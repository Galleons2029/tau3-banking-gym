"""Apply a plan to a snapshot and emit a runnable variant domain.

The output is a build artifact: a renamed copy of the domain's nine Python
modules plus its data, written under the variant workspace and never inside the
canonical domain.  Nothing here opens a canonical file for writing -- input
comes from a frozen snapshot -- so the official corpus keeps working untouched
and ``environment_fingerprint()`` is unchanged, which is what keeps already
published bundles loadable.

Every replacement count is checked against the plan.  A rename that reaches a
different number of sites than predicted aborts the build; that is the
mechanism that turns "one spelling was missed" into a failure instead of a
quietly degraded variant.
"""

import ast
import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tau3.perturb import scan
from tau3.perturb import snapshot as snapshot_module
from tau3.perturb.models import PerturbationPlan
from tau3.perturb.rewrite import Rewriter, build_rewriter
from tau3.perturb.storage import write_json
from tau3.perturb.workspace import assert_outside_domain

#: Canonical package path, rewritten to the variant package in every import.
CANONICAL_PACKAGE = "tau3.domains.banking_knowledge"


class MaterializeError(RuntimeError):
    """A variant could not be built exactly as planned."""


@dataclass(frozen=True)
class Materialized:
    """Where a variant landed and what it cost."""

    root: Path
    plan: PerturbationPlan
    file_count: int
    replacements: int

    @property
    def code_dir(self) -> Path:
        """Directory to put on ``sys.path`` to import the variant package."""
        return self.root / "code"

    @property
    def data_dir(self) -> Path:
        """The variant's domain data directory."""
        return self.root / "data"


def _structural_rewrites(plan: PerturbationPlan) -> dict[str, tuple[tuple[str, str, int], ...]]:
    """Per-module ``(pattern, replacement, expected_count)`` code edits.

    These are the edits a symbol rename cannot express: where the package lives,
    where its data lives, what the domain is called, and one cache fix.
    """
    package = plan.package_name
    # code/<package>/<module>.py -> parents[2] is the variant root.  The
    # canonical modules derive their paths from DATA_DIR and never import Path,
    # so the replacement carries its own aliased import rather than assuming
    # one is in scope.
    data_root = '_VariantPath(__file__).resolve().parents[2] / "data"'
    path_import = "from pathlib import Path as _VariantPath\n\n"
    return {
        "*": (
            (rf"\b{re.escape(CANONICAL_PACKAGE)}\b", package, -1),
        ),
        "utils.py": (
            (
                r'KNOWLEDGE_DATA_DIR = DATA_DIR / "tau3" / "domains" / "banking_knowledge"',
                f"{path_import}KNOWLEDGE_DATA_DIR = {data_root}",
                1,
            ),
        ),
        "retrieval.py": (
            (
                r'PROMPTS_DIR = DATA_DIR / "tau3" / "domains" / "banking_knowledge" / "prompts"',
                f'{path_import}PROMPTS_DIR = {data_root} / "prompts"',
                1,
            ),
            (
                r"from tau3\.knowledge\.embeddings_cache import get_cached_docs, set_cached_docs",
                _DOCS_CACHE_SHIM,
                1,
            ),
        ),
        "environment.py": (
            (r'domain_name="banking_knowledge"', f'domain_name="{plan.domain_name}"', 1),
        ),
    }


#: Replaces the shared document cache with a variant-local one.
#:
#: ``tau3.knowledge.embeddings_cache`` holds a single process-global list with
#: no key, so a variant environment built after a canonical one in the same
#: process would silently index the canonical documents -- and look like
#: unusually good retrieval rather than a bug.
_DOCS_CACHE_SHIM = '''_VARIANT_DOCS_CACHE = {}


def get_cached_docs(key="default"):
    """Variant-local replacement for the shared, process-global doc cache."""
    return _VARIANT_DOCS_CACHE.get(key)


def set_cached_docs(docs, key="default"):
    """Cache this variant's documents without touching other domains."""
    _VARIANT_DOCS_CACHE[key] = docs'''


def _variant_meta(plan: PerturbationPlan, environment_hash: str) -> str:
    """Source for the generated ``_variant_meta`` module."""
    return (
        '"""Generated. Identifies this variant and the inputs it was built from."""\n\n'
        f'VARIANT_ID = {plan.variant_id!r}\n'
        f'DOMAIN_NAME = {plan.domain_name!r}\n'
        f'PACKAGE_NAME = {plan.package_name!r}\n'
        f'BASE_DOMAIN = {plan.base_domain!r}\n'
        f'PLAN_DIGEST = {plan.plan_digest!r}\n'
        f'CORPUS_DIGEST = {plan.corpus_digest!r}\n'
        f'BASE_ENVIRONMENT_HASH = {environment_hash!r}\n'
    )


def _apply_structural(text: str, key: str, rules: dict) -> tuple[str, list[str]]:
    """Apply structural code edits, reporting any whose count was wrong."""
    name = key.rsplit("/", 1)[-1]
    problems: list[str] = []
    for scope in ("*", name):
        for pattern, replacement, expected in rules.get(scope, ()):
            text, found = re.subn(pattern, replacement.replace("\\", "\\\\"), text)
            if expected >= 0 and found != expected:
                problems.append(
                    f"{key}: expected {expected} match(es) of {pattern!r}, found {found}"
                )
    return text, problems


#: Corpus kinds whose content is JSON and must be rewritten structurally.
JSON_KINDS = ("document", "task", "tasks_aggregate", "db")


def rewrite_corpus(
    corpus, values: Rewriter, keys: Rewriter
) -> tuple[dict[str, str], Counter]:
    """Apply the symbol rename to every corpus file.

    Shared by the plan's dry run and the real build so the two can never
    disagree.  JSON is rewritten structurally rather than as text: in raw JSON
    an escape like ``\n`` ends in a word character, which corrupts the word
    boundary in front of the next token, so raw-text matching silently
    undercounts.  The parsed string is what the rewriter -- and the agent --
    actually sees.
    """
    outputs: dict[str, str] = {}
    hits: Counter = Counter()
    for corpus_file in corpus:
        if corpus_file.kind in JSON_KINDS:
            rewritten, found = values.apply_json_text(corpus_file.text, keys=keys)
        else:
            rewritten, found = values.apply(corpus_file.text)
        outputs[corpus_file.key] = rewritten
        hits.update(found)
    return outputs, hits


def materialize(
    plan: PerturbationPlan,
    *,
    snapshot_root: Path,
    destination: Path,
    environment_hash: str = "",
    force: bool = False,
) -> Materialized:
    """Build the variant described by ``plan``.

    Args:
        plan: The plan to apply.
        snapshot_root: Frozen corpus to read from.
        destination: Variant root directory; replaced when ``force``.
        environment_hash: Canonical environment fingerprint, recorded so a
            stale variant can be detected later.
        force: Overwrite an existing variant.

    Returns:
        A :class:`Materialized` describing the result.

    Raises:
        MaterializeError: If any replacement count differs from the plan, if a
            generated module does not parse, or if a structural edit misfired.
    """
    destination = Path(destination)
    assert_outside_domain(destination)

    snap = snapshot_module.load(snapshot_root)
    if snap.digest != plan.corpus_digest:
        raise MaterializeError(
            "Plan was built against a different corpus: "
            f"plan {plan.corpus_digest[:12]} vs snapshot {snap.digest[:12]}"
        )

    pairs = plan.replacement_pairs()
    values: Rewriter = build_rewriter(pairs)
    # Only identifiers are renamed as JSON keys; a product's prose name must
    # never become a schema field.
    keys: Rewriter = build_rewriter(
        [
            (form.source, form.target)
            for concept in plan.concepts
            if concept.kind == "identifier"
            for form in concept.forms
        ]
    )

    corpus = scan.load_corpus(snapshot_root)
    rules = _structural_rewrites(plan)
    problems: list[str] = []

    rewritten_by_key, hits = rewrite_corpus(corpus, values, keys)

    outputs: dict[str, str] = {}
    for corpus_file in corpus:
        key = corpus_file.key
        rewritten = rewritten_by_key[key]

        if corpus_file.kind == "python":
            rewritten, issues = _apply_structural(rewritten, key, rules)
            problems.extend(issues)
            try:
                ast.parse(rewritten)
            except SyntaxError as error:
                problems.append(f"{key}: generated module does not parse: {error}")

        # A document's filename must follow its renamed id.
        if corpus_file.kind == "document":
            new_id = json.loads(rewritten)["id"]
            key = f"data/documents/{new_id}.json"
        outputs[key] = rewritten

    expected = plan.expected_counts()
    for source, planned in sorted(expected.items()):
        actual = hits.get(source, 0)
        if actual != planned:
            problems.append(
                f"rename count mismatch for {source!r}: plan {planned}, applied {actual}"
            )
    for source in sorted(set(hits) - set(expected)):
        problems.append(f"replaced {source!r} which the plan does not list")

    if len(outputs) != len(corpus):
        problems.append(
            f"output file count {len(outputs)} != input {len(corpus)}; "
            "two documents likely collapsed onto one renamed id"
        )

    if problems:
        raise MaterializeError(
            "Variant does not match its plan:\n  " + "\n  ".join(problems[:20])
        )

    if destination.exists():
        if not force:
            raise MaterializeError(f"Variant already exists: {destination} (use force)")
        shutil.rmtree(destination)

    package_dir = destination / "code" / plan.package_name
    for key, text in sorted(outputs.items()):
        target = (
            package_dir / key[len("code/") :]
            if key.startswith("code/")
            else destination / key
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    (package_dir / "_variant_meta.py").write_text(_variant_meta(plan, environment_hash))
    write_json(destination / "plan.json", plan.model_dump())
    _write_mapping(destination / "mapping.csv", plan)

    return Materialized(
        root=destination,
        plan=plan,
        file_count=len(outputs),
        replacements=sum(hits.values()),
    )


def _write_mapping(path: Path, plan: PerturbationPlan) -> None:
    """Write the human-reviewable old -> new table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "concept_id", "style", "from", "to", "occurrences"])
        for concept in plan.concepts:
            for form in concept.forms:
                writer.writerow(
                    [
                        concept.kind,
                        concept.concept_id,
                        form.style or "literal",
                        form.source,
                        form.target,
                        form.count,
                    ]
                )
