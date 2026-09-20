"""Prove a variant is complete, isolated and reproducible.

Every check fails closed.  The one that matters most is the leak scan: a
missed rename is invisible at runtime -- the variant still loads, still
answers, still scores -- but it leaves a canonical symbol in place for a model
to recognise, which is precisely the failure the whole pipeline exists to
prevent.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from tau3.perturb import scan
from tau3.perturb.models import PerturbationPlan
from tau3.perturb.storage import read_json


@dataclass
class Report:
    """Outcome of verifying one variant."""

    variant_id: str
    stages: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no stage produced a finding."""
        return not self.findings

    def record(self, stage: str, problems: Iterable[str]) -> None:
        """Record a stage's outcome."""
        problems = list(problems)
        self.stages[stage] = "ok" if not problems else f"{len(problems)} finding(s)"
        self.findings.extend(f"[{stage}] {problem}" for problem in problems)

    def to_dict(self) -> dict:
        """Serializable form, written to ``report.json``."""
        return {
            "variant_id": self.variant_id,
            "ok": self.ok,
            "stages": self.stages,
            "findings": self.findings,
        }


def load_plan(variant_root: Path) -> PerturbationPlan:
    """Read the plan a variant was built from."""
    return PerturbationPlan.model_validate(read_json(Path(variant_root) / "plan.json"))


#: Artifacts of running the variant, not of generating it.
_IGNORED_NAMES = frozenset({"plan.json", "mapping.csv", "report.json"})
_IGNORED_DIRS = frozenset({"__pycache__"})


def variant_files(variant_root: Path) -> list[Path]:
    """Every generated file in a variant tree.

    Excludes ``__pycache__``: importing the variant leaves bytecode behind, and
    that is a product of running it rather than of building it -- it must not
    count as generated content for the leak or determinism checks.
    """
    root = Path(variant_root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in _IGNORED_NAMES
        and not _IGNORED_DIRS.intersection(path.parts)
    )


def _matchable(path: Path) -> str:
    """Text to scan, decoding JSON so escapes cannot hide a symbol."""
    text = path.read_text()
    if path.suffix == ".json":
        try:
            return "\n".join(scan._strings_in(json.loads(text)))
        except json.JSONDecodeError:
            return text
    return text


def check_leaks(variant_root: Path, plan: PerturbationPlan) -> list[str]:
    """No canonical symbol the plan renamed may survive anywhere in the variant.

    Includes filenames: a document id lives in its name as well as its body.
    """
    from tau3.perturb.rewrite import FormCounter

    sources = [form.source for concept in plan.concepts for form in concept.forms]
    if not sources:
        return ["plan renames nothing"]
    counter = FormCounter(sources)

    problems: list[str] = []
    for path in variant_files(variant_root):
        relative = path.relative_to(variant_root).as_posix()
        for form, count in sorted(counter.count(_matchable(path)).items()):
            problems.append(f"{relative}: {count}x canonical symbol {form!r}")
        for form, _ in sorted(counter.count(path.name).items()):
            problems.append(f"{relative}: canonical symbol {form!r} in filename")
    return problems


def check_structure(variant_root: Path, plan: PerturbationPlan) -> list[str]:
    """Generated modules parse, and the package imports in a clean process."""
    problems: list[str] = []
    package_dir = Path(variant_root) / "code" / plan.package_name
    if not package_dir.is_dir():
        return [f"missing package directory: {package_dir}"]

    for path in sorted(package_dir.glob("*.py")):
        try:
            ast.parse(path.read_text())
        except SyntaxError as error:
            problems.append(f"{path.name}: {error}")
    if problems:
        return problems

    script = f"""
import sys, json
sys.path.insert(0, {str(package_dir.parent)!r})
from {plan.package_name} import environment as env_mod

def names(tools):
    # Environment.get_tools() yields Tool objects; toolkits yield name->callable.
    if isinstance(tools, dict):
        return sorted(tools)
    return sorted(tool.name for tool in tools)

env = env_mod.get_environment(retrieval_variant="bm25_grep", retrieval_kwargs={{"top_k": 10}})
kb = env_mod.get_knowledge_base()
print("RESULT" + json.dumps({{
    "domain": env.domain_name,
    "visible": names(env.get_tools()),
    "discoverable": names(env.tools.get_discoverable_tools()),
    "user_visible": names(env.get_user_tools()),
    "user_discoverable": names(env.user_tools.get_discoverable_tools()),
    "documents": len(kb.documents),
    "tasks": len(env_mod.get_tasks()),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    if completed.returncode != 0:
        return [f"variant package failed to import:\n{completed.stderr[-2000:]}"]

    payload = next(
        (line[len("RESULT") :] for line in completed.stdout.splitlines()
         if line.startswith("RESULT")),
        None,
    )
    if payload is None:
        return ["import produced no result line"]
    info = json.loads(payload)

    if info["domain"] != plan.domain_name:
        problems.append(f"domain is {info['domain']!r}, expected {plan.domain_name!r}")
    expected_docs = plan.expected.get("document_count")
    if expected_docs and info["documents"] != expected_docs:
        problems.append(f"{info['documents']} documents, expected {expected_docs}")
    expected_tasks = plan.expected.get("task_count")
    if expected_tasks and info["tasks"] != expected_tasks:
        problems.append(f"{info['tasks']} tasks, expected {expected_tasks}")

    renamed = {c.root: c.new_root for c in plan.concepts if c.kind == "tool"}
    live = set().union(*(
        info[key]
        for key in ("visible", "discoverable", "user_visible", "user_discoverable")
    ))
    for canonical, new_name in sorted(renamed.items()):
        if canonical in live:
            problems.append(f"canonical tool still exposed: {canonical}")
        if new_name not in live:
            problems.append(f"renamed tool missing from toolkit: {new_name}")
    return problems


def check_retrieval_isolation(variant_root: Path, plan: PerturbationPlan) -> list[str]:
    """A variant built after the canonical domain must index its own documents.

    ``tau3.knowledge.embeddings_cache`` keeps one process-global, unkeyed list
    of documents.  Materialization replaces it with a variant-local cache; this
    check is the regression guard, and it deliberately builds the canonical
    environment first so a reintroduced bug would actually fire.
    """
    package_dir = Path(variant_root) / "code" / plan.package_name
    script = f"""
import sys, json, re
sys.path.insert(0, {str(package_dir.parent)!r})
from tau3.domains.banking_knowledge import environment as canonical
canonical.get_environment(retrieval_variant="bm25_grep", retrieval_kwargs={{"top_k": 10}})
from {plan.package_name} import environment as variant
env = variant.get_environment(retrieval_variant="bm25_grep", retrieval_kwargs={{"top_k": 10}})
found = env.tools.KB_search(query="account monthly maintenance fee eligibility")
print("RESULT" + json.dumps(sorted(set(re.findall(r"doc_[A-Za-z0-9_()\\-]+", found)))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=900
    )
    if completed.returncode != 0:
        return [f"isolation probe failed:\n{completed.stderr[-1500:]}"]
    payload = next(
        (line[len("RESULT") :] for line in completed.stdout.splitlines()
         if line.startswith("RESULT")),
        None,
    )
    if payload is None:
        return ["isolation probe produced no result line"]

    returned = json.loads(payload)
    if not returned:
        return ["retrieval returned no documents"]
    known = {
        path.stem for path in (Path(variant_root) / "data" / "documents").glob("*.json")
    }
    foreign = [doc for doc in returned if doc not in known]
    if foreign:
        return [
            "retrieval returned documents that are not this variant's "
            f"(shared cache leak?): {foreign[:5]}"
        ]
    return []


def check_determinism(
    variant_root: Path, plan: PerturbationPlan, snapshot_root: Path
) -> list[str]:
    """Rebuilding from the same plan and snapshot must be byte-identical."""
    from tau3.perturb.materialize import materialize

    with tempfile.TemporaryDirectory() as temporary:
        rebuilt_root = Path(temporary) / "rebuild"
        materialize(plan, snapshot_root=snapshot_root, destination=rebuilt_root, force=True)
        original = {
            path.relative_to(variant_root).as_posix(): path.read_bytes()
            for path in variant_files(variant_root)
        }
        rebuilt = {
            path.relative_to(rebuilt_root).as_posix(): path.read_bytes()
            for path in variant_files(rebuilt_root)
        }

    problems: list[str] = []
    for key in sorted(set(original) - set(rebuilt)):
        problems.append(f"only in original build: {key}")
    for key in sorted(set(rebuilt) - set(original)):
        problems.append(f"only in rebuild: {key}")
    for key in sorted(set(original) & set(rebuilt)):
        if original[key] != rebuilt[key]:
            problems.append(f"differs between builds: {key}")
    return problems[:20]


def check_disjoint(plan: PerturbationPlan, others: Iterable[PerturbationPlan]) -> list[str]:
    """Variants used for training and evaluation must share no new symbol."""
    mine = {form.target for concept in plan.concepts for form in concept.forms}
    problems: list[str] = []
    for other in others:
        if other.variant_id == plan.variant_id:
            continue
        theirs = {form.target for concept in other.concepts for form in concept.forms}
        shared = sorted(mine & theirs)
        if shared:
            problems.append(
                f"shares {len(shared)} symbol(s) with variant {other.variant_id}: "
                f"{shared[:5]}"
            )
    return problems


def verify(
    variant_root: Path,
    *,
    snapshot_root: Optional[Path] = None,
    others: Iterable[PerturbationPlan] = (),
    skip_slow: bool = False,
) -> Report:
    """Run every check against a materialized variant.

    Args:
        variant_root: The variant to verify.
        snapshot_root: Snapshot to rebuild from for the determinism check.
        others: Plans of sibling variants, checked for symbol overlap.
        skip_slow: Skip the subprocess and rebuild checks.

    Returns:
        A :class:`Report`; ``report.ok`` is False if anything failed.
    """
    variant_root = Path(variant_root)
    plan = load_plan(variant_root)
    report = Report(variant_id=plan.variant_id)

    report.record("leak", check_leaks(variant_root, plan))
    report.record("disjoint", check_disjoint(plan, others))
    if not skip_slow:
        report.record("structure", check_structure(variant_root, plan))
        report.record("retrieval_isolation", check_retrieval_isolation(variant_root, plan))
        if snapshot_root is not None:
            report.record("determinism", check_determinism(variant_root, plan, snapshot_root))
    return report
