"""Stage E checks that need the rendered text rather than the plan.

These run on the published artifact, not on the draft, so they also catch a
substitution bug in Pass 2 and not just a model that misbehaved in Pass 1. L1 is
the important one: a figure in a document that traces to no assigned variable is
a fact the database does not contain, and every task that touches it becomes
unsolvable.
"""

import hashlib
import re
import statistics

from tau3.worldgen.linters import LintReport
from tau3.worldgen.models import DocPlan, WorldSchema
from tau3.worldgen.render import ALLOWED_DIGITS, format_value

NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
SHINGLE = 5
NEAR_DUPLICATE_JACCARD = 0.72
# Length band around an archetype's target, from the corpus's own spread.
LENGTH_BAND = (0.45, 2.2)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def minhash(text: str, permutations: int = 64) -> list[int]:
    """A cheap MinHash over word shingles; no dependency for one similarity."""
    words = _tokens(text)
    shingles = {
        " ".join(words[i : i + SHINGLE])
        for i in range(max(1, len(words) - SHINGLE + 1))
    }
    if not shingles:
        return [0] * permutations
    signature = []
    for index in range(permutations):
        signature.append(
            min(
                int(hashlib.sha1(f"{index}|{s}".encode()).hexdigest()[:12], 16)
                for s in shingles
            )
        )
    return signature


def similarity(left: list[int], right: list[int]) -> float:
    """Estimated Jaccard similarity from two signatures."""
    if not left or not right:
        return 0.0
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


def check_numeric_provenance(
    schema: WorldSchema, documents: list[DocPlan], rendered: dict[str, str]
) -> LintReport:
    """L1: every figure in a document traces to a value assigned to it."""
    report = LintReport()
    for document in documents:
        text = rendered.get(document.doc_id)
        if text is None:
            continue
        expected = set()
        renderings = []
        for variable_id in document.variable_ids:
            variable = schema.variable(variable_id)
            rendering = format_value(variable)
            renderings.append(rendering)
            expected.add(rendering)
            expected.add(rendering.lstrip("$").rstrip("%").replace(",", ""))
        masked = text
        # Mask only the exact rendered forms. Masking the normalized variants
        # too would blank a bare "0" everywhere it appears, so a document
        # carrying a zero-valued variable would swallow the digits of any figure
        # the model invented.
        for rendering in sorted(renderings, key=len, reverse=True):
            if rendering:
                masked = masked.replace(rendering, " ")
        for pattern in ALLOWED_DIGITS:
            masked = pattern.sub(" ", masked)
        for found in NUMBER.findall(masked):
            normalized = found.lstrip("$").rstrip("%").replace(",", "")
            if found in expected or normalized in expected:
                continue
            report.add(
                "L1",
                document.doc_id,
                f"States {found!r}, which traces to no variable allocated to it",
            )
    return report


def check_cross_document_agreement(
    schema: WorldSchema, documents: list[DocPlan], rendered: dict[str, str]
) -> LintReport:
    """L2: a variable stated in two documents must read the same in both."""
    report = LintReport()
    for variable in schema.variables:
        if variable.derived or variable.value is None:
            continue
        homes = [d for d in documents if variable.id in d.variable_ids]
        if len(homes) < 2:
            continue
        rendering = format_value(variable)
        for document in homes:
            text = rendered.get(document.doc_id)
            if text is None:
                continue
            if rendering not in text and rendering.lstrip("$") not in text:
                report.add(
                    "L2",
                    f"{variable.id}:{document.doc_id}",
                    f"Should state {rendering!r} but the rendered text does not",
                )
    return report


def check_length_band(
    documents: list[DocPlan],
    rendered: dict[str, str],
    corpus_mean: float,
    tolerance=4.0,
) -> LintReport:
    """L7: documents sit near their archetype's length, and the corpus near the target."""
    report = LintReport()
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(encoding.encode(text))
    except ImportError:  # pragma: no cover - tiktoken ships with the extra

        def count(text: str) -> int:
            return len(text.split())

    lengths = []
    for document in documents:
        text = rendered.get(document.doc_id)
        if text is None:
            continue
        tokens = count(f"{document.title}\n{text}")
        lengths.append(tokens)
        low, high = LENGTH_BAND
        if not low * document.target_tokens <= tokens <= high * document.target_tokens:
            report.add(
                "L7",
                document.doc_id,
                f"{tokens} tokens is outside "
                f"[{low * document.target_tokens:.0f}, "
                f"{high * document.target_tokens:.0f}] for {document.archetype}",
                severity="advice",
            )
    if lengths:
        mean = statistics.fmean(lengths)
        if abs(mean - corpus_mean) > tolerance:
            report.add(
                "L7-corpus",
                "corpus",
                f"Mean document length {mean:.1f} against the measured target "
                f"{corpus_mean:.1f} ({mean - corpus_mean:+.1f})",
                severity="advice",
            )
    return report


def check_near_duplicates(rendered: dict[str, str]) -> LintReport:
    """L8: two documents saying the same thing add no retrieval difficulty."""
    report = LintReport()
    signatures = {doc_id: minhash(text) for doc_id, text in rendered.items()}
    ordered = sorted(signatures)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            score = similarity(signatures[left], signatures[right])
            if score > NEAR_DUPLICATE_JACCARD:
                report.add(
                    "L8",
                    f"{left}:{right}",
                    f"Estimated Jaccard {score:.2f} exceeds {NEAR_DUPLICATE_JACCARD}",
                )
    return report


def check_tool_coverage(
    tools: list[dict], documents: list[DocPlan], rendered: dict[str, str]
) -> LintReport:
    """L9: every discoverable tool is named somewhere, and no invented tool is."""
    report = LintReport()
    aliases = {tool["alias"] for tool in tools}
    official = {tool["official_name"] for tool in tools}
    corpus = "\n".join(rendered.values())

    for tool in tools:
        if tool["alias"] not in corpus:
            report.add(
                "L9",
                tool["official_name"],
                f"No document names {tool['alias']}, so it can never be unlocked",
            )
    mentioned = set(re.findall(r"\b[a-z][a-z_]*_\d{4}\b", corpus))
    for name in sorted(mentioned - aliases):
        # An official suffix in this world's text is a leak, not just an unknown.
        kind = "official name leaked" if name in official else "unknown tool"
        report.add("L9", name, f"Documents name a tool that does not exist: {kind}")
    del documents
    return report


def lint_rendered(
    schema: WorldSchema,
    documents: list[DocPlan],
    rendered: dict[str, str],
    tools: list[dict],
    corpus_mean: float,
) -> LintReport:
    """Run every check that needs the finished text."""
    report = LintReport()
    for part in (
        check_numeric_provenance(schema, documents, rendered),
        check_cross_document_agreement(schema, documents, rendered),
        check_length_band(documents, rendered, corpus_mean),
        check_near_duplicates(rendered),
        check_tool_coverage(tools, documents, rendered),
    ):
        report.findings.extend(part.findings)
    return report
