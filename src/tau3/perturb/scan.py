"""Inventory the canonical corpus: what symbols exist, spelled how, how often.

Discovery is *evidence-driven*, never derivation-driven.  Candidate spellings
are proposed from a concept's root (via :mod:`tau3.perturb.naming`), but only
the ones that actually occur are kept -- the corpus is full of irregular
casing (``EcoCard``, ``BNPL Bronze``, ``Split-the-Bill``) that no style
function would produce.  Spellings a concept could have but does not are still
useful: the leak scan denies them so a variant can never drift into one.

Counts recorded here become assertions the materializer checks, so they are
produced by :class:`~tau3.perturb.rewrite.FormCounter`, which shares the
rewriter's matching semantics exactly.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from tau3.perturb import naming
from tau3.perturb.rewrite import FormCounter

#: The canonical banking brand, in root form.
BRAND_ROOT = "Rho Bank"

#: Document ids are ``doc_<category>_<product>_<NNN>``.  Categories are generic
#: domain vocabulary ("checking accounts") and are deliberately NOT renamed --
#: a variant should still be a bank.
DOC_CATEGORIES: tuple[str, ...] = (
    "bank_accounts",
    "business_checking_accounts",
    "business_credit_cards",
    "business_savings_accounts",
    "buy_now_pay_later",
    "checking_accounts",
    "credit_cards",
    "customer_support",
    "everyone_pay",
    "personal_subscriptions",
    "savings_accounts",
)

#: Segments that name a topic rather than a product.
NON_PRODUCT_SLUGS: frozenset[str] = frozenset(
    {
        "automatic_sweep_program",
        "bnpl_management_dashboard",
        "credit_card_account_logistics",
        "credit_card_replacements",
        "everyone_pay",
        "joint_business_holders_+_user_roles",
        "special_support_codes",
        "virtual_card_management",
    }
)

#: Products whose display spelling no style function reproduces.
DISPLAY_OVERRIDES: dict[str, str] = {
    "bnpl_bronze": "BNPL Bronze",
    "bnpl_diamond": "BNPL Diamond",
    "bnpl_gold": "BNPL Gold",
    "bnpl_platinum": "BNPL Platinum",
    "bnpl_silver": "BNPL Silver",
    "crypto-cash_back": "Crypto-Cash Back",
    "ecocard": "EcoCard",
    "green_fee-free_account": "Green Fee-Free Account",
    "qr_transfers": "QR Transfers",
    "split-the-bill": "Split-the-Bill",
}

#: Product roots that own no document segment of their own but are used as
#: identifiers elsewhere -- ``"green account"`` is a dict key in ``tools.py``
#: while the documents only ever say "Green Account (checking)/(savings)".
#: Registering the bare root keeps both spellings on one concept.
EXTRA_PRODUCT_ROOTS: tuple[str, ...] = ("Green Account",)

#: Identifiers that embed the brand and are load-bearing as symbols: a
#: parameter name in ``apply_for_credit_card`` and ``generate_application_id``,
#: a JSON key in task arguments, and a docstring mention.  The snake spelling
#: ``rho_bank`` never counts as a standalone brand hit -- it is always followed
#: by ``_``, which fails the boundary rule -- so it needs its own concept.
IDENTIFIERS: tuple[str, ...] = ("rho_bank_subscription",)

#: Tools whose names the harness depends on by literal string.  Renaming these
#: would break ``scenarios.Actions.call``, ``validation.validate_references``
#: and ``build._derive_read_log_allowlist``.
FROZEN_TOOLS: frozenset[str] = frozenset(
    {
        "calculate",
        "call_discoverable_agent_tool",
        "call_discoverable_user_tool",
        "give_discoverable_user_tool",
        "grep",
        "KB_search",
        "KB_search_bm25",
        "KB_search_dense",
        "list_discoverable_agent_tools",
        "list_discoverable_user_tools",
        "request_human_agent_transfer",
        "rewrite_context",
        "shell",
        "think",
        "transfer_to_human_agents",
        "unlock_discoverable_agent_tool",
    }
)

_DOC_ID = re.compile(r"^doc_(?P<segment>.+)_(?P<index>\d{3})$")

#: Styles the brand is spelled in.  Only the brand gets first-word forms:
#: "Rho" really is shorthand for "Rho Bank".
BRAND_STYLES: tuple[str, ...] = (
    "title_hyphen", "upper_hyphen", "lower_hyphen", "mixed_hyphen",
    "title_space", "upper_space", "lower_space", "snake", "upper_snake",
    "titleconcat", "lowerconcat", "upperconcat",
    "title_first", "upper_first", "lower_first",
)

#: Styles a product name is spelled in.  Deliberately excludes first-word
#: forms: "Bronze" is not shorthand for "Bronze Rewards Card", it is a tier
#: token shared with six other products, and claiming it here would make
#: several concepts fight over the same spelling.
PRODUCT_STYLES: tuple[str, ...] = (
    "title_space", "upper_space", "lower_space", "snake", "upper_snake",
)


@dataclass(frozen=True)
class CorpusFile:
    """One canonical file, with a stable key used in plans and reports."""

    key: str
    kind: str  # document | task | tasks_aggregate | db | prompt | python | markdown
    path: Path
    text: str


@dataclass(frozen=True)
class ObservedForm:
    """A spelling that actually occurs, and how often."""

    form: str
    style: Optional[str]
    count: int


@dataclass(frozen=True)
class Concept:
    """A renameable symbol and every spelling of it found in the corpus."""

    concept_id: str
    kind: str  # brand | product | tool | identifier
    root: str
    slug: Optional[str] = None
    base: Optional[str] = None
    detail: Optional[str] = None
    forms: tuple[ObservedForm, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Total occurrences across all spellings."""
        return sum(form.count for form in self.forms)


@dataclass(frozen=True)
class Proposal:
    """A concept and its candidate spellings, before counting."""

    concept_id: str
    kind: str
    root: str
    slug: Optional[str]
    detail: Optional[str]
    candidates: dict[str, Optional[str]]


@dataclass(frozen=True)
class Inventory:
    """Everything a plan needs to know about the canonical corpus."""

    corpus: tuple[CorpusFile, ...]
    concepts: tuple[Concept, ...]
    document_ids: tuple[str, ...]
    task_ids: tuple[str, ...]

    def by_kind(self, kind: str) -> tuple[Concept, ...]:
        """Concepts of one kind, in declaration order."""
        return tuple(c for c in self.concepts if c.kind == kind)

    def concept(self, concept_id: str) -> Concept:
        """Look up one concept by id."""
        for candidate in self.concepts:
            if candidate.concept_id == concept_id:
                return candidate
        raise KeyError(concept_id)


# -- corpus loading -------------------------------------------------------


def domain_data_dir() -> Path:
    """The canonical domain's data directory."""
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DATA_DIR

    return Path(KNOWLEDGE_DATA_DIR)


def domain_code_dir() -> Path:
    """The canonical domain's Python package."""
    import tau3.domains.banking_knowledge as package

    return Path(package.__file__).parent


def _strings_in(node) -> list[str]:
    """Every string inside a decoded JSON document, keys included."""
    found: list[str] = []
    if isinstance(node, str):
        found.append(node)
    elif isinstance(node, dict):
        for key, child in node.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_strings_in(child))
    elif isinstance(node, list):
        for item in node:
            found.extend(_strings_in(item))
    return found


def matchable_text(corpus_file: "CorpusFile") -> str:
    """Text to match symbols against.

    JSON is decoded first.  In raw JSON an escape such as ``\n`` is a
    backslash followed by the letter ``n`` -- a word character -- which defeats
    the word boundary in front of the very next token and silently undercounts
    it.  What the rewriter edits, and what the agent reads, is the decoded
    string.
    """
    if corpus_file.kind in ("document", "task", "tasks_aggregate", "db"):
        return "\n".join(_strings_in(json.loads(corpus_file.text)))
    return corpus_file.text


def kind_for(key: str) -> str:
    """Classify a corpus file by its stable key."""
    if key.startswith("code/"):
        return "python" if key.endswith(".py") else "markdown"
    if key == "data/db.json":
        return "db"
    if key == "data/tasks.json":
        return "tasks_aggregate"
    if key.startswith("data/documents/"):
        return "document"
    if key.startswith("data/tasks/"):
        return "task"
    if key.startswith("data/prompts/"):
        return "prompt"
    raise ValueError(f"Unclassifiable corpus key: {key}")


def _canonical_paths() -> list[tuple[str, Path]]:
    """Enumerate ``(key, path)`` for the live canonical corpus."""
    data, code = domain_data_dir(), domain_code_dir()
    found: list[tuple[str, Path]] = []
    for path in (data / "documents").glob("*.json"):
        found.append((f"data/documents/{path.name}", path))
    for path in (data / "tasks").glob("task_*.json"):
        found.append((f"data/tasks/{path.name}", path))
    for name in ("tasks.json", "db.json"):
        if (data / name).exists():
            found.append((f"data/{name}", data / name))
    for path in (data / "prompts").rglob("*.md"):
        found.append((f"data/prompts/{path.relative_to(data / 'prompts').as_posix()}", path))
    for pattern in ("*.py", "*.md"):
        for path in code.glob(pattern):
            found.append((f"code/{path.name}", path))
    return sorted(found)


@lru_cache(maxsize=4)
def load_corpus(snapshot_root: Optional[Path] = None) -> tuple[CorpusFile, ...]:
    """Load every canonical file a rename has to propagate through.

    Args:
        snapshot_root: Read from this frozen snapshot instead of the live
            domain directory.  Preferred: the canonical corpus is shared with
            other running jobs, and a snapshot is both a stable view and a
            guarantee that nothing here opens an original for writing.

    Cached: this is 800-odd reads off a network filesystem, and the scan, the
    plan builder and the verifier all want the same bytes.
    """
    if snapshot_root is not None:
        from tau3.perturb import snapshot as snapshot_module

        snap = snapshot_module.load(Path(snapshot_root))
        archive = Path(snapshot_root) / snapshot_module.ARCHIVE
        return tuple(
            CorpusFile(key=key, kind=kind_for(key), path=archive, text=text)
            for key, text in sorted(snap.files.items())
        )
    return tuple(
        CorpusFile(key=key, kind=kind_for(key), path=path, text=path.read_text())
        for key, path in _canonical_paths()
    )


# -- concept discovery ----------------------------------------------------


def _contains_at_boundary(haystack: str, needle: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` on word boundaries."""
    if needle == haystack:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])"
    return re.search(pattern, haystack) is not None


def _candidate_forms(
    root: str, slug: Optional[str], styles: tuple[str, ...] = BRAND_STYLES
) -> dict[str, Optional[str]]:
    """Propose spellings for a concept: its literal, its slug, and each style."""
    candidates: dict[str, Optional[str]] = {root: "literal"}
    if slug:
        candidates.setdefault(slug, "slug")
    for form, style in naming.all_forms(root, styles).items():
        candidates.setdefault(form, style)
    return candidates


def _split_segment(segment: str) -> tuple[str, str]:
    """Split ``<category>_<product>`` using the longest matching category."""
    for category in sorted(DOC_CATEGORIES, key=len, reverse=True):
        if segment.startswith(category + "_"):
            return category, segment[len(category) + 1 :]
    raise ValueError(f"Document segment has no known category: {segment}")


def product_roots(document_ids: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Discover ``(slug, display_root)`` for every product, from document ids."""
    segments = set()
    for document_id in document_ids:
        match = _DOC_ID.match(document_id)
        if not match:
            raise ValueError(f"Unparseable document id: {document_id}")
        segments.add(match.group("segment"))

    roots: list[tuple[str, str]] = []
    for segment in sorted(segments):
        _, slug = _split_segment(segment)
        if slug.endswith("_(general)") or slug in NON_PRODUCT_SLUGS:
            continue
        display = DISPLAY_OVERRIDES.get(slug)
        if display is None:
            display = " ".join(
                word if word.isupper() else word.capitalize()
                for word in slug.replace("_", " ").split()
            )
        roots.append((slug, display))
    for extra in EXTRA_PRODUCT_ROOTS:
        roots.append((naming.render(extra, "snake"), extra))
    return tuple(sorted(set(roots)))


def tool_names() -> tuple[tuple[str, str], ...]:
    """Discover ``(name, kind)`` for every renameable domain tool.

    The toolkits are introspected with a null database: tool discovery reads
    only class attributes, and nothing here ever calls a tool.
    """
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    found: list[tuple[str, str]] = []
    for toolkit, owner in ((KnowledgeTools(None), "agent"), (KnowledgeUserTools(None), "user")):
        discoverable = set(toolkit.get_discoverable_tools())
        for name in sorted(toolkit.tools):
            if name in FROZEN_TOOLS:
                continue
            suffix = "discoverable" if name in discoverable else "visible"
            found.append((name, f"{owner}_{suffix}"))
    return tuple(found)


def inventory(snapshot_root: Optional[Path] = None) -> Inventory:
    """Scan the corpus and return every renameable concept.

    Args:
        snapshot_root: Scan a frozen snapshot instead of the live domain.
    """
    corpus = load_corpus(snapshot_root)
    document_ids = tuple(
        sorted(f.key.rsplit("/", 1)[-1][: -len(".json")] for f in corpus if f.kind == "document")
    )

    # Propose every candidate spelling first, then count them all through one
    # counter so that longest-match interaction is accounted for: occurrences
    # of "Green Account" inside "Dark Green Account" belong to the latter.
    proposals: list[Proposal] = [
        Proposal(
            "brand.primary", "brand", BRAND_ROOT, None, None,
            _candidate_forms(BRAND_ROOT, None, BRAND_STYLES),
        )
    ]
    for slug, display in product_roots(document_ids):
        proposals.append(
            Proposal(
                f"product.{slug}", "product", display, slug, None,
                _candidate_forms(display, slug, PRODUCT_STYLES),
            )
        )
    for name, kind in tool_names():
        # A tool's only spelling is its identifier.
        proposals.append(
            Proposal(f"tool.{name}", "tool", name, None, kind, {name: "literal"})
        )
    for name in IDENTIFIERS:
        proposals.append(
            Proposal(f"identifier.{name}", "identifier", name, None, None, {name: "literal"})
        )

    # Most proposed spellings never occur.  A plain substring test is a sound
    # prefilter -- every boundary-aware match is also a substring -- and it
    # cuts the alternation the regex has to carry by roughly 7x.
    matchable = {f.key: matchable_text(f) for f in corpus}
    blob = "\n".join(matchable.values())
    present = {
        form
        for proposal in proposals
        for form in proposal.candidates
        if form in blob
    }

    counter = FormCounter(present)
    totals: dict[str, int] = {}
    for corpus_file in corpus:
        for form, hits in counter.count(matchable[corpus_file.key]).items():
            totals[form] = totals.get(form, 0) + hits

    # A product name that is a boundary-substring of a longer product name
    # composes from it, so their new names stay consistent ("Business Silver
    # Rewards Card" must follow whatever "Silver Rewards Card" becomes).
    # The brand participates: "Rho Bank Plus" is a product named after the
    # bank, so its new name has to follow whatever the brand becomes.
    roots = {
        p.concept_id: p.root for p in proposals if p.kind in ("product", "brand")
    }
    bases: dict[str, Optional[str]] = {}
    for concept_id, root in roots.items():
        contained = [
            (other_id, other_root)
            for other_id, other_root in roots.items()
            if other_id != concept_id and _contains_at_boundary(root, other_root)
        ]
        bases[concept_id] = (
            max(contained, key=lambda item: len(item[1]))[0] if contained else None
        )

    concepts = []
    for proposal in proposals:
        forms = tuple(
            ObservedForm(form=form, style=style, count=totals.get(form, 0))
            for form, style in sorted(
                proposal.candidates.items(), key=lambda kv: (-len(kv[0]), kv[0])
            )
            if totals.get(form, 0) > 0
        )
        concepts.append(
            Concept(
                concept_id=proposal.concept_id,
                kind=proposal.kind,
                root=proposal.root,
                slug=proposal.slug,
                base=bases.get(proposal.concept_id),
                detail=proposal.detail,
                forms=forms,
            )
        )

    task_ids = tuple(
        sorted(json.loads(f.text)["id"] for f in corpus if f.kind == "task")
    )
    return Inventory(
        corpus=corpus,
        concepts=tuple(concepts),
        document_ids=document_ids,
        task_ids=task_ids,
    )


def unobserved_forms(concepts: Iterable[Concept]) -> dict[str, tuple[str, ...]]:
    """Spellings a concept could take but does not, for the leak denylist."""
    denied: dict[str, tuple[str, ...]] = {}
    for concept in concepts:
        if concept.kind == "tool":
            continue
        observed = {form.form for form in concept.forms}
        styles = BRAND_STYLES if concept.kind == "brand" else PRODUCT_STYLES
        candidates = set(_candidate_forms(concept.root, concept.slug, styles))
        extra = tuple(sorted(candidates - observed))
        if extra:
            denied[concept.concept_id] = extra
    return denied
