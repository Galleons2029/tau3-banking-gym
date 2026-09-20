"""Build a perturbation plan: assign every canonical symbol its variant name.

Naming happens at the level of *identity tokens*, not whole product names.  The
canonical line reuses a small vocabulary -- ``Green`` appears in nine product
names, ``Gold`` in eight -- and renaming each product independently would split
families that belong together (``Gold Account`` and ``Gold Plus Account`` are
the same product line; ``Business Silver Rewards Card`` must follow whatever
``Silver Rewards Card`` becomes).  Mapping the shared token once keeps all of
that consistent by construction, and makes composed names fall out for free.

Structural words (``Account``, ``Card``, ``Rewards``), generic modifiers
(``Light``, ``Navy``, ``Sky``) and plain feature descriptions (``Scheduled
Payments``) are deliberately left alone: renaming them would damage meaning
without removing anything memorisable, because they are ordinary English rather
than arbitrary symbols a model could only have learned from this corpus.
"""

import random
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

from tau3.perturb import naming, pools, scan
from tau3.perturb.models import ConceptMap, FormMap, PerturbationPlan, Recipe
from tau3.perturb.rewrite import Rewriter, build_rewriter
from tau3.perturb.storage import digest

#: Tier-like identity tokens, drawn from one lexical family.
TIER_TOKENS: tuple[str, ...] = (
    "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Emerald",
)

#: Colour-like identity tokens, drawn from another.
COLOR_TOKENS: tuple[str, ...] = (
    "Blue", "Green", "Purple", "Beige", "Bluest", "Evergreen",
)

#: Standalone product marks with no shared vocabulary.
MARK_TOKENS: tuple[str, ...] = ("EcoCard", "Crypto-Cash", "Zoom")

#: Every token that gets a new name.
IDENTITY_TOKENS: tuple[str, ...] = TIER_TOKENS + COLOR_TOKENS + MARK_TOKENS

#: Which pool family each identity group draws from.
TOKEN_FAMILY: dict[str, str] = {
    "tier": "mineral",
    "color": "botanical",
    "mark": "avian",
}

#: Words that carry structure or plain meaning and are never renamed.
KEPT_TOKENS: tuple[str, ...] = (
    # structural
    "Account", "Card", "Rewards", "Saver", "Plus", "Business", "Elite",
    "Vault", "Reserve", "Years", "Back", "Fee-Free", "BNPL", "(checking)",
    "(savings)",
    # generic modifiers
    "Light", "Dark", "Sky", "Navy", "True", "World", "Cobalt", "Hunter", "Lime",
    # plain feature descriptions
    "QR", "Transfers", "Scheduled", "Payments", "Sending", "Limits", "User",
    "Blocking", "Split-the-Bill",
)

_SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<digits>\d{4})$")


class PlanError(ValueError):
    """A plan could not be built consistently."""


def _draw(rng: random.Random, pool: Sequence[str], count: int, used: set[str]) -> list[str]:
    """Draw ``count`` distinct unused words from ``pool``."""
    available = [word for word in pool if word not in used]
    if len(available) < count:
        raise PlanError(
            f"Pool exhausted: need {count} names, {len(available)} available "
            "after exclusions. Add words to tau3.perturb.pools."
        )
    chosen = rng.sample(available, count)
    used.update(chosen)
    return chosen


def _token_map(rng: random.Random, used: set[str]) -> dict[str, str]:
    """Assign every identity token a replacement, family by family."""
    mapping: dict[str, str] = {}
    for group, tokens in (
        ("tier", TIER_TOKENS),
        ("color", COLOR_TOKENS),
        ("mark", MARK_TOKENS),
    ):
        family = pools.FAMILIES[TOKEN_FAMILY[group]]
        for token, replacement in zip(tokens, _draw(rng, family, len(tokens), used)):
            mapping[token] = replacement
    return mapping


def _rename_root(root: str, token_map: dict[str, str], brand: tuple[str, str]) -> str:
    """Rewrite a product root through the brand and the token map."""
    canonical_brand, new_brand = brand
    if canonical_brand in root:
        root = root.replace(canonical_brand, new_brand)
    return " ".join(token_map.get(token, token) for token in root.split(" "))


def _rename_tool(
    name: str, rng: random.Random, used_names: set[str], used_suffixes: set[str]
) -> str:
    """Rename a tool by its leading verb and its disambiguating suffix.

    The noun phrase is kept: a discoverable tool's docstring is parsed at
    runtime into the definition the agent sees, so the name has to keep
    describing what the tool actually does.  Changing the verb and the four
    digits still forces the exact callable name to be retrieved rather than
    recalled.
    """
    match = _SUFFIX.match(name)
    stem = match.group("stem") if match else name
    parts = stem.split("_")
    verb = parts[0]

    candidates = pools.TOOL_VERBS.get(verb, ())
    for replacement in list(candidates) + [verb]:
        renamed_stem = "_".join([replacement, *parts[1:]])
        if renamed_stem == stem and candidates:
            continue
        if match:
            for _ in range(64):
                digits = f"{rng.randint(1000, 9999)}"
                if digits in used_suffixes:
                    continue
                candidate = f"{renamed_stem}_{digits}"
                if candidate not in used_names:
                    used_suffixes.add(digits)
                    used_names.add(candidate)
                    return candidate
        elif renamed_stem not in used_names:
            used_names.add(renamed_stem)
            return renamed_stem
    raise PlanError(f"Could not find a free name for tool {name!r}")


def _rename_document(document_id: str, slug_map: dict[str, str]) -> str:
    """Rewrite a document id through the product slug it names."""
    match = scan._DOC_ID.match(document_id)
    if not match:
        raise PlanError(f"Unparseable document id: {document_id}")
    segment = match.group("segment")
    _, slug = scan._split_segment(segment)
    replacement = slug_map.get(slug)
    if replacement is None:
        return document_id
    return document_id.replace(slug, replacement, 1)


def _target_form(new_root: str, source: str, style: Optional[str]) -> str:
    """Render one spelling of a concept's new name."""
    if style in (None, "literal"):
        return new_root
    if style == "slug":
        return naming.render(new_root, "snake")
    return naming.render(new_root, style)


def build_plan(
    seed: int,
    *,
    snapshot_root: Path,
    recipe: Optional[Recipe] = None,
    exclude: Iterable[str] = (),
) -> PerturbationPlan:
    """Assign variant names to every canonical symbol.

    Args:
        seed: Drives every draw; the same seed and snapshot give the same plan.
        snapshot_root: Frozen corpus to plan against.
        recipe: Which axes to perturb.  Defaults to names-only.
        exclude: Names already taken by other variants, kept disjoint so a
            model trained on one variant cannot recognise another.

    Returns:
        A validated :class:`PerturbationPlan`.

    Raises:
        PlanError: If names collide or a pool is exhausted.
    """
    recipe = recipe or Recipe()
    from tau3.perturb import snapshot as snapshot_module

    snap = snapshot_module.load(snapshot_root)
    inventory = scan.inventory(snapshot_root)

    rng = random.Random(seed)
    used: set[str] = set(exclude)

    brand_root = scan.BRAND_ROOT
    new_brand = _draw(rng, pools.BRAND_ROOTS, 1, used)[0] if recipe.brand else brand_root
    token_map = _token_map(rng, used) if recipe.products else {}

    concepts: list[ConceptMap] = []

    def emit(concept: scan.Concept, new_root: str) -> None:
        if new_root == concept.root:
            return  # nothing renamed for this concept
        forms = [
            FormMap(
                source=form.form,
                target=_target_form(new_root, form.form, form.style),
                style=form.style,
                count=form.count,
            )
            for form in concept.forms
        ]
        forms = [form for form in forms if form.source != form.target]
        if not forms:
            return
        concepts.append(
            ConceptMap(
                concept_id=concept.concept_id,
                kind=concept.kind,
                root=concept.root,
                new_root=new_root,
                slug=concept.slug,
                base=concept.base,
                detail=concept.detail,
                forms=forms,
            )
        )

    if recipe.brand:
        emit(inventory.concept("brand.primary"), new_brand)

    if recipe.products:
        for concept in inventory.by_kind("product"):
            emit(concept, _rename_root(concept.root, token_map, (brand_root, new_brand)))

    if recipe.products:
        # Document ids must be their own concepts.  A product slug sits inside
        # an id surrounded by underscores -- word characters -- so the boundary
        # rule correctly refuses to match it there.  Renaming the whole id as
        # one token is what carries the 933 ``required_documents`` references.
        slug_map = {
            concept.slug: naming.render(concept.new_root, "snake")
            for concept in concepts
            if concept.kind == "product" and concept.slug
        }
        for document_id in inventory.document_ids:
            new_id = _rename_document(document_id, slug_map)
            if new_id == document_id:
                continue
            concepts.append(
                ConceptMap(
                    concept_id=f"document.{document_id}",
                    kind="document",
                    root=document_id,
                    new_root=new_id,
                    forms=[
                        FormMap(source=document_id, target=new_id, style="literal", count=0)
                    ],
                )
            )

    if recipe.identifiers:
        brand_snake = naming.render(brand_root, "snake")
        new_brand_snake = naming.render(new_brand, "snake")
        for concept in inventory.by_kind("identifier"):
            emit(concept, concept.root.replace(brand_snake, new_brand_snake))

    if recipe.tools:
        used_names = {concept.root for concept in inventory.by_kind("tool")}
        used_suffixes: set[str] = set()
        for name in sorted(used_names):
            suffix = _SUFFIX.match(name)
            if suffix:
                used_suffixes.add(suffix.group("digits"))
        for concept in inventory.by_kind("tool"):
            emit(concept, _rename_tool(concept.root, rng, used_names, used_suffixes))

    concepts = _recount(concepts, snapshot_root)
    _validate(concepts, inventory)

    variant_id = digest(
        [seed, recipe.model_dump(), snap.digest, sorted(exclude)]
    )[:8]
    plan = PerturbationPlan(
        variant_id=variant_id,
        domain_name=f"banking_knowledge__{variant_id}",
        package_name=f"tau3_bk_{variant_id}",
        corpus_digest=snap.digest,
        seed=seed,
        recipe=recipe,
        token_map=token_map,
        concepts=concepts,
        frozen_tools=sorted(scan.FROZEN_TOOLS),
        kept_tokens=sorted(KEPT_TOKENS),
        expected={
            "document_count": len(inventory.document_ids),
            "task_count": len(inventory.task_ids),
            "renamed_concepts": len(concepts),
            "total_replacements": sum(c.total for c in concepts),
        },
    )
    payload = plan.model_dump(exclude={"plan_digest"})
    plan.plan_digest = digest(payload)
    return plan


def _recount(concepts: list[ConceptMap], snapshot_root: Path) -> list[ConceptMap]:
    """Replace planned counts with what the real rewrite actually does.

    The plan and the build then share one code path, so a count assertion at
    build time means what it should -- the corpus is unchanged and the rewrite
    is deterministic -- rather than re-deriving occurrences a second way and
    disagreeing with itself over JSON escaping.

    Forms that match nothing are dropped; a form that never occurs cannot
    shadow another, so removing it leaves every remaining count untouched.
    """
    from tau3.perturb.materialize import rewrite_corpus

    pairs = [(form.source, form.target) for c in concepts for form in c.forms]
    values = build_rewriter(pairs)
    keys = build_rewriter(
        [
            (form.source, form.target)
            for c in concepts
            if c.kind == "identifier"
            for form in c.forms
        ]
    )
    _, hits = rewrite_corpus(scan.load_corpus(snapshot_root), values, keys)

    recounted: list[ConceptMap] = []
    for concept in concepts:
        forms = [
            form.model_copy(update={"count": hits.get(form.source, 0)})
            for form in concept.forms
            if hits.get(form.source, 0) > 0
        ]
        if forms:
            recounted.append(concept.model_copy(update={"forms": forms}))
    return recounted


def _validate(concepts: Sequence[ConceptMap], inventory: scan.Inventory) -> None:
    """Fail closed on any inconsistency a plan could carry into a build."""
    pairs = [(form.source, form.target) for c in concepts for form in c.forms]
    # The rewriter's own collision rules do the heavy lifting: duplicate
    # sources, two sources sharing a target, and any target that is also a
    # canonical symbol are all rejected here.
    build_rewriter(pairs)

    new_roots: dict[str, str] = {}
    for concept in concepts:
        clash = new_roots.get(concept.new_root)
        if clash is not None:
            raise PlanError(
                f"{concept.root!r} and {clash!r} would both become {concept.new_root!r}"
            )
        new_roots[concept.new_root] = concept.root

    # A composed product must agree with the concept it composes from, or the
    # same product would end up with two names in different files.
    by_id = {concept.concept_id: concept for concept in concepts}
    for concept in concepts:
        if not concept.base or concept.base not in by_id:
            continue
        base = by_id[concept.base]
        expected = concept.root.replace(base.root, base.new_root)
        if concept.new_root != expected:
            raise PlanError(
                f"Composed name is inconsistent: {concept.root!r} became "
                f"{concept.new_root!r}, but {base.root!r} became {base.new_root!r} "
                f"(expected {expected!r})"
            )
