"""Stage C: decide which document reveals which facts, before any text exists.

This is the step that sets how hard retrieval is. Knowledge is deliberately
fragmented: a task's evidence is spread across several documents so that no
single retrieval answers it, and roughly a third of variables appear twice so a
single miss is not fatal. The anti-aggregation bound is the knob that decides
how many documents a task needs -- tighten it and gold sets grow, loosen it and
the task collapses into one lookup.
"""

import hashlib
import math
from dataclasses import dataclass

from tau3.worldgen.models import ARCHETYPES, DocPlan, Rule, Variable, WorldSchema

# How many variables a document of each archetype carries.
CAPACITY = {
    "product_overview": (5, 9),
    "faq": (3, 6),
    "internal_protocol": (0, 3),
    "tool_doc": (0, 2),
    "program_terms": (2, 5),
    "howto_short": (1, 2),
    "eligibility_matrix": (2, 4),
    "promo_notice": (1, 3),
}
# Archetypes that may carry internal-only material and tool signatures.
INTERNAL_ARCHETYPES = {"internal_protocol", "tool_doc", "eligibility_matrix"}
# Mean number of documents a variable appears in. Above one so a single missed
# retrieval is survivable, well below two so it is not free.
TARGET_DEGREE = 1.35
# How much prose an archetype spends on one fact, relative to a plain listing.
# An overview explains, a how-to answers one question, a table states.
PROSE_PER_FACT = {
    "product_overview": 1.4,
    "faq": 1.3,
    "internal_protocol": 1.6,
    "tool_doc": 1.4,
    "program_terms": 1.5,
    "howto_short": 1.6,
    "eligibility_matrix": 1.2,
    "promo_notice": 1.4,
}
# Share of variables that appear in a second document.
DUPLICATE_SHARE = 0.35
# Of those, how many are a cross-reference rather than a repeated value.
CROSSREF_SHARE = 0.30

TITLES = {
    "product_overview": "{entity} at a Glance",
    "faq": "{entity} Questions and Answers",
    "howto_short": "How do I check my {entity} {topic}?",
    "program_terms": "{entity} Programme Terms",
    "eligibility_matrix": "{entity} Eligibility",
    "promo_notice": "{entity} Limited-Time Offer",
    "internal_protocol": "Internal: {entity} Procedure",
    "tool_doc": "Internal: {entity} Tooling",
}


@dataclass
class AllocationReport:
    """What the allocation produced, so its shape can be checked at a glance."""

    documents: int
    archetypes: dict[str, int]
    mean_degree: float
    duplicated: int
    crossrefs: int
    uncovered: list[str]
    tools_bound: int
    tools_unbound: list[str]


def _order(items, salt: str):
    """A stable shuffle: allocation must not drift between runs of one seed."""
    return sorted(
        items, key=lambda item: hashlib.sha256(f"{salt}|{item}".encode()).hexdigest()
    )


def archetype_quota(total: int) -> dict[str, int]:
    """Split a document budget across archetypes in the corpus's proportions."""
    quota = {name: int(total * spec["share"]) for name, spec in ARCHETYPES.items()}
    # Largest remainders take the rounding slack, so the total is exact.
    remainder = total - sum(quota.values())
    by_loss = sorted(
        ARCHETYPES,
        key=lambda name: total * ARCHETYPES[name]["share"] - quota[name],
        reverse=True,
    )
    for name in by_loss[:remainder]:
        quota[name] += 1
    return quota


def _product_archetypes(quota: dict[str, int]) -> list[str]:
    """Archetypes available to product documents, most plentiful first."""
    return [name for name in quota if name not in {"internal_protocol", "tool_doc"}]


def length_scale(quota: dict[str, int], corpus_mean: float) -> float:
    """Scale archetype lengths so the mix reproduces the measured corpus mean.

    The per-archetype figures come from the design document, which calibrated
    them against the paper's reported average. The corpus installed here is
    shorter, and a mix that reproduces the paper misses the corpus by more than
    the acceptance band allows.
    """
    planned = sum(ARCHETYPES[name]["tokens"] * count for name, count in quota.items())
    documents = sum(quota.values())
    if not documents or not planned:
        return 1.0
    return corpus_mean / (planned / documents)


def allocate(
    schema: WorldSchema,
    tools: list[dict],
    total_documents: int,
    salt: str,
    protected: dict[str, set[str]] | None = None,
    leak_divisor: int = 3,
    corpus_mean: float | None = None,
) -> tuple[list[DocPlan], AllocationReport]:
    """Spread variables, rules and tool signatures across a document budget.

    `protected` maps a task or family id to the knowledge it needs, so the
    anti-aggregation bound can be enforced while the allocation is still being
    built rather than rejected afterwards by a linter.
    """
    protected = protected or {}
    quota = archetype_quota(total_documents)
    scale = length_scale(quota, corpus_mean) if corpus_mean else 1.0
    documents: list[DocPlan] = []

    products = [f for f in schema.features if f.entity_kind in {"account", "card"}]
    protocols = [f for f in schema.features if f.entity_kind == "protocol"]
    # Internal documents are bounded by how many tools there are to write about:
    # at full scale the archetype mix asks for more procedures than the tool set
    # can fill, and the leftover budget belongs to the product documents rather
    # than being silently dropped from the corpus.
    tool_budget = min(quota["internal_protocol"] + quota["tool_doc"], len(tools))
    product_budget = total_documents - tool_budget

    # ---- product documents -------------------------------------------------
    # Documents are shaped first and filled second. Filling each to capacity in
    # turn would let one overview absorb a product's whole attribute set and
    # leave the rest of the budget with nothing to say.
    shapes: list[tuple[str, object]] = []
    available = _product_archetypes(quota)
    if products:
        # Scale the product archetypes up together so the mix is preserved while
        # the reclaimed internal budget is used.
        planned = sum(quota[name] for name in available) or 1
        scale = product_budget / planned
        for archetype in available:
            for slot in range(max(1, round(quota[archetype] * scale))):
                shapes.append((archetype, products[len(shapes) % len(products)]))
    shapes = shapes[:product_budget]

    pools: dict[str, list[str]] = {}
    for feature in products:
        variables = [
            v for v in schema.variables if v.feature_id == feature.id and not v.derived
        ]
        customer = _order(
            [v.id for v in variables if v.visibility == "customer_facing"], salt
        )
        internal = [v.id for v in variables if v.visibility == "internal_only"]
        pools[feature.id] = customer + internal

    # How many facts each document may carry is derived from how many its
    # feature actually has. Filling to the archetype's ceiling regardless would
    # force the pool to be recycled, and a corpus where every fact appears three
    # times makes a single missed retrieval harmless -- which is the opposite of
    # what the fragmentation is for.
    wants: dict[int, int] = {}
    for feature in products:
        indices = [
            i
            for i, (_, f) in enumerate(shapes)
            if f.id is feature.id or f.id == feature.id
        ]
        ceilings = [CAPACITY[shapes[i][0]][1] for i in indices]
        total_ceiling = sum(ceilings) or 1
        budget = max(len(pools[feature.id]) * TARGET_DEGREE, float(len(indices)))
        for index, ceiling in zip(indices, ceilings, strict=True):
            share = budget * ceiling / total_ceiling
            wants[index] = max(1, min(ceiling, int(round(share))))

    cursors = dict.fromkeys(pools, 0)
    counters: dict[str, int] = {}
    for position, (archetype, feature) in enumerate(shapes):
        pool = pools[feature.id]
        if not pool:
            continue
        want = wants.get(position, 1)

        chosen: list[str] = []
        attempts = 0
        while len(chosen) < want and attempts < len(pool):
            variable_id = pool[cursors[feature.id] % len(pool)]
            cursors[feature.id] += 1
            attempts += 1
            if variable_id in chosen:
                continue
            if _would_over_reveal(variable_id, chosen, protected, leak_divisor):
                continue
            chosen.append(variable_id)
        if not chosen:
            continue

        counters[feature.id] = counters.get(feature.id, 0) + 1
        stem = feature.id.removeprefix("feat_")
        # Length follows content here too: an overview asked for four hundred
        # tokens about five facts either pads or comes in short, and the length
        # band then fails a document that is otherwise correct.
        ceiling = ARCHETYPES[archetype]["tokens"] * scale
        documents.append(
            DocPlan(
                doc_id=f"doc_{feature.category_id.removeprefix('cat_')}_{stem}_{counters[feature.id]:03d}",
                title=_title(archetype, feature.entity_name, chosen, schema),
                archetype=archetype,
                feature_id=feature.id,
                variable_ids=chosen,
                target_tokens=max(
                    90,
                    int(
                        min(ceiling, 70 + 45 * len(chosen) * PROSE_PER_FACT[archetype])
                    ),
                ),
            )
        )

    # Every variable must appear at least once, whatever the shapes allowed.
    placed = {v for document in documents for v in document.variable_ids}
    for feature in products:
        missing = [v for v in pools[feature.id] if v not in placed]
        for variable_id in missing:
            counters[feature.id] = counters.get(feature.id, 0) + 1
            stem = feature.id.removeprefix("feat_")
            documents.append(
                DocPlan(
                    doc_id=f"doc_{feature.category_id.removeprefix('cat_')}_{stem}_{counters[feature.id]:03d}",
                    title=_title(
                        "howto_short", feature.entity_name, [variable_id], schema
                    ),
                    archetype="howto_short",
                    feature_id=feature.id,
                    variable_ids=[variable_id],
                    target_tokens=int(ARCHETYPES["howto_short"]["tokens"] * scale),
                )
            )

    # ---- internal documents carrying rules and tool signatures -------------
    documents.extend(
        _tool_documents(
            schema,
            tools,
            protocols,
            quota,
            salt,
            documents,
            scale,
            protected,
            leak_divisor,
        )
    )

    # ---- redundancy: a share of variables appears a second time ------------
    duplicated, crossrefs = _add_redundancy(
        documents, schema, salt, protected, leak_divisor
    )
    if corpus_mean and documents:
        # Rescale against the mix that was actually built. The quota is only a
        # plan: uncovered variables add short documents at the end, and scaling
        # the plan leaves the realized corpus below its target.
        realized = sum(d.target_tokens for d in documents) / len(documents)
        if realized:
            correction = corpus_mean / realized
            for document in documents:
                document.target_tokens = max(
                    1, int(document.target_tokens * correction)
                )

    counted: dict[str, int] = {}
    for document in documents:
        for variable_id in document.variable_ids:
            counted[variable_id] = counted.get(variable_id, 0) + 1
    all_variables = [v.id for v in schema.variables if not v.derived]
    uncovered = [v for v in all_variables if v not in counted]

    mentioned = {tool for document in documents for tool in document.tool_ids}
    unbound = [t["official_name"] for t in tools if t["official_name"] not in mentioned]

    report = AllocationReport(
        documents=len(documents),
        archetypes={
            name: sum(1 for d in documents if d.archetype == name)
            for name in ARCHETYPES
        },
        mean_degree=(sum(counted.values()) / len(counted)) if counted else 0.0,
        duplicated=duplicated,
        crossrefs=crossrefs,
        uncovered=uncovered,
        tools_bound=len(mentioned),
        tools_unbound=unbound,
    )
    return documents, report


def _next_archetype(available: list[str], remaining: dict[str, int], index: int) -> str:
    """Take from the archetype with budget left, rotating for variety."""
    for offset in range(len(available)):
        name = available[(index + offset) % len(available)]
        if remaining.get(name, 0) > 0:
            remaining[name] -= 1
            return name
    return "faq"


def _would_over_reveal(
    variable_id: str,
    chosen: list[str],
    protected: dict[str, set[str]],
    leak_divisor: int,
) -> bool:
    """Whether adding this variable would let one document answer a task."""
    for required in protected.values():
        if variable_id not in required:
            continue
        budget = math.ceil(len(required) / leak_divisor)
        if len(set(chosen) & required) + 1 > budget:
            return True
    return False


def _title(archetype: str, entity: str, chosen: list[str], schema: WorldSchema) -> str:
    """A title in the style of the archetype it belongs to."""
    topic = "details"
    if chosen:
        try:
            topic = schema.variable(chosen[0]).name.replace("_", " ")
        except KeyError:
            pass
    return TITLES[archetype].format(entity=entity, topic=topic)


def _tool_documents(
    schema: WorldSchema,
    tools: list[dict],
    protocols: list,
    quota: dict[str, int],
    salt: str,
    existing: list[DocPlan],
    scale: float = 1.0,
    protected: dict[str, set[str]] | None = None,
    leak_divisor: int = 3,
) -> list[DocPlan]:
    """Internal documents that carry policy rules and tool signatures.

    Every discoverable tool must be named somewhere, or an agent can never
    unlock it however well it searches.
    """
    budget = max(1, min(quota["internal_protocol"] + quota["tool_doc"], len(tools)))
    ordered = _order([t["official_name"] for t in tools], salt)
    by_name = {t["official_name"]: t for t in tools}
    per_document = math.ceil(len(ordered) / budget)

    rules = list(schema.rules)
    documents: list[DocPlan] = []
    # Batching by position alone put every tool an ordering task needs into one
    # procedure, which is the whole task in a single retrieval. Tools are placed
    # one at a time so the same anti-aggregation bound applies here.
    pending = list(ordered)
    for index in range(budget):
        batch: list[str] = []
        deferred: list[str] = []
        while pending and len(batch) < per_document:
            candidate = pending.pop(0)
            if _would_over_reveal(candidate, batch, protected or {}, leak_divisor):
                deferred.append(candidate)
                continue
            batch.append(candidate)
        pending = deferred + pending
        if not batch:
            break
        archetype = by_name[batch[0]]["document_archetype"]
        # Deliberately unowned: a procedure covering closure tools is not about
        # the "Account Opening" feature it would otherwise be assigned by
        # rotation, and the retrievability check would then demand it name the
        # wrong subject. These documents are found by the tool names they carry.
        # A rule counts towards the same bound as a tool: a procedure holding one
        # tool and the policy that governs it is still two thirds of an ordering
        # task in a single retrieval.
        rule = None
        for candidate in list(rules):
            if not _would_over_reveal(
                candidate.id, batch, protected or {}, leak_divisor
            ):
                rule = candidate
                rules.remove(candidate)
                break
        # Length follows content. A long target on a document with nothing to
        # state leaves a model padding with procedural detail it has to invent,
        # which the placeholder gate then rejects over and over.
        load = len(batch) + (1 if rule else 0)
        ceiling = ARCHETYPES[archetype]["tokens"] * scale
        documents.append(
            DocPlan(
                doc_id=f"doc_protocol_{index + 1:03d}",
                title=_tool_title(archetype, batch, by_name),
                archetype=archetype,
                feature_id=None,
                tool_ids=batch,
                rule_ids=[rule.id] if rule else [],
                target_tokens=max(90, int(min(ceiling, 80 + 90 * load))),
            )
        )
    for leftover in pending:
        documents.append(
            DocPlan(
                doc_id=f"doc_protocol_{len(documents) + 1:03d}",
                title=_tool_title("tool_doc", [leftover], by_name),
                archetype="tool_doc",
                feature_id=None,
                tool_ids=[leftover],
                target_tokens=max(
                    90, int(min(ARCHETYPES["tool_doc"]["tokens"] * scale, 170))
                ),
            )
        )
    del existing
    return documents


def _tool_title(archetype: str, batch: list[str], by_name: dict) -> str:
    """Name an internal document after the action its tools perform."""
    lead = batch[0]
    action = " ".join(part.capitalize() for part in lead.split("_")[:-1]) or "Servicing"
    del by_name
    if archetype == "tool_doc":
        return f"Internal: {action} Tooling"
    return f"Internal: {action} Procedure"


def _add_redundancy(
    documents: list[DocPlan],
    schema: WorldSchema,
    salt: str,
    protected: dict[str, set[str]],
    leak_divisor: int,
) -> tuple[int, int]:
    """Repeat a share of variables, some as values and some as pointers."""
    placements: dict[str, list[DocPlan]] = {}
    for document in documents:
        for variable_id in document.variable_ids:
            placements.setdefault(variable_id, []).append(document)

    singles = _order([v for v, docs in placements.items() if len(docs) == 1], salt)
    # Top up towards the target degree rather than adding a fixed share on top:
    # the capacity pass already places most variables more than once, and adding
    # unconditionally would push the corpus past the redundancy it is meant to
    # have.
    placed = sum(len(docs) for docs in placements.values())
    shortfall = int(round(len(placements) * TARGET_DEGREE)) - placed
    wanted = max(0, min(len(singles), shortfall))
    duplicated = crossrefs = 0

    for position, variable_id in enumerate(singles[: wanted + len(singles) // 5]):
        if duplicated >= wanted and position % 10 >= CROSSREF_SHARE * 10:
            continue
        source = placements[variable_id][0]
        target = _second_home(documents, source, variable_id, protected, leak_divisor)
        if target is None:
            continue
        if position % 10 < CROSSREF_SHARE * 10:
            if source.doc_id not in target.crossrefs:
                target.crossrefs.append(source.doc_id)
                crossrefs += 1
        else:
            target.variable_ids.append(variable_id)
            duplicated += 1
    return duplicated, crossrefs


def _second_home(
    documents: list[DocPlan],
    source: DocPlan,
    variable_id: str,
    protected: dict[str, set[str]],
    leak_divisor: int,
) -> DocPlan | None:
    """A different document of the same feature that can take one more fact."""
    for document in documents:
        if document is source or document.feature_id != source.feature_id:
            continue
        if variable_id in document.variable_ids:
            continue
        low, high = CAPACITY[document.archetype]
        if len(document.variable_ids) >= high:
            continue
        if _would_over_reveal(
            variable_id, document.variable_ids, protected, leak_divisor
        ):
            continue
        return document
    return None


def rules_for(schema: WorldSchema) -> list[Rule]:
    """Policy rules the allocation has to place."""
    return list(schema.rules)


def variables_of(schema: WorldSchema, feature_id: str) -> list[Variable]:
    """Non-derived variables of one feature."""
    return [v for v in schema.variables if v.feature_id == feature_id and not v.derived]
