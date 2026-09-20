"""Scale is a setting: the deterministic stages have to hold at any size."""

import time

import pytest

from tau3.worldgen.allocate import allocate
from tau3.worldgen.models import Feature, WorldSchema
from tau3.worldgen.runtime import discoverable_tool_names, suffix_alias_map
from tau3.worldgen.schema_gen import offline_skeletons, to_variables
from tau3.worldgen.slice import build_families, build_products
from tau3.worldgen.solver import solve
from tau3.worldgen.tasks_gen import prospective_requirements


def tool_bindings():
    aliases = suffix_alias_map(discoverable_tool_names(), "scale-salt")
    return [
        {
            "official_name": official,
            "alias": alias,
            "document_archetype": "internal_protocol",
            "arguments": [],
        }
        for alias, official in aliases.items()
    ]


def build_world(topics: int):
    """A world of the given size, using offline skeletons so no model is called."""
    products = build_products(topics)
    families = build_families(products)
    features, variables, ranges = [], [], {}
    for spec in products:
        skeletons = offline_skeletons(spec)
        product_variables = to_variables(spec, skeletons)
        for skeleton, variable in zip(skeletons, product_variables, strict=True):
            ranges[variable.id] = (skeleton.plausible_min, skeleton.plausible_max)
        variables.extend(product_variables)
        features.append(
            Feature(
                id=spec.feature_id,
                category_id=spec.category_id,
                entity_name=spec.entity_name,
                entity_kind="card" if spec.kind == "card" else "account",
                variable_ids=[v.id for v in product_variables],
            )
        )
    schema = WorldSchema(features=features, variables=variables)
    # The solved families are what matters: a trap family's discriminator is a
    # derived value the solver adds, and the raw constraints alone do not single
    # out anything.
    assignment = solve(schema, families, ranges, 42, products)
    return schema, products, assignment.families


@pytest.mark.parametrize("topics", [11, 71, 300])
def test_products_and_families_grow_without_collisions(topics):
    products = build_products(topics)
    families = build_families(products)
    assert len({p.feature_id for p in products}) == len(products)
    assert len({p.entity_name for p in products}) == len(products)
    # The customer names no shortlist, only the tier they are shopping in, so a
    # family ranges over a whole tier and its answer must be unique across it.
    # Exactly two of those are placed one constraint away; the rest are clearly
    # out.
    assert families
    assert all(f.answer in f.candidates for f in families)
    assert all(len(f.near_misses) == 2 for f in families)
    assert all(f.answer not in f.near_misses for f in families)
    by_segment = {}
    for product in products:
        by_segment.setdefault((product.kind, product.segment), []).append(product)
    for family in families:
        answer = next(p for p in products if p.feature_id == family.answer)
        assert family.segment == answer.segment
        assert len(family.candidates) == len(by_segment[(answer.kind, family.segment)])

    # Tiers partition a kind: two families that shared a product would be
    # setting the same values against each other and each would overwrite the
    # other's answer.
    seen: set[str] = set()
    for family in families:
        assert not seen & set(family.candidates)
        seen |= set(family.candidates)
    # A tier has to be a range rather than a shortlist, or the scope it puts on
    # a task is doing no work.
    assert all(len(f.candidates) >= 3 for f in families)


def test_all_three_trap_kinds_survive_scaling():
    families = build_families(build_products(300))
    # A bigger world must carry all three traps, not more of the easiest one.
    assert {f.kind for f in families} == {"selection", "promotional", "temporal"}


def test_layering_buys_families_rather_than_bigger_ones():
    """Scale must add scenarios, not repeat one scenario at more customers.

    Every task of a product kind sharing one answer and one evidence set is what
    drove gold-document reuse to twice the corpus being imitated.
    """
    small = build_families(build_products(11))
    large = build_families(build_products(71))
    assert len({f.segment for f in small}) == 1
    assert len({f.segment for f in large}) > 1
    assert len(large) > len(small)
    # Distinct answers, not the same product reached from more directions.
    assert len({f.answer for f in large}) == len(large)


@pytest.mark.parametrize("topics,documents", [(71, 698), (300, 3000)])
def test_allocation_holds_its_invariants_at_scale(topics, documents):
    schema, products, families = build_world(topics)
    payload = [
        {
            "family_id": f.family_id,
            "candidates": f.candidates,
            "answer": f.answer,
            "constraints": f.constraints,
        }
        for f in families
    ]
    protected = prospective_requirements(schema, payload)

    started = time.monotonic()
    plans, report = allocate(
        schema, tool_bindings(), documents, "scale-salt", protected, 6
    )
    elapsed = time.monotonic() - started

    assert report.documents == documents
    assert report.uncovered == []
    # Every discoverable tool is named somewhere, or it can never be unlocked.
    assert report.tools_unbound == []
    assert len({p.doc_id for p in plans}) == len(plans)
    # The deterministic stages are not where scaling costs anything; the model
    # calls are. A budget here catches an accidental quadratic.
    assert elapsed < 30, f"allocation took {elapsed:.1f}s at {documents} documents"


def test_scaling_costs_grow_no_worse_than_linearly():
    schema, _, families = build_world(300)
    payload = [
        {
            "family_id": f.family_id,
            "candidates": f.candidates,
            "answer": f.answer,
            "constraints": f.constraints,
        }
        for f in families
    ]
    protected = prospective_requirements(schema, payload)
    tools = tool_bindings()

    timings = {}
    for documents in (1500, 3000):
        started = time.monotonic()
        allocate(schema, tools, documents, "scale-salt", protected, 6)
        timings[documents] = time.monotonic() - started
    # Doubling the corpus must not cost far more than double the work.
    assert timings[3000] < max(timings[1500] * 6, 1.0)


@pytest.mark.parametrize("topics", [71, 600, 3000])
def test_product_names_never_contain_a_digit(topics):
    import re

    # A digit in a product name is not cosmetic. Documents state the name, and
    # the renderer's guard against invented figures cannot tell that digit from
    # a fee the model made up -- which rejected a third of a full-scale corpus
    # before names were composed instead of numbered.
    names = [p.entity_name for p in build_products(topics)]
    assert not [n for n in names if re.search(r"\d", n)]
    assert len(set(names)) == len(names)


def test_running_out_of_names_fails_loudly():
    from tau3.worldgen import slice as slice_module

    original = slice_module.MODIFIERS
    try:
        slice_module.MODIFIERS = ["Only"]
        with pytest.raises(ValueError, match="Name supply exhausted"):
            # Two products sharing a name would share documents and quietly ruin
            # every task that has to tell them apart.
            build_products(300)
    finally:
        slice_module.MODIFIERS = original


@pytest.mark.parametrize("topics", [40, 120])
def test_the_answer_is_unique_across_the_whole_catalogue(topics):
    """A customer who names no shortlist is comparing everything.

    Before the instructions stopped naming products, uniqueness was only ever
    proved against two chosen rivals -- and a measurement over a real world
    found six products satisfying every constraint. A task with six valid
    answers is not a task.
    """
    from tau3.worldgen.linters import solutions
    from tau3.worldgen.models import TaskSpec

    schema, _products, families = build_world(topics)
    for family in families:
        spec = TaskSpec(
            task_id=family.family_id,
            archetype="selection",
            candidate_features=family.candidates,
            near_misses=family.near_misses,
            constraints=family.constraints,
            unique_answer=family.answer,
        )
        assert solutions(schema, spec) == [family.answer], family.family_id


@pytest.mark.parametrize("topics", [40, 120])
def test_distractors_are_shaped_to_the_kind_of_family_they_belong_to(topics):
    """A selection family and a trap family reject products differently.

    In a selection family two products are placed one constraint away and the
    rest are clearly out, so the agent has a real comparison to make but not an
    ambiguous one. In a trap family every product clears the stated constraints
    and only a value that has to be worked out -- a total rate, or a rate that
    depends on today's date -- separates them; that nothing looks wrong on the
    surface is the trap.
    """
    from tau3.worldgen.linters import satisfies

    schema, _products, families = build_world(topics)
    for family in families:
        failures = [
            sum(1 for c in family.constraints if not satisfies(schema, feature_id, c))
            for feature_id in family.candidates
            if feature_id != family.answer
        ]
        assert all(count >= 1 for count in failures), family.family_id
        if family.kind == "selection":
            assert failures.count(1) == 2, family.family_id
            assert all(count >= 2 for count in failures if count != 1)
        else:
            # Only the derived discriminator rejects anything.
            assert all(count == 1 for count in failures), family.family_id
