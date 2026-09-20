"""Stage A2: well-posed tasks, real near misses, and traps that actually bite."""

import pytest

from tau3.worldgen.linters import (
    check_near_misses,
    check_official_name_collisions,
    check_unique_answers,
    satisfies,
    solutions,
)
from tau3.worldgen.models import Feature, TaskSpec, Variable, WorldSchema
from tau3.worldgen.schema_gen import (
    VariableSkeleton,
    merge_required,
    offline_skeletons,
    snake_case,
    to_variables,
)
from tau3.worldgen.slice import FAMILIES, PRODUCTS, REQUIRED_VARIABLES
from tau3.worldgen.solver import solve


def build_unassigned() -> tuple[WorldSchema, dict[str, tuple[float, float]]]:
    """The slice as Stage A1 leaves it: shapes with no values."""
    features, variables, ranges = [], [], {}
    for spec in PRODUCTS:
        skeletons = offline_skeletons(spec)
        # A generated product carries far more attributes than the families
        # constrain. Without one here every value would be pinned by a
        # constraint and the seed would have nothing left to vary, which is not
        # what a real world looks like.
        skeletons.append(
            VariableSkeleton(
                name="statement_delivery_fee",
                type="currency",
                unit="USD",
                plausible_min=0.0,
                plausible_max=25.0,
            )
        )
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
    return WorldSchema(features=features, variables=variables), ranges


@pytest.fixture
def solved():
    schema, ranges = build_unassigned()
    assignment = solve(schema, FAMILIES, ranges, 42, PRODUCTS)
    return schema, assignment


def specs_from(assignment):
    return [
        TaskSpec(
            task_id=family.family_id,
            archetype="selection",
            candidate_features=family.candidates,
            constraints=family.constraints,
            unique_answer=family.answer,
        )
        for family in assignment.families
    ]


def test_every_value_is_assigned(solved):
    schema, _ = solved
    assert not [v.id for v in schema.variables if v.value is None]


def test_every_family_has_exactly_one_answer(solved):
    schema, assignment = solved
    for spec, family in zip(specs_from(assignment), assignment.families, strict=True):
        assert solutions(schema, spec) == [family.answer]
    assert check_unique_answers(schema, specs_from(assignment)).passed


def test_every_distractor_fails_exactly_one_constraint(solved):
    schema, assignment = solved
    report = check_near_misses(schema, specs_from(assignment))
    assert report.passed, [f.message for f in report.findings]


def test_all_three_trap_classes_are_instantiated(solved):
    _, assignment = solved
    kinds = {family.kind for family in assignment.families}
    assert kinds == {"selection", "promotional", "temporal"}
    assert all(family.trap for family in assignment.families)


def test_the_advertised_boost_cannot_close_the_base_rate_gap(solved):
    schema, assignment = solved
    family = next(f for f in assignment.families if f.kind == "promotional")
    answer_stem = family.answer.removeprefix("feat_")
    answer_total = float(schema.variable(f"var_{answer_stem}_effective_apy").value)
    for feature_id in family.candidates:
        if feature_id == family.answer:
            continue
        stem = feature_id.removeprefix("feat_")
        base = float(schema.variable(f"var_{stem}_base_apy").value)
        boost = float(schema.variable(f"var_{stem}_promotional_apy_boost").value)
        # Chasing the promotion is the losing move, and the rival still clears
        # every constraint the customer actually stated.
        assert base + boost < answer_total
        assert boost > 0
        assert satisfies(schema, feature_id, "base_apy >= 3.0")


def test_the_higher_headline_promotion_is_the_expired_one(solved):
    schema, assignment = solved
    family = next(f for f in assignment.families if f.kind == "temporal")
    loser = next(c for c in family.candidates if c != family.answer)
    winner_promo = schema.variable(
        f"var_{family.answer.removeprefix('feat_')}_promotional_cashback_rate"
    )
    loser_promo = schema.variable(
        f"var_{loser.removeprefix('feat_')}_promotional_cashback_rate"
    )
    assert float(loser_promo.value) > float(winner_promo.value)
    assert loser_promo.window["end"] < "11/14/2025".replace("/", "/")
    assert winner_promo.volatility == loser_promo.volatility == "promotional"


def test_derived_values_are_marked_and_never_customer_facing(solved):
    schema, _ = solved
    derived = [v for v in schema.variables if v.derived]
    assert derived
    assert all(v.visibility == "internal_only" for v in derived)


def test_assignment_is_deterministic_under_the_seed():
    first, ranges = build_unassigned()
    second, _ = build_unassigned()
    solve(first, FAMILIES, ranges, 42, PRODUCTS)
    solve(second, FAMILIES, ranges, 42, PRODUCTS)
    assert {v.id: v.value for v in first.variables} == {
        v.id: v.value for v in second.variables
    }

    third, _ = build_unassigned()
    solve(third, FAMILIES, ranges, 43, PRODUCTS)
    assert {v.id: v.value for v in third.variables} != {
        v.id: v.value for v in first.variables
    }


def test_solving_twice_does_not_duplicate_derived_values():
    schema, ranges = build_unassigned()
    solve(schema, FAMILIES, ranges, 42, PRODUCTS)
    once = len(schema.variables)
    solve(schema, FAMILIES, ranges, 42, PRODUCTS)
    assert len(schema.variables) == once


def test_slice_names_do_not_reuse_the_official_corpus(solved):
    schema, _ = solved
    assert check_official_name_collisions(schema).passed


def test_required_variables_survive_a_generator_that_renames_them():
    spec = next(p for p in PRODUCTS if p.kind == "checking")
    generated = [
        VariableSkeleton(
            name="Monthly Maintenance Fee",
            type="percent",
            unit="wrong",
            plausible_min=1.0,
            plausible_max=9.0,
            visibility="internal_only",
        ),
        VariableSkeleton(
            name="ATM Network Size",
            type="int_count",
            unit="atms",
            plausible_min=100.0,
            plausible_max=60000.0,
            visibility="customer_facing",
        ),
    ]
    merged = merge_required(spec, generated)
    names = [v.name for v in merged]
    assert names[: len(REQUIRED_VARIABLES["checking"])] == list(
        REQUIRED_VARIABLES["checking"]
    )
    # The required type wins over whatever the generator proposed, since the
    # solver and the families depend on it.
    monthly = next(v for v in merged if v.name == "monthly_fee")
    assert monthly.type == "currency"
    # The range is still taken from the generator's suggestion, matched through
    # the wordier name it proposed.
    assert (monthly.plausible_min, monthly.plausible_max) == (1.0, 9.0)
    assert "atm_network_size" in names
    # And the wordier name is not carried as a second copy of the same fact.
    assert "monthly_maintenance_fee" not in names


def test_generated_names_are_normalized():
    assert snake_case("Monthly  Maintenance-Fee ") == "monthly_maintenance_fee"
    assert snake_case("APY (%)") == "apy"


def test_enum_variables_are_assigned_from_their_candidates():
    from tau3.worldgen.solver import seed_baseline

    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_compounding_frequency",
                feature_id="feat_x",
                name="compounding_frequency",
                type="enum",
                choices=["daily", "monthly", "quarterly"],
                value=None,
            )
        ]
    )
    seed_baseline(schema, {}, 42)
    assert schema.variables[0].value in {"daily", "monthly", "quarterly"}


def test_an_enum_without_candidates_is_an_error_not_a_missing_value():
    from tau3.worldgen.solver import seed_baseline

    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_tier",
                feature_id="feat_x",
                name="tier",
                type="enum",
                choices=[],
                value=None,
            )
        ]
    )
    with pytest.raises(ValueError, match="no candidates"):
        seed_baseline(schema, {}, 42)


def test_generic_product_words_are_not_treated_as_official_names():
    # "Saver" and "Rewards" appear across many official topics: reusing them is
    # what makes a synthesized catalogue read like the real one. Only the
    # distinctive part of a name may not collide.
    generic = WorldSchema(
        features=[
            Feature(
                id="f1",
                category_id="c",
                entity_name="Amber Saver",
                entity_kind="account",
            ),
            Feature(
                id="f2",
                category_id="c",
                entity_name="Basalt Rewards",
                entity_kind="card",
            ),
        ]
    )
    assert check_official_name_collisions(generic).passed

    distinctive = WorldSchema(
        features=[
            Feature(
                id="f3",
                category_id="c",
                entity_name="Evergreen Saver",
                entity_kind="account",
            )
        ]
    )
    assert not check_official_name_collisions(distinctive).passed


def test_a_value_does_not_depend_on_what_else_exists():
    """Adding a variable must not move the values of the others.

    Drawing sequentially from one shared stream made every value depend on how
    many variables preceded it, so introducing a single derived variable shifted
    the whole catalogue and invalidated every rendered document -- which is the
    opposite of what incremental refresh needs.
    """
    from tau3.worldgen.solver import seed_baseline

    def world(extra: bool):
        variables = [
            Variable(
                id=f"var_{name}",
                feature_id="feat_a",
                name=name,
                type="currency",
                value=None,
            )
            for name in ("alpha", "beta", "gamma")
        ]
        if extra:
            variables.insert(
                0,
                Variable(
                    id="var_aaa_new",
                    feature_id="feat_a",
                    name="new",
                    type="currency",
                    value=None,
                ),
            )
        schema = WorldSchema(variables=variables)
        ranges = {v.id: (0.0, 1000.0) for v in variables}
        seed_baseline(schema, ranges, 42)
        return {v.id: v.value for v in schema.variables}

    without, with_extra = world(False), world(True)
    for variable_id, value in without.items():
        assert with_extra[variable_id] == value, variable_id
