"""Structural linters: unique answers, real distractors, and no single-document shortcut."""

import pytest

from tau3.worldgen.linters import (
    check_near_misses,
    check_scope_is_documented,
    check_plan_references,
    check_single_document_leakage,
    check_unique_answers,
    lint_world,
    satisfies,
    solutions,
)
from tau3.worldgen.seed import build_plan, build_schema


@pytest.fixture
def schema():
    return build_schema()


@pytest.fixture
def plan():
    return build_plan()


def spec_named(plan, task_id):
    return next(spec for spec in plan.tasks if spec.task_id == task_id)


def document_named(plan, doc_id):
    return next(doc for doc in plan.documents if doc.doc_id == doc_id)


def test_seed_world_is_clean(schema, plan):
    report = lint_world(schema, plan.documents, plan.tasks, leak_divisor=3)
    assert report.passed, [f.message for f in report.findings]


def test_selection_task_has_exactly_one_answer(schema, plan):
    spec = spec_named(plan, "task_001")
    assert solutions(schema, spec) == ["feat_copper"]


def test_a_second_qualifying_product_is_rejected(schema, plan):
    # Give Zinc the early deposit it lacks and two products now qualify.
    schema.variable("var_zinc_early_direct_deposit_days").value = 3
    report = check_unique_answers(schema, plan.tasks)
    assert not report.passed
    assert "select 2 products" in report.findings[0].message


def test_expected_answer_must_match_the_constraints(schema, plan):
    spec = spec_named(plan, "task_001")
    spec.unique_answer = "feat_slate"
    report = check_unique_answers(schema, [spec])
    assert not report.passed
    assert "expects feat_slate" in report.findings[0].message


def test_distractor_failing_two_constraints_is_rejected(schema, plan):
    # Slate already fails on fee; failing on deposit limit too would let the
    # agent discard it from one document instead of comparing.
    schema.variable("var_slate_mobile_deposit_limit").value = 500
    report = check_near_misses(schema, plan.tasks)
    assert not report.passed
    assert "fails 2 constraints" in report.findings[0].message


def test_missing_variable_fails_the_constraint_rather_than_passing(schema, plan):
    schema.variables = [v for v in schema.variables if v.name != "monthly_fee"]
    assert not satisfies(schema, "feat_copper", "monthly_fee == 0")


def test_single_document_cannot_reveal_a_whole_task(schema, plan):
    # Fold the tenure number back into the policy document: two of the three
    # things task 2 needs would then come from one place.
    policy = document_named(plan, "doc_protocol_opening_002")
    policy.variable_ids = ["var_savings_min_tenure_days"]
    report = check_single_document_leakage(plan.tasks, plan.documents, leak_divisor=3)
    assert not report.passed
    assert any("task_002" in f.subject for f in report.findings)


def test_leakage_check_skips_tasks_with_no_required_knowledge(plan):
    grounding = spec_named(plan, "task_005")
    assert not grounding.requires
    report = check_single_document_leakage([grounding], plan.documents)
    assert report.passed


def test_dangling_cross_reference_is_caught(schema, plan):
    document_named(plan, "doc_protocol_opening_002").crossrefs = ["doc_does_not_exist"]
    report = check_plan_references(schema, plan.documents)
    assert not report.passed
    assert report.findings[0].check == "L6"


def test_documents_cannot_reveal_unknown_knowledge(schema, plan):
    document_named(plan, "doc_checking_copper_001").variable_ids.append("var_invented")
    report = check_plan_references(schema, plan.documents)
    assert any("Unknown variable" in f.message for f in report.findings)


def test_a_document_that_never_names_its_subject_is_rejected(schema, plan):
    from tau3.worldgen.linters import check_retrievable_subjects

    # Retrieval indexes content, so this document is unreachable even though its
    # title says "Copper" -- the failure that a real run surfaced.
    rendered = {
        d.doc_id: "## Fees and Limits\n\n| Item | Value |\n|---|---|\n| Monthly fee | $0 |\n"
        for d in plan.documents
    }
    report = check_retrievable_subjects(schema, plan.documents, rendered)
    assert not report.passed
    assert all(f.check == "L11" for f in report.findings)
    assert any("doc_checking_copper_001" == f.subject for f in report.findings)


def test_rendered_seed_documents_name_their_subjects(schema, plan):
    from tau3.worldgen.linters import check_retrievable_subjects
    from tau3.worldgen.seed import render_documents

    rendered = {
        d["id"]: d["content"] for d in render_documents(schema, plan, lambda name: name)
    }
    assert check_retrievable_subjects(schema, plan.documents, rendered).passed


def test_a_planned_but_unrendered_document_is_reported(schema, plan):
    from tau3.worldgen.linters import check_retrievable_subjects

    report = check_retrievable_subjects(schema, plan.documents, {})
    assert any("not rendered" in f.message for f in report.findings)


def _tier_world():
    """One product kind layered into two tiers, one task scoped to one of them."""
    from tau3.worldgen.models import Feature, TaskSpec, Variable, WorldSchema

    variables, features = [], []
    for feature_id, tier, fee in (
        ("feat_a", "Everyday", 0),
        ("feat_b", "Everyday", 5),
        ("feat_c", "Signature", 0),
    ):
        features.append(
            Feature(
                id=feature_id,
                category_id="c",
                entity_name=feature_id,
                entity_kind="account",
            )
        )
        variables.append(
            Variable(
                id=f"var_{feature_id}_service_tier",
                feature_id=feature_id,
                name="service_tier",
                type="string",
                value=tier,
            )
        )
        variables.append(
            Variable(
                id=f"var_{feature_id}_monthly_fee",
                feature_id=feature_id,
                name="monthly_fee",
                type="currency",
                value=fee,
            )
        )
    schema = WorldSchema(features=features, variables=variables)
    spec = TaskSpec(
        task_id="task_001",
        archetype="selection",
        candidate_features=["feat_a", "feat_b"],
        constraints=["monthly_fee == 0"],
        unique_answer="feat_a",
        scope="Everyday",
    )
    return schema, spec


def test_a_task_may_not_ask_for_a_tier_its_answer_is_not_in():
    # The tier is the only thing separating the answers of two families of the
    # same product kind, so a tier the documents disagree with points the
    # customer at a product the evidence places somewhere else.
    schema, spec = _tier_world()
    schema.variable("var_feat_a_service_tier").value = "Landmark"
    report = check_scope_is_documented(schema, [spec])
    assert report.errors
    assert "documented as 'Landmark'" in report.errors[0].message


def test_a_correctly_scoped_task_passes_but_reports_what_the_tier_is_doing():
    schema, spec = _tier_world()
    report = check_scope_is_documented(schema, [spec])
    assert report.passed
    # feat_c meets the stated requirement and is excluded by its tier alone.
    # That is the point of layering, and it is reported so the difficulty it
    # adds to the task is visible rather than implicit.
    assert any("outside the Everyday tier" in f.message for f in report.advice)
