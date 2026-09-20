"""Stage I: a value that moves must not cost a regeneration."""

import json

import pytest

from tau3.worldgen.models import DocPlan, TaskSpec, Variable, WorldPlan, WorldSchema
from tau3.worldgen.refresh import (
    assess,
    plan_hashes,
    refill,
    task_index,
    template_hash,
    variable_index,
)


@pytest.fixture
def world():
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_a_fee",
                feature_id="feat_a",
                name="monthly_fee",
                type="currency",
                value=0,
            ),
            Variable(
                id="var_a_limit",
                feature_id="feat_a",
                name="mobile_limit",
                type="currency",
                value=2500,
            ),
            Variable(
                id="var_b_fee",
                feature_id="feat_b",
                name="monthly_fee",
                type="currency",
                value=12,
            ),
        ]
    )
    plan = WorldPlan(
        documents=[
            DocPlan(
                doc_id="doc_a",
                title="Alpha",
                archetype="product_overview",
                feature_id="feat_a",
                variable_ids=["var_a_fee", "var_a_limit"],
            ),
            DocPlan(
                doc_id="doc_b",
                title="Beta",
                archetype="faq",
                feature_id="feat_b",
                variable_ids=["var_b_fee"],
            ),
            DocPlan(
                doc_id="doc_c",
                title="Gamma",
                archetype="howto_short",
                feature_id="feat_a",
                variable_ids=["var_a_fee"],
            ),
        ],
        tasks=[
            TaskSpec(
                task_id="task_001",
                archetype="selection",
                requires_variables=["var_a_fee", "var_b_fee"],
            ),
            TaskSpec(
                task_id="task_002",
                archetype="denial",
                requires_variables=["var_a_limit"],
            ),
        ],
    )
    return schema, plan


def test_indexes_point_from_knowledge_back_to_what_uses_it(world):
    _, plan = world
    assert variable_index(plan)["var_a_fee"] == ["doc_a", "doc_c"]
    assert task_index(plan)["var_a_fee"] == ["task_001"]


def test_a_changed_value_touches_only_its_documents_and_tasks(world):
    schema, plan = world
    impact = assess(plan, schema, ["var_a_fee"])
    assert impact.documents == ["doc_a", "doc_c"]
    assert impact.tasks == ["task_001"]
    # The point of the two-pass renderer: a value that moved needs no model.
    assert impact.needs_model == []


def test_a_template_hash_ignores_values_but_not_the_brief(world):
    _, plan = world
    document = plan.documents[0]
    before = template_hash(document, "model-x", seed=1)
    assert template_hash(document, "model-x", seed=1) == before
    # A different writer or a different seed is a different document.
    assert template_hash(document, "model-y", seed=1) != before
    assert template_hash(document, "model-x", seed=2) != before
    # A variable arriving changes what the document is written from.
    document.variable_ids.append("var_b_fee")
    assert template_hash(document, "model-x", seed=1) != before


def test_a_changed_brief_is_reported_as_needing_the_model(world):
    schema, plan = world
    styles = {d.doc_id: "model-x" for d in plan.documents}
    recorded = plan_hashes(plan, styles, seed=1)

    # Retitling doc_b is a rewrite, not a substitution.
    plan.documents[1].title = "Beta, revised"
    impact = assess(plan, schema, [], recorded, styles, seed=1)
    assert impact.needs_model == ["doc_b"]
    assert "doc_b" in impact.documents


def test_a_removed_variable_forces_a_rewrite_of_what_stated_it(world):
    schema, plan = world
    schema.variables = [v for v in schema.variables if v.id != "var_a_limit"]
    impact = assess(plan, schema, ["var_a_limit"])
    assert impact.needs_model == ["doc_a"]


def test_refill_restates_values_without_calling_a_model(tmp_path, world):
    schema, plan = world
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "doc_a.md").write_text(
        "## Alpha\n\n- Monthly fee: [[var_a_fee]]\n- Mobile limit: [[var_a_limit]]\n"
    )
    written = refill(tmp_path, schema, [plan.documents[0]], aliases={})
    assert written == ["doc_a"]
    content = json.loads((tmp_path / "documents" / "doc_a.json").read_text())["content"]
    assert "$0" in content and "$2,500" in content

    # Move the value and refill again: the prose is untouched, the figure is not.
    schema.variable("var_a_fee").value = 15
    refill(tmp_path, schema, [plan.documents[0]], aliases={})
    updated = json.loads((tmp_path / "documents" / "doc_a.json").read_text())["content"]
    assert "$15" in updated
    assert "Monthly fee" in updated


def test_refill_skips_a_document_that_was_never_rendered(tmp_path, world):
    schema, plan = world
    (tmp_path / "render").mkdir()
    assert refill(tmp_path, schema, plan.documents, aliases={}) == []


def test_refill_refuses_a_template_written_from_a_different_allocation(tmp_path, world):
    """Re-planning after rendering means re-rendering, not refilling.

    A template whose allocation has changed still holds placeholders for
    variables the document no longer owns. Filling them produces a document
    stating another document's figures, which L1 then reports as invented
    values -- from prose that was never wrong.
    """
    schema, plan = world
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "doc_b.md").write_text(
        "## Beta\n\n- Fee: [[var_b_fee]]\n- Limit: [[var_a_limit]]\n"
    )
    document = plan.documents[1]
    assert "var_a_limit" not in document.variable_ids

    with pytest.raises(ValueError, match="different allocation"):
        refill(tmp_path, schema, [document], aliases={})
