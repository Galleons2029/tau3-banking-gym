"""Stage C and D: allocation shape, the placeholder gate, and value substitution."""

import pytest

from tau3.worldgen.allocate import TARGET_DEGREE, allocate, archetype_quota
from tau3.worldgen.models import ARCHETYPES, DocPlan, Variable, WorldConfig, WorldSchema
from tau3.worldgen.render import (
    HallucinatedValue,
    fill,
    format_value,
    missing_placeholders,
    style_for,
    unknown_placeholders,
    unresolved_digits,
)
from tau3.worldgen.slice import FAMILIES, PRODUCTS
from tau3.worldgen.solver import solve
from tau3.worldgen.text_linters import (
    check_cross_document_agreement,
    check_numeric_provenance,
    check_tool_coverage,
    minhash,
    similarity,
)


@pytest.fixture(scope="module")
def world():
    from test_banking_knowledge.test_worldgen_solver import build_unassigned

    schema, ranges = build_unassigned()
    solve(schema, FAMILIES, ranges, 42, PRODUCTS)
    tools = [
        {
            "official_name": "open_bank_account_4821",
            "alias": "open_bank_account_5888",
            "owner": "agent",
            "arguments": ["user_id", "account_type", "account_class"],
            "document_archetype": "internal_protocol",
        },
        {
            "official_name": "close_bank_account_7392",
            "alias": "close_bank_account_9480",
            "owner": "agent",
            "arguments": ["account_id", "reason"],
            "document_archetype": "tool_doc",
        },
    ]
    documents, report = allocate(schema, tools, 40, "salt")
    return schema, documents, report, tools


def test_archetype_quota_is_exact_and_proportional():
    quota = archetype_quota(80)
    assert sum(quota.values()) == 80
    assert quota["howto_short"] > quota["product_overview"]
    assert set(quota) == set(ARCHETYPES)


def test_every_variable_reaches_at_least_one_document(world):
    _, _, report, _ = world
    assert report.uncovered == []


def test_redundancy_sits_near_the_target_degree(world):
    _, _, report, _ = world
    # Above one so a single missed retrieval is survivable, well below two so it
    # is not free.
    assert 1.0 < report.mean_degree < TARGET_DEGREE + 0.35


def test_every_tool_is_named_by_some_document(world):
    _, _, report, _ = world
    assert report.tools_unbound == []


def test_no_document_reveals_a_whole_family(world):
    schema, documents, _, tools = world
    import math

    protected = {
        "fam": {
            v.id
            for v in schema.variables
            if v.feature_id in {"feat_copper", "feat_zinc", "feat_slate"}
            and v.name in {"monthly_fee", "early_direct_deposit_days"}
        }
    }
    documents, _ = allocate(schema, tools, 40, "salt", protected, leak_divisor=3)
    budget = math.ceil(len(protected["fam"]) / 3)
    for document in documents:
        assert len(set(document.variable_ids) & protected["fam"]) <= budget


def test_style_rotation_keeps_internal_documents_internal():
    models = ["model-a", "model-b"]
    internal = DocPlan(
        doc_id="doc_protocol_001", title="Internal", archetype="internal_protocol"
    )
    customer = DocPlan(doc_id="doc_p_001", title="Glance", archetype="product_overview")
    assert "internal" in style_for(internal, models, "salt").register
    assert "customer" in style_for(customer, models, "salt").register
    # Both models must actually get used across a corpus, or the rotation is
    # decorative and the corpus clusters by author.
    chosen = {
        style_for(
            DocPlan(doc_id=f"doc_{i:03d}", title="t", archetype="faq"), models, "salt"
        ).model
        for i in range(20)
    }
    assert chosen == set(models)


def test_a_draft_that_states_a_figure_is_rejected():
    assert unresolved_digits("The fee is $12 per month.") == ["1", "2"]
    assert unresolved_digits("The fee is [[var_x_fee]] per month.") == []
    # Structure is not a fact: list numbering, years, and identifiers with digits.
    assert unresolved_digits("1. Do this\n2. Do that") == []
    assert unresolved_digits("Effective 2025 onwards.") == []
    assert unresolved_digits("Use open_bank_account_4821 with last_4_digits.") == []


def test_missing_and_invented_placeholders_are_both_caught():
    document = DocPlan(
        doc_id="d",
        title="t",
        archetype="faq",
        variable_ids=["var_a", "var_b"],
        tool_ids=["open_bank_account_4821"],
    )
    body = "Fee [[var_a]] and [[var_zzz]] via [[tool:open_bank_account_4821]]."
    assert missing_placeholders(body, document) == ["var_b"]
    assert unknown_placeholders(body, document) == ["var_zzz"]


def test_substitution_uses_this_world_s_tool_names():
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_fee",
                feature_id="feat_x",
                name="monthly_fee",
                type="currency",
                value=0,
            )
        ]
    )
    document = DocPlan(
        doc_id="d",
        title="t",
        archetype="faq",
        variable_ids=["var_x_fee"],
        tool_ids=["open_bank_account_4821"],
    )
    filled = fill(
        "Fee [[var_x_fee]]; use [[tool:open_bank_account_4821]].",
        document,
        schema,
        {"open_bank_account_4821": "open_bank_account_5888"},
    )
    assert filled == "Fee $0; use open_bank_account_5888."
    assert "4821" not in filled


def test_values_render_the_way_a_knowledge_base_states_them():
    def variable(kind, value):
        return Variable(id="v", feature_id="f", name="n", type=kind, value=value)

    assert format_value(variable("currency", 1250)) == "$1,250"
    assert format_value(variable("percent", 2.50)) == "2.5%"
    assert format_value(variable("int_days", 3)) == "3"
    assert format_value(variable("bool", True)) == "Yes"


def test_numeric_provenance_catches_a_figure_from_nowhere():
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_fee",
                feature_id="feat_x",
                name="monthly_fee",
                type="currency",
                value=0,
            )
        ]
    )
    documents = [
        DocPlan(doc_id="d", title="t", archetype="faq", variable_ids=["var_x_fee"])
    ]
    clean = check_numeric_provenance(schema, documents, {"d": "The fee is $0."})
    assert clean.passed

    invented = check_numeric_provenance(
        schema, documents, {"d": "The fee is $0 and the limit is $2,500."}
    )
    assert not invented.passed
    assert "2,500" in invented.findings[0].message


def test_an_enum_value_is_not_read_back_as_loose_figures():
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_hours",
                feature_id="feat_x",
                name="support_hours",
                type="enum",
                choices=["24_7"],
                value="24_7",
            )
        ]
    )
    documents = [
        DocPlan(doc_id="d", title="t", archetype="faq", variable_ids=["var_x_hours"])
    ]
    assert check_numeric_provenance(schema, documents, {"d": "Support: 24_7."}).passed


def test_a_variable_stated_twice_must_agree():
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_x_fee",
                feature_id="feat_x",
                name="monthly_fee",
                type="currency",
                value=12,
            )
        ]
    )
    documents = [
        DocPlan(doc_id="a", title="t", archetype="faq", variable_ids=["var_x_fee"]),
        DocPlan(doc_id="b", title="t", archetype="faq", variable_ids=["var_x_fee"]),
    ]
    agreeing = {"a": "Fee $12.", "b": "The fee is $12 a month."}
    assert check_cross_document_agreement(schema, documents, agreeing).passed

    contradicting = {"a": "Fee $12.", "b": "The fee is $15 a month."}
    report = check_cross_document_agreement(schema, documents, contradicting)
    assert not report.passed


def test_an_unnamed_tool_can_never_be_unlocked():
    tools = [
        {"official_name": "close_bank_account_7392", "alias": "close_bank_account_9480"}
    ]
    missing = check_tool_coverage(tools, [], {"d": "Nothing here."})
    assert not missing.passed
    assert "never be unlocked" in missing.findings[0].message

    present = check_tool_coverage(
        tools, [], {"d": "Use close_bank_account_9480 to close."}
    )
    assert present.passed


def test_an_official_tool_name_in_the_text_is_reported_as_a_leak():
    tools = [
        {"official_name": "close_bank_account_7392", "alias": "close_bank_account_9480"}
    ]
    report = check_tool_coverage(
        tools,
        [],
        {"d": "Use close_bank_account_9480, formerly close_bank_account_7392."},
    )
    assert not report.passed
    assert "official name leaked" in report.findings[0].message


def test_near_duplicate_detection_separates_copies_from_neighbours():
    text = "The Copper Account charges no monthly maintenance fee for customers."
    assert similarity(minhash(text), minhash(text)) == 1.0
    different = "Business closures require a manager approval code before processing."
    assert similarity(minhash(text), minhash(different)) < 0.5


def test_offline_rendering_is_never_publishable():
    config = WorldConfig(text_mode="template")
    assert config.text_mode == "template"
    with pytest.raises(HallucinatedValue):
        raise HallucinatedValue("placeholder gate")


def test_a_draft_reports_what_it_consumed():
    """Throughput is endpoint-bound, so a run has to record its own usage."""
    from types import SimpleNamespace

    from tau3.worldgen.models import DocPlan, Variable, WorldConfig, WorldSchema
    from tau3.worldgen.render import Style, draft

    document = DocPlan(
        doc_id="doc_a",
        title="Alpha",
        archetype="faq",
        feature_id="feat_a",
        variable_ids=["var_a_fee"],
    )
    schema = WorldSchema(
        variables=[
            Variable(
                id="var_a_fee",
                feature_id="feat_a",
                name="monthly_fee",
                type="currency",
                value=0,
            )
        ]
    )

    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content="## Alpha\n\n- Monthly fee: [[var_a_fee]]\n",
            usage={"prompt_tokens": 120, "completion_tokens": 900},
        )

    import tau3.utils.llm_utils as llm_utils

    original = llm_utils.generate
    llm_utils.generate = fake_generate
    try:
        _body, info = draft(
            document,
            schema,
            tools={},
            titles={},
            style=Style(model="m", persona="p", register="internal procedural"),
            config=WorldConfig(),
        )
    finally:
        llm_utils.generate = original

    assert info["prompt_tokens"] == 120
    assert info["completion_tokens"] == 900
    assert info["attempts"] == 1


def test_offline_drafts_report_no_usage():
    from tau3.worldgen.models import DocPlan, WorldConfig, WorldSchema
    from tau3.worldgen.render import Style, draft

    document = DocPlan(doc_id="doc_a", title="Alpha", archetype="faq")
    _body, info = draft(
        document,
        WorldSchema(),
        tools={},
        titles={},
        style=Style(model="m", persona="p", register="internal procedural"),
        config=WorldConfig(text_mode="template"),
    )
    assert info == {
        "model": "offline",
        "attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
