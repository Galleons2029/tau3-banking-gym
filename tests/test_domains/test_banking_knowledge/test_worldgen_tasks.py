"""Stage F: rules that the environment really enforces, and tasks built on them."""

import json
import random

import pytest

from tau3.worldgen.models import DocPlan, TaskSpec
from tau3.worldgen.rules import RULES, RuleSpec, environment_rules, verify_rules
from tau3.worldgen.tasks_gen import (
    build_tasks,
    customer,
    evidence_for,
    gold_statistics,
    routed,
)


def test_every_declared_rule_is_enforced_by_the_environment():
    # Not a style check: a rule the official tools do not apply would be
    # documented, believed, and then contradicted at execution time.
    assert verify_rules() == []
    rules = environment_rules()
    assert {r.id for r in rules} == {spec.id for spec in RULES}
    assert all(r.enforced_by_tool for r in rules)


def test_a_rule_the_environment_does_not_enforce_is_reported():
    invented = RuleSpec(
        id="rule_invented",
        statement="Personal checking requires a minimum age of ninety.",
        tool="open_bank_account_4821",
        # The environment applies no such rule, so the "violating" case succeeds.
        violating_accounts=[],
        satisfying_accounts=[],
        call={"account_type": "checking", "account_class": "Probe Account"},
    )
    broken = verify_rules([invented])
    assert broken and "was not refused" in broken[0]


def test_environment_rules_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "tau3.worldgen.rules.verify_rules", lambda specs=None: ["rule_x: drifted"]
    )
    with pytest.raises(ValueError, match="no longer enforces"):
        environment_rules()


def test_reference_actions_route_through_unlock_and_call():
    calls = routed("close_bank_account_9480", {"account_id": "a1"}, "001_3")
    assert [c["name"] for c in calls] == [
        "unlock_discoverable_agent_tool",
        "call_discoverable_agent_tool",
    ]
    assert calls[0]["arguments"]["agent_tool_name"] == "close_bank_account_9480"
    assert json.loads(calls[1]["arguments"]["arguments"]) == {"account_id": "a1"}


def test_evidence_is_a_cover_with_nothing_spare():
    documents = [
        DocPlan(doc_id="a", title="t", archetype="faq", variable_ids=["v1", "v2"]),
        DocPlan(doc_id="b", title="t", archetype="faq", variable_ids=["v3"]),
        DocPlan(doc_id="c", title="t", archetype="faq", variable_ids=["v1"]),
    ]
    chosen = evidence_for({"v1", "v2", "v3"}, documents)
    assert set(chosen) == {"a", "b"}
    # "c" adds nothing the others do not already provide.
    assert "c" not in chosen


def test_a_rotated_cover_is_still_a_cover_with_nothing_spare():
    # Two customers asking the same thing should not always be answered from the
    # same documents: a corpus that holds a fact in several places is more
    # spread out than a greedy pass starting at one end makes it look.
    documents = [
        DocPlan(doc_id="a", title="t", archetype="faq", variable_ids=["v1"]),
        DocPlan(doc_id="b", title="t", archetype="faq", variable_ids=["v1"]),
        DocPlan(doc_id="c", title="t", archetype="faq", variable_ids=["v2"]),
    ]
    covers = {
        tuple(sorted(evidence_for({"v1", "v2"}, documents, rotation)))
        for rotation in range(3)
    }
    assert len(covers) > 1
    for cover in covers:
        revealed = set()
        for doc_id in cover:
            revealed |= next(d for d in documents if d.doc_id == doc_id).reveals
        assert {"v1", "v2"} <= revealed
        assert len(cover) == 2


def test_a_selection_task_states_the_tier_and_proves_the_answer_is_in_it():
    from tau3.worldgen.tasks_gen import selection_requirements, selection_task

    schema, documents, families = _tiny_world()
    family = {**families[0], "segment": "Signature"}
    generated = selection_task(
        1,
        family,
        schema,
        documents,
        {t: t for t in _TOOLS},
        customer(random.Random(1), 1),
    )
    instructions = generated.task["user_scenario"]["instructions"]
    # The tier is the scope of the task: it is what lets two customers of the
    # same product kind be asking for different products.
    assert "Signature tier" in instructions
    assert generated.spec.scope == "Signature"
    # And the tier the answer is documented as being in is evidence, not an
    # assumption: without it the gold set proves a product qualifies but not
    # that it is one of the ones the customer asked about.
    tier = next(
        v
        for v in schema.variables
        if v.feature_id == family["answer"] and v.name == "service_tier"
    )
    assert tier.id in selection_requirements(family, schema)


def test_evidence_for_nothing_is_nothing():
    assert evidence_for(set(), [DocPlan(doc_id="a", title="t", archetype="faq")]) == []


def test_customers_are_distinct_and_verifiable():
    import random

    rng = random.Random(1)
    people = [customer(rng, i) for i in range(20)]
    assert len({p["user_id"] for p in people}) == 20
    for person in people:
        # Two of four identity fields are what a verification needs.
        assert person["email"] and person["date_of_birth"]
        assert person["phone_number"] and person["address"]


def test_the_database_holds_more_people_than_the_tasks_are_about():
    from tau3.worldgen.slice import FAMILIES

    schema, documents, families = _tiny_world()
    tasks, db = build_tasks(
        schema, documents, families, {t: t for t in _TOOLS}, 4, 42, noise=12
    )
    task_users = {
        action["arguments"].get("user_id")
        for task in tasks
        for action in task.task["evaluation_criteria"]["actions"]
        if action["name"] == "log_verification"
    }
    # Noise customers are the point: an agent that never identified anybody must
    # not be able to find the only matching record.
    assert len(db.users.data) > len(task_users)
    assert len(db.users.data) >= 12
    del FAMILIES


def test_gold_statistics_describe_the_task_set():
    stats = gold_statistics([])
    assert stats["mean_gold_documents"] == 0.0

    schema, documents, families = _tiny_world()
    tasks, _ = build_tasks(
        schema, documents, families, {t: t for t in _TOOLS}, 4, 42, noise=0
    )
    stats = gold_statistics(tasks)
    assert stats["mean_gold_documents"] > 0
    assert sum(stats["archetypes"].values()) == len(tasks)


_TOOLS = [
    "get_all_user_accounts_by_user_id_3847",
    "open_bank_account_4821",
    "close_bank_account_7392",
]


def _tiny_world():
    from tau3.worldgen.models import Feature, Variable, WorldSchema
    from tau3.worldgen.rules import environment_rules
    from tau3.worldgen.slice import business_feature_id

    business = business_feature_id()
    variables = []
    for feature_id in ("feat_a", "feat_b", business):
        for name, value in (
            ("account_class", f"{feature_id} Account"),
            ("service_tier", "Signature"),
            ("monthly_fee", 0),
        ):
            variables.append(
                Variable(
                    id=f"var_{feature_id}_{name}",
                    feature_id=feature_id,
                    name=name,
                    type=(
                        "string"
                        if name in {"account_class", "service_tier"}
                        else "currency"
                    ),
                    value=value,
                )
            )
    schema = WorldSchema(
        features=[
            Feature(
                id="feat_a", category_id="c", entity_name="Alpha", entity_kind="account"
            ),
            Feature(
                id="feat_b", category_id="c", entity_name="Beta", entity_kind="account"
            ),
            # An ordering task opens a business account, and a world documenting
            # no class string for one cannot carry that task at all.
            Feature(
                id=business,
                category_id="cat_business",
                entity_name="Quarry Business",
                entity_kind="account",
            ),
        ],
        variables=variables,
        rules=environment_rules(),
    )
    documents = [
        DocPlan(
            doc_id="doc_a",
            title="Alpha",
            archetype="product_overview",
            feature_id="feat_a",
            variable_ids=[v.id for v in variables if v.feature_id == "feat_a"],
        ),
        DocPlan(
            doc_id="doc_b",
            title="Beta",
            archetype="product_overview",
            feature_id="feat_b",
            variable_ids=[v.id for v in variables if v.feature_id == "feat_b"],
        ),
        DocPlan(
            doc_id="doc_tools",
            title="Internal",
            archetype="internal_protocol",
            tool_ids=_TOOLS,
            rule_ids=["rule_savings_tenure", "rule_business_no_closed"],
        ),
    ]
    families = [
        {
            "family_id": "fam",
            "kind": "selection",
            "candidates": ["feat_a", "feat_b"],
            "answer": "feat_a",
            "constraints": ["monthly_fee == 0"],
            "trap": "",
        }
    ]
    return schema, documents, families


def test_task_specs_carry_what_stage_g_checks():
    schema, documents, families = _tiny_world()
    tasks, _ = build_tasks(
        schema, documents, families, {t: t for t in _TOOLS}, 4, 42, noise=0
    )
    ordering = next((t for t in tasks if t.spec.archetype == "ordering"), None)
    if ordering is not None:
        # A value the evaluator compares verbatim must be declared, or Stage G
        # cannot check that the customer ever says it.
        assert ordering.spec.verbatim_arguments
        assert all(
            value in ordering.task["user_scenario"]["instructions"]
            for value in ordering.spec.verbatim_arguments
        )
    for task in tasks:
        assert isinstance(task.spec, TaskSpec)
        basis = task.task["evaluation_criteria"]["reward_basis"]
        if task.spec.archetype in {"denial", "grounding"}:
            assert basis == ["DB", "NL_ASSERTION"]
            assert task.task["evaluation_criteria"]["nl_assertions"]
        else:
            assert basis == ["DB"]


def test_a_shopping_customer_holds_out_until_the_account_is_open():
    """Two ways a selection task ends with nothing done, both observed live.

    A customer who accepts the first recommendation never states the rest of
    their requirements, and one who thanks the agent for naming a product never
    gets it opened. Either way the trajectory ends in agreement with the
    database untouched, which scores as a failure that says nothing about how
    hard the task was.
    """
    from tau3.worldgen.tasks_gen import selection_task

    schema, documents, families = _tiny_world()
    generated = selection_task(
        1,
        families[0],
        schema,
        documents,
        {t: t for t in _TOOLS},
        customer(random.Random(1), 1),
    )
    instructions = generated.task["user_scenario"]["instructions"]
    assert "until you have stated every one of them" in instructions
    assert "until they have told you the account is open" in instructions


def test_an_ordering_customer_names_the_account_they_want_opened():
    """The evaluator compares the opened class string verbatim.

    Which business account to open is not what an ordering task asks about, and
    a customer who leaves it open hands the agent a catalogue to guess from.
    """
    schema, documents, families = _tiny_world()
    tasks, _ = build_tasks(
        schema, documents, families, {t: t for t in _TOOLS}, 4, 42, noise=0
    )
    ordering = next(t for t in tasks if t.spec.archetype == "ordering")
    opened = [
        json.loads(action["arguments"]["arguments"])["account_class"]
        for action in ordering.task["evaluation_criteria"]["actions"]
        if action["name"] == "call_discoverable_agent_tool"
        and "account_class" in action["arguments"]["arguments"]
    ]
    assert opened
    assert opened[0] in ordering.task["user_scenario"]["instructions"]
