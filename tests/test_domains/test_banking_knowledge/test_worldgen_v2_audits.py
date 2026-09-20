"""Adversarial isolation, calibration, budget and admission regressions; offline."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
from tau3.worldgen.v2.audit import AuditIncomplete, AuditSession
from tau3.worldgen.v2.blind import (
    Witness,
    calculate,
    check_frozen_witness,
    check_public_witness,
    public_problem,
)
from tau3.worldgen.v2.catalog import scaffold
from tau3.worldgen.v2.pipeline import build, publish, write_json
from tau3.worldgen.v2.settings import load_settings
from tau3.worldgen.v2.specs import WorldSpec


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "world"
    assert (
        build(root, WorldSpec(categories=[scaffold("checking", "checking")]))["status"]
        == "PASS"
    )
    publish(root)
    return root


def test_private_answer_changes_cannot_change_solver_input(bundle, monkeypatch):
    from tau3.worldgen.v2 import environment

    original = public_problem(bundle, "task_checking_service")
    env = environment.get_environment(bundle, retrieval_variant="no_knowledge")
    monkeypatch.setattr(environment, "get_environment", lambda *a, **kw: env)
    path = bundle / "tasks/task_checking_service.json"
    task = json.loads(path.read_text())
    task["evaluation_criteria"] = {"PRIVATE_CANARY": "ANSWER_SHOULD_NEVER_LEAK"}
    task["required_documents"] = ["SECRET_GOLD"]
    write_json(path, task)
    write_json(bundle / "private/canary.json", {"reference": "SECRET_GOAL"})
    projected = public_problem(bundle, "task_checking_service")
    assert projected == original
    encoded = json.dumps(projected)
    assert all(
        s not in encoded
        for s in ["PRIVATE_CANARY", "SECRET_GOLD", "SECRET_GOAL", "evaluation_criteria"]
    )


@pytest.mark.parametrize(
    "expression,result",
    [
        ("0.1 + 0.2", Decimal("0.3")),
        ("0.10000000000000000001 + 0", Decimal("0.10000000000000000001")),
        ("days('2025-11-14', '2025-11-13')", Decimal(1)),
        ("50 - 20", Decimal(30)),
        ("365>=14", True),
        ("20 > 0 and 20 <= 50", True),
        ("2025-11-01 <= 2025-11-14 <= 2025-12-31", True),
    ],
)
def test_independent_exact_calculation(expression, result):
    assert calculate(expression) == result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "(1).__class__",
        "True + 1",
        "10**999999",
        "'NaN'",
        "[0]*99999",
    ],
)
def test_independent_calculation_rejects_unbounded_or_executable_input(expression):
    with pytest.raises((ValueError, TypeError)):
        calculate(expression)


def service_witness(bundle):
    from tau3.worldgen.v2.environment import get_environment

    env = get_environment(bundle, retrieval_variant="no_knowledge")
    scenario = env.tools.spec.categories[0].scenarios[1]
    step = scenario.steps[0]
    alias = next(
        a for a, op in env.tools._runtime().aliases.items() if op == step.operation
    )
    row = dict(env.tools.db.tables["checking_records"]["checking_record_1"])
    row["balance"] = "30"
    return Witness.model_validate(
        {
            "answer": "Withdrew 20 USD; 30 USD remains.",
            "evidence": [{"source": "request", "quote": "20 USD"}],
            "steps": [
                {"capability": alias, "actor": "assistant", "arguments": step.arguments}
            ],
            "changed_rows": [
                {
                    "table": "checking_records",
                    "row_id": "checking_record_1",
                    "fields": row,
                }
            ],
            "calculations": [{"expression": "50-20", "result": "30"}],
        }
    )


def test_frozen_prediction_and_wrong_private_goal_disagree(bundle, monkeypatch):
    from tau3.worldgen.v2 import environment

    witness = service_witness(bundle)
    assert (
        check_frozen_witness(bundle, "task_checking_service", witness)["status"]
        == "PASS"
    )
    spec = environment.load_spec(bundle)
    spec.categories[0].scenarios[1].goals[0].fields["balance"] = "35"
    monkeypatch.setattr(environment, "load_spec", lambda *a, **kw: spec)
    with pytest.raises(ValueError, match="private expected"):
        check_frozen_witness(bundle, "task_checking_service", witness)


def test_fabricated_quote_and_wrong_math_fail_before_private_replay(bundle):
    problem = public_problem(bundle, "task_checking_service")
    witness = service_witness(bundle)
    check_public_witness(problem, witness)
    witness.evidence[0].quote = "The bank authorizes theft"
    with pytest.raises(ValueError, match="quotation"):
        check_public_witness(problem, witness)
    witness.evidence[0].quote = "20 USD"
    witness.calculations[0].result = "31"
    with pytest.raises(ValueError, match="calculation"):
        check_public_witness(problem, witness)


def test_audit_reserves_failures_and_never_reuses_different_inputs(
    bundle, tmp_path, monkeypatch
):
    import tau3.utils.llm_utils as llm

    session = AuditSession(bundle, tmp_path / "audit", load_settings(), 1, "test")
    monkeypatch.setattr(
        llm, "generate", lambda **kw: SimpleNamespace(content='{"ok":true}')
    )
    assert session.ask("model", "system", {"a": 1}) == {"ok": True}
    assert session.ask("model", "system", {"a": 1}) == {"ok": True}
    with pytest.raises(AuditIncomplete, match="budget"):
        session.ask("model", "system", {"a": 2})
    assert len(list((tmp_path / "audit/calls").glob("*.json"))) == 1


@pytest.mark.parametrize("budget", [1, 2])
@pytest.mark.parametrize("session_type", [AuditSession, ConcurrentAuditSession])
def test_transport_retry_records_every_attempt_and_consumes_budget(
    bundle, tmp_path, monkeypatch, budget, session_type
):
    from litellm import Timeout

    import tau3.utils.llm_utils as llm

    session = session_type(bundle, tmp_path / "audit", load_settings(), budget, "test")
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        assert kwargs["num_retries"] == 0
        if len(calls) == 1:
            raise Timeout(message="test timeout", model="model", llm_provider="openai")
        return SimpleNamespace(content='{"ok":false}')

    monkeypatch.setattr(llm, "generate", respond)
    if budget == 1:
        with pytest.raises(AuditIncomplete, match="budget"):
            session.ask("model", "system", {"a": 1})
    else:
        assert session.ask("model", "system", {"a": 1}) == {"ok": False}
        assert session.ask("model", "system", {"a": 1}) == {"ok": False}
    assert len(calls) == budget
    record = json.loads(next((tmp_path / "audit/calls").glob("*.json")).read_text())
    assert len(record["attempts"]) == budget
    assert record["attempts"][0]["error"] == "Timeout"
    assert record["attempts"][0]["transport_failure"] is True
    with pytest.raises(AuditIncomplete):
        session.ask("model", "system", {"a": 2})
    assert len(calls) == budget


@pytest.mark.parametrize("session_type", [AuditSession, ConcurrentAuditSession])
def test_nontransport_failure_is_not_resampled(
    bundle, tmp_path, monkeypatch, session_type
):
    import tau3.utils.llm_utils as llm

    session = session_type(bundle, tmp_path / "audit", load_settings(), 5, "test")
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        raise ValueError("invalid content")

    monkeypatch.setattr(llm, "generate", respond)
    for _ in range(2):
        with pytest.raises(AuditIncomplete):
            session.ask("model", "system", {})
    assert len(calls) == 1


def test_real_calibration_fixture_labels_and_alternative_state(bundle):
    from tau3.data_model.tasks import Task
    from tau3.worldgen.v2.calibration import controls, trajectory
    from tau3.worldgen.v2.environment import get_environment

    cases = controls(bundle)
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["kind"] for c in cases} == {
        "selection",
        "service",
        "ordering",
        "denial",
        "grounding",
    }
    case = next(c for c in cases if c["id"] == "service_alternative_two_payments")
    task = Task.model_validate_json(
        (bundle / "tasks" / f"{case['task_id']}.json").read_text()
    )
    messages = trajectory(bundle, task, case["actions"], case["answer"])
    env = get_environment(bundle, retrieval_variant="no_knowledge")
    env.set_state(
        initialization_data=None, initialization_actions=None, message_history=messages
    )
    assert env.tools.check_scenario_outcome("checking_service")


def test_expansion_requires_actual_readiness(bundle, tmp_path):
    from tau3.worldgen.v2.expansion import expand

    with pytest.raises(ValueError, match="readiness"):
        expand(bundle, tmp_path / "expanded", [])


def test_missing_evidence_never_grants_readiness(bundle, tmp_path):
    from tau3.worldgen.v2.readiness import assess

    report = assess(
        bundle,
        tmp_path / "blind",
        tmp_path / "calib",
        tmp_path / "online",
        tmp_path / "readiness.json",
    )
    assert report["status"] == "INCONCLUSIVE"


def test_independent_training_admission_cannot_relax_legacy_worlds(bundle, tmp_path):
    from tau3.worldgen.v2.readiness import assess

    with pytest.raises(ValueError, match="requires a targeted world"):
        assess(
            bundle,
            tmp_path / "blind",
            tmp_path / "calib",
            tmp_path / "online",
            tmp_path / "readiness.json",
            admission_mode="targeted_training",
        )


def test_online_budget_checked_before_model_calls(bundle, tmp_path):
    from tau3.worldgen.v2.rollouts import run_matrix

    with pytest.raises(ValueError, match="budget"):
        run_matrix(bundle, tmp_path / "matrix", max_rollouts=1)


def test_scoring_mutations_are_actual_wrong_operations(bundle):
    from tau3.data_model.tasks import Task
    from tau3.worldgen.v2.calibration import controls, trajectory
    from tau3.worldgen.v2.environment import get_environment

    tested = set()
    for case in controls(bundle):
        if not any(
            token in case["id"]
            for token in (
                "wrong_",
                "no_operation",
                "reversed_operations",
                "missing_permission",
            )
        ):
            continue
        task = Task.model_validate_json(
            (bundle / "tasks" / f"{case['task_id']}.json").read_text()
        )
        messages = trajectory(bundle, task, case["actions"], case["answer"])
        env = get_environment(bundle, retrieval_variant="no_knowledge")
        env.set_state(None, None, messages)
        assert not env.tools.check_scenario_outcome(
            case["task_id"].removeprefix("task_")
        ), case["id"]
        tested.add(case["id"])
    assert {
        "selection_wrong_product_id",
        "service_wrong_amount",
        "ordering_missing_permission",
    } <= tested


def test_public_repair_is_bounded_and_uses_only_public_feedback(bundle):
    from tau3.worldgen.v2.blind import solve_public

    problem = public_problem(bundle, "task_checking_service")
    witness = service_witness(bundle).model_dump()
    invalid = json.loads(json.dumps(witness))
    invalid["evidence"][0]["quote"] = "fabricated quote"
    requests = []

    def ask(model, prompt, payload):
        requests.append(payload)
        if model == "reviewer":
            return {
                k: True
                for k in (
                    "intent_satisfied",
                    "evidence_sufficient",
                    "state_correct",
                    "permissions_correct",
                    "calculations_complete",
                )
            } | {"issues": []}
        return invalid if len(requests) == 1 else witness

    result, _, attempts = solve_public(problem, "solver", "reviewer", ask)
    assert result == witness and attempts == 2 and len(requests) == 3
    assert requests[1]["problem"] == problem
    assert "quotation" in requests[1]["public_feedback"]
    assert "goals" not in json.dumps(requests)
    count = []

    def broken(*args):
        count.append(1)
        return invalid

    with pytest.raises(ValueError, match="quotation"):
        solve_public(problem, "solver", "reviewer", broken)
    assert len(count) == 4


def test_public_protocol_repairs_array_source_and_unbound_calculation(bundle):
    from tau3.worldgen.v2.blind import solve_public

    problem = public_problem(bundle, "task_checking_service")
    valid = service_witness(bundle).model_dump()
    bad_source = json.loads(json.dumps(valid))
    bad_source["evidence"] = [{"source": "checking_records"}]
    bad_source["calculations"] = [{"expression": "balance-20", "result": "30"}]
    bad_calculation = json.loads(json.dumps(valid))
    bad_calculation["calculations"] = bad_source["calculations"]
    requests = []

    def ask(model, system, payload):
        requests.append(payload)
        if model == "reviewer":
            return {
                k: True
                for k in (
                    "intent_satisfied",
                    "evidence_sufficient",
                    "state_correct",
                    "permissions_correct",
                    "calculations_complete",
                )
            } | {"issues": []}
        index = len(requests)
        if index == 1:
            raise ValueError("Expected one JSON object")
        return bad_source if index == 2 else bad_calculation if index == 3 else valid

    result, _, attempts = solve_public(problem, "solver", "reviewer", ask)
    assert result == valid and attempts == 4 and len(requests) == 5
    assert "checking_records" in requests[2]["public_feedback"]
    assert "balance-20" in requests[3]["public_feedback"]
    assert "variables are not bound" in requests[3]["public_feedback"]
    assert all(p["problem"] == problem for p in requests[1:])


@pytest.mark.parametrize("content", ['```json\n{"ok": true}\n```', '{"ok":true}'])
def test_complete_single_json_fence_is_supported(content):
    from tau3.worldgen.v2.json_output import parse_object

    assert parse_object(content) == {"ok": True}


@pytest.mark.parametrize(
    "content",
    [
        '{"ok":true} trailing',
        '```json\n{"ok":',
        '{"ok":true}{"ok":false}',
        '{"ok":false,"ok":true}',
        '{"nested":{"value":0,"value":1}}',
        '{"value":NaN}',
        "[]",
    ],
)
def test_json_noise_truncation_and_multiple_verdicts_are_rejected(content):
    from tau3.worldgen.v2.json_output import parse_object

    with pytest.raises(ValueError):
        parse_object(content)


def test_user_can_discover_only_granted_contracts_and_execute(bundle):
    from tau3.worldgen.v2.environment import get_environment

    env = get_environment(bundle, retrieval_variant="no_knowledge")
    runtime = env.tools._runtime()
    scenario = env.tools.spec.categories[0].scenarios[2]
    assert json.loads(env.user_tools.list_granted_tools()) == []
    for step in scenario.steps:
        alias = next(a for a, op in runtime.aliases.items() if op == step.operation)
        if runtime.operations[step.operation][1].actor == "assistant":
            env.tools.unlock_discoverable_agent_tool(alias)
            env.tools.call_discoverable_agent_tool(alias, json.dumps(step.arguments))
        else:
            with pytest.raises(ValueError, match="not granted"):
                env.user_tools.call_discoverable_user_tool(
                    alias, json.dumps(step.arguments)
                )
            granted = json.loads(env.tools.give_discoverable_user_tool(alias))
            contracts = json.loads(env.user_tools.list_granted_tools())
            assert contracts == [granted]
            env.user_tools.call_discoverable_user_tool(
                contracts[0]["name"], json.dumps(step.arguments)
            )
    assert env.tools.check_scenario_outcome(scenario.id)
    other = get_environment(bundle, retrieval_variant="no_knowledge")
    assert json.loads(other.user_tools.list_granted_tools()) == []


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_valid_json_never_becomes_a_successful_audit(
    bundle, tmp_path, monkeypatch, finish_reason
):
    import tau3.utils.llm_utils as llm

    session = AuditSession(bundle, tmp_path / "audit", load_settings(), 2, "test")
    monkeypatch.setattr(
        llm,
        "generate",
        lambda **kw: SimpleNamespace(
            content='{"ok":true}',
            raw_data={"choices": [{"finish_reason": finish_reason}]},
        ),
    )
    for _ in range(2):
        with pytest.raises(AuditIncomplete):
            session.ask("model", "system", {"a": 1})
    records = list((tmp_path / "audit/calls").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["response"] == '{"ok":true}'
    assert record["finish_reason"] == finish_reason


def test_public_checker_rejects_invented_wrapper_tool(bundle):
    problem = public_problem(bundle, "task_checking_service")
    witness = service_witness(bundle)
    witness.steps[0].capability = "invented_unlock_wrapper"
    with pytest.raises(ValueError, match="public"):
        check_public_witness(problem, witness)


def test_semantic_judge_rejects_valid_json_with_truncated_finish(monkeypatch):
    from tau3.worldgen.v2 import semantic

    monkeypatch.setattr(
        semantic,
        "generate",
        lambda **kw: SimpleNamespace(
            content='{"results":[{"index":0,"met":true,"reason":"Yes"}]}',
            raw_data={"choices": [{"finish_reason": "length"}]},
        ),
    )
    checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions([], ["Correct"])
    assert len(checks) == 1 and not checks[0].met
    assert checks[0].justification.startswith("INCONCLUSIVE:")


def test_calibration_covers_each_business_structure(tmp_path):
    from tau3.worldgen.v2.calibration import controls
    from tau3.worldgen.v2.diversity import structural_fingerprint
    from tau3.worldgen.v2.environment import load_spec

    root = tmp_path / "multiple"
    spec = WorldSpec(
        categories=[scaffold("checking", "checking"), scaffold("card", "credit_card")]
    )
    assert build(root, spec)["status"] == "PASS"
    publish(root)
    spec = load_spec(root)
    expected = {
        structural_fingerprint(category, scenario)
        for category in spec.categories
        for scenario in category.scenarios
    }
    cases = controls(root)
    assert {c["structure"] for c in cases} == expected
    for structure in expected:
        assert {c["expected"] for c in cases if c["structure"] == structure} == {0, 1}


@pytest.mark.parametrize("transient", [False, True])
def test_independent_replay_rejects_extra_same_customer_effects(
    bundle, monkeypatch, transient
):
    from tau3.worldgen.v2 import environment

    witness = service_witness(bundle)
    env = environment.get_environment(bundle, retrieval_variant="no_knowledge")
    original = env.tools.call_discoverable_agent_tool

    def altered(capability, arguments):
        result = original(capability, arguments)
        row = dict(env.tools.db.tables["checking_records"]["checking_record_1"])
        if transient:
            changed = row | {"opened_on": "2020-01-01"}
            env.tools.db.events.append(
                {
                    "changes": [
                        {
                            "table": "checking_records",
                            "row_id": "checking_record_1",
                            "before": row,
                            "after": changed,
                        },
                        {
                            "table": "checking_records",
                            "row_id": "checking_record_1",
                            "before": changed,
                            "after": row,
                        },
                    ]
                }
            )
        else:
            env.tools.db.tables["checking_records"]["unrequested"] = row
        return result

    if not transient:
        extra = witness.changed_rows[0].model_copy(deep=True)
        extra.row_id = "unrequested"
        witness.changed_rows.append(extra)
    monkeypatch.setattr(env.tools, "call_discoverable_agent_tool", altered)
    monkeypatch.setattr(environment, "get_environment", lambda *args, **kw: env)
    with pytest.raises(ValueError, match="Forbidden side effect"):
        check_frozen_witness(bundle, "task_checking_service", witness)


def test_native_json_settings_preserve_high_and_do_not_affect_rollouts():
    from tau3.worldgen.v2.json_output import json_llm_args

    original = {"max_tokens": 65536, "extra_body": {"reasoning_effort": "high"}}
    configured = json_llm_args(original)
    assert configured["extra_body"] == {
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
    }
    assert original == {"max_tokens": 65536, "extra_body": {"reasoning_effort": "high"}}


def test_reviewer_format_repair_reaches_reviewer_without_resolving_again(bundle):
    from tau3.worldgen.v2.blind import solve_public

    problem = public_problem(bundle, "task_checking_service")
    candidate = service_witness(bundle).model_dump()
    calls = []

    def ask(model, prompt, payload):
        calls.append((model, payload))
        if model == "solver":
            return candidate
        if len(calls) == 2:
            raise ValueError("Extra data after JSON")
        return {
            "intent_satisfied": True,
            "evidence_sufficient": True,
            "state_correct": True,
            "permissions_correct": True,
            "calculations_complete": True,
            "issues": [],
        }

    result, _, attempts = solve_public(problem, "solver", "reviewer", ask)
    assert result == candidate and attempts == 1
    assert [c[0] for c in calls] == ["solver", "reviewer", "reviewer"]
    assert calls[-1][1]["solution"] == candidate
    assert calls[-1][1]["problem"] == problem
    assert "Extra data" in calls[-1][1]["public_format_feedback"]


@pytest.mark.parametrize("answer", ["", "   ", "..."])
def test_model_cannot_accept_missing_final_communication(monkeypatch, answer):
    from tau3.data_model.message import AssistantMessage, ToolCall
    from tau3.worldgen.v2 import semantic

    calls = []

    def accepts(**kw):
        calls.append(kw)
        return SimpleNamespace(
            content='{"results":[{"index":0,"met":true,"reason":"Tools succeeded"}]}'
        )

    monkeypatch.setattr(semantic, "generate", accepts)
    messages = [
        AssistantMessage(role="assistant", content="Hello, how can I help?"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="t", name="open", arguments={}, requestor="assistant")
            ],
        ),
        AssistantMessage(role="assistant", content=answer),
    ]
    checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions(
        messages, [semantic.COMMUNICATION_INTEGRITY]
    )
    assert len(calls) == 1  # Retain the actual model judgment for calibration.
    assert not checks[0].met
    assert checks[0].justification.startswith("DETERMINISTIC_VETO:")
    messages[-1] = AssistantMessage(role="assistant", content="The account is open.")
    checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions(
        messages, [semantic.COMMUNICATION_INTEGRITY]
    )
    assert checks[0].met


@pytest.mark.parametrize(
    "invalid",
    [
        '{"results":[{"index":0,"met":true,"reason":"a"},{"index":0,"met":false,"reason":"b"}]}',
        '{"results":[{"index":0,"met":true,"reason":"a"}]}',
        '{"results":[{"index":0,"met":"false","reason":"a"},{"index":1,"met":false,"reason":"b"}]}',
        '{"results":[null,null]}',
        '{"results":',
    ],
)
def test_grader_repairs_only_protocol_once(monkeypatch, invalid):
    from tau3.worldgen.v2 import semantic

    valid = '{"results":[{"index":1,"met":false,"reason":"Unsupported date"},{"index":0,"met":true,"reason":"Explains refusal"}]}'
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=invalid if len(calls) == 1 else valid)

    monkeypatch.setattr(semantic, "generate", respond)
    trace = {}
    token = semantic._protocol_context.set(trace)
    try:
        checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions(
            [], ["Explain refusal", "Use correct dates"]
        )
    finally:
        semantic._protocol_context.reset(token)
    assert [c.met for c in checks] == [True, False]
    assert len(calls) == 2 and calls[0]["model"] == calls[1]["model"]
    assert calls[1]["messages"][:2] == calls[0]["messages"]
    feedback = json.loads(calls[1]["messages"][-1].content)
    assert feedback["invalid_response"] == invalid
    assert feedback["required_indices"] == [0, 1]
    assert set(feedback) == {
        "instruction",
        "protocol_error",
        "required_indices",
        "invalid_response",
    }
    assert trace["repaired"] and trace["status"] == "VALID"
    assert trace["attempts"][0]["response"] == invalid
    assert trace["attempts"][0]["status"] == "INVALID_SCHEMA"


@pytest.mark.parametrize(
    "response",
    ['{"results":[]}', '{"results":[{"index":0,"met":false,"reason":"Wrong answer"}]}'],
)
def test_grader_exhausts_repair_but_never_retries_valid_negative(monkeypatch, response):
    from tau3.worldgen.v2 import semantic

    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=response)

    monkeypatch.setattr(semantic, "generate", respond)
    checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions([], ["Correct"])
    assert not checks[0].met
    if response == '{"results":[]}':
        assert len(calls) == 2
        assert checks[0].justification.startswith("INCONCLUSIVE:")
    else:
        assert len(calls) == 1 and checks[0].justification == "Wrong answer"


@pytest.mark.parametrize("budget", [1, 2])
def test_grader_repair_preserves_raw_calls_and_obeys_audit_budget(
    bundle, tmp_path, monkeypatch, budget
):
    import tau3.utils.llm_utils as llm
    from tau3.data_model.tasks import Task
    from tau3.worldgen.v2 import semantic
    from tau3.worldgen.v2.calibration import protocol_summary

    monkeypatch.setenv("TAU3_SYNTH_WORLD", str(bundle))
    task = Task.model_validate_json(
        (bundle / "tasks/task_checking_service.json").read_text()
    )
    assertions = task.evaluation_criteria.nl_assertions
    valid = json.dumps(
        {
            "results": [
                {"index": i, "met": False, "reason": "No answer"}
                for i in range(len(assertions))
            ]
        }
    )
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"results":[]}' if len(calls) == 1 else valid)

    monkeypatch.setattr(llm, "generate", respond)
    session = AuditSession(bundle, tmp_path / "audit", load_settings(), budget, "test")
    with semantic.audited_judge(session, "judge"):
        reward = semantic.StrictSemanticEvaluator.calculate_reward(task, [])
    trace = reward.info["protocol"]
    assert len(calls) == budget
    assert len(list((tmp_path / "audit/calls").glob("*.json"))) == budget
    assert trace["attempts"][0]["response"] == '{"results":[]}'
    assert trace["status"] == ("VALID" if budget == 2 else "INCONCLUSIVE")
    assert all(
        c.justification.startswith("INCONCLUSIVE:") for c in reward.nl_assertions
    ) == (budget == 1)
    row = {
        "status": "PASS" if budget == 2 else "INCONCLUSIVE",
        "reward": {"info": {"nl": reward.info}},
    }
    stats = protocol_summary([row])
    assert stats["first_valid_rate"] == 0
    assert stats["repair_attempts"] == 1
    assert stats["repair_successes"] == budget - 1
    assert stats["final_correct_rate"] == budget - 1
    # Resume reuses the exact first response and the distinct repair request.
    with semantic.audited_judge(session, "judge"):
        repeated = semantic.StrictSemanticEvaluator.calculate_reward(task, [])
    assert repeated.info == reward.info and len(calls) == budget


def test_public_catalog_exact_reading_and_no_knowledge_isolation(bundle):
    from tau3.worldgen.v2.environment import get_environment

    env = get_environment(bundle, retrieval_variant="bm25")
    catalog = json.loads(env.tools.KB_list_documents("checking"))
    assert catalog["documents"] and catalog["next_offset"] is None
    ids = [d["id"] for d in catalog["documents"][:10]]
    docs = json.loads(env.tools.KB_read_documents(ids))
    assert [d["id"] for d in docs] == ids
    assert all(d["content"] for d in docs)
    assert json.loads(env.tools.KB_list_documents("nonexistent"))["total"] == 0
    with pytest.raises(ValueError):
        env.tools.KB_read_documents(["private/reference.json"])
    with pytest.raises(ValueError):
        env.tools.KB_read_documents(ids * 11)
    with pytest.raises(ValueError):
        env.tools.KB_list_documents(offset=-1)
    isolated = get_environment(bundle, retrieval_variant="no_knowledge")
    assert "KB_list_documents" not in isolated.tools.tools
    assert "KB_read_documents" not in isolated.tools.tools


def test_public_projection_uses_actual_variant_instructions(bundle):
    original = public_problem(bundle, "task_checking_service")
    instructions = original["request"] + " Please keep the answer concise."
    variant = public_problem(
        bundle, "variant_without_seed_file", instructions=instructions
    )
    assert variant == {**original, "request": instructions}


def test_task_expression_guards_preserve_facts():
    from tau3.synthesis.world_tasks import expression_errors

    source = "You are user_1, email user_1@example.invalid. Pay 20 dollars to record_2."
    assert not expression_errors(source, source + " Please be concise.")
    assert expression_errors(source, source.replace("20", "21"))
    assert expression_errors(source, source.replace("record_2", "record_3"))
    assert expression_errors(source, source.replace("example.invalid", "other.invalid"))
    assert expression_errors(source, source + " Call hidden_operation.")
    assert expression_errors(source, None)


def test_world_task_synthesis_refuses_unqualified_seed_before_calls(
    bundle, tmp_path, monkeypatch
):
    import tau3.utils.llm_utils as llm
    from tau3.synthesis.world_tasks import synthesize_world_tasks

    path = tmp_path / "readiness.json"
    write_json(path, {"status": "INCONCLUSIVE"})
    monkeypatch.setattr(
        llm, "generate", lambda **kwargs: pytest.fail("Unqualified seed called a model")
    )
    with pytest.raises(ValueError, match="readiness has not passed"):
        synthesize_world_tasks(
            bundle,
            path,
            tmp_path / "tasks",
            20,
            __import__("pathlib").Path(
                "configs/worldgen/v2-verification-catalog-high.yaml"
            ),
            200,
        )
    assert not (tmp_path / "tasks/manifest.json").exists()


def test_world_task_loader_never_accepts_draft(tmp_path):
    from tau3.synthesis.bundle import load_bundle_manifest

    write_json(
        tmp_path / "manifest.json", {"domain": "banking_synth", "status": "draft"}
    )
    with pytest.raises(ValueError, match="not published"):
        load_bundle_manifest(tmp_path)


@pytest.mark.parametrize(
    "expression,result", [("5=0", False), ("5 = 5", True), ("3.5 = 3.50", True)]
)
def test_public_calculator_accepts_literal_mathematical_equality(expression, result):
    assert calculate(expression) is result


@pytest.mark.parametrize("expression", ["value=0", "1/0", "5 === 0"])
def test_invalid_calculation_feedback_identifies_public_expression(expression):
    with pytest.raises(ValueError, match="Invalid public calculation") as error:
        calculate(expression)
    assert repr(expression) in str(error.value)


def test_citation_spans_extract_exact_public_text_without_fixing_false_quotes():
    from tau3.worldgen.v2.blind import materialize_citations

    problem = {"documents": {"d": {"content": "Heading\n\nRate is 4%.\nNo guarantee."}}}
    raw = {"evidence": [{"source": "d", "start_line": 3, "end_line": 4}]}
    result = materialize_citations(problem, raw)
    assert result["evidence"][0]["quote"] == "Rate is 4%.\nNo guarantee."
    assert "quote" not in raw["evidence"][0]
    assert materialize_citations(problem, result) == result
    false_quote = {"evidence": [{"source": "d", "quote": "Rate is 5%."}]}
    assert materialize_citations(problem, false_quote) == false_quote
    for source, start, end in [
        ("private/goal", 1, 1),
        ("d", 0, 2),
        ("d", 1, 5),
        ("d", True, 2),
    ]:
        with pytest.raises(ValueError):
            materialize_citations(
                problem,
                {
                    "evidence": [
                        {"source": source, "start_line": start, "end_line": end}
                    ]
                },
            )


@pytest.mark.parametrize("online_reward", [0, 1, "infra", "judge_unknown", "partial"])
def test_world_task_candidates_are_bounded_and_publication_requires_both_runs(
    bundle, tmp_path, monkeypatch, online_reward
):
    from pathlib import Path

    from tau3.data_model.tasks import Task
    from tau3.domains.banking_synth import environment as domain
    from tau3.runner import batch
    from tau3.synthesis import world_tasks
    from tau3.synthesis.bundle import load_task_bundle

    anchor = Task.model_validate_json(
        (bundle / "tasks/task_checking_service.json").read_text()
    )
    monkeypatch.setattr(domain, "get_tasks", lambda: [anchor])
    monkeypatch.setattr(
        world_tasks, "check_readiness", lambda *args: {"status": "PASS"}
    )
    proof = tmp_path / "ready.json"
    write_json(proof, {"status": "PASS", "sources": {}})
    monkeypatch.setattr(
        world_tasks,
        "solve_public",
        lambda *args: (service_witness(bundle).model_dump(), {}, 1),
    )
    attempts = []

    def ask(self, model, system, payload):
        if "instructions" in payload:
            attempt = payload["variant"][1]
            attempts.append(attempt)
            if attempt == 0:
                return {
                    "request": anchor.user_scenario.instructions + " Pay 999 dollars."
                }
            return {
                "request": anchor.user_scenario.instructions
                + (" Please be concise." if attempt == 1 else " Thank you.")
            }
        return {"equivalent": True, "no_solution_added": True, "issues": []}

    monkeypatch.setattr(AuditSession, "ask", ask)
    online_calls = []
    saved_results = {}

    def run(config, tasks, save_path, console_display):
        online_calls.append((config.llm_agent, [t.id for t in tasks]))
        write_json(
            save_path, {"task_ids": [t.id for t in tasks], "test_reward": online_reward}
        )
        result = SimpleNamespace(
            tasks=tasks,
            info=SimpleNamespace(
                world_artifact_hash=world_tasks.digest(
                    world_tasks.artifact_hashes(bundle)
                ),
                world_implementation_hash=world_tasks.implementation_hash(),
                world_evaluation_hash=world_tasks.digest(
                    world_tasks.load_settings(kwargs["config"]).model_dump()
                ),
                agent_info=SimpleNamespace(llm=config.llm_agent),
                user_info=SimpleNamespace(llm=config.llm_user),
                max_steps=config.max_steps,
                seed=config.seed,
                retrieval_config="bm25",
            ),
            simulations=[
                SimpleNamespace(
                    task_id=t.id,
                    termination_reason=SimpleNamespace(
                        value="infrastructure_error"
                        if online_reward == "infra"
                        else "user_stop"
                    ),
                    reward_info=SimpleNamespace(
                        reward=online_reward
                        if isinstance(online_reward, int)
                        else int(online_reward == "partial"),
                        nl_assertions=[
                            SimpleNamespace(justification="INCONCLUSIVE: Timeout")
                        ]
                        if online_reward == "judge_unknown"
                        else [],
                    ),
                )
                for t in tasks
            ],
        )
        if online_reward == "partial" and len(online_calls) == 1:
            result.simulations = []
        saved_results[save_path] = result
        return result

    monkeypatch.setattr(batch, "run_tasks", run)
    if online_reward == "partial":
        from tau3.data_model.simulation import Results

        monkeypatch.setattr(Results, "load", lambda path: saved_results[path])
    out = tmp_path / "generated"
    kwargs = dict(
        world=bundle,
        readiness=proof,
        output=out,
        count=1,
        config=Path("configs/worldgen/v2-verification-catalog-high.yaml"),
        max_calls=30,
    )
    if online_reward == "partial":
        with pytest.raises(ValueError, match="Incomplete task conversation membership"):
            world_tasks.synthesize_world_tasks(**kwargs)
        manifest = world_tasks.synthesize_world_tasks(**kwargs, resume=True)
        assert manifest["status"] == "published" and len(online_calls) == 3
        assert list((out / "validation").glob("*_checkpoint_*.json"))
    elif online_reward in {"infra", "judge_unknown"}:
        with pytest.raises(AuditIncomplete, match="infrastructure/judge"):
            world_tasks.synthesize_world_tasks(**kwargs)
        assert len(online_calls) == 1 and attempts == [0, 1]
        assert json.loads((out / "manifest.json").read_text())["status"] == "draft"
    elif online_reward == 0:
        with pytest.raises(ValueError, match="three candidates"):
            world_tasks.synthesize_world_tasks(**kwargs)
        assert json.loads((out / "manifest.json").read_text())["status"] == "draft"
        assert attempts == [0, 1, 2] and len(online_calls) == 4
        assert not (out / "tasks.json").exists()
    else:
        manifest = world_tasks.synthesize_world_tasks(**kwargs)
        assert attempts == [0, 1] and len(online_calls) == 2
        assert manifest["admitted_tasks"] == 1
        tasks = load_task_bundle(out)
        assert tasks[0].evaluation_criteria == anchor.evaluation_criteria
        assert tasks[0].id != anchor.id
        assert tasks[0].id.startswith(f"synth_{manifest['bundle_id']}_")
        assert manifest["task_anchors"][tasks[0].id] == anchor.id
        assert (
            manifest["task_groups"][tasks[0].id]
            == world_tasks.task_structures(bundle)[anchor.id]
        )
        seed_test = json.loads((bundle / "splits.json").read_text())["test"]
        actual_test = json.loads((out / "split_tasks.json").read_text())["validation"]
        assert (tasks[0].id in actual_test) == (anchor.id in seed_test)
        assert world_tasks.synthesize_world_tasks(**kwargs, resume=True) == manifest
        assert len(online_calls) == 2
        other = world_tasks.synthesize_world_tasks(
            **{**kwargs, "output": tmp_path / "second-batch"}
        )
        assert other["bundle_id"] != manifest["bundle_id"]
        assert load_task_bundle(tmp_path / "second-batch")[0].id != tasks[0].id
        (out / "validation/online_round_1_0.json").write_text("{}")
        with pytest.raises(ValueError, match="evidence changed"):
            load_task_bundle(out)


def test_large_world_task_batch_requires_real_pilot(bundle, tmp_path):
    from pathlib import Path

    from tau3.synthesis.world_tasks import synthesize_world_tasks

    with pytest.raises(ValueError, match="published 20-task pilot"):
        synthesize_world_tasks(
            bundle,
            tmp_path / "no-readiness.json",
            tmp_path / "batch",
            200,
            Path("configs/worldgen/v2-verification-catalog-high.yaml"),
            6000,
        )
    assert not (tmp_path / "batch/manifest.json").exists()


@pytest.mark.parametrize("outcome", ["pass", "worker_failure", "budget_overflow"])
def test_blind_shards_preserve_budget_and_require_native_replay(
    bundle, tmp_path, monkeypatch, outcome
):
    import importlib.util
    from pathlib import Path

    from tau3.worldgen.v2 import blind

    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/validate_worldgen_v2_independent.py"
    )
    spec = importlib.util.spec_from_file_location("qualification_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    replayed = []

    def replay(world, output, ids, **kwargs):
        replayed.append(ids)
        assert len(list((output / "calls").glob("*.json"))) == 4
        assert json.loads((output / "audit.json").read_text())["max_calls"] == 8
        return {"status": "PASS", "task_ids": ids}

    monkeypatch.setattr(blind, "run_blind", replay)

    def worker(command, **kwargs):
        output = Path(command[command.index("--worker-output") + 1])
        budget = int(command[command.index("--worker-max-calls") + 1])
        assert budget == 2
        session = AuditSession(bundle, output, load_settings(), budget, "blind-v1")
        ids = json.loads((output / "tasks.json").read_text())
        write_json(
            output / "report.json",
            {
                "status": "FAIL" if outcome == "worker_failure" else "PASS",
                "task_ids": ids,
                "identity": session.identity,
            },
        )
        write_json(
            output / "calls" / f"worker_{output.name}.json",
            {
                "status": "COMPLETE",
                "request": {"model": output.name, "messages": []},
                "response": "{}",
                "finish_reason": "stop",
                "attempts": [{"status": "COMPLETE"}]
                * (3 if outcome == "budget_overflow" else 1),
            },
        )
        return SimpleNamespace(wait=lambda: 2 if outcome == "worker_failure" else 0)

    monkeypatch.setattr(driver.subprocess, "Popen", worker)
    output = tmp_path / "full/blind"
    if outcome == "pass":
        assert (
            driver.sharded_blind(bundle, output, None, max_calls=8)["status"] == "PASS"
        )
        assert len(replayed) == 1
        assert (
            json.loads((output / "shard-provenance.json").read_text())["physical_calls"]
            == 4
        )
    else:
        with pytest.raises(ValueError, match="incomplete|budget"):
            driver.sharded_blind(bundle, output, None, max_calls=8)
        assert not replayed


@pytest.mark.parametrize("worker_failure", [False, True])
@pytest.mark.parametrize("request_limit", [None, 128])
def test_online_model_workers_use_native_results_and_fixed_settings(
    bundle, tmp_path, monkeypatch, worker_failure, request_limit
):
    import importlib.util
    from pathlib import Path

    from tau3.runner import batch
    from tau3.worldgen.v2 import behaviour, rollouts

    if request_limit is not None:
        monkeypatch.setenv("TAU3_LLM_CONCURRENCY", str(request_limit))
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/validate_worldgen_v2_independent.py"
    )
    spec = importlib.util.spec_from_file_location("online_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    config = Path("configs/worldgen/v2-verification-transport-high.yaml")
    settings = load_settings(config)
    commands, aggregate = [], []

    def worker(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(wait=lambda: int(worker_failure))

    def summarize_workers(world, output, **kwargs):
        aggregate.append(kwargs)
        matrix = json.loads((output / "matrix.json").read_text())
        assert matrix["settings_hash"] == driver.digest(settings.model_dump())
        assert matrix["num_tasks"] == len(kwargs["task_ids"])
        assert matrix["max_steps"] == 80 and matrix["retrieval"] == "bm25"
        return {"status": "PASS"}

    monkeypatch.setattr(driver.subprocess, "Popen", worker)
    monkeypatch.setattr(rollouts, "run_rollouts", summarize_workers)
    output = tmp_path / "online"
    if worker_failure:
        with pytest.raises(ValueError, match="Model worker incomplete"):
            driver.online_condition(bundle, output, "bm25", config)
        assert not aggregate
        return
    assert driver.online_condition(bundle, output, "bm25", config)["status"] == "PASS"
    assert len(aggregate) == 1
    assert {c[c.index("--online-model") + 1] for c in commands} == set(
        settings.agent_models
    )
    executed = []

    def run(run_config, tasks, **kwargs):
        executed.append(run_config)
        assert run_config.llm_agent == settings.agent_models[0]
        assert run_config.max_concurrency == (32 if request_limit else 2)
        assert run_config.max_retries == 0
        assert run_config.llm_args_agent == settings.llm_args
        assert (
            kwargs["save_path"].name
            == f"results_{driver.digest(settings.agent_models[0])[:12]}.json"
        )
        return SimpleNamespace()

    monkeypatch.setattr(batch, "run_tasks", run)
    monkeypatch.setattr(
        behaviour, "summarize", lambda result, expected: {"tasks": len(expected)}
    )
    assert (
        driver.online_condition(
            bundle, output, "bm25", config, settings.agent_models[0]
        )["status"]
        == "PASS"
    )
    assert len(executed) == 1


def test_response_reuse_preserves_invalid_content_and_original_identity(
    bundle, tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    import tau3.utils.llm_utils as llm

    path = (
        Path(__file__).resolve().parents[3] / "scripts/qualify_world_task_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("reuse_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    source, output = tmp_path / "old", tmp_path / "new"
    settings = load_settings()
    session = AuditSession(bundle, source, settings, 3, "blind-v1")
    replies = iter(['{"ok":true}', '[{"invalid":"array"}]'])
    monkeypatch.setattr(
        llm, "generate", lambda **kwargs: SimpleNamespace(content=next(replies))
    )
    assert session.ask("model", "system", {"a": 1}) == {"ok": True}
    with pytest.raises(ValueError):
        session.ask("model", "system", {"a": 2})
    old = json.loads((source / "audit.json").read_text())
    old["implementation"] = "previous_checker"
    write_json(source / "audit.json", old)
    report = driver.reuse_blind_responses(source, output, bundle, None)
    assert len(report["imported"]) == 2 and report["excluded"] == []
    assert not (output / "report.json").exists()
    for p in (output / "calls").glob("*.json"):
        assert json.loads(p.read_text())["origin"]["audit_identity"] == old
    monkeypatch.setattr(
        llm,
        "generate",
        lambda **kwargs: pytest.fail("Reused request made a new API call"),
    )
    replay = AuditSession(bundle, output, settings, 3, "blind-v1")
    assert replay.ask("model", "system", {"a": 1}) == {"ok": True}
    with pytest.raises(ValueError):
        replay.ask("model", "system", {"a": 2})
    old["settings"] = "different_settings"
    write_json(source / "audit.json", old)
    with pytest.raises(ValueError, match="identical public artifacts/settings"):
        driver.reuse_blind_responses(source, tmp_path / "incompatible", bundle, None)


def test_larger_batch_rejects_pilot_expressions_before_blind_calls(
    bundle, tmp_path, monkeypatch
):
    from pathlib import Path

    from tau3.data_model.tasks import Task
    from tau3.domains.banking_synth import environment as domain
    from tau3.synthesis import bundle as bundles
    from tau3.synthesis import world_tasks

    anchor = Task.model_validate_json(
        (bundle / "tasks/task_checking_service.json").read_text()
    )
    pilot_expression = anchor.user_scenario.instructions + " Please be concise."
    pilot = [anchor.model_copy(deep=True) for _ in range(20)]
    for index, task in enumerate(pilot):
        task.id = f"pilot_{index}"
        task.user_scenario.instructions = pilot_expression
    config = Path("configs/worldgen/v2-verification-transport-high.yaml")
    settings = load_settings(config)
    monkeypatch.setattr(bundles, "load_task_bundle", lambda *args: pilot)
    monkeypatch.setattr(
        world_tasks,
        "load_world_task_manifest",
        lambda *args: {
            "settings_hash": world_tasks.digest(settings.model_dump()),
        },
    )
    monkeypatch.setattr(world_tasks, "check_readiness", lambda *args: {})
    monkeypatch.setattr(domain, "get_tasks", lambda: [anchor])
    monkeypatch.setattr(
        world_tasks,
        "solve_public",
        lambda *args: pytest.fail("Pilot duplicate reached blind solving"),
    )
    monkeypatch.setattr(
        AuditSession,
        "ask",
        lambda self, model, system, payload: {"request": pilot_expression}
        if "instructions" in payload
        else {"equivalent": True, "no_solution_added": True, "issues": []},
    )
    ready = tmp_path / "readiness.json"
    write_json(ready, {"status": "PASS", "sources": {}})
    output = tmp_path / "new-batch"
    with pytest.raises(ValueError, match="three candidates"):
        world_tasks.synthesize_world_tasks(
            bundle, ready, output, 21, config, 100, pilot_bundle=tmp_path / "pilot"
        )
    report = json.loads((output / "validation/admission.json").read_text())
    assert report["admitted"] == 0
    assert all("Duplicate" in row["reason"] for row in report["rejected"])


def test_namespaced_task_uses_actual_request_in_complete_evaluator(bundle, monkeypatch):
    from tau3.data_model.simulation import SimulationRun, TerminationReason
    from tau3.data_model.tasks import Task
    from tau3.domains.banking_synth.environment import get_environment
    from tau3.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau3.registry import registry
    from tau3.worldgen.v2 import semantic
    from tau3.worldgen.v2.calibration import trajectory

    monkeypatch.setenv("TAU3_SYNTH_WORLD", str(bundle))
    if "banking_synth" not in registry.get_domains():
        registry.register_domain(get_environment, "banking_synth")
    task = Task.model_validate_json(
        (bundle / "tasks/task_checking_service.json").read_text()
    )
    task.id = "synth_batch_namespace_000000_a0_task_checking_service"
    task.user_scenario.instructions += " Please be concise."
    assert not (bundle / "tasks" / f"{task.id}.json").exists()
    messages = trajectory(
        bundle,
        task,
        [a.model_dump(mode="json") for a in task.evaluation_criteria.actions],
        "Withdrew 20 USD. The remaining balance is 30 USD.",
    )
    observed = []

    def judge(**kwargs):
        payload = json.loads(kwargs["messages"][-1].content)
        observed.append(payload)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "index": i,
                            "met": True,
                            "reason": "The supplied test answer agrees with the successful tool result.",
                        }
                        for i in range(len(payload["assertions"]))
                    ]
                }
            )
        )

    monkeypatch.setattr(semantic, "generate", judge)
    simulation = SimulationRun(
        id="namespaced-test",
        task_id=task.id,
        start_time="2025-11-14T00:00:00",
        end_time="2025-11-14T00:00:01",
        duration=1,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
    )
    reward = evaluate_simulation(
        simulation,
        task,
        EvaluationType.ALL,
        False,
        "banking_synth",
        env_kwargs={"retrieval_variant": "no_knowledge"},
    )
    assert reward.reward == 1 and len(observed) == 1
    assert observed[0]["public_context"]["request"] == task.user_scenario.instructions


def test_concurrent_audit_reserves_budget_and_reuses_identical_requests(
    bundle, tmp_path, monkeypatch
):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    from tau3.data_model.message import AssistantMessage
    from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
    from tau3.utils import llm_utils

    count, guard = [], Lock()

    def generate(**kwargs):
        with guard:
            count.append(kwargs)
        time.sleep(0.02)
        return AssistantMessage(role="assistant", content='{"ok":true}')

    monkeypatch.setattr(llm_utils, "generate", generate)
    settings = load_settings()
    session = ConcurrentAuditSession(bundle, tmp_path / "audit", settings, 4, "test")
    with ThreadPoolExecutor(max_workers=16) as executor:
        replies = list(
            executor.map(
                lambda _: session.ask("model", "public", {"same": True}), range(16)
            )
        )
    assert len(count) == 1 and all(reply == {"ok": True} for reply in replies)

    def unique(index):
        try:
            return session.ask("model", "public", {"index": index})
        except AuditIncomplete:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        replies = list(executor.map(unique, range(16)))
    assert len(count) == 4
    assert sum(reply is not None for reply in replies) == 3
    assert session.transport_summary()["physical_calls"] == 4
    resumed = ConcurrentAuditSession(bundle, tmp_path / "audit", settings, 4, "test")
    assert resumed.ask("model", "public", {"same": True}) == {"ok": True}
    assert len(count) == 4


def test_concurrent_audit_does_not_resample_interrupted_request(
    bundle, tmp_path, monkeypatch
):
    from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
    from tau3.utils import llm_utils

    def fail(**kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(llm_utils, "generate", fail)
    session = ConcurrentAuditSession(
        bundle, tmp_path / "audit", load_settings(), 10, "test"
    )
    with pytest.raises(AuditIncomplete):
        session.ask("model", "public", {})
    monkeypatch.setattr(
        llm_utils, "generate", lambda **kwargs: pytest.fail("must not retry")
    )
    with pytest.raises(AuditIncomplete, match="Previous request"):
        session.ask("model", "public", {})


def test_concurrent_audit_keeps_precharged_crash_slot(bundle, tmp_path, monkeypatch):
    from tau3.utils import llm_utils

    output = tmp_path / "audit"
    settings = load_settings()
    session = ConcurrentAuditSession(bundle, output, settings, 1, "test")
    budget = json.loads((output / "budget.json").read_text())
    # Simulate a crash between persisting the budget and the request record.
    budget.update(used=1, last_reservation={"request": "interrupted", "attempt": 1})
    write_json(output / "budget.json", budget)
    session = ConcurrentAuditSession(bundle, output, settings, 1, "test")
    monkeypatch.setattr(
        llm_utils, "generate", lambda **kw: pytest.fail("charged slot reused")
    )
    with pytest.raises(AuditIncomplete, match="budget"):
        session.ask("model", "public", {})


@pytest.mark.parametrize("full_code", [0, 2])
def test_qualification_resume_keeps_pilot_gate_and_prepare_only(
    tmp_path, monkeypatch, full_code
):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3] / "scripts/qualify_world_task_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("qualification_resume_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    write_json(tmp_path / "local-worlds.json", {"scale": "/tmp/identical-scale"})
    commands = []

    def call(command, **kwargs):
        commands.append(command)
        return full_code if len(commands) == 1 else 0

    monkeypatch.setattr(driver.subprocess, "call", call)
    with pytest.raises(SystemExit) as exc:
        driver.resume_full(tmp_path, Path("config.yaml"))
    assert exc.value.code == full_code
    state = json.loads((tmp_path / "stage-state.json").read_text())
    if full_code:
        assert len(commands) == 1 and state["phase"] == "full_incomplete"
    else:
        assert len(commands) == 2 and state["phase"] == "READY"
        assert "--prepare-only" in commands[1]
    assert "--stage" in commands[0] and "full" in commands[0]


def test_task_candidates_generate_and_verify_in_parallel_before_rollouts(
    bundle, tmp_path, monkeypatch
):
    from threading import Barrier

    from tau3.data_model.tasks import Task
    from tau3.domains.banking_synth import environment as domain
    from tau3.runner import batch
    from tau3.synthesis import world_tasks

    anchor = Task.model_validate_json(
        (bundle / "tasks/task_checking_service.json").read_text()
    )
    monkeypatch.setattr(domain, "get_tasks", lambda: [anchor])
    monkeypatch.setattr(world_tasks, "check_readiness", lambda *args: {})
    proof = tmp_path / "ready.json"
    write_json(proof, {"sources": {}})
    generation, verification = Barrier(2, timeout=10), Barrier(2, timeout=10)

    def ask(self, model, system, payload):
        if "variant" in payload:
            generation.wait()
            suffix = [" Please be concise.", " Please use a formal tone."][
                payload["variant"][0]
            ]
            return {"request": anchor.user_scenario.instructions + suffix}
        return {"equivalent": True, "no_solution_added": True, "issues": []}

    witness = service_witness(bundle).model_dump()

    def solve(*args):
        verification.wait()
        return witness, {}, 1

    def rollout(config, tasks, **kwargs):
        assert len(tasks) == 2
        assert "_000000_" in tasks[0].id and "_000001_" in tasks[1].id
        assert len(list((tmp_path / "tasks/validation").glob("*/blind_*.json"))) == 4
        raise RuntimeError("test stops after verifying parallel candidates")

    monkeypatch.setattr(AuditSession, "ask", ask)
    monkeypatch.setattr(world_tasks, "solve_public", solve)
    monkeypatch.setattr(batch, "run_tasks", rollout)
    with pytest.raises(RuntimeError, match="parallel candidates"):
        world_tasks.synthesize_world_tasks(
            bundle,
            proof,
            tmp_path / "tasks",
            2,
            __import__("pathlib").Path(
                "configs/worldgen/v2-verification-transport-high.yaml"
            ),
            60,
        )
