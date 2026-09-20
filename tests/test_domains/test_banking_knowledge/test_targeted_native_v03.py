"""v03 regression contracts: protocol, clean export, policy branches and frozen quotas."""

import json
from collections import Counter

import pytest

from tau3.synthesis.storage import digest, write_json
from tau3.synthesis.targeted.native.models import NativeConfig, NativeSlot
from tau3.synthesis.targeted.native.protocol import (
    audit_protocol,
    is_protocol_misuse,
    tool_error,
)
from tau3.synthesis.targeted.native.scenarios import compile_candidate
from tau3.synthesis.targeted.native.v03_planning import recipe_counts
from tau3.synthesis.targeted.native.validation import validate_static


def slot(family, branch, difficulty="10-14", **overrides):
    values = dict(index=0, split="pilot", family=family, branch_id=branch,
                  difficulty=difficulty, origin="current", seed=43000,
                  trial_seeds=[1, 2, 3, 4], evidence_ids=["finding:0"], v03_labels=["A", "B", "F"])
    return NativeSlot(**(values | overrides))


def invocation(identifier, name, arguments, result="OK", error=False):
    return [{"role": "assistant", "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}]},
            {"role": "tool", "id": identifier, "content": result, "error": error}]


def test_protocol_real_unlock_and_call_and_same_batch_rejection():
    name = "get_all_user_accounts_by_user_id_3847"
    unlock = invocation("u", "unlock_discoverable_agent_tool", {"agent_tool_name": name})
    call = invocation("c", "call_discoverable_agent_tool", {"agent_tool_name": name, "arguments": '{"user_id":"customer"}'})
    assert audit_protocol(unlock + call)["clean"]
    assert not audit_protocol(call)["clean"]
    batch = [{"role": "assistant", "tool_calls": unlock[0]["tool_calls"] + call[0]["tool_calls"]}, unlock[1], call[1]]
    assert "call_without_prior_unlock" in {e["issue"] for e in audit_protocol(batch)["errors"]}


@pytest.mark.parametrize("arguments,expected", [
    ({"capability": "get_all_user_accounts_by_user_id_3847", "arguments": "{}"}, "invalid_selector_or_wrapper_signature"),
    ({"agent_tool_name": "get_all_user_accounts_by_user_id_3847", "arguments": {}}, "nested_arguments_not_string"),
    ({"agent_tool_name": "get_all_user_accounts_by_user_id_3847", "arguments": "{"}, "invalid_nested_json"),
    ({"agent_tool_name": "get_all_user_accounts_by_user_id_3847", "arguments": '{"user_id":false}'}, "argument_type"),
    ({"agent_tool_name": "get_all_user_accounts_by_user_id_0000", "arguments": "{}"}, "unknown_tool"),
])
def test_protocol_rejects_invalid_calls_even_if_runtime_returns_success(arguments, expected):
    evidence = audit_protocol(invocation("c", "call_discoverable_agent_tool", arguments))
    assert expected in {e["issue"] for e in evidence["errors"]}


def test_tool_errors_do_not_match_retrieved_document_examples():
    assert tool_error({"content": "Error: Unknown tool"})
    assert tool_error({"content": '{"error":"bad arguments"}'})
    assert not tool_error({"content": "Retrieved document\nError: example of an invalid call"})


def test_missing_business_argument_is_not_misclassified_as_unlock_failure():
    name = "get_all_user_accounts_by_user_id_3847"
    messages = invocation("u", "unlock_discoverable_agent_tool", {"agent_tool_name": name})
    messages += invocation("c", "call_discoverable_agent_tool", {"agent_tool_name": name, "arguments": "{}"},
                           result="Error: missing user_id")
    audit = audit_protocol(messages)
    assert not audit["clean"]
    assert any(e["issue"] == "argument_signature" for e in audit["errors"])
    assert not any(is_protocol_misuse(e) for e in audit["errors"])


@pytest.mark.parametrize("family,branch,difficulty", [
    ("credit_limit", "deny:high_utilization", "10-14"),
    ("credit_limit", "deny:cooldown_period_active", "10-14"),
    ("credit_limit", "overlimit", "1-4"),
    ("transaction_disputes", "credit", "10-14"),
    ("transaction_disputes", "debit", "10-14"),
    ("replacement_closure", "close", "20-29"),
    ("replacement_closure", "replace", "5-9"),
])
def test_v03_business_replay_and_adversarial_calls(family, branch, difficulty):
    candidate = compile_candidate(slot(family, branch, difficulty))
    checks = validate_static(candidate)
    assert checks["positive"] and checks["alternative_query"]
    if difficulty != "1-4":
        assert checks["counterexamples"]["wrong_selector"]
        assert checks["counterexamples"]["hidden_bypass"]


def test_official_approval_collision_is_not_mislabeled_valid():
    candidate = compile_candidate(slot("credit_limit", "approve"))
    with pytest.raises(ValueError, match="Unsatisfied cli_decision"):
        validate_static(candidate)


@pytest.mark.parametrize("revision,status", [("official", "CLOSED"), ("native_cli_approval_v1", "CLOSED"),
                                              ("native_card_lifecycle_v2", "ACTIVE")])
def test_replacement_revision_separates_physical_card_from_account(revision, status):
    from tau3.synthesis.targeted.native.validation import (
        check_goals,
        reference_operations,
    )
    from tau3.synthesis.validation import replay

    candidate = compile_candidate(slot("replacement_closure", "replace", "5-9", runtime_revision=revision))
    env, _ = replay(candidate.task)
    goal = next(g for g in candidate.goals if g["kind"] == "row_match")
    cid = goal["selector"]["credit_card_account_id"]
    account = env.tools.db.credit_card_accounts.data[cid]
    assert account["status"] == status
    orders = [r for r in env.tools.db.credit_card_orders.data.values() if r.get("credit_card_account_id") == cid]
    assert len(orders) == 1 and orders[0]["old_card_cancelled"] is True
    if revision == "native_card_lifecycle_v2":
        assert "closed_date" not in account
        assert not check_goals(candidate, env, reference_operations(candidate))
        account["status"] = "CLOSED"
        assert check_goals(candidate, env, reference_operations(candidate))
    assert validate_static(candidate)["positive"]


def test_audit_indices_are_explicit_bounded_and_do_not_modify_capture():
    from pydantic import ValidationError

    from tau3.synthesis.targeted.native.models import NativeQuality
    from tau3.synthesis.targeted.native.quality import indexed_audit_context

    sample = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                           {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "done"}]}
    original = json.dumps(sample)
    context, schema = indexed_audit_context(sample)
    assert [m["assistant_turn_index"] for m in context["messages"] if m["role"] == "assistant"] == [0, 1]
    assert schema["properties"]["erroneous_assistant_turns"]["items"]["enum"] == [0, 1]
    assert json.dumps(sample) == original
    review = dict(reasoning_correct=True, grounded_in_visible_history=True, user_compliant=True,
                  handoff_correct=True, final_answer_correct=True, no_private_leakage=True,
                  erroneous_assistant_turns=[True], unlocalizable_error=False, explanation="bad index")
    with pytest.raises(ValidationError):
        NativeQuality.model_validate(review)


@pytest.mark.parametrize("revision", ["official", "native_cli_approval_v1", "native_card_lifecycle_v2"])
def test_statement_credit_is_account_scoped_only_in_new_runtime(revision):
    from tau3.synthesis.validation import replay

    candidate = compile_candidate(slot("replacement_closure", "retention_credit", "20-29",
                                       runtime_revision=revision))
    credits = [g for g in candidate.goals if g["kind"] == "required_operation"
               and g["name"] == "apply_statement_credit_8472"]
    assert len(credits) >= 2
    if revision != "native_card_lifecycle_v2":
        with pytest.raises(ValueError, match="Failed to apply statement credit"):
            validate_static(candidate)
        return
    checks = validate_static(candidate)
    assert checks["positive"] and all(checks["counterexamples"].values())
    environment, _ = replay(candidate.task)
    rows = [r for r in environment.tools.db.credit_card_transaction_history.data.values()
            if r.get("user_id") == candidate.facts["identity"]["user_id"]]
    assert len(rows) == len(credits)
    assert len({r["transaction_id"] for r in rows}) == len(credits)
    assert {r["credit_card_account_id"] for r in rows} == {
        g["arguments"]["credit_card_account_id"] for g in credits}
    before = environment.tools.db.model_dump(mode="json")
    duplicate = environment.tools.apply_statement_credit_8472(
        user_id=candidate.facts["identity"]["user_id"], **credits[0]["arguments"])
    assert duplicate.startswith("Error:")
    assert environment.tools.db.model_dump(mode="json") == before


@pytest.mark.parametrize("difficulty", ["5-9", "10-14"])
def test_replacement_does_not_require_closure_only_dispute_query(difficulty):
    from tau3.data_model.tasks import Action
    from tau3.synthesis.targeted.native.validation import reference_operations
    from tau3.synthesis.validation import hashes, replay

    candidate = compile_candidate(slot("replacement_closure", "replace", difficulty,
                                       runtime_revision="native_card_lifecycle_v2"))
    assert "get_user_dispute_history_7291" not in {c["name"] for c in reference_operations(candidate)}
    reference, _ = replay(candidate.task)
    optional = [Action(action_id="optional_unlock", name="unlock_discoverable_agent_tool",
                       arguments={"agent_tool_name": "get_user_dispute_history_7291"}),
                Action(action_id="optional_read", name="call_discoverable_agent_tool",
                       arguments={"agent_tool_name": "get_user_dispute_history_7291",
                                  "arguments": json.dumps({"user_id": candidate.facts["identity"]["user_id"]})})]
    alternative, _ = replay(candidate.task, [*candidate.task.evaluation_criteria.actions, *optional])
    assert hashes(reference) == hashes(alternative)
    closure = compile_candidate(slot("replacement_closure", "close", "20-29",
                                     runtime_revision="native_card_lifecycle_v2"))
    assert "get_user_dispute_history_7291" in {c["name"] for c in reference_operations(closure)}


def test_admission_evidence_contains_actual_public_unlock_schema_only():
    from tau3.synthesis.targeted.native.workflow import public_unlock_evidence

    messages = invocation("u", "unlock_discoverable_agent_tool", {"agent_tool_name": "real_tool"},
                          result="Tool unlocked: real_tool\nParameters: required user_id")
    messages += invocation("bad", "unlock_discoverable_agent_tool", {"agent_tool_name": "unknown"},
                           result="Error: unknown tool")
    messages += [{"role": "user", "content": "private customer facts"},
                 {"role": "tool", "id": "unrelated", "content": "private tool result"}]
    assert public_unlock_evidence({"simulation": {"messages": messages}}) == [
        {"tool_name": "real_tool", "public_reply": "Tool unlocked: real_tool\nParameters: required user_id"}]


def test_independent_unlock_probe_covers_omitted_blind_tool_without_business_writes():
    from tau3.synthesis.targeted.native.workflow import public_unlock_probe

    candidate = compile_candidate(slot("transaction_disputes", "debit", "10-14"))
    before = candidate.model_dump(mode="json")
    evidence = public_unlock_probe(candidate)
    assert evidence["business_state_unchanged"]
    expected = {a.arguments["agent_tool_name"] for a in candidate.task.evaluation_criteria.actions
                if a.name == "unlock_discoverable_agent_tool"}
    assert {t["tool_name"] for t in evidence["tools"]} == expected
    assert expected and all(t["document_ids"] and t["public_reply"].startswith("Tool unlocked:")
                            for t in evidence["tools"])
    assert candidate.model_dump(mode="json") == before
    candidate.task.required_documents = []
    with pytest.raises(Exception, match="No public discovery document"):
        public_unlock_probe(candidate)


def test_unlock_probe_rejects_documented_but_unavailable_tool(monkeypatch):
    from types import SimpleNamespace

    from tau3.synthesis import validation
    from tau3.synthesis.targeted.native.workflow import public_unlock_probe
    from tau3.worldgen.v2.audit import AuditIncomplete

    candidate = compile_candidate(slot("transaction_disputes", "debit", "10-14"))
    env = validation.fresh_environment(candidate.task)
    monkeypatch.setattr(type(env), "get_response", lambda *args: SimpleNamespace(error=True, content="Error: unavailable"))
    monkeypatch.setattr(validation, "fresh_environment", lambda task: env)
    with pytest.raises(AuditIncomplete, match="cannot be unlocked"):
        public_unlock_probe(candidate)


def test_admission_includes_captured_public_policy_and_regular_tools_only():
    from tau3.synthesis.targeted.native.workflow import public_agent_context
    from tau3.worldgen.v2.audit import AuditIncomplete

    schema = {"type": "function", "function": {"name": "change_user_email", "parameters": {}}}
    capture = {"sample": {"tools": [schema], "messages": [
        {"role": "system", "content": "Verify identity before email changes."},
        {"role": "user", "content": "private conversation"},
        {"role": "assistant", "content": "not policy", "reasoning_content": "private reasoning"},
    ]}, "simulation": {"user_policy": "private simulator instructions"}}
    public = public_agent_context(capture)
    assert public == {"tools": [schema], "system_messages": [capture["sample"]["messages"][0]]}
    public["tools"][0]["function"]["name"] = "changed"
    assert capture["sample"]["tools"][0]["function"]["name"] == "change_user_email"
    with pytest.raises(AuditIncomplete, match="public agent policy"):
        public_agent_context({"sample": {"tools": [schema], "messages": []}})


@pytest.mark.parametrize("change", ["none", "runtime", "checker", "static", "fresh", "outcome", "tamper"])
def test_admission_reuse_requires_unchanged_bound_certificate(tmp_path, monkeypatch, change):
    from copy import deepcopy

    from tau3.synthesis.targeted.native import reuse

    old = compile_candidate(slot("credit_limit", "overlimit", "1-4"))
    decision = dict(valid=True, publicly_solvable=True, goals_correct=True, no_leak=True, explanation="bound public proof")
    model = "teacher"
    source_capture = {"identity": {"round": "parent"}, "sample": {"messages": ["actual public context"]}}
    current_capture = {**deepcopy(source_capture), "identity": {"round": "repair"}}
    source_result = {"capture_hash": digest(source_capture), "status": "COMPLETE",
                     "environment_success": False, "obligation_failures": ["original teacher failure"]}
    result = {**source_result, "capture_hash": digest(current_capture)}
    old.checks = {"static": {"positive": True, "reference_db_hashes": ["agent", "user"]}, "blind": {model: {"result_hash": digest(source_result)}},
                  "independent_review": {model: decision}}
    candidate = old.model_copy(deep=True)
    candidate.checks["static"]["reference_db_hashes"] = ("agent", "user")
    parent, root = tmp_path / "parent", tmp_path / "repair"
    work = root / "tasks/pilot/00000/candidates/0"
    directory = work / "blind" / digest(model)[:12]
    source = parent / directory.relative_to(root)
    write_json(source / "result.json", source_result)
    write_json(directory / "capture.json", current_capture)
    origin = {"source": str(source), "strict_replay": True,
              "source_capture_hash": digest(source_capture), "source_result_hash": digest(source_result),
              "payload_hash": digest({k: v for k, v in source_capture.items() if k != "identity"})}
    if change != "fresh":
        write_json(directory / "adopted-capture.json", origin)
    hashes = {"runtime/domains/banking_knowledge/tools.py": "same",
              "runtime/synthesis/targeted/native/workflow.py": "previous_review_input"}
    write_json(parent / "snapshot/manifest.json", {"hashes": hashes})
    hashes["runtime/synthesis/targeted/native/workflow.py"] = "complete_public_review_input"
    if change == "runtime":
        hashes["runtime/domains/banking_knowledge/tools.py"] = "different"
    if change == "checker":
        hashes["runtime/synthesis/targeted/native/business_checks.py"] = "new_independent_policy_check"
    write_json(root / "snapshot/manifest.json", {"hashes": hashes})
    if change == "static":
        candidate.checks["static"]["new_obligation"] = True
    if change == "outcome":
        result["environment_success"] = True
    if change == "tamper":
        current_capture["sample"]["messages"] = ["substituted context"]
        write_json(directory / "capture.json", current_capture)
        result["capture_hash"] = digest(current_capture)
    monkeypatch.setattr(reuse, "parent_candidate", lambda *args: (parent, old))
    if change == "tamper":
        with pytest.raises(ValueError, match="evidence changed"):
            reuse.reuse_admission_review(root, NativeConfig(), candidate, work, model, result)
    else:
        actual = reuse.reuse_admission_review(root, NativeConfig(), candidate, work, model, result)
        assert actual == (decision if change == "none" else None)
        if actual:
            assert candidate.checks["admission_review_origins"][model]["decision_hash"] == digest(decision)


def test_capture_reuse_requires_same_business_problem():
    from tau3.synthesis.targeted.native.reuse import same_business_task

    old = compile_candidate(slot("credit_limit", "overlimit", "1-4", runtime_revision="native_cli_approval_v1"))
    new = compile_candidate(old.slot.model_copy(update={"runtime_revision": "native_card_lifecycle_v2"}))
    assert same_business_task(old, new)
    assert old.task.initial_state.initialization_data.agent_data["task_config"]["data"]["native_runtime"]["revision"] == "native_cli_approval_v1"
    new.facts["goal"] += " Change the target."
    assert not same_business_task(old, new)


@pytest.mark.parametrize("entry", ["incident", "boundary"])
def test_bureau_incident_uses_system_error_and_rejects_self_consistent_wrong_gold(entry):
    from tau3.synthesis.targeted.native.business_checks import check_initial_consistency
    from tau3.synthesis.targeted.native.scenarios import Builder, boundary
    from tau3.synthesis.targeted.native.v03_supplements import incident

    b = Builder(slot("escalation_boundary", "emergency", "1-4"), 0, 2)
    (incident if entry == "incident" else boundary)(b, 1, 2)
    candidate = b.finish()
    assert next(g for g in candidate.goals if g["kind"] == "transfer")["reason"] == "technical_system_error"
    assert validate_static(candidate)["positive"]
    # Corrupt the reference AND goal together: replay agreement must not hide
    # a reference answer that contradicts the publicly documented cause.
    for goal in candidate.goals:
        if goal["kind"] == "transfer":
            goal["reason"] = "fraud_or_security_concern"
    for action in candidate.task.evaluation_criteria.actions:
        if action.name == "transfer_to_human_agents":
            action.arguments["reason"] = "fraud_or_security_concern"
    with pytest.raises(ValueError, match="public system-error transfer reason"):
        check_initial_consistency(candidate)


@pytest.mark.parametrize("split,enabled", [("pilot", True), ("train", False), ("train", True), ("validation", True)])
def test_parent_reuse_requires_explicit_split_and_matching_business(tmp_path, split, enabled):
    from tau3.synthesis.targeted.native.reuse import parent_candidate

    candidate = compile_candidate(slot("credit_limit", "overlimit", "1-4"))
    candidate.slot.split = split
    parent, root = tmp_path / "parent", tmp_path / "repair"
    config = NativeConfig(parent_round=str(parent), reuse_parent_pilot=True,
                          reuse_parent_splits=[split] if enabled and split != "pilot" else [])
    raw = candidate.model_dump(mode="json")
    directory = parent / "tasks" / split / "00000"
    write_json(directory / "candidate.json", raw)
    write_json(directory / "provenance.json", {"status": "VALID", "candidate_hash": digest(raw)})
    write_json(root / "budget-inheritance.json", {"parent": str(parent)})
    write_json(parent / "config.json", config.model_dump(mode="json"))
    snapshot = dict(public_tool_schema_hash="same", policy_hash="same", retrieval="same", hashes={})
    for path in [parent, root]:
        write_json(path / "snapshot/manifest.json", snapshot)
    found = parent_candidate(root, config, candidate)
    assert (found is not None) is enabled
    candidate.goals.append({"kind": "forbidden", "tools": ["another_operation"]})
    assert parent_candidate(root, config, candidate) is None


def test_explicit_split_reuse_requires_parent_round():
    with pytest.raises(ValueError, match="declared parent round"):
        NativeConfig(reuse_parent_splits=["train"])


@pytest.mark.parametrize("tamper", [False, True])
def test_reuse_crosses_stopped_budget_bound_repair_without_formal_tasks(tmp_path, tamper):
    from tau3.synthesis.targeted.native.reuse import parent_candidate

    candidate = compile_candidate(slot("credit_limit", "overlimit", "1-4", split="train"))
    root, parent, ancestor = [tmp_path / p for p in ("new", "pilot_only", "formal")]
    config = NativeConfig(parent_round=str(parent), reuse_parent_splits=["train"])
    write_json(root / "budget-inheritance.json", {"parent": str(parent)})
    binding = {"parent": str(ancestor), "identity": {"id": "frozen"},
               "budget": {"calls": 25}, "blind_budget": {"rollouts": 2}}
    for name, filename in (("identity", "identity.json"), ("budget", "budget.json"), ("blind_budget", "blind-budget.json")):
        write_json(ancestor / filename, binding[name])
    write_json(parent / "budget-inheritance.json", {**binding, "hash": digest(binding)})
    write_json(ancestor / "supervisor.json", {"status": "INCOMPLETE"})
    raw = candidate.model_dump(mode="json")
    write_json(ancestor / "tasks/train/00000/candidate.json", raw)
    write_json(ancestor / "tasks/train/00000/provenance.json", {"status": "VALID", "candidate_hash": digest(raw)})
    write_json(ancestor / "config.json", config.model_dump(mode="json"))
    snapshot = dict(public_tool_schema_hash="same", policy_hash="same", retrieval="same", hashes={})
    for path in (root, ancestor):
        write_json(path / "snapshot/manifest.json", snapshot)
    if tamper:
        write_json(ancestor / "budget.json", {"calls": 26})
        with pytest.raises(ValueError, match="ancestor identity or budget changed"):
            parent_candidate(root, config, candidate)
    else:
        found = parent_candidate(root, config, candidate)
        assert found is not None and found[0] == ancestor


def test_admission_user_context_contains_only_initially_available_user_tools():
    from tau3.synthesis.targeted.native.workflow import public_user_tools

    candidate = compile_candidate(slot("handoff", "purchase", "1-4"))
    context = public_user_tools(candidate)
    assert context["visibility"] == "user_only"
    assert {tool["function"]["name"] for tool in context["tools"]} == {
        "submit_transaction", "call_discoverable_user_tool"
    }
    assert set(context) == {"visibility", "tools"}
    assert "planned_purchase" not in json.dumps(context)
    # A discoverable handoff starts with the wrapper, not an implicitly granted
    # hidden operation. Its real grant remains mandatory.
    candidate = compile_candidate(slot("handoff", None, "5-9"))
    context = public_user_tools(candidate)
    names = {tool["function"]["name"] for tool in context["tools"]}
    assert "call_discoverable_user_tool" in names
    assert "get_one_time_user_bypass_code" not in names


def test_opening_requires_own_account_evidence_before_write():
    from tau3.synthesis.targeted.native.validation import (
        check_goals,
        operations,
        reference_operations,
    )
    from tau3.synthesis.validation import hashes, replay

    candidate = compile_candidate(slot("accounts_funds", "open", "5-9"))
    checks = validate_static(candidate)
    assert checks["counterexamples"]["late_eligibility_check"]
    assert checks["counterexamples"]["wrong_customer_eligibility_check"]
    actions = candidate.task.evaluation_criteria.actions
    index = next(i for i, c in enumerate(reference_operations(candidate))
                 if c["name"] == "get_all_user_accounts_by_user_id_3847")
    expected, _ = replay(candidate.task)
    late, messages = replay(candidate.task, [*actions[:index], *actions[index + 1:], actions[index]])
    # A final-state-only grader cannot detect this ordering violation.
    assert hashes(late) == hashes(expected)
    assert check_goals(candidate, late, operations(messages))
    legal, messages = replay(candidate.task, [*actions[:index + 1], actions[index], *actions[index + 1:]])
    assert hashes(legal) == hashes(expected)
    assert not check_goals(candidate, legal, operations(messages))
    with pytest.raises(ValueError, match="No supported graph"):
        compile_candidate(slot("accounts_funds", "open", "1-4"))


def test_short_native_account_service_does_not_skip_opening_eligibility():
    from tau3.synthesis.targeted.native.scenarios import Builder, v03_holdout_groups
    from tau3.synthesis.targeted.native.v03_scenarios import install_generators
    from tau3.synthesis.targeted.native.validation import reference_operations

    candidate = compile_candidate(slot("accounts_funds", None, "1-4"))
    assert {c["name"] for c in reference_operations(candidate)} == {"log_verification", "change_user_email"}
    assert validate_static(candidate)["positive"]
    for count in range(1, 9):
        builder = Builder(candidate.slot, 0, 0)
        install_generators()["accounts_funds"](builder, count, 0)
        assert len(builder.actions.items) == 2
    assert not v03_holdout_groups("accounts_funds", None, "1-4", False, None)
    with pytest.raises(ValueError, match="No supported graph"):
        compile_candidate(slot("accounts_funds", None, "1-4", split="validation"))
    # The new curriculum does not rewrite the legacy compiler's contract.
    legacy = compile_candidate(slot("accounts_funds", None, "1-4", v03_labels=[]))
    assert "open_bank_account_4821" in {c["name"] for c in reference_operations(legacy)}


@pytest.mark.parametrize("split", ["pilot", "validation"])
def test_short_optimization_cannot_hide_an_unchecked_account_opening(split):
    candidate = compile_candidate(slot("optimization", None, "1-4", split=split))
    assert not any(g["kind"] == "opened_account" for g in candidate.goals)
    assert validate_static(candidate)["positive"]


@pytest.mark.parametrize("runtime_changed,corrupt", [(False, False), (True, False), (False, True)])
def test_replacement_capture_reuse_binds_runtime_and_actual_replies(tmp_path, monkeypatch, runtime_changed, corrupt):
    from tau3.data_model.simulation import SimulationRun
    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native import reuse
    from tau3.synthesis.validation import replay

    candidate = compile_candidate(slot("replacement_closure", "replace", "5-9", runtime_revision="native_card_lifecycle_v2"))
    _, messages = replay(candidate.task)
    if corrupt:
        messages[-1].content = "An unrelated successful reply"
    simulation = SimulationRun(id="original", task_id=candidate.task.id,
        start_time="2026-09-19T00:00:00", end_time="2026-09-19T00:00:01", duration=1,
        termination_reason="user_stop", messages=messages)
    parent, root = tmp_path / "parent", tmp_path / "repair"
    directory = root / "tasks/pilot/00000/trials/0"
    source = parent / directory.relative_to(root)
    old_identity = {"model": "teacher", "seed": 1, "blind": False, "round": "parent"}
    identity = {**old_identity, "round": "repair"}
    capture = {"identity": old_identity, "simulation": simulation.model_dump(mode="json")}
    write_json(source / "capture.json", capture)
    write_json(source / "result.json", {"capture_hash": digest(capture)})
    write_json(source / "started.json", old_identity)
    key = "runtime/domains/banking_knowledge/tools.py"
    write_json(parent / "snapshot/manifest.json", {"hashes": {key: "original"}})
    write_json(root / "snapshot/manifest.json", {"hashes": {key: "changed" if runtime_changed else "original"}})
    monkeypatch.setattr(reuse, "parent_candidate", lambda *args: (parent, candidate))
    if corrupt:
        with pytest.raises(ValueError, match="Tool call:"):
            reuse.reuse_capture(root, NativeConfig(), candidate, directory, identity)
        assert not (directory / "adopted-capture.json").exists()
    else:
        assert reuse.reuse_capture(root, NativeConfig(), candidate, directory, identity) is not runtime_changed
        if not runtime_changed:
            imported = read_json(directory / "capture.json")
            assert imported["simulation"] == capture["simulation"]
            assert imported["identity"] == identity


@pytest.mark.parametrize("branch", ["approve", "deny:insufficient_account_age", "deny:cooldown_period_active"])
def test_credit_chronology_for_every_supported_tier_variant(branch):
    from datetime import datetime

    from tau3.domains.banking_knowledge.utils import get_today
    from tau3.synthesis.targeted.native.business_checks import check_initial_consistency
    from tau3.synthesis.targeted.native.scenarios import Builder
    from tau3.synthesis.targeted.native.v03_scenarios import credit_limit

    checked = 0
    for variant in range(48):
        b = Builder(slot("credit_limit", branch, runtime_revision="native_cli_approval_v1"), 0, variant)
        try:
            credit_limit(b, 1, variant)
        except ValueError:
            assert branch == "deny:insufficient_account_age" and variant % 3 != 1
            continue
        candidate = b.finish()
        check_initial_consistency(candidate)
        data = candidate.task.initial_state.initialization_data.agent_data
        payments = list(data["payment_history"]["data"].values())
        months = sorted({p["payment_date"][:7] for p in payments})
        assert len(months) == len(payments)
        if branch == "deny:insufficient_account_age":
            card = data["credit_card_accounts"]["data"][payments[0]["credit_card_account_id"]]
            assert (get_today() - datetime.strptime(card["date_of_account_open"], "%m/%d/%Y").date()).days == 89
            assert len(months) == 3
        checked += 1
    assert checked


@pytest.mark.parametrize("invalid_date", ["2000-01-01", "2099-01-01"])
def test_static_admission_rejects_impossible_payment_dates(invalid_date):
    candidate = compile_candidate(slot("credit_limit", "deny:high_utilization"))
    rows = candidate.task.initial_state.initialization_data.agent_data["payment_history"]["data"]
    next(iter(rows.values()))["payment_date"] = invalid_date
    with pytest.raises(ValueError, match="between account opening"):
        validate_static(candidate)


def test_debit_dispute_facts_are_available_and_liability_is_grounded():
    from tau3.synthesis.targeted.native.business_checks import check_initial_consistency

    candidate = compile_candidate(slot("transaction_disputes", "debit"))
    check_initial_consistency(candidate)
    known = json.loads(candidate.task.user_scenario.instructions.known_info)
    assert "online" in known["purchase_channel"]
    goal = next(g for g in candidate.goals if g["kind"] == "row_match")
    assert goal["expected"]["customer_max_liability_amount"] == min(50, goal["expected"]["disputed_amount"])
    bad = candidate.model_copy(deep=True)
    next(g for g in bad.goals if g["kind"] == "row_match")["expected"]["customer_max_liability_amount"] = 0
    with pytest.raises(ValueError, match="liability"):
        check_initial_consistency(bad)
    del known["purchase_channel"]
    candidate.task.user_scenario.instructions.known_info = json.dumps(known)
    with pytest.raises(ValueError, match="not available to the user"):
        check_initial_consistency(candidate)


def test_repair_budget_is_cumulative_and_resume_cannot_reset_it(tmp_path):
    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.revision import inherit_budget
    from tau3.synthesis.targeted.native.workflow import budget_forecast

    parent, child = tmp_path / "parent", tmp_path / "child"
    prior = {"calls": 100, "audit_calls": 40, "tokens": 1000, "rollouts": 76}
    write_json(parent / "budget.json", prior)
    write_json(parent / "blind-budget.json", {"rollouts": 40})
    write_json(parent / "identity.json", {"snapshot_hash": "prior"})
    write_json(parent / "supervisor.json", {"status": "INCOMPLETE"})
    config = NativeConfig(parent_round=str(parent), budget_parent_round=str(parent), max_rollouts=12156)
    inherit_budget(child, config)
    current = {"calls": 120, "audit_calls": 50, "tokens": 1300, "rollouts": 156}
    write_json(child / "budget.json", current)
    inherit_budget(child, config)
    assert read_json(child / "budget.json") == current
    forecast = budget_forecast(child, config)
    scale = (3000 + 200 + 20) / 20 * 1.3
    assert forecast["estimated"]["calls"] == 100 + int(20 * scale + 1)
    assert forecast["checks"]["teacher_rollouts"]
    write_json(child / "budget.json", {**current, "calls": 0})
    with pytest.raises(ValueError, match="lost inherited"):
        inherit_budget(child, config)
    write_json(child / "budget.json", current)
    write_json(parent / "budget.json", {**prior, "calls": 101})
    with pytest.raises(ValueError, match="evidence changed"):
        inherit_budget(child, config)


@pytest.mark.parametrize("parent_reused", [False, True])
def test_reused_pilot_does_not_understate_production_budget(tmp_path, parent_reused):
    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.revision import inherit_budget
    from tau3.synthesis.targeted.native.workflow import budget_forecast

    parent, child = tmp_path / "parent", tmp_path / "child"
    prior = {"calls": 100, "audit_calls": 40, "tokens": 1000, "rollouts": 148}
    earlier = {"calls": 60, "audit_calls": 20, "tokens": 600, "rollouts": 76}
    write_json(parent / "budget.json", prior)
    carried = {"calls": 30, "audit_calls": 10, "tokens": 300} if parent_reused else {}
    write_json(parent / "budget-inheritance.json", {"budget": earlier, "reused_pilot_cost": carried})
    measurement = {"budget": prior, "budget_hash": digest(prior)}
    write_json(parent / "pilot/budget-measurement.json", measurement)
    write_json(parent / "blind-budget.json", {"rollouts": 78})
    write_json(parent / "identity.json", {"snapshot_hash": "prior"})
    write_json(parent / "supervisor.json", {"status": "INCOMPLETE"})
    config = NativeConfig(parent_round=str(parent), budget_parent_round=str(parent),
                          reuse_parent_pilot=True, max_rollouts=12228)
    inherit_budget(child, config)
    current = {"calls": 110, "audit_calls": 50, "tokens": 1100, "rollouts": 148}
    write_json(child / "budget.json", current)
    inherit_budget(child, config)
    assert read_json(child / "budget.json") == current
    forecast = budget_forecast(child, config)
    assert forecast["reused_pilot_cost"] == {
        key: value + carried.get(key, 0) for key, value in {"calls": 40, "audit_calls": 20, "tokens": 400}.items()}
    scale = (3000 + 200 + 20) / 20 * 1.3
    assert forecast["estimated"]["calls"] == 100 + int((50 + carried.get("calls", 0)) * scale + 1)
    assert forecast["checks"]["teacher_rollouts"]
    # An internally consistent rewrite is still different parent evidence.
    changed = {**prior, "calls": 101}
    write_json(parent / "pilot/budget-measurement.json", {"budget": changed, "budget_hash": digest(changed)})
    with pytest.raises(ValueError, match="evidence changed"):
        inherit_budget(child, config)


@pytest.mark.parametrize("branch", ["approve", "mixed"])
def test_native_approval_revision_passes_independent_validation(branch):
    candidate = compile_candidate(slot("credit_limit", branch,
        difficulty="20-29" if branch == "mixed" else "10-14",
        runtime_revision="native_cli_approval_v1"))
    assert validate_static(candidate)["positive"]


@pytest.mark.parametrize("revision", ["official", "native_cli_approval_v1"])
def test_approval_revision_preserves_submission_and_official_default(revision):
    from tau3.synthesis.targeted.planning import unpack
    from tau3.synthesis.validation import fresh_environment

    candidate = compile_candidate(slot("credit_limit", "approve", runtime_revision=revision))
    env = fresh_environment(candidate.task)
    args = next(args for action in candidate.task.evaluation_criteria.actions
                for name, args in [unpack(action.name, action.arguments)]
                if name == "submit_credit_limit_increase_request_7392")
    assert not env.tools.submit_credit_limit_increase_request_7392(**args).startswith("Error:")
    records = env.tools.db.credit_limit_increase_requests.data
    request_id = next(key for key, row in records.items() if row.get("user_id") == args["user_id"])
    before = dict(records[request_id])
    card = env.tools.db.credit_card_accounts.data[args["credit_card_account_id"]]
    new_limit = int(float(card["credit_limit"].replace("$", "").replace(",", ""))) + args["requested_increase_amount"]
    response = env.tools.approve_credit_limit_increase_5847(
        args["credit_card_account_id"], args["user_id"], new_limit)
    assert not response.startswith("Error:")
    assert card["credit_limit"] == f"${new_limit:.2f}"
    if revision == "official":
        assert records[request_id] == before
    else:
        assert records[request_id]["status"] == "APPROVED"
        assert records[request_id]["submitted_at"] == before["submitted_at"]
        assert records[request_id]["requested_increase_amount"] == before["requested_increase_amount"]


@pytest.mark.parametrize("defect", ["missing", "amount", "owner", "account", "status", "ineligible"])
def test_native_approval_rejects_invalid_request_without_partial_write(defect):
    from copy import deepcopy

    from tau3.synthesis.targeted.planning import unpack
    from tau3.synthesis.validation import fresh_environment

    candidate = compile_candidate(slot("credit_limit", "approve", runtime_revision="native_cli_approval_v1"))
    env = fresh_environment(candidate.task)
    args = next(args for action in candidate.task.evaluation_criteria.actions
                for name, args in [unpack(action.name, action.arguments)]
                if name == "submit_credit_limit_increase_request_7392")
    db = env.tools.db
    card = db.credit_card_accounts.data[args["credit_card_account_id"]]
    new_limit = int(float(card["credit_limit"].replace("$", "").replace(",", ""))) + args["requested_increase_amount"]
    if defect != "missing":
        env.tools.submit_credit_limit_increase_request_7392(**args)
        pending = next(row for row in db.credit_limit_increase_requests.data.values()
                       if row.get("user_id") == args["user_id"])
        mutations = {"amount": ("requested_increase_amount", -1), "owner": ("user_id", "someone_else"),
                     "account": ("credit_card_account_id", "another_account"), "status": ("status", "APPROVED")}
        if defect in mutations:
            key, value = mutations[defect]
            pending[key] = value
        if defect == "ineligible":
            card["account_status"] = "PAST_DUE"
    before = deepcopy(db.model_dump())
    assert env.tools.approve_credit_limit_increase_5847(
        args["credit_card_account_id"], args["user_id"], new_limit).startswith("Error:")
    assert db.model_dump() == before


@pytest.mark.parametrize("family,branch", [
    ("accounts_funds", "email"), ("accounts_funds", "checking_fee"), ("accounts_funds", "open"),
    ("debit_security", "activate"), ("debit_security", "clear_block"), ("debit_security", "temporary_limit"),
    ("debit_security", "replacement"), ("debit_security", "unfreeze"),
    ("replacement_closure", "retention_credit"), ("replacement_closure", "retention_waiver"),
    ("handoff", "referral_link"), ("handoff", "purchase"),
    ("escalation_boundary", "payment_incident"), ("escalation_boundary", "decline_incident"),
    ("escalation_boundary", "emergency"), ("transaction_disputes", "rewards"),
])
def test_minority_operations_have_executable_business_obligations(family, branch):
    for difficulty in ("1-4", "5-9", "10-14", "15-19", "20-29"):
        try:
            candidate = compile_candidate(slot(family, branch, difficulty))
        except ValueError:
            continue
        checks = validate_static(candidate)
        assert checks["positive"] and checks["expected_db_diff"] or family == "escalation_boundary"
        return
    pytest.fail(f"No executable curriculum for {family}/{branch}")


def test_full_credit_quotas_and_legacy_defaults():
    recipes = recipe_counts(3000)
    assert recipes["credit_limit/approve"] == 250
    assert sum(n for key, n in recipes.items() if key.startswith("credit_limit/deny:")) == 300
    assert recipes["credit_limit/overlimit"] + recipes["credit_limit/missing_amount"] == 100
    assert recipes["credit_limit/mixed"] == 100
    assert NativeConfig().curriculum == "native_v2"
    counts = Counter()
    for key, n in recipe_counts(400).items():
        counts[key.split("/")[0]] += n
    assert counts == dict(credit_limit=100, transaction_disputes=80, replacement_closure=60,
                          accounts_funds=48, debit_security=40, optimization=32, handoff=20, escalation_boundary=20)


def test_credit_pair_changes_only_one_source_condition():
    from tau3.synthesis.targeted.native.coverage import pair_report

    left = slot("credit_limit", "approve", pair_id="unit-pair", pair_side="control")
    right = slot("credit_limit", "deny:high_utilization", index=1, pair_id="unit-pair", pair_side="boundary")
    candidates = [compile_candidate(s) for s in (left, right)]
    result = pair_report(candidates, [left, right])
    assert result["status"] == "PASS"
    assert result["pairs"]["unit-pair"]["changed_fields"][0][-1] == "current_balance"


def test_v03_gate_does_not_silently_relax_format_success_ceiling():
    from tau3.synthesis.targeted.native.gate import v03_diagnostic_checks
    from tau3.synthesis.targeted.native.models import GatePolicy

    baseline = {f"task_{i:03d}": {str(seed): 0 for seed in range(5)} for i in range(50, 55)}
    candidate = {t: {str(seed): int(seed < 3) for seed in range(5)} for t in baseline}
    left = {"unknown_tool_events": 9, "unknown_tool_task_rate": 0.1, "protocol_error_task_rate": 0.5,
            "first_discoverable_format_success": 0.96, "wrong_entity_write_task_rate": 0.1}
    right = {**left, "protocol_error_task_rate": 0.3, "first_discoverable_format_success": 1.0}
    checks = v03_diagnostic_checks(left, right, baseline, candidate, GatePolicy(v03_checks=True))
    assert checks["unknown_tool_reduction"]
    assert checks["protocol_error_reduction"] and checks["task_051_floor"]
    assert not checks["first_discoverable_format_gain"]


def test_successful_recovery_stays_out_of_clean_partition(tmp_path, monkeypatch):
    from tau3.synthesis.targeted.native import quality

    candidate = compile_candidate(slot("credit_limit", "overlimit", "1-4"))
    messages = invocation("bad", "made_up_tool", {}, result="Error: Unknown tool")
    messages.append({"role": "assistant", "content": "I have left your request unsubmitted."})
    sample_messages = [{"role": "user", "content": "Please check my request."}]
    for message in messages:
        m = dict(message)
        if m["role"] == "tool":
            m = {"role": "tool", "tool_call_id": m["id"], "content": m["content"]}
        else:
            m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                               for c in m.get("tool_calls", [])]
        sample_messages.append(m)
    sample = {"schema_version": 2, "messages": sample_messages, "tools": [], "seed": 1}
    simulation = {"messages": messages}
    result = {"status": "COMPLETE", "environment_success": True, "simulation": simulation}
    provenance = {"task_id": candidate.task.id}
    write_json(tmp_path / "result.json", result)
    write_json(tmp_path / "capture.json", {"sample": sample, "simulation": simulation})
    review = {key: True for key in ("reasoning_correct", "grounded_in_visible_history", "user_compliant", "handoff_correct", "final_answer_correct", "no_private_leakage")}
    review.update(erroneous_assistant_turns=[], unlocalizable_error=False, explanation="localized tool error")
    write_json(tmp_path / "quality.json", {"identity": digest([result, sample, provenance]), "review": review})
    monkeypatch.setattr(quality, "ask", lambda *a, **k: pytest.fail("cached qualification must not call LLM"))
    row = quality.qualify(tmp_path, NativeConfig(clean_only=True), candidate, tmp_path, provenance)
    assert row["metadata"]["dataset_partition"] == "recovery"
    assert row["metadata"]["masked_assistant_turns"] == [0]
    assert row["loss_mask"] == [0, 0, 0, 1]
