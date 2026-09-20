"""Native curricula must prove real business outcomes without live model calls."""

from collections import Counter

import pytest

from tau3.synthesis.targeted.models import Finding
from tau3.synthesis.targeted.native.models import (
    FAMILIES,
    NativeConfig,
    NativeProfile,
    NativeSlot,
)
from tau3.synthesis.targeted.native.planning import compile_plan
from tau3.synthesis.targeted.native.scenarios import compile_candidate
from tau3.synthesis.targeted.native.validation import validate_static
from tau3.synthesis.targeted.planning import operation_kind, unpack
from tau3.synthesis.world_sft import validate_sample


def slot(family, difficulty="5-9", split="pilot"):
    return NativeSlot(index=0, split=split, family=family, difficulty=difficulty,
                      origin="current", seed=43000, trial_seeds=[1, 2, 3, 4], evidence_ids=["finding:0"])


def test_real_user_selector_and_write_metadata():
    assert unpack("call_discoverable_user_tool", {"discoverable_tool_name": "deposit_check_3847", "arguments": '{"check_amount":25}'}) == ("deposit_check_3847", {"check_amount": 25})
    assert operation_kind("call_discoverable_agent_tool") == "write"
    assert operation_kind("get_bank_account_transactions_9173") == "read_or_generic"
    assert operation_kind("give_discoverable_user_tool") == "protocol"
    assert operation_kind("deposit_check_3847", "user") == "write"


def test_selective_mask_preserves_context_and_v1_contract():
    sample = {"schema_version": 2, "messages": [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "bad"}, {"role": "assistant", "content": "correct"}], "loss_mask": [0, 0, 1]}
    validate_sample(sample)
    with pytest.raises(ValueError):
        validate_sample({**sample, "schema_version": 1})
    with pytest.raises(ValueError):
        validate_sample({**sample, "loss_mask": [1, 0, 1]})


def test_frozen_prefix_and_disjoint_validation_groups():
    profile = NativeProfile(round_id="test", base_model="base", report_hash="report", findings=[
        Finding(label=label, observation="x", interpretation="x", source="report", quote="x", confidence="high", severity="high")
        for label in ["F1", "F2", "F3", "F3b", "F4", "F5", "F6", "F7", "F8"]])
    plan = compile_plan(NativeConfig(), profile, "snapshot")
    assert len(plan.slots) == 3000 and len(plan.pilot) == 20 and len(plan.validation) == 200
    assert Counter(s.family for s in plan.slots[:400]) == {k: v * 4 for k, v in FAMILIES.items()}
    assert Counter(s.family for s in plan.slots) == {k: v * 30 for k, v in FAMILIES.items()}
    assert Counter(s.difficulty for s in plan.slots[:400]) == {"1-4": 60, "5-9": 100, "10-14": 100, "15-19": 80, "20-29": 60}
    assert sum(s.difficulty in {"15-19", "20-29"} for s in plan.pilot) >= 6
    train_groups = {compile_candidate(s).group_id for s in plan.pilot}
    validation_groups = {compile_candidate(s).group_id for s in plan.validation[:20]}
    assert not train_groups & validation_groups


@pytest.mark.parametrize("family,difficulty", [
    ("disputes", "10-14"), ("debit", "20-29"), ("accounts", "20-29"),
    ("optimization", "1-4"), ("identity", "10-14"), ("handoff", "15-19"), ("boundary", "1-4"),
])
def test_native_positive_and_negative_workflows(family, difficulty):
    candidate = compile_candidate(slot(family, difficulty))
    checks = validate_static(candidate)
    assert checks["positive"] and checks["alternative_query"]
    assert checks["counterexamples"]["empty"]
    assert checks["counterexamples"]["extra_write"]


def test_seeds_are_distinct_and_not_replaced():
    with pytest.raises(ValueError):
        slot("accounts").model_validate({**slot("accounts").model_dump(), "trial_seeds": [1, 1, 2, 3]})


def test_blind_budget_failure_does_not_replace_a_valid_task():
    from tau3.synthesis.targeted.native.workflow import blind_attempt_evidence
    from tau3.worldgen.v2.audit import AuditIncomplete

    result = {"status": "INCONCLUSIVE", "environment_success": False,
              "simulation": {"termination_reason": "max_steps", "messages": [{"role": "assistant", "content": "Searching"}]}}
    evidence = blind_attempt_evidence(result)
    assert evidence["status"] == "INCONCLUSIVE" and evidence["success"] is False
    with pytest.raises(AuditIncomplete, match="infrastructure"):
        blind_attempt_evidence({**result, "simulation": {"termination_reason": "infrastructure_error", "messages": []}})


def test_response_journal_never_resends_a_completed_request(tmp_path, monkeypatch):
    from litellm import ModelResponse

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.runner import journal
    from tau3.utils import llm_utils

    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                             usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
    monkeypatch.setattr(llm_utils, "completion", transport)
    kwargs = {"model": "openai/GLM-5.3-Flash", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    for _ in range(2):
        with journal(tmp_path, NativeConfig(), tmp_path / "responses", lambda _: "teacher"):
            assert llm_utils.completion(**kwargs).choices[0].message.content == "hello"
    assert len(calls) == 1
    assert read_json(tmp_path / "budget.json")["calls"] == 1
    assert read_json(tmp_path / "budget.json")["tokens"] == 7


@pytest.mark.parametrize("family", ["accounts", "handoff"])
def test_captured_native_episode_grades_without_resampling(tmp_path, monkeypatch, family):
    from tau3.data_model.simulation import SimulationRun, TerminationReason
    from tau3.synthesis.storage import digest, write_json
    from tau3.synthesis.targeted.native.runner import sample
    from tau3.synthesis.validation import replay
    from tau3.utils import llm_utils

    candidate = compile_candidate(slot(family))
    _, messages = replay(candidate.task)
    config = NativeConfig()
    binding = {"snapshot": "test"}
    identity = {"task_hash": digest(candidate.task.model_dump(mode="json")),
                "model": config.teacher_model, "seed": 42, "binding": binding,
                "blind": True, "config_hash": digest(config.model_dump(mode="json"))}
    simulation = SimulationRun(id="captured", task_id=candidate.task.id,
        start_time="2026-09-18", end_time="2026-09-18", duration=1,
        termination_reason=TerminationReason.USER_STOP, messages=messages)
    directory = tmp_path / "trial"
    write_json(directory / "started.json", identity)
    write_json(directory / "capture.json", {"identity": identity,
        "simulation": simulation.model_dump(mode="json"), "sample": {}, "elapsed_seconds": 1})

    def forbid(*args, **kwargs):
        pytest.fail("A complete capture must grade without a new model request")

    monkeypatch.setattr(llm_utils, "completion", forbid)
    result = sample(tmp_path, config, candidate, directory, config.teacher_model,
                    42, binding, blind=True)
    assert result["status"] == "COMPLETE"
    assert result["environment_success"] is True
    assert result["obligation_failures"] == []
    assert sample(tmp_path, config, candidate, directory, config.teacher_model,
                  42, binding, blind=True) == result
    assert not (tmp_path / "budget.json").exists()


def test_gate_bootstraps_tasks_and_requires_complete_paired_seeds():
    from tau3.synthesis.targeted.native.gate import paired_statistics
    from tau3.synthesis.targeted.native.models import GatePolicy

    seeds = {str(i): 0 for i in range(5)}
    baseline = {str(i): dict(seeds) for i in range(10)}
    candidate = {str(i): dict.fromkeys(seeds, 1) for i in range(10)}
    result = paired_statistics(baseline, candidate, GatePolicy(bootstrap_replicates=100))
    assert result["gain"] == result["ci_low"] == 1
    del candidate["0"]["1"]
    with pytest.raises(ValueError):
        paired_statistics(baseline, candidate, GatePolicy(bootstrap_replicates=100))


def test_real_dots_template_preserves_reasoning_and_error_mask():
    from pathlib import Path

    from tau3.synthesis.targeted.native.training import artifacts, encode_sample

    checkpoint = "/cpfs/user/liujialong/model/tau3.v0.2/iter_0000074_hf_fp8_infra"
    if not Path(checkpoint).exists():
        pytest.skip("User-supplied local tokenizer asset is unavailable")
    sample = {"schema_version": 2, "tools": [{"type": "function", "function": {"name": "call_discoverable_agent_tool",
              "description": "Call a tool", "parameters": {"type": "object", "properties": {"agent_tool_name": {"type": "string"}, "arguments": {"type": "string"}}}}}],
              "messages": [{"role": "system", "content": "Banking assistant"}, {"role": "user", "content": "Please proceed"},
                  {"role": "assistant", "content": "Incorrect attempt", "reasoning_content": "masked reasoning"},
                  {"role": "assistant", "content": "", "reasoning_content": "Use the visible schema.",
                   "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "call_discoverable_agent_tool", "arguments": '{"agent_tool_name":"correct_tool","arguments":"{\\"amount\\":33,\\"enabled\\":true}"}'}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "Success"},
                  {"role": "assistant", "content": "Completed."}], "loss_mask": [0, 0, 0, 1, 0, 1]}
    encoded = encode_sample(sample, checkpoint)
    tokenizer, _, _ = artifacts(checkpoint)
    supervised = tokenizer.decode([token for token in encoded["labels"] if token != -100], skip_special_tokens=False)
    assert "Use the visible schema." in supervised
    assert "masked reasoning" not in supervised
    assert "Incorrect attempt" not in supervised
    assert "Success" not in supervised
    assert encoded["audit"]["tool_roundtrip"]


def test_overlong_success_is_quarantined_without_truncation(tmp_path):
    from pathlib import Path

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.training import check_training_readiness
    from tau3.worldgen.v2.audit import AuditIncomplete

    checkpoint = "/cpfs/user/liujialong/model/tau3.v0.2/iter_0000074_hf_fp8_infra"
    if not Path(checkpoint).exists():
        pytest.skip("User tokenizer unavailable")
    row = {"schema_version": 2, "tools": [], "messages": [
        {"role": "user", "content": "Please proceed."},
        {"role": "assistant", "content": "Completed.", "reasoning_content": "Confirmed the actual successful result."}],
        "loss_mask": [0, 1]}
    contract = {"checkpoint": checkpoint, "max_length": 1}
    assert not check_training_readiness(row, contract, tmp_path / "short")
    assert read_json(tmp_path / "short/training-audit.json")["status"] == "REJECTED_LENGTH"
    assert row["messages"][1]["reasoning_content"] == "Confirmed the actual successful result."
    assert not check_training_readiness(row, contract, tmp_path / "short")
    with pytest.raises(AuditIncomplete, match="changed"):
        check_training_readiness(row, {**contract, "max_length": 131072}, tmp_path / "short")
    assert check_training_readiness(row, {**contract, "max_length": 131072}, tmp_path / "supported")


def test_invalid_selector_has_explicit_signature_error():
    from tau3.data_model.message import ToolCall
    from tau3.synthesis.validation import fresh_environment

    candidate = compile_candidate(slot("accounts"))
    env = fresh_environment(candidate.task)
    response = env.get_response(ToolCall(id="invalid", name="call_discoverable_agent_tool",
        arguments={"tool_name": "open_bank_account_4821", "arguments": "{}"}, requestor="assistant"))
    assert response.error
    assert "Invalid arguments for call_discoverable_agent_tool" in response.content
    assert "multiple values" not in response.content


def test_failed_actions_do_not_satisfy_user_handoff_or_unlock():
    from tau3.synthesis.targeted.native.validation import operations, protocol_errors

    messages = [{"role": "assistant", "tool_calls": [{"id": "1", "name": "unlock_discoverable_agent_tool", "arguments": {"agent_tool_name": "reset_debit_card_pin_6284"}}]},
                {"role": "tool", "id": "1", "error": True, "content": "Error: failed"}]
    calls = operations(messages)
    assert calls[0]["succeeded"] is False
    assert protocol_errors(calls) == []


def test_apy_oracle_rejects_wrong_amount_missing_report_and_duplicate():
    native_slot = slot("optimization", "10-14").model_copy(update={"mechanism": "savings_correction"})
    candidate = compile_candidate(native_slot)
    checks = validate_static(candidate)
    assert checks["positive"]
    assert checks["counterexamples"]["duplicate_credit"]
    assert any("submit_interest_discrepancy_report" in name for name in checks["counterexamples"])
    assert any(name.endswith("_amount") for name in checks["counterexamples"])


def test_valid_tools_can_follow_wrong_security_workflow():
    from tau3.synthesis.targeted.native.scenarios import Builder, debit

    b = Builder(slot("debit", "10-14"), 0, 5)
    debit(b, 1, 5)
    candidate = b.finish()
    candidate.slot.difficulty = "5-9"
    checks = validate_static(candidate)
    assert checks["counterexamples"]["wrong_successful_security_workflow"]


def test_entry_replacement_wait_is_a_policy_obligation_not_a_teacher_failure():
    import json

    from tau3.data_model.tasks import Action
    from tau3.synthesis.targeted.native.scenarios import Builder, debit

    entry = Builder(slot("debit", "1-4"), 0, 3)
    debit(entry, 1, 3)
    candidate = entry.finish()
    assert any(g["kind"] == "replacement_wait" for g in candidate.goals)
    checks = validate_static(candidate)
    assert checks["positive"]
    assert checks["counterexamples"]["premature_entry_replacement"]
    account_id = next(g["account_id"] for g in candidate.goals if g["kind"] == "replacement_wait")
    # Reproduce the old, executable but policy-invalid reference, even with a
    # larger action bucket and a corresponding order goal.
    invalid = candidate.model_copy(deep=True)
    invalid.slot.difficulty = "5-9"
    invalid.goals = [g for g in invalid.goals if g["kind"] != "replacement_wait"]
    invalid.goals.append({"kind": "replacement", "account_id": account_id,
                          "delivery_option": "STANDARD", "delivery_fee": 0,
                          "card_design": "CLASSIC", "design_fee": 0})
    invalid.task.evaluation_criteria.actions += [
        Action(action_id="bad_unlock", name="unlock_discoverable_agent_tool", arguments={"agent_tool_name": "order_debit_card_5739"}),
        Action(action_id="bad_order", name="call_discoverable_agent_tool", arguments={
            "agent_tool_name": "order_debit_card_5739", "arguments": json.dumps({
                "account_id": account_id, "user_id": entry.uid,
                "delivery_option": "STANDARD", "delivery_fee": 0,
                "card_design": "CLASSIC", "design_fee": 0,
                "shipping_address": entry.user["address"]})})]
    with pytest.raises(ValueError, match="48-hour"):
        validate_static(invalid)

    mid = Builder(slot("debit", "10-14"), 0, 2)
    debit(mid, 2, 2)  # First card: freeze; second, Blue-account card: replace now.
    mid_candidate = mid.finish()
    assert any(g["kind"] == "replacement" for g in mid_candidate.goals)
    assert not any(g["kind"] == "replacement_wait" for g in mid_candidate.goals)
    assert validate_static(mid_candidate)["positive"]


def test_gate_write_identity_preserves_types_but_normalizes_json_numbers():
    from tau3.synthesis.targeted.native.gate import operation_identity

    name = "apply_savings_account_credit_6831"
    arguments = {"account_id": "a", "amount": 33, "credit_type": "interest_correction"}
    assert operation_identity("assistant", name, arguments) == operation_identity("assistant", name, {**arguments, "amount": 33.0})
    assert operation_identity("assistant", name, arguments) != operation_identity("assistant", name, {**arguments, "amount": "33"})


def test_native_bundle_public_loader_checks_corpus_and_task_hashes(tmp_path, monkeypatch):
    import hashlib

    from tau3.synthesis.bundle import load_task_bundle
    from tau3.synthesis.storage import digest, write_json
    from tau3.synthesis.targeted.native import bundle
    from tau3.synthesis.targeted.native.models import NativePlan

    candidate = compile_candidate(slot("accounts"))
    files = {"runtime/test.py": "frozen"}
    monkeypatch.setattr(bundle, "sources", lambda: files)
    snapshot = {"hashes": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in files.items()}}
    snapshot["snapshot_hash"] = digest(snapshot)
    write_json(tmp_path / "snapshot/manifest.json", snapshot)
    plan = NativePlan(profile_hash="x", snapshot_hash=snapshot["snapshot_hash"], slots=[], pilot=[], validation=[])
    bundle.publish(tmp_path, "pilot", [candidate], plan, "train")
    target = tmp_path / "pilot/bundle"
    assert load_task_bundle(target)[0] == candidate.task
    files["runtime/test.py"] = "changed"
    with pytest.raises(ValueError, match="runtime or corpus"):
        load_task_bundle(target)
    files["runtime/test.py"] = "frozen"
    write_json(target / "tasks.json", [])
    with pytest.raises(ValueError, match="artifacts changed"):
        load_task_bundle(target)


def test_native_first_initialization_archives_before_publishing_manifest(tmp_path, monkeypatch):
    import gzip
    import json

    from tau3.synthesis.targeted.native.workflow import initialize

    # initialize intentionally configures its isolated worker process. Scope
    # these settings in the test so later V2 tests cannot inherit a native pool.
    for key in ("TAU3_LLM_CONCURRENCY", "TAU3_LLM_POOL", "TAU3_HTTP_TIMING_DIR"):
        monkeypatch.setenv(key, "")
    config = NativeConfig(training_contract="configs/synthesis/targeted-native-training.json")
    first = initialize(tmp_path / "fresh", config)
    assert first == initialize(tmp_path / "fresh", config)
    with gzip.open(tmp_path / "fresh/snapshot/sources.json.gz", "rt") as handle:
        assert "data/db.json" in json.load(handle)


def test_interest_and_consolidation_share_one_unambiguous_retained_account():
    from tau3.synthesis.targeted.native.scenarios import (
        Builder,
        accounts,
        savings_correction,
    )

    b = Builder(slot("optimization", "15-19"), 0, 10)
    savings_correction(b, 10)
    accounts(b, 2, 10)
    assert sum(row["level"] == "Blue Account" for row in b.data["accounts"]["data"].values()) == 1
    candidate = b.finish()
    count = len(candidate.task.evaluation_criteria.actions)
    candidate.slot.difficulty = "15-19" if count >= 15 else "10-14"
    assert validate_static(candidate)["positive"]


@pytest.mark.parametrize("role", ["teacher_model", "reviewer_model"])
def test_native_structured_request_uses_strict_parsing_without_gateway_json_mode(tmp_path, monkeypatch, role):
    from litellm import ModelResponse

    from tau3.synthesis.targeted.budget import ask
    from tau3.utils import llm_utils

    captured = []
    def complete(**kwargs):
        captured.append(kwargs)
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": '{"valid":true}'}, "finish_reason": "stop"}], usage={"total_tokens": 10})
    monkeypatch.setattr(llm_utils, "completion", complete)
    config = NativeConfig()
    assert ask(tmp_path, config, "analysis", getattr(config, role), "JSON", {}) == {"valid": True}
    assert captured[0]["max_tokens"] == (32768 if role == "reviewer_model" else 16384)
    if role == "reviewer_model":
        assert captured[0]["timeout"] == 900
    assert "response_format" not in captured[0]
    assert "response_format" not in captured[0]["extra_body"]
    assert captured[0]["extra_body"]["reasoning_effort"] == "low"


def test_section_analysis_binds_quotes_without_inventing_counts(tmp_path, monkeypatch):
    from tau3.synthesis.targeted.native import analysis

    report = tmp_path / "report.md"
    report.write_text("The model wrote an extra account update.\n\n" * 120)
    assert "".join(analysis.sections(report.read_text())) == report.read_text()
    def reply(*args, **kwargs):
        return {"findings": [{"label": "F4", "observation": "Extra writes", "interpretation": "Possible restraint gap",
            "source": "report", "quote": "The model wrote an extra account update.", "confidence": "high", "severity": "high",
            "affected_task_count": 123, "mechanisms": ["accounts"], "quarantined": False}], "quarantine": []}
    monkeypatch.setattr(analysis, "ask", reply)
    profile = analysis.analyze(tmp_path / "round", NativeConfig(), report, "test", "base")
    assert len(profile.findings) == 1
    assert profile.findings[0].affected_task_count is None
    assert profile.findings[0].source == str(report.resolve())


def test_source_quote_repairs_only_typography_and_rejects_changed_facts():
    from tau3.synthesis.targeted.native.analysis import ground_quote

    source = "截图中的“130 条、45 个成功”不能作为当前最终成绩口径。"
    assert ground_quote('截图中的"130 条、45 个成功"不能作为当前最终成绩口径', source) == source[:-1]
    with pytest.raises(ValueError):
        ground_quote('截图中的"97 条、45 个成功"不能作为当前最终成绩口径', source)


def test_verification_after_a_successful_write_is_not_valid():
    from tau3.synthesis.targeted.native.validation import check_goals, operations
    from tau3.synthesis.validation import replay

    candidate = compile_candidate(slot("accounts", "1-4"))
    actions = candidate.task.evaluation_criteria.actions
    assert actions[0].name == "log_verification"
    env, messages = replay(candidate.task, [*actions[1:], actions[0]])
    assert any("verification" in error for error in check_goals(candidate, env, operations(messages)))


def test_resume_cannot_accept_a_truncated_physical_response(tmp_path, monkeypatch):
    from litellm import ModelResponse

    from tau3.synthesis.targeted.native.runner import journal
    from tau3.utils import llm_utils
    from tau3.worldgen.v2.audit import AuditIncomplete

    attempts = []
    def truncated(**kwargs):
        attempts.append(kwargs)
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": "partial"}, "finish_reason": "length"}], usage={"total_tokens": 10})
    monkeypatch.setattr(llm_utils, "completion", truncated)
    for _ in range(2):
        with pytest.raises(AuditIncomplete), journal(tmp_path, NativeConfig(), tmp_path / "responses", lambda _: "teacher"):
            llm_utils.completion(model="openai/GLM-5.3-Flash", messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert len(attempts) == 1


@pytest.mark.parametrize("failure", ["empty", "timeout"])
def test_native_user_recovery_keeps_seed_and_replays_completed_prefix(tmp_path, monkeypatch, failure):
    from litellm import ModelResponse

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.runner import journal
    from tau3.utils import llm_utils

    requests = []
    def transport(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1 and failure == "timeout":
            raise TimeoutError("network")
        text = "" if len(requests) == 1 else "Please close my account."
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], usage={"total_tokens": 10})
    monkeypatch.setattr(llm_utils, "completion", transport)
    kwargs = {"model": "openai/gemini-3.5-flash", "messages": [{"role": "system", "content": "Be the user"}], "seed": 123, "max_tokens": 100}
    for _ in range(2):
        with journal(tmp_path, NativeConfig(), tmp_path / "responses", lambda _: "teacher_user"):
            assert llm_utils.completion(**kwargs).choices[0].message.content
    assert len(requests) == 2
    assert [r["seed"] for r in requests] == [123, 123]
    assert kwargs["messages"][0]["content"] == "Be the user"
    assert len(read_json(tmp_path / "responses/00000.json")["history"]) == 1
    assert read_json(tmp_path / "budget.json")["calls"] == 2


def test_empty_user_recovery_is_bounded_across_resume(tmp_path, monkeypatch):
    from litellm import ModelResponse

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.native.runner import journal
    from tau3.utils import llm_utils
    from tau3.worldgen.v2.audit import AuditIncomplete

    monkeypatch.setattr(llm_utils, "completion", lambda **kw: ModelResponse(choices=[{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}], usage={"total_tokens": 10}))
    for _ in range(2):
        with pytest.raises(AuditIncomplete, match="three attempts"), journal(tmp_path, NativeConfig(), tmp_path / "responses", lambda _: "teacher_user"):
            llm_utils.completion(model="openai/gemini-3.5-flash", messages=[{"role": "system", "content": "User"}], max_tokens=100)
    assert read_json(tmp_path / "budget.json")["calls"] == 3


def test_malformed_native_audit_retries_original_evidence_not_verdict(tmp_path, monkeypatch):
    from litellm import ModelResponse

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.budget import ask
    from tau3.utils import llm_utils

    requests = []
    def transport(**kwargs):
        requests.append(kwargs)
        text = '{"valid":true' if len(requests) == 1 else '{"valid":false}'
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], usage={"total_tokens": 10})
    monkeypatch.setattr(llm_utils, "completion", transport)
    config = NativeConfig()
    for _ in range(2):
        assert ask(tmp_path, config, "admission_review", config.reviewer_model, "Review", {"evidence": 1}) == {"valid": False}
    assert len(requests) == 2
    assert requests[0]["messages"][1] == requests[1]["messages"][1]
    assert read_json(tmp_path / "budget.json")["calls"] == 2
    assert len(list((tmp_path / "requests").glob("*.json"))) == 2


def test_retrieval_replay_preserves_original_timing_but_rejects_document_drift(tmp_path):
    from types import SimpleNamespace

    from tau3.data_model.message import ToolCall, ToolMessage
    from tau3.synthesis.storage import write_json
    from tau3.synthesis.targeted.native.runner import restore_retrieval_observations
    from tau3.worldgen.v2.audit import AuditIncomplete

    original = 'Policy text\n[Timing: retrieval=2ms, reranking=0ms, total=2ms]'
    current = 'Policy text\n[Timing: retrieval=1ms, reranking=0ms, total=1ms]'
    directory = tmp_path / 'trial'
    write_json(directory / 'responses/00000.json', {'physical_id': 'p'})
    write_json(tmp_path / 'physical_calls/p.json', {'request': {'messages': [
        {'role': 'assistant', 'tool_calls': [{'id': 'call1', 'function': {'name': 'KB_search', 'arguments': '{"query":"accounts"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call1', 'content': original}]}})
    env = SimpleNamespace(get_response=lambda call: ToolMessage(id=call.id, content=current, requestor='assistant', role='tool', error=False))
    restore_retrieval_observations(tmp_path, directory, env)
    call = ToolCall(id='call1', name='KB_search', arguments={'query': 'accounts'}, requestor='assistant')
    assert env.get_response(call).content == original
    assert len(list((directory / 'retrieval-replay').glob('*.json'))) == 1
    current = 'Different policy text\n[Timing: retrieval=1ms, reranking=0ms, total=1ms]'
    with pytest.raises(AuditIncomplete, match='document contents'):
        env.get_response(call)
    with pytest.raises(AuditIncomplete, match='call changed'):
        env.get_response(call.model_copy(update={'arguments': {'query': 'cards'}}))


def test_quality_schema_and_index_errors_retry_without_forcing_approval(tmp_path, monkeypatch):
    import json

    from litellm import ModelResponse

    from tau3.synthesis.targeted.budget import ask
    from tau3.utils import llm_utils

    replies = []
    valid = dict(reasoning_correct=False, grounded_in_visible_history=True,
                 user_compliant=True, handoff_correct=True, final_answer_correct=True,
                 no_private_leakage=True, erroneous_assistant_turns=[],
                 unlocalizable_error=False, explanation='Wrong reasoning, reject.')
    malformed = {**valid, 'grounded_in_isible_history': True}
    del malformed['grounded_in_visible_history']
    sequence = [malformed, {**valid, 'erroneous_assistant_turns': [4]}, valid]
    def complete(**kwargs):
        replies.append(kwargs)
        return ModelResponse(choices=[{'message': {'role': 'assistant', 'content': json.dumps(sequence[len(replies)-1])}, 'finish_reason': 'stop'}], usage={'total_tokens': 10})
    monkeypatch.setattr(llm_utils, 'completion', complete)
    cfg = NativeConfig()
    for _ in range(2):
        result = ask(tmp_path, cfg, 'reasoning_quality', cfg.reviewer_model, 'Audit',
                     {'agent_context': {'messages': [{'role': 'assistant', 'content': 'x'}]}})
        assert result['reasoning_correct'] is False
    assert len(replies) == 3


def test_native_transient_gateway_quota_retry_is_counted(tmp_path, monkeypatch):
    from litellm import ModelResponse

    from tau3.synthesis.storage import read_json
    from tau3.synthesis.targeted.budget import Budget

    class GatewayError(Exception):
        status_code = 403
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise GatewayError('Backend error (glm): QS quota exceeded')
        return ModelResponse(choices=[{'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}], usage={'total_tokens': 10})
    monkeypatch.setattr('tau3.synthesis.targeted.budget.time.sleep', lambda _: None)
    config = NativeConfig()
    config.settings.audit_transport_attempts = 3
    result = Budget(tmp_path, config).call('blind', transport, model='openai/GLM-5.3-Flash', messages=[], max_tokens=100)
    assert result.choices[0].message.content == 'OK'
    assert len(calls) == read_json(tmp_path / 'budget.json')['calls'] == 2
    assert len(list((tmp_path / 'physical_calls').glob('*.json'))) == 2


def test_pilot_forecast_does_not_count_formal_spend_as_pilot(tmp_path):
    from tau3.synthesis.storage import write_json
    from tau3.synthesis.targeted.native.workflow import budget_forecast

    write_json(tmp_path / 'budget.json', {'calls': 100, 'audit_calls': 50, 'tokens': 1000})
    first = budget_forecast(tmp_path, NativeConfig())
    write_json(tmp_path / 'budget.json', {'calls': 90000, 'audit_calls': 50000, 'tokens': 900000000})
    assert budget_forecast(tmp_path, NativeConfig()) == first


def test_physical_evidence_parent_is_read_only_and_snapshot_bound(tmp_path):
    from tau3.synthesis.storage import write_json
    from tau3.synthesis.targeted.native.runner import physical_record
    from tau3.worldgen.v2.audit import AuditIncomplete

    parent, child = tmp_path / 'parent', tmp_path / 'child'
    write_json(parent / 'snapshot/manifest.json', {'snapshot_hash': 'original'})
    write_json(parent / 'physical_calls/p.json', {'status': 'COMPLETE', 'id': 'p'})
    write_json(child / 'evidence-parent.json', {'root': str(parent), 'snapshot_hash': 'original'})
    assert physical_record(child, 'p')['status'] == 'COMPLETE'
    assert not (child / 'physical_calls/p.json').exists()
    write_json(parent / 'snapshot/manifest.json', {'snapshot_hash': 'changed'})
    with pytest.raises(AuditIncomplete, match='snapshot changed'):
        physical_record(child, 'p')
