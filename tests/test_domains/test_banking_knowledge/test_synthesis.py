"""Offline correctness and admission tests for the fixed-KB training pipeline."""

import json
from types import SimpleNamespace

import pytest

from tau3.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau3.synthesis.bundle import load_task_bundle
from tau3.synthesis.catalog import build_catalog
from tau3.synthesis.llm import Review, visible_sft_sample
from tau3.synthesis.models import SynthesisConfig
from tau3.synthesis.scenarios import (
    cli_decision,
    reward_points,
    sample_candidate,
    solve_selection,
)
from tau3.synthesis.storage import (
    digest,
    environment_fingerprint,
    read_json,
    retrieval_settings,
    write_json,
)
from tau3.synthesis.validation import (
    check_business,
    fresh_environment,
    replay,
    validate_references,
    validate_static,
)
from tau3.synthesis.workflow import (
    checkpoint,
    export_bundle,
    grouped_split,
    validate_bundle,
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.fixture
def candidates(catalog):
    return [sample_candidate(catalog, 42 + i * 1009, i) for i in range(20)]


def draft(root, catalog, items):
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "domain": "banking_knowledge",
            "status": "draft",
            "environment_hash": environment_fingerprint(),
            "requested_tasks": len(items),
            "retrieval": retrieval_settings(),
            "config": SynthesisConfig().model_dump(),
        },
    )
    write_json(root / "catalog.json", catalog.model_dump(mode="json"))
    for item in items:
        checkpoint(root, item)


def mark_admitted(record):
    """Unit-test fixture for publication, not a replacement for live validation."""
    record.static_passed = record.text_checked = record.accepted = True
    record.text_mode = "llm"
    record.trials = {
        f"bm25_grep:{i}": {"mode": "bm25_grep", "passed": True} for i in range(2)
    }
    record.checks["validated_content_hash"] = digest(
        [record.task.model_dump(mode="json"), record.skeleton.model_dump(mode="json")]
    )


def test_catalog_evidence_and_glm_defaults(catalog):
    assert all(e.quote and len(e.sha256) == 64 for e in catalog.documents.values())
    assert catalog.rules["cli_tiers"]["entry"]["age"] == 120
    config = SynthesisConfig()
    assert {
        config.generator_model,
        config.teacher_model,
        config.user_model,
        config.judge_model,
    } == {"openai/GLM5.3-agentic-qs-h20"}
    settings = retrieval_settings()
    assert settings["name"] == "bm25_grep"
    assert settings["grep"]["case_sensitive"] is False
    assert settings["kb_search"]["top_k"] == settings["grep"]["top_k"] == 10


def test_unique_selection_and_boundaries(catalog):
    def solve(amount):
        return solve_selection(
            catalog.products,
            {"mobile_check_deposit_per_day": amount, "early_days": 0, "fee_cap": "25"},
            age=20,
        )

    assert solve(500) == "Light Green Account"
    assert solve(501) == "Blue Account"
    assert solve(2501) == "Green Account (checking)"
    with pytest.raises(ValueError, match="No eligible"):
        solve(3001)
    products = {**catalog.products, "Clone": catalog.products["Light Green Account"]}
    with pytest.raises(ValueError, match="Ambiguous"):
        solve_selection(
            products,
            {"mobile_check_deposit_per_day": 500, "early_days": 0, "fee_cap": "25"},
            age=20,
        )


@pytest.mark.parametrize("tier", ["entry", "mid", "premium"])
def test_cli_boundaries(catalog, tier):
    rules = catalog.rules["cli_tiers"][tier]
    state = {
        "age": rules["age"],
        "balance": "0",
        "limit": "10000",
        "increase": "1000",
        "pending_disputes": False,
        "pending_replacement": False,
        "past_due": False,
        "on_time_months": rules["months"],
    }
    assert cli_decision(state, rules) == "approved"
    assert (
        cli_decision({**state, "age": rules["age"] - 1}, rules)
        == "insufficient_account_age"
    )
    assert (
        cli_decision({**state, "balance": str(rules["utilization"] * 100)}, rules)
        == "high_utilization"
    )
    assert (
        cli_decision({**state, "on_time_months": rules["months"] - 1}, rules)
        == "insufficient_payment_history"
    )
    with pytest.raises(ValueError, match="adjusted"):
        cli_decision({**state, "increase": "9000"}, rules)


def test_reward_units_and_unspecified_rounding(catalog):
    assert reward_points("100", "Travel", "Silver Rewards Card", catalog) == 400
    assert reward_points("100", "Groceries", "Silver Rewards Card", catalog) == 100
    with pytest.raises(ValueError, match="rounding"):
        reward_points("100.01", "Groceries", "Silver Rewards Card", catalog)


@pytest.mark.parametrize("slot", range(4))
def test_real_reference_replay_and_counterexamples(catalog, slot):
    record = sample_candidate(catalog, 42 + slot, slot)
    result = validate_static(record, catalog)
    assert result["counterexamples"]["empty"] == "rejected"
    assert result["counterexamples"]["extra_write"] == "rejected"
    assert record.task.evaluation_criteria.reward_basis == ["DB"]


def test_independent_business_check_rejects_permissive_tool(catalog):
    record = sample_candidate(catalog, 42, 0)
    actions = [a.model_copy(deep=True) for a in record.task.evaluation_criteria.actions]
    inner = json.loads(actions[-1].arguments["arguments"])
    inner["account_class"] = "Invented Account"
    actions[-1].arguments["arguments"] = json.dumps(inner)
    env, _ = replay(record.task, actions)
    with pytest.raises(ValueError, match="Wrong selected"):
        check_business(record, env, catalog)


def test_reference_error_text_and_state_isolation(catalog):
    record = sample_candidate(catalog, 42, 0)
    with pytest.raises(ValueError, match="Reference action failed"):
        replay(record.task, [record.task.evaluation_criteria.actions[-1]])
    first, second = fresh_environment(record.task), fresh_environment(record.task)
    first.tools.db.users.data[record.skeleton.private["user_id"]]["email"] = (
        "changed@example.com"
    )
    assert first.tools.db is first.user_tools.db
    assert (
        second.tools.db.users.data[record.skeleton.private["user_id"]]["email"]
        != "changed@example.com"
    )


def test_broken_reference_and_user_permissions(catalog):
    record = sample_candidate(catalog, 43, 1)
    record.task.user_tools = []
    with pytest.raises(ValueError, match="User tool"):
        validate_references(record, catalog)
    record.task.user_tools = ["call_discoverable_user_tool"]
    row = next(
        iter(
            record.task.initial_state.initialization_data.agent_data[
                "credit_card_accounts"
            ]["data"].values()
        )
    )
    row["user_id"] = "missing_user"
    with pytest.raises(ValueError, match="Broken reference"):
        validate_references(record, catalog)


def test_grouped_split_keeps_literal_variants_together(candidates):
    candidates[1].skeleton.group_id = candidates[5].skeleton.group_id
    splits = grouped_split(candidates)
    groups = {r.task.id: r.skeleton.group_id for r in candidates}
    assert not (
        {groups[i] for i in splits["train"]} & {groups[i] for i in splits["validation"]}
    )
    assert set(splits["base"]) == set(splits["train"]) | set(splits["validation"])


def test_cannot_publish_offline_or_golden_only(tmp_path, catalog, candidates):
    for record in candidates:
        record.static_passed = True
        record.trials = {"golden_retrieval": {"passed": True}}
    draft(tmp_path, catalog, candidates)
    with pytest.raises(ValueError, match="No task"):
        export_bundle(tmp_path)
    with pytest.raises(ValueError, match="publication"):
        load_task_bundle(tmp_path)


@pytest.fixture
def published(tmp_path, catalog, candidates):
    for record in candidates:
        mark_admitted(record)
    draft(tmp_path, catalog, candidates)
    export_bundle(tmp_path)
    return tmp_path


def test_bundle_loading_tamper_and_unknown_split(published):
    assert len(load_task_bundle(published)) == 20
    with pytest.raises(ValueError, match="Unknown task split"):
        load_task_bundle(published, "missing")
    tasks = read_json(published / "tasks.json")
    tasks.append(tasks[0])
    manifest = read_json(published / "manifest.json")
    manifest["tasks_hash"] = digest(tasks)
    write_json(published / "tasks.json", tasks)
    write_json(published / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        load_task_bundle(published)


def test_runner_and_gym_bundle_defaults(published):
    from tau3.data_model.simulation import TextRunConfig
    from tau3.gym.gym_agent import AgentGymEnv, UserGymEnv
    from tau3.runner.batch import _load_run_tasks

    tasks = load_task_bundle(published, "train")
    config = TextRunConfig(task_bundle=str(published), task_split_name="train")
    assert config.retrieval_config == "bm25_grep"
    assert [t.id for t in _load_run_tasks(config)] == [t.id for t in tasks]
    for cls in (AgentGymEnv, UserGymEnv):
        env = cls(
            domain="banking_knowledge",
            task_id=tasks[0].id,
            task_bundle=str(published),
            task_split_name="train",
        )
        assert env._get_task() == tasks[0]
        assert env.retrieval_config == "bm25_grep"
        assert env.retrieval_config_kwargs == {"top_k": 10}
        env.close()
    diagnostic = TextRunConfig(
        task_bundle=str(published), retrieval_config="no_knowledge"
    )
    assert diagnostic.retrieval_config == "no_knowledge"


def test_review_rejects_missing_boolean_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Review.model_validate({"explanation": "looks good"})


def test_sft_uses_visible_agent_state_only():
    state = SimpleNamespace(
        system_messages=[SystemMessage(role="system", content="public policy")],
        messages=[
            UserMessage(role="user", content="help"),
            AssistantMessage(role="assistant", content="I can help"),
        ],
    )
    orchestrator = SimpleNamespace(
        agent_state=state,
        agent=SimpleNamespace(tools=[]),
        user_state=SimpleNamespace(
            messages=[
                ToolMessage(
                    role="tool", id="secret", requestor="user", content="PRIVATE_RESULT"
                )
            ]
        ),
    )
    sample = visible_sft_sample(orchestrator, "test")
    assert "PRIVATE_RESULT" not in json.dumps(sample)
    assert sample["loss_mask"] == [0, 0, 1]


def test_live_validation_resume_does_not_repeat_completed_calls(
    tmp_path, catalog, candidates, monkeypatch
):
    import tau3.synthesis.workflow as workflow

    items = candidates[:4]
    for record in items:
        record.text_mode = "llm"
        record.text_checked = True
        record.checks["reviewed_text_hash"] = digest(
            record.task.user_scenario.model_dump(mode="json")
        )
    draft(tmp_path, catalog, items)
    calls = []
    monkeypatch.setattr(workflow, "probe", lambda config: None)
    monkeypatch.setattr(workflow, "validate_static", lambda *args: {})

    def trial(root, record, config, mode, seed):
        calls.append((record.task.id, mode, seed))
        return {"mode": mode, "passed": True}

    monkeypatch.setattr(workflow, "run_trial", trial)
    validate_bundle(tmp_path)
    count = len(calls)
    validate_bundle(tmp_path, resume=True)
    assert len(calls) == count
    record = workflow.records(tmp_path)[0]
    record.task.user_scenario.instructions.reason_for_call = "Changed after review"
    checkpoint(tmp_path, record)
    validate_bundle(tmp_path, resume=True)
    assert not workflow.records(tmp_path)[0].accepted


def test_narrative_resume_reuses_saved_output(catalog, monkeypatch):
    import tau3.synthesis.llm as llm

    record = sample_candidate(catalog, 42, 0)
    calls = []
    saved = []

    def response(model, system, payload, config, call_name):
        calls.append(payload)
        assert set(payload) == {"goal"}
        return {"reason_for_call": "Please help me compare my shortlisted accounts."}, {
            "cost": None
        }

    monkeypatch.setattr(llm, "json_response", response)
    monkeypatch.setattr(
        llm, "review", lambda *args: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(TimeoutError):
        llm.narrate(
            record,
            SynthesisConfig(),
            on_update=lambda: saved.append(record.model_dump(mode="json")),
        )
    assert saved[-1]["checks"]["phase"] == "text_review"
    assert record.checks["pending_narrative"]
    monkeypatch.setattr(
        llm,
        "review",
        lambda *args: (
            Review(
                fact_consistent=True,
                no_answer_leak=True,
                follows_user_constraints=True,
                explanation="ok",
            ),
            {},
        ),
    )
    llm.narrate(record, SynthesisConfig())
    assert record.text_checked
    assert len(calls) == 1


def test_validation_uses_only_bm25_grep(tmp_path, catalog, candidates, monkeypatch):
    import tau3.synthesis.workflow as workflow

    record = candidates[0]
    record.text_checked = True
    record.checks["reviewed_text_hash"] = digest(
        record.task.user_scenario.model_dump(mode="json")
    )
    draft(tmp_path, catalog, [record])
    monkeypatch.setattr(workflow, "probe", lambda config: None)
    monkeypatch.setattr(workflow, "validate_static", lambda *args: {})

    def trial(root, record, config, mode, seed):
        assert mode == "bm25_grep"
        return {"mode": mode, "passed": True}

    monkeypatch.setattr(workflow, "run_trial", trial)
    validate_bundle(tmp_path)
    result = workflow.records(tmp_path)[0]
    assert result.accepted
    assert set(result.trials) == {"bm25_grep:0", "bm25_grep:1"}


def test_generation_refills_failed_live_slots_only(
    tmp_path, catalog, candidates, monkeypatch
):
    import tau3.synthesis.workflow as workflow

    items = candidates[:4]
    for record in items:
        record.static_passed = record.text_checked = True
        record.trials = {
            f"bm25_grep:{i}": {"mode": "bm25_grep", "passed": False} for i in range(2)
        }
    items[1].trials["bm25_grep:1"]["infrastructure_error"] = True
    items[2].accepted = True
    items[2].trials["bm25_grep:0"]["passed"] = True
    items[3].trials = {}
    draft(tmp_path, catalog, items)
    monkeypatch.setattr(workflow, "probe", lambda config: None)
    monkeypatch.setattr(workflow, "validate_static", lambda *args: {})
    monkeypatch.setattr(
        workflow,
        "narrate",
        lambda record, config, **kw: setattr(record, "text_checked", True),
    )
    workflow.generate_bundle(tmp_path, 4, SynthesisConfig(), resume=True)
    result = workflow.records(tmp_path)
    assert [r.attempt for r in result] == [1, 0, 0, 0]
    assert result[0].task.id != items[0].task.id
    assert result[1].task == items[1].task
    assert (tmp_path / "rejections/000000_00.json").exists()


def test_sft_rejects_changed_candidate(published, monkeypatch):
    import tau3.synthesis.workflow as workflow

    training_ids = {t.id for t in load_task_bundle(published, "train")}
    for record in workflow.records(published):
        if record.task.id in training_ids:
            record.task.user_scenario.instructions.reason_for_call = (
                "Changed after publication"
            )
            checkpoint(published, record)
    monkeypatch.setattr(workflow, "probe", lambda config: None)
    with pytest.raises(ValueError, match="after publication"):
        workflow.collect_sft(published)


def test_trial_resume_reuses_completed_simulation(tmp_path, catalog, monkeypatch):
    import tau3.runner.build as build
    import tau3.runner.simulation as runner
    import tau3.synthesis.workflow as workflow
    from tau3.data_model.simulation import RewardInfo, SimulationRun, TerminationReason

    record = sample_candidate(catalog, 42, 0)
    state = SimpleNamespace(
        system_messages=[SystemMessage(role="system", content="public policy")],
        messages=[AssistantMessage(role="assistant", content="Done")],
    )
    orchestrator = SimpleNamespace(agent_state=state, agent=SimpleNamespace(tools=[]))
    simulation = SimulationRun(
        id="resume-test",
        task_id=record.task.id,
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=state.messages,
        reward_info=RewardInfo(reward=1),
    )
    calls = []
    monkeypatch.setattr(build, "build_text_orchestrator", lambda *a, **kw: orchestrator)
    monkeypatch.setattr(build, "build_env_kwargs", lambda *a: {})

    def run(*args, **kwargs):
        calls.append(1)
        return simulation

    monkeypatch.setattr(runner, "run_simulation", run)
    monkeypatch.setattr(
        workflow, "review", lambda *a: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(TimeoutError):
        workflow.run_trial(tmp_path, record, SynthesisConfig(), "bm25_grep", 42)
    monkeypatch.setattr(
        workflow,
        "review",
        lambda *a: (
            Review(
                fact_consistent=True,
                no_answer_leak=True,
                follows_user_constraints=True,
                explanation="ok",
            ),
            {},
        ),
    )
    result = workflow.run_trial(tmp_path, record, SynthesisConfig(), "bm25_grep", 42)
    assert len(calls) == 1
    assert result["mode"] == "bm25_grep"
    assert result["passed"]
    simulation.reward_info = RewardInfo(reward=0)
    result = workflow.run_trial(tmp_path, record, SynthesisConfig(), "bm25_grep", 43)
    assert not result["passed"] and result["review"] is None
    assert result["review_usage"]["model_review_skipped"]


def test_batch_stops_short_pilot_before_publication(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    source = Path(__file__).resolve().parents[3] / "scripts/run_synthesis_batch.py"
    spec = importlib.util.spec_from_file_location("synthesis_batch", source)
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    calls = []
    monkeypatch.setattr(
        batch, "generate_bundle", lambda *a, **kw: calls.append("generate")
    )
    monkeypatch.setattr(batch, "validate_bundle", lambda *a, **kw: {"accepted": 19})
    monkeypatch.setattr(batch, "records", lambda root: [])
    monkeypatch.setattr(batch, "export_bundle", lambda root: calls.append("export"))
    with pytest.raises(ValueError, match="No admission progress"):
        batch.build_admitted_bundle(
            tmp_path, 20, SynthesisConfig(), lambda *a, **kw: None
        )
    assert calls == ["generate"] * 3


def test_strict_schema_and_failure_capture(monkeypatch):
    import tau3.synthesis.llm as llm

    outputs = iter(["", '{"reason_for_call":"Please compare my accounts."}'])
    calls, saved = [], []

    def generate(**kwargs):
        calls.append(kwargs)
        content = next(outputs)
        return SimpleNamespace(
            content=content,
            usage={"completion_tokens": 4096},
            cost=None,
            raw_data={"choices": [{"finish_reason": "stop" if content else "length"}]},
        )

    monkeypatch.setattr(llm, "generate", generate)
    with llm.response_log(saved.append):
        obj, _ = llm.json_response(
            "openai/GLM5.3-agentic-qs-h20",
            "Return JSON",
            {},
            SynthesisConfig(),
            "synthesis_narrative",
        )
    assert obj["reason_for_call"]
    assert len(saved) == 2 and saved[0]["content"] == ""
    assert saved[0]["finish_reasons"] == ["length"]
    assert calls[1]["max_tokens"] == 8192
    assert calls[1]["timeout"] > calls[0]["timeout"]
    for call in calls:
        fmt = call["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        schema = fmt["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["reason_for_call"]
        assert call["num_retries"] == call["max_retries"] == 0


def test_structured_errors_preserve_candidate_budget(
    tmp_path, catalog, candidates, monkeypatch
):
    import tau3.synthesis.workflow as workflow
    from tau3.synthesis.llm import StructuredResponseError

    items = candidates[:4]
    draft(tmp_path, catalog, items)
    monkeypatch.setattr(workflow, "probe", lambda c: None)
    monkeypatch.setattr(workflow, "validate_static", lambda *a: {})
    monkeypatch.setattr(
        workflow,
        "narrate",
        lambda *a, **kw: (_ for _ in ()).throw(StructuredResponseError("invalid JSON")),
    )
    workflow.generate_bundle(tmp_path, 4, SynthesisConfig(), resume=True)
    workflow.generate_bundle(tmp_path, 4, SynthesisConfig(), resume=True)
    result = workflow.records(tmp_path)
    assert [r.task.id for r in result] == [r.task.id for r in items]
    assert all(r.attempt == 0 and r.checks["retryable_error"] for r in result)
    assert not list((tmp_path / "rejections").glob("*.json"))


def test_exhausted_business_candidates_are_not_replayed(tmp_path, monkeypatch):
    import tau3.synthesis.workflow as workflow

    calls = []

    def invalid(*args):
        calls.append(1)
        raise ValueError("invalid business state")

    monkeypatch.setattr(workflow, "validate_static", invalid)
    config = SynthesisConfig(text_mode="template", candidate_attempts=1)
    workflow.generate_bundle(tmp_path, 4, config)
    workflow.generate_bundle(tmp_path, 4, config, resume=True)
    assert len(calls) == 4
    assert all(r.checks["candidate_exhausted"] for r in workflow.records(tmp_path))


def test_fixed_official_approval_conflict_is_quarantined(catalog):
    record = sample_candidate(catalog, 42, 2)
    env = fresh_environment(record.task)
    uid, cid = record.skeleton.private["user_id"], record.skeleton.private["card_id"]
    # Isolate the insertion conflict from eligibility checks without changing tools.
    env.tools.db.credit_card_accounts.data[cid]["account_status"] = "CURRENT"
    for table in (env.tools.db.transaction_disputes, env.tools.db.credit_card_orders):
        for row in table.data.values():
            if row.get("user_id") == uid:
                row["status"] = "RESOLVED"
    increase = int(record.skeleton.private["state"]["increase"])
    current = int(record.skeleton.private["state"]["limit"])
    env.tools.submit_credit_limit_increase_request_7392(cid, uid, increase)
    response = env.tools.approve_credit_limit_increase_5847(
        cid, uid, current + increase
    )
    assert "approved!" in response
    rows = [
        r
        for r in env.tools.db.credit_limit_increase_requests.data.values()
        if r.get("user_id") == uid
    ]
    assert len(rows) == 1 and rows[0]["status"] == "PENDING"
    record.skeleton.private["decision"] = "approved"
    with pytest.raises(ValueError, match="Quarantined"):
        validate_references(record, catalog)
    for seed in range(30):
        assert (
            sample_candidate(catalog, seed, 2).skeleton.private["decision"]
            != "approved"
        )
    assert any("CLI approvals" in text for text in catalog.exclusions)


def test_repair_fork_preserves_source_and_accepted_tasks(tmp_path, catalog, candidates):
    from tau3.synthesis.workflow import fork_repaired_bundle, records

    source, destination = tmp_path / "old", tmp_path / "new"
    items = candidates[:4]
    mark_admitted(items[0])
    items[2].skeleton.private["decision"] = "approved"
    draft(source, catalog, items)
    before = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    fork_repaired_bundle(source, destination, SynthesisConfig())
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")}
    assert before == after
    result = records(destination)
    assert result[0].accepted and result[0].trials == items[0].trials
    assert not (destination / "candidates/000002.json").exists()
    assert (destination / "quarantine/000002.json").exists()
    assert (
        read_json(destination / "manifest.json")["environment_hash"]
        == environment_fingerprint()
    )


def test_review_projection_preserves_user_evidence_and_native_messages():
    from copy import deepcopy

    from tau3.synthesis.llm import review_transcript

    messages = [
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [
                {"id": "kb", "name": "KB_search", "arguments": {"query": "fees"}}
            ],
        },
        {
            "role": "tool",
            "id": "kb",
            "requestor": "assistant",
            "content": "LARGE_PRIVATE_RETRIEVAL",
            "error": False,
        },
        {
            "role": "tool",
            "id": "business",
            "requestor": "assistant",
            "content": "BANK_RECORD",
        },
        {"role": "assistant", "content": "The fee is $20."},
        {
            "role": "tool",
            "id": "customer",
            "requestor": "user",
            "content": "ACTUAL_PRIVATE_APPROVAL",
        },
        {"role": "user", "content": "I received approval."},
    ]
    original = deepcopy(messages)
    projected = review_transcript(messages)
    assert messages == original
    assert "LARGE_PRIVATE_RETRIEVAL" not in json.dumps(projected)
    assert projected[1]["omitted_content_hash"] == digest("LARGE_PRIVATE_RETRIEVAL")
    assert projected[2:] == original[2:]
    assert projected[0] == original[0]


def test_explicit_deposit_boundary_cannot_be_weakened(catalog):
    from tau3.synthesis.llm import explicit_requirement_errors

    record = sample_candidate(catalog, 42, 0)
    record.skeleton.facts["preferences"]["mobile_check_deposit_per_day"] = 2501
    bad = [
        {
            "role": "user",
            "content": "I need to be able to mobile deposit at least $2,500 a day in checks.",
        }
    ]
    assert explicit_requirement_errors(record, bad)
    good = [
        {
            "role": "user",
            "content": "I need to be able to mobile deposit at least $2,501 a day in checks.",
        }
    ]
    assert not explicit_requirement_errors(record, good)
    question = [
        {"role": "user", "content": "Can I mobile deposit at least $2,500 a day?"}
    ]
    assert not explicit_requirement_errors(record, question)


def test_light_green_age_boundary_and_evidence(catalog):
    from tau3.synthesis.scenarios import customer_age

    constraints = {
        "mobile_check_deposit_per_day": 500,
        "early_days": 0,
        "fee_cap": "25",
    }
    assert (
        solve_selection(catalog.products, constraints, age=24) == "Light Green Account"
    )
    assert solve_selection(catalog.products, constraints, age=25) == "Blue Account"
    assert customer_age("11/14/2000") == 25
    assert customer_age("11/15/2000") == 24
    assert "doc_checking_accounts_light_green_account_002" in catalog.documents


def test_cli_submit_uses_visible_discovery_route(catalog):
    record = sample_candidate(catalog, 42, 2)
    actions = record.task.evaluation_criteria.actions
    assert not any(
        a.name == "submit_credit_limit_increase_request_7392" for a in actions
    )
    assert any(
        a.name == "call_discoverable_agent_tool"
        and a.arguments.get("agent_tool_name")
        == "submit_credit_limit_increase_request_7392"
        for a in actions
    )
    assert validate_static(record, catalog)
    # A direct hidden call can execute internally but is not a valid agent reference.
    call = next(
        a
        for a in actions
        if a.arguments.get("agent_tool_name")
        == "submit_credit_limit_increase_request_7392"
        and a.name == "call_discoverable_agent_tool"
    )
    call.name = "submit_credit_limit_increase_request_7392"
    call.arguments = json.loads(call.arguments["arguments"])
    with pytest.raises(ValueError, match="hidden tool"):
        validate_references(record, catalog)


def test_ordering_reason_is_explicit_user_requirement(catalog):
    record = sample_candidate(catalog, 42, 3)
    assert "Record my closure reason exactly" in record.skeleton.facts["goal"]
    assert "Proactively" in record.skeleton.facts["interaction"]
    for action in record.task.evaluation_criteria.actions:
        if (
            action.name == "call_discoverable_agent_tool"
            and action.arguments["agent_tool_name"] == "close_bank_account_7392"
        ):
            assert (
                json.loads(action.arguments["arguments"])["reason"]
                == "Customer requested closure"
            )


def test_known_factual_failure_skips_model_review(catalog, monkeypatch):
    import tau3.synthesis.llm as llm

    record = sample_candidate(catalog, 42, 0)
    record.skeleton.facts["preferences"]["mobile_check_deposit_per_day"] = 2501
    monkeypatch.setattr(
        llm,
        "json_response",
        lambda *a: pytest.fail("Model must not run after a deterministic rejection"),
    )
    result, usage = llm.review(
        record,
        SynthesisConfig(),
        [
            {
                "role": "user",
                "content": "I need to mobile deposit at least $2,500 a day.",
            }
        ],
    )
    assert not result.passed
    assert usage["model_review_skipped"]
    assert usage["unassessed"] == ["no_answer_leak", "follows_user_constraints"]


def test_reference_route_repair_regrades_unchanged_dialogue(tmp_path, catalog):
    from tau3.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
    from tau3.synthesis.workflow import repair_reference_routes, trial_config

    record = sample_candidate(catalog, 42, 2)
    _, messages = replay(record.task)
    # Test fixture: reproduce the old compiler's direct hidden submit action.
    old_actions = []
    for action in record.task.evaluation_criteria.actions:
        if (
            action.arguments.get("agent_tool_name")
            == "submit_credit_limit_increase_request_7392"
        ):
            if action.name == "unlock_discoverable_agent_tool":
                continue
            action = action.model_copy(deep=True)
            action.name = "submit_credit_limit_increase_request_7392"
            action.arguments = json.loads(action.arguments["arguments"])
        old_actions.append(action)
    record.task.evaluation_criteria.actions = old_actions
    before = record.task.model_dump(mode="json")
    config = SynthesisConfig()
    for i in range(2):
        seed = 42 + i
        path = (
            tmp_path
            / "trajectories"
            / record.task.id
            / f"validation_bm25_grep_{seed}.json"
        )
        sim = SimulationRun(
            id=f"test-{i}",
            task_id=record.task.id,
            start_time="start",
            end_time="end",
            duration=1,
            termination_reason=TerminationReason.USER_STOP,
            messages=messages,
            reward_info=RewardInfo(reward=0),
        )
        write_json(path, sim.model_dump(mode="json"))
        write_json(
            path.with_suffix(".capture.json"),
            {
                "input_hash": digest(
                    [
                        before,
                        trial_config(config, "bm25_grep", seed).model_dump(mode="json"),
                    ]
                ),
                "retry_simulation": False,
            },
        )
        record.trials[f"bm25_grep:{i}"] = {
            "mode": "bm25_grep",
            "seed": seed,
            "reward": 0,
            "passed": False,
            "trajectory": str(path.relative_to(tmp_path)),
            "review": {
                "fact_consistent": True,
                "no_answer_leak": True,
                "follows_user_constraints": True,
            },
        }
    repair_reference_routes(tmp_path, record, config, catalog)
    assert record.task.user_scenario.model_dump(mode="json") == before["user_scenario"]
    assert record.task.initial_state.model_dump(mode="json") == before["initial_state"]
    assert record.accepted
    assert all(t["reward"] == 1 for t in record.trials.values())
    for t in record.trials.values():
        native = read_json(tmp_path / t["trajectory"])
        assert native["messages"] == [m.model_dump(mode="json") for m in messages]
    assert (
        tmp_path / "history/reference_routes" / record.task.id / "task_before.json"
    ).exists()


def load_delivery_auditor():
    """Load the standalone auditor without changing synthesis implementation hashes."""
    import importlib.util
    from pathlib import Path

    source = Path(__file__).resolve().parents[3] / "scripts/audit_synthesis_bundle.py"
    spec = importlib.util.spec_from_file_location("synthesis_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delivery_audit_rejects_declared_success_without_native_evidence(published):
    report = load_delivery_auditor().audit(published, expected=20)
    assert report["artifact_checks_passed"] is False
    assert "SFT export is missing" in report["errors"]
    assert any("static" in error for error in report["errors"])


def test_delivery_audit_checks_actual_context_and_assistant_mask():
    import copy

    check = load_delivery_auditor().check_sft_sample
    sample = {
        "task_id": "task",
        "retrieval_config": "bm25_grep",
        "messages": [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "hello", "weight": 1},
        ],
        "loss_mask": [0, 1],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in ("KB_search", "grep")
        ],
    }
    assert check(sample, sample, "task") == []
    changed = copy.deepcopy(sample)
    changed["messages"].append({"role": "tool", "content": "PRIVATE USER RESULT"})
    assert any("actual agent-visible" in e for e in check(changed, sample, "task"))
    changed["messages"][-1]["tool_call_id"] = "private"
    native = [{"role": "tool", "requestor": "user", "id": "private"}]
    assert any("Native private" in e for e in check(changed, changed, "task", native))
    changed = copy.deepcopy(sample)
    changed["loss_mask"] = [1, 1]
    assert any("Loss mask" in e for e in check(changed, sample, "task"))
    changed = copy.deepcopy(sample)
    changed["retrieval_config"] = "golden_retrieval"
    assert any("retrieval mode" in e for e in check(changed, sample, "task"))


def test_delivery_audit_rejects_untracked_or_unfinished_sft_attempts():
    check = load_delivery_auditor().check_sft_attempts
    assert check({"sft:0": {"seed": 142}}, {"sft:0": {"seed": 142}}, 42) == []
    assert check({}, {"sft:0": {"seed": 142}}, 42)
    assert check({"sft:0": {"seed": 142}}, {}, 42)
    assert check({"sft:0": {"seed": 142}}, {"sft:0": {"seed": 143}}, 42)
    assert check({"sft:4": {"seed": 146}}, {"sft:4": {"seed": 146}}, 42)


def test_export_does_not_drop_combined_operation_coverage(published):
    from tau3.synthesis.workflow import records

    before = (published / "tasks.json").read_bytes()
    manifest = read_json(published / "manifest.json")
    manifest["sampling_requirements"] = {"combined_ordering_tasks": 1}
    write_json(published / "manifest.json", manifest)
    for record in records(published):
        if record.skeleton.private.get("open_business"):
            record.accepted = False
            checkpoint(published, record)
    with pytest.raises(ValueError, match="combined opening/closure coverage"):
        export_bundle(published)
    assert (published / "tasks.json").read_bytes() == before
