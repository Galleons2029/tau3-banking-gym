"""Offline regressions for streamed admission, visible SFT and frozen response reuse."""

import json
from types import SimpleNamespace

import pytest

from tau3.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau3.data_model.simulation import RewardInfo, SimulationRun
from tau3.data_model.tasks import Task
from tau3.synthesis import streaming_tasks as stream
from tau3.synthesis import world_sft as sft
from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.catalog import scaffold
from tau3.worldgen.v2.pipeline import build, publish, write_json
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import load_settings
from tau3.worldgen.v2.specs import WorldSpec


def test_migration_is_bound_to_exact_code_world_and_source(world, tmp_path):
    from tau3.synthesis.stream_resume import check_migration
    from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash

    previous, output = tmp_path / "previous", tmp_path / "resume"
    write_json(previous / "manifest.json", {"identity": "original"})
    settings = load_settings()
    report = {
        "status": "PASS",
        "source": str(previous.resolve()),
        "source_manifest": digest({"identity": "original"}),
        "implementation": implementation_hash(),
        "artifacts": artifact_hashes(world),
        "settings": digest(settings.model_dump()),
    }
    write_json(output / "migration.json", report)
    assert check_migration(output, previous, world, settings) == report
    write_json(previous / "manifest.json", {"identity": "changed"})
    with pytest.raises(AuditIncomplete, match="stale"):
        check_migration(output, previous, world, settings)


def test_legacy_negative_teacher_is_preserved_and_tampering_rejected(
    world, tmp_path, monkeypatch
):
    from tau3.synthesis import stream_resume as resume

    task = Task.model_validate_json(
        (world / "tasks/task_checking_grounding.json").read_text()
    )
    settings = load_settings()
    identity = {
        "task": digest(task.model_dump(mode="json")),
        "model": settings.agent_models[0],
        "producer": "old",
    }
    captured = {"identity": identity, "simulation": {}, "sample": sample()}
    result = {
        "identity": identity,
        "capture_hash": digest(captured),
        "simulation": simulation(task.id, RewardInfo(reward=0)).model_dump(mode="json"),
    }
    write_json(tmp_path / "started.json", identity)
    write_json(tmp_path / "capture.json", captured)
    write_json(tmp_path / "result.json", result)
    monkeypatch.setattr(
        resume.EnvironmentEvaluator,
        "calculate_reward",
        lambda *a, **k: RewardInfo(reward=0),
    )
    before = (tmp_path / "result.json").read_bytes()
    assert resume.check_trial(tmp_path, task, world, settings) == result
    assert (tmp_path / "result.json").read_bytes() == before
    captured["sample"]["task_id"] = "tampered"
    write_json(tmp_path / "capture.json", captured)
    with pytest.raises(AuditIncomplete, match="binding"):
        resume.check_trial(tmp_path, task, world, settings)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "world"
    assert (
        build(root, WorldSpec(categories=[scaffold("checking", "checking")]))["status"]
        == "PASS"
    )
    publish(root)
    return root


def simulation(task_id, reward=None):
    return SimulationRun(
        id="test",
        task_id=task_id,
        start_time="2025-11-14T00:00:00",
        end_time="2025-11-14T00:00:01",
        duration=1,
        termination_reason="user_stop",
        reward_info=reward,
        messages=[UserMessage(role="user", content="PRIVATE_USER_TOOL_CANARY")],
    )


def sample():
    return {
        "task_id": "task",
        "messages": [
            {"role": "system", "content": "Public policy"},
            {"role": "user", "content": "Request"},
            {"role": "assistant", "content": "Answer", "weight": 1},
        ],
        "tools": [],
        "loss_mask": [0, 0, 1],
    }


def test_sft_rejects_validation_masks_and_broken_tool_results(tmp_path):
    row = sample()
    with pytest.raises(ValueError, match="training"):
        sft.export_shard(tmp_path / "invalid.jsonl", [row], "validation", {})
    row["loss_mask"] = [1, 1, 1]
    with pytest.raises(ValueError, match="mask"):
        sft.validate_sample(row)
    row = sample()
    row["messages"].insert(
        2, {"role": "tool", "content": "unseen", "tool_call_id": "unknown"}
    )
    row["loss_mask"] = [0, 0, 0, 1]
    with pytest.raises(ValueError, match="visible"):
        sft.validate_sample(row)
    sft.export_shard(tmp_path / "ready.jsonl", [sample()], "train", {"proof": "bound"})
    assert len((tmp_path / "ready.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("reward", [-1, 0, 1])
def test_capture_is_visible_only_persisted_before_grading_and_resumable(
    world, tmp_path, monkeypatch, reward
):
    from tau3.evaluator import evaluator
    from tau3.runner import build as runner

    task = Task.model_validate_json(
        (world / "tasks/task_checking_grounding.json").read_text()
    )
    task.id = "synth_actual_customer_request"
    task.user_scenario.instructions += " Keep this actual request."
    directory = tmp_path / "trial"
    count = []
    state = SimpleNamespace(
        system_messages=[SystemMessage(role="system", content="PUBLIC_POLICY")],
        messages=[
            UserMessage(role="user", content="VISIBLE_REQUEST"),
            AssistantMessage(role="assistant", content="VISIBLE_ANSWER"),
        ],
    )

    def make(config, actual, seed):
        count.append(actual.id)
        assert config.agent == "llm_agent" and config.domain == "banking_synth"
        return SimpleNamespace(
            agent_state=state,
            agent=SimpleNamespace(tools=[]),
            environment=SimpleNamespace(get_policy=lambda: "PUBLIC_POLICY"),
            run=lambda: simulation(task.id),
        )

    def grade(**kwargs):
        assert (directory / "capture.json").exists()
        assert (
            kwargs["env_kwargs"]["task"].user_scenario.instructions
            == task.user_scenario.instructions
        )
        assert kwargs["domain"] == "banking_synth"
        if reward == -1:
            raise RuntimeError("grading interrupted")
        return RewardInfo(reward=reward)

    monkeypatch.setattr(runner, "build_text_orchestrator", make)
    monkeypatch.setattr(evaluator, "evaluate_simulation", grade)
    settings = load_settings()
    if reward == -1:
        with pytest.raises(RuntimeError, match="grading interrupted"):
            sft.capture_trial(task, settings, "teacher", directory, 42, {})
        with pytest.raises(AuditIncomplete, match="Interrupted grading"):
            sft.capture_trial(task, settings, "teacher", directory, 42, {})
        assert count == [task.id]
        return
    sft.capture_trial(task, settings, "teacher", directory, 42, {})
    sft.capture_trial(task, settings, "teacher", directory, 42, {})
    assert count == [task.id]
    if reward:
        row = sft.trial_sample(directory)
        assert "PRIVATE_USER_TOOL_CANARY" not in json.dumps(row)
        assert "VISIBLE_ANSWER" in json.dumps(row) and row["loss_mask"] == [0, 0, 1]
        result = json.loads((directory / "result.json").read_text())
        result["capture_hash"] = "changed"
        write_json(directory / "result.json", result)
        with pytest.raises(ValueError, match="binding"):
            sft.trial_sample(directory)
    else:
        with pytest.raises(ValueError, match="Failed"):
            sft.trial_sample(directory)


def test_interrupted_teacher_never_silently_restarts(world, tmp_path, monkeypatch):
    from tau3.runner import build as runner

    task = Task.model_validate_json(
        (world / "tasks/task_checking_grounding.json").read_text()
    )
    attempts = []

    def crash(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(runner, "build_text_orchestrator", crash)
    settings = load_settings()
    with pytest.raises(RuntimeError):
        sft.capture_trial(task, settings, "teacher", tmp_path / "trial", 42, {})
    with pytest.raises(AuditIncomplete, match="Interrupted"):
        sft.capture_trial(task, settings, "teacher", tmp_path / "trial", 42, {})
    assert len(attempts) == 1


def test_reuse_copies_negative_raw_response_and_charges_once(
    world, tmp_path, monkeypatch
):
    from tau3.utils import llm_utils

    session = ConcurrentAuditSession(
        world, tmp_path / "new", load_settings(), 10, "test"
    )
    request = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "public"},
            {"role": "user", "content": json.dumps({"a": 1})},
        ],
    }
    record = {
        "request": request,
        "status": "COMPLETE",
        "response": '{"ok":false}',
        "finish_reason": "stop",
        "attempts": [{"status": "COMPLETE"}],
    }
    old = tmp_path / "old"
    write_json(old / "calls" / f"{digest(request)}.json", record)
    stream.reuse_request(session, old, request)
    stream.reuse_request(session, old, request)
    monkeypatch.setattr(
        llm_utils,
        "generate",
        lambda **kw: pytest.fail("reused response must not call model"),
    )
    assert session.ask("model", "public", {"a": 1}) == {"ok": False}
    assert json.loads((session.output / "budget.json").read_text())["used"] == 1
    assert json.loads(
        (session.output / "calls" / f"{digest(request)}.json").read_text()
    )["origin"]["record_hash"] == digest(record)


def test_dedup_claim_survives_restart(tmp_path):
    stream.claim_expression(tmp_path, "same", "first")
    stream.claim_expression(tmp_path, "same", "first")
    with pytest.raises(ValueError, match="Duplicate"):
        stream.claim_expression(tmp_path, "same", "second")


def test_stream_exports_first_task_before_later_slot_finishes_and_resumes(
    world, tmp_path, monkeypatch
):
    # Earlier V2 rollout tests configure their worker through this process-global
    # variable. This test explicitly exercises an empty/default settings file.
    monkeypatch.delenv("TAU3_SYNTH_ROLLOUT_CONFIG", raising=False)
    from threading import Event

    from tau3.synthesis import bundle as bundles
    from tau3.synthesis.world_tasks import adapter_hash
    from tau3.utils import llm_utils
    from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash

    source, output = tmp_path / "source", tmp_path / "stream"
    settings = load_settings()
    proof = tmp_path / "ready.json"
    write_json(proof, {"status": "PASS"})
    source_manifest = {
        "source_artifacts": artifact_hashes(world),
        "implementation": implementation_hash(),
        "adapter_hash": adapter_hash(),
        "settings_hash": digest(settings.model_dump()),
        "requested_tasks": 8000,
        "source_world": str(world),
        "readiness": str(proof),
        "bundle_id": "source8k",
    }
    write_json(source / "manifest.json", source_manifest)
    ConcurrentAuditSession(
        world, source / "audit", settings, 240000, "task-expression-v1"
    )
    seed_test = set(json.loads((world / "splits.json").read_text())["test"])
    anchor = next(
        Task.model_validate_json(p.read_text())
        for p in (world / "tasks").glob("*.json")
        if p.stem not in seed_test
    )
    pilot_tasks = []
    for i in range(20):
        task = anchor.model_copy(deep=True)
        task.id = f"pilot_{i}"
        task.user_scenario.instructions = f"Pilot expression {i}"
        pilot_tasks.append(task)
    pilot = tmp_path / "pilot"
    write_json(pilot / "manifest.json", {"status": "published"})
    monkeypatch.setattr(bundles, "load_task_bundle", lambda *a, **k: pilot_tasks)
    monkeypatch.setattr(stream, "check_readiness", lambda *a: {})
    monkeypatch.setattr(stream, "balanced_anchors", lambda *a: [anchor])

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][-1].content)
        if "variant" in payload:
            response = {
                "request": anchor.user_scenario.instructions
                + [" Please be concise.", " Please use simple language."][
                    payload["variant"][0]
                ]
            }
        else:
            response = {"equivalent": True, "no_solution_added": True, "issues": []}
        return AssistantMessage(role="assistant", content=json.dumps(response))

    monkeypatch.setattr(llm_utils, "generate", generate)
    witness = {
        "answer": "Answer",
        "evidence": [{"source": "records"}],
        "steps": [],
        "changed_rows": [],
        "calculations": [],
    }
    monkeypatch.setattr(stream, "solve_public", lambda *a: (witness, {}, 1))
    monkeypatch.setattr(stream, "check_frozen_witness", lambda *a: None)
    first_published = Event()
    runs = []

    def capture(task, settings, model, directory, seed, binding):
        if "_000001_" in task.id:
            assert first_published.wait(10), (
                "A later slot blocked the earlier SFT export"
            )
        assert len(list(directory.parent.glob("blind_*.json"))) == 2
        runs.append(task.id)
        result = {
            "simulation": simulation(task.id, RewardInfo(reward=1)).model_dump(
                mode="json"
            )
        }
        write_json(directory / "result.json", result)
        return result

    monkeypatch.setattr(stream, "capture_trial", capture)
    monkeypatch.setattr(
        stream,
        "trial_sample",
        lambda p: sample()
        | {
            "task_id": json.loads((p.parent / "candidate.json").read_text())["task"][
                "id"
            ],
            "teacher_model": p.name,
        },
    )
    export = stream.export_shard

    def publish(path, samples, split, source):
        value = export(path, samples, split, source)
        if path.name == "000000.jsonl":
            first_published.set()
        return value

    monkeypatch.setattr(stream, "export_shard", publish)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    result = stream.stream_tasks(
        world, source, pilot, config, output, count=2, workers=2
    )
    assert result == {"status": "COMPLETE", "admitted_tasks": 2, "sft_samples": 4}
    assert len(runs) == 4
    assert len(list((output / "sft/shards").glob("*.jsonl"))) == 2
    stream.stream_tasks(world, source, pilot, config, output, count=2, workers=2)
    assert len(runs) == 4, "Resume repeated completed model conversations"
