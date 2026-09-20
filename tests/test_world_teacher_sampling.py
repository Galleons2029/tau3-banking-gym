"""Regression checks for model-specific teacher sampling and immutable reuse."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau3.synthesis import teacher_sampling as teachers
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.pipeline import write_json
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import RolloutSettings

GLM = "openai/GLM-5.3-Flash"
GEMINI = "openai/gemini-3.5-flash"


@pytest.fixture
def inputs():
    return SimpleNamespace(model_dump=lambda **kw: {"id": "task"}), RolloutSettings(
        agent_models=[GEMINI, GLM]
    )


def historical(path, task, settings, model=GLM, seed=43, reward=1):
    identity = {
        "task": digest(task.model_dump()),
        "settings": digest(settings.model_dump()),
        "model": model,
        "seed": seed,
        "producer": "historical",
    }
    write_json(path / "started.json", identity)
    write_json(path / "result.json", {"reward": reward})
    return path


def test_two_seeds_and_single_teacher_are_explicit():
    policy = teachers.TeacherPolicy(model=GLM)
    assert policy.seeds == (42, 43)
    with pytest.raises(ValidationError):
        teachers.TeacherPolicy(model=GLM, seeds=[42, 42])
    with pytest.raises(ValidationError):
        teachers.TeacherPolicy(model="")


@pytest.mark.parametrize("reward", [0, 1])
def test_reuse_glm_preserves_positive_and_negative_verdicts(
    tmp_path, inputs, monkeypatch, reward
):
    task, settings = inputs
    source = historical(tmp_path / "old", task, settings, reward=reward)
    before = (source / "result.json").read_bytes()
    calls = []
    monkeypatch.setattr(
        teachers,
        "check_trial",
        lambda d, *args: json.loads((d / "result.json").read_text()),
    )

    def capture(*args):
        return calls.append(args)

    destination = tmp_path / "new"
    for _ in range(2):
        result = teachers.run_teacher_trial(
            task, settings, GLM, destination, 43, {}, tmp_path, [source], capture
        )
        assert result == {"reward": reward}
    assert not calls
    assert (source / "result.json").read_bytes() == before
    write_json(source / "result.json", {"reward": "tampered"})
    with pytest.raises(AuditIncomplete, match="evidence changed"):
        teachers.resolve_trial(destination)


def test_gemini_never_substitutes_for_glm_and_seeds_do_not_alias(
    tmp_path, inputs, monkeypatch
):
    task, settings = inputs
    gemini = historical(tmp_path / "gemini", task, settings, GEMINI, 42)
    glm = historical(tmp_path / "glm", task, settings, GLM, 43)
    calls = []
    monkeypatch.setattr(teachers, "check_trial", lambda d, *args: {"reward": 1})

    def capture(task, settings, model, directory, seed, binding):
        calls.append((model, seed))
        return {"reward": 1}

    for seed in (42, 43):
        teachers.run_teacher_trial(
            task,
            settings,
            GLM,
            tmp_path / str(seed),
            seed,
            {},
            tmp_path,
            [gemini, glm],
            capture,
        )
    assert calls == [(GLM, 42)]
    assert settings.agent_models == [GEMINI, GLM]


def test_interrupted_glm_is_not_resampled_and_binding_cannot_change(tmp_path, inputs):
    task, settings = inputs
    source = historical(tmp_path / "old", task, settings)
    (source / "result.json").unlink()

    def forbidden(*args):
        pytest.fail("Interrupted GLM must not be silently resampled")

    with pytest.raises(AuditIncomplete, match="interrupted GLM"):
        teachers.run_teacher_trial(
            task, settings, GLM, tmp_path / "new", 43, {}, tmp_path, [source], forbidden
        )


def test_export_rejects_mislabeled_model_or_seed(tmp_path, inputs, monkeypatch):
    task, settings = inputs
    source = historical(tmp_path / "old", task, settings)
    monkeypatch.setattr(
        teachers, "trial_sample", lambda d: {"teacher_model": GEMINI, "seed": 43}
    )
    with pytest.raises(AuditIncomplete, match="binding"):
        teachers.teacher_sample(source, GLM, 43)


def test_current_checkpoint_is_reused_by_capture_identity_guard(tmp_path, inputs):
    task, settings = inputs
    source = historical(tmp_path / "current", task, settings)
    calls = []
    teachers.run_teacher_trial(
        task,
        settings,
        GLM,
        source,
        43,
        {"generation": 2},
        tmp_path,
        [],
        lambda *args: calls.append(args),
    )
    assert calls[0][2] == GLM and calls[0][4] == 43
    assert calls[0][5] == {"generation": 2}


def test_stream_routes_glm_only_keeps_dual_verifiers_and_resumes(tmp_path, monkeypatch):
    from tau3.data_model.simulation import RewardInfo, SimulationRun
    from tau3.data_model.tasks import Task, UserScenario
    from tau3.domains.banking_synth import environment
    from tau3.synthesis import bundle
    from tau3.synthesis import streaming_tasks as stream

    settings = RolloutSettings(agent_models=[GEMINI, GLM])
    task = Task(id="anchor", user_scenario=UserScenario(instructions="My request"))
    world, source, pilot, output = [
        tmp_path / n for n in ("world", "source", "pilot", "out")
    ]
    write_json(world / "splits.json", {"test": []})
    write_json(pilot / "manifest.json", {})
    readiness = tmp_path / "ready.json"
    write_json(readiness, {"status": "PASS"})
    write_json(
        source / "manifest.json",
        {
            "requested_tasks": 1,
            "source_artifacts": {},
            "implementation": "runtime",
            "adapter_hash": "adapter",
            "settings_hash": digest(settings.model_dump()),
            "readiness": str(readiness),
            "bundle_id": "test",
        },
    )
    write_json(
        source / "audit/audit.json",
        {
            "artifacts": {},
            "implementation": "runtime",
            "settings": digest(settings.model_dump()),
            "kind": "task-expression-v1",
        },
    )
    monkeypatch.setattr(stream, "configure", lambda *a: settings)
    monkeypatch.setattr(stream, "artifact_hashes", lambda *a: {})
    monkeypatch.setattr(stream, "implementation_hash", lambda: "runtime")
    monkeypatch.setattr(stream, "adapter_hash", lambda: "adapter")
    monkeypatch.setattr(stream, "check_certificate", lambda *a: None)
    monkeypatch.setattr(stream, "check_readiness", lambda *a: None)
    monkeypatch.setattr(stream, "balanced_anchors", lambda *a: [task])
    monkeypatch.setattr(stream, "task_structures", lambda *a: {task.id: "group"})
    monkeypatch.setattr(environment, "get_tasks", lambda: [task])
    monkeypatch.setattr(bundle, "load_task_bundle", lambda *a: [task] * 20)
    monkeypatch.setattr(stream, "expression_errors", lambda *a: [])
    monkeypatch.setattr(stream, "reuse_request", lambda *a: None)
    monkeypatch.setattr(stream, "public_problem", lambda *a, **k: {})
    monkeypatch.setattr(stream, "Witness", SimpleNamespace(model_validate=lambda x: x))
    monkeypatch.setattr(stream, "check_frozen_witness", lambda *a: None)
    verifiers, captures = [], []

    def solve(problem, model, reviewer, ask):
        verifiers.append((model, reviewer))
        return {}, {}, []

    monkeypatch.setattr(stream, "solve_public", solve)

    class Session:
        def __init__(self, world, output, *a):
            self.output = output

        def ask(self, model, system, payload):
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            }
            write_json(
                self.output / "calls" / f"{digest(request)}.json", {"request": request}
            )
            return (
                {"request": "My rewritten request"}
                if system == stream.REWRITE
                else {"equivalent": True, "no_solution_added": True, "issues": []}
            )

    monkeypatch.setattr(stream, "ConcurrentAuditSession", Session)

    def capture(task, settings, model, directory, seed, binding):
        captures.append((model, seed))
        signature = {
            "task": digest(task.model_dump(mode="json")),
            "settings": digest(settings.model_dump()),
            "model": model,
            "seed": seed,
            "binding": binding,
        }
        simulation = SimulationRun(
            id=f"trial-{seed}",
            task_id=task.id,
            start_time="2026-09-14T00:00:00",
            end_time="2026-09-14T00:00:01",
            duration=1,
            termination_reason="agent_stop",
            reward_info=RewardInfo(reward=1),
            messages=[],
        ).model_dump(mode="json")
        captured = {
            "identity": signature,
            "simulation": simulation,
            "sample": {
                "task_id": task.id,
                "teacher_model": model,
                "seed": seed,
                "messages": [{"role": "assistant", "content": "Answer"}],
                "loss_mask": [1],
            },
        }
        result = {
            "identity": signature,
            "simulation": simulation,
            "capture_hash": digest(captured),
        }
        write_json(directory / "started.json", signature)
        write_json(directory / "capture.json", captured)
        write_json(directory / "result.json", result)
        return result

    monkeypatch.setattr(stream, "capture_trial", capture)
    monkeypatch.setattr(
        teachers, "check_trial", lambda d, *args: teachers.read(d / "result.json")
    )
    policy = teachers.TeacherPolicy(model=GLM)
    for _ in range(2):
        report = stream.stream_tasks(
            world,
            source,
            pilot,
            tmp_path / "config",
            output,
            1,
            workers=1,
            teacher_policy=policy,
        )
        assert report == {"status": "COMPLETE", "admitted_tasks": 1, "sft_samples": 2}
    assert captures == [(GLM, 42), (GLM, 43)]
    assert verifiers == [(GEMINI, GLM), (GLM, GEMINI)]
    rows = [
        json.loads(line)
        for line in (output / "sft/shards/000000.jsonl").read_text().splitlines()
    ]
    assert {r["teacher_model"] for r in rows} == {GLM}
    assert {r["seed"] for r in rows} == {42, 43}
    with pytest.raises(ValueError, match="identity changed"):
        stream.stream_tasks(
            world,
            source,
            pilot,
            tmp_path / "config",
            output,
            1,
            teacher_policy=teachers.TeacherPolicy(model=GEMINI),
        )


def test_older_generation_rejections_survive_missing_or_stale_newer_copy(tmp_path):
    latest, oldest = tmp_path / "latest", tmp_path / "oldest"
    write_json(
        oldest / "rejected.json",
        {
            "status": "REJECTED",
            "reason": "Real model conversation failed",
            "attempt": 0,
        },
    )
    expected = {
        "status": "REJECTED",
        "reason": "Real model conversation failed",
        "attempt": 0,
        "origin": str(oldest),
    }
    assert teachers.historical_rejection([latest, oldest]) == expected
    write_json(latest / "rejected.json", {"reason": "Stale validation certificate"})
    assert teachers.historical_rejection([latest, oldest]) == expected
    assert teachers.historical_rejection([latest]) is None
