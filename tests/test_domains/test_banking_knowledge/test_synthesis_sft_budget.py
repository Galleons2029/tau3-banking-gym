from types import SimpleNamespace

import pytest

from tau3.synthesis.models import SynthesisConfig
from tau3.synthesis.storage import digest, write_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import tau3.synthesis.workflow as module

    task = SimpleNamespace(
        id="budget-task", model_dump=lambda **kw: {"id": "budget-task"}
    )
    record = SimpleNamespace(
        task=task, skeleton=SimpleNamespace(seed=42), checks={}, trials={}, errors=[]
    )
    config = SynthesisConfig()
    write_json(tmp_path / "manifest.json", {"config": config.model_dump()})
    import tau3.synthesis.bundle as bundle

    monkeypatch.setattr(bundle, "load_task_bundle", lambda *a: [task])
    monkeypatch.setattr(module, "records", lambda *a: [record])
    monkeypatch.setattr(module, "probe", lambda *a: None)
    monkeypatch.setattr(module, "checkpoint", lambda *a: None)
    return module, tmp_path, record, config


def test_failed_simulations_are_not_repeated_on_resume(setup, monkeypatch):
    module, root, record, config = setup
    calls = []

    def run(root, record, config, mode, seed, purpose):
        assert f"sft:{seed - 142}" in record.checks["sft_attempts_started"]
        calls.append(seed)
        return {
            "mode": mode,
            "seed": seed,
            "passed": False,
            "infrastructure_error": True,
        }

    monkeypatch.setattr(module, "run_trial", run)
    module.collect_sft(root)
    module.collect_sft(root)
    assert calls == [142, 143, 144, 145]
    assert len(record.checks["sft_attempts_started"]) == 4


def test_interruption_before_capture_consumes_attempt(setup, monkeypatch):
    module, root, record, config = setup
    record.checks["sft_attempts_started"] = {"sft:0": {"seed": 142}}
    calls = []

    def run(*args):
        calls.append(args[4])
        return {"mode": "bm25_grep", "passed": False}

    monkeypatch.setattr(module, "run_trial", run)
    module.collect_sft(root)
    assert calls == [143, 144, 145]
    assert record.trials["sft:0"]["infrastructure_error"] is True


def test_review_retry_reuses_native_and_does_not_consume_another_run(
    setup, monkeypatch
):
    module, root, record, config = setup
    native_runs = []
    judge_calls = []

    def run(root, record, config, mode, seed, purpose):
        capture = (
            root
            / "trajectories"
            / record.task.id
            / f"sft_bm25_grep_{seed}.capture.json"
        )
        judge_calls.append(seed)
        if not capture.exists():
            native_runs.append(seed)
            write_json(
                capture,
                {
                    "retry_simulation": False,
                    "input_hash": digest(
                        [
                            record.task.model_dump(),
                            module.trial_config(config, mode, seed).model_dump(
                                mode="json"
                            ),
                        ]
                    ),
                },
            )
            raise RuntimeError("Judge service unavailable after native run saved")
        return {
            "mode": mode,
            "seed": seed,
            "passed": False,
            "infrastructure_error": False,
        }

    monkeypatch.setattr(module, "run_trial", run)
    for _ in range(6):
        module.collect_sft(root)
    assert native_runs == [142, 143, 144, 145]
    assert judge_calls == [142, 142, 143, 143, 144, 144, 145, 145]


def test_changed_cached_inputs_fail_without_new_simulation(setup, monkeypatch):
    module, root, record, config = setup
    write_json(
        root / "trajectories" / record.task.id / "sft_bm25_grep_142.capture.json",
        {"retry_simulation": False, "input_hash": "wrong"},
    )
    monkeypatch.setattr(
        module, "run_trial", lambda *a: pytest.fail("Must not resample")
    )
    with pytest.raises(ValueError, match="cached run inputs changed"):
        module.collect_sft(root)
