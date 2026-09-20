import importlib
from types import SimpleNamespace

import pytest

from tau3.synthesis.models import SynthesisConfig
from tau3.synthesis.storage import digest, write_json


def load_candidate(name, filename):
    return importlib.import_module("tau3.synthesis." + filename.removesuffix(".py"))


def test_review_budget_exhaustion_is_distinct_from_transport(monkeypatch):
    llm = load_candidate("pending_budget_llm", "llm.py")
    budgets = []

    def truncated(**kwargs):
        budgets.append(kwargs["max_tokens"])
        return SimpleNamespace(
            content="",
            raw_data={"choices": [{"finish_reason": "length"}]},
            usage={},
            cost=0,
        )

    monkeypatch.setattr(llm, "generate", truncated)
    with pytest.raises(llm.ReviewBudgetExceeded):
        llm.json_response(
            "endpoint", "judge", {}, SynthesisConfig(), "synthesis_review"
        )
    assert budgets == [8192, 16384]
    monkeypatch.setattr(
        llm, "generate", lambda **kw: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(llm.StructuredResponseError) as error:
        llm.json_response(
            "endpoint", "judge", {}, SynthesisConfig(), "synthesis_review"
        )
    assert not isinstance(error.value, llm.ReviewBudgetExceeded)


def test_inconclusive_review_rejects_cached_db_success_without_resampling(
    tmp_path, monkeypatch
):
    import tau3.runner.build as build
    import tau3.synthesis.llm as installed_llm
    from tau3.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
    from tau3.synthesis.catalog import build_catalog
    from tau3.synthesis.scenarios import sample_candidate

    llm = load_candidate("pending_budget_llm", "llm.py")
    monkeypatch.setattr(
        installed_llm, "ReviewBudgetExceeded", llm.ReviewBudgetExceeded, raising=False
    )
    workflow = load_candidate("pending_budget_workflow", "workflow.py")
    config = SynthesisConfig()
    record = sample_candidate(build_catalog(), 42, 0)
    directory = tmp_path / "trajectories" / record.task.id
    native = SimulationRun(
        id="budget-cached",
        task_id=record.task.id,
        seed=42,
        start_time="start",
        end_time="end",
        duration=1,
        messages=[],
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1),
    )
    write_json(
        directory / "validation_bm25_grep_42.json", native.model_dump(mode="json")
    )
    write_json(
        directory / "validation_bm25_grep_42.capture.json",
        {
            "input_hash": digest(
                [
                    record.task.model_dump(mode="json"),
                    workflow.trial_config(config, "bm25_grep", 42).model_dump(
                        mode="json"
                    ),
                ]
            ),
            "retry_simulation": False,
            "visible_sample": {"messages": [], "tools": [], "loss_mask": []},
        },
    )
    monkeypatch.setattr(
        build,
        "build_text_orchestrator",
        lambda *a, **kw: pytest.fail("Cached native must not be sampled again"),
    )
    monkeypatch.setattr(
        workflow,
        "review",
        lambda *a: (_ for _ in ()).throw(llm.ReviewBudgetExceeded("exhausted")),
    )
    result = workflow.run_trial(tmp_path, record, config, "bm25_grep", 42)
    assert result["reward"] == 1
    assert result["passed"] is False
    assert result["review"] is None
    assert result["infrastructure_error"] is False
    assert result["review_usage"]["model_review_inconclusive"] is True
