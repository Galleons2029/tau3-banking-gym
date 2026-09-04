"""Tests for rescoring trajectories with tau2.scripts.evaluate_trajectories."""

from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.run import get_tasks
from tau2.scripts import evaluate_trajectories as evaluate_trajectories_module
from tau2.scripts.evaluate_trajectories import compute_simulation_rewards

# ---- Fixtures ----


def _make_info(user_implementation: str = "user_simulator") -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation=user_implementation),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_half_duplex_sim(task_id: str) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )


# ---- Re-grading options: strict_replay, env_kwargs, fresh tasks ----


class TestRegradingOptions:
    def _capture_eval_kwargs(self, monkeypatch, results, **compute_kwargs):
        captured = []

        def fake_evaluate_simulation(**kwargs):
            captured.append(kwargs)
            return RewardInfo(reward=1.0)

        monkeypatch.setattr(
            evaluate_trajectories_module,
            "evaluate_simulation",
            fake_evaluate_simulation,
        )
        compute_simulation_rewards(results, **compute_kwargs)
        return captured

    def test_rescoring_uses_lenient_replay(self, monkeypatch):
        """Re-grading replays historical trajectories whose recorded tool
        outputs may cosmetically predate current tool code; the replay must
        not abort on output-text drift."""
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["strict_replay"] is False

    def test_rescoring_banking_passes_read_log_allowlist(self, monkeypatch):
        """banking_knowledge live grading logs golden-trajectory reads to the
        agent_discoverable_tools table via a per-task allowlist; re-grading
        must pass the same allowlist or required-read assertions silently
        stop discriminating."""
        from tau2.data_model.tasks import Action

        task = _make_task("t0")
        task.evaluation_criteria = EvaluationCriteria(
            actions=[
                Action(
                    action_id="t0_0",
                    requestor="assistant",
                    name="call_discoverable_agent_tool",
                    arguments={
                        "agent_tool_name": "get_bank_account_transactions_9173",
                        "arguments": "{}",
                    },
                )
            ]
        )
        results = Results(
            info=_make_info(),
            tasks=[task],
            simulations=[_make_half_duplex_sim("t0")],
        )
        results.info.environment_info = EnvironmentInfo(
            domain_name="banking_knowledge", policy="test policy"
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["env_kwargs"] == {
            "read_log_allowlist": {"get_bank_account_transactions_9173"}
        }

    def test_non_banking_domain_gets_no_env_kwargs(self, monkeypatch):
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["env_kwargs"] is None

    def test_fresh_tasks_reloads_task_definitions(self, monkeypatch):
        """--fresh-tasks must grade against the current data-dir task
        definitions, not the ones embedded in the results file."""
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()  # stale criteria
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_half_duplex_sim(embedded_task.id)],
        )

        captured = self._capture_eval_kwargs(monkeypatch, results, fresh_tasks=True)
        current_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        assert (
            captured[0]["task"].evaluation_criteria == current_task.evaluation_criteria
        )
        assert captured[0]["task"].evaluation_criteria != EvaluationCriteria()

    def test_embedded_tasks_used_by_default(self, monkeypatch):
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_half_duplex_sim(embedded_task.id)],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["task"].evaluation_criteria == EvaluationCriteria()
