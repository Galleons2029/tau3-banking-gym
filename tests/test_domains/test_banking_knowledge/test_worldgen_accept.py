"""Stage H: comparisons against the official corpus, and what they refuse to claim."""

import json

import pytest

from tau3.worldgen.accept import (
    Comparison,
    GateReport,
    compare_behaviour,
    compare_text,
    official_control_sample,
    summarize_run,
    text_statistics,
)


def write_run(path, rewards, terminations):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": f"task_{i:03d}",
                        "reward_info": {"reward": reward},
                        "termination_reason": termination,
                    }
                    for i, (reward, termination) in enumerate(
                        zip(rewards, terminations, strict=True)
                    )
                ]
            }
        )
    )
    return path


def test_a_comparison_reports_its_own_distance():
    close = Comparison("pass_rate", 0.2, 0.3, tolerance=0.15)
    assert close.within and close.delta == pytest.approx(-0.1)
    far = Comparison("pass_rate", 0.9, 0.3, tolerance=0.15)
    assert not far.within


def test_an_advisory_gate_reports_without_blocking():
    report = GateReport(
        gate="text",
        comparisons=[Comparison("mean_tokens", 400.0, 250.0, tolerance=10.0)],
        advisory=True,
    )
    # The measurement is out of tolerance and still does not block: style
    # similarity is the goal, exact length agreement is not.
    assert not report.comparisons[0].within
    assert report.passed
    assert report.as_dict()["advisory"] is True


def test_a_blocking_gate_fails_when_a_comparison_is_out_of_tolerance():
    report = GateReport(
        gate="retrieval",
        comparisons=[Comparison("gold_recall_at_10", 0.9, 0.4, tolerance=0.2)],
    )
    assert not report.passed


def test_text_statistics_describe_structure_not_only_length():
    bulleted = {"a": "# Fees\n\n- Monthly fee: $0\n- Limit: $2,500\n"}
    tabular = {"a": "# Fees\n\n| Item | Value |\n|---|---|\n| Fee | $0 |\n"}
    assert text_statistics(bulleted)["bullet_line_share"] > 0
    assert text_statistics(tabular)["table_line_share"] > 0
    assert text_statistics(bulleted)["table_line_share"] == 0


def test_the_official_control_sample_spans_the_range_of_task_sizes():
    from pathlib import Path

    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_TASK_SET_PATH

    sample = official_control_sample(8, seed=42)
    assert len(sample) == 8
    assert len(set(sample)) == 8
    # Comparing against whichever tasks came first would confound difficulty
    # with task length, so the sample has to reach both ends.
    sizes = []
    for task_id in sample:
        task = json.loads(
            (Path(KNOWLEDGE_TASK_SET_PATH) / f"{task_id}.json").read_text()
        )
        sizes.append(len((task["evaluation_criteria"] or {}).get("actions") or []))
    assert max(sizes) - min(sizes) >= 3
    assert official_control_sample(8, seed=42) == sample


def test_a_control_sample_larger_than_the_corpus_is_the_whole_corpus():
    assert len(official_control_sample(500, seed=1)) == 97


def test_a_run_summary_separates_truncation_from_failure(tmp_path):
    path = write_run(
        tmp_path / "r.json",
        [1.0, 0.0, 0.0, 0.0],
        ["user_stop", "user_stop", "max_steps", "max_steps"],
    )
    summary = summarize_run(path)
    assert summary["pass_rate"] == pytest.approx(0.25)
    # Half the failures ran out of room; that is not evidence about difficulty.
    assert summary["max_steps"] == 2
    assert summary["max_steps_share"] == pytest.approx(0.5)


def test_behaviour_gate_warns_when_truncation_dominates(tmp_path):
    synthetic = write_run(
        tmp_path / "s.json", [0.0] * 4, ["max_steps"] * 3 + ["user_stop"]
    )
    official = write_run(tmp_path / "o.json", [1.0, 0.0, 0.0, 0.0], ["user_stop"] * 4)
    report = compare_behaviour(synthetic, official, tolerance=0.3)
    assert any("truncated" in note.lower() for note in report.notes)
    assert any("step budget" in note for note in report.notes)


def test_behaviour_gate_compares_the_same_agent_on_both_corpora(tmp_path):
    synthetic = write_run(tmp_path / "s.json", [1.0, 0.0, 0.0, 0.0], ["user_stop"] * 4)
    official = write_run(tmp_path / "o.json", [1.0, 1.0, 0.0, 0.0], ["user_stop"] * 4)
    report = compare_behaviour(synthetic, official, tolerance=0.3)
    comparison = report.comparisons[0]
    assert comparison.synthetic == pytest.approx(0.25)
    assert comparison.official == pytest.approx(0.5)
    assert comparison.within


def test_an_empty_run_is_not_reported_as_a_perfect_score(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"simulations": []}))
    assert summarize_run(path)["pass_rate"] == 0.0


def test_the_text_gate_is_advisory_against_the_installed_corpus(tmp_path):
    world = tmp_path / "world"
    (world / "documents").mkdir(parents=True)
    (world / "documents" / "doc_a.json").write_text(
        json.dumps({"id": "doc_a", "title": "Copper", "content": "- Fee: $0\n"})
    )
    report = compare_text(world)
    assert report.advisory and report.passed
    assert any(c.metric == "mean_tokens" for c in report.comparisons)
    assert any("Style similarity" in note for note in report.notes)


def test_a_service_error_is_excluded_rather_than_counted_as_a_failure(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": "a",
                        "reward_info": {"reward": 1.0},
                        "termination_reason": "user_stop",
                    },
                    {
                        "task_id": "b",
                        "reward_info": {"reward": 0.0},
                        "termination_reason": "user_stop",
                    },
                    # Never attempted: no reward, no messages.
                    {
                        "task_id": "c",
                        "reward_info": None,
                        "termination_reason": "infrastructure_error",
                    },
                ]
            }
        )
    )
    summary = summarize_run(path)
    # One of two scored trajectories passed; the outage does not make it a third.
    assert summary["pass_rate"] == pytest.approx(0.5)
    assert summary["scored"] == 2
    assert summary["infrastructure_errors"] == 1
    assert summary["infrastructure_error_tasks"] == ["c"]


def test_the_behaviour_gate_says_when_it_cannot_tell_noise_from_a_difference(tmp_path):
    from tau3.worldgen.accept import delta_uncertainty

    synthetic = write_run(tmp_path / "s.json", [1.0, 0.0], ["user_stop"] * 2)
    official = write_run(tmp_path / "o.json", [1.0, 1.0], ["user_stop"] * 2)
    report = compare_behaviour(synthetic, official, tolerance=0.3)
    # Two tasks a side cannot resolve a 0.3 gap, and the report has to say so
    # rather than let "passed" be read as evidence of similarity.
    assert delta_uncertainty(summarize_run(synthetic), summarize_run(official)) > 0.3
    assert any("cannot yet distinguish" in note for note in report.notes)


def test_an_all_service_error_run_is_not_a_perfect_score(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": "a",
                        "reward_info": None,
                        "termination_reason": "infrastructure_error",
                    }
                ]
            }
        )
    )
    summary = summarize_run(path)
    assert summary["scored"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["infrastructure_errors"] == 1


def test_the_behaviour_gate_may_load_the_draft_it_is_judging(tmp_path, monkeypatch):
    """A world is a draft until these gates pass, so the gate must load drafts.

    Without this the only world the behaviour layer could ever measure is one
    already published -- which is the decision the measurement is for.
    """
    import subprocess

    from tau3.worldgen.workflow import _behaviour_runs
    from tau3.worldgen.world import DRAFT_ENV_VAR, WORLD_ENV_VAR

    seen = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        seen.append(kwargs["env"])
        return Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    _behaviour_runs(tmp_path, "openai/model", ["task_1"], 1, 200)

    assert seen and len(seen) == 2
    for environment in seen:
        assert environment[WORLD_ENV_VAR] == str(tmp_path)
        assert environment[DRAFT_ENV_VAR] == "1"


def test_a_failed_behaviour_run_says_why(tmp_path, monkeypatch):
    import subprocess

    from tau3.worldgen.workflow import _behaviour_runs

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "ValueError: the world documents no business account class"

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="documents no business account class"):
        _behaviour_runs(tmp_path, "openai/model", ["task_1"], 1, 200)
