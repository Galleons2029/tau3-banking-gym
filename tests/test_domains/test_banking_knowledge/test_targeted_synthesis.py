"""Offline behavioral tests for failure-directed business tasks and SFT accounting."""

import argparse
import json
from collections import Counter
from decimal import Decimal

import pytest

from tau3.synthesis.targeted import planning, workflow
from tau3.synthesis.targeted.budget import Budget
from tau3.synthesis.targeted.models import (
    DIFFICULTIES,
    RECIPES,
    FailureProfile,
    Finding,
    RoundConfig,
    Slot,
    SynthesisPlan,
)
from tau3.synthesis.targeted.recipes import compile_world, verify_mechanisms
from tau3.worldgen.v2 import pipeline
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.environment import get_environment
from tau3.worldgen.v2.pipeline import initial_database, write_json
from tau3.worldgen.v2.runtime import OperationRuntime, digest, goal_errors


def slot(recipe="discovery", difficulty="1-4", **kwargs):
    return Slot(
        index=0,
        recipe=recipe,
        difficulty=difficulty,
        origin="current",
        seed=42000,
        **kwargs,
    )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_all_mechanisms_reject_targeted_errors(recipe, difficulty):
    allocation = slot(recipe, difficulty)
    spec = compile_world(allocation, "offline")
    result = verify_mechanisms(spec, allocation)
    assert result["status"] == "PASS", result
    assert all(result["counterexamples"].values())
    assert result["business_operations"] > 0


@pytest.fixture
def world(tmp_path, monkeypatch):
    allocation = slot("handoff", "15-19", secondary="escalation")
    spec = compile_world(allocation, "world")
    # This fixture replaces semantic category admission ONLY for offline tests.
    # It is never exported as real model validation evidence.
    monkeypatch.setattr(pipeline, "admitted", lambda *args: True)
    root = tmp_path / "world"
    assert pipeline.build(root, spec)["status"] == "PASS"
    pipeline.publish(root)
    return root, spec


def test_queries_paginate_project_and_never_mutate_business_state(world):
    root, spec = world
    env = get_environment(root, retrieval_variant="bm25_grep")
    assert "get_customer_records" not in env.tools.tools
    assert "KB_list_documents" not in env.tools.tools
    assert "KB_read_documents" not in env.tools.tools
    assert "KB_search" in env.tools.tools and "grep" in env.tools.tools
    before = env.tools.db.model_copy(deep=True)
    public = env.tools.public_customer_records(spec.categories[0].scenarios[0].user_id)
    assert public == before.tables
    assert env.tools.db == before
    ops = [o for o in spec.categories[0].operations if o.kind == "read"]
    runtime = OperationRuntime(spec, before.model_copy(deep=True))
    op = ops[0]
    alias = next(a for a, oid in runtime.aliases.items() if oid == op.id)
    args = {
        "user_id": "missing",
        "product_id": spec.categories[0].products[0].id,
        "offset": 0,
    }
    with pytest.raises(ValueError, match="customer"):
        runtime.execute(alias, args)
    args["user_id"] = spec.categories[0].scenarios[0].user_id
    args["offset"] = -1
    with pytest.raises(ValueError):
        runtime.execute(alias, args)
    assert runtime.db == before


def test_blind_projection_uses_public_queries(world, monkeypatch):
    from tau3.worldgen.v2.blind import public_problem
    from tau3.worldgen.v2.environment import WorldTools

    root, spec = world
    monkeypatch.setattr(
        WorldTools,
        "get_customer_records",
        lambda *_: pytest.fail("Private snapshot shortcut"),
    )
    task_id = "task_" + spec.categories[0].scenarios[0].id
    problem = public_problem(root, task_id)
    assert problem["records"] == initial_database(spec).tables
    assert not {"goals", "gold", "steps", "evaluation_criteria"} & problem.keys()


def test_legal_alternative_order_is_accepted(world):
    _, spec = world
    scenario = spec.categories[0].scenarios[0]
    initial = initial_database(spec)
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    ops = {o.id: o for o in spec.categories[0].operations}
    aliases = {op: alias for alias, op in runtime.aliases.items()}
    for step in reversed(scenario.steps):
        op = ops[step.operation]
        runtime.execute(aliases[step.operation], step.arguments, op.actor)
    assert not goal_errors(initial, runtime.db, scenario.goals, scenario.user_id, spec)


def test_calibration_entry_dispatches_controls_and_returns_real_messages(
    world, monkeypatch
):
    from tau3.data_model.simulation import SimulationRun, TerminationReason
    from tau3.data_model.tasks import Task
    from tau3.worldgen.v2.calibration import controls, trajectory
    from tau3.worldgen.v2.environment import WorldTools

    root, _ = world
    monkeypatch.setattr(
        WorldTools,
        "get_customer_records",
        lambda *_: pytest.fail("Calibration answer used a private DB snapshot"),
    )
    tasks = {
        task.id: task
        for path in (root / "tasks").glob("*.json")
        for task in [Task.model_validate_json(path.read_text())]
    }
    cases = controls(root)
    assert all("Service account" not in case["answer"] for case in cases)
    assert any(c["id"].endswith("missing_user_execution") for c in cases)
    assert any(
        c["id"].endswith("legal_alternative_order") and c["expected"] == 1
        for c in cases
    )
    for case in cases:
        messages = trajectory(
            root, tasks[case["task_id"]], case["actions"], case["answer"]
        )
        simulation = SimulationRun(
            id=case["id"],
            task_id=case["task_id"],
            seed=0,
            start_time="2026-09-14T00:00:00",
            end_time="2026-09-14T00:00:01",
            duration=1,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=messages,
        )
        assert simulation.messages[0].role == "user"
        assert simulation.messages[-1].role == "assistant"


def test_independent_training_readiness_keeps_hard_tasks_but_requires_all_proofs(
    world, tmp_path, monkeypatch
):
    from tau3.worldgen.v2 import blind, readiness
    from tau3.worldgen.v2.calibration import controls

    root, spec = world
    settings = RoundConfig().settings
    settings_path = tmp_path / "settings.json"
    write_json(settings_path, settings.model_dump())
    task_id = "task_" + spec.categories[0].scenarios[0].id
    identity = {
        "artifacts": pipeline.artifact_hashes(root),
        "implementation": pipeline.implementation_hash(),
        "settings": digest(settings.model_dump()),
    }
    b, c, online = (tmp_path / name for name in ("blind", "calibration", "online"))
    # Structural gate test only: no fake audit evidence leaves this fixture.
    verified = []
    monkeypatch.setattr(
        blind, "verify_saved_blind", lambda *args: verified.append(args)
    )
    write_json(
        b / "report.json",
        {
            "identity": identity,
            "status": "PASS",
            "task_ids": [task_id],
            "results": [
                {"task_id": task_id, "model": model, "status": "PASS"}
                for model in settings.agent_models
            ],
        },
    )
    suite = controls(root)
    calibrated = {
        "identity": identity,
        "status": "PASS",
        "suite_hash": digest(suite),
        "results": [
            {"case_id": case["id"], "model": model, "status": "PASS"}
            for case in suite
            for model in settings.agent_models
        ],
    }
    write_json(c / "report.json", calibrated)
    output = tmp_path / "readiness.json"
    legacy = readiness.assess(root, b, c, online, output, settings_path)
    assert legacy["status"] == "INCONCLUSIVE"
    result = readiness.assess(
        root, b, c, online, output, settings_path, admission_mode="targeted_training"
    )
    assert result["status"] == "PASS" and set(result["checks"]) == {
        "blind",
        "calibration",
    }
    assert len(verified) == 4
    assert readiness.check_readiness(root, output, settings_path) == result
    calibrated["results"].pop()
    write_json(c / "report.json", calibrated)
    assert (
        readiness.assess(
            root,
            b,
            c,
            online,
            output,
            settings_path,
            admission_mode="targeted_training",
        )["status"]
        == "INCONCLUSIVE"
    )


def test_targeted_admission_never_samples_teachers_to_select_tasks(
    tmp_path, monkeypatch
):
    from tau3.worldgen.v2 import blind, calibration, readiness, rollouts

    monkeypatch.setattr(blind, "run_blind", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(
        calibration, "run_calibration", lambda *a, **k: {"status": "PASS"}
    )
    monkeypatch.setattr(
        rollouts,
        "run_matrix",
        lambda *a, **k: pytest.fail("Teacher success filtered admission"),
    )
    modes = []

    def assess(*args, **kwargs):
        modes.append(kwargs["admission_mode"])
        return {"status": "PASS"}

    monkeypatch.setattr(readiness, "assess", assess)
    workflow.qualify(tmp_path, RoundConfig(), tmp_path / "world", tmp_path / "audit")
    assert modes == ["targeted_training"]
    assert not (tmp_path / "audit/matrix-started.json").exists()
    assert not (tmp_path / "budget.json").exists()


def test_same_shape_date_pairs_flip_only_policy_input():
    first = slot("policy")
    second = first.model_copy(update={"seed": first.seed + 1})
    specs = [compile_world(s, "paired") for s in [first, second]]
    tables = [initial_database(s).tables for s in specs]
    name = next(t for t in tables[0] if t.endswith("_cases"))
    rid = next(iter(tables[0][name]))
    a, b = tables[0][name][rid], tables[1][name][rid]
    assert {k for k in a if a[k] != b[k]} == {"reported_on"}
    assert [
        s.categories[0].scenarios[0].goals[0].fields["liability"] for s in specs
    ] == ["25", "250"]


def test_normalized_business_dedup_ignores_renaming():
    s = slot()
    a = verify_mechanisms(compile_world(s, "a"), s)
    b = verify_mechanisms(compile_world(s, "b"), s)
    assert a["business_fingerprint"] == b["business_fingerprint"]


def test_raw_missing_tool_hypothesis_does_not_become_a_causal_quota(
    tmp_path, monkeypatch
):
    findings = [
        Finding(
            label=label,
            source="report",
            quote="evidence",
            observation="symptom",
            interpretation="cause",
            confidence="medium",
            severity="high",
        )
        for label in ["F1", "F2"]
    ]
    profile = FailureProfile(
        round_id="r",
        base_model="b",
        report_hash="h",
        findings=findings,
        task_observations={"t": {"hypothesis_labels": ["F1"]}},
        task_attributions={"t": [{"label": "F2", "reason": "wrong band"}]},
    )
    write_json(tmp_path / "profile.json", profile.model_dump())
    monkeypatch.setattr(
        planning,
        "ask",
        lambda *a, **k: {
            "strategies": [
                {"recipe": "discovery", "evidence_labels": ["F1"]},
                {"recipe": "policy", "evidence_labels": ["F2"]},
            ]
        },
    )
    plan = planning.make_plan(tmp_path, RoundConfig(num_tasks=20))
    assert plan.weights["discovery"] == 0
    assert plan.weights["policy"] == 1


def test_attribution_requires_exact_failed_trial_evidence(tmp_path, monkeypatch):
    findings = [
        Finding(
            label="F2",
            source="report",
            quote="evidence",
            observation="symptom",
            interpretation="cause",
            confidence="medium",
            severity="high",
        )
    ]
    observations = {
        "t": {
            "hypothesis_labels": ["F1"],
            "trials": [
                {
                    "reward": 0,
                    "source": "run.json",
                    "source_hash": "h",
                    "hypothesis_labels": ["F1"],
                    "differences": [{"fields": ["amount"]}],
                }
            ],
        }
    }
    response = {
        "tasks": [
            {
                "task_id": "t",
                "causes": [
                    {
                        "label": "F2",
                        "reason": "wrong band",
                        "confidence": "medium",
                        "evidence_ids": [],
                    }
                ],
            }
        ]
    }

    def ask(*args, **kwargs):
        assert "hypothesis_labels" not in json.dumps(args[5])
        cause = response["tasks"][0]["causes"][0]
        if not cause["evidence_ids"]:
            cause["evidence_ids"] = [args[5]["tasks"]["t"]["0"]["evidence"][0]["id"]]
        return response

    monkeypatch.setattr(planning, "ask", ask)
    result = planning.attribute_tasks(
        tmp_path, RoundConfig(), observations, findings, []
    )
    assert result["t"][0]["evidence"][0]["source_hash"] == "h"
    response["tasks"][0]["causes"][0]["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="Unknown attribution evidence ID"):
        planning.attribute_tasks(tmp_path, RoundConfig(), observations, findings, [])


def test_attribution_batches_bound_context_without_losing_trial_addresses():
    tasks = [
        ("t", {"trials": [{"reward": 0, "evidence": "字" * 30} for _ in range(5)]})
    ]
    batches = list(planning.attribution_batches(tasks, 180))
    assert len(batches) == 5
    assert [key for b in batches for key in b["t"]] == ["0", "1", "2", "3", "4"]
    assert all(len(json.dumps(b, ensure_ascii=False).encode()) <= 180 for b in batches)
    with pytest.raises(ValueError, match="One trial exceeds"):
        list(planning.attribution_batches(tasks, 20))


def test_public_business_names_do_not_expose_curriculum_taxonomy():
    from tau3.synthesis.targeted.recipes import PUBLIC_SERVICES

    for recipe in RECIPES:
        spec = compile_world(slot(recipe), "private-labels")
        category = spec.categories[0]
        cases = next(
            rows
            for table, rows in initial_database(spec).tables.items()
            if table.endswith("_cases")
        )
        assert {r["service_kind"] for r in cases.values()} == {PUBLIC_SERVICES[recipe]}
        assert all(not o.id.endswith("_resolve_" + recipe) for o in category.operations)
        assert category.scenarios[0].structural_family == recipe


def test_planner_freezes_exact_quota_totals_and_supported_evidence(
    tmp_path, monkeypatch
):
    findings = [
        Finding(
            label=labels[0],
            source="report",
            quote="evidence",
            observation="missing",
            interpretation="hypothesis",
            confidence="medium",
            severity="high",
        )
        for labels in RECIPES.values()
    ]
    profile = FailureProfile(
        round_id="round1", base_model="base", report_hash="hash", findings=findings
    )
    write_json(tmp_path / "profile.json", profile.model_dump())
    monkeypatch.setattr(
        planning,
        "ask",
        lambda *a, **k: {
            "strategies": [
                {
                    "recipe": r,
                    "evidence_labels": [labels[0]],
                    "parameters": {},
                    "expected_behavior": "correct operation",
                }
                for r, labels in RECIPES.items()
            ]
            + [{"recipe": "unknown", "evidence_labels": []}],
            "extensions": [],
        },
    )
    config = RoundConfig()
    plan = planning.make_plan(tmp_path, config)
    assert len(plan.slots) == 3000 and len(plan.pilot) == 20
    assert Counter(s.origin for s in plan.slots) == {
        "current": 2100,
        "replay": 600,
        "explore": 300,
    }
    assert Counter(s.difficulty for s in plan.slots) == planning.allocate(
        3000, DIFFICULTIES
    )
    assert all(s.difficulty != "1-4" for s in plan.slots if s.secondary)
    assert len(plan.backlog) == 1
    assert plan == planning.make_plan(tmp_path, config)
    assert sum(s.difficulty in {"15-19", "20-29"} for s in plan.pilot) >= 6


def test_budget_counts_failed_calls_and_restarts(tmp_path):
    config = RoundConfig(max_total_calls=2, max_total_tokens=100000)
    budget = Budget(tmp_path, config)
    record = budget.reserve({"max_tokens": 10}, "analysis")
    budget.complete(record, error=TimeoutError())
    first = json.loads((tmp_path / "budget.json").read_text())
    assert first["calls"] == 1 and first["tokens"] > 0
    budget = Budget(tmp_path, config)
    record = budget.reserve({"max_tokens": 10}, "teacher")
    budget.complete(record, {"usage": {"total_tokens": 5}})
    with pytest.raises(AuditIncomplete):
        budget.reserve({"max_tokens": 10}, "teacher")
    assert (
        json.loads((tmp_path / "budget.json").read_text())["tokens"]
        == first["tokens"] + 5
    )


def test_truncated_json_cannot_pass_through_budget_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tau3.utils import llm_utils

    payload = {
        "choices": [
            {"finish_reason": "length", "message": {"content": '{"passed":true}'}}
        ],
        "usage": {"total_tokens": 12},
    }
    monkeypatch.setattr(
        llm_utils,
        "completion",
        lambda **kw: SimpleNamespace(model_dump=lambda **kw: payload),
    )
    with Budget(tmp_path, RoundConfig()).installed("world_build"):
        with pytest.raises(AuditIncomplete, match="finish normally"):
            llm_utils.completion(model="test", messages=[])
    records = [
        json.loads(p.read_text()) for p in (tmp_path / "physical_calls").glob("*.json")
    ]
    assert len(records) == 1 and records[0]["response"] == payload


def test_sft_excludes_private_reasoning_and_preserves_visible_context():
    sample = {
        "task_id": "t",
        "retrieval_config": "bm25_grep",
        "tools": [],
        "messages": [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "done", "reasoning_content": "private"},
        ],
        "loss_mask": [0, 1],
    }
    result = workflow.clean_sample(sample)
    assert result["messages"][1] == {"role": "assistant", "content": "done"}
    assert sample["messages"][1]["reasoning_content"] == "private"
    with pytest.raises(ValueError):
        workflow.clean_sample({**sample, "loss_mask": [1, 1]})


def test_report_keeps_incomplete_tasks_in_denominator(tmp_path):
    profile = {"source": "test"}
    write_json(tmp_path / "profile.json", profile)
    slots = [slot().model_copy(update={"index": i}) for i in range(3)]
    plan = SynthesisPlan(
        profile_hash=digest(profile), proposal={}, weights={}, slots=slots, pilot=slots
    )
    write_json(tmp_path / "plan.json", plan.model_dump())
    for task_index in range(2):
        for trial in range(4):
            write_json(
                tmp_path
                / f"formal/slots/{task_index:06d}/teacher/{trial}/quality.json",
                {
                    "status": "COMPLETE",
                    "environment_success": task_index == 0,
                    "sft_qualified": task_index == 0,
                },
            )
    result = workflow.report(tmp_path, RoundConfig())
    assert result["teacher_pass@4"] == 1 / 3
    assert result["four_trial_coverage"] == 2 / 3
    assert result["qualified_samples"] == 4
    assert result["hard_tasks"] == 1 and result["incomplete_tasks"] == 1
    assert result["status"] == "INCOMPLETE"


def test_targeted_cli_defaults_and_required_identity():
    from tau3.synthesis.cli import add_synthesis_parser

    parser = argparse.ArgumentParser()
    add_synthesis_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            "synthesize",
            "targeted",
            "collect",
            "--round-dir",
            "/tmp/round",
            "--config",
            "config.yaml",
        ]
    )
    assert args.targeted_stage == "collect" and args.phase == "formal"


def test_no_single_model_or_changed_trial_count():
    with pytest.raises(ValueError):
        RoundConfig(trials=3)
    with pytest.raises(ValueError):
        RoundConfig(reviewer_model="openai/GLM-5.3-Flash")
    assert Decimal(".1") + Decimal(".2") == Decimal(".3")


def test_targeted_default_concurrency_is_128_and_supports_256():
    config = RoundConfig()
    assert config.workers == config.llm_concurrency == 128
    assert RoundConfig(llm_concurrency=256).llm_concurrency == 256
    with pytest.raises(ValueError):
        RoundConfig(llm_concurrency=257)


def test_supervisor_telemetry_io_errors_do_not_stop_stage_dispatch(
    tmp_path, monkeypatch
):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(__file__).resolve().parents[3] / "scripts/run_targeted_synthesis.py"
    spec = importlib.util.spec_from_file_location("supervisor_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "attribution-progress.json").write_text("{}")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "attribution-progress.json":
            raise OSError(5, "simulated shared filesystem read failure")
        return original_read(path, *args, **kwargs)

    original_write = pipeline.write_json
    failed = []

    def write(path, value):
        if path.name == "run-state.json" and not failed:
            failed.append(True)
            raise OSError(5, "simulated heartbeat write failure")
        return original_write(path, value)

    stages = []

    def popen(command, **kwargs):
        stages.append(command[5])
        return SimpleNamespace(pid=1, poll=lambda: 0)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(pipeline, "write_json", write)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--round-dir",
            str(tmp_path),
            "--config",
            "config.yaml",
            "--report",
            "report.md",
            "--round-id",
            "r",
            "--base-model",
            "b",
        ],
    )
    assert module.main() == 0
    assert stages == [
        "analyze",
        "plan",
        "pilot",
        "generate",
        "collect",
        "export",
        "report",
    ]


def test_provider_timeout_is_serialized_as_data_for_world_workers():
    import pickle

    from litellm.exceptions import Timeout

    def timeout():
        raise Timeout(message="test", model="test", llm_provider="openai")

    result = workflow.isolated_worker(timeout)
    assert pickle.loads(pickle.dumps(result)) == result
    assert result["status"] == "INCONCLUSIVE" and "Timeout" in result["reason"]


@pytest.mark.parametrize(
    "stage,expected_attempts", [("teacher", 3), ("calibration", 1)]
)
def test_backend_auth_recovery_keeps_request_and_accounts_every_attempt(
    tmp_path, monkeypatch, stage, expected_attempts
):
    from tau3.worldgen.v2.audit import backend_auth_failure, transport_failure

    class BackendError(Exception):
        status_code = 401

    error = BackendError(
        "Backend error (gemini_mock): brpc [example:8080][E401]user auth failed"
    )
    assert backend_auth_failure(error) and transport_failure(error)
    assert not transport_failure(BackendError("Invalid API key"))
    monkeypatch.setattr("tau3.synthesis.targeted.budget.time.sleep", lambda _: None)
    seen = []

    def fail(**kwargs):
        seen.append(kwargs)
        raise error

    config = RoundConfig()
    config.settings.audit_transport_attempts = 3
    with pytest.raises(BackendError):
        Budget(tmp_path, config).call(stage, fail, seed=123, messages=["fixed"])
    assert len(seen) == expected_attempts
    assert all(request == seen[0] for request in seen)
    assert seen[0]["seed"] == 123
    assert (
        json.loads((tmp_path / "budget.json").read_text())["calls"] == expected_attempts
    )
    records = [
        json.loads(p.read_text()) for p in (tmp_path / "physical_calls").glob("*.json")
    ]
    assert len(records) == expected_attempts
    assert all(record["status"] == "INCONCLUSIVE" for record in records)


def test_backend_auth_recovery_returns_the_first_success(tmp_path, monkeypatch):
    class BackendError(Exception):
        status_code = 401

    class Reply:
        def model_dump(self, **kwargs):
            return {
                "choices": [{"finish_reason": "stop"}],
                "usage": {"total_tokens": 1},
            }

    attempts = []

    def transport(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise BackendError(
                "Backend error (gemini_mock): brpc [E401]user auth failed"
            )
        return Reply()

    monkeypatch.setattr("tau3.synthesis.targeted.budget.time.sleep", lambda _: None)
    config = RoundConfig()
    config.settings.audit_transport_attempts = 3
    assert isinstance(
        Budget(tmp_path, config).call("teacher", transport, seed=7), Reply
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]


@pytest.mark.parametrize("missing_explanation", [False, True])
def test_quality_protocol_repair_is_bounded_and_keeps_negative_verdicts(
    tmp_path, missing_explanation
):
    from tau3.synthesis.targeted.quality import review_quality

    verdict = {
        name: True
        for name in workflow.QualityReview.model_fields
        if name != "explanation"
    }
    verdict["fact_consistent"] = False
    if not missing_explanation:
        verdict["explanation"] = "Unsupported final claim"
    calls = []

    def ask(*args, evidence):
        calls.append(args)
        evidence.update(path=f"call-{len(calls)}", hash=str(len(calls)))
        return (
            verdict
            if len(calls) == 1
            else {**verdict, "explanation": "Unsupported final claim"}
        )

    evidence, protocol = {}, {}
    review = review_quality(
        tmp_path, RoundConfig(), {"conversation": ["original"]}, evidence, protocol, ask
    )
    assert not review.passed
    assert len(calls) == 1 + missing_explanation
    assert evidence["path"] == f"call-{len(calls)}"
    if missing_explanation:
        assert protocol["attempts"][0]["status"] == "INVALID_SCHEMA"
        assert calls[1][2] == "quality_protocol_repair"
        assert calls[1][5]["original_review"] == {"conversation": ["original"]}


def test_quality_protocol_repair_cannot_retry_until_accepted(tmp_path):
    from pydantic import ValidationError

    from tau3.synthesis.targeted.quality import review_quality

    calls = []

    def ask(*args, evidence):
        calls.append(args)
        return {}

    protocol = {}
    with pytest.raises(ValidationError):
        review_quality(tmp_path, RoundConfig(), {}, {}, protocol, ask)
    assert len(calls) == 2
    assert all(p["status"] == "INVALID_SCHEMA" for p in protocol["attempts"])


def test_one_provider_timeout_does_not_destroy_other_attribution_batches(
    tmp_path, monkeypatch
):
    from litellm.exceptions import Timeout

    def batch(root, config, public, findings, **kwargs):
        index = int(next(iter(public["task"])))
        if index == 1:
            raise Timeout(message="test", model="test", llm_provider="openai")
        return {"task": [{"trial": index}]}

    monkeypatch.setattr(planning, "attribute_batch", batch)
    observations = {
        "task": {"trials": [{"reward": 0, "evidence": "a" * 7000} for _ in range(4)]}
    }
    with pytest.raises(AuditIncomplete, match="1 attribution batches"):
        planning.attribute_tasks(
            tmp_path, RoundConfig(analysis_batch_bytes=10000), observations, [], []
        )
    progress = json.loads((tmp_path / "attribution-progress.json").read_text())
    assert progress["completed_batches"] == 3 and progress["inconclusive_batches"] == 1


def test_structured_parallel_calls_do_not_mutate_global_transport(
    tmp_path, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tau3.synthesis.targeted.budget import ask
    from tau3.utils import llm_utils

    barrier = threading.Barrier(4, timeout=5)

    class Reply:
        def __init__(self, value):
            self.value = value

        def model_dump(self, **kwargs):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.value)},
                    }
                ],
                "usage": {"total_tokens": 1},
            }

    def completion(**kwargs):
        barrier.wait()
        return Reply(json.loads(kwargs["messages"][-1]["content"]))

    monkeypatch.setattr(llm_utils, "completion", completion)
    config = RoundConfig()

    def run(index):
        return ask(
            tmp_path,
            config,
            "attribution",
            config.teacher_model,
            "JSON",
            {"index": index},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(4))) == [{"index": i} for i in range(4)]
    assert llm_utils.completion is completion
    assert json.loads((tmp_path / "budget.json").read_text())["calls"] == 4


def test_json_mode_recovery_preserves_failed_attempt_and_counts_both_calls(
    tmp_path, monkeypatch
):
    from tau3.synthesis.targeted.budget import ask
    from tau3.utils import llm_utils

    class Reply:
        def __init__(self, constrained):
            self.constrained = constrained

        def model_dump(self, **kwargs):
            return {
                "choices": [
                    {
                        "finish_reason": "length" if self.constrained else "stop",
                        "message": {
                            "content": "\n" * 20 if self.constrained else '{"ok":true}'
                        },
                    }
                ],
                "usage": {"total_tokens": 20},
            }

    monkeypatch.setattr(
        llm_utils,
        "completion",
        lambda **kwargs: Reply("response_format" in kwargs["extra_body"]),
    )
    config = RoundConfig()
    args = (
        tmp_path,
        config,
        "attribution",
        config.teacher_model,
        "Return JSON",
        {"task": "test"},
    )
    with pytest.raises(AuditIncomplete):
        ask(*args)
    assert ask(*args, recovery_attempt=1) == {"ok": True}
    assert ask(*args, recovery_attempt=1) == {"ok": True}
    with pytest.raises(AuditIncomplete, match="Previous structured request"):
        ask(*args)
    records = [
        json.loads(p.read_text()) for p in (tmp_path / "requests").glob("*.json")
    ]
    assert sorted(r["status"] for r in records) == ["COMPLETE", "RESERVED"]
    assert json.loads((tmp_path / "budget.json").read_text())["calls"] == 2


def test_attribution_dispatches_concurrently_and_merges_in_input_order(
    tmp_path, monkeypatch
):
    import threading
    import time

    barrier = threading.Barrier(4, timeout=5)

    def fake_batch(root, config, public, findings):
        index = int(next(iter(public["task"])))
        barrier.wait()
        time.sleep((3 - index) * 0.01)
        return {"task": [{"trial": index}]}

    monkeypatch.setattr(planning, "attribute_batch", fake_batch)
    observations = {
        "task": {"trials": [{"reward": 0, "evidence": "a" * 7000} for _ in range(4)]}
    }
    result = planning.attribute_tasks(
        tmp_path, RoundConfig(analysis_batch_bytes=10000), observations, [], []
    )
    assert result == {"task": [{"trial": i} for i in range(4)]}
    progress = json.loads((tmp_path / "attribution-progress.json").read_text())
    assert progress["workers"] == progress["completed_batches"] == 4
    assert progress["llm_concurrency"] == 128 and progress["status"] == "COMPLETE"


def test_physical_request_pool_limits_calls_and_releases_failed_calls(
    tmp_path, monkeypatch
):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from tau3.utils import llm_utils
    from tau3.utils.llm_concurrency import RequestPool

    class Reply:
        def model_dump(self, **kwargs):
            return {
                "choices": [{"finish_reason": "stop"}],
                "usage": {"total_tokens": 1},
            }

    active = peak = 0
    guard = threading.Lock()

    def fake_completion(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            if kwargs.get("fail"):
                raise TimeoutError("simulated")
            return Reply()
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    config = RoundConfig(llm_concurrency=3)
    with Budget(tmp_path, config).installed("analysis"):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: llm_utils.completion(max_tokens=10), range(16)))
        with pytest.raises(TimeoutError):
            llm_utils.completion(max_tokens=10, fail=True)
    state = RequestPool(tmp_path / "llm_pool", 3).snapshot()
    assert peak == state["peak"] == 3
    assert not state["active"]
    assert state["admitted"] == state["released"] == 17
    assert json.loads((tmp_path / "budget.json").read_text())["calls"] == 17


def test_unrecognized_planner_evidence_is_quarantined_not_scheduled(
    tmp_path, monkeypatch
):
    f = Finding(
        label="F8",
        source="report",
        quote="evidence",
        observation="missing",
        interpretation="hypothesis",
        confidence="high",
        severity="high",
    )
    profile = FailureProfile(
        round_id="r", base_model="base", report_hash="h", findings=[f]
    )
    write_json(tmp_path / "profile.json", profile.model_dump())
    monkeypatch.setattr(
        planning,
        "ask",
        lambda *a, **k: {
            "strategies": [
                {"recipe": "boundary", "evidence_labels": ["F8", "escalation"]}
            ],
            "extensions": [],
        },
    )
    plan = planning.make_plan(tmp_path, RoundConfig(num_tasks=20))
    assert plan.weights["boundary"] > 0
    assert plan.backlog[0]["unsupported_evidence"] == ["escalation"]


@pytest.mark.parametrize("passed_trials", [set(), {1}, {0, 1, 2, 3}])
def test_four_teacher_trials_run_even_after_success_and_resume_without_resampling(
    tmp_path, monkeypatch, passed_trials
):
    from types import SimpleNamespace

    from tau3.synthesis.targeted.models import TaskProvenance

    allocation = slot()
    spec = compile_world(allocation, "collect")
    scenario = spec.categories[0].scenarios[0]
    runtime = OperationRuntime(spec, initial_database(spec))
    task_payload = pipeline.task_payload(
        spec, scenario, [], runtime.aliases, runtime.db
    )
    world_path = tmp_path / "world"
    write_json(world_path / "tasks" / f"{task_payload['id']}.json", task_payload)
    provenance = TaskProvenance(
        task_id=task_payload["id"],
        world=str(world_path),
        world_hash="world",
        task_hash="task",
        plan_hash="plan",
        business_fingerprint="business",
        slot=allocation,
        candidate=0,
        reference_actions=4,
        business_operations=1,
        checks={},
        status="VALID",
    )
    monkeypatch.setattr(workflow, "admitted_record", lambda *a: provenance)
    config = RoundConfig()
    monkeypatch.setattr(workflow, "configure", lambda *a: config.settings)
    monkeypatch.setattr(
        workflow,
        "SimulationRun",
        SimpleNamespace(
            model_validate=lambda x: SimpleNamespace(
                messages=[],
                passed=x["passed"],
                model_dump=lambda **kw: {"messages": []},
            )
        ),
    )
    monkeypatch.setattr(workflow, "successful", lambda sim: sim.passed)
    calls = []

    def capture(task, settings, model, directory, seed, binding):
        trial = int(directory.name)
        calls.append((trial, seed))
        sample = {
            "task_id": task.id,
            "retrieval_config": "bm25_grep",
            "tools": [],
            "messages": [{"role": "assistant", "content": "Review complete."}],
            "loss_mask": [1],
        }
        write_json(directory / "capture.json", {"sample": sample})
        return {"simulation": {"passed": trial in passed_trials}}

    monkeypatch.setattr(workflow, "capture_trial", capture)
    monkeypatch.setattr(
        workflow,
        "ask",
        lambda *a, **kw: {
            "fact_consistent": True,
            "user_compliant": True,
            "handoff_correct": True,
            "no_private_leakage": True,
            "final_answer_correct": True,
            "explanation": "valid",
        },
    )
    results = workflow.collect_slot(
        str(tmp_path), config.model_dump(), allocation.model_dump(), "formal", "plan"
    )
    assert len(calls) == len(results) == 4
    assert len({seed for _, seed in calls}) == 4
    assert sum(r["sft_qualified"] for r in results) == len(passed_trials)
    assert all(r["status"] == "COMPLETE" for r in results)
    again = workflow.collect_slot(
        str(tmp_path), config.model_dump(), allocation.model_dump(), "formal", "plan"
    )
    assert results == again and len(calls) == 4
