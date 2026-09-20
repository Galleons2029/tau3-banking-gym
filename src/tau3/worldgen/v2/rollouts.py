"""Run the configured Gemini/GLM comparison without implicit model fallbacks."""

import json
import os
from pathlib import Path

from tau3.worldgen.v2.pipeline import (
    artifact_hashes,
    check_certificate,
    implementation_hash,
    write_json,
)
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import load_settings


def run_rollouts(
    root: Path,
    output: Path,
    retrieval: str = "bm25",
    num_tasks: int = 5,
    num_trials: int = 1,
    max_steps: int | None = None,
    settings_path: Path | None = None,
    task_ids: list[str] | None = None,
    all_tasks: bool = False,
) -> dict:
    """Execute a bounded two-model matrix and preserve incomplete results as such.

    This command owns its worker process's configured world and settings. Separate
    world matrices should run in separate processes, as with the Tau domain CLI.
    """
    from tau3.data_model.simulation import Results, TextRunConfig
    from tau3.domains.banking_synth.environment import get_environment, get_tasks
    from tau3.registry import registry
    from tau3.runner.batch import run_tasks
    from tau3.worldgen.v2.behaviour import summarize

    max_steps = (
        max_steps
        if max_steps is not None
        else load_settings(settings_path).simulation_max_steps
    )
    if min(num_tasks, num_trials, max_steps) < 1:
        raise ValueError("Task/trial/step budgets must be positive")
    check_certificate(root)
    os.environ["TAU3_SYNTH_WORLD"] = str(root.resolve())
    if settings_path:
        os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(settings_path.resolve())
    settings = load_settings(settings_path)
    if "banking_synth" not in registry.get_domains():
        registry.register_domain(get_environment, "banking_synth")
    available = {t.id: t for t in get_tasks()}
    if task_ids is not None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or not set(task_ids) <= available.keys()
        ):
            raise ValueError("Empty, duplicate or unknown task IDs")
        tasks = [available[tid] for tid in task_ids]
        num_tasks = len(tasks)
    elif all_tasks:
        tasks = list(available.values())
        num_tasks = len(tasks)
    else:
        tasks = list(available.values())[:num_tasks]
    if len(tasks) != num_tasks:
        raise ValueError("Requested more tasks than the world contains")
    config = {
        "artifacts": artifact_hashes(root),
        "implementation": implementation_hash(),
        "settings_hash": digest(settings.model_dump()),
        "retrieval": retrieval,
        "num_tasks": num_tasks,
        "num_trials": num_trials,
        "max_steps": max_steps,
        "task_ids": [t.id for t in tasks],
    }
    path = output / "matrix.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Stale rollout configuration; choose a fresh output directory")
    write_json(path, config)
    expected = {t.id: t.model_dump(mode="json") for t in tasks}
    reports = []
    for model in settings.agent_models:
        result_path = output / f"results_{digest(model)[:12]}.json"
        if result_path.exists():
            result = Results.load(result_path)
            if (
                result.info.world_artifact_hash != digest(config["artifacts"])
                or result.info.world_implementation_hash != config["implementation"]
            ):
                raise ValueError("Stale cached rollout")
        complete = False
        if result_path.exists():
            try:
                summarize(result, expected)
                complete = True
            except ValueError:
                pass
        if not complete:
            run_config = TextRunConfig(
                domain="banking_synth",
                llm_agent=model,
                llm_user=settings.user_model,
                llm_args_agent=settings.llm_args,
                llm_args_user=settings.llm_args,
                num_trials=num_trials,
                max_steps=max_steps,
                timeout=settings.simulation_timeout,
                max_errors=5,
                max_concurrency=2,
                max_retries=0,
                auto_resume=True,
                retrieval_config=retrieval,
                seed=42,
                log_level="ERROR",
            )
            result = run_tasks(
                run_config, tasks, save_path=result_path, console_display=False
            )
        try:
            summary = summarize(result, expected)
            if summary["tasks"] != num_tasks:
                raise ValueError("Incomplete task membership")
            report = {
                "model": model,
                "status": "PASS",
                "completion": "complete",
                **summary,
            }
        except ValueError as exc:
            report = {"model": model, "status": "INCONCLUSIVE", "reason": str(exc)}
        report["results"] = str(result_path)
        report["cells"] = [
            {
                "task_id": s.task_id,
                "trial": s.trial,
                "termination": s.termination_reason.value,
                "reward": s.reward_info.reward if s.reward_info else None,
                "judge_inconclusive": bool(
                    s.reward_info
                    and any(
                        c.justification.startswith("INCONCLUSIVE:")
                        for c in s.reward_info.nl_assertions or []
                    )
                ),
            }
            for s in result.simulations
        ]
        reports.append(report)
        write_json(
            output / "summary.json",
            {
                "status": "PASS"
                if len(reports) == len(settings.agent_models)
                and all(r["status"] == "PASS" for r in reports)
                else "INCONCLUSIVE",
                "runs": reports,
                "scope": "Execution completeness; pass rate is reported separately",
            },
        )
    return json.loads((output / "summary.json").read_text())


def run_matrix(
    root: Path,
    output: Path,
    task_ids: list[str] | None = None,
    num_trials: int = 1,
    max_steps: int | None = None,
    settings_path: Path | None = None,
    max_rollouts: int = 500,
) -> dict:
    """Run both public retrieval conditions and report exact type/structure coverage."""
    from tau3.worldgen.v2.diversity import structural_fingerprint
    from tau3.worldgen.v2.specs import WorldSpec

    check_certificate(root)
    spec = WorldSpec.model_validate_json((root / "spec.json").read_text())
    membership = {
        f"task_{s.id}": {
            "kind": s.kind,
            "structure": structural_fingerprint(c, s),
            "category": c.id,
        }
        for c in spec.categories
        for s in c.scenarios
    }
    ids = task_ids if task_ids is not None else sorted(membership)
    settings = load_settings(settings_path)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not set(ids) <= membership.keys()
        or len(set(settings.agent_models)) != 2
    ):
        raise ValueError("Matrix needs valid unique task IDs and two distinct models")
    expected_cells = len(ids) * len(settings.agent_models) * 2 * num_trials
    max_steps = max_steps if max_steps is not None else settings.simulation_max_steps
    if expected_cells > max_rollouts or min(num_trials, max_steps, max_rollouts) < 1:
        raise ValueError("Matrix exceeds the declared rollout budget")
    runs = {}
    for retrieval in ("full_kb", "bm25"):
        runs[retrieval] = run_rollouts(
            root,
            output / retrieval,
            retrieval,
            len(ids),
            num_trials,
            max_steps,
            settings_path,
            task_ids=ids,
        )
        rows = [
            {
                **cell,
                "model": r["model"],
                "retrieval": variant,
                **membership[cell["task_id"]],
            }
            for variant, result in runs.items()
            for r in result["runs"]
            for cell in r["cells"]
        ]
        coverage = {
            key: {
                value: {
                    "cells": sum(r[key] == value for r in rows),
                    "successes": sum(
                        r[key] == value
                        and r["reward"] == 1
                        and not r["judge_inconclusive"]
                        for r in rows
                    ),
                }
                for value in sorted({membership[tid][key] for tid in ids})
            }
            for key in ("kind", "structure", "category")
        }
        complete = len(runs) == 2 and all(v["status"] == "PASS" for v in runs.values())
        report = {
            "status": "PASS" if complete else "INCONCLUSIVE",
            "task_ids": ids,
            "all_world_tasks": set(ids) == membership.keys(),
            "expected_cells": expected_cells,
            "identity": {
                "artifacts": artifact_hashes(root),
                "implementation": implementation_hash(),
                "settings": digest(settings.model_dump()),
            },
            "coverage": coverage,
            "cells": rows,
            "scope": "Complete execution coverage; success rates and expansion readiness are separate",
        }
        write_json(output / "report.json", report)
    return report
