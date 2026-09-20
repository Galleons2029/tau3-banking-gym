"""Revalidate existing snapshots, run small gates, then the fixed full matrix.

No additional categories or rewritten documents are synthesized by this script.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tau3.utils.llm_concurrency import configured_concurrency, install_request_limit
from tau3.worldgen.v2.category_review import admitted, review_category
from tau3.worldgen.v2.pipeline import (
    artifact_hashes,
    check_certificate,
    implementation_hash,
    initial_database,
    publish,
    task_payload,
    validate_world,
    write_json,
)
from tau3.worldgen.v2.runtime import OperationRuntime, digest
from tau3.worldgen.v2.settings import load_settings
from tau3.worldgen.v2.specs import WorldSpec


def execution_world(root: Path, name: str, mirrors: dict) -> Path:
    """Use an optional local copy only after verifying the complete artifact identity."""
    source = root / name
    if name not in mirrors:
        return source
    candidate = Path(mirrors[name])
    check_certificate(source)
    check_certificate(candidate)
    if artifact_hashes(source) != artifact_hashes(candidate):
        raise ValueError("Local world mirror differs from the recorded source")
    return candidate


def refresh(source: Path, destination: Path):
    """Copy a prior snapshot and rebuild scoring metadata, preserving public prose."""
    if destination.exists():
        try:
            check_certificate(destination)
            return
        except ValueError:
            pass  # Revalidate the dedicated copy after an implementation change.
    else:
        shutil.copytree(source, destination)
    write_json(destination / "manifest.json", {"schema_version": 2, "status": "draft"})
    spec = WorldSpec.model_validate_json((source / "spec.json").read_text())
    initial = initial_database(spec)
    aliases = OperationRuntime(spec, initial).aliases
    config = json.loads((source / "build.json").read_text())
    settings = load_settings()
    config["spec_hash"] = digest(spec.model_dump())
    write_json(destination / "spec.json", spec.model_dump(mode="json"))
    write_json(destination / "build.json", config)
    for category in spec.categories:
        path = destination / "private/category_reviews" / f"{category.id}.json"
        reviews = json.loads(path.read_text()) if path.exists() else []
        if not admitted(
            category, spec.clock, initial, reviews, config["review_models"]
        ):
            reviews = [
                review_category(category, spec.clock, initial, model, settings.llm_args)
                for model in config["review_models"]
            ]
            write_json(path, reviews)
        for scenario in category.scenarios:
            task = json.loads(
                (source / "tasks" / f"task_{scenario.id}.json").read_text()
            )
            write_json(
                destination / "tasks" / f"task_{scenario.id}.json",
                task_payload(
                    spec, scenario, task["required_documents"], aliases, initial
                ),
            )
    report = validate_world(destination)
    if report["status"] != "PASS":
        raise ValueError(f"Revalidation failed: {report.get('errors')}")
    publish(destination)
    write_json(
        destination / "migration.json",
        {
            "source": str(source),
            "scope": "Existing documents and business scenarios; updated evaluation metadata",
            "source_spec_hash": digest(json.loads((source / "spec.json").read_text())),
        },
    )


def sharded_blind(world: Path, output: Path, config: Path | None, max_calls=1000):
    """Partition fixed tasks/budget, then replay merged raw calls with the checker."""
    from tau3.worldgen.v2.audit import AuditSession
    from tau3.worldgen.v2.blind import run_blind

    ids = sorted(p.stem for p in (world / "tasks").glob("*.json"))
    workers = min(4, len(ids))
    shards = output.parent / "blind-shards"
    plan = {
        "task_ids": ids,
        "workers": workers,
        "max_calls": max_calls,
        "partitions": [ids[i::workers] for i in range(workers)],
    }
    plan_path = shards / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Blind shard plan changed")
    write_json(plan_path, plan)
    children = []
    for i, task_ids in enumerate(plan["partitions"]):
        part = shards / str(i)
        write_json(part / "tasks.json", task_ids)
        budget = max_calls // workers + int(i < max_calls % workers)
        command = [
            sys.executable,
            __file__,
            "--stage",
            "full",
            "--blind-worker",
            "--worker-world",
            str(world),
            "--worker-output",
            str(part),
            "--worker-max-calls",
            str(budget),
        ]
        if config:
            command += ["--config", str(config)]
        with (part / "worker.log").open("a") as log:
            children.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            )
    codes = [child.wait() for child in children]
    write_json(shards / "workers.json", {"exit_codes": codes})
    if any(codes):
        raise ValueError("Blind shard incomplete; original evidence retained")
    session = AuditSession(world, output, load_settings(config), max_calls, "blind-v1")
    records, provenance, physical_calls = {}, [], 0
    for i in range(workers):
        part = shards / str(i)
        audit = json.loads((part / "audit.json").read_text())
        if any(
            audit.get(k) != v for k, v in session.identity.items() if k != "max_calls"
        ):
            raise ValueError("Blind shard identity changed")
        report = json.loads((part / "report.json").read_text())
        if report["status"] != "PASS" or report["task_ids"] != plan["partitions"][i]:
            raise ValueError("Blind shard membership incomplete")
        provenance.append({"audit": audit, "report": report})
        for p in (part / "calls").glob("*.json"):
            record = json.loads(p.read_text())
            if p.name in records:
                raise ValueError(
                    "Overlapping shard requests; cannot merge call budgets"
                )
            records[p.name] = record
            physical_calls += len(record.get("attempts", [None]))
    if physical_calls > max_calls:
        raise ValueError("Shards exceeded total physical call budget")
    existing = {p.name for p in (output / "calls").glob("*.json")}
    if not existing <= records.keys():
        raise ValueError("Unexpected calls in merged blind audit")
    for name, record in records.items():
        target = output / "calls" / name
        if target.exists() and json.loads(target.read_text()) != record:
            raise ValueError("Merged blind response differs from original shard")
        write_json(target, record)
    write_json(
        output / "shard-provenance.json",
        {
            "plan": plan,
            "physical_calls": physical_calls,
            "shards": provenance,
        },
    )
    # No PASS flag is invented by merging: normal validation repeats from the
    # exact saved model responses, including every public repair attempt.
    return run_blind(
        world, output, ids, settings_path=config, max_calls=max_calls, fail_fast=True
    )


def online_condition(world, output, retrieval, config, model=None):
    """Run model workers into distinct native result files, then use native checks."""
    from tau3.data_model.simulation import Results, TextRunConfig
    from tau3.domains.banking_synth.environment import get_environment, get_tasks
    from tau3.registry import registry
    from tau3.runner.batch import run_tasks
    from tau3.worldgen.v2.behaviour import summarize
    from tau3.worldgen.v2.rollouts import run_rollouts

    check_certificate(world)
    settings = load_settings(config)
    ids = sorted(p.stem for p in (world / "tasks").glob("*.json"))
    if not ids or len(ids) * len(settings.agent_models) > 240:
        raise ValueError("Online condition exceeds fixed full-matrix budget")
    identity = {
        "artifacts": artifact_hashes(world),
        "implementation": implementation_hash(),
        "settings_hash": digest(settings.model_dump()),
        "retrieval": retrieval,
        "num_tasks": len(ids),
        "num_trials": 1,
        "max_steps": settings.simulation_max_steps,
        "task_ids": ids,
    }
    matrix = output / "matrix.json"
    if matrix.exists() and json.loads(matrix.read_text()) != identity:
        raise ValueError("Online worker configuration changed")
    if model is None:
        write_json(matrix, identity)  # One parent owns this shared metadata file.
        children = []
        for selected in settings.agent_models:
            command = [
                sys.executable,
                __file__,
                "--stage",
                "full",
                "--online-worker",
                retrieval,
                "--online-model",
                selected,
                "--worker-world",
                str(world),
                "--worker-output",
                str(output),
            ]
            if config:
                command += ["--config", str(config)]
            with (output / f"worker_{digest(selected)[:12]}.log").open("a") as log:
                children.append(
                    subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                )
        codes = [child.wait() for child in children]
        write_json(
            output / "model-workers.json",
            {"models": settings.agent_models, "exit_codes": codes},
        )
        if any(codes):
            raise ValueError(
                "Model worker incomplete; no completed failures are resampled"
            )
        return run_rollouts(
            world, output, retrieval=retrieval, task_ids=ids, settings_path=config
        )

    if model not in settings.agent_models or not matrix.exists():
        raise ValueError("Undeclared online model worker")
    os.environ["TAU3_SYNTH_WORLD"] = str(world.resolve())
    if config:
        os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(config.resolve())
    if "banking_synth" not in registry.get_domains():
        registry.register_domain(get_environment, "banking_synth")
    available = {t.id: t for t in get_tasks()}
    tasks = [available[tid] for tid in ids]
    expected = {t.id: t.model_dump(mode="json") for t in tasks}
    path = output / f"results_{digest(model)[:12]}.json"
    result = None
    if path.exists():
        result = Results.load(path)
        if (
            {t.id: t.model_dump(mode="json") for t in result.tasks} != expected
            or result.info.world_artifact_hash != digest(identity["artifacts"])
            or result.info.world_implementation_hash != identity["implementation"]
            or result.info.world_evaluation_hash != identity["settings_hash"]
            or result.info.agent_info.llm != model
            or result.info.user_info.llm != settings.user_model
            or result.info.num_trials != 1
            or result.info.max_steps != settings.simulation_max_steps
            or result.info.seed != 42
            or result.info.retrieval_config != retrieval
        ):
            raise ValueError("Cached model worker identity changed")
        if any(
            s.termination_reason.value == "infrastructure_error"
            for s in result.simulations
        ):
            raise ValueError("Saved infrastructure failure requires diagnosis")
        if len(result.simulations) < len(tasks):
            result = None
    if result is None:
        run_config = TextRunConfig(
            domain="banking_synth",
            llm_agent=model,
            llm_user=settings.user_model,
            llm_args_agent=settings.llm_args,
            llm_args_user=settings.llm_args,
            num_trials=1,
            max_steps=settings.simulation_max_steps,
            timeout=settings.simulation_timeout,
            max_errors=5,
            max_concurrency=(
                max(1, configured_concurrency() // (2 * len(settings.agent_models)))
                if "TAU3_LLM_CONCURRENCY" in os.environ
                else 2
            ),
            max_retries=0,
            auto_resume=True,
            retrieval_config=retrieval,
            seed=42,
            log_level="ERROR",
        )
        result = run_tasks(run_config, tasks, save_path=path, console_display=False)
    summary = summarize(result, expected)
    return {"status": "PASS", **summary}


def main():
    """Run a finite stage; later stages refuse to proceed after inconclusive gates."""
    install_request_limit()
    from tau3.worldgen.v2.blind import run_blind
    from tau3.worldgen.v2.calibration import run_calibration
    from tau3.worldgen.v2.readiness import assess
    from tau3.worldgen.v2.rollouts import run_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic/worldgen-v2-independent")
    )
    parser.add_argument("--stage", choices=["prepare", "smoke", "full"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--parallel-online", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--online-worker", choices=["full_kb", "bm25"])
    parser.add_argument("--online-model")
    parser.add_argument("--blind-worker", action="store_true")
    parser.add_argument("--worker-max-calls", type=int, default=250)
    parser.add_argument("--worker-world", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument(
        "--local-worlds",
        type=Path,
        help="Optional JSON mapping of world/scale/natural to identical local copies",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="For smoke: aggregate existing reports without making model calls",
    )
    args = parser.parse_args()
    if args.blind_worker:
        if not args.worker_world or not args.worker_output:
            parser.error("Blind workers require explicit world and output")
        report = run_blind(
            args.worker_world,
            args.worker_output,
            json.loads((args.worker_output / "tasks.json").read_text()),
            settings_path=args.config,
            max_calls=args.worker_max_calls,
            fail_fast=True,
        )
        raise SystemExit(0 if report["status"] == "PASS" else 2)
    if args.online_worker:
        if not args.worker_world or not args.worker_output:
            parser.error("Online workers require explicit world and output")
        report = online_condition(
            args.worker_world,
            args.worker_output,
            args.online_worker,
            args.config,
            args.online_model,
        )
        raise SystemExit(0 if report["status"] == "PASS" else 2)
    if args.summarize_only and args.stage != "smoke":
        parser.error("--summarize-only requires --stage smoke")
    root = args.output
    mirrors = json.loads(args.local_worlds.read_text()) if args.local_worlds else {}
    if args.stage == "prepare":
        for name, source in [
            ("world", "worldgen-v2-final-expanded"),
            ("scale", "worldgen-v2-final-scale/categories_024"),
            ("natural", "worldgen-v2-final-natural"),
        ]:
            refresh(Path("data/synthetic") / source, root / name)
            print(f"Revalidated {name}", flush=True)
        return
    if args.stage == "smoke":
        spec = WorldSpec.model_validate_json((root / "world/spec.json").read_text())
        ids = [f"task_{s.id}" for s in spec.categories[0].scenarios]
        ids.extend(f"task_{s.id}" for s in spec.categories[-1].scenarios)
        if args.summarize_only:
            reports = {
                name: json.loads((root / "smoke" / name / "report.json").read_text())
                for name in ("blind", "calibration", "online")
            }
        else:
            world = execution_world(root, "world", mirrors)
            reports = {}
            reports["blind"] = run_blind(
                world,
                root / "smoke/blind",
                ids,
                settings_path=args.config,
                max_calls=48,
            )
            print({"blind": reports["blind"]["status"]}, flush=True)
            reports["calibration"] = run_calibration(
                world,
                root / "smoke/calibration",
                settings_path=args.config,
                max_calls=200,
            )
            print({"calibration": reports["calibration"]["status"]}, flush=True)
            reports["online"] = run_matrix(
                world,
                root / "smoke/online",
                ids,
                settings_path=args.config,
                max_rollouts=24,
            )
            print({"online": reports["online"]["status"]}, flush=True)
        identity = {
            "artifacts": artifact_hashes(root / "world"),
            "implementation": implementation_hash(),
            "settings": digest(load_settings(args.config).model_dump()),
        }
        # Complete execution alone does not unlock the larger matrix.
        successes = {
            r["task_id"]
            for r in reports["online"]["cells"]
            if r["retrieval"] == "full_kb"
            and r["reward"] == 1
            and not r["judge_inconclusive"]
        }
        passed = (
            all(r["status"] == "PASS" for r in reports.values())
            and set(ids) <= successes
            and all(
                r.get("identity", {}).get(k) == v
                for r in reports.values()
                for k, v in identity.items()
            )
            and all(
                set(reports[k].get("task_ids", [])) == set(ids)
                for k in ("blind", "online")
            )
        )
        report = {
            "status": "PASS" if passed else "INCONCLUSIVE",
            "task_ids": ids,
            "reports": {k: r["status"] for k, r in reports.items()},
        }
        write_json(root / "smoke/report.json", report)
        if not passed:
            raise SystemExit(2)
    else:
        smoke = json.loads((root / "smoke/report.json").read_text())
        if smoke["status"] != "PASS":
            raise ValueError(
                "Small validation gates have not passed; full run remains locked"
            )
        identity = {
            "artifacts": artifact_hashes(root / "world"),
            "implementation": implementation_hash(),
            "settings": digest(load_settings(args.config).model_dump()),
        }
        for name in ("blind", "calibration", "online"):
            evidence = json.loads((root / "smoke" / name / "report.json").read_text())
            if evidence.get("status") != "PASS" or any(
                evidence.get("identity", {}).get(k) != v for k, v in identity.items()
            ):
                raise ValueError(
                    "Small gate evidence became stale; full run remains locked"
                )
        for name in ("scale", "natural"):
            world = execution_world(root, name, mirrors)
            blind = sharded_blind(world, root / "full" / name / "blind", args.config)
            if blind["status"] != "PASS":
                raise ValueError(f"Blind verification incomplete for {name}")
            calibration = run_calibration(
                world,
                root / "full" / name / "calibration",
                settings_path=args.config,
                max_calls=400,
            )
            if calibration["status"] != "PASS":
                raise ValueError(f"Scoring calibration incomplete for {name}")
            if args.parallel_online:
                children = []
                for retrieval in ("full_kb", "bm25"):
                    command = [
                        sys.executable,
                        __file__,
                        "--stage",
                        "full",
                        "--online-worker",
                        retrieval,
                        "--worker-world",
                        str(world),
                        "--worker-output",
                        str(root / "full" / name / "online" / retrieval),
                    ]
                    if args.config:
                        command += ["--config", str(args.config)]
                    with (root / "full" / name / f"{retrieval}.log").open("a") as log:
                        children.append(
                            subprocess.Popen(
                                command, stdout=log, stderr=subprocess.STDOUT
                            )
                        )
                codes = [child.wait() for child in children]
                write_json(
                    root / "full" / name / "online-workers.json", {"exit_codes": codes}
                )
                if any(codes):
                    raise ValueError(
                        "Online worker incomplete; preserve evidence for diagnosis"
                    )
                # Aggregate completed evidence without repeating failed cells.
            report = run_matrix(
                world,
                root / "full" / name / "online",
                settings_path=args.config,
                max_rollouts=480,
            )
            print({name: report["status"]}, flush=True)
            if artifact_hashes(root / name) != artifact_hashes(world):
                raise ValueError("Canonical seed changed during full validation")
            readiness = assess(
                world,
                root / "full" / name / "blind",
                root / "full" / name / "calibration",
                root / "full" / name / "online",
                root / "full" / name / "readiness.json",
                args.config,
            )
            if readiness["status"] != "PASS":
                raise ValueError(
                    f"Controlled expansion readiness incomplete for {name}"
                )
        # New cancellation workflow in the full public corpus is covered by the
        # frozen smoke matrix; its four cells are included in the 488-cell scope.


if __name__ == "__main__":
    main()
