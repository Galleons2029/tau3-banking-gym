"""Requalify a frozen release offline, then resume the authorized 6k stream."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def clone(source, target):
    """Copy immutable inputs efficiently; writers must always replace atomically."""
    if target.exists():
        return
    try:
        shutil.copytree(source, target, copy_function=os.link)
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)


def check_adapter_compatibility(expected, tau=None):
    """Allow only the exact additive targeted CLI dispatch, with all other bytes pinned."""
    from tau3.synthesis import world_tasks
    from tau3.worldgen.v2.runtime import digest

    tau = tau or Path(world_tasks.__file__).parents[1]
    files = {
        name: (tau / "synthesis" / name).read_text()
        for name in ("world_tasks.py", "bundle.py", "cli.py", "concurrent_audit.py")
    }
    files["llm_concurrency.py"] = (tau / "utils/llm_concurrency.py").read_text()
    current = digest(files)
    changes = []
    if current != expected:
        for fragment in (
            "    from tau3.synthesis.targeted.cli import add_parser\n\n"
            "    add_parser(stages)\n",
            '    if args.synthesis_stage == "targeted":\n'
            "        from tau3.synthesis.targeted.cli import run\n\n"
            "        return run(args)\n",
        ):
            if files["cli.py"].count(fragment) != 1:
                raise ValueError("Unrecognized task adapter change")
            files["cli.py"] = files["cli.py"].replace(fragment, "", 1)
            changes.append(fragment)
        if digest(files) != expected:
            raise ValueError(
                "Task adapter has changes beyond additive targeted dispatch"
            )
    return {"original": expected, "current": current, "allowed_additions": changes}


def main():
    """Keep original evidence untouched and admit only an exact compatible release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from tau3.data_model.simulation import SimulationRun
    from tau3.data_model.tasks import Task
    from tau3.evaluator.evaluator_env import EnvironmentEvaluator
    from tau3.synthesis.stream_resume import read
    from tau3.synthesis.world_tasks import adapter_hash
    from tau3.utils import llm_utils
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.blind import run_blind
    from tau3.worldgen.v2.calibration import run_calibration
    from tau3.worldgen.v2.environment import get_environment
    from tau3.worldgen.v2.pipeline import (
        artifact_hashes,
        implementation_hash,
        publish,
        validate_world,
        write_json,
    )
    from tau3.worldgen.v2.readiness import assess, evidence_files
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.settings import load_settings

    args.root = args.root.resolve()
    args.output = args.output.resolve()

    def report_failure(error_type, error, traceback):
        state = {
            "phase": "blocked",
            "error_type": error_type.__name__,
            "reason": str(error),
            "updated_at": time.time(),
            "pid": os.getpid(),
            "output": str(args.output),
        }
        write_json(args.output / "migration-state.json", state)
        write_json(args.root / "resume-state.json", state)
        write_json(args.root / "batch-state.json", {**state, "status": "BLOCKED"})
        sys.__excepthook__(error_type, error, traceback)

    sys.excepthook = report_failure
    previous = args.root / "task-batch-6000"
    out = args.output / "task-batch-6000"
    with file_lock(args.output / "migration.lock"):

        def state(phase, **values):
            report = {
                "phase": phase,
                "updated_at": time.time(),
                "pid": os.getpid(),
                **values,
            }
            write_json(args.output / "migration-state.json", report)
            write_json(args.root / "resume-state.json", report)
            if phase == "offline_compatibility":
                write_json(
                    args.root / "batch-state.json",
                    {
                        **report,
                        "status": "REQUALIFYING",
                        "request_concurrency": int(os.environ["TAU3_LLM_CONCURRENCY"]),
                        "output": str(out),
                    },
                )

        state("offline_compatibility")
        settings = load_settings(args.config)
        normalized = settings.model_dump()
        for name, default in (
            ("agent_llm_args", {}),
            ("user_llm_args", {}),
            ("capture_retrieval", "bm25"),
        ):
            if normalized.pop(name) != default:
                raise ValueError(
                    "Migration requires unchanged effective actor settings"
                )
        source_manifest = read(args.root / "task-batch-8000/manifest.json")
        if digest(normalized) != source_manifest["settings_hash"]:
            raise ValueError("Effective model settings changed")
        adapter_compatibility = check_adapter_compatibility(
            source_manifest["adapter_hash"]
        )
        old_readiness = read(Path(source_manifest["readiness"]))
        if old_readiness["status"] != "PASS":
            raise ValueError("Previous seed qualification did not pass")
        for source in old_readiness["sources"].values():
            if evidence_files(Path(source["directory"])) != source["files"]:
                raise ValueError("Original seed evidence changed")
        old_world = Path(read(args.root / "local-worlds.json")["scale"])
        durable_world = args.output / "scale"
        clone(old_world, durable_world)
        mirror_path = args.output / "execution-world.json"
        if mirror_path.exists():
            world = Path(read(mirror_path)["scale"])
            if not world.exists():
                raise ValueError(
                    "Execution mirror missing; explicit restoration required"
                )
        else:
            world = Path(tempfile.mkdtemp(prefix="worldgen-resume-256-")) / "scale"
            clone(durable_world, world)
            write_json(
                mirror_path, {"scale": str(world), "durable_source": str(durable_world)}
            )
        if artifact_hashes(world) != source_manifest["source_artifacts"]:
            raise ValueError("Seed artifacts changed")
        validation = validate_world(world)
        if validation["status"] != "PASS":
            raise ValueError("Current seed structural validation failed")
        old_validation = read(old_world / "validation/report.json")
        if validation["scenarios"] != old_validation["scenarios"]:
            raise ValueError("Seed scenario results changed")
        publish(world)
        os.environ["TAU3_SYNTH_WORLD"] = str(world)
        os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(args.config)

        # Offline means offline: missing/changed cached prompts cannot silently
        # issue fresh calls until every compatibility gate has passed.
        original_generate = llm_utils.generate

        def forbidden_generate(**kwargs):
            raise RuntimeError(
                "Offline compatibility requires the exact historical model response"
            )

        llm_utils.generate = forbidden_generate
        try:
            gates = {}
            for name, run in (("blind", run_blind), ("calibration", run_calibration)):
                source = Path(old_readiness["sources"][name]["directory"])
                destination = args.output / "full/scale" / name
                clone(source / "calls", destination / "calls")
                report = run(
                    world,
                    destination,
                    settings_path=args.config,
                    max_calls=read(source / "audit.json")["max_calls"],
                )
                if report["status"] != "PASS":
                    raise ValueError(f"Offline {name} replay did not pass")
                write_json(
                    destination / "historical-origin.json",
                    {
                        "source": str(source),
                        "files": old_readiness["sources"][name]["files"],
                        "scope": "Exact cached raw replies; all checks rerun under the current code; no new model calls",
                    },
                )
                gates[name] = len(report["results"])
                state("offline_compatibility", completed=gates)

            source = Path(old_readiness["sources"]["online"]["directory"])
            destination = args.output / "full/scale/online"
            clone(source, destination)
            replayed = 0
            for path in sorted(source.glob("*/results_*.json")):
                results = read(path)
                tasks = {t["id"]: Task.model_validate(t) for t in results["tasks"]}
                for raw in results["simulations"]:
                    simulation = SimulationRun.model_validate(raw)
                    replay = EnvironmentEvaluator.calculate_reward(
                        lambda **kwargs: get_environment(
                            world, retrieval_variant=path.parent.name
                        ),
                        tasks[simulation.task_id],
                        simulation.messages,
                        strict_replay=True,
                    )
                    if (
                        replay.env_assertions != simulation.reward_info.env_assertions
                        or replay.db_check != simulation.reward_info.db_check
                    ):
                        raise ValueError("Historical online outcome changed")
                    replayed += 1
            report = read(source / "report.json")
            if replayed != len(report["cells"]):
                raise ValueError("Incomplete online compatibility replay")
            report["identity"] = {
                **report["identity"],
                "implementation": implementation_hash(),
                "settings": digest(settings.model_dump()),
            }
            report["compatibility"] = {
                "source_report": str(source / "report.json"),
                "source_hash": digest(read(source / "report.json")),
                "strict_tool_replays": replayed,
                "scope": "Historical model answers and grades retained; current runtime strictly replayed every tool output and outcome",
            }
            write_json(destination / "report.json", report)
            readiness = args.output / "full/scale/readiness.json"
            report = assess(
                world,
                args.output / "full/scale/blind",
                args.output / "full/scale/calibration",
                destination,
                readiness,
                args.config,
            )
            if report["status"] != "PASS":
                raise ValueError("Current controlled expansion readiness did not pass")
        finally:
            llm_utils.generate = original_generate

        # Reissue compatibility metadata only in a new root. Raw model calls,
        # old manifests, admissions, grades and failed attempts stay untouched.
        pilot = args.output / "task-pilot-20"
        clone(args.root / "task-pilot-20", pilot)
        original = read(pilot / "manifest.json")
        reissued = {
            **original,
            "adapter_hash": adapter_hash(),
            "implementation": implementation_hash(),
            "settings_hash": digest(settings.model_dump()),
            "readiness": str(readiness),
            "readiness_hash": digest(read(readiness)),
            "compatibility_origin": {
                "manifest": str(args.root / "task-pilot-20/manifest.json"),
                "hash": digest(original),
                "scope": "Historical pilot; compatible unchanged task adapter and fully replayed seed gates",
            },
        }
        write_json(pilot / "manifest.json", reissued)
        draft = args.output / "task-batch-8000"
        write_json(
            draft / "manifest.json",
            {
                **source_manifest,
                "adapter_hash": adapter_hash(),
                "implementation": implementation_hash(),
                "settings_hash": digest(settings.model_dump()),
                "readiness": str(readiness),
                "readiness_hash": digest(read(readiness)),
                "compatibility_origin": {
                    "manifest": str(args.root / "task-batch-8000/manifest.json"),
                    "hash": digest(source_manifest),
                },
            },
        )
        old_audit = read(args.root / "task-batch-8000/audit/audit.json")
        write_json(
            draft / "audit/audit.json",
            {
                **old_audit,
                "implementation": implementation_hash(),
                "settings": digest(settings.model_dump()),
            },
        )
        calls = draft / "audit/calls"
        if not calls.exists():
            calls.symlink_to(
                args.root / "task-batch-8000/audit/calls", target_is_directory=True
            )
        write_json(args.output / "local-worlds.json", {"scale": str(world)})
        migration = {
            "status": "PASS",
            "source": str(previous),
            "source_manifest": digest(read(previous / "manifest.json")),
            "implementation": implementation_hash(),
            "artifacts": artifact_hashes(world),
            "settings": digest(settings.model_dump()),
            "readiness": str(readiness),
            "readiness_hash": digest(read(readiness)),
            "checks": {**gates, "online_strict_replays": replayed},
            "adapter_compatibility": adapter_compatibility,
            "policy": "Preserve original evidence and all true failures. Reopen only stale-certificate candidate attempts. Revalidate every reused admission and teacher trajectory. Defer interrupted teacher attempts without resampling.",
            "pilot_sft": {
                "path": str(previous / "pilot-sft/sft.jsonl"),
                "progress": read(previous / "pilot-sft/progress.json"),
                "policy": "Keep completed attempts and failed grades; no resampling",
            },
        }
        write_json(out / "migration.json", migration)
        state(
            "streaming",
            checks=migration["checks"],
            output=str(out),
            request_concurrency=int(os.environ["TAU3_LLM_CONCURRENCY"]),
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_world_streaming.py")),
            "--root",
            str(args.output),
            "--config",
            str(args.config),
            "--output",
            str(out),
            "--count",
            "6000",
            "--resume-from",
            str(previous),
        ]
        result = subprocess.call(command)
        state(
            "complete" if result == 0 else "incomplete",
            output=str(out),
            exit_code=result,
        )
        raise SystemExit(result)


if __name__ == "__main__":
    main()
