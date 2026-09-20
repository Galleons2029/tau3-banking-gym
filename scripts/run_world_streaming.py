"""Run the authorized 6k stream and pilot teacher sampling in separate processes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Supervise bounded producers and persist independent completion states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--teacher-policy", type=Path)
    parser.add_argument("--mode", choices=["all", "tasks", "pilot-sft"], default="all")
    args = parser.parse_args()
    if args.teacher_policy and args.mode == "pilot-sft":
        parser.error(
            "Teacher policy currently applies to the task stream; pilot needs a separate output"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        from tau3.synthesis.streaming_tasks import stream_tasks
        from tau3.synthesis.teacher_sampling import TeacherPolicy
        from tau3.synthesis.world_sft import pilot_sft

        mirrors = json.loads((args.root / "local-worlds.json").read_text())
        world = Path(mirrors["scale"])
        if args.mode == "pilot-sft":
            result = pilot_sft(
                world,
                args.root / "task-pilot-20",
                args.config,
                args.output / "pilot-sft",
            )
        else:
            result = stream_tasks(
                world,
                args.root / "task-batch-8000",
                args.root / "task-pilot-20",
                args.config,
                args.output,
                args.count,
                workers=int(os.environ.get("TAU3_LLM_CONCURRENCY", "128")),
                resume_from=args.resume_from,
                teacher_policy=TeacherPolicy.model_validate_json(
                    args.teacher_policy.read_text()
                )
                if args.teacher_policy
                else None,
            )
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result["status"] == "COMPLETE" else 2)

    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.pipeline import write_json

    with file_lock(args.output / "supervisor.lock"):
        children = {}
        # All producers share the request pool, including agent/user/judge calls.
        # Completed pilot attempts remain immutable during a version migration.
        modes = (
            ("tasks",)
            if args.resume_from or args.teacher_policy
            else ("pilot-sft", "tasks")
        )
        for mode in modes:
            command = [
                sys.executable,
                __file__,
                "--root",
                str(args.root),
                "--config",
                str(args.config),
                "--output",
                str(args.output),
                "--count",
                str(args.count),
                "--mode",
                mode,
            ]
            if args.teacher_policy:
                command.extend(["--teacher-policy", str(args.teacher_policy)])
            if args.resume_from:
                command.extend(["--resume-from", str(args.resume_from)])
            with (args.output / f"{mode}.log").open("a") as log:
                children[mode] = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        while True:
            states = {
                name: {"pid": p.pid, "exit_code": p.poll()}
                for name, p in children.items()
            }
            done = all(item["exit_code"] is not None for item in states.values())
            status = (
                "RUNNING"
                if not done
                else (
                    "COMPLETE"
                    if all(v["exit_code"] == 0 for v in states.values())
                    else "INCOMPLETE"
                )
            )
            state = {
                "status": status,
                "supervisor_pid": os.getpid(),
                "requested_tasks": args.count,
                "request_concurrency": int(
                    os.environ.get("TAU3_LLM_CONCURRENCY", "128")
                ),
                "previous_output": str(args.resume_from) if args.resume_from else None,
                "workers": states,
                "updated_at": time.time(),
                "output": str(args.output.resolve()),
            }
            write_json(args.output / "run-state.json", state)
            write_json(args.root / "batch-state.json", state)
            if status_root := os.environ.get("TAU3_STREAM_STATUS_ROOT"):
                write_json(Path(status_root) / "batch-state.json", state)
            if done:
                break
            time.sleep(10)
        if status != "COMPLETE":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
