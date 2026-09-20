"""Run the approved report-to-SFT stages, stopping at the first incomplete gate."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Supervise stage processes and persist a heartbeat without bypassing gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--previous-plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from tau3.utils.llm_concurrency import RequestPool, file_lock
    from tau3.worldgen.v2.pipeline import write_json

    root = args.round_dir.resolve()
    with file_lock(root / "supervisor.lock"):
        for stage in (
            "analyze",
            "plan",
            "pilot",
            "generate",
            "collect",
            "export",
            "report",
        ):
            command = [
                sys.executable,
                "-m",
                "tau3.cli",
                "synthesize",
                "targeted",
                stage,
                "--round-dir",
                str(root),
                "--config",
                str(args.config.resolve()),
            ]
            if args.resume:
                command.append("--resume")
            if stage == "analyze":
                command += [
                    "--report",
                    str(args.report.resolve()),
                    "--round-id",
                    args.round_id,
                    "--base-model",
                    args.base_model,
                ]
                if args.run:
                    command += ["--run", str(args.run.resolve())]
            elif stage == "plan" and args.previous_plan:
                command += ["--previous-plan", str(args.previous_plan.resolve())]
            with (root / f"{stage}.log").open("a") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            while True:
                code = process.poll()
                state = {
                    "stage": stage,
                    "pid": process.pid,
                    "updated_at": time.time(),
                    "exit_code": code,
                    "command": command,
                    "status": "RUNNING"
                    if code is None
                    else "COMPLETE"
                    if code == 0
                    else "INCOMPLETE",
                }
                try:
                    pool_path = root / "llm_pool/status.json"
                    if pool_path.exists():
                        limit = json.loads(pool_path.read_text())["limit"]
                        pool = RequestPool(pool_path.parent, limit).snapshot()
                        state["llm"] = {
                            "limit": limit,
                            "active": len(pool["active"]),
                            "peak": pool["peak"],
                            "admitted": pool["admitted"],
                        }
                    progress_path = root / "attribution-progress.json"
                    if stage == "analyze" and progress_path.exists():
                        state["attribution"] = json.loads(progress_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    # Telemetry is optional; a transient shared-filesystem read
                    # must not orphan the active stage or prevent the next one.
                    state["telemetry_warning"] = f"{type(exc).__name__}: {exc}"
                    print(state["telemetry_warning"], file=sys.stderr, flush=True)
                try:
                    write_json(root / "run-state.json", state)
                except OSError as exc:
                    print(f"Heartbeat write failed: {exc}", file=sys.stderr, flush=True)
                if code is not None:
                    break
                time.sleep(10)
            print(json.dumps(state), flush=True)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
