"""Durable native pilot -> 400-task package -> external-evaluation handoff."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tau3.synthesis.storage import read_json, write_json
from tau3.utils.llm_concurrency import file_lock


def main():
    """Stop on explicit gaps; never silently change slots, seeds, budgets or curricula."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--additional-report", type=Path, action="append", default=[])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--start-index", type=int, choices=range(7), default=0)
    args = parser.parse_args()
    root = args.round_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    common = ["--round-dir", str(root), "--config", str(args.config.resolve()), "--resume"]
    analyze = ["analyze", "--report", str(args.report.resolve()), "--round-id", args.round_id, "--base-model", args.base_model]
    if args.run:
        analyze += ["--run", str(args.run.resolve())]
    for report in args.additional_report:
        analyze += ["--additional-report", str(report.resolve())]
    stages = [analyze, ["plan"], ["pilot"], ["generate", "--stage", "small"],
              ["collect", "--stage", "small"], ["export", "--stage", "small"],
              ["generate", "--split", "validation"]]
    with file_lock(root / "supervisor.lock"):
        incomplete_stages = []
        for index, stage in enumerate(stages):
            if index < args.start_index:
                continue
            command = [sys.executable, "-m", "tau3.cli", "synthesize", "targeted", *stage, *common]
            log = root / "logs" / f"{index:02d}-{stage[0]}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            state = {"pid": os.getpid(), "stage_index": index, "stage": stage,
                     "started_at": started, "log": str(log), "status": "RUNNING"}
            write_json(root / "supervisor.json", state)
            with log.open("a") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           env={**os.environ, "LOGURU_LEVEL": "ERROR"})
                write_json(root / "supervisor.json", {**state, "child_pid": process.pid})
                code = process.wait()
            if code:
                if code == 2 and index in {3, 4, 5}:
                    # Publication gaps or incomplete slots must not suppress the
                    # valid tasks' four samples or already qualified SFT parts.
                    incomplete_stages.append({"stage": stage, "exit_code": code})
                    write_json(root / "partial-stages.json", incomplete_stages)
                    continue
                write_json(root / "supervisor.json", {**state, "status": "INCOMPLETE", "exit_code": code,
                           "finished_at": time.time(), "next_action": "Inspect stage evidence; resume identical inputs after resolving the cause."})
                return code
        training = read_json(root / "small/training_manifest.json")
        write_json(root / "supervisor.json", {"pid": os.getpid(), "status": "WAITING_EXTERNAL_EVALUATION" if training["status"] == "READY" else "INCOMPLETE",
                   "finished_at": time.time(), "training_manifest": str(root / "small/training_manifest.json"),
                   "validation_bundle": str(root / "validation/bundle"),
                   "next_action": "External training and paired evaluation; import receipt with targeted accept-evaluation before full generation."})
        return 0 if training["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
