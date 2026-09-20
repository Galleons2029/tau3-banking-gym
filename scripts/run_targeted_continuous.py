"""Resume fixed targeted slots while keeping export and telemetry off dispatch."""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


class ExportProcess:
    """Run one exporter at a time and remember which completed jobs it covers."""

    def __init__(self, command, log_path, interval=600):
        self.command, self.log_path, self.interval = command, log_path, interval
        self.process = None
        self.started_revision = -1
        self.finished_revision = -1
        self.last_start = float("-inf")
        self.exit_code = None

    def tick(self, revision, final=False):
        """Poll without waiting; failed exports never count as successful."""
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return
            self.exit_code = code
            if code == 0:
                self.finished_revision = self.started_revision
            self.process = None
        if self.finished_revision >= revision:
            return
        if time.monotonic() - self.last_start < (30 if final else self.interval):
            return
        with self.log_path.open("a") as log:
            self.process = subprocess.Popen(
                self.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        self.started_revision = revision
        self.last_start = time.monotonic()


def atomic(path, value):
    """Replace a small status file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def job(root, raw, slot, plan_hash):
    """Persist one isolated world result before handing control back to dispatch."""
    from tau3.synthesis.targeted import workflow as w
    from tau3.worldgen.v2.pipeline import write_json

    result = w.isolated_worker(w.generate_slot, root, raw, slot, "formal", plan_hash)
    output = {"slot": slot["index"], "generation": result}
    if result.get("status") == "VALID":
        output["trials"] = w.isolated_worker(
            w.collect_slot, root, raw, slot, "formal", plan_hash
        )
    write_json(Path(root) / "continuous-results" / f"{slot['index']:06d}.json", output)
    return {"slot": slot["index"], "status": result.get("status")}


def auxiliary(args):
    """Slow shared-filesystem work lives only in dedicated subprocesses."""
    from tau3.synthesis.targeted.models import RoundConfig
    from tau3.utils.llm_concurrency import RequestPool, file_lock

    root = args.round_dir
    config = RoundConfig.model_validate_json((root / "config.json").read_text())
    if args.mode == "export":
        with file_lock(root / "continuous-export.lock"):
            helper = root / "export-runtime"
            if (helper / "targeted_incremental_export.py").exists():
                # Independent offline exporter revision; teacher runtime stays frozen.
                sys.path.insert(0, str(helper))
                from targeted_incremental_export import export_incremental
            else:
                from tau3.synthesis.targeted.incremental_export import (
                    export_incremental,
                )
            export_incremental(
                root,
                config,
                "formal",
                workers=8,
                full=(root / "formal/full-export-audit-required.json").exists(),
            )
        return
    pool = RequestPool(root / "llm_pool", config.llm_concurrency)
    while True:
        state = json.loads(args.heartbeat.read_text())
        try:
            snapshot = pool.snapshot()
            state["llm"] = {
                "limit": snapshot["limit"],
                "active": len(snapshot["active"]),
                "peak": snapshot["peak"],
                "admitted": snapshot["admitted"],
            }
            atomic(root / "run-state.json", state)
        except (OSError, ValueError) as exc:
            print(repr(exc), flush=True)
        if state["status"] in ("COMPLETE", "INCOMPLETE", "FAILED"):
            return
        time.sleep(10)


def main(args):
    from tau3.synthesis.targeted import workflow as w
    from tau3.synthesis.targeted.models import RoundConfig
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.runtime import digest

    root = args.round_dir
    with file_lock(root / "supervisor.lock"):
        config = RoundConfig.model_validate_json((root / "config.json").read_text())
        w.initialize(root, config)
        w.check_pilot(root, config)
        w.check_forecast(root, config)
        plan = w.load_plan(root)
        plan_hash = digest(plan.model_dump())
        slots, skipped = [], 0
        for slot in plan.slots:
            d = root / "formal/slots" / f"{slot.index:06d}"
            if (d / "task.json").exists() and all(
                (d / "teacher" / str(i) / "quality.json").exists() for i in range(4)
            ):
                skipped += 1  # Includes unresolved evidence; final report retains it.
            else:
                slots.append(slot)
        slots.sort(
            key=lambda s: (
                not (root / "formal/slots" / f"{s.index:06d}" / "task.json").exists(),
                s.index,
            )
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--round-dir",
            str(root),
            "--heartbeat",
            str(args.heartbeat),
        ]
        exporter = ExportProcess(
            command + ["--mode", "export"], Path(str(args.heartbeat) + ".export.log")
        )
        state = {
            "pid": os.getpid(),
            "status": "RUNNING",
            "stage": "dispatch",
            "skipped_recorded_tasks": skipped,
            "llm_limit": config.llm_concurrency,
        }
        atomic(args.heartbeat, {**state, "updated_at": time.time()})
        with Path(str(args.heartbeat) + ".telemetry.log").open("a") as log:
            telemetry = subprocess.Popen(
                command + ["--mode", "telemetry"],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        completed = 0
        try:
            pending = iter(slots)
            with ProcessPoolExecutor(
                max_workers=config.workers,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
            ) as pool:
                futures = {}

                def submit():
                    slot = next(pending, None)
                    if slot is not None:
                        futures[
                            pool.submit(
                                job,
                                str(root),
                                config.model_dump(),
                                slot.model_dump(),
                                plan_hash,
                            )
                        ] = slot.index

                for _ in range(config.workers):
                    submit()
                while futures:
                    done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = futures.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            print(
                                f"Slot {index} remains incomplete: {exc!r}", flush=True
                            )
                        completed += 1
                        submit()
                    exporter.tick(completed)
                    atomic(
                        args.heartbeat,
                        {
                            **state,
                            "updated_at": time.time(),
                            "completed_jobs": completed,
                            "active_jobs": len(futures),
                            "export_pid": exporter.process.pid
                            if exporter.process
                            else None,
                            "export_revision": exporter.finished_revision,
                        },
                    )
            while exporter.finished_revision < completed:
                exporter.tick(completed, final=True)
                atomic(
                    args.heartbeat,
                    {
                        **state,
                        "stage": "final_export",
                        "updated_at": time.time(),
                        "completed_jobs": completed,
                        "active_jobs": 0,
                        "export_pid": exporter.process.pid
                        if exporter.process
                        else None,
                    },
                )
                time.sleep(5)
            report = json.loads((root / "formal/report.json").read_text())
            atomic(
                args.heartbeat,
                {
                    **state,
                    "status": report["status"],
                    "stage": "finished",
                    "updated_at": time.time(),
                    "completed_jobs": completed,
                },
            )
        except BaseException:
            atomic(
                args.heartbeat, {**state, "status": "FAILED", "updated_at": time.time()}
            )
            raise
        finally:
            # On graceful exit the telemetry process publishes the terminal status.
            if telemetry.poll() is not None:
                print(f"Telemetry exited: {telemetry.returncode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--mode", choices=["run", "export", "telemetry"], default="run")
    args = parser.parse_args()
    args.round_dir = args.round_dir.resolve()
    if args.mode == "run":
        atomic(
            args.heartbeat,
            {
                "pid": os.getpid(),
                "status": "RUNNING",
                "stage": "initializing",
                "updated_at": time.time(),
            },
        )
        try:
            main(args)
        except BaseException as exc:
            atomic(
                args.heartbeat,
                {
                    "pid": os.getpid(),
                    "status": "FAILED",
                    "reason": repr(exc),
                    "updated_at": time.time(),
                },
            )
            raise
    else:
        auxiliary(args)
