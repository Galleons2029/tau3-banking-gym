"""Resume the pilot, then scale and collect SFT only after complete admission."""

import argparse
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tau3.synthesis.storage import write_json
from tau3.synthesis.workflow import (
    collect_sft,
    error_info,
    export_bundle,
    generate_bundle,
    read_config,
    records,
    validate_bundle,
)


def build_admitted_bundle(root, count, config, progress):
    """Bound candidate replacement and service retries; never scale a short pilot."""
    previous = None
    stalled_rounds = 0
    for round_index in range(config.candidate_attempts + 3):
        progress("generate", bundle=str(root), count=count, round=round_index)
        generate_bundle(root, count, config, resume=(root / "manifest.json").exists())
        exhausted = [
            r.slot for r in records(root) if r.checks.get("candidate_exhausted")
        ]
        if exhausted:
            raise ValueError(
                f"Candidate budget exhausted for slots {exhausted}; no further validation scheduled"
            )
        progress("validate", bundle=str(root), count=count, round=round_index)
        report = validate_bundle(root, resume=True)
        progress("validated", bundle=str(root), report=report)
        if report["accepted"] == count:
            export_bundle(root)
            from tau3.synthesis.bundle import load_task_bundle

            published = load_task_bundle(root)
            if len(published) != count:
                raise ValueError(
                    f"Publication deduplication left {len(published)}/{count} tasks"
                )
            return
        current = [(r.task.id, r.accepted) for r in records(root)]
        stalled_rounds = stalled_rounds + 1 if current == previous else 0
        if stalled_rounds >= 2:
            raise ValueError(
                f"No admission progress after retries: {report['accepted']}/{count}; "
                "inspect candidate errors and resume after resolving the failure"
            )
        previous = current
    raise ValueError(f"Candidate budget exhausted for {root}; see report.json")


def main():
    """Execute authorized stages and persist a durable, secret-free job status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--collect-sft", action="store_true")
    args = parser.parse_args()
    logger.remove()

    lock = threading.Lock()
    stopped = threading.Event()
    active = {}

    def progress(phase, **details):
        status = {
            "phase": phase,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        with lock:
            active.clear()
            active.update(status, pid=os.getpid())
            write_json(args.status, active)
        print(status, flush=True)

    def heartbeat():
        while not stopped.wait(30):
            with lock:
                if not active:
                    continue
                try:
                    items = (
                        records(Path(active["bundle"])) if active.get("bundle") else []
                    )
                    active.update(
                        observed_at=datetime.now(timezone.utc).isoformat(),
                        candidates=len(items),
                        static_passed=sum(r.static_passed for r in items),
                        text_checked=sum(r.text_checked for r in items),
                        accepted=sum(r.accepted for r in items),
                    )
                    write_json(args.status, active)
                except (OSError, ValueError):
                    pass  # Stage execution owns errors; observation must not abort it.

    observer = threading.Thread(target=heartbeat, daemon=True)
    observer.start()

    try:
        config = read_config(args.config)
        build_admitted_bundle(args.pilot_bundle, 20, config, progress)
        build_admitted_bundle(args.output, 200, config, progress)
        if args.collect_sft:
            progress("collect-sft", bundle=str(args.output))
            report = collect_sft(args.output)
            progress("complete", bundle=str(args.output), sft=report)
        else:
            progress("complete", bundle=str(args.output))
    except Exception as exc:
        progress("stopped", error=error_info(exc))
        raise SystemExit(1) from None
    finally:
        stopped.set()
        observer.join(timeout=1)


if __name__ == "__main__":
    main()
