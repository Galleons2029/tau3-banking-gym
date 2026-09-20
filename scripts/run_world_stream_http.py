"""Apply an audited transport-only repair to an unchanged frozen task producer."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    """Read deployment metadata without importing the mutable workspace."""
    return json.loads(Path(path).read_text())


def load_transport(path, expected):
    """Load only the exact transport module recorded in the deployment receipt."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("HTTP transport release changed")
    spec = importlib.util.spec_from_file_location("tau3_http_transport_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_recovery(output, deployment, write_json, digest):
    """Archive only inconclusive current trials; retain every completed verdict.

    A trial is eligible once per transport repair. Its previous bytes are moved
    outside admitted evidence, never deleted. Previously admitted slots are never
    modified. Old-generation interruption records remain in their source tree.
    """
    root = output / "transport" / deployment["id"]
    plan_path = root / "recovery.json"
    if plan_path.exists():
        plan = read(plan_path)
        if plan["status"] == "ARCHIVED":
            return plan
        return finish_recovery(output, root, plan, write_json, digest)
    previous = Path(deployment["resume_from"])
    # Counting copied legacy markers twice is deliberately conservative.
    charged = sum(
        1 for _ in (output / "slots").glob("*/attempt_*/online_*/started.json")
    )
    charged += sum(
        1 for _ in (previous / "slots").glob("*/attempt_*/online_*/started.json")
    )
    rows = []
    for marker in sorted((output / "slots").glob("*/attempt_*/online_*/started.json")):
        directory = marker.parent
        slot = directory.parent.parent.name
        if (output / "admitted" / f"{slot}.json").exists():
            continue
        result_path = directory / "result.json"
        reason = None
        if result_path.exists():
            if read(result_path)["simulation"]["termination_reason"] == "timeout":
                reason = "rollout_timeout_before_HTTP_pool_repair"
        elif not (directory / "capture.json").exists():
            reason = "interrupted_without_completed_capture_at_transport_cutover"
        if reason is None:
            continue
        relative = directory.relative_to(output)
        # The frozen importer reads original legacy trials directly. Do not
        # pretend that moving their copies repairs those legacy interruptions.
        if (previous / relative / "started.json").exists():
            continue
        archive = root / "prior_trials" / relative
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        }
        row = {
            "trial": str(relative),
            "archive": str(archive),
            "reason": reason,
            "files": hashes,
        }
        rows.append(row)
    plan = {
        "status": "PREPARED",
        "trials": rows,
        "audit_calls": [
            {
                "source": str(path),
                "archive": str(root / "prior_calls" / path.name),
                "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted((output / "audit/calls").glob("*.json"))
            if (record := read(path)).get("status") != "COMPLETE"
            and not record.get("response")
        ],
        "already_charged_rollouts": charged,
        "max_rollouts": read(output / "manifest.json")["identity"]["max_rollouts"],
        "policy": "One explicit transport recovery; normal failed grades and admitted evidence are immutable",
    }
    write_json(plan_path, plan)
    return finish_recovery(output, root, plan, write_json, digest)


def finish_recovery(output, root, plan, write_json, digest):
    """Complete a prepared archive idempotently after a process interruption."""
    for row in plan.get("audit_calls", []):
        source, archive = Path(row["source"]), Path(row["archive"])
        present = archive if archive.exists() else source
        if hashlib.sha256(present.read_bytes()).hexdigest() != row["hash"]:
            raise ValueError("Interrupted audit response changed")
        if archive.exists():
            if source.exists():
                raise ValueError("Both archived and active audit records exist")
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        source.rename(archive)
    # The existing raw audit budget is not reduced when incomplete calls move.
    for row in plan["trials"]:
        directory, archive = output / row["trial"], Path(row["archive"])
        source = archive if archive.exists() else directory
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        }
        if hashes != row["files"]:
            raise ValueError("Recovery trial changed after preparation")
        if archive.exists():
            if directory.exists():
                raise ValueError("Both archived and active recovery trials exist")
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
    write_json(
        root / "rollout-budget.json",
        {
            "used": plan["already_charged_rollouts"],
            "limit": plan["max_rollouts"],
            "recovery": digest(plan),
        },
    )
    plan["status"] = "ARCHIVED"
    write_json(root / "recovery.json", plan)
    return plan


def worker(deployment):
    """Install HTTP clients before the first SDK client is constructed."""
    sys.path.insert(0, str(Path(deployment["frozen_code"]) / "src"))
    import litellm

    from tau3.synthesis import streaming_tasks as stream
    from tau3.synthesis.world_sft import producer_hash
    from tau3.utils import llm_utils
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.audit import AuditIncomplete
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest

    output = Path(deployment["output"])
    if producer_hash() != deployment["producer_hash"]:
        raise ValueError("Frozen task producer changed; transport-only resume refused")
    if (
        digest(read(output / "manifest.json")["identity"])
        != deployment["stream_identity"]
    ):
        raise ValueError("Stream identity changed")
    module = load_transport(
        Path(deployment["transport_module"]), deployment["transport_sha256"]
    )
    old_sync, old_async = litellm.client_session, litellm.aclient_session
    new_sync, new_async = module.make_http_clients(10, 5)
    litellm.client_session, litellm.aclient_session = new_sync, new_async
    old_sync.close()
    asyncio.run(old_async.aclose())
    actual = new_sync._transport._pool._max_connections
    if actual != int(os.environ["TAU3_LLM_CONCURRENCY"]):
        raise ValueError("HTTP capacity does not match request admission")
    if llm_utils.completion is not litellm.completion:
        raise ValueError("Unexpected preinstalled completion wrapper")
    runtime = output / "transport" / deployment["id"]
    recovery = read(runtime / "recovery.json")
    if recovery["status"] != "ARCHIVED":
        raise ValueError("Recovery archive is incomplete")
    write_json(
        runtime / "worker.json",
        {
            "pid": os.getpid(),
            "deployment": deployment,
            "http_max_connections": actual,
            "http_max_keepalive_connections": new_sync._transport._pool._max_keepalive_connections,
            "async_http_max_connections": new_async._transport._pool._max_connections,
            "started_at": time.time(),
        },
    )
    capture = stream.capture_trial

    def bounded_capture(task, settings, model, directory, seed, binding):
        # The original code has a finite slot/attempt budget. Explicit recovery
        # adds physical attempts, so additionally enforce its global rollout cap.
        with file_lock(runtime / "rollout-budget.lock"):
            reservation = (
                runtime / "rollout-reservations" / f"{digest(str(directory))}.json"
            )
            if not (directory / "started.json").exists() and not reservation.exists():
                budget = read(runtime / "rollout-budget.json")
                if budget["used"] >= budget["limit"]:
                    raise AuditIncomplete("Physical rollout budget exhausted")
                budget["used"] += 1
                write_json(runtime / "rollout-budget.json", budget)
                write_json(
                    reservation,
                    {"trial": str(directory), "model": model, "at": time.time()},
                )
        return capture(task, settings, model, directory, seed, binding)

    stream.capture_trial = bounded_capture
    root = Path(deployment["root"])
    world = Path(read(root / "local-worlds.json")["scale"])
    result = stream.stream_tasks(
        world,
        root / "task-batch-8000",
        root / "task-pilot-20",
        Path(deployment["config"]),
        output,
        6000,
        workers=int(os.environ["TAU3_LLM_CONCURRENCY"]),
        resume_from=Path(deployment["resume_from"]),
    )
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


def main():
    """Supervise one resumed producer with durable transport-specific provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    deployment = read(args.deployment)
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != deployment["launcher_sha256"]
    ):
        raise ValueError("Transport launcher changed")
    if args.worker:
        raise SystemExit(worker(deployment))
    sys.path.insert(0, str(Path(deployment["frozen_code"]) / "src"))
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest

    output = Path(deployment["output"])
    with file_lock(output / "supervisor.lock"):
        # No actor is running while recovery modifies non-admitted checkpoints.
        with file_lock(output / "job.lock"):
            prepare_recovery(output, deployment, write_json, digest)
        with (output / "tasks.log").open("a") as handle:
            child = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--deployment",
                    str(args.deployment),
                    "--worker",
                ],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        while True:
            code = child.poll()
            state = {
                "status": "RUNNING"
                if code is None
                else "COMPLETE"
                if code == 0
                else "INCOMPLETE",
                "supervisor_pid": os.getpid(),
                "requested_tasks": 6000,
                "request_concurrency": int(os.environ["TAU3_LLM_CONCURRENCY"]),
                "http_max_connections": int(os.environ["TAU3_LLM_CONCURRENCY"]),
                "workers": {"tasks": {"pid": child.pid, "exit_code": code}},
                "updated_at": time.time(),
                "output": str(output),
                "transport_deployment": str(args.deployment),
                "transport_timing": os.environ["TAU3_HTTP_TIMING_DIR"],
            }
            write_json(output / "run-state.json", state)
            write_json(Path(deployment["status_root"]) / "batch-state.json", state)
            write_json(
                Path(deployment["status_root"]) / "resume-state.json",
                {
                    **state,
                    "phase": "streaming_transport_repaired"
                    if code is None
                    else "finished",
                },
            )
            if code is not None:
                raise SystemExit(code)
            time.sleep(10)


if __name__ == "__main__":
    main()
