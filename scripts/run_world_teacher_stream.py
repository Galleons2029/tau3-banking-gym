"""Supervise a separately identified teacher-policy migration of a qualified world."""

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
    """Read deployment metadata before importing the frozen release."""
    return json.loads(Path(path).read_text())


def worker(deployment):
    """Keep seed verification settings fixed and run the selected teacher policy."""
    import litellm

    from tau3.synthesis import streaming_tasks as stream
    from tau3.synthesis.teacher_sampling import TeacherPolicy
    from tau3.synthesis.world_sft import producer_hash
    from tau3.utils import llm_utils
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.audit import AuditIncomplete, AuditSession
    from tau3.worldgen.v2.pipeline import (
        artifact_hashes,
        check_certificate,
        implementation_hash,
        write_json,
    )
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.settings import load_settings

    output = Path(deployment["output"])
    previous = Path(deployment["resume_from"])
    world = Path(deployment["world"])
    settings = load_settings(Path(deployment["config"]))
    if producer_hash() != deployment["producer_hash"]:
        raise ValueError("Teacher producer release changed")
    if implementation_hash() != deployment["implementation"]:
        raise ValueError("Qualified world runtime changed")
    if digest(settings.model_dump()) != deployment["settings_hash"]:
        raise ValueError("Independent verification settings changed")
    if digest(read(previous / "manifest.json")) != deployment["source_manifest"]:
        raise ValueError("Stopped source manifest changed")
    check_certificate(world)
    module_path = Path(deployment["transport_module"])
    if (
        hashlib.sha256(module_path.read_bytes()).hexdigest()
        != deployment["transport_sha256"]
    ):
        raise ValueError("HTTP transport release changed")
    spec = importlib.util.spec_from_file_location("teacher_http_transport", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sync, asynchronous = module.make_http_clients(10, 5)
    old_sync, old_async = litellm.client_session, litellm.aclient_session
    litellm.client_session, litellm.aclient_session = sync, asynchronous
    old_sync.close()
    asyncio.run(old_async.aclose())
    if llm_utils.completion is not litellm.completion:
        raise ValueError("Unexpected prior request wrapper")
    limit = int(os.environ["TAU3_LLM_CONCURRENCY"])
    if (
        limit != deployment["request_concurrency"]
        or sync._transport._pool._max_connections != limit
    ):
        raise ValueError("HTTP/request admission capacity mismatch")
    policy = TeacherPolicy.model_validate(deployment["teacher_policy"])
    write_json(
        output / "worker.json",
        {
            "pid": os.getpid(),
            "producer_hash": producer_hash(),
            "teacher_policy": policy.model_dump(mode="json"),
            "verification_models": settings.agent_models,
            "user_model": settings.user_model,
            "judge_model": settings.judge_model,
            "api_base": settings.llm_args["api_base"],
            "http_max_connections": sync._transport._pool._max_connections,
            "http_max_keepalive_connections": sync._transport._pool._max_keepalive_connections,
            "async_http_max_connections": asynchronous._transport._pool._max_connections,
            "workers": limit,
            "started_at": time.time(),
        },
    )
    # A new output identity records the policy change. Old seed certificates,
    # task records, raw grades and actor settings are never rewritten.
    migration = {
        "status": "PASS",
        "source": str(previous.resolve()),
        "source_manifest": deployment["source_manifest"],
        "implementation": implementation_hash(),
        "artifacts": artifact_hashes(world),
        "settings": digest(settings.model_dump()),
        "historical_sources": deployment["historical_sources"],
        "teacher_policy": policy.model_dump(mode="json"),
        "producer_hash": producer_hash(),
        "release_checks": deployment["release_checks"],
        "scope": "Seed runtime and verifier settings unchanged; each imported admission and trajectory is verified before reuse; new teacher selection has a separate stream identity",
    }
    migration_path = output / "migration.json"
    if migration_path.exists() and read(migration_path) != migration:
        raise ValueError("Teacher migration identity changed")
    write_json(migration_path, migration)
    budget_path = output / "physical-rollout-budget.json"
    if not budget_path.exists():
        write_json(
            budget_path,
            {
                "used": deployment["charged_rollouts"],
                "limit": deployment["max_rollouts"],
            },
        )
    audit_budget = output / "audit/budget.json"
    if not audit_budget.exists():
        session = AuditSession(
            world,
            output / "audit",
            settings,
            deployment["requested_tasks"] * 30,
            "task-expression-v1",
        )
        write_json(
            audit_budget,
            {
                "identity": digest(session.identity),
                "used": deployment["charged_audit_calls"],
            },
        )
    capture = stream.capture_trial

    def bounded_capture(task, settings, model, directory, seed, binding):
        if model != policy.model:
            raise ValueError("A non-policy teacher was requested")
        with file_lock(output / "physical-rollout-budget.lock"):
            reservation = (
                output / "rollout-reservations" / f"{digest(str(directory))}.json"
            )
            if not (directory / "started.json").exists() and not reservation.exists():
                budget = read(budget_path)
                if budget["used"] >= budget["limit"]:
                    raise AuditIncomplete("Physical rollout budget exhausted")
                budget["used"] += 1
                write_json(budget_path, budget)
                write_json(
                    reservation,
                    {
                        "model": model,
                        "seed": seed,
                        "directory": str(directory),
                        "at": time.time(),
                    },
                )
        return capture(task, settings, model, directory, seed, binding)

    stream.capture_trial = bounded_capture
    root = Path(deployment["root"])
    result = stream.stream_tasks(
        world,
        root / "task-batch-8000",
        root / "task-pilot-20",
        Path(deployment["config"]),
        output,
        deployment["requested_tasks"],
        workers=limit,
        resume_from=previous,
        teacher_policy=policy,
    )
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


def main():
    """Run one background worker, exposing authoritative liveness and model policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    deployment = read(args.deployment)
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != deployment["launcher_sha256"]
    ):
        raise ValueError("Teacher launcher changed")
    sys.path.insert(0, str(Path(deployment["frozen_code"]) / "src"))
    if args.worker:
        raise SystemExit(worker(deployment))
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.pipeline import write_json

    output = Path(deployment["output"])
    with file_lock(output / "supervisor.lock"):
        with (output / "tasks.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--deployment",
                    str(args.deployment),
                    "--worker",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
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
                "workers": {"tasks": {"pid": child.pid, "exit_code": code}},
                "requested_tasks": deployment["requested_tasks"],
                "request_concurrency": deployment["request_concurrency"],
                "http_max_connections": deployment["request_concurrency"],
                "teacher_policy": deployment["teacher_policy"],
                "output": str(output),
                "updated_at": time.time(),
                "deployment": str(args.deployment),
                "phase": "glm_teacher_stream",
            }
            write_json(output / "run-state.json", state)
            for name in ("batch-state.json", "resume-state.json"):
                write_json(Path(deployment["status_root"]) / name, state)
            if code is not None:
                raise SystemExit(code)
            time.sleep(10)


if __name__ == "__main__":
    main()
