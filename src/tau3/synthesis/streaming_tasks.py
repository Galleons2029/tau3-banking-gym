"""Retarget frozen drafts and admit/export each task without a whole-batch barrier."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau3.data_model.simulation import SimulationRun
from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
from tau3.synthesis.stream_resume import check_migration, check_trial, replay_blind
from tau3.synthesis.teacher_sampling import (
    TeacherPolicy,
    historical_rejection,
    run_teacher_trial,
    teacher_sample,
)
from tau3.synthesis.world_sft import (
    capture_trial,
    configure,
    export_shard,
    producer_hash,
    successful,
    trial_sample,
)
from tau3.synthesis.world_tasks import (
    STYLES,
    adapter_hash,
    balanced_anchors,
    expression_errors,
    task_structures,
)
from tau3.utils.llm_concurrency import file_lock
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.blind import (
    Witness,
    check_frozen_witness,
    public_problem,
    solve_public,
)
from tau3.worldgen.v2.pipeline import (
    artifact_hashes,
    check_certificate,
    implementation_hash,
    write_json,
)
from tau3.worldgen.v2.readiness import check_readiness, evidence_files
from tau3.worldgen.v2.runtime import digest

REWRITE = 'Rewrite customer-simulator instructions in the requested style. Preserve every fact, identifier, numeric literal, requirement and consent condition. Do not add an answer, product choice, operation name or bank policy. Return JSON {"request":"complete rewritten instructions"}.'
REVIEW = 'Compare the original and rewritten customer instructions. Require the same identity, facts, numbers, requested operations, preferences, permission requirements and disclosure behavior. Reject any new answer, policy, tool name or product choice. Return JSON {"equivalent":true,"no_solution_added":true,"issues":[]}.'


def reuse_request(session, source: Path, request: dict):
    """Copy exact raw responses with provenance; charge old attempts conservatively."""
    key = digest(request)
    target = session.output / "calls" / f"{key}.json"
    excluded = session.output / "source_interruptions" / f"{key}.json"
    with file_lock(session.output / "locks" / f"{key}.lock"):
        original = source / "calls" / f"{key}.json"
        if target.exists() or excluded.exists() or not original.exists():
            return
        record = json.loads(original.read_text())
        if record["request"] != request:
            raise AuditIncomplete("Source request hash mismatch")
        origin = {"source": str(original.resolve()), "record_hash": digest(record)}
        # Completed negative/invalid responses are copied too. Only a request with
        # no returned content may be retried after the explicit 8k -> 6k cutover.
        interrupted = record.get("status") != "COMPLETE" and not record.get("response")
        with file_lock(session.output / "budget.lock"):
            path = session.output / "budget.json"
            budget = json.loads(path.read_text())
            budget["used"] += len(record.get("attempts", [None]))
            if budget["used"] > session.max_calls:
                raise AuditIncomplete("Imported requests exhausted the audit budget")
            write_json(path, budget)
            if interrupted:
                write_json(
                    excluded,
                    {
                        "origin": origin,
                        "record": record,
                        "reason": "Explicit retarget restart; no model response was available",
                    },
                )
            else:
                write_json(target, {**record, "origin": origin})


def claim_expression(output: Path, expression: str, owner: str):
    """Persist dedup ownership so completion order cannot change it on resume."""
    path = output / "expressions" / f"{digest(expression)}.json"
    with file_lock(output / "expressions.lock"):
        if path.exists() and json.loads(path.read_text())["owner"] != owner:
            raise ValueError("Duplicate task expression")
        write_json(path, {"owner": owner})


def check_admitted(output: Path, record: dict, identity: dict):
    """Check immutable per-task evidence before reuse or training export."""
    if record["identity_hash"] != digest(identity) or record["status"] != "published":
        raise ValueError("Stale task admission")
    evidence_root = Path(record.get("evidence_root", output))
    directory = evidence_root / "slots" / f"{record['slot']:06d}"
    if evidence_files(directory) != record["evidence"]:
        raise ValueError("Task verification evidence changed")
    for key, expected in record["audit_refs"].items():
        if (
            digest(
                json.loads((evidence_root / "audit/calls" / f"{key}.json").read_text())
            )
            != expected
        ):
            raise ValueError("Raw audit response changed")


def stream_tasks(
    world: Path,
    source: Path,
    pilot: Path,
    config: Path,
    output: Path,
    count=6000,
    workers=128,
    resume_from: Path | None = None,
    teacher_policy: TeacherPolicy | None = None,
):
    """Reuse original public requests; publish each independently admitted task."""
    from tau3.domains.banking_synth.environment import get_tasks
    from tau3.synthesis.bundle import load_task_bundle

    with file_lock(output / "job.lock"):
        settings = configure(world, config)
        migration = (
            check_migration(output, resume_from, world, settings)
            if resume_from
            else None
        )
        history = ([resume_from] if resume_from else []) + [
            Path(p) for p in (migration or {}).get("historical_sources", [])
        ]
        if len(set(settings.agent_models)) != 2:
            raise ValueError(
                "Two distinct independent verification models are required"
            )
        if teacher_policy and teacher_policy.model not in settings.agent_models:
            raise ValueError("Teacher must be one of the qualified models")
        previous = json.loads((source / "manifest.json").read_text())
        source_audit = json.loads((source / "audit/audit.json").read_text())
        artifacts = artifact_hashes(world)
        if not 0 < count <= previous["requested_tasks"]:
            raise ValueError("Retarget count must fit the original slot plan")
        for key, value in {
            "source_artifacts": artifacts,
            "implementation": implementation_hash(),
            "adapter_hash": adapter_hash(),
            "settings_hash": digest(settings.model_dump()),
        }.items():
            if previous.get(key) != value:
                raise ValueError("Source draft world/adapter/settings changed")
        if any(
            source_audit.get(k) != v
            for k, v in {
                "artifacts": artifacts,
                "implementation": implementation_hash(),
                "settings": digest(settings.model_dump()),
                "kind": "task-expression-v1",
            }.items()
        ):
            raise ValueError("Source audit identity changed")
        pilot_tasks = load_task_bundle(pilot)
        if len(pilot_tasks) != 20:
            raise ValueError("A published 20-task pilot is required")
        readiness = Path(previous["readiness"])
        check_readiness(world, readiness, config)
        anchors = balanced_anchors(world, get_tasks())
        structures = task_structures(world)
        seed_test = set(json.loads((world / "splits.json").read_text())["test"])
        training_target = sum(
            anchors[i % len(anchors)].id not in seed_test for i in range(count)
        )
        identity = {
            "schema_version": 1,
            "producer": "streaming-world-tasks",
            "producer_hash": producer_hash(),
            "source_manifest": digest(previous),
            "source": str(source.resolve()),
            "source_artifacts": artifacts,
            "settings_hash": digest(settings.model_dump()),
            "readiness_hash": digest(json.loads(readiness.read_text())),
            "pilot_hash": digest(json.loads((pilot / "manifest.json").read_text())),
            "requested_tasks": count,
            "training_task_target": training_target,
            "sft_target": training_target * 2,
            "max_calls": count * 30,
            "max_rollouts": count * 6,
            "slot_policy": "First requested source slots, unchanged anchor and source task namespace",
            "scope": "Existing business structures; immutable per-task admission enables incremental SFT",
        }
        if teacher_policy:
            identity["teacher_policy"] = teacher_policy.model_dump(mode="json")
        if migration:
            identity["migration_hash"] = digest(migration)
        manifest = output / "manifest.json"
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["identity"] != identity
        ):
            raise ValueError("Streaming job identity changed")
        write_json(manifest, {"identity": identity, "status": "running"})
        for task in pilot_tasks:
            claim_expression(
                output, task.user_scenario.instructions, "pilot:" + task.id
            )
        if resume_from:
            for path in (resume_from / "expressions").glob("*.json"):
                target = output / "expressions" / path.name
                value = json.loads(path.read_text())
                if target.exists() and json.loads(target.read_text()) != value:
                    raise AuditIncomplete("Previous expression ownership changed")
                write_json(target, value)
        session = ConcurrentAuditSession(
            world, output / "audit", settings, count * 30, "task-expression-v1"
        )
        binding = {"stream_identity": digest(identity), "world": artifacts}

        def publish_sft(record):
            if record["split"] != "train":
                return 0
            trial_dir = (
                Path(record.get("evidence_root", output))
                / "slots"
                / f"{record['slot']:06d}"
                / f"attempt_{record['attempt']}"
            )
            if teacher_policy:
                from tau3.data_model.tasks import Task

                task = Task.model_validate(record["task"])
                samples, failures = [], []
                for seed in teacher_policy.seeds:
                    destination = (
                        output / "teacher-trials" / f"{record['slot']:06d}" / str(seed)
                    )
                    try:
                        run_teacher_trial(
                            task,
                            settings,
                            teacher_policy.model,
                            destination,
                            seed,
                            {**binding, "admission_hash": digest(record)},
                            world,
                            [trial_dir / f"online_{i}" for i in range(2)],
                            capture_trial,
                        )
                        samples.append(
                            teacher_sample(destination, teacher_policy.model, seed)
                        )
                    except (AuditIncomplete, ValueError) as exc:
                        failures.append(
                            {
                                "seed": seed,
                                "type": type(exc).__name__,
                                "reason": str(exc),
                            }
                        )
                write_json(
                    output / "teacher-trials" / f"{record['slot']:06d}" / "report.json",
                    {
                        "model": teacher_policy.model,
                        "samples": len(samples),
                        "failures": failures,
                        "admission_hash": digest(record),
                    },
                )
            else:
                samples = [trial_sample(trial_dir / f"online_{i}") for i in range(2)]
            return export_shard(
                output / "sft/shards" / f"{record['slot']:06d}.jsonl",
                samples,
                "train",
                {
                    "admission": str(
                        (output / "admitted" / f"{record['slot']:06d}.json").resolve()
                    ),
                    "admission_hash": digest(record),
                    "producer_hash": producer_hash(),
                },
            )

        def slot(index):
            # Certificate/code errors are infrastructure failures, never rejected
            # customer expressions. Frozen releases make drift unlikely, not safe.
            try:
                check_certificate(world)
            except ValueError as exc:
                raise RuntimeError("Stream world integrity changed") from exc
            admitted_path = output / "admitted" / f"{index:06d}.json"
            if admitted_path.exists():
                record = json.loads(admitted_path.read_text())
                check_admitted(output, record, identity)
                return record, publish_sft(record)
            if resume_from:
                old_path = resume_from / "admitted" / f"{index:06d}.json"
                if old_path.exists():
                    record = json.loads(old_path.read_text())
                    old_identity = json.loads(
                        (resume_from / "manifest.json").read_text()
                    )["identity"]
                    check_admitted(resume_from, record, old_identity)
                    evidence_root = Path(record.get("evidence_root", resume_from))
                    replay_blind(world, evidence_root, record, settings)
                    from tau3.data_model.tasks import Task

                    task = Task.model_validate(record["task"])
                    trial_dir = (
                        evidence_root
                        / "slots"
                        / f"{index:06d}"
                        / f"attempt_{record['attempt']}"
                    )
                    for i in range(2):
                        check_trial(trial_dir / f"online_{i}", task, world, settings)
                    record = {
                        **record,
                        "identity_hash": digest(identity),
                        "evidence_root": str(evidence_root.resolve()),
                        "origin": {
                            "record": str(old_path.resolve()),
                            "hash": digest(json.loads(old_path.read_text())),
                        },
                    }
                    write_json(admitted_path, record)
                    return record, publish_sft(record)
            anchor = anchors[index % len(anchors)]
            directory = output / "slots" / f"{index:06d}"
            refs = {}

            def ask(model, system, payload):
                request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                }
                key = digest(request)
                for historical in history:
                    reuse_request(session, historical / "audit", request)
                reuse_request(session, source / "audit", request)
                result = session.ask(model, system, payload)
                refs[key] = digest(
                    json.loads((session.output / "calls" / f"{key}.json").read_text())
                )
                return result

            for attempt in range(3):
                trial_dir = directory / f"attempt_{attempt}"
                rejected = trial_dir / "rejected.json"
                if rejected.exists():
                    continue
                old_trial = (
                    resume_from / "slots" / f"{index:06d}" / f"attempt_{attempt}"
                    if resume_from
                    else None
                )
                prior_rejection = historical_rejection(
                    [
                        historical / "slots" / f"{index:06d}" / f"attempt_{attempt}"
                        for historical in history
                    ]
                )
                if prior_rejection:
                    write_json(rejected, prior_rejection)
                    continue
                try:
                    style = STYLES[
                        (index + index // len(anchors) + attempt) % len(STYLES)
                    ]
                    response = ask(
                        settings.agent_models[index % 2],
                        REWRITE,
                        {
                            "instructions": anchor.user_scenario.instructions,
                            "style": style,
                            "variant": [index, attempt],
                            "batch": previous["bundle_id"],
                        },
                    )
                    instructions = response.get("request")
                    errors = expression_errors(
                        anchor.user_scenario.instructions, instructions
                    )
                    if errors:
                        raise ValueError(
                            "Task expression rejected: " + "; ".join(errors)
                        )
                    review = ask(
                        settings.agent_models[1 - index % 2],
                        REVIEW,
                        {
                            "original": anchor.user_scenario.instructions,
                            "rewritten": instructions,
                        },
                    )
                    if (
                        review.get("equivalent") is not True
                        or review.get("no_solution_added") is not True
                        or review.get("issues") != []
                    ):
                        raise ValueError("Task expression review failed")
                    task = anchor.model_copy(deep=True)
                    task.id = f"synth_{previous['bundle_id']}_{index:06d}_a{attempt}_{anchor.id}"
                    task.user_scenario.instructions = instructions
                    if instructions == anchor.user_scenario.instructions:
                        raise ValueError("Unchanged task expression")
                    claim_expression(output, instructions, task.id)
                    write_json(
                        trial_dir / "candidate.json",
                        {
                            "task": task.model_dump(mode="json"),
                            "review": review,
                            "anchor_id": anchor.id,
                        },
                    )
                    problem = public_problem(world, task.id, instructions=instructions)
                    for i, model in enumerate(settings.agent_models):
                        solution, cross_review, attempts = solve_public(
                            problem, model, settings.agent_models[1 - i], ask
                        )
                        check_frozen_witness(
                            world, anchor.id, Witness.model_validate(solution)
                        )
                        write_json(
                            trial_dir / f"blind_{i}.json",
                            {
                                "status": "PASS",
                                "model": model,
                                "solution": solution,
                                "review": cross_review,
                                "attempts": attempts,
                            },
                        )
                    outcomes = []
                    for i, verification_model in enumerate(settings.agent_models):
                        model = (
                            teacher_policy.model
                            if teacher_policy
                            else verification_model
                        )
                        seed = teacher_policy.seeds[i] if teacher_policy else 42 + i
                        destination = trial_dir / f"online_{i}"
                        if teacher_policy:
                            outcome = run_teacher_trial(
                                task,
                                settings,
                                model,
                                destination,
                                seed,
                                binding,
                                world,
                                [
                                    historical
                                    / "slots"
                                    / f"{index:06d}"
                                    / f"attempt_{attempt}"
                                    / f"online_{j}"
                                    for historical in history
                                    for j in range(2)
                                ],
                                capture_trial,
                            )
                        else:
                            old_online = (
                                old_trial / f"online_{i}" if old_trial else None
                            )
                            if old_online and (old_online / "started.json").exists():
                                if not (old_online / "result.json").exists():
                                    raise AuditIncomplete(
                                        "Previous interrupted teacher retained; slot deferred"
                                    )
                                outcome = check_trial(old_online, task, world, settings)
                                # Preserve the original identities and bytes. The new
                                # per-task admission records compatibility separately.
                                import shutil

                                shutil.copytree(
                                    old_online,
                                    trial_dir / f"online_{i}",
                                    dirs_exist_ok=True,
                                )
                            else:
                                outcome = capture_trial(
                                    task,
                                    settings,
                                    model,
                                    trial_dir / f"online_{i}",
                                    42 + i,
                                    binding,
                                )
                        outcomes.append(outcome)
                    passes = [
                        successful(SimulationRun.model_validate(item["simulation"]))
                        for item in outcomes
                    ]
                    if not all(passes):
                        raise ValueError("At least one real model conversation failed")
                    # Capture validation is required before admitting a task whose
                    # successful train conversations will become teacher examples.
                    for i in range(2):
                        if teacher_policy:
                            teacher_sample(
                                trial_dir / f"online_{i}",
                                teacher_policy.model,
                                teacher_policy.seeds[i],
                            )
                        else:
                            trial_sample(trial_dir / f"online_{i}")
                    record = {
                        "status": "published",
                        "identity_hash": digest(identity),
                        "slot": index,
                        "attempt": attempt,
                        "task": task.model_dump(mode="json"),
                        "anchor_id": anchor.id,
                        "group": structures[anchor.id],
                        "split": "validation" if anchor.id in seed_test else "train",
                        "audit_refs": refs,
                        "evidence": evidence_files(directory),
                    }
                    write_json(admitted_path, record)
                    return record, publish_sft(record)
                except AuditIncomplete:
                    raise
                except ValueError as exc:
                    try:
                        check_certificate(world)
                    except ValueError as integrity_error:
                        raise RuntimeError(
                            "Stream world integrity changed"
                        ) from integrity_error
                    write_json(
                        rejected,
                        {"status": "REJECTED", "reason": str(exc), "attempt": attempt},
                    )
            raise ValueError("Slot exhausted three candidates; evidence retained")

        records, failures, samples = {}, [], 0
        cancelled = 0

        def progress(status):
            write_json(
                output / "progress.json",
                {
                    "status": status,
                    "requested_tasks": count,
                    "admitted_tasks": len(records),
                    "sft_samples": samples,
                    "sft_target": training_target * 2,
                    "cancelled_slots": cancelled,
                    "completed_slots": len(records) + len(failures),
                    "failures": failures,
                },
            )

        progress("RUNNING")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(slot, i): i for i in range(count)}
            for future in as_completed(futures):
                index = futures[future]
                if future.cancelled():
                    cancelled += 1
                    continue
                try:
                    record, added = future.result()
                    records[index] = record
                    samples += added
                except Exception as exc:
                    failures.append(
                        {"slot": index, "reason": str(exc), "type": type(exc).__name__}
                    )
                    # A local transport/teacher interruption must not cancel all
                    # other independent slots. Global integrity failures do.
                    if isinstance(exc, RuntimeError):
                        for pending in futures:
                            pending.cancel()
                progress("RUNNING")
        ordered = [records[i] for i in sorted(records)]
        write_json(output / "tasks.json", [r["task"] for r in ordered])
        write_json(
            output / "split_tasks.json",
            {
                "base": [r["task"]["id"] for r in ordered],
                "train": [r["task"]["id"] for r in ordered if r["split"] == "train"],
                "validation": [
                    r["task"]["id"] for r in ordered if r["split"] == "validation"
                ],
            },
        )
        status = (
            "COMPLETE"
            if len(records) == count and samples == training_target * 2
            else "INCOMPLETE"
        )
        write_json(
            manifest,
            {
                "identity": identity,
                "status": status,
                "admitted_tasks": len(records),
                "sft_samples": samples,
                "failures": failures,
            },
        )
        progress(status)
        return {
            "status": status,
            "admitted_tasks": len(records),
            "sft_samples": samples,
        }
