"""Checkpointed batch generation, live acceptance, publication and SFT collection."""

import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from tau3.synthesis.catalog import build_catalog
from tau3.synthesis.llm import (
    ReviewBudgetExceeded,
    StructuredResponseError,
    explicit_requirement_errors,
    json_response,
    narrate,
    response_log,
    review,
    visible_sft_sample,
)
from tau3.synthesis.models import (
    FAMILIES,
    VERSION,
    GenerationRecord,
    RuleCatalog,
    SynthesisConfig,
)
from tau3.synthesis.scenarios import customer_age, sample_candidate, solve_selection
from tau3.synthesis.storage import (
    digest,
    environment_fingerprint,
    generator_fingerprint,
    read_json,
    retrieval_settings,
    write_json,
)
from tau3.synthesis.validation import validate_static


def error_info(exc: Exception) -> str:
    """Avoid persisting provider exception strings, which can contain credentials."""
    if isinstance(
        exc, (StructuredResponseError, ValueError, AssertionError, KeyError, TypeError)
    ):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: status={getattr(exc, 'status_code', None)}"


def read_config(path=None) -> SynthesisConfig:
    """Read JSON or YAML configuration; defaults come from the model."""
    if path is None:
        return SynthesisConfig()
    import yaml

    return SynthesisConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def records(root: Path) -> list[GenerationRecord]:
    """Read candidate checkpoints in stable slot order."""
    return [
        GenerationRecord.model_validate(read_json(p))
        for p in sorted((root / "candidates").glob("*.json"))
    ]


def checkpoint(root: Path, record: GenerationRecord):
    write_json(
        root / "candidates" / f"{record.slot:06d}.json", record.model_dump(mode="json")
    )


def check_workspace(root: Path):
    manifest = read_json(root / "manifest.json")
    if manifest["environment_hash"] != environment_fingerprint():
        raise ValueError(
            "Environment changed since generation; use a fresh output directory"
        )
    return manifest


def probe(config: SynthesisConfig):
    """Fail early on unavailable model services before scheduling an expensive batch."""
    obj, _ = json_response(
        config.generator_model,
        'Return JSON {"ready": true}.',
        {},
        config,
        "synthesis_probe",
    )
    if not obj["ready"]:
        raise StructuredResponseError("Endpoint probe did not report ready")


def generate_bundle(root: Path, count: int, config: SynthesisConfig, resume=False):
    """Generate bounded candidates; only export can publish admitted tasks."""
    if count < 4 or count % 4:
        raise ValueError("num_tasks must be a positive multiple of four")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output exists; use --resume or a fresh output directory")
        manifest = check_workspace(root)
        if (
            manifest["config"] != config.model_dump()
            or manifest["requested_tasks"] != count
        ):
            raise ValueError("Resume configuration/count differs from checkpoint")
        catalog = RuleCatalog.model_validate(read_json(root / "catalog.json"))
    else:
        catalog = build_catalog()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        manifest = {
            "schema_version": VERSION,
            "benchmark": "tau3-AA",
            "domain": "banking_knowledge",
            "status": "draft",
            "environment_hash": environment_fingerprint(),
            "git_commit": result.stdout.strip(),
            "generator_hash": generator_fingerprint(),
            "synthesis_revision": "fixed-tools-v3",
            "config": config.model_dump(),
            "retrieval": retrieval_settings(),
            "requested_tasks": count,
            "sampling_requirements": {"combined_ordering_tasks": count // 20},
            "catalog_hash": digest(catalog.model_dump(mode="json")),
        }
        write_json(root / "catalog.json", catalog.model_dump(mode="json"))
        write_json(manifest_path, manifest)
    if config.text_mode == "llm":
        with response_log(
            lambda evidence: write_json(
                root / "llm_calls" / "probe" / f"{uuid4().hex}.json", evidence
            )
        ):
            probe(config)

    def worker(slot):
        path = root / "candidates" / f"{slot:06d}.json"
        old = (
            GenerationRecord.model_validate(read_json(path)) if path.exists() else None
        )
        failed_live = (
            old is not None
            and not old.accepted
            and all(
                key in old.trials
                and not old.trials[key].get("infrastructure_error")
                and not old.trials[key].get("passed")
                and not old.trials[key].get("review_pending")
                for key in ("bm25_grep:0", "bm25_grep:1")
            )
        )
        if (
            old
            and old.static_passed
            and (old.text_checked or config.text_mode == "template")
            and not failed_live
        ):
            return
        if old and old.checks.get("candidate_exhausted"):
            return
        start = old.attempt if old else 0
        if failed_live:
            start = old.attempt + 1
            write_json(
                root / "rejections" / f"{slot:06d}_{old.attempt:02d}.json",
                old.model_dump(mode="json"),
            )
            if start >= config.candidate_attempts:
                old.checks["candidate_exhausted"] = True
                checkpoint(root, old)
                return
            old = None
        for attempt in range(start, config.candidate_attempts):
            # Seed namespace is stable when increasing worker count or resuming.
            candidate_seed = config.seed + slot * 1009 + attempt * 1000003
            record = (
                old
                if old and old.attempt == attempt and old.static_passed
                else sample_candidate(catalog, candidate_seed, slot, attempt)
            )
            record.text_mode = config.text_mode
            checkpoint(root, record)
            try:
                if not record.static_passed:
                    record.checks["static"] = validate_static(record, catalog)
                    record.static_passed = True
                    checkpoint(root, record)
                with response_log(
                    lambda evidence: write_json(
                        root / "llm_calls" / record.task.id / f"{uuid4().hex}.json",
                        evidence,
                    )
                ):
                    narrate(record, config, on_update=lambda: checkpoint(root, record))
                record.checks.pop("retryable_error", None)
                checkpoint(root, record)
                return
            except (StructuredResponseError, json.JSONDecodeError) as exc:
                record.errors.append(error_info(exc))
                record.checks["retryable_error"] = True
                checkpoint(root, record)
                return  # Keep this exact business skeleton; service retries have their own budget.
            except Exception as exc:
                record.errors.append(error_info(exc))
                checkpoint(root, record)
                if not isinstance(
                    exc, (ValueError, AssertionError, KeyError, TypeError)
                ):
                    raise  # Provider failure must not consume ten new business candidates.
                write_json(
                    root / "rejections" / f"{slot:06d}_{attempt:02d}.json",
                    record.model_dump(mode="json"),
                )
                if attempt + 1 == config.candidate_attempts:
                    record.checks["candidate_exhausted"] = True
                    checkpoint(root, record)
                old = None

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(worker, range(count)))
    return summarize(root)


def trial_config(config: SynthesisConfig, mode: str, seed: int):
    """Resolve identical dialogue settings for live runs and score-only migrations."""
    from tau3.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain="banking_knowledge",
        llm_agent=config.teacher_model,
        llm_user=config.user_model,
        llm_args_agent=config.llm_args,
        llm_args_user=config.llm_args,
        max_steps=config.max_steps,
        retrieval_config=mode,
        retrieval_config_kwargs={"top_k": 10} if mode in ("bm25_grep", "bm25") else {},
        seed=seed,
    )


def run_trial(
    root: Path,
    record: GenerationRecord,
    config: SynthesisConfig,
    mode: str,
    seed: int,
    purpose="validation",
) -> dict:
    """Save full simulation and actual teacher context before quality review."""
    from tau3.data_model.simulation import SimulationRun
    from tau3.runner.build import build_env_kwargs, build_text_orchestrator
    from tau3.runner.simulation import run_simulation

    run_config = trial_config(config, mode, seed)
    stem = f"{purpose}_{mode}_{seed}"
    directory = root / "trajectories" / record.task.id
    capture_path = directory / f"{stem}.capture.json"
    signature = digest(
        [record.task.model_dump(mode="json"), run_config.model_dump(mode="json")]
    )
    capture = read_json(capture_path) if capture_path.exists() else None
    if (
        capture
        and capture["input_hash"] == signature
        and not capture["retry_simulation"]
    ):
        simulation = SimulationRun.model_validate(read_json(directory / f"{stem}.json"))
        sample = capture["visible_sample"]
    else:
        orchestrator = build_text_orchestrator(run_config, record.task, seed=seed)
        kwargs = build_env_kwargs(
            "banking_knowledge", record.task, mode, run_config.retrieval_config_kwargs
        )
        simulation = run_simulation(orchestrator, env_kwargs=kwargs)
        sample = visible_sft_sample(orchestrator, record.task.id)
        sample["retrieval_config"] = mode
        write_json(directory / f"{stem}.json", simulation.model_dump(mode="json"))
        write_json(
            capture_path,
            {
                "input_hash": signature,
                "generator_hash": generator_fingerprint(),
                "retry_simulation": simulation.termination_reason.value
                in {
                    "infrastructure_error",
                    "agent_error",
                    "user_error",
                    "unexpected_error",
                    "timeout",
                },
                "visible_sample": sample,
            },
        )

    # The reviewer projects only agent-private retrieval bodies out of this native
    # transcript. Private user results remain available to check real notifications.
    # This review projection never enters agent inputs, scoring, or SFT exports.
    transcript = [
        m.model_dump(mode="json", exclude={"raw_data"})
        for m in simulation.get_messages()
    ]
    reward = simulation.reward_info.reward if simulation.reward_info else 0
    infrastructure = simulation.termination_reason.value in {
        "infrastructure_error",
        "agent_error",
        "user_error",
        "unexpected_error",
        "timeout",
    }
    reviewed, usage = (
        None,
        {
            "model_review_skipped": True,
            "reason": "DB failure or infrastructure failure",
        },
    )
    if reward == 1 and not infrastructure:
        with response_log(
            lambda evidence: write_json(
                directory / "llm_calls" / f"{uuid4().hex}.json", evidence
            )
        ):
            try:
                reviewed, usage = review(record, config, transcript)
            except ReviewBudgetExceeded:
                reviewed = None
                usage = {
                    "model_review_inconclusive": True,
                    "reason": "Structured review output budget exhausted",
                    "unassessed": [
                        "fact_consistent",
                        "no_answer_leak",
                        "follows_user_constraints",
                    ],
                }
                record.errors.append(
                    "Quality review inconclusive: output budget exhausted"
                )
    visible = sample["messages"]
    calls = [
        c
        for m in simulation.get_messages()
        if m.role == "assistant"
        for c in (getattr(m, "tool_calls", None) or [])
    ]
    retrieved = set()
    for message in visible:
        if message["role"] == "tool":
            retrieved.update(
                re.findall(r"\bID: (doc_[^\n]+)", message.get("content") or "")
            )
    reward = simulation.reward_info.reward if simulation.reward_info else 0
    infrastructure = simulation.termination_reason.value in {
        "infrastructure_error",
        "agent_error",
        "user_error",
        "unexpected_error",
        "timeout",
    }
    result = {
        "mode": mode,
        "seed": seed,
        "reward": reward,
        "passed": reward == 1
        and reviewed is not None
        and reviewed.passed
        and not infrastructure,
        "infrastructure_error": infrastructure,
        "review": reviewed.model_dump() if reviewed is not None else None,
        "review_usage": usage,
        "termination": simulation.termination_reason.value,
        "agent_cost": simulation.agent_cost,
        "user_cost": simulation.user_cost,
        "usage": [
            m.usage for m in simulation.get_messages() if getattr(m, "usage", None)
        ],
        "retrieval_calls": sum(c.name in {"KB_search", "grep"} for c in calls),
        "retrieved_evidence": sorted(retrieved & set(record.task.required_documents)),
        "evidence_coverage": len(retrieved & set(record.task.required_documents))
        / len(record.task.required_documents),
        "visible_context_chars": len(json.dumps(visible)),
        "trajectory": str((directory / f"{stem}.json").relative_to(root)),
    }
    if purpose == "sft" and mode == "bm25_grep" and result["passed"]:
        write_json(directory / f"{stem}.sft.json", sample)
        result["sft_path"] = str((directory / f"{stem}.sft.json").relative_to(root))
    return result


def validate_bundle(root: Path, resume=False, offline=False):
    """Run strict gates and two bm25_grep conversations per candidate."""
    manifest = check_workspace(root)
    config = SynthesisConfig.model_validate(manifest["config"])
    catalog = RuleCatalog.model_validate(read_json(root / "catalog.json"))
    items = records(root)
    if not offline:
        if config.text_mode != "llm":
            raise ValueError(
                "Template-only candidates cannot pass live admission; regenerate with text_mode=llm"
            )
        with response_log(
            lambda evidence: write_json(
                root / "llm_calls" / "probe" / f"{uuid4().hex}.json", evidence
            )
        ):
            probe(config)

    def worker(record):
        try:
            record.static_passed = False
            record.checks["static"] = validate_static(record, catalog)
            record.static_passed = True
            if offline:
                checkpoint(root, record)
                return
            if not record.text_checked:
                raise ValueError("Narrative has not passed independent review")
            if record.checks.get("reviewed_text_hash") != digest(
                record.task.user_scenario.model_dump(mode="json")
            ):
                raise ValueError("User scenario changed after text review")
            for trial in range(2):
                key = f"bm25_grep:{trial}"
                previous = record.trials.get(key)
                if previous and previous.get("trajectory"):
                    native = read_json(root / previous["trajectory"])
                    issues = explicit_requirement_errors(
                        record, native.get("messages") or []
                    )
                    previous["deterministic_checks"] = issues
                    if issues:
                        previous["passed"] = False
                if (
                    not resume
                    or not previous
                    or previous.get("infrastructure_error")
                    or previous.get("review_pending")
                ):
                    record.trials[key] = run_trial(
                        root, record, config, "bm25_grep", record.skeleton.seed + trial
                    )
                    checkpoint(root, record)
            record.accepted = any(
                record.trials[f"bm25_grep:{i}"]["passed"] for i in range(2)
            )
            record.checks["validated_content_hash"] = digest(
                [
                    record.task.model_dump(mode="json"),
                    record.skeleton.model_dump(mode="json"),
                ]
            )
        except Exception as exc:
            record.accepted = False
            record.errors.append(error_info(exc))
        checkpoint(root, record)

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(worker, items))
    return summarize(root)


def grouped_split(items: list[GenerationRecord]) -> dict[str, list[str]]:
    """Choose approximately 20% per family, keeping every structural group intact."""
    validation = set()
    for family in FAMILIES:
        grouped = defaultdict(list)
        for record in items:
            if record.skeleton.family == family:
                grouped[record.skeleton.group_id].append(record.task.id)
        target = round(sum(map(len, grouped.values())) / 5)
        possible = {0: []}
        for group, ids in sorted(grouped.items()):
            for size, selected in list(possible.items()):
                possible.setdefault(size + len(ids), selected + [group])
        best = min(possible, key=lambda n: (abs(n - target), n))
        for group in possible[best]:
            validation.update(grouped[group])
    ids = [r.task.id for r in items]
    return {
        "base": ids,
        "train": [i for i in ids if i not in validation],
        "validation": [i for i in ids if i in validation],
    }


def normalized_words(text: str) -> set:
    return set(re.findall(r"[a-z]+", re.sub(r"\b\w*\d\w*\b", "", text.lower())))


def export_bundle(root: Path):
    """Publish only independently reviewed tasks with successful target-mode trials."""
    from tau3.domains.banking_knowledge.environment import get_tasks

    manifest = check_workspace(root)
    original = [normalized_words(str(t.user_scenario)) for t in get_tasks()]
    admitted, rejected = [], []
    seen = set()
    for record in records(root):
        trials = [record.trials.get(f"bm25_grep:{i}", {}) for i in range(2)]
        ready = (
            record.accepted
            and record.static_passed
            and record.text_checked
            and record.text_mode == "llm"
        )
        ready = ready and record.checks.get("validated_content_hash") == digest(
            [
                record.task.model_dump(mode="json"),
                record.skeleton.model_dump(mode="json"),
            ]
        )
        ready = (
            ready
            and all(t.get("mode") == "bm25_grep" for t in trials)
            and any(t.get("passed") for t in trials)
        )
        words = normalized_words(str(record.task.user_scenario))
        duplicate = any(
            len(words & other) / max(1, len(words | other)) > 0.85 for other in original
        )
        fingerprint = digest([record.skeleton.family, record.skeleton.private])
        if not ready or duplicate or fingerprint in seen:
            rejected.append(
                {
                    "id": record.task.id,
                    "reason": "near_duplicate" if duplicate else "not_admitted",
                    "errors": record.errors,
                }
            )
            continue
        seen.add(fingerprint)
        admitted.append(record)
    if not admitted:
        write_json(
            root / "publication_report.json", {"published": 0, "rejected": rejected}
        )
        raise ValueError(
            "No task passed all publication gates; draft remains unpublished"
        )
    combined = sum(
        bool(r.skeleton.private.get("open_business"))
        for r in admitted
        if r.skeleton.family == "ordering"
    )
    required_combined = manifest.get("sampling_requirements", {}).get(
        "combined_ordering_tasks", 0
    )
    if combined < required_combined:
        raise ValueError(
            f"Missing combined opening/closure coverage: {combined}/{required_combined}"
        )
    splits = grouped_split(admitted)
    tasks = [r.task.model_dump(mode="json") for r in admitted]
    write_json(root / "tasks.json", tasks)
    write_json(root / "split_tasks.json", splits)
    for filename, rows in (
        (
            "metadata.jsonl",
            [r.model_dump(mode="json", exclude={"task"}) for r in admitted],
        ),
        ("rejected.jsonl", rejected),
    ):
        path = root / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        temporary.replace(path)
    manifest.update(
        {
            "status": "published",
            "tasks_hash": digest(tasks),
            "splits_hash": digest(splits),
            "task_groups": {r.task.id: r.skeleton.group_id for r in admitted},
            "published_tasks": len(tasks),
        }
    )
    write_json(root / "manifest.json", manifest)
    return summarize(root)


def collect_sft(root: Path, split="train"):
    """Sample at most four attempts and keep at most two reviewed successes per task."""
    from tau3.synthesis.bundle import load_task_bundle

    if split != "train":
        raise ValueError("SFT collection is restricted to the training split")
    published_tasks = {t.id: t for t in load_task_bundle(root, split)}
    tasks = set(published_tasks)
    config = SynthesisConfig.model_validate(read_json(root / "manifest.json")["config"])
    with response_log(
        lambda evidence: write_json(
            root / "llm_calls" / "probe" / f"{uuid4().hex}.json", evidence
        )
    ):
        probe(config)

    def worker(record):
        if record.task.id not in tasks:
            return
        if record.task != published_tasks[record.task.id]:
            raise ValueError("Candidate changed after publication; cannot collect SFT")
        successes = 0
        for index in range(4):
            key = f"sft:{index}"
            result = record.trials.get(key)
            if result is None:
                seed = record.skeleton.seed + 100 + index
                directory = root / "trajectories" / record.task.id
                capture_path = directory / f"sft_bm25_grep_{seed}.capture.json"
                native_path = directory / f"sft_bm25_grep_{seed}.json"
                capture = read_json(capture_path) if capture_path.exists() else None
                started = record.checks.setdefault("sft_attempts_started", {})
                consumed = key in started or capture is not None or native_path.exists()
                started.setdefault(key, {"seed": seed})
                # An interrupted simulation consumes a slot. Only its saved successful
                # native run may resume quality review without another teacher run.
                if consumed and (not capture or capture.get("retry_simulation")):
                    record.trials[key] = {
                        "mode": "bm25_grep",
                        "seed": seed,
                        "reward": 0,
                        "passed": False,
                        "infrastructure_error": True,
                        "reason": "Consumed SFT attempt has no reusable completed run",
                    }
                    checkpoint(root, record)
                    continue
                if capture and capture["input_hash"] != digest(
                    [
                        record.task.model_dump(mode="json"),
                        trial_config(config, "bm25_grep", seed).model_dump(mode="json"),
                    ]
                ):
                    raise ValueError(
                        "SFT cached run inputs changed; cannot resample slot"
                    )
                checkpoint(
                    root, record
                )  # Persist the budget before any simulation call.
                try:
                    result = run_trial(
                        root,
                        record,
                        config,
                        "bm25_grep",
                        seed,
                        "sft",
                    )
                    record.trials[key] = result
                    checkpoint(root, record)
                except Exception as exc:
                    record.errors.append(error_info(exc))
                    checkpoint(root, record)
                    break
            successes += int(result.get("passed", False))
            if successes >= 2:
                break

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(worker, records(root)))

    # A train task may exhaust all four dedicated SFT sampling slots despite having
    # a validated, quality-reviewed primary teacher trajectory. Reuse one such
    # trajectory as a fallback sample instead of exceeding the sampling budget.
    for record in records(root):
        if record.task.id not in tasks:
            continue
        sft_success = any(
            trial.get("passed")
            for key, trial in record.trials.items()
            if key.startswith("sft:")
        )
        if sft_success:
            continue
        source_key = next(
            (
                key
                for key in ("bm25_grep:0", "bm25_grep:1")
                if record.trials.get(key, {}).get("passed")
                and record.trials[key].get("trajectory")
            ),
            None,
        )
        if source_key is None:
            continue
        source = record.trials[source_key]
        capture = read_json((root / source["trajectory"]).with_suffix(".capture.json"))
        fallback_path = (
            root
            / "trajectories"
            / record.task.id
            / f"{source_key.replace(':', '_')}.sft.json"
        )
        write_json(fallback_path, capture["visible_sample"])
        record.checks["sft_primary_fallback"] = {
            "source_key": source_key,
            "sft_path": str(fallback_path.relative_to(root)),
        }
        checkpoint(root, record)

    samples, seen = [], set()
    for record in records(root):
        if record.task.id not in tasks:
            continue
        for index in range(4):
            result = record.trials.get(f"sft:{index}", {})
            if (
                result.get("passed")
                and result.get("mode") == "bm25_grep"
                and "sft_path" in result
            ):
                sample = read_json(root / result["sft_path"])
                signature = digest(sample["messages"])
                if signature not in seen:
                    samples.append(sample)
                    seen.add(signature)
        fallback = record.checks.get("sft_primary_fallback")
        if fallback:
            sample = read_json(root / fallback["sft_path"])
            signature = digest(sample["messages"])
            if signature not in seen:
                samples.append(sample)
                seen.add(signature)
    path = root / "sft.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
    )
    temporary.replace(path)
    covered = {sample["task_id"] for sample in samples}
    return {
        "sft_samples": len(samples),
        "training_tasks": len(tasks),
        "tasks_with_samples": len(covered),
        "tasks_without_success": sorted(tasks - covered),
    }


def summarize(root: Path) -> dict:
    """Persist actual counts, cost information and failures without inferring success."""
    items = records(root)
    report = {
        "requested": read_json(root / "manifest.json")["requested_tasks"],
        "candidates": len(items),
        "static_passed": sum(r.static_passed for r in items),
        "text_checked": sum(r.text_checked for r in items),
        "accepted": sum(r.accepted for r in items),
        "by_family": dict(Counter(r.skeleton.family for r in items)),
        "business_groups": len({r.skeleton.group_id for r in items}),
        "exhausted_slots": [
            r.slot for r in items if r.checks.get("candidate_exhausted")
        ],
        "coverage_exclusions": read_json(root / "catalog.json").get("exclusions", []),
        "failures": [
            {"task_id": r.task.id, "errors": r.errors} for r in items if r.errors
        ],
        "known_conversation_cost": sum(
            (t.get("agent_cost") or 0) + (t.get("user_cost") or 0)
            for r in items
            for t in r.trials.values()
        ),
        "cost_note": "Only provider-reported costs; missing prices are not zero-cost claims.",
    }
    write_json(root / "report.json", report)
    return report


def fork_repaired_bundle(source: Path, destination: Path, config: SynthesisConfig):
    """Preserve the old draft and fork checkpoints under a new synthesis revision.

    Only incompatible approval candidates are quarantined. Official environment
    checksums and successful compatible task/trial checkpoints remain unchanged.
    """
    import shutil

    source, destination = source.resolve(), destination.resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Repair destination must be separate from the source bundle")
    if destination.exists():
        raise ValueError(
            "Repair destination exists; resume it instead of forking again"
        )
    manifest = check_workspace(source)
    if manifest["status"] != "draft":
        raise ValueError("Repair only draft bundles; published bundles are immutable")
    old_config = SynthesisConfig.model_validate(manifest["config"])
    if config.text_mode != "llm" or old_config.text_mode != "llm":
        raise ValueError("Repair requires LLM-mode source and destination")
    dialogue_fields = (
        "llm_args",
        "teacher_model",
        "user_model",
        "max_steps",
        "retrieval_config",
        "top_k",
    )
    if any(
        getattr(old_config, field) != getattr(config, field)
        for field in dialogue_fields
    ):
        raise ValueError(
            "Repair must preserve dialogue settings to reuse completed trials"
        )
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.tmp"))
    history = destination / "history" / digest(manifest)[:12]
    write_json(history / "source_manifest.json", manifest)
    if (destination / "rejections").exists():
        (destination / "rejections").rename(history / "rejections")
    catalog = build_catalog()
    quarantined = []
    for record in records(destination):
        reason = None
        if (
            record.skeleton.family == "credit_limit"
            and record.skeleton.private.get("decision") == "approved"
        ):
            reason = "official submit/approve request ID collision"
        elif (
            record.skeleton.family == "ordering"
            and "Record my closure reason exactly" not in record.skeleton.facts["goal"]
        ):
            reason = "closure reason was not an explicit upfront user requirement"
        elif record.skeleton.family == "selection":
            private = record.skeleton.private
            expected = solve_selection(
                catalog.products,
                private["constraints"],
                age=customer_age(record.skeleton.facts["identity"]["date_of_birth"]),
            )
            missing_age_evidence = (
                private["constraints"]["mobile_check_deposit_per_day"] <= 500
                and catalog.products["Light Green Account"]["age_document"]
                not in record.task.required_documents
            )
            if expected != private["selected_product"] or missing_age_evidence:
                reason = "selection age eligibility or evidence incomplete"
        if reason:
            record.accepted = False
            record.errors.append("Quarantined: " + reason)
            write_json(
                destination / "quarantine" / f"{record.slot:06d}.json",
                record.model_dump(mode="json"),
            )
            (destination / "candidates" / f"{record.slot:06d}.json").unlink()
            quarantined.append(record.slot)
        else:
            repair_reference_routes(destination, record, config, catalog)
            record.checks["carried_from"] = str(source)
            if not record.text_checked:
                record.checks["narrative_attempts"] = 0
            checkpoint(destination, record)
    write_json(destination / "catalog.json", catalog.model_dump(mode="json"))
    manifest.update(
        config=config.model_dump(),
        generator_hash=generator_fingerprint(),
        catalog_hash=digest(catalog.model_dump(mode="json")),
        parent_bundle=str(source),
        quarantined_slots=quarantined,
        synthesis_revision="fixed-tools-v3",
    )
    write_json(destination / "manifest.json", manifest)
    return summarize(destination)


def repair_reference_routes(
    root: Path, record: GenerationRecord, config: SynthesisConfig, catalog: RuleCatalog
):
    """Repair hidden reference calls and regrade unchanged native dialogue inputs.

    The transformation is defined by official discovery metadata, never by the
    observed agent actions. User instructions, initial state, tools and policies
    are preserved. Only required write-tool audit records change in the target DB.
    """
    from tau3.data_model.simulation import SimulationRun
    from tau3.domains.banking_knowledge.environment import get_environment
    from tau3.environment.toolkit import MUTATES_STATE_ATTR
    from tau3.evaluator.evaluator_env import EnvironmentEvaluator
    from tau3.runner.build import build_env_kwargs
    from tau3.synthesis.scenarios import Actions
    from tau3.synthesis.validation import fresh_environment

    before = record.task.model_copy(deep=True)
    env = fresh_environment(before)
    hidden = env.tools.get_discoverable_tools()
    if not any(
        a.requestor == "assistant" and a.name in hidden
        for a in before.evaluation_criteria.actions
    ):
        return
    transformed = Actions()
    for action in before.evaluation_criteria.actions:
        if action.requestor == "assistant" and action.name in hidden:
            if not getattr(hidden[action.name], MUTATES_STATE_ATTR, False):
                raise ValueError("Score-only migration cannot add required read calls")
            transformed.call(action.name, **action.arguments)
        else:
            transformed.add(action.name, action.arguments, action.requestor)
            if action.name == "unlock_discoverable_agent_tool":
                transformed.unlocked.add(action.arguments["agent_tool_name"])
    record.task.evaluation_criteria.actions = transformed.items
    comparable = record.task.model_dump(mode="json")
    comparable["evaluation_criteria"]["actions"] = before.model_dump(mode="json")[
        "evaluation_criteria"
    ]["actions"]
    if comparable != before.model_dump(mode="json"):
        raise ValueError("Reference route repair changed dialogue inputs")
    record.checks["static"] = validate_static(record, catalog)
    record.static_passed = True
    archive = root / "history" / "reference_routes" / record.task.id
    write_json(archive / "task_before.json", before.model_dump(mode="json"))
    write_json(archive / "trials_before.json", record.trials)
    for key, trial in record.trials.items():
        if not trial.get("trajectory"):
            continue
        path = root / trial["trajectory"]
        native = read_json(path)
        write_json(archive / path.name, native)
        simulation = SimulationRun.model_validate(native)
        mode, seed = trial["mode"], trial["seed"]
        run_config = trial_config(config, mode, seed)
        kwargs = build_env_kwargs(
            "banking_knowledge", record.task, mode, run_config.retrieval_config_kwargs
        )
        # Run the unchanged official evaluator, including strict tool-output replay.
        reward = EnvironmentEvaluator.calculate_reward(
            get_environment,
            record.task,
            simulation.get_messages(),
            env_kwargs=kwargs,
            strict_replay=True,
        )
        simulation.reward_info = reward
        write_json(path, simulation.model_dump(mode="json"))
        review_result = trial.get("review") or {}
        checked = all(
            review_result.get(k) is True
            for k in ("fact_consistent", "no_answer_leak", "follows_user_constraints")
        )
        trial.update(
            reward=reward.reward,
            passed=reward.reward == 1
            and checked
            and not trial.get("infrastructure_error"),
            review_pending=reward.reward == 1 and not review_result,
            regraded_from_reference_hash=digest(
                before.evaluation_criteria.model_dump(mode="json")
            ),
        )
        capture_path = path.with_suffix(".capture.json")
        if capture_path.exists():
            capture = read_json(capture_path)
            expected = digest(
                [before.model_dump(mode="json"), run_config.model_dump(mode="json")]
            )
            if capture["input_hash"] != expected:
                raise ValueError(
                    "Cannot migrate a capture with mismatched original inputs"
                )
            capture["input_hash"] = digest(
                [
                    record.task.model_dump(mode="json"),
                    run_config.model_dump(mode="json"),
                ]
            )
            capture["score_only_migration"] = str(archive.relative_to(root))
            write_json(capture_path, capture)
    primary = [record.trials.get(f"bm25_grep:{i}") for i in range(2)]
    record.accepted = all(primary) and any(t.get("passed") for t in primary)
    record.checks["validated_content_hash"] = digest(
        [record.task.model_dump(mode="json"), record.skeleton.model_dump(mode="json")]
    )
    record.checks["reference_route_repair"] = {
        "source_task_hash": digest(before.model_dump(mode="json")),
        "target_task_hash": digest(record.task.model_dump(mode="json")),
        "dialogue_inputs_unchanged": True,
    }
