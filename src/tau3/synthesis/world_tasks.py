"""Bounded task-expression synthesis over an independently admitted V2 seed world.

Business structures, customer state and private goals remain fixed. New expressions
are never accepted solely because their seed task was accepted.
"""

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tau3.data_model.tasks import Task
from tau3.synthesis.concurrent_audit import ConcurrentAuditSession
from tau3.utils.llm_concurrency import (
    configured_concurrency,
    file_lock,
    install_request_limit,
)
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.blind import (
    Witness,
    check_frozen_witness,
    public_problem,
    solve_public,
)
from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash, write_json
from tau3.worldgen.v2.readiness import check_readiness, evidence_files
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import load_settings

STYLES = ("concise", "conversational", "formal", "plain language", "polite", "direct")


def adapter_hash() -> str:
    """Bind the task adapter and its public loading/CLI entry points."""
    return digest(
        {
            name: (Path(__file__).parent / name).read_text()
            for name in ("world_tasks.py", "bundle.py", "cli.py", "concurrent_audit.py")
        }
        | {
            "llm_concurrency.py": (
                Path(__file__).parents[1] / "utils/llm_concurrency.py"
            ).read_text()
        }
    )


def expression_errors(source: str, candidate: str) -> list[str]:
    """Reject changed numeric literals and newly invented machine identifiers."""
    if not isinstance(candidate, str) or not candidate.strip():
        return ["Missing nonempty request"]

    def numbers(value):
        return Counter(re.findall(r"(?<![\w])\d+(?:\.\d+)?(?![\w])", value))

    def identifiers(value):
        return Counter(
            re.findall(r"[\w.+-]+@[\w.-]+\.[\w-]+|\b[a-zA-Z][\w]*_[\w]+\b", value)
        )

    errors = []
    if numbers(source) != numbers(candidate):
        errors.append("Numeric literals changed")
    if identifiers(source) != identifiers(candidate):
        errors.append("Customer/record identifiers changed")
    return errors


def task_structures(world: Path) -> dict[str, str]:
    """Read the seed's structural families for stable cross-batch grouping."""
    from tau3.worldgen.v2.diversity import structural_fingerprint
    from tau3.worldgen.v2.specs import WorldSpec

    spec = WorldSpec.model_validate_json((world / "spec.json").read_text())
    return {
        f"task_{scenario.id}": structural_fingerprint(category, scenario)
        for category in spec.categories
        for scenario in category.scenarios
    }


def balanced_anchors(world: Path, tasks: list[Task]) -> list[Task]:
    """Cover distinct existing business structures before adding more seed tasks."""
    structures = task_structures(world)
    seen, first, rest = set(), [], []
    for task in sorted(tasks, key=lambda task: task.id):
        key = structures[task.id]
        (rest if key in seen else first).append(task)
        seen.add(key)
    return first + rest


def load_world_task_manifest(root: Path) -> dict:
    """Reject altered tasks, validation evidence, seed worlds or adapter versions."""
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "published"
        or manifest.get("domain") != "banking_synth"
    ):
        raise ValueError("V2 task bundle is not published")
    if (
        manifest.get("adapter_hash") != adapter_hash()
        or manifest.get("implementation") != implementation_hash()
    ):
        raise ValueError("Task adapter or world implementation changed")
    from tau3.worldgen.world import configured_world_root

    world = configured_world_root()
    if artifact_hashes(world) != manifest["source_artifacts"]:
        raise ValueError(
            "Configure TAU3_SYNTH_WORLD to the task bundle's admitted seed world"
        )
    readiness = Path(manifest["readiness"])
    if digest(json.loads(readiness.read_text())) != manifest["readiness_hash"]:
        raise ValueError("Seed readiness changed")
    for source in json.loads(readiness.read_text())["sources"].values():
        if evidence_files(Path(source["directory"])) != source["files"]:
            raise ValueError("Seed readiness evidence changed")
    if evidence_files(root / "validation") != manifest["validation_files"]:
        raise ValueError("Task validation evidence changed")
    if evidence_files(root / "audit") != manifest["audit_files"]:
        raise ValueError("Task synthesis evidence changed")
    return manifest


def synthesize_world_tasks(
    world: Path,
    readiness: Path,
    output: Path,
    count: int,
    config: Path,
    max_calls: int,
    resume: bool = False,
    local_world: Path | None = None,
    pilot_bundle: Path | None = None,
) -> dict:
    """Own the bundle exclusively while generating or resuming atomic evidence."""
    install_request_limit()
    with file_lock(output / "bundle.lock"):
        return _synthesize_world_tasks(
            world,
            readiness,
            output,
            count,
            config,
            max_calls,
            resume,
            local_world,
            pilot_bundle,
        )


def _synthesize_world_tasks(
    world: Path,
    readiness: Path,
    output: Path,
    count: int,
    config: Path,
    max_calls: int,
    resume: bool = False,
    local_world: Path | None = None,
    pilot_bundle: Path | None = None,
) -> dict:
    """Generate, independently solve, replay and admit every new task expression."""
    from tau3.data_model.simulation import Results, TextRunConfig
    from tau3.domains.banking_synth.environment import get_environment, get_tasks
    from tau3.registry import registry
    from tau3.runner.batch import run_tasks

    if count < 1 or max_calls < 1:
        raise ValueError("Positive task and model-call budgets required")
    source_world = world
    if local_world is not None:
        if artifact_hashes(world) != artifact_hashes(local_world):
            raise ValueError(
                "Local task-synthesis mirror differs from the source world"
            )
        world = local_world
    settings = load_settings(config)
    if len(set(settings.agent_models)) != 2:
        raise ValueError("Two independent model roles required")
    os.environ["TAU3_SYNTH_WORLD"] = str(world.resolve())
    os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(config.resolve())
    pilot_hash = None
    pilot_expressions = set()
    if count > 20:
        if pilot_bundle is None:
            raise ValueError(
                "A batch above 20 tasks requires a current published 20-task pilot"
            )
        from tau3.synthesis.bundle import load_task_bundle

        pilot_tasks = load_task_bundle(pilot_bundle)
        pilot_manifest = load_world_task_manifest(pilot_bundle)
        if len(pilot_tasks) != 20 or pilot_manifest.get("settings_hash") != digest(
            settings.model_dump()
        ):
            raise ValueError(
                "Pilot must have 20 admitted tasks under the same settings"
            )
        pilot_hash = digest(pilot_manifest)
        pilot_expressions = {digest(t.user_scenario.instructions) for t in pilot_tasks}
    check_readiness(world, readiness, config)
    if "banking_synth" not in registry.get_domains():
        registry.register_domain(get_environment, "banking_synth")
    identity = {
        "schema_version": 1,
        "domain": "banking_synth",
        "bundle_id": digest(str(output.resolve()))[:16],
        "source_world": str(source_world.resolve()),
        "source_artifacts": artifact_hashes(world),
        "implementation": implementation_hash(),
        "adapter_hash": adapter_hash(),
        "readiness": str(readiness.resolve()),
        "readiness_hash": digest(json.loads(readiness.read_text())),
        "settings_hash": digest(settings.model_dump()),
        "config": str(config.resolve()),
        "requested_tasks": count,
        "max_calls": max_calls,
        "pilot_bundle": str(pilot_bundle.resolve()) if pilot_bundle else None,
        "pilot_hash": pilot_hash,
        "candidate_attempts": 3,
        "max_rollouts": count * 6,
        "retrieval": {"name": "bm25", "kwargs": {}},
        "scope": "New task expressions over existing business skeletons; no new business structures or customer states",
    }
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if not resume or any(existing.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Existing task bundle requires --resume with identical inputs"
            )
        if existing["status"] == "published":
            return load_world_task_manifest(output)
    else:
        write_json(path, {**identity, "status": "draft"})
    session = ConcurrentAuditSession(
        world, output / "audit", settings, max_calls, "task-expression-v1"
    )
    anchors = balanced_anchors(world, get_tasks())
    seen, accepted, rejected = set(pilot_expressions), {}, []

    def make_candidate(index, attempt):
        anchor = anchors[index % len(anchors)]
        task_id = f"synth_{identity['bundle_id']}_{index:06d}_a{attempt}_{anchor.id}"
        candidate_path = output / "validation" / task_id / "candidate.json"
        response = session.ask(
            settings.agent_models[index % 2],
            'Rewrite customer-simulator instructions in the requested style. Preserve every fact, identifier, numeric literal, requirement and consent condition. Do not add an answer, product choice, operation name or bank policy. Return JSON {"request":"complete rewritten instructions"}.',
            {
                "instructions": anchor.user_scenario.instructions,
                "style": STYLES[
                    (index + index // len(anchors) + attempt) % len(STYLES)
                ],
                "variant": [index, attempt],
                "batch": identity["bundle_id"],
            },
        )
        instructions = response.get("request")
        errors = expression_errors(anchor.user_scenario.instructions, instructions)
        if errors:
            raise ValueError("Task expression rejected: " + "; ".join(errors))
        review = session.ask(
            settings.agent_models[1 - index % 2],
            'Compare the original and rewritten customer instructions. Require the same identity, facts, numbers, requested operations, preferences, permission requirements and disclosure behavior. Reject any new answer, policy, tool name or product choice. Return JSON {"equivalent":true,"no_solution_added":true,"issues":[]}.',
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
        task.id = task_id
        task.user_scenario.instructions = instructions
        candidate = {
            "anchor_id": anchor.id,
            "requested_style": STYLES[
                (index + index // len(anchors) + attempt) % len(STYLES)
            ],
            "task": task.model_dump(mode="json"),
            "review": review,
        }
        if (
            candidate_path.exists()
            and json.loads(candidate_path.read_text()) != candidate
        ):
            raise ValueError("Candidate differs from exact saved generation responses")
        write_json(candidate_path, candidate)
        return Task.model_validate(candidate["task"])

    def verify_candidate(index, task):
        anchor = anchors[index % len(anchors)]
        problem = public_problem(
            world, task.id, instructions=task.user_scenario.instructions
        )
        for i, model in enumerate(settings.agent_models):
            solution, review, attempts = solve_public(
                problem, model, settings.agent_models[1 - i], session.ask
            )
            # No private-state feedback is ever returned to generation or solving.
            check_frozen_witness(world, anchor.id, Witness.model_validate(solution))
            write_json(
                output / "validation" / task.id / f"blind_{i}.json",
                {
                    "model": model,
                    "solution": solution,
                    "review": review,
                    "attempts": attempts,
                    "status": "PASS",
                },
            )
        return task

    def parallel_candidates(function, jobs, attempt):
        def run_job(index, value):
            try:
                return function(index, value), None
            except AuditIncomplete:
                raise  # An outage/budget failure must stop, not resample candidates.
            except ValueError as exc:
                return None, {
                    "slot": index,
                    "attempt": attempt,
                    "stage": "generation_or_blind",
                    "reason": str(exc),
                }

        with ThreadPoolExecutor(max_workers=configured_concurrency()) as executor:
            futures = {
                i: executor.submit(run_job, i, value) for i, value in jobs.items()
            }
            results = {}
            try:
                for index, future in futures.items():
                    candidate, rejection = future.result()
                    if rejection is None:
                        results[index] = candidate
                    else:
                        rejected.append(rejection)
                        write_json(
                            output / "validation" / f"rejected_{index}_{attempt}.json",
                            rejection,
                        )
            except BaseException:
                for future in futures.values():
                    future.cancel()
                raise
        return results

    for attempt in range(3):
        generated = parallel_candidates(
            make_candidate,
            {i: attempt for i in range(count) if i not in accepted},
            attempt,
        )
        unique = {}
        # Slot-order ownership makes duplicate rejection stable across resumes.
        for index, task in generated.items():
            fingerprint = digest(task.user_scenario.instructions)
            anchor = anchors[index % len(anchors)]
            if (
                fingerprint in seen
                or task.user_scenario.instructions == anchor.user_scenario.instructions
            ):
                rejection = {
                    "slot": index,
                    "attempt": attempt,
                    "stage": "generation_or_blind",
                    "reason": "Duplicate or unchanged task expression",
                }
                rejected.append(rejection)
                write_json(
                    output / "validation" / f"rejected_{index}_{attempt}.json",
                    rejection,
                )
            else:
                seen.add(fingerprint)
                unique[index] = task
        candidates = parallel_candidates(verify_candidate, unique, attempt)
        passing = set(candidates)
        for i, model in enumerate(settings.agent_models):
            if not candidates:
                break
            tasks = list(candidates.values())
            result_path = output / "validation" / f"online_round_{attempt}_{i}.json"
            run_config = TextRunConfig(
                domain="banking_synth",
                llm_agent=model,
                llm_user=settings.user_model,
                llm_args_agent=settings.llm_args,
                llm_args_user=settings.llm_args,
                num_trials=1,
                max_steps=settings.simulation_max_steps,
                timeout=settings.simulation_timeout,
                max_errors=5,
                max_concurrency=configured_concurrency(),
                max_retries=0,
                auto_resume=True,
                retrieval_config="bm25",
                seed=42 + i,
                log_level="ERROR",
            )
            expected_tasks = {t.id: t.model_dump(mode="json") for t in tasks}
            result = None
            if result_path.exists():
                result = Results.load(result_path)
                if {
                    t.id: t.model_dump(mode="json") for t in result.tasks
                } != expected_tasks:
                    raise ValueError("Cached task conversation inputs changed")
                if (
                    result.info.world_artifact_hash
                    != digest(identity["source_artifacts"])
                    or result.info.world_implementation_hash
                    != identity["implementation"]
                    or result.info.world_evaluation_hash != identity["settings_hash"]
                    or result.info.agent_info.llm != model
                    or result.info.user_info.llm != settings.user_model
                    or result.info.max_steps != settings.simulation_max_steps
                    or result.info.seed != run_config.seed
                    or result.info.retrieval_config != "bm25"
                ):
                    raise ValueError("Cached task conversation world/settings changed")
                previous_ids = [sim.task_id for sim in result.simulations]
                if (
                    len(previous_ids) != len(set(previous_ids))
                    or not set(previous_ids) <= expected_tasks.keys()
                ):
                    raise ValueError("Invalid cached task conversation membership")
                if any(
                    sim.termination_reason.value == "infrastructure_error"
                    for sim in result.simulations
                ):
                    raise AuditIncomplete(
                        "Saved infrastructure failure requires diagnosis; it cannot be resampled as a new candidate"
                    )
                if len(previous_ids) < len(tasks):
                    # Keep a pre-resume checkpoint as evidence; the runner only owes
                    # missing cells, and completed failures remain completed.
                    write_json(
                        result_path.with_name(
                            result_path.stem
                            + "_checkpoint_"
                            + digest(previous_ids)[:12]
                            + ".json"
                        ),
                        json.loads(result_path.read_text()),
                    )
                    result = None
            if result is None:
                result = run_tasks(
                    run_config, tasks, save_path=result_path, console_display=False
                )
            if {sim.task_id for sim in result.simulations} != {
                t.id for t in tasks
            } or len(result.simulations) != len(tasks):
                raise ValueError(
                    "Incomplete task conversation membership; resume evidence required"
                )
            by_id = {sim.task_id: sim for sim in result.simulations}
            for index, task in candidates.items():
                sim = by_id[task.id]
                if sim.termination_reason.value == "infrastructure_error" or (
                    sim.reward_info
                    and any(
                        c.justification.startswith("INCONCLUSIVE:")
                        for c in sim.reward_info.nl_assertions or []
                    )
                ):
                    raise AuditIncomplete(
                        "Task replay infrastructure/judge evidence is incomplete"
                    )
                if (
                    sim.termination_reason.value not in {"user_stop", "agent_stop"}
                    or not sim.reward_info
                    or sim.reward_info.reward != 1
                ):
                    passing.discard(index)
                    rejected.append(
                        {
                            "slot": index,
                            "attempt": attempt,
                            "task_id": task.id,
                            "stage": "online",
                            "model": model,
                            "termination": sim.termination_reason.value,
                            "reward": sim.reward_info.reward
                            if sim.reward_info
                            else None,
                        }
                    )
        accepted.update({index: candidates[index] for index in passing})
        write_json(
            output / "validation" / "admission.json",
            {
                "status": "PASS" if len(accepted) == count else "INCONCLUSIVE",
                "admitted": len(accepted),
                "requested": count,
                "attempt_rounds": attempt + 1,
                "rejected": rejected,
            },
        )
        write_json(
            output / "progress.json",
            {
                "status": "draft",
                "admitted": len(accepted),
                "requested": count,
                "attempt_rounds": attempt + 1,
            },
        )
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(
            "Task slots exhausted their three candidates; draft and all rejections retained"
        )
    tasks = [accepted[index] for index in range(count)]
    task_anchors = {
        accepted[index].id: anchors[index % len(anchors)].id for index in range(count)
    }
    structures = task_structures(world)
    groups = {tid: structures[anchor_id] for tid, anchor_id in task_anchors.items()}
    if artifact_hashes(source_world) != identity["source_artifacts"]:
        raise ValueError("Canonical seed world changed during task synthesis")
    raw = [t.model_dump(mode="json") for t in tasks]
    seed_test = set(json.loads((world / "splits.json").read_text())["test"])
    splits = {
        "base": [t.id for t in tasks],
        "train": [t.id for t in tasks if task_anchors[t.id] not in seed_test],
        "validation": [t.id for t in tasks if task_anchors[t.id] in seed_test],
    }
    write_json(output / "tasks.json", raw)
    write_json(output / "split_tasks.json", splits)
    manifest = {
        **identity,
        "status": "published",
        "tasks_hash": digest(raw),
        "splits_hash": digest(splits),
        "task_groups": groups,
        "task_anchors": task_anchors,
        "split_policy": "Inherit the admitted seed's structural train/test partition",
        "validation_files": evidence_files(output / "validation"),
        "audit_files": evidence_files(output / "audit"),
        "admitted_tasks": len(tasks),
    }
    write_json(path, manifest)
    return manifest
