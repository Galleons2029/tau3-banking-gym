"""Capture and publish training-only teacher context for admitted V2 tasks."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from tau3.data_model.simulation import SimulationRun, TextRunConfig
from tau3.synthesis.llm import visible_sft_sample
from tau3.synthesis.world_tasks import adapter_hash
from tau3.utils.llm_concurrency import file_lock, install_request_limit
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.pipeline import (
    artifact_hashes,
    check_certificate,
    implementation_hash,
    write_json,
)
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import load_settings


@lru_cache(maxsize=1)
def producer_hash():
    """Pin capture, streaming admission and their runtime dependencies."""
    tau = Path(__file__).parents[1]
    paths = [
        Path(__file__),
        Path(__file__).with_name("streaming_tasks.py"),
        Path(__file__).with_name("stream_resume.py"),
        Path(__file__).with_name("teacher_sampling.py"),
        tau / "synthesis/llm.py",
        tau / "runner/build.py",
        tau / "runner/simulation.py",
        tau / "agent/llm_agent.py",
    ]
    return digest(
        {
            "adapter": adapter_hash(),
            "world": implementation_hash(),
            "files": {str(p.relative_to(tau)): p.read_text() for p in paths},
        }
    )


def configure(world: Path, config: Path):
    """Configure one process for a fixed admitted world and its model settings."""
    from tau3.domains.banking_synth.environment import get_environment
    from tau3.registry import registry

    check_certificate(world)
    os.environ["TAU3_SYNTH_WORLD"] = str(world.resolve())
    os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(config.resolve())
    install_request_limit()
    if "banking_synth" not in registry.get_domains():
        registry.register_domain(get_environment, "banking_synth")
    return load_settings(config)


def successful(simulation: SimulationRun) -> bool:
    """Only normally finished conversations with conclusive perfect scores pass."""
    reward = simulation.reward_info
    if simulation.termination_reason.value in {
        "infrastructure_error",
        "timeout",
        "unexpected_error",
        "agent_error",
        "user_error",
        "context_window_exceeded",
    }:
        raise AuditIncomplete("Conversation infrastructure/timeout is inconclusive")
    if reward and any(
        c.justification.startswith("INCONCLUSIVE:") for c in reward.nl_assertions or []
    ):
        raise AuditIncomplete("Conversation judge is inconclusive")
    return simulation.termination_reason.value in {"user_stop", "agent_stop"} and bool(
        reward and reward.reward == 1
    )


def validate_sample(sample: dict):
    """Require teacher-only supervision and coherent visible tool call/result IDs."""
    messages = sample["messages"]
    mask = [int(m["role"] == "assistant") for m in messages]
    supplied = sample.get("loss_mask")
    selective = sample.get("schema_version") == 2
    valid = (
        isinstance(supplied, list)
        and len(supplied) == len(mask)
        and any(supplied)
        and all(x in (0, 1) and (not x or allowed) for x, allowed in zip(supplied, mask))
    )
    if not valid or (not selective and supplied != mask):
        raise ValueError("Invalid assistant-only supervision mask")
    pending = set()
    seen = set()
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unexpected role in teacher context")
        if role == "assistant":
            if pending:
                raise ValueError("Missing visible tool results")
            for call in message.get("tool_calls") or []:
                if call["id"] in seen:
                    raise ValueError("Duplicate tool call ID")
                pending.add(call["id"])
                seen.add(call["id"])
        elif role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Tool result is not in teacher's visible context")
            pending.remove(message["tool_call_id"])
    if pending:
        raise ValueError("Unfinished tool calls cannot be training examples")


def capture_trial(task, settings, model, directory: Path, seed: int, binding: dict):
    """Persist actual teacher context before scoring; never rerun interrupted trials."""
    from tau3.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau3.runner.build import build_env_kwargs, build_text_orchestrator

    signature = {
        "task": digest(task.model_dump(mode="json")),
        "model": model,
        "settings": digest(settings.model_dump()),
        "seed": seed,
        "producer": producer_hash(),
        "binding": binding,
    }
    with file_lock(directory / "trial.lock"):
        marker = directory / "started.json"
        if marker.exists() and json.loads(marker.read_text()) != signature:
            raise AuditIncomplete("Teacher trial identity changed")
        capture_path, result_path = (
            directory / "capture.json",
            directory / "result.json",
        )
        if result_path.exists():
            if not marker.exists() or not capture_path.exists():
                raise AuditIncomplete("Incomplete saved teacher evidence")
            return json.loads(result_path.read_text())
        if capture_path.exists():
            if not marker.exists():
                raise AuditIncomplete("Unbound capture")
            capture = json.loads(capture_path.read_text())
            if capture["identity"] != signature:
                raise AuditIncomplete("Capture identity changed")
            if (directory / "grading-started.json").exists():
                raise AuditIncomplete("Interrupted grading cannot be silently repeated")
            simulation = SimulationRun.model_validate(capture["simulation"])
        else:
            if marker.exists():
                raise AuditIncomplete(
                    "Interrupted teacher request retained; no automatic resampling"
                )
            write_json(marker, signature)
            run_config = TextRunConfig(
                domain="banking_synth",
                agent="llm_agent",
                llm_agent=model,
                llm_user=settings.user_model,
                llm_args_agent={**settings.llm_args, **settings.agent_llm_args},
                llm_args_user={**settings.llm_args, **settings.user_llm_args},
                max_steps=settings.simulation_max_steps,
                timeout=settings.simulation_timeout,
                max_errors=5,
                max_retries=0,
                retrieval_config=settings.capture_retrieval,
                seed=seed,
            )
            orchestrator = build_text_orchestrator(run_config, task, seed=seed)
            simulation = orchestrator.run()
            simulation.policy = orchestrator.environment.get_policy()
            sample = visible_sft_sample(orchestrator, task.id)
            sample.update(
                retrieval_config=settings.capture_retrieval,
                teacher_model=model,
                seed=seed,
            )
            capture = {
                "identity": signature,
                "simulation": simulation.model_dump(mode="json"),
                "sample": sample,
            }
            write_json(capture_path, capture)
        write_json(directory / "grading-started.json", signature)
        simulation.reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            domain="banking_synth",
            solo_mode=False,
            evaluation_type=EvaluationType.ALL,
            env_kwargs=build_env_kwargs(
                "banking_synth", task, settings.capture_retrieval
            ),
        )
        # Persist even a negative/inconclusive verdict before inspecting it.
        result = {
            "identity": signature,
            "simulation": simulation.model_dump(mode="json"),
            "capture_hash": digest(capture),
            "status": "GRADED",
        }
        write_json(result_path, result)
        return result


def trial_sample(directory: Path):
    """Verify the exact graded capture before deriving a training row."""
    result = json.loads((directory / "result.json").read_text())
    capture = json.loads((directory / "capture.json").read_text())
    marker = json.loads((directory / "started.json").read_text())
    if (
        result["identity"] != marker
        or capture["identity"] != marker
        or result["capture_hash"] != digest(capture)
    ):
        raise ValueError("Teacher capture/result binding changed")
    if not successful(SimulationRun.model_validate(result["simulation"])):
        raise ValueError("Failed teacher conversation")
    sample = capture["sample"]
    validate_sample(sample)
    return {
        **sample,
        "evidence": {
            "directory": str(directory.resolve()),
            "result_hash": digest(result),
            "capture_hash": digest(capture),
        },
    }


def export_shard(path: Path, samples: list[dict], split: str, source: dict):
    """Atomically publish one training shard; validation examples are rejected."""
    if split != "train":
        raise ValueError("Only training tasks may enter SFT")
    rows = []
    for sample in samples:
        validate_sample(sample)
        rows.append({**sample, "split": "train", "source": source})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)
    return len(rows)


def pilot_sft(world: Path, pilot: Path, config: Path, output: Path):
    """Sample both configured teachers for already published pilot train tasks."""
    from tau3.synthesis.bundle import load_task_bundle

    with file_lock(output / "job.lock"):
        settings = configure(world, config)
        tasks = load_task_bundle(pilot, split="train")
        binding = {
            "pilot_manifest": digest(json.loads((pilot / "manifest.json").read_text())),
            "world": artifact_hashes(world),
        }
        identity = {
            "producer": producer_hash(),
            "binding": binding,
            "settings": digest(settings.model_dump()),
            "target": len(tasks) * len(settings.agent_models),
        }
        path = output / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Pilot SFT identity changed")
        write_json(path, identity)
        rows, failures = [], []

        def run(task, i, model):
            directory = output / "trials" / task.id / str(i)
            capture_trial(task, settings, model, directory, 80042 + i, binding)
            return trial_sample(directory)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(run, task, i, model): (task.id, i)
                for task in tasks
                for i, model in enumerate(settings.agent_models)
            }
            for future in as_completed(futures):
                task_id, model_index = futures[future]
                try:
                    sample = future.result()
                    export_shard(
                        output / "shards" / f"{task_id}_{model_index}.jsonl",
                        [sample],
                        "train",
                        binding,
                    )
                    rows.append(sample)
                except Exception as exc:
                    failures.append(
                        {
                            "task_id": task_id,
                            "model_index": model_index,
                            "reason": str(exc),
                        }
                    )
                export_shard(
                    output / "sft.jsonl",
                    sorted(rows, key=lambda r: (r["task_id"], r["teacher_model"])),
                    "train",
                    binding,
                )
                write_json(
                    output / "progress.json",
                    {
                        "status": "RUNNING",
                        "target": identity["target"],
                        "sft_samples": len(rows),
                        "failures": failures,
                    },
                )
        report = {
            "status": "COMPLETE" if len(rows) == identity["target"] else "INCOMPLETE",
            "target": identity["target"],
            "sft_samples": len(rows),
            "failures": failures,
        }
        write_json(output / "progress.json", report)
        return report
