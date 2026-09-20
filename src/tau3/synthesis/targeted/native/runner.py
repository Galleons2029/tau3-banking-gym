"""Native trajectories with a durable, replayable physical-response journal."""

import re
import time
from contextlib import contextmanager
from pathlib import Path

from tau3.data_model.simulation import SimulationRun, TextRunConfig
from tau3.synthesis.llm import visible_sft_sample
from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.budget import Budget
from tau3.synthesis.targeted.native.validation import (
    check_goals,
    operations,
    protocol_errors,
    replacement_policy_errors,
)
from tau3.synthesis.validation import fresh_environment
from tau3.utils.llm_concurrency import file_lock
from tau3.worldgen.v2.audit import AuditIncomplete


def physical_record(root, physical_id):
    """Read inherited immutable requests without copying the entire request store."""
    path = root / "physical_calls" / f"{physical_id}.json"
    if path.exists():
        return read_json(path)
    reference = read_json(root / "evidence-parent.json")
    parent = Path(reference["root"])
    if read_json(parent / "snapshot/manifest.json")["snapshot_hash"] != reference["snapshot_hash"]:
        raise AuditIncomplete("Inherited physical evidence snapshot changed")
    return read_json(parent / "physical_calls" / f"{physical_id}.json")


class JournalBudget(Budget):
    """Bind physical reservations before transmission, closing the response crash gap."""

    journal_path = None
    request_identity = None

    def reserve(self, request, stage, kind="call"):
        if kind == "call" and self.journal_path is not None and self.journal_path.exists():
            previous = read_json(self.journal_path)
            if len(previous.get("history", [])) + 1 >= 3:
                raise AuditIncomplete("Physical request recovery exhausted (three attempts retained)")
        record = super().reserve(request, stage, kind)
        if kind == "call" and self.journal_path is not None:
            previous = read_json(self.journal_path) if self.journal_path.exists() else {}
            history = previous.get("history", [])
            if previous.get("physical_id"):
                history = [*history, previous["physical_id"]]
            write_json(self.journal_path, {"identity": self.request_identity,
                       "physical_id": record["id"], "history": history, "status": "RESERVED"})
        return record


def restore_retrieval_observations(root, directory, environment):
    """Replay original KB timing footers only after checking all document bytes."""
    calls, contents = {}, {}
    for path in sorted((directory / "responses").glob("*.json")):
        entry = read_json(path)
        physical = physical_record(root, entry["physical_id"])
        for message in physical["request"].get("messages", []):
            for call in message.get("tool_calls") or []:
                if call.get("function", {}).get("name") == "KB_search":
                    calls[call["id"]] = call["function"]
            if message.get("role") == "tool" and message.get("tool_call_id") in calls:
                key, value = message["tool_call_id"], message.get("content")
                if key in contents and contents[key] != value:
                    raise AuditIncomplete("Recorded retrieval observations disagree")
                contents[key] = value
    original = environment.get_response
    footer = re.compile(r"\n\[Timing: retrieval=\d+ms, reranking=\d+ms, total=\d+ms\]$")

    def get_response(call):
        response = original(call)
        if call.id in contents:
            import json

            expected = calls[call.id]
            arguments = expected["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if call.name != "KB_search" or call.arguments != arguments:
                raise AuditIncomplete("Retrieval replay call changed")
            actual, recorded = response.content, contents[call.id]
            if not isinstance(actual, str) or not isinstance(recorded, str) or response.error:
                raise AuditIncomplete("Retrieval replay returned an invalid observation")
            if footer.sub("", actual) != footer.sub("", recorded):
                raise AuditIncomplete("Retrieval document contents changed during replay")
            if actual != recorded:
                write_json(directory / "retrieval-replay" / f"{digest(call.id)}.json",
                           {"tool_call_id": call.id, "actual_hash": digest(actual),
                            "recorded_hash": digest(recorded), "difference": "timing_footer_only"})
            response.content = recorded
        return response

    environment.get_response = get_response


@contextmanager
def journal(root, config, directory, role, deadline=None):
    """Replay completed responses; unresolved transmissions never silently resample."""
    from litellm import ModelResponse

    from tau3.utils import llm_utils

    original = llm_utils.completion
    budget = JournalBudget(root, config)
    cursor = 0

    def completion(*args, **kwargs):
        nonlocal cursor
        path = directory / f"{cursor:05d}.json"
        cursor += 1
        identity = digest({k: v for k, v in kwargs.items() if k not in {"api_key", "headers", "extra_headers", "timeout"}})
        stage = role(kwargs)
        while True:
            attempts = 0
            if path.exists():
                saved = read_json(path)
                if saved["identity"] != identity:
                    raise AuditIncomplete("Response-journal prefix changed; cannot resume")
                attempts = len(saved.get("history", [])) + 1
                physical = physical_record(root, saved["physical_id"])
                if physical["status"] == "COMPLETE":
                    choices = physical["response"].get("choices", [])
                    if not choices or any(c.get("finish_reason") not in {"stop", "tool_calls"} for c in choices):
                        raise AuditIncomplete("Truncated journal response retained without continuation")
                    empty = all(not (c.get("message", {}).get("content") or c.get("message", {}).get("tool_calls")) for c in choices)
                    if not empty:
                        if saved["status"] != "COMPLETE":
                            write_json(path, {**saved, "status": "COMPLETE"})
                        return ModelResponse(**physical["response"])
                    if not stage.endswith("_user"):
                        raise AuditIncomplete("Empty teacher response retained without resampling")
                elif physical.get("error") not in {"Timeout", "APITimeoutError", "TimeoutError", "APIConnectionError"} and physical["id"] not in (
                    read_json(root / "transport-recovery-authorization.json").get("physical_ids", [])
                    if (root / "transport-recovery-authorization.json").exists() else []
                ):
                    raise AuditIncomplete("Unresolved physical request retained without resampling")
                if attempts >= 3:
                    raise AuditIncomplete("Physical request recovery exhausted (three attempts retained)")
            budget.journal_path, budget.request_identity = path, identity
            call_kwargs = dict(kwargs)
            if attempts and stage.endswith("_user"):
                call_kwargs["messages"] = [dict(m) for m in kwargs["messages"]]
                call_kwargs["messages"][0]["content"] += (
                    "\nAlways output a nonempty in-character user message or a valid "
                    "granted tool call. An empty response is not a valid turn.")
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise AuditIncomplete("Logical trajectory wall-clock budget exhausted")
                call_kwargs["timeout"] = min(float(kwargs.get("timeout") or remaining), remaining)
            try:
                budget.call(stage, original, *args, **call_kwargs)
            except Exception as exc:
                if type(exc).__name__ not in {"Timeout", "APITimeoutError", "TimeoutError", "APIConnectionError"}:
                    raise
            # Inspect the durable physical outcome before accepting it. Retrying
            # never changes teacher context, task, seed or sampling settings.

    llm_utils.completion = completion
    try:
        yield
    finally:
        llm_utils.completion = original


def sample(root, config, candidate, directory, model, seed, binding, *, blind=False):
    """Capture real agent state before grading; each logical seed has one history."""
    from tau3.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau3.runner.build import build_env_kwargs, build_text_orchestrator

    directory = Path(directory)
    identity = {"task_hash": digest(candidate.task.model_dump(mode="json")), "model": model,
                "seed": seed, "binding": binding, "blind": blind,
                "config_hash": digest(config.model_dump(mode="json"))}
    with file_lock(directory / "trial.lock"):
        marker, capture_path, result_path = [directory / name for name in ("started.json", "capture.json", "result.json")]
        if marker.exists() and read_json(marker) != identity:
            raise AuditIncomplete("Native trial identity changed")
        if not marker.exists() and (config.reuse_parent_pilot or config.reuse_parent_splits):
            from tau3.synthesis.targeted.native.reuse import reuse_capture

            reuse_capture(root, config, candidate, directory, identity)
        if result_path.exists():
            result = read_json(result_path)
            if result["identity"] != identity or digest(read_json(capture_path)) != result["capture_hash"]:
                raise AuditIncomplete("Native result lost evidence binding")
            return result
        if not marker.exists():
            if blind:
                with file_lock(root / "budget.lock"):
                    path = root / "blind-budget.json"
                    count = read_json(path)["rollouts"] if path.exists() else 0
                    if count >= config.max_blind_rollouts:
                        raise AuditIncomplete("Blind-solve budget exhausted")
                    write_json(path, {"rollouts": count + 1})
            else:
                Budget(root, config).reserve({"task": candidate.task.id, "seed": seed}, "teacher", kind="rollout")
            write_json(marker, identity)
            write_json(directory / "clock.json", {"deadline": time.time() + config.settings.simulation_timeout})
        if not capture_path.exists():
            recovery_path = directory / "clock-recovery.json"
            if recovery_path.exists():
                recovery = read_json(recovery_path)
                if recovery.get("pending"):
                    if recovery["identity"] != identity:
                        raise AuditIncomplete("Outage recovery identity changed")
                    remaining = recovery["remaining_active_seconds"]
                    if not 0 < remaining <= config.settings.simulation_timeout:
                        raise AuditIncomplete("Invalid outage recovery time allowance")
                    # Persist the grant once. A subsequent crash cannot reset it.
                    recovery.update(pending=False, resumed_at=time.time())
                    write_json(recovery_path, recovery)
                    write_json(directory / "clock.json", {"deadline": recovery["resumed_at"] + remaining})
            # One deadline survives restart; restarting cannot grant a fresh 1800 seconds.
            deadline = read_json(directory / "clock.json")["deadline"]
            remaining = deadline - time.time()
            if remaining <= 0:
                raise AuditIncomplete("Logical trajectory deadline exhausted; seed retained")
            run = TextRunConfig(
                domain="banking_knowledge", agent="llm_agent", llm_agent=model,
                llm_user=config.settings.user_model,
                llm_args_agent={**config.settings.llm_args, **config.settings.agent_llm_args, "temperature": 0 if blind else 0.7},
                llm_args_user={**config.settings.llm_args, **config.settings.user_llm_args, "temperature": 0},
                max_steps=config.settings.simulation_max_steps, timeout=remaining,
                max_errors=5, max_retries=0, retrieval_config="bm25_grep", seed=seed,
            )
            orchestrator = build_text_orchestrator(run, candidate.task, seed=seed)
            if config.user_role_guard:
                orchestrator.user.customer_role_guard = config.user_role_guard
            restore_retrieval_observations(root, directory, orchestrator.environment)
            started = time.monotonic()
            stage = "blind" if blind else "teacher"
            def role(kwargs):
                if getattr(orchestrator, "to_role", None) is not None and str(orchestrator.to_role.value) == "user":
                    return stage + "_user"
                return stage
            with journal(root, config, directory / "responses", role, deadline):
                simulation = orchestrator.run()
            simulation.policy = orchestrator.environment.get_policy()
            sample_row = visible_sft_sample(orchestrator, candidate.task.id)
            sample_row.update(schema_version=2, seed=seed, teacher_model=model)
            capture = {"identity": identity, "simulation": simulation.model_dump(mode="json"),
                       "sample": sample_row, "elapsed_seconds": time.monotonic() - started}
            write_json(capture_path, capture)
        capture = read_json(capture_path)
        simulation = SimulationRun.model_validate(capture["simulation"])
        with journal(root, config, directory / "grading-responses", lambda _: "grading"):
            simulation.reward_info = evaluate_simulation(
                simulation=simulation, task=candidate.task, domain="banking_knowledge", solo_mode=False,
                evaluation_type=EvaluationType.ALL,
                env_kwargs=build_env_kwargs("banking_knowledge", candidate.task, "bm25_grep"),
            )
        raw_messages = simulation.model_dump(mode="json")["messages"]
        env = fresh_environment(candidate.task)
        env.set_state(initialization_data=candidate.task.initial_state.initialization_data,
                      initialization_actions=candidate.task.initial_state.initialization_actions,
                      message_history=simulation.messages)
        calls = operations(raw_messages)
        failures = (check_goals(candidate, env, calls, raw_messages) + protocol_errors(calls)
                    + replacement_policy_errors(candidate, calls))
        success = (simulation.termination_reason.value in {"user_stop", "agent_stop"}
                   and simulation.reward_info.reward == 1 and not failures)
        incomplete = simulation.termination_reason.value in {
            "infrastructure_error", "timeout", "unexpected_error", "context_window_exceeded",
            "agent_error", "user_error",
            "max_steps", "too_many_errors",
        }
        result = {"identity": identity, "capture_hash": digest(capture),
                  "simulation": simulation.model_dump(mode="json"),
                  "environment_success": success, "obligation_failures": failures,
                  "status": "INCONCLUSIVE" if incomplete else "COMPLETE"}
        write_json(result_path, result)
        return result
