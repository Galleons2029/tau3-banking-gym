"""Process-safe physical request accounting, including failed calls and tokens."""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from tau3.utils.llm_concurrency import RequestPool, file_lock
from tau3.worldgen.v2.audit import AuditIncomplete, backend_auth_failure
from tau3.worldgen.v2.pipeline import write_json
from tau3.worldgen.v2.runtime import digest


class Budget:
    """Reserve conservative input/output token bounds before every network call."""

    def __init__(self, root: Path, config):
        self.root, self.config = root, config

    def reserve(self, request, stage, kind="call"):
        """Reservations survive crashes; missing usage never means zero cost."""
        token_bound = len(json.dumps(request, ensure_ascii=False).encode()) + int(
            request.get("max_tokens", self.config.settings.llm_args["max_tokens"])
        )
        with file_lock(self.root / "budget.lock"):
            path = self.root / "budget.json"
            state = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "calls": 0,
                    "audit_calls": 0,
                    "tokens": 0,
                    "rollouts": 0,
                }
            )
            if kind == "rollout":
                if state["rollouts"] >= self.config.max_rollouts:
                    raise AuditIncomplete("Round rollout budget exhausted")
                state["rollouts"] += 1
            else:
                audit = stage != "teacher"
                if (
                    state["calls"] >= self.config.max_total_calls
                    or state["tokens"] + token_bound > self.config.max_total_tokens
                    or (audit and state["audit_calls"] >= self.config.max_audit_calls)
                ):
                    raise AuditIncomplete("Round physical-call/token budget exhausted")
                state["calls"] += 1
                state["audit_calls"] += int(audit)
                state["tokens"] += token_bound
                if self.config.schema_version == 2:
                    model = request.get("model", "unknown")
                    state.setdefault("by_model", {})[model] = state.get("by_model", {}).get(model, 0) + 1
                    state.setdefault("by_stage", {})[stage] = state.get("by_stage", {}).get(stage, 0) + 1
            write_json(path, state)
        call_id = uuid4().hex
        record = {
            "id": call_id,
            "stage": stage,
            "kind": kind,
            "status": "RESERVED",
            "token_reservation": token_bound,
            "request": request,
            "reserved_at": time.time(),
        }
        write_json(self.root / "physical_calls" / f"{call_id}.json", record)
        return record

    def complete(self, record, response=None, error=None):
        """Settle only trustworthy provider usage; otherwise keep the reservation."""
        record = dict(record)
        record["finished_at"] = time.time()
        if error:
            record.update(status="INCONCLUSIVE", error=type(error).__name__)
        else:
            raw = (
                response.model_dump(mode="json")
                if hasattr(response, "model_dump")
                else response
            )
            record.update(status="COMPLETE", response=raw)
            usage = (raw or {}).get("usage") or {}
            used = usage.get("total_tokens")
            if isinstance(used, int) and used >= 0:
                with file_lock(self.root / "budget.lock"):
                    path = self.root / "budget.json"
                    state = json.loads(path.read_text())
                    state["tokens"] += used - record["token_reservation"]
                    write_json(path, state)
                record["actual_tokens"] = used
        write_json(self.root / "physical_calls" / f"{record['id']}.json", record)

    def call(self, stage, transport, *args, **kwargs):
        """Account one physical call without mutating a process-global function."""
        kwargs = {**kwargs, "num_retries": 0, "caching": False}
        public = {
            k: v
            for k, v in kwargs.items()
            if k not in {"api_key", "headers", "extra_headers"}
        }
        pool = RequestPool(self.root / "llm_pool", self.config.llm_concurrency)
        # AuditSession owns audit retries and its own physical-call limit. Other
        # callers need the same bounded backend recovery without changing a
        # logical teacher slot, its seed, messages, or sampling parameters.
        attempts = (
            1
            if stage in {"blind", "calibration"} and self.config.schema_version != 2
            else self.config.settings.audit_transport_attempts
        )
        for attempt in range(attempts):
            with pool.slot():
                record = self.reserve(public, stage)
                try:
                    result = transport(*args, **kwargs)
                except Exception as exc:
                    self.complete(record, error=exc)
                    gateway_glm_error = (
                        self.config.schema_version == 2
                        and getattr(exc, "status_code", None) == 403
                        and "Backend error (glm)" in str(exc)
                        and "QS quota exceeded" in str(exc)
                    )
                    if not (backend_auth_failure(exc) or gateway_glm_error) or attempt + 1 == attempts:
                        raise
                else:
                    self.complete(record, result)
                    break
            time.sleep(min(2**attempt, 4))
        raw = result.model_dump(mode="json")
        if any(
            c.get("finish_reason") not in {"stop", "tool_calls"}
            for c in raw.get("choices", [])
        ):
            raise AuditIncomplete("Physical model response did not finish normally")
        return result

    @contextmanager
    def installed(self, stage):
        """Capture world-runtime calls in its isolated owning process."""
        from tau3.utils import llm_utils

        original = llm_utils.completion

        def completion(*args, **kwargs):
            return self.call(stage, original, *args, **kwargs)

        llm_utils.completion = completion
        try:
            yield
        finally:
            llm_utils.completion = original


def ask(root, config, stage, model, system, payload, evidence=None, recovery_attempt=0):
    """Retry malformed native audits with bounded, independently retained evidence."""
    attempts = 3 if config.schema_version == 2 else 1
    for attempt in range(attempts):
        try:
            result = _ask_once(root, config, stage, model, system, payload, evidence,
                               recovery_attempt, native_attempt=attempt)
            if config.schema_version == 2 and stage == "reasoning_quality":
                from tau3.synthesis.targeted.native.models import NativeQuality

                quality = NativeQuality.model_validate(result)
                count = sum(m["role"] == "assistant" for m in payload["agent_context"]["messages"])
                if any(type(i) is not int or not 0 <= i < count for i in quality.erroneous_assistant_turns):
                    raise ValueError("Audit returned an invalid assistant index")
            return result
        except Exception as exc:
            recoverable = isinstance(exc, (ValueError, AuditIncomplete)) or type(exc).__name__ in {
                "Timeout", "APITimeoutError", "TimeoutError", "APIConnectionError"}
            if not recoverable or attempt + 1 == attempts:
                raise


def _ask_once(root, config, stage, model, system, payload, evidence=None,
              recovery_attempt=0, native_attempt=0):
    """Freeze exact structured requests and never erase invalid model responses."""
    from tau3.utils import llm_utils
    from tau3.worldgen.v2.json_output import json_llm_args, parse_object

    if native_attempt:
        # Re-audit the original evidence, never repair a verdict or inject feedback.
        system += (" Return one complete JSON object, including its final closing brace. "
                   "No Markdown, preamble or trailing text. Keep explanation under 1500 "
                   "characters; escape quotes inside strings. Do not repeat the schema.")
    if recovery_attempt:
        if stage != "attribution" or recovery_attempt != 1:
            raise ValueError("Only one explicit attribution recovery is supported")
        system += " Return exactly one complete JSON object. No Markdown or whitespace padding."
    request = {
        "model": model,
        "system": system,
        "payload": payload,
        "settings": digest(config.settings.model_dump()),
        "stage": stage,
    }
    if native_attempt:
        request["native_retry"] = native_attempt
    prompt_json = getattr(config, "structured_output_mode", None) == "prompt_json"
    if prompt_json:
        request["native_json_mode"] = False
        request["structured_max_tokens"] = config.structured_max_tokens
        request["structured_reasoning_effort"] = config.structured_reasoning_effort
        if model == config.reviewer_model:
            request["structured_reviewer_reasoning_effort"] = config.structured_reasoning_effort
            request["structured_reviewer_max_tokens"] = config.structured_reviewer_max_tokens
            request["structured_reviewer_timeout"] = config.structured_reviewer_timeout
    overrides = (
        {"timeout": config.attribution_timeout} if stage == "attribution" else {}
    )
    if overrides:
        request["transport_overrides"] = overrides
    if recovery_attempt:
        request["recovery_attempt"] = recovery_attempt
        request["native_json_mode"] = False
    path = root / "requests" / f"{digest(request)}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["status"] != "COMPLETE":
            raise AuditIncomplete("Previous structured request is incomplete")
        if evidence is not None:
            evidence.update(path=str(path.resolve()), hash=digest(record))
        return parse_object(record["response"])
    write_json(path, {"request": request, "status": "RESERVED"})
    call_args = {**json_llm_args(config.settings.llm_args), **overrides}
    if recovery_attempt or prompt_json:
        # This endpoint can loop on whitespace under constrained JSON mode.
        # Prompt-only recovery still passes the exact same strict JSON parser.
        call_args.pop("response_format", None)
        call_args["extra_body"].pop("response_format", None)
    if prompt_json:
        call_args["max_tokens"] = config.structured_max_tokens
        call_args["extra_body"]["reasoning_effort"] = config.structured_reasoning_effort
        if model == config.reviewer_model:
            call_args["max_tokens"] = config.structured_reviewer_max_tokens
            call_args["timeout"] = config.structured_reviewer_timeout
    reply = Budget(root, config).call(
        stage,
        llm_utils.completion,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tools=None,
        tool_choice=None,
        **call_args,
    )
    choice = (reply.model_dump(mode="json").get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    content = (choice.get("message") or {}).get("content")
    record = {
        "request": request,
        "response": content,
        "status": "COMPLETE" if finish == "stop" else "INCONCLUSIVE",
    }
    write_json(path, record)
    if record["status"] != "COMPLETE":
        raise AuditIncomplete("Structured output did not finish normally")
    if evidence is not None:
        evidence.update(path=str(path.resolve()), hash=digest(record))
    return parse_object(content)
