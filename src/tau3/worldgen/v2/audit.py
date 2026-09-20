"""Bounded, resumable model calls with exact inputs and responses for audits."""

import json
import time
from pathlib import Path

from tau3.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash, write_json
from tau3.worldgen.v2.runtime import digest


class AuditIncomplete(ValueError):
    """An audit lacks evidence, including exhausted or failed model calls."""


def backend_auth_failure(exc: Exception) -> bool:
    """Recognize the relay's intermittent Gemini backend failure, not bad API keys."""
    message = str(exc)
    return (
        getattr(exc, "status_code", None) == 401
        and "Backend error (gemini_mock)" in message
        and "[E401]user auth failed" in message
    )


def transport_failure(exc: Exception) -> bool:
    """Retry only failures without a model verdict, never content/schema errors."""
    from litellm import APIConnectionError, RateLimitError, Timeout

    return (
        backend_auth_failure(exc)
        or isinstance(exc, (Timeout, APIConnectionError, RateLimitError))
        or (
            isinstance(getattr(exc, "status_code", None), int)
            and 500 <= exc.status_code < 600
        )
    )


class AuditSession:
    """One sequential worker owns a bounded audit directory; never store API keys."""

    def __init__(self, root: Path, output: Path, settings, max_calls: int, kind: str):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.output, self.settings = output, settings
        self.identity = {
            "artifacts": artifact_hashes(root),
            "implementation": implementation_hash(),
            "settings": digest(settings.model_dump()),
            "kind": kind,
            "max_calls": max_calls,
        }
        path = output / "audit.json"
        if path.exists() and json.loads(path.read_text()) != self.identity:
            raise ValueError("Stale audit identity; use a fresh directory")
        write_json(path, self.identity)
        self.max_calls = max_calls

    def generate(self, *, model, messages, **kwargs):
        """Reserve before calling; interrupted/failed calls consume the budget too."""
        from tau3.utils.llm_utils import generate

        request = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        path = self.output / "calls" / f"{digest(request)}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("status") != "COMPLETE" or record.get(
                "finish_reason"
            ) not in (None, "stop"):
                raise AuditIncomplete("Previous request failed or was interrupted")
            return AssistantMessage(
                role="assistant",
                content=record["response"],
                raw_data={"choices": [{"finish_reason": record.get("finish_reason")}]},
            )
        record = {"request": request, "status": "RESERVED", "attempts": []}
        # Disable hidden SDK retries: every physical audit request consumes budget.
        kwargs = {**kwargs, "num_retries": 0}
        for index in range(self.settings.audit_transport_attempts):
            used = sum(
                len(item.get("attempts", [None]))
                for p in (self.output / "calls").glob("*.json")
                for item in [json.loads(p.read_text())]
            )
            if used >= self.max_calls:
                if record["attempts"]:
                    record.update(status="INCONCLUSIVE", error="BudgetExhausted")
                    write_json(path, record)
                raise AuditIncomplete("Model call budget exhausted")
            attempt = {"attempt": index + 1, "status": "RESERVED"}
            record["attempts"].append(attempt)
            write_json(path, record)
            try:
                reply = generate(model=model, messages=messages, **kwargs)
            except Exception as exc:
                retryable = transport_failure(exc)
                attempt.update(
                    status="INCONCLUSIVE",
                    error=type(exc).__name__,
                    transport_failure=retryable,
                )
                record.update(status="INCONCLUSIVE", error=type(exc).__name__)
                write_json(path, record)
                if retryable and index + 1 < self.settings.audit_transport_attempts:
                    time.sleep(min(2**index, 4))
                    continue
                raise AuditIncomplete(type(exc).__name__) from exc
            raw = getattr(reply, "raw_data", None) or {}
            outcome = dict(
                status="COMPLETE",
                response=reply.content,
                usage=getattr(reply, "usage", None),
                finish_reason=(raw.get("choices") or [{}])[0].get("finish_reason"),
            )
            attempt.update(outcome)
            record.update(outcome)
            record.pop("error", None)
            if record["finish_reason"] not in (None, "stop"):
                attempt["status"] = record["status"] = "INCONCLUSIVE"
                write_json(path, record)
                raise AuditIncomplete("Model response did not finish normally")
            write_json(path, record)
            return reply

    def transport_summary(self) -> dict:
        """Expose recovered outages separately from first-response validity."""
        records = [
            json.loads(p.read_text()) for p in (self.output / "calls").glob("*.json")
        ]
        return {
            "requests": len(records),
            "physical_calls": sum(len(r.get("attempts", [None])) for r in records),
            "transport_failures": sum(
                a.get("transport_failure") is True
                for r in records
                for a in r.get("attempts", [])
            ),
            "recovered_requests": sum(
                r.get("status") == "COMPLETE" and len(r.get("attempts", [])) > 1
                for r in records
            ),
        }

    def ask(self, model: str, system: str, payload: dict) -> dict:
        """Call a model using a frozen public payload and parse an explicit object."""
        from tau3.worldgen.v2.json_output import json_llm_args, parse_object

        reply = self.generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=system),
                UserMessage(
                    role="user", content=json.dumps(payload, ensure_ascii=False)
                ),
            ],
            call_name="worldgen_v2_audit",
            **json_llm_args(self.settings.llm_args),
        )
        return parse_object(reply.content)
