"""Concurrent task audits retaining the sequential audit's exact evidence format.

One request lock prevents duplicate calls; a separate short budget lock makes
reservation atomic without holding that lock across network requests.
"""

import json
import time

from tau3.data_model.message import AssistantMessage
from tau3.utils.llm_concurrency import file_lock
from tau3.worldgen.v2.audit import AuditIncomplete, AuditSession, transport_failure
from tau3.worldgen.v2.pipeline import write_json
from tau3.worldgen.v2.runtime import digest


class ConcurrentAuditSession(AuditSession):
    """Share a bounded audit directory across candidate threads/processes."""

    def __init__(self, root, output, settings, max_calls, kind):
        with file_lock(output / "identity.lock"):
            super().__init__(root, output, settings, max_calls, kind)
            with file_lock(output / "budget.lock"):
                used = sum(
                    len(json.loads(p.read_text()).get("attempts", [None]))
                    for p in (output / "calls").glob("*.json")
                )
                path = output / "budget.json"
                if path.exists():
                    budget = json.loads(path.read_text())
                    if (
                        budget.get("identity") != digest(self.identity)
                        or budget["used"] < used
                    ):
                        raise AuditIncomplete(
                            "Concurrent audit budget identity/count changed"
                        )
                else:
                    write_json(path, {"identity": digest(self.identity), "used": used})

    def generate(self, *, model, messages, **kwargs):
        """Reuse one completed response for identical concurrent requests."""
        request = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        with file_lock(self.output / "locks" / f"{digest(request)}.lock"):
            return self._generate(model=model, messages=messages, **kwargs)

    def _generate(self, *, model, messages, **kwargs):
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
            with file_lock(self.output / "budget.lock"):
                budget_path = self.output / "budget.json"
                budget = json.loads(budget_path.read_text())
                used = budget["used"]
                if used >= self.max_calls:
                    if record["attempts"]:
                        record.update(status="INCONCLUSIVE", error="BudgetExhausted")
                        write_json(path, record)
                    raise AuditIncomplete("Model call budget exhausted")
                attempt = {"attempt": index + 1, "status": "RESERVED"}
                # Charge first. A crash before the request record is written leaves
                # a conservative charged slot, never an unbudgeted model request.
                write_json(
                    budget_path,
                    {
                        "identity": digest(self.identity),
                        "used": used + 1,
                        "last_reservation": {
                            "request": path.stem,
                            "attempt": index + 1,
                        },
                    },
                )
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
