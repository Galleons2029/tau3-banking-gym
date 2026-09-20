"""Strict model JSON with an optional single Markdown fence, never text salvage."""

import json
import re
from copy import deepcopy


def json_llm_args(args: dict) -> dict:
    """Request native JSON only for structured audits; retain high reasoning.

    extra_body preserves this parameter for gateway model IDs that LiteLLM's
    drop_params would otherwise treat as unknown. Parsing remains strict even
    when a gateway claims to support JSON mode.
    """
    configured = deepcopy(args)
    configured.pop("response_format", None)
    configured["extra_body"] = {
        **(configured.get("extra_body") or {}),
        "response_format": {"type": "json_object"},
    }
    return configured


def parse_object(content: str | None) -> dict:
    """Accept one complete object; reject truncation, multiple objects and prose."""
    text = (content or "").strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
    )
    if fence:
        text = fence[1]

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(
        text, object_pairs_hook=unique_object, parse_constant=invalid_constant
    )
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value
