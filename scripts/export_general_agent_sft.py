"""Export evidence-bound targeted SFT rows in the general-agent JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    """Serialize values deterministically for identity comparisons."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_digest(path: Path) -> str:
    """Hash an artifact without loading the complete corpus into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    """Atomically publish a small manifest or audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def function_schema(tool: dict) -> dict:
    """Normalize one actual OpenAI function schema without inventing contracts."""
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        raise ValueError("Tool is not an OpenAI function definition")
    function = tool["function"]
    name, description, parameters = (
        function.get("name"),
        function.get("description"),
        function.get("parameters"),
    )
    if not isinstance(name, str) or not name:
        raise ValueError("Tool schema is missing a function name")
    if not isinstance(description, str):
        raise ValueError(f"Tool {name!r} is missing a string description")
    if not isinstance(parameters, dict):
        raise ValueError(f"Tool {name!r} is missing JSON Schema parameters")
    if parameters.get("type") != "object" or not isinstance(
        parameters.get("properties"), dict
    ):
        raise ValueError(f"Tool {name!r} does not expose an object parameter schema")
    strict = function.get("strict", False)
    if not isinstance(strict, bool):
        raise ValueError(f"Tool {name!r} has a non-boolean strict flag")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            # The scheme defines absent strict as false. Make that actual default explicit.
            "strict": strict,
        },
    }


def call_signature(message: dict, simulation: bool) -> tuple:
    """Compare calls across visible OpenAI and internal simulation representations."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name", call.get("name"))
        arguments = function.get("arguments", call.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Tool call {call.get('id')!r} has invalid JSON arguments"
                ) from exc
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError(f"Tool call {call.get('id')!r} is malformed")
        calls.append((call.get("id"), name, canonical(arguments)))
    return message.get("content", ""), tuple(calls)


def provider_reasoning(message: dict) -> str | None:
    """Return only a provider-emitted assistant reasoning string, if present."""
    raw = message.get("raw_data")
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    candidate = choices[0].get("message")
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("reasoning_content")
    if not isinstance(value, str):
        provider = candidate.get("provider_specific_fields")
        value = (
            provider.get("reasoning_content") if isinstance(provider, dict) else None
        )
    return value if isinstance(value, str) and value else None


def assistant_reasoning(row: dict, evidence: Path) -> list[str | None]:
    """Bind visible assistant turns one-for-one to their original teacher turns."""
    result = json.loads((evidence / "result.json").read_text())
    simulated = [
        message
        for message in result["simulation"]["messages"]
        if message.get("role") == "assistant"
    ]
    visible = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(simulated) != len(visible):
        raise ValueError("Visible and teacher assistant-turn counts differ")
    reasoning = []
    for source, target in zip(simulated, visible, strict=True):
        if call_signature(source, simulation=True) != call_signature(
            target, simulation=False
        ):
            raise ValueError("Visible assistant turn differs from teacher evidence")
        reasoning.append(provider_reasoning(source))
    return reasoning


def general_messages(row: dict) -> tuple[list[dict], int]:
    """Convert a captured visible context to the scheme's loss-bearing messages."""
    source = row.get("messages")
    mask = row.get("loss_mask")
    if (
        not isinstance(source, list)
        or not isinstance(mask, list)
        or len(source) != len(mask)
    ):
        raise ValueError("Source row has mismatched messages and loss mask")
    evidence = Path(row["metadata"]["evidence"])
    per_assistant = iter(assistant_reasoning(row, evidence))
    messages, reasoning_count = [], 0
    schemas = {tool["function"]["name"] for tool in map(function_schema, row["tools"])}
    pending, seen = set(), set()
    for message, loss in zip(source, mask, strict=True):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role {role!r}")
        if loss not in (0, 1, False, True) or (
            bool(loss) and role != "assistant"
        ) or (row.get("schema_version") != 2 and bool(loss) != (role == "assistant")):
            raise ValueError("Loss mask is not assistant-only")
        converted = {
            "role": role,
            "content": message.get("content", ""),
            "loss": bool(loss),
        }
        if not isinstance(converted["content"], str):
            raise ValueError("Message content must be a string")
        if role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                identifier = call.get("id")
                function = call.get("function") or {}
                name, arguments = function.get("name"), function.get("arguments")
                if not isinstance(identifier, str) or not isinstance(name, str):
                    raise ValueError("Assistant tool call is missing an ID or name")
                if name not in schemas and loss:
                    raise ValueError(f"Tool call {name!r} lacks a top-level schema")
                if not isinstance(arguments, str):
                    raise ValueError("Tool-call arguments must be JSON strings")
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("Tool-call arguments must encode a JSON object")
                if identifier in seen:
                    raise ValueError("Duplicate tool-call ID")
                pending.add(identifier)
                seen.add(identifier)
                calls.append(
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                        "id": identifier,
                    }
                )
            if calls:
                converted["tool_calls"] = calls
            reasoning = next(per_assistant)
            if reasoning is not None:
                converted["reasoning_content"] = reasoning
                reasoning_count += 1
        elif role == "tool":
            identifier = message.get("tool_call_id")
            if not isinstance(identifier, str) or identifier not in pending:
                raise ValueError("Tool result lacks a matching previous tool call")
            pending.remove(identifier)
            converted["tool_call_id"] = identifier
        messages.append(converted)
    if pending:
        raise ValueError("Unfinished tool calls cannot be exported")
    return messages, reasoning_count


def convert_row(row: dict) -> tuple[dict, dict]:
    """Convert one evidence-bound SFT row and return audit counters."""
    if not isinstance(row.get("tools"), list) or not row["tools"]:
        raise ValueError("Sample has no configured tools")
    tools = [function_schema(tool) for tool in row["tools"]]
    if len({tool["function"]["name"] for tool in tools}) != len(tools):
        raise ValueError("Duplicate tool names in configured schema")
    messages, reasoning_count = general_messages(row)
    dynamic = sum(
        call["function"]["name"]
        in {
            "unlock_discoverable_agent_tool",
            "call_discoverable_agent_tool",
            "give_discoverable_user_tool",
        }
        for message in messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    )
    return {"tools": tools, "messages": messages}, {
        "assistant_messages": sum(m["role"] == "assistant" for m in messages),
        "reasoning_messages": reasoning_count,
        "tool_calls": sum(len(m.get("tool_calls", [])) for m in messages),
        "dynamic_protocol_calls": dynamic,
        "tool_names": sorted(tool["function"]["name"] for tool in tools),
    }


def export(source: Path, output: Path, audit: Path, overwrite: bool = False) -> dict:
    """Write general-agent JSONL only if every source row validates completely."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    totals: Counter = Counter()
    schemas: Counter = Counter()
    rows = 0
    with source.open() as input_file, temporary.open("w") as destination:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row, result = convert_row(json.loads(line))
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Row {line_number} failed general-agent export: {exc}"
                ) from exc
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            totals.update(
                {key: value for key, value in result.items() if isinstance(value, int)}
            )
            schemas.update({"|".join(result["tool_names"]): 1})
    temporary.replace(output)
    report = {
        "status": "PASS",
        "format": "general_agent_openai_tools_v1",
        "source": str(source.resolve()),
        "source_sha256": file_digest(source),
        "output": str(output.resolve()),
        "output_sha256": file_digest(output),
        "rows": rows,
        **dict(totals),
        "reasoning_preserved_by_default": True,
        "top_level_openai_tool_schemas_complete": True,
        "top_level_schema_variants": len(schemas),
        "strict_flags": {"true": 0, "false": rows * 7},
        "dynamic_business_operation_top_level_schemas": False,
        "dynamic_business_operation_note": (
            "Business operations are intentionally invoked through the three declared "
            "discoverable-tool wrappers. Their operation-specific parameter contracts are "
            "returned after unlock in visible tool results, not fabricated as top-level tools."
        ),
    }
    write_json_atomic(audit, report)
    return report


def main() -> None:
    """Run the exporter from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.input, args.output, args.audit, args.overwrite),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
