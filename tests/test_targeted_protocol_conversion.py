"""Checks for loss-safe, semantics-preserving protocol conversion."""

import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

_spec = importlib.util.spec_from_file_location(
    "protocol_converter",
    Path(__file__).parents[1] / "scripts" / "convert_targeted_protocol.py",
)
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)


def sample():
    """One failed call followed by a successful correction."""
    name = "call_discoverable_agent_tool"
    calls = [
        {"arguments": '{"capability": "business field", "amount": 30.0}'},
        {"capability": "bank_write", "arguments": '{"amount": 30.0}'},
    ]
    messages = [{"role": "system", "content": "Read the actual schemas."}]
    for index, args in enumerate(calls):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": str(index), "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                },
            }]},
            {"role": "tool", "tool_call_id": str(index), "content": (
                "Error: WorldTools.call_discoverable_agent_tool() missing 1 required positional argument: 'capability'"
                if index == 0 else '{"ok": true}'
            )},
        ])
    return {
        "messages": messages, "loss_mask": [0, 1, 0, 1, 0],
        "tools": [{"type": "function", "function": {
            "name": name, "parameters": {"type": "object", "properties": {
                "capability": {"type": "string"},
                "arguments": {"type": "string"},
            }, "required": ["capability", "arguments"]},
        }}],
        "metadata": {"gold": "must never enter general output"},
    }


def test_only_outer_selector_changes_and_roundtrips():
    args = {"capability": "x", "arguments": '{ "capability": "y", "amount": 30.0 }'}
    mapped = converter.rename_arguments("call_discoverable_agent_tool", args)
    assert mapped == {"agent_tool_name": "x", "arguments": args["arguments"]}
    assert converter.rename_arguments("call_discoverable_agent_tool", mapped, True) == args
    with pytest.raises(ValueError, match="collision"):
        converter.rename_arguments("call_discoverable_agent_tool", {**args, "agent_tool_name": "z"})


def test_text_rewrites_only_explicit_schema_references():
    ordinary = "This capability lets you act. **Granted capability:** `u`."
    assert converter.transform_text(ordinary, {"u": "user"}, "assistant") == ordinary
    assert converter.transform_text("- capability: `u`", {"u": "user"}, "assistant") == "- discoverable_tool_name: `u`"
    with pytest.raises(ValueError, match="Unresolved"):
        converter.transform_text("capability: `unknown`", {}, "assistant")


def test_failed_call_retained_but_unsupervised_in_both_arms():
    original = sample()
    converted, control, audit = converter.transform_row(original, {}, {"0"})
    assert original["loss_mask"] == [0, 1, 0, 1, 0]
    assert converted["loss_mask"] == control["loss_mask"] == [0, 0, 0, 1, 0]
    assert audit["masked_assistant_indices"] == [1]
    assert "agent_tool_name" in converted["messages"][2]["content"]
    assert "agent_tool_name" not in json.loads(converted["messages"][1]["tool_calls"][0]["function"]["arguments"])
    converter.validate(converted)
    converter.validate(control)
    output = converter.general(converted)
    assert set(output) == {"messages", "tools"}
    assert not output["messages"][1]["loss"]
    assert output["messages"][3]["loss"]
    assert "must never enter" not in json.dumps(output)
    converted["loss_mask"][1] = 1
    with pytest.raises(ValidationError):
        converter.validate(converted)


def test_orphan_result_rejected():
    row = sample()
    row["messages"][2]["tool_call_id"] = "unknown"
    row["loss_mask"][1] = 0
    with pytest.raises(ValueError, match="without call"):
        converter.validate(row)
