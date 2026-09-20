"""Check evidence-bound conversion to the documented general-agent format."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "general_agent_export",
    Path(__file__).parents[1] / "scripts/export_general_agent_sft.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def source_row(evidence: Path) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Follow policy."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "lookup", "arguments": '{"id":"1"}'},
                        "type": "function",
                    }
                ],
                "weight": 1,
            },
            {"role": "tool", "content": '{"name":"one"}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Done.", "tool_calls": None, "weight": 1},
        ],
        "loss_mask": [0, 1, 0, 1],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
            }
        ],
        "metadata": {"evidence": str(evidence)},
    }


def write_evidence(path: Path, first_reasoning="first thought") -> None:
    path.mkdir(parents=True)
    simulation = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "lookup", "arguments": {"id": "1"}}
                ],
                "raw_data": {
                    "choices": [{"message": {"reasoning_content": first_reasoning}}]
                },
            },
            {
                "role": "assistant",
                "content": "Done.",
                "tool_calls": None,
                "raw_data": {"choices": [{"message": {"content": "Done."}}]},
            },
        ]
    }
    (path / "result.json").write_text(json.dumps({"simulation": simulation}))


def test_general_agent_conversion_preserves_real_reasoning_only(tmp_path):
    evidence = tmp_path / "teacher"
    write_evidence(evidence)
    converted, stats = exporter.convert_row(source_row(evidence))
    assert converted["messages"][0] == {
        "role": "system",
        "content": "Follow policy.",
        "loss": False,
    }
    assistant = converted["messages"][1]
    assert assistant["reasoning_content"] == "first thought"
    assert assistant["loss"] is True
    assert assistant["tool_calls"] == [
        {
            "type": "function",
            "id": "call_1",
            "function": {"name": "lookup", "arguments": '{"id":"1"}'},
        }
    ]
    assert "reasoning_content" not in converted["messages"][-1]
    assert converted["tools"][0]["function"]["strict"] is False
    assert stats["reasoning_messages"] == 1


def test_export_rejects_unbound_calls_and_does_not_publish(tmp_path):
    evidence = tmp_path / "teacher"
    write_evidence(evidence)
    row = source_row(evidence)
    row["messages"][1]["tool_calls"][0]["function"]["name"] = "not_configured"
    source, output, audit = (
        tmp_path / "source.jsonl",
        tmp_path / "output.jsonl",
        tmp_path / "audit.json",
    )
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="differs from teacher evidence"):
        exporter.export(source, output, audit)
    assert not output.exists()
    assert not audit.exists()


def test_export_writes_scheme_only_rows_and_audit(tmp_path):
    evidence = tmp_path / "teacher"
    write_evidence(evidence)
    source, output, audit = (
        tmp_path / "source.jsonl",
        tmp_path / "output.jsonl",
        tmp_path / "audit.json",
    )
    source.write_text(json.dumps(source_row(evidence)) + "\n")
    report = exporter.export(source, output, audit)
    exported = json.loads(output.read_text())
    assert set(exported) == {"tools", "messages"}
    assert report["rows"] == 1
    assert report["top_level_openai_tool_schemas_complete"] is True
    assert report["reasoning_messages"] == 1
