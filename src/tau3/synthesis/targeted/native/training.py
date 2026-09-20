"""Reference Dots training adapter with original template and exact token masks."""

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.world_sft import validate_sample


@lru_cache(maxsize=2)
def artifacts(checkpoint):
    """Load tokenizer assets locally, without loading weights or executing model code."""
    root = Path(checkpoint)
    config = json.loads((root / "tokenizer_config.json").read_text())
    template = (root / "chat_template.jinja").read_text()
    if config.get("chat_template") and config["chat_template"].strip() != template.strip():
        raise ValueError("Tokenizer config and standalone chat template differ")
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    def fail(message):
        raise ValueError(message)
    environment.globals["raise_exception"] = fail
    tokenizer_path = root / "tokenizer.json"
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json")}
    return Tokenizer.from_file(str(tokenizer_path)), environment.from_string(template), hashes


def template_messages(sample):
    """Parse only outer tool arguments; nested wrapper JSON stays a string."""
    messages = copy.deepcopy(sample["messages"])
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            if not isinstance(function["arguments"], dict):
                raise ValueError("Outer tool arguments are not an object")
    return messages


def encode_sample(sample, checkpoint, max_length=131072):
    """Render complete history and assign labels only to unmasked assistant spans."""
    validate_sample(sample)
    tokenizer, template, hashes = artifacts(str(checkpoint))
    messages = template_messages(sample)
    def render(prefix):
        return template.render(messages=prefix, tools=sample["tools"],
                               enable_thinking=True, clear_thinking=False, add_generation_prompt=False)
    text = render(messages)
    spans = []
    system = messages[0] if messages[0]["role"] == "system" else {"role": "system", "content": "You are a helpful assistant."}
    header = render([system])
    pieces = [header]
    offset = len(header)
    index = int(messages[0]["role"] == "system")
    while index < len(messages):
        message, loss = messages[index], sample["loss_mask"][index]
        end = index + 1
        if message["role"] == "tool":
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
        rendered = render([system, *messages[index:end]])
        if not rendered.startswith(header):
            raise ValueError("Template header is not stable")
        piece = rendered[len(header):]
        pieces.append(piece)
        start, stop = offset + len("<|assistant|>"), offset + len(piece)
        offset = stop
        index = end
        if message["role"] != "assistant":
            continue
        if not piece.startswith("<|assistant|>"):
            raise ValueError("Unsupported assistant boundary")
        segment = piece[len("<|assistant|>"):]
        reasoning = message.get("reasoning_content")
        if reasoning and reasoning.strip() not in segment:
            raise ValueError("Template dropped teacher reasoning")
        # Verify function and parameter serialization through the actual Dots template.
        rendered_calls = re.findall(r'<invoke name="([^"]+)">(.*?)</invoke>', segment, re.S)
        expected_calls = message.get("tool_calls") or []
        if len(rendered_calls) != len(expected_calls):
            raise ValueError("Tool-call count changed during rendering")
        for (name, body), call in zip(rendered_calls, expected_calls, strict=True):
            expected = call["function"]
            parsed = dict(re.findall(r'<parameter name="([^"]+)">\n(.*?)\n</parameter>', body, re.S))
            values = {key: value if isinstance(expected["arguments"].get(key), str) else json.loads(value)
                      for key, value in parsed.items()}
            if name != expected["name"] or values != expected["arguments"]:
                raise ValueError("Tool arguments changed during template round trip")
        spans.append((start, stop, bool(loss)))
    if "".join(pieces) != text:
        raise ValueError("Incremental template segments differ from full-history rendering")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) > max_length:
        raise ValueError(f"LENGTH_UNSUPPORTED: {len(encoded.ids)} > {max_length}; no truncation")
    labels = [-100] * len(encoded.ids)
    cursor = 0
    for i, (start, stop) in enumerate(encoded.offsets):
        while cursor < len(spans) and start >= spans[cursor][1]:
            cursor += 1
        if cursor >= len(spans):
            break
        low, high, enabled = spans[cursor]
        if stop > start and enabled and low <= start and stop <= high:
            labels[i] = encoded.ids[i]
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised assistant tokens")
    return {"input_ids": encoded.ids, "labels": labels,
            "audit": {"tokenizer_hashes": hashes, "tokens": len(labels),
                      "supervised_tokens": sum(v != -100 for v in labels),
                      "mask_hash": digest(labels), "rendered_hash": digest(text),
                      "reasoning_preserved": True, "tool_roundtrip": True,
                      "adapter_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def check_training_readiness(sample, contract, directory):
    """Quarantine an overlong whole trajectory without discarding its successful task."""
    from tau3.worldgen.v2.audit import AuditIncomplete

    path = Path(directory) / "training-audit.json"
    identity = digest([sample, contract])
    if path.exists():
        record = read_json(path)
        if record["identity"] != identity:
            raise AuditIncomplete("Token-level training evidence changed")
    else:
        try:
            audit = encode_sample(sample, contract["checkpoint"], contract["max_length"])["audit"]
            record = {"identity": identity, "status": "READY", "audit": audit}
        except ValueError as exc:
            record = {"identity": identity,
                      "status": "REJECTED_LENGTH" if str(exc).startswith("LENGTH_UNSUPPORTED:") else "INCONCLUSIVE",
                      "reason": str(exc)}
        write_json(path, record)
    if record["status"] == "INCONCLUSIVE":
        raise AuditIncomplete(record["reason"])
    return record["status"] == "READY"
