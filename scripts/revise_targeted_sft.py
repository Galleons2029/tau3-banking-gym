"""Revise an evidence-bound SFT corpus without resampling teachers.

Run conversion with the frozen V2 runtime; audit tokens in a separate process
using the current training adapter. Original worlds and captures are immutable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import multiprocessing
import re
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import convert_targeted_protocol as prior

VERSION = "targeted-data-revision-v2"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDENTITIES = Path(
    "/newcpfs/user/liujialong/data/common_data/common_id_datasets"
)
SOURCE = ROOT / "data/synthetic/targeted-v1-protocol-compatible-20260917"
GIVEN_NAMES = "Aaron Adam Adrian Aisha Alex Alexander Alice Amanda Amelia Amy Andrea Andrew Angela Anna Anthony Arthur Ashley Austin Barbara Benjamin Beth Brandon Brian Bruce Caleb Cameron Carla Carlos Catherine Charles Charlotte Chloe Christian Christopher Claire Clara Daniel David Diana Dylan Edward Elena Elizabeth Emily Emma Eric Ethan Eva Evelyn Felix Fiona Francis Frank Gabriel George Grace Hannah Harry Hazel Helen Henry Ian Irene Isaac Isabel Isabella Jack Jacob James Jane Janet Jason Jay Jean Jennifer Jessica John Jonathan Jordan Joseph Joshua Julia Julian Justin Karen Katherine Kelly Kevin Laura Lauren Leah Leo Liam Lily Linda Lisa Logan Lucas Lucy Luke Maria Mark Martha Martin Mary Matthew Maya Megan Michael Michelle Molly Monica Nathan Natalie Nicholas Nicole Noah Nora Oliver Olivia Omar Oscar Owen Patrick Paul Peter Philip Rachel Rebecca Richard Robert Robin Rose Ryan Samuel Sandra Sara Sarah Scott Sean Sebastian Simon Sophia Sophie Stella Stephen Steven Susan Thomas Timothy Tina Tom Tony Victor Victoria Vincent Walter William Zachary Zoe".split()
DOMAINS = "gmail.com outlook.com yahoo.com hotmail.com aol.com protonmail.com icloud.com live.com mail.com".split()


def file_hash(path):
    """Hash without retaining large corpus bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity_pool(root):
    """Use local name/surname samples and provider domains to compose identities."""
    first = next((root / "person_names").rglob("English_Names_Corpus*txt"))
    surname = root / "person_names/en/us_census_2010_surnames/names.zip"
    domains = root / "email/domains/free_email_providers/domains.json"
    given = set(first.read_text(encoding="utf-8-sig").lower().splitlines())
    given = [n for n in GIVEN_NAMES if n.lower() in given]
    with zipfile.ZipFile(surname) as archive:
        records = csv.DictReader(
            io.StringIO(archive.read("Names_2010Census.csv").decode())
        )
        family = [r["name"].title() for r in records if r["name"].isalpha()][:2000]
    available = set(prior.read(domains))
    providers = [d for d in DOMAINS if d in available]
    if len(given) < 50 or len(family) < 500 or not providers:
        raise ValueError("Insufficient local identity samples")
    return {
        "given": given,
        "family": family,
        "domains": providers,
        "sources": {str(p): file_hash(p) for p in (first, surname, domains)},
        "method": "Synthetic combinations of corpus names, census surnames and provider domains; not copied real-person email pairs.",
    }


def make_identities(users, slot, pool):
    """Create deterministic, collision-free customer identities within the corpus."""
    records = []
    for index, (uid, info) in enumerate(sorted(users.items())):
        number = int(slot) * 100 + index + 100000
        seed = int(prior.sha(uid)[:16], 16)
        given = pool["given"][seed % len(pool["given"])]
        family = pool["family"][(seed // 1009) % len(pool["family"])]
        domain = pool["domains"][(seed // 100003) % len(pool["domains"])]
        records.append(
            {
                "old": {"user_id": uid, "name": info["name"], "email": info["email"]},
                "new": {
                    "user_id": f"cust_{number}",
                    "name": f"{given} {family}",
                    "email": f"{given.lower()}.{family.lower()}{number}@{domain}",
                },
            }
        )
    return records


class IdentityMap:
    """Simultaneous replacement in JSON values, keys, embedded JSON and prose."""

    def __init__(self, records, reverse=False):
        a, b = ("new", "old") if reverse else ("old", "new")
        self.mapping = {
            record[a][key]: record[b][key]
            for record in records
            for key in ("user_id", "name", "email")
        }
        self.pattern = re.compile(
            "|".join(re.escape(k) for k in sorted(self.mapping, key=len, reverse=True))
        )

    def text(self, value):
        """Replace longest identities first without replacing newly emitted text."""
        return self.pattern.sub(lambda m: self.mapping[m[0]], value)

    def __call__(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self(v) for v in value]
        if isinstance(value, dict):
            result = {self.text(k): self(v) for k, v in value.items()}
            if len(result) != len(value):
                raise ValueError("Identity map collapsed keys")
            return result
        return value


def render_contract(contract, grant=False):
    """Present existing contracts in the official parameter-list style."""
    lines = [
        ("Tool granted to customer: " if grant else "Tool unlocked: ")
        + contract["name"],
        "Description: " + contract["description"],
        "",
        "Tool: " + contract["name"],
        "Parameters:",
    ]
    for key, prop in contract["parameters"].items():
        kind = "number" if prop["type"] == "decimal" else prop["type"]
        details = []
        for label in ("minimum", "maximum", "scale", "unit"):
            if prop.get(label) not in (None, ""):
                details.append(f"{label}: {prop[label]}")
        if prop.get("choices"):
            details.append("allowed values: " + ", ".join(prop["choices"]))
        lines.append(
            f"  - {key}: {kind} (required)"
            + (" - " + "; ".join(details) if details else "")
        )
    wrapper = "call_discoverable_user_tool" if grant else "call_discoverable_agent_tool"
    selector = "discoverable_tool_name" if grant else "agent_tool_name"
    lines.append(
        f"\n{'Only the customer may' if grant else 'You can now'} call `{wrapper}` with {selector}='{contract['name']}' and arguments containing a JSON object encoded as a string. Use exactly the listed fields."
    )
    return "\n".join(lines)


def normalize_numbers(arguments, operation):
    """Normalize numeric strings only where the operation explicitly declares numbers."""
    result = copy.deepcopy(arguments)
    count = 0
    if "arguments" not in result:
        return result, count
    inner = json.loads(result["arguments"])
    for name, prop in operation.parameters.items():
        if (
            name not in inner
            or not isinstance(inner[name], str)
            or prop.type not in {"decimal", "integer"}
        ):
            continue
        number = prop.coerce(inner[name])
        value = int(number) if prop.type == "integer" else float(number)
        if Decimal(str(value)) != number:
            raise ValueError("Numeric conversion loses precision")
        inner[name] = value
        count += 1
    if count:
        result["arguments"] = json.dumps(inner, ensure_ascii=False)
    return result, count


def numeric_view(value, numeric_fields, field=None):
    """Compare only declared numeric fields by value, preserving all identifiers."""
    if isinstance(value, dict):
        return {k: numeric_view(v, numeric_fields, k) for k, v in value.items()}
    if isinstance(value, list):
        return [numeric_view(v, numeric_fields, field) for v in value]
    if (
        field in numeric_fields
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ):
        try:
            return ("decimal", Decimal(str(value)))
        except ArithmeticError:
            pass
    return value


def replay_revision(row, native, world, mapper):
    """Replay original and revised worlds, checking every result, state and grant."""
    from tau3.worldgen.v2.environment import WorldTools, WorldUserTools
    from tau3.worldgen.v2.runtime import (
        OperationRuntime,
        WorldDB,
        digest,
        goal_errors,
        validate_db,
    )
    from tau3.worldgen.v2.specs import WorldSpec

    spec, initial = prior.world_material(str(world), row["metadata"]["world_hash"])
    new_spec = WorldSpec.model_validate(mapper(spec.model_dump()))
    new_initial = WorldDB.model_validate(mapper(initial.model_dump()))
    validate_db(new_spec, new_initial)
    old = WorldTools(spec, initial.model_copy(deep=True), initial)
    new = WorldTools(new_spec, new_initial.model_copy(deep=True), new_initial)
    old_user, new_user = WorldUserTools(old), WorldUserTools(new)
    adapter = prior.ProtocolAdapter(new, new_user)
    runtime = OperationRuntime(spec, initial)
    numeric_fields = {
        k
        for cat in spec.categories
        for entity in cat.entities
        for k, v in entity.fields.items()
        if v.type in {"decimal", "integer"}
    }
    numeric_fields |= {
        k
        for _, op in runtime.operations.values()
        for k, v in op.parameters.items()
        if v.type in {"decimal", "integer"}
    }
    pending = defaultdict(deque)
    replacements = {}
    counts = Counter()
    for message in native["simulation"]["messages"]:
        for call in message.get("tool_calls") or []:
            pending[(call.get("requestor", message["role"]), call["id"])].append(call)
        if message["role"] != "tool":
            continue
        actor = message.get("requestor", "assistant")
        key = actor, message["id"]
        call = pending[key].popleft()
        name, args = call["name"], call["arguments"]
        if name in {"KB_search", "grep"}:
            counts["retrieval_results_preserved_with_identity_map"] += 1
            continue
        source_tools = old if actor == "assistant" else old_user
        old_error, old_result = prior.outcome(getattr(source_tools, name), args)
        changed = prior.rename_arguments(name, mapper(args))
        if (
            name in {"call_discoverable_agent_tool", "call_discoverable_user_tool"}
            and not old_error
        ):
            alias = args["capability"]
            changed, count = normalize_numbers(
                changed, runtime.operations[runtime.aliases[alias]][1]
            )
            counts["numeric_values_normalized"] += count
        target = (
            adapter
            if name in prior.SELECTORS
            else (new if actor == "assistant" else new_user)
        )
        new_error, new_result = prior.outcome(getattr(target, name), changed)
        if old_error != new_error:
            raise ValueError("Error outcome changed")
        if old_error:
            if prior.error_identity(mapper(old_result)) != prior.error_identity(
                new_result
            ):
                raise ValueError("Error meaning changed")
            if prior.error_identity(old_result) != prior.error_identity(
                message["content"]
            ):
                raise ValueError("Captured failure is not reproducible")
        else:
            if prior.normalized(old_result) != prior.normalized(message["content"]):
                raise ValueError("Original result is not reproducible: " + name)
            if numeric_view(
                mapper(prior.normalized(old_result)), numeric_fields
            ) != numeric_view(prior.normalized(new_result), numeric_fields):
                raise ValueError(
                    "Result changed beyond identity/numeric representation: " + name
                )
        if numeric_view(mapper(old.db.model_dump()), numeric_fields) != numeric_view(
            new.db.model_dump(), numeric_fields
        ):
            raise ValueError("Database state divergence")
        if old.unlocked != new.unlocked or old.user_grants != new.user_grants:
            raise ValueError("Grant/permission divergence")
        public_result = new_result
        if not new_error and name in {
            "unlock_discoverable_agent_tool",
            "give_discoverable_user_tool",
        }:
            public_result = render_contract(
                json.loads(new_result), name.startswith("give_")
            )
            counts["contract_results_rendered"] += 1
        if actor == "assistant":
            replacements[call["id"]] = {
                "arguments": changed,
                "result": public_result,
                "error": new_error,
            }
        counts["replayed_calls"] += 1
        counts["user_calls"] += actor == "user"
    if any(pending.values()):
        raise ValueError("Incomplete native call results")
    scenario = next(
        s
        for c in new_spec.categories
        for s in c.scenarios
        if "task_" + s.id == row["task_id"]
    )
    errors = goal_errors(
        new_initial, new.db, scenario.goals, scenario.user_id, new_spec
    )
    if errors:
        raise ValueError("Revised goals failed: " + repr(errors))
    return replacements, {
        "status": "PASS",
        **counts,
        "source_state_hash": digest(old.db.model_dump()),
        "revised_state_hash": digest(new.db.model_dump()),
        "numeric_comparison": "Only fields declared decimal/integer; identifiers compare exactly.",
    }


VERIFICATION_PATTERNS = [
    (
        r"\byou(?:\s+are|'re|’re)\s+(?:now\s+)?(?:fully\s+)?verified\b",
        "your customer profile has been located",
    ),
    (
        r"\byour\s+identity\s+(?:has\s+been\s+|is\s+)?(?:successfully\s+)?verified\b",
        "your customer profile has been located",
    ),
    (r"\bidentity\s+verified\b", "Customer profile located"),
    (
        r"\bidentity\s+verification\s+(?:is\s+)?(?:complete|completed|successful)\b",
        "Customer profile lookup complete",
    ),
    (r"\b(?:verified|confirmed)\s+your\s+identity\b", "located your customer profile"),
    (r"\b(?:verify|confirm)\s+your\s+identity\b", "locate your customer profile"),
]
RISKY_CLAIM = re.compile(
    r"\b(?:I(?:'ve| have|’ve)\s+(?:successfully\s+)?(?:transferred you|escalated (?:this|your|the)|issued (?:a |the )?refund|credited your|paid you)|your (?:case|issue|problem)(?:s)?\s+(?:is|are|has been|have been)\s+(?:fully\s+)?resolved)\b",
    re.I,
)


def clean_text(text, located):
    """Correct profile-lookup claims; quarantine uncertain business assertions."""
    edits = 0
    for pattern, replacement in VERIFICATION_PATTERNS:
        # A requested lookup does not require a previously successful lookup.
        if not located and replacement != "locate your customer profile":
            continue
        text, number = re.subn(pattern, replacement, text, flags=re.I)
        edits += number
    return text, edits, bool(RISKY_CLAIM.search(text))


def revise_row(source, mapper, replacements):
    """Apply evidence-bound substitutions and split dependency-bearing call batches."""
    row = copy.deepcopy(source)
    row["schema_version"] = 2
    row["messages"] = mapper(row["messages"])
    row["tools"] = mapper(row["tools"])
    counts = Counter()
    edits, candidates = [], []
    located = False
    calls = {}
    for index, message in enumerate(row["messages"]):
        if message["role"] == "system":
            old = message["content"]
            message["content"] = old.replace(
                "Always make sure you generate valid JSON only.",
                "Use the model's native structured tool-call format. Only the wrapper arguments field contains a JSON object encoded as a string. Use natural language for customer-facing answers.",
            )
            message["content"] = message["content"].replace(
                "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\nYou cannot do both at the same time.",
                "You can send a customer-facing message or make structured tool calls. A tool-calling message may include a brief explanation, as supported by this environment. Do not claim that an operation succeeded before receiving its result.",
            )
            message["content"] += (
                "\nFinding a customer profile is not identity authentication. Report case estimates and routing labels as recorded; do not describe them as payments, final issue resolution, or a completed human transfer. Follow actual tool results and user reports."
            )
            counts["system_revised"] += 1
        if message["role"] == "assistant":
            old = message.get("content") or ""
            text, number, risky = clean_text(old, located)
            message["content"] = text
            counts["identity_claim_edits"] += number
            if number:
                edits.append(
                    {
                        "message": index,
                        "kind": "profile_lookup_wording",
                        "before_hash": prior.sha(old),
                        "after_hash": prior.sha(text),
                    }
                )
            if risky:
                # Preserve dialogue context without training an unsupported assertion.
                row["loss_mask"][index] = 0
                message["weight"] = 0
                candidates.append(
                    {
                        "message": index,
                        "reason": "Unverified resolution/payment/human-transfer claim; preserved as context only",
                        "text": text,
                    }
                )
                counts["claim_turns_masked"] += 1
            for call in message.get("tool_calls") or []:
                name = call["function"]["name"]
                calls[call["id"]] = name
                if call["id"] in replacements:
                    call["function"]["arguments"] = json.dumps(
                        replacements[call["id"]]["arguments"], ensure_ascii=False
                    )
                # Remove a redundant non-OpenAI key, never the canonical function name.
                call.pop("name", None)
        elif message["role"] == "tool":
            entry = replacements.get(message["tool_call_id"])
            if entry:
                message["content"] = entry["result"]
                if (
                    calls.get(message["tool_call_id"]) == "find_customer"
                    and not entry["error"]
                    and json.loads(entry["result"])
                ):
                    located = True
    messages, masks = [], []
    index = 0
    unlocked = set()
    while index < len(row["messages"]):
        message = row["messages"][index]
        batch = message.get("tool_calls") or []
        unlocks = {
            json.loads(c["function"]["arguments"]).get("agent_tool_name")
            for c in batch
            if c["function"]["name"] == "unlock_discoverable_agent_tool"
        }
        dependent = any(
            c["function"]["name"] == "call_discoverable_agent_tool"
            and json.loads(c["function"]["arguments"]).get("agent_tool_name")
            in unlocks - unlocked
            for c in batch
        )
        if dependent:
            responses = row["messages"][index + 1 : index + 1 + len(batch)]
            by_id = {m.get("tool_call_id"): m for m in responses if m["role"] == "tool"}
            if set(by_id) != {c["id"] for c in batch}:
                raise ValueError("Cannot safely split dependent batch")
            for j, call in enumerate(batch):
                part = copy.deepcopy(message)
                part["tool_calls"] = [call]
                part["content"] = message.get("content", "") if j == 0 else ""
                messages.extend([part, by_id[call["id"]]])
                masks.extend([row["loss_mask"][index], 0])
            counts["dependent_batches_split"] += 1
            index += len(batch) + 1
            unlocked |= unlocks
            continue
        messages.append(message)
        masks.append(row["loss_mask"][index])
        if (
            message["role"] == "tool"
            and calls.get(message.get("tool_call_id"))
            == "unlock_discoverable_agent_tool"
        ):
            if not replacements[message["tool_call_id"]]["error"]:
                unlocked.add(
                    replacements[message["tool_call_id"]]["arguments"][
                        "agent_tool_name"
                    ]
                )
        index += 1
    row["messages"], row["loss_mask"] = messages, masks
    row["metadata"]["data_revision"] = {
        "version": VERSION,
        "source_row_hash": prior.sha(prior.canonical(source)),
        "derived": True,
    }
    prior.validate(row)
    if not any(masks):
        raise ValueError("No supervised messages remain")
    assert not any(
        old
        in json.dumps({"messages": messages, "tools": row["tools"]}, ensure_ascii=False)
        for old in mapper.mapping
    )
    return row, {
        "counts": dict(counts),
        "edits": edits,
        "review_candidates": candidates,
    }


def focus_candidates(row):
    """Extract a real next action with full prefix, keeping following results masked."""
    picked = {}
    previous_error = False
    for index, (message, loss) in enumerate(
        zip(row["messages"], row["loss_mask"], strict=True)
    ):
        if message["role"] == "tool":
            previous_error |= message["content"].startswith("Error:")
            continue
        if message["role"] != "assistant" or not loss or not message.get("tool_calls"):
            continue
        types = []
        for call in message["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            if name == "give_discoverable_user_tool":
                types.append("handoff")
            elif name == "unlock_discoverable_agent_tool":
                types.append("discovery")
            elif name == "call_discoverable_agent_tool":
                inner = json.loads(args["arguments"])
                types.append(
                    "pagination" if inner.get("offset", 0) > 0 else "execution"
                )
        if previous_error:
            types.insert(0, "recovery")
        for kind in types:
            if kind in picked:
                continue
            end = index + 1
            while end < len(row["messages"]) and row["messages"][end]["role"] == "tool":
                end += 1
            sample = copy.deepcopy(row)
            sample["messages"] = sample["messages"][:end]
            sample["loss_mask"] = [int(i == index) for i in range(end)]
            for i, m in enumerate(sample["messages"]):
                if m["role"] == "assistant":
                    m["weight"] = int(i == index)
            sample["metadata"]["focus"] = {
                "kind": kind,
                "target_message": index,
                "results_are_after_target_and_unsupervised": True,
            }
            prior.validate(sample)
            picked[kind] = sample
        previous_error = False
    return picked


def write_jsonl(path, rows):
    """Publish one complete shard atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    return file_hash(path)


def process_group(rows, output, identity_hash, pool):
    """Convert a single world, preserving original evidence and derived world material."""
    from tau3.worldgen.v2.runtime import digest

    slot = Path(rows[0]["metadata"]["evidence"]).parents[1].name
    directory = Path(output) / "shards" / slot
    report_file = directory / "report.json"
    if report_file.exists():
        report = prior.read(report_file)
        if report["identity_hash"] != identity_hash or any(
            file_hash(directory / n) != h for n, h in report["files"].items()
        ):
            raise ValueError("Frozen conversion shard changed")
        return report
    directory.mkdir(parents=True, exist_ok=True)
    world = Path(
        prior.read(Path(rows[0]["metadata"]["evidence"]).parents[1] / "task.json")[
            "world"
        ]
    )
    db = prior.read(world / "db.json")
    records = make_identities(db["users"], slot, pool)
    mapper = IdentityMap(records)
    files = {}
    world_view = {
        "source_world": str(world),
        "source_world_hash": rows[0]["metadata"]["world_hash"],
        "derived_not_original_certificate": True,
        "identity_map": records,
        "spec": mapper(prior.read(world / "spec.json")),
        "db": mapper(db),
        "tasks": {
            p.name: mapper(prior.read(p))
            for p in sorted((world / "tasks").glob("*.json"))
        },
        "documents": {
            str(p.relative_to(world / "documents")): mapper.text(p.read_text())
            for p in sorted((world / "documents").rglob("*"))
            if p.is_file()
        },
    }
    prior.atomic(directory / "world_view.json", world_view)
    files["world_view.json"] = file_hash(directory / "world_view.json")
    converted, audits, candidates = [], [], {}
    counters = Counter()
    for row in rows:
        trial = Path(row["metadata"]["evidence"])
        native, capture, quality = (
            prior.read(trial / n)
            for n in ("result.json", "capture.json", "quality.json")
        )
        if (
            not quality["sft_qualified"]
            or digest(native) != quality["native_hash"]
            or digest(capture) != quality["capture_hash"]
            or native["capture_hash"] != digest(capture)
        ):
            raise ValueError("Native evidence binding mismatch")
        replacements, replay = replay_revision(row, native, world, mapper)
        revised, audit = revise_row(row, mapper, replacements)
        revised["metadata"]["data_revision"]["world_view_hash"] = files[
            "world_view.json"
        ]
        revised["metadata"]["data_revision"]["world_view"] = str(
            directory / "world_view.json"
        )
        converted.append(revised)
        for kind, candidate in focus_candidates(revised).items():
            candidates.setdefault(kind, candidate)
        audits.append(
            {
                "task_id": row["task_id"],
                "trial": row["metadata"]["trial"],
                "source_row_hash": prior.sha(prior.canonical(row)),
                "revised_row_hash": prior.sha(prior.canonical(revised)),
                "replay": replay,
                **audit,
            }
        )
        counters.update(audit["counts"])
        counters.update({k: v for k, v in replay.items() if isinstance(v, int)})
    kinds = [
        k
        for k in ("recovery", "handoff", "pagination", "execution", "discovery")
        if k in candidates
    ][:2]
    focused = [candidates[k] for k in kinds]
    for name, values in (
        ("sft.jsonl", converted),
        ("focus.jsonl", focused),
        ("audit.jsonl", audits),
    ):
        files[name] = write_jsonl(directory / name, values)
    report = {
        "status": "PASS",
        "slot": slot,
        "identity_hash": identity_hash,
        "rows": len(rows),
        "focus_rows": len(focused),
        "counts": dict(counters),
        "files": files,
    }
    prior.atomic(report_file, report)
    return report


def safe_group(*args):
    """Capture a world failure without publishing partial aggregate files."""
    try:
        return process_group(*args)
    except Exception as exc:
        import traceback

        rows, output, *_ = args
        slot = Path(rows[0]["metadata"]["evidence"]).parents[1].name
        record = {
            "status": "FAIL",
            "slot": slot,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        prior.atomic(Path(output) / "failures" / (slot + ".json"), record)
        return record


def aggregate(output, reports):
    """Read each shard once, publishing aligned full/focus and task-balanced files."""
    names = (
        "sft.jsonl",
        "general_agent.jsonl",
        "focus.sft.jsonl",
        "focus.general_agent.jsonl",
        "balanced.sft.jsonl",
        "balanced.general_agent.jsonl",
        "sampling.jsonl",
        "audit.jsonl",
    )
    handles = {n: (output / (n + ".tmp")).open("w") for n in names}
    counts = Counter()
    try:
        for report in sorted(reports, key=lambda r: r["slot"]):
            directory = output / "shards" / report["slot"]
            for name, expected in report["files"].items():
                if file_hash(directory / name) != expected:
                    raise ValueError("Shard changed before aggregate")
            rows = [json.loads(line) for line in (directory / "sft.jsonl").open()]
            focus = [json.loads(line) for line in (directory / "focus.jsonl").open()]
            for prefix, samples in (("", rows), ("focus.", focus)):
                for row in samples:
                    handles[prefix + "sft.jsonl"].write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    handles[prefix + "general_agent.jsonl"].write(
                        json.dumps(prior.general(row), ensure_ascii=False) + "\n"
                    )
            # Four whole trajectories + one focused step per task: equal task counts.
            chosen = [("full", i % len(rows), rows[i % len(rows)]) for i in range(4)]
            if focus:
                pos = int(report["slot"]) % len(focus)
                chosen.append(("focus", pos, focus[pos]))
            else:
                chosen.append(("full", 0, rows[0]))
            for family, local_index, row in chosen:
                handles["balanced.sft.jsonl"].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                handles["balanced.general_agent.jsonl"].write(
                    json.dumps(prior.general(row), ensure_ascii=False) + "\n"
                )
                handles["sampling.jsonl"].write(
                    json.dumps(
                        {
                            "line": counts["balanced_rows"] + 1,
                            "slot": report["slot"],
                            "source_kind": family,
                            "shard_line": local_index + 1,
                            "sample_hash": prior.sha(prior.canonical(row)),
                        }
                    )
                    + "\n"
                )
                counts["balanced_rows"] += 1
                counts["balanced_" + family] += 1
            for line in (directory / "audit.jsonl").open():
                handles["audit.jsonl"].write(line)
            counts["rows"] += len(rows)
            counts["focus_rows"] += len(focus)
    finally:
        for handle in handles.values():
            handle.close()
    hashes = {name: file_hash(output / (name + ".tmp")) for name in names}
    for name in names:
        (output / (name + ".tmp")).replace(output / name)
    return dict(counts), hashes


def main():
    """Convert the source, leaving final training readiness to token audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, default=DEFAULT_IDENTITIES)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--slots", help="Comma-separated slot ids for a bounded preflight"
    )
    args = parser.parse_args()
    source_report = prior.read(args.source / "report.json")
    if (
        source_report["status"] != "PASS"
        or file_hash(args.source / "sft.jsonl") != source_report["hashes"]["sft.jsonl"]
    ):
        raise ValueError("Source version failed integrity check")
    pool = identity_pool(args.identity_root)
    identity = {
        "version": VERSION,
        "source": str(args.source.resolve()),
        "source_hash": source_report["hashes"]["sft.jsonl"],
        "converter_hash": file_hash(__file__),
        "parent_converter_hash": file_hash(prior.__file__),
        "identity_sources": pool["sources"],
        "slots": args.slots,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "identity.json").exists() and prior.read(
        args.output / "identity.json"
    ) != identity:
        raise ValueError("Output is version-frozen; use a new directory")
    prior.atomic(args.output / "identity.json", identity)
    prior.atomic(args.output / "identity_sources.json", pool)
    groups = defaultdict(list)
    requested = set(args.slots.split(",")) if args.slots else None
    for line in (args.source / "sft.jsonl").open():
        row = json.loads(line)
        slot = Path(row["metadata"]["evidence"]).parents[1].name
        if requested is None or slot in requested:
            groups[slot].append(row)
    if requested and set(groups) != requested:
        raise ValueError("Requested preflight slot missing")
    reports = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                safe_group,
                rows,
                str(args.output.resolve()),
                prior.sha(prior.canonical(identity)),
                pool,
            )
            for rows in groups.values()
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            if (
                len(reports) % 50 == 0
                or reports[-1]["status"] != "PASS"
                or len(reports) == len(groups)
            ):
                progress = {
                    "status": "CONVERTING",
                    "worlds": len(groups),
                    "processed": len(reports),
                    "failures": [r for r in reports if r["status"] != "PASS"],
                }
                prior.atomic(args.output / "progress.json", progress)
                print(
                    json.dumps({**progress, "failures": len(progress["failures"])}),
                    flush=True,
                )
    if any(r["status"] != "PASS" for r in reports):
        raise SystemExit("Some worlds failed: no aggregate published")
    totals, hashes = aggregate(args.output, reports)
    if file_hash(args.source / "sft.jsonl") != identity["source_hash"]:
        raise ValueError("Source mutated during conversion")
    counts = Counter()
    for report in reports:
        counts.update(report["counts"])
    result = {
        "status": "CONVERSION_PASS_PENDING_TOKEN_AUDIT",
        "worlds": len(groups),
        **totals,
        "counts": dict(counts),
        "hashes": hashes,
        "source_unchanged": True,
        "full_task_exposure": "four per task",
        "focus_task_exposure": "one per task",
        "identity": identity,
    }
    prior.atomic(args.output / "conversion_report.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"hashes", "identity"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
