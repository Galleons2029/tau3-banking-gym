"""Polish residual lookup language, reusing unchanged world and token evidence."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import audit_targeted_revision_training as training
import convert_targeted_protocol as prior
import revise_targeted_sft as revision

CLAIM = re.compile(
    r"\b(?:you(?:\s+are|'re|’re)\s+(?:now\s+)?verified|(?:your\s+)?identity\s+(?:is\s+)?verified|identity\s+verification\s+(?:is\s+)?complete(?:d)?)\b",
    re.I,
)
FUTURE = re.compile(
    r"\b(once|when|after)\s+(?:you(?:\s+are|'re|’re)\s+(?:now\s+)?verified|(?:your\s+)?identity\s+(?:is\s+)?verified|identity\s+verification\s+(?:is\s+)?complete(?:d)?)\b",
    re.I,
)


@lru_cache(maxsize=8)
def display_names(world_view):
    """Only display-name spelling is case insensitive; IDs/emails remain exact."""
    return [
        (
            re.compile(re.escape(person["old"]["name"]), re.I),
            person["new"]["name"],
            person["new"]["email"].split("@")[1],
        )
        for person in prior.read(Path(world_view))["identity_map"]
    ]


def polish(row):
    """Fix conditional lookup wording and mask unsupported pre-lookup assertions."""
    result = copy.deepcopy(row)
    counts = Counter()
    located = False
    calls = {}
    edits = []
    supplied_customer_text = ""
    world_view = row.get("metadata", {}).get("data_revision", {}).get("world_view")
    name_rules = display_names(world_view) if world_view else []
    for index, message in enumerate(result["messages"]):
        if message["role"] == "user":
            supplied_customer_text += "\n" + (message.get("content") or "")
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = call["function"]["name"]
            if call["function"]["name"] == "find_customer":
                email = json.loads(call["function"]["arguments"]).get("email")
                if (
                    isinstance(email, str)
                    and "@example.invalid" in email
                    and email not in supplied_customer_text
                    and result["loss_mask"][index]
                ):
                    result["loss_mask"][index] = 0
                    message["weight"] = 0
                    counts["unsupported_lookup_inputs_masked"] += 1
                    edits.append(
                        {
                            "message": index,
                            "kind": "mask_lookup_email_not_supplied_by_user",
                            "input": email,
                        }
                    )
        if (
            message["role"] == "tool"
            and calls.get(message.get("tool_call_id")) == "find_customer"
        ):
            try:
                located |= bool(json.loads(message["content"]))
            except ValueError:
                pass
        if message["role"] != "assistant":
            continue
        text, number = FUTURE.subn(
            lambda m: m[1] + " your customer profile is located",
            message.get("content") or "",
        )
        counts["conditional_lookup_edits"] += number
        for pattern, replacement, domain in name_rules:
            text, number = pattern.subn(replacement, text)
            counts["display_name_case_edits"] += number
            if located:
                text, number = re.subn(
                    r"\bverified\s+as\s+" + re.escape(replacement),
                    "customer profile located for " + replacement,
                    text,
                    flags=re.I,
                )
                counts["named_profile_claim_edits"] += number
                if len(name_rules) == 1:
                    text, number = re.subn(
                        r"(?<![A-Za-z0-9._%+-])@example\.invalid\b",
                        "@" + domain,
                        text,
                        flags=re.I,
                    )
                    counts["abbreviated_email_domain_edits"] += number
        if CLAIM.search(text):
            if located:
                text, number = CLAIM.subn("the customer profile has been located", text)
                counts["remaining_lookup_assertion_edits"] += number
            elif result["loss_mask"][index]:
                result["loss_mask"][index] = 0
                message["weight"] = 0
                counts["premature_lookup_turns_masked"] += 1
                edits.append(
                    {
                        "message": index,
                        "kind": "mask_assertion_before_lookup",
                        "text": text,
                    }
                )
        if text != message.get("content", ""):
            edits.append(
                {
                    "message": index,
                    "kind": "conditional_or_lookup_wording",
                    "before_hash": prior.sha(message.get("content") or ""),
                    "after_hash": prior.sha(text),
                }
            )
        message["content"] = text
    if edits:
        result["metadata"]["text_polish"] = {
            "parent_row_hash": prior.sha(prior.canonical(row)),
            "edits": edits,
        }
    # Only assistant text and removal of supervision may differ. All observations,
    # calls, arguments, call order and customer messages remain exactly identical.
    assert result["tools"] == row["tools"]
    for old, new, old_loss, new_loss in zip(
        row["messages"],
        result["messages"],
        row["loss_mask"],
        result["loss_mask"],
        strict=True,
    ):
        if old["role"] == "assistant":
            assert {k: v for k, v in old.items() if k not in {"content", "weight"}} == {
                k: v for k, v in new.items() if k not in {"content", "weight"}
            }
        else:
            assert old == new
        assert new_loss <= old_loss
        if new_loss:
            assert not CLAIM.search(new.get("content") or "")
    prior.validate(result)
    return result, {"counts": dict(counts), "edits": edits}


def process(slot, parent, output, identity):
    """Adopt untouched proofs; regenerate actual token proofs for edited worlds."""
    source, target = Path(parent) / "shards" / slot, Path(output) / "shards" / slot
    target.mkdir(parents=True, exist_ok=True)
    original = prior.read(source / "report.json")
    for name, expected in original["files"].items():
        if revision.file_hash(source / name) != expected:
            raise ValueError("Parent world shard changed")
    token = prior.read(source / "token_report.json")
    if (
        token["status"] != "PASS"
        or revision.file_hash(source / "token_audit.jsonl") != token["audit_hash"]
    ):
        raise ValueError("Parent token evidence changed")
    rows = [json.loads(line) for line in (source / "sft.jsonl").open()]
    revised, audits, counts = [], [], Counter()
    for row in rows:
        value, audit = polish(row)
        revised.append(value)
        audits.append(audit)
        counts.update(audit["counts"])
    changed = any(a["edits"] for a in audits)
    if (target / "polish_report.json").exists():
        cached = prior.read(target / "polish_report.json")
        if cached["identity"] != identity:
            raise ValueError("Polished version changed")
        for name, expected in cached["files"].items():
            if revision.file_hash(target / name) != expected:
                raise ValueError("Polished shard changed")
        return cached
    for name in original["files"]:
        (target / name).write_bytes((source / name).read_bytes())
    files = original["files"].copy()
    if changed:
        files["sft.jsonl"] = revision.write_jsonl(target / "sft.jsonl", revised)
        candidates = {}
        for row in revised:
            for kind, sample in revision.focus_candidates(row).items():
                candidates.setdefault(kind, sample)
        kinds = [
            k
            for k in ("recovery", "handoff", "pagination", "execution", "discovery")
            if k in candidates
        ][:2]
        focused = [candidates[k] for k in kinds]
        files["focus.jsonl"] = revision.write_jsonl(target / "focus.jsonl", focused)
        source_audits = [json.loads(line) for line in (source / "audit.jsonl").open()]
        for entry, row, audit in zip(source_audits, revised, audits, strict=True):
            entry["parent_revised_row_hash"] = entry["revised_row_hash"]
            entry["revised_row_hash"] = prior.sha(prior.canonical(row))
            entry["text_polish"] = audit
        files["audit.jsonl"] = revision.write_jsonl(
            target / "audit.jsonl", source_audits
        )
    report = {
        **original,
        "files": files,
        "identity_hash": prior.sha(prior.canonical(identity)),
    }
    prior.atomic(target / "report.json", report)
    if changed:
        native_identity = {**token["identity"], "text_polish": identity}
        token = training.audit_group(
            str(target), token["identity"]["config"], native_identity
        )
    else:
        (target / "token_audit.jsonl").write_bytes(
            (source / "token_audit.jsonl").read_bytes()
        )
        (target / "token_report.json").write_bytes(
            (source / "token_report.json").read_bytes()
        )
    files = {
        **files,
        "token_audit.jsonl": revision.file_hash(target / "token_audit.jsonl"),
        "token_report.json": revision.file_hash(target / "token_report.json"),
    }
    result = {
        **report,
        "files": files,
        "identity": identity,
        "text_changed": changed,
        "polish_counts": dict(counts),
        "tokens_reused": not changed,
        "token_report": token,
    }
    prior.atomic(target / "polish_report.json", result)
    return result


def safe_process(*args):
    """Persist actionable failures without omitting samples from the result."""
    try:
        return process(*args)
    except Exception as exc:
        import traceback

        record = {
            "status": "FAIL",
            "slot": args[0],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        prior.atomic(Path(args[2]) / "failures" / (args[0] + ".json"), record)
        return record


def main():
    """Publish the final independent corpus after bounded language corrections."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    parent = prior.read(args.parent / "report.json")
    if parent["status"] != "PASS":
        raise ValueError("Parent revision is not ready")
    for name, expected in parent["hashes"].items():
        if revision.file_hash(args.parent / name) != expected:
            raise ValueError("Parent aggregate changed")
    identity = {
        "version": "targeted-data-revision-v2-final",
        "parent": str(args.parent.resolve()),
        "parent_report_hash": revision.file_hash(args.parent / "report.json"),
        "polisher_hash": revision.file_hash(__file__),
        "token_auditor_hash": revision.file_hash(training.__file__),
        "converter_hash": revision.file_hash(revision.__file__),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "identity.json").exists() and prior.read(
        args.output / "identity.json"
    ) != identity:
        raise ValueError("Final output is frozen")
    prior.atomic(args.output / "identity.json", identity)
    reports = []
    slots = [p.name for p in sorted((args.parent / "shards").iterdir())]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                safe_process, slot, str(args.parent), str(args.output), identity
            )
            for slot in slots
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            if (
                len(reports) % 100 == 0
                or reports[-1]["status"] != "PASS"
                or len(reports) == len(slots)
            ):
                progress = {
                    "status": "FINAL_POLISH",
                    "worlds": len(slots),
                    "processed": len(reports),
                    "failures": sum(r["status"] != "PASS" for r in reports),
                }
                prior.atomic(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if any(r["status"] != "PASS" for r in reports):
        raise SystemExit("Polish failed; no aggregate published")
    totals, hashes = revision.aggregate(args.output, reports)
    counts, token_totals, verified = Counter(), Counter(), set()
    token_path = args.output / "token_audit.jsonl"
    with token_path.open("w") as stream:
        for report in sorted(reports, key=lambda r: r["slot"]):
            counts.update(report["polish_counts"])
            detail = report["token_report"]["counts"].copy()
            maximum = detail.pop("longest_sample")
            token_totals.update(detail)
            token_totals["longest_sample"] = max(
                token_totals["longest_sample"], maximum
            )
            for line in (
                args.output / "shards" / report["slot"] / "token_audit.jsonl"
            ).open():
                verified.add(json.loads(line)["sample_hash"])
                stream.write(line)
    exposure = Counter()
    for prefix in ("", "focus.", "balanced."):
        with (
            (args.output / (prefix + "sft.jsonl")).open() as source,
            (args.output / (prefix + "general_agent.jsonl")).open() as public,
        ):
            for left, right in zip(source, public, strict=True):
                sample = json.loads(left)
                if prior.sha(prior.canonical(sample)) not in verified or prior.general(
                    sample
                ) != json.loads(right):
                    raise ValueError("Final row lacks matching token proof")
                if prefix == "balanced.":
                    exposure[sample["metadata"]["business_fingerprint"]] += 1
                for message, loss in zip(
                    sample["messages"], sample["loss_mask"], strict=True
                ):
                    if loss and CLAIM.search(message.get("content") or ""):
                        raise ValueError("Residual supervised lookup claim")
    if len(exposure) != parent["worlds"] or set(exposure.values()) != {5}:
        raise ValueError("Uneven final task exposure")
    if (
        revision.file_hash(args.parent / "report.json")
        != identity["parent_report_hash"]
    ):
        raise ValueError("Parent version changed")
    source_identity = parent.get(
        "source_identity", parent.get("parent_identity", parent["identity"])
    )
    if (
        revision.file_hash(Path(source_identity["source"]) / "sft.jsonl")
        != source_identity["source_hash"]
    ):
        raise ValueError("Original protocol corpus changed")
    for path, expected in source_identity["identity_sources"].items():
        if revision.file_hash(path) != expected:
            raise ValueError("Identity source changed")
    final = {
        **parent,
        "status": "PASS",
        **totals,
        "hashes": hashes,
        "parent_identity": parent["identity"],
        "source_identity": source_identity,
        "parent_polish_counts": parent.get("polish_counts", {}),
        "identity": identity,
        "polish_counts": dict(counts),
        "worlds_reencoded": sum(r["text_changed"] for r in reports),
        "worlds_reusing_exact_token_proofs": sum(r["tokens_reused"] for r in reports),
        "training_audit": {
            **parent["training_audit"],
            **token_totals,
            "token_audit_hash": revision.file_hash(token_path),
            "proof_strategy": "Exact unchanged sample hashes adopt parent native-token proof; edited worlds rerun the same encoder, tokenizer and loss checks.",
        },
        "residual_supervised_lookup_claims": 0,
        "business_replay_strategy": "Inherited parent dual replay; all tool schemas, calls, arguments, results, order, and user messages asserted unchanged by final polish.",
    }
    prior.atomic(args.output / "report.json", final)
    prior.atomic(
        args.output / "progress.json",
        {
            "status": "COMPLETE",
            "rows": final["rows"],
            "focus_rows": final["focus_rows"],
            "balanced_rows": final["balanced_rows"],
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                **totals,
                "polish_counts": dict(counts),
                "worlds_reencoded": final["worlds_reencoded"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
