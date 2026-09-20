"""Audit revised SFT with the configured native tokenizer and training adapter."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import convert_targeted_protocol as prior
from revise_targeted_sft import ROOT, file_hash


def general_training_sample(row):
    """Convert public per-message loss flags into the reference encoder contract."""
    sample = {
        "schema_version": 2,
        "tools": copy.deepcopy(row["tools"]),
        "messages": [],
        "loss_mask": [],
    }
    for message in row["messages"]:
        if type(message.get("loss")) is not bool:
            raise ValueError("Every public message requires an explicit boolean loss")
        if message["loss"] and message["role"] != "assistant":
            raise ValueError("Non-assistant supervision is forbidden")
        sample["loss_mask"].append(int(message["loss"]))
        sample["messages"].append(
            {k: copy.deepcopy(v) for k, v in message.items() if k != "loss"}
        )
    return sample


def audit_group(directory, config, identity):
    """Validate all full and focus samples of one world at token/label level."""
    from tau3.synthesis.targeted.native.training import encode_sample

    directory = Path(directory)
    conversion = prior.read(directory / "report.json")
    for name, expected in conversion["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("World/shard changed before token audit")
    world = prior.read(directory / "world_view.json")
    report_path = directory / "token_report.json"
    if report_path.exists():
        report = prior.read(report_path)
        if (
            report["identity"] != identity
            or file_hash(directory / "token_audit.jsonl") != report["audit_hash"]
        ):
            raise ValueError("Frozen token audit changed")
        return report
    records = []
    totals = Counter()
    for family, filename in (("full", "sft.jsonl"), ("focus", "focus.jsonl")):
        for line_index, line in enumerate((directory / filename).open()):
            sample = json.loads(line)
            if (
                sample["metadata"]["data_revision"]["world_view_hash"]
                != conversion["files"]["world_view.json"]
            ):
                raise ValueError("Sample refers to a different derived world")
            provenance_encoded = encode_sample(
                sample, config["checkpoint"], config["max_length"]
            )
            training_sample = general_training_sample(prior.general(sample))
            encoded = encode_sample(
                training_sample, config["checkpoint"], config["max_length"]
            )
            ids, labels = encoded["input_ids"], encoded["labels"]
            if len(ids) != len(labels) or any(
                label not in (-100, token)
                for label, token in zip(labels, ids, strict=True)
            ):
                raise ValueError("Malformed token supervision")
            record = {
                "family": family,
                "line": line_index + 1,
                "sample_hash": prior.sha(prior.canonical(sample)),
                "general_training_sample_hash": prior.sha(
                    prior.canonical(training_sample)
                ),
                "provenance_format_audit": provenance_encoded["audit"],
                "audit_format": "general_agent_via_general_training_sample",
                **encoded["audit"],
            }
            records.append(record)
            totals[family + "_rows"] += 1
            totals[family + "_tokens"] += len(ids)
            totals[family + "_supervised_tokens"] += sum(
                label != -100 for label in labels
            )
            totals["longest_sample"] = max(totals["longest_sample"], len(ids))
    from revise_targeted_sft import write_jsonl

    audit_hash = write_jsonl(directory / "token_audit.jsonl", records)
    result = {
        "status": "PASS",
        "slot": directory.name,
        "identity": identity,
        "counts": dict(totals),
        "audit_hash": audit_hash,
        "identities": [record["new"] for record in world["identity_map"]],
    }
    prior.atomic(report_path, result)
    return result


def safe_audit(*args):
    """Return explicit failure evidence instead of omitting an invalid trajectory."""
    try:
        return audit_group(*args)
    except Exception as exc:
        import traceback

        result = {
            "status": "FAIL",
            "slot": Path(args[0]).name,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        prior.atomic(Path(args[0]) / "token_failure.json", result)
        return result


def main():
    """Mark ready only after all rows and balanced selections pass native audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/synthesis/targeted-native-training.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.round_dir
    conversion = prior.read(root / "conversion_report.json")
    config = prior.read(args.config)
    if (
        config.get("enable_thinking") is not True
        or config.get("clear_thinking") is not False
    ):
        raise ValueError(
            "Current reference encoder only supports the declared thinking settings"
        )
    for name, expected in conversion["hashes"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Aggregate integrity mismatch: " + name)
    checkpoint = Path(config["checkpoint"])
    identity = {
        "conversion_hash": file_hash(root / "conversion_report.json"),
        "config": config,
        "auditor_hash": file_hash(__file__),
        "adapter_hash": file_hash(
            ROOT / "src/tau3/synthesis/targeted/native/training.py"
        ),
        "validator_hash": file_hash(ROOT / "src/tau3/synthesis/world_sft.py"),
        "tokenizer_hashes": {
            name: file_hash(checkpoint / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        },
    }
    directories = sorted((root / "shards").iterdir())
    reports = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(safe_audit, str(d), config, identity) for d in directories
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            if (
                len(reports) % 50 == 0
                or reports[-1]["status"] != "PASS"
                or len(reports) == len(directories)
            ):
                progress = {
                    "status": "TOKEN_AUDIT",
                    "processed": len(reports),
                    "worlds": len(directories),
                    "failures": sum(r["status"] != "PASS" for r in reports),
                }
                prior.atomic(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if any(r["status"] != "PASS" for r in reports):
        prior.atomic(
            root / "report.json",
            {
                "status": "FAIL",
                "phase": "token_audit",
                "failures": [r for r in reports if r["status"] != "PASS"],
            },
        )
        raise SystemExit(1)
    totals = Counter()
    verified = set()
    identity_ids, identity_emails = set(), set()
    for report in reports:
        for person in report["identities"]:
            if person["user_id"] in identity_ids or person["email"] in identity_emails:
                raise ValueError("Cross-world customer identity collision")
            identity_ids.add(person["user_id"])
            identity_emails.add(person["email"])
    temporary = root / "token_audit.jsonl.tmp"
    with temporary.open("w") as target:
        for report in sorted(reports, key=lambda r: r["slot"]):
            count = report["counts"].copy()
            maximum = count.pop("longest_sample")
            totals.update(count)
            totals["longest_sample"] = max(totals["longest_sample"], maximum)
            path = root / "shards" / report["slot"] / "token_audit.jsonl"
            if file_hash(path) != report["audit_hash"]:
                raise ValueError("Token evidence changed")
            for line in path.open():
                verified.add(json.loads(line)["sample_hash"])
                target.write(line)
    temporary.replace(root / "token_audit.jsonl")
    task_exposure = Counter()
    with (
        (root / "balanced.sft.jsonl").open() as samples,
        (root / "sampling.jsonl").open() as schedule,
    ):
        for line, selection in zip(samples, schedule, strict=True):
            row, selected = json.loads(line), json.loads(selection)
            sample_hash = prior.sha(prior.canonical(row))
            if sample_hash not in verified or sample_hash != selected["sample_hash"]:
                raise ValueError("Balanced selection lacks token-level proof")
            task_exposure[row["metadata"]["business_fingerprint"]] += 1
    if len(task_exposure) != conversion["worlds"] or set(task_exposure.values()) != {5}:
        raise ValueError("Task exposure is not uniform")
    # General-agent files must be exactly the public projection of audited samples.
    for prefix in ("", "focus.", "balanced."):
        with (
            (root / (prefix + "sft.jsonl")).open() as source,
            (root / (prefix + "general_agent.jsonl")).open() as public,
        ):
            for left, right in zip(source, public, strict=True):
                if prior.general(json.loads(left)) != json.loads(right):
                    raise ValueError(
                        "Public training projection differs from audited row"
                    )
    for name, expected in conversion["hashes"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Aggregate changed during training audit")
    if (
        file_hash(Path(conversion["identity"]["source"]) / "sft.jsonl")
        != conversion["identity"]["source_hash"]
    ):
        raise ValueError("Original corpus changed")
    for path, expected in conversion["identity"]["identity_sources"].items():
        if file_hash(path) != expected:
            raise ValueError("Identity source changed")
    final = {
        **conversion,
        "status": "PASS",
        "training_audit": {
            "status": "PASS",
            "identity": identity,
            **totals,
            "token_audit_hash": file_hash(root / "token_audit.jsonl"),
            "uniform_task_exposure": 5,
            "general_agent_projection": "PASS",
            "unique_customer_ids": len(identity_ids),
            "unique_customer_emails": len(identity_emails),
            "model_training_started": False,
        },
        "scope": "Offline identity/contract/text/supervision revision, typed-equivalent dual replay. Retrieval observations retained with mapped identities; no fresh retrieval or LLM quality recertification. Actual external training loader must use the audited reference adapter or equivalent semantics.",
    }
    prior.atomic(root / "report.json", final)
    prior.atomic(
        root / "progress.json",
        {
            "status": "COMPLETE",
            "worlds": final["worlds"],
            "rows": final["rows"],
            "focus_rows": final["focus_rows"],
            "balanced_rows": final["balanced_rows"],
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": final["rows"],
                "focus_rows": final["focus_rows"],
                "balanced_rows": final["balanced_rows"],
                **totals,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
