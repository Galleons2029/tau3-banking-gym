"""Bind a unique-trajectory SFT experiment before the external scale-up gate."""

import hashlib
import json

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.planning import stage_slots


def content_identity(row):
    """Count real visible trajectories rather than replicas or provenance edits."""
    return digest({key: row[key] for key in ("messages", "tools", "loss_mask")})


def audit_package(root, plan, stage):
    """Verify a whole fixed-four cohort and its actual unique training rows."""
    policy = plan.proposal.get("expansion", {})
    if stage not in policy.get("stages", {}):
        raise ValueError("Expansion stage is not in the frozen plan")
    folder = root / stage
    manifest = read_json(folder / "training_manifest.json")
    report = read_json(folder / "report.json")
    slots = stage_slots(plan, stage)
    checks = {
        "ready_manifest": manifest["status"] == "READY",
        "complete_cohort": report["status"] == "COMPLETE",
        "all_fixed_slots": report["complete_slots"] == report["expected_slots"] == 4 * len(slots),
        "all_tasks_valid": report["valid_tasks"] == report["expected_tasks"] == len(slots),
        "training_identity": manifest["round_identity"] == read_json(root / "identity.json"),
        "coverage": read_json(folder / "operation_coverage.json")["status"] == "PASS",
        "pairs": read_json(folder / "pair_checks.json")["status"] == "PASS",
    }
    content_ids, task_seeds = set(), set()
    row_hashes, stream_hash = [], hashlib.sha256()
    eligible = {(s.index, seed): s for s in slots for seed in s.trial_seeds}
    qualified_slots, exported_slots = set(), set()
    complete_slots = 0
    for slot in slots:
        for trial, seed in enumerate(slot.trial_seeds):
            path = root / "tasks/train" / f"{slot.index:05d}" / "trials" / str(trial) / "slot.json"
            if not path.exists():
                continue
            saved = read_json(path)
            if saved.get("seed") != seed:
                raise ValueError("Recorded sampling seed differs from frozen slot")
            complete_slots += saved["status"] == "COMPLETE"
            if saved.get("sft_qualified"):
                qualified_slots.add((slot.index, seed))
    checks["actual_complete_slots"] = complete_slots == 4 * len(slots)
    validation_groups = set(read_json(root / "validation/index.json")["group_ids"])
    with (folder / "sft.jsonl").open("rb") as handle:
        for line in handle:
            stream_hash.update(line)
            row = json.loads(line)
            source = row["metadata"]["provenance"]["slot"]
            slot = eligible.get((source["index"], row["seed"]))
            if slot is None or source != slot.model_dump(mode="json"):
                raise ValueError("Export contains a task outside the frozen cohort")
            if source["split"] != "train" or row["metadata"].get("dataset_partition") != "clean":
                raise ValueError("Non-training or recovery row in main export")
            if row["metadata"]["provenance"]["group_id"] in validation_groups:
                raise ValueError("Training trajectory overlaps a validation graph group")
            if row["loss_mask"] != [int(m["role"] == "assistant") for m in row["messages"]]:
                raise ValueError("Main export does not preserve assistant-only supervision")
            trial = slot.trial_seeds.index(row["seed"])
            saved = read_json(root / "tasks/train" / f"{slot.index:05d}" / "trials" / str(trial) / "slot.json")
            if saved["status"] != "COMPLETE" or not saved["sft_qualified"]:
                raise ValueError("Export lacks a qualified complete source slot")
            content_ids.add(content_identity(row))
            task_seeds.add((row["task_id"], row["seed"]))
            exported_slots.add((slot.index, row["seed"]))
            row_hashes.append(digest(row))
    count = len(row_hashes)
    checks.update(
        file_hash=stream_hash.hexdigest() == manifest["sft_sha256"],
        row_index=row_hashes == [r["row_hash"] for r in manifest["rows_index"]],
        row_count=count == manifest["rows"] == report["qualified_rows"],
        unique_task_seeds=len(task_seeds) == count,
        unique_content=len(content_ids) == count,
        all_qualified_slots_exported=qualified_slots == exported_slots,
        complete_token_audits=len(manifest["token_audits"]) == count and not manifest["training_failures"],
    )
    adapter_hash = hashlib.sha256()
    with (folder / "general_agent_reasoning.jsonl").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            adapter_hash.update(chunk)
    checks["adapter_file_hash"] = adapter_hash.hexdigest() == manifest["general_agent_sha256"]
    parent_retained = True
    config = read_json(root / "config.json")
    if config.get("retained_sft_manifest"):
        from tau3.synthesis.targeted.native.retention import audit_retention

        retained = audit_retention(root, content_ids)
        parent_retained = retained["passed"]
        checks["rejected_parent_rows_excluded"] = all(
            passed for label, passed in retained["checks"].items() if label != "KEEP")
    elif config.get("parent_round"):
        from pathlib import Path

        parent = Path(config["parent_round"]) / "small"
        previous = read_json(parent / "training_manifest.json")
        original_hash = hashlib.sha256()
        originals = set()
        with (parent / "sft.jsonl").open("rb") as handle:
            for line in handle:
                original_hash.update(line)
                originals.add(content_identity(json.loads(line)))
        parent_retained = (original_hash.hexdigest() == previous["sft_sha256"]
                           and originals <= content_ids)
    checks["parent_main_trajectories_retained"] = parent_retained
    enough = (len(content_ids) >= policy["target_unique_rows"]
              and len(slots) >= policy.get("minimum_handoff_tasks", 0))
    result = {
        "stage": stage, "plan_hash": digest(plan.model_dump(mode="json")),
        "manifest_hash": digest(manifest), "checks": checks,
        "unique_rows": len(content_ids), "target_unique_rows": policy["target_unique_rows"],
        "remaining_rows": max(0, policy["target_unique_rows"] - len(content_ids)),
        "status": "INCOMPLETE" if not all(checks.values()) else "READY" if enough else "MORE_TASKS_REQUIRED",
    }
    write_json(folder / "expansion-audit.json", result)
    return result


def handoff(root, plan, stage):
    """Publish an evaluation binding only after the unique-row target is met."""
    audit = audit_package(root, plan, stage)
    if audit["status"] != "READY":
        raise ValueError("Expansion has not met its complete unique-trajectory target")
    binding = {"status": "READY", "stage": stage, "manifest_hash": audit["manifest_hash"],
               "plan_hash": audit["plan_hash"], "audit_hash": digest(audit)}
    write_json(root / "training-handoff.json", binding)
    return binding


def evaluation_manifest(root, plan):
    """Legacy receipts bind small; expansion receipts bind the audited handoff."""
    if not plan.proposal.get("expansion"):
        return read_json(root / "small/training_manifest.json")
    binding = read_json(root / "training-handoff.json")
    if binding["status"] != "READY" or binding["plan_hash"] != digest(plan.model_dump(mode="json")):
        raise ValueError("Expansion evaluation handoff changed")
    audit = audit_package(root, plan, binding["stage"])
    if (audit["status"] != "READY" or digest(audit) != binding["audit_hash"]
            or audit["manifest_hash"] != binding["manifest_hash"]):
        raise ValueError("Expansion training evidence changed after handoff")
    return read_json(root / binding["stage"] / "training_manifest.json")
