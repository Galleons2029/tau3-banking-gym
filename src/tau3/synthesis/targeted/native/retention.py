"""Bind the independently requalified historical subset without rerolling seeds."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.user_roles import (
    role_context,
    role_disposition,
    validate_role_review,
)


def sha256(path):
    """Hash immutable retention artifacts using their actual file bytes."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bind_retention(root, config):
    """Freeze the manifest/index identity once; later mutation fails closed."""
    if not getattr(config, "retained_sft_manifest", None):
        return
    path = Path(config.retained_sft_manifest).resolve()
    manifest = read_json(path)
    index = path.parent / "sampling_index.json"
    if manifest.get("status") != "COMPLETE" or sha256(index) != manifest["mapping_sha256"]:
        raise ValueError("Filtered retention is incomplete or its index changed")
    binding = {"manifest": str(path), "manifest_sha256": sha256(path),
               "mapping_sha256": sha256(index), "source_sha256": manifest["source_sha256"]}
    destination = root / "retained-sft-binding.json"
    if destination.exists() and read_json(destination) != binding:
        raise ValueError("Filtered retention binding changed")
    write_json(destination, binding)
    retention_index(str(path), binding["manifest_sha256"])


@lru_cache(maxsize=2)
def retention_index(manifest_path, manifest_hash):
    """Load only the small review index, never the full SFT into every worker."""
    path = Path(manifest_path)
    if sha256(path) != manifest_hash:
        raise ValueError("Filtered manifest changed")
    manifest = read_json(path)
    index = path.parent / "sampling_index.json"
    if sha256(index) != manifest["mapping_sha256"]:
        raise ValueError("Filtered review index changed")
    rows = read_json(index)
    keys = [(r["task_id"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != manifest["source_rows"]:
        raise ValueError("Filtered index contains duplicate or missing source slots")
    return dict(zip(keys, rows, strict=True))


def inherited_role_review(root, config, directory, result):
    """Reuse role judgments only for an exact, already adopted historical capture."""
    if not getattr(config, "retained_sft_manifest", None) or not (directory / "adopted-capture.json").exists():
        return None
    binding = read_json(root / "retained-sft-binding.json")
    entry = retention_index(binding["manifest"], binding["manifest_sha256"]).get(
        (result["simulation"]["task_id"], result["identity"]["seed"]))
    if entry is None:
        return None
    context = role_context(result["simulation"]["messages"])
    reviews = entry["reviews"]
    complete = [r for r in reviews if r.get("status") == "COMPLETE"]
    if not complete or any(r["context_hash"] != digest(context) for r in complete):
        raise ValueError("Inherited role judgment does not match captured customer context")
    for review in complete:
        validate_role_review(review["review"], context)
    if {r["model"] for r in reviews} != {config.teacher_model, config.reviewer_model}:
        raise ValueError("Historical role reviewers differ from the configured models")
    disposition = role_disposition(reviews)
    # Human-confirmed errors can remain rejected even if a machine disagreed.
    if entry["disposition"] == "KEEP" and disposition != "KEEP":
        raise ValueError("Historical retained row lacks two complete role approvals")
    return {"result_hash": digest(result), "disposition": entry["disposition"],
            "reviews": reviews, "origin": {**binding, "source_line": entry["source_line"],
             "source_line_sha256": entry["source_line_sha256"],
             "historical_user_prompt_preserved": True}}


def audit_retention(root, content_ids):
    """Require every retained row and reject every discarded/quarantined row."""
    from tau3.synthesis.targeted.native.expansion import content_identity

    binding = read_json(root / "retained-sft-binding.json")
    path = Path(binding["manifest"])
    if sha256(path) != binding["manifest_sha256"]:
        raise ValueError("Retained manifest changed before handoff")
    manifest = read_json(path)
    if sha256(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("Original retention source changed")
    checks, counts = {}, {}
    for disposition, record in manifest["files"].items():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Filtered partition bytes changed")
        with Path(record["path"]).open() as handle:
            identities = [content_identity(json.loads(line)) for line in handle]
        if len(identities) != len(set(identities)) or len(identities) != manifest["counts"][disposition]:
            raise ValueError("Filtered partition has duplicates or a count mismatch")
        counts[disposition] = len(identities)
        checks[disposition] = (set(identities) <= content_ids if disposition == "KEEP"
                               else not set(identities) & content_ids)
    return {"passed": all(checks.values()), "checks": checks, "counts": counts,
            "manifest_sha256": binding["manifest_sha256"]}
