"""Publish native task bundles for the existing evaluation runner."""

import hashlib

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.snapshot import sources


def publish(root, folder, candidates, plan, split):
    """Export tasks only after every slot in this stage passes validity admission."""
    target = root / folder / "bundle"
    tasks = [c.task.model_dump(mode="json") for c in candidates]
    ids = [t["id"] for t in tasks]
    splits = {"base": ids, "train": ids if split == "train" else [],
              "validation": ids if split == "validation" else []}
    snapshot = read_json(root / "snapshot/manifest.json")
    manifest = {"schema_version": 2, "backend": "banking_native", "domain": "banking_knowledge",
                "status": "published", "retrieval": {"name": "bm25_grep"},
                "tasks_hash": digest(tasks), "splits_hash": digest(splits),
                "task_groups": {c.task.id: c.group_id for c in candidates},
                "plan_hash": digest(plan.model_dump(mode="json")), "snapshot": snapshot}
    write_json(target / "tasks.json", tasks)
    write_json(target / "split_tasks.json", splits)
    write_json(target / "manifest.json", manifest)


def validate_manifest(root, manifest):
    """Check actual runtime and public corpus; keep official tasks out of the bundle."""
    if manifest.get("domain") != "banking_knowledge" or manifest.get("status") != "published":
        raise ValueError("Unpublished or incompatible native bundle")
    if manifest.get("retrieval", {}).get("name") != "bm25_grep":
        raise ValueError("Native bundle requires bm25_grep")
    snapshot = manifest["snapshot"]
    if snapshot["snapshot_hash"] != digest({k: v for k, v in snapshot.items() if k != "snapshot_hash"}):
        raise ValueError("Native snapshot manifest changed")
    actual = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources().items()}
    if actual != snapshot["hashes"]:
        raise ValueError("Native evaluation runtime or corpus differs from frozen snapshot")
    return manifest
