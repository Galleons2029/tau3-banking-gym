"""Strict public loader for published, validated task bundles."""

from pathlib import Path

from tau3.data_model.tasks import Task
from tau3.synthesis.models import VERSION
from tau3.synthesis.storage import digest, environment_fingerprint, read_json


def load_bundle_manifest(path: str | Path) -> dict:
    """Reject unpublished bundles and incompatible environment versions."""
    manifest = read_json(Path(path) / "manifest.json")
    if manifest.get("domain") == "banking_synth":
        from tau3.synthesis.world_tasks import load_world_task_manifest

        return load_world_task_manifest(Path(path))
    if manifest.get("schema_version") == 2 and manifest.get("backend") == "banking_native":
        from tau3.synthesis.targeted.native.bundle import validate_manifest

        return validate_manifest(Path(path), manifest)
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("domain") != "banking_knowledge"
    ):
        raise ValueError("Unsupported task bundle schema or domain")
    if manifest.get("status") != "published":
        raise ValueError("Task bundle has not passed publication gates")
    if manifest.get("environment_hash") != environment_fingerprint():
        raise ValueError(
            "Task bundle environment version mismatch; regenerate and validate"
        )
    if manifest.get("retrieval", {}).get("name") != "bm25_grep":
        raise ValueError("Unsupported training retrieval configuration")
    return manifest


def load_task_bundle(path: str | Path, split: str | None = None) -> list[Task]:
    """Load a validated bundle split; defaults to base without silent fallbacks."""
    root = Path(path)
    manifest = load_bundle_manifest(root)
    raw = read_json(root / "tasks.json")
    splits = read_json(root / "split_tasks.json")
    if digest(raw) != manifest.get("tasks_hash") or digest(splits) != manifest.get(
        "splits_hash"
    ):
        raise ValueError("Bundle artifacts changed after validation")
    tasks = [Task.model_validate(item) for item in raw]
    by_id = {t.id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task IDs in bundle")
    for name, ids in splits.items():
        if (
            not isinstance(ids, list)
            or len(ids) != len(set(ids))
            or set(ids) - by_id.keys()
        ):
            raise ValueError(f"Invalid split: {name}")
    if set(splits.get("base", [])) != set(by_id):
        raise ValueError("Base split must contain every task")
    train, validation = set(splits.get("train", [])), set(splits.get("validation", []))
    if train & validation or train | validation != set(by_id):
        raise ValueError("Train/validation must partition the bundle")
    groups = manifest.get("task_groups", {})
    if set(groups) != set(by_id):
        raise ValueError("Incomplete grouping metadata")
    if {groups[i] for i in train} & {groups[i] for i in validation}:
        raise ValueError("Business skeleton leaks across splits")
    selected = split or "base"
    if selected not in splits:
        raise ValueError(f"Unknown task split: {selected}")
    return [by_id[task_id] for task_id in splits[selected]]
