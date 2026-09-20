"""World bundle layout, integrity and publication state.

A bundle mirrors an installed domain directory (`documents/`, `db.json`,
`tasks/`) so the synthesized world loads through the benchmark's own models. The
loader fails closed: a draft, altered or environment-incompatible bundle is
never silently accepted, matching `synthesis.bundle.load_task_bundle`.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from tau3.synthesis.storage import (
    digest,
    environment_fingerprint,
    read_json,
    write_json,
)
from tau3.worldgen.models import (
    VERSION,
    WorldConfig,
    WorldManifest,
    WorldPlan,
    WorldSchema,
)

MANIFEST = "manifest.json"
TARGETS = "targets.json"
DRAFT_ENV_VAR = "TAU3_SYNTH_ALLOW_DRAFT"
WORLD_ENV_VAR = "TAU3_SYNTH_WORLD"


@dataclass(frozen=True)
class World:
    """A loaded world bundle."""

    root: Path
    manifest: WorldManifest

    @property
    def alias_map(self) -> dict[str, str]:
        return self.manifest.alias_map

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def db_path(self) -> Path:
        return self.root / "db.json"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"


def _hash_tree(path: Path) -> str:
    """Hash a file, or a directory by its sorted relative paths and contents."""
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.exists():
        return ""
    members = {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }
    return digest(members)


def artifact_hashes(root: Path) -> dict[str, str]:
    """Hash the artifacts a published world is scored and replayed against."""
    return {
        name: _hash_tree(root / name)
        for name in ("documents", "db.json", "tasks")
        if (root / name).exists()
    }


def new_manifest(
    name: str, config: WorldConfig, alias_map: dict[str, str], targets_hash: str = ""
) -> WorldManifest:
    """Create a draft manifest pinned to the current environment and config."""
    from tau3.synthesis.storage import retrieval_settings

    return WorldManifest(
        name=name,
        seed=config.seed,
        suffix_salt=config.suffix_salt,
        alias_map=alias_map,
        environment_hash=environment_fingerprint(),
        config_hash=digest(config.model_dump(mode="json")),
        targets_hash=targets_hash,
        retrieval=retrieval_settings(),
        models={
            "generator": config.generator_model,
            "teacher": config.teacher_model,
            "user": config.user_model,
            "judge": config.judge_model,
        },
    )


def write_world_manifest(root: Path, manifest: WorldManifest) -> None:
    """Persist a manifest atomically."""
    write_json(Path(root) / MANIFEST, manifest.model_dump(mode="json"))


def read_world_manifest(root: Path) -> WorldManifest:
    """Read a manifest without checking publication state."""
    return WorldManifest.model_validate(read_json(Path(root) / MANIFEST))


def allow_draft() -> bool:
    """Whether draft worlds may be loaded; development only."""
    return os.environ.get(DRAFT_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def load_world(root: str | Path, *, require_published: bool | None = None) -> World:
    """Load a bundle, rejecting drafts, tampering and environment drift."""
    root = Path(root)
    if require_published is None:
        require_published = not allow_draft()
    manifest = read_world_manifest(root)

    if manifest.schema_version != VERSION:
        raise ValueError("Unsupported world bundle schema version")
    if manifest.environment_hash != environment_fingerprint():
        raise ValueError(
            "World bundle environment version mismatch; regenerate and validate"
        )
    if require_published and manifest.status != "published":
        raise ValueError(
            f"World bundle {root} has not passed publication gates "
            f"(set {DRAFT_ENV_VAR}=1 for development loads)"
        )
    if require_published:
        raise ValueError(
            "Legacy published flags are not validation certificates; "
            "regenerate with build-v2 or explicitly allow development loading"
        )
    if manifest.artifact_hashes:
        current = artifact_hashes(root)
        changed = [
            name
            for name, value in manifest.artifact_hashes.items()
            if current.get(name) != value
        ]
        if changed:
            raise ValueError(f"World artifacts changed after validation: {changed}")
    if manifest.alias_map:
        official = set(manifest.alias_map.values())
        if set(manifest.alias_map) & official:
            raise ValueError("Alias map collides with official tool names")
    return World(root=root, manifest=manifest)


def configured_world_root() -> Path:
    """Path of the world the synth domain serves."""
    value = os.environ.get(WORLD_ENV_VAR, "").strip()
    if not value:
        raise ValueError(
            f"{WORLD_ENV_VAR} is not set; the banking_synth domain needs a world bundle"
        )
    return Path(value)


SCHEMA_FILE = "schema/world.json"
PLAN_FILE = "plan/world.json"


def write_schema(root: Path, schema: WorldSchema) -> None:
    """Persist the structured layer."""
    write_json(Path(root) / SCHEMA_FILE, schema.model_dump(mode="json"))


def load_schema(root: Path) -> WorldSchema:
    """Read the structured layer."""
    return WorldSchema.model_validate(read_json(Path(root) / SCHEMA_FILE))


def write_plan(root: Path, plan: WorldPlan) -> None:
    """Persist document allocation and task specifications."""
    write_json(Path(root) / PLAN_FILE, plan.model_dump(mode="json"))


def load_plan(root: Path) -> WorldPlan:
    """Read document allocation and task specifications."""
    return WorldPlan.model_validate(read_json(Path(root) / PLAN_FILE))


def load_tasks(root: Path) -> list:
    """Read the benchmark-facing task files in stable order."""
    from tau3.data_model.tasks import Task

    return [
        Task.model_validate(read_json(path))
        for path in sorted((Path(root) / "tasks").glob("task_*.json"))
    ]
