"""Environment for a synthesized banking world.

V2 bundles use their own typed database and declarative operations. Retrieval
components are shared with `banking_knowledge`; business state and policy are
world-local. Legacy bundles remain available only for explicit development.
"""

import json
from pathlib import Path
from typing import Optional

from tau3.data_model.tasks import Task
from tau3.domains.banking_knowledge.data_model import KnowledgeBase, TransactionalDB
from tau3.domains.banking_knowledge.retrieval import (
    DEFAULT_RETRIEVAL_VARIANT,
    build_policy,
    build_tools,
    resolve_variant,
)
from tau3.domains.banking_knowledge.tools import KnowledgeUserTools
from tau3.environment.environment import Environment
from tau3.worldgen.runtime import alias_toolkits
from tau3.worldgen.world import World, configured_world_root, load_world

DOMAIN_NAME = "banking_synth"


def get_world() -> World:
    """Load the world bundle this process serves."""
    return load_world(configured_world_root())


def get_db(world: Optional[World] = None) -> TransactionalDB:
    """Load the synthesized transactional database."""
    root = configured_world_root()
    if _is_v2(root):
        from tau3.worldgen.v2.runtime import WorldDB

        return WorldDB.model_validate_json((root / "db.json").read_text())
    return TransactionalDB.load(str((world or get_world()).db_path))


def get_knowledge_base(world: Optional[World] = None) -> KnowledgeBase:
    """Load the synthesized knowledge base."""
    root = configured_world_root()
    if _is_v2(root):
        from tau3.worldgen.v2.environment import load_spec
        from tau3.worldgen.world import allow_draft

        load_spec(root, allow_draft())
        return KnowledgeBase.load(str(root / "documents"))
    return KnowledgeBase.load(str((world or get_world()).documents_dir))


def _is_v2(root: Path) -> bool:
    """Inspect only the artifact format; the V2 loader validates integrity."""
    path = root / "manifest.json"
    return path.exists() and json.loads(path.read_text()).get("schema_version") == 2


def get_environment(
    db: Optional[TransactionalDB] = None,
    retrieval_variant: Optional[str] = None,
    retrieval_kwargs: Optional[dict] = None,
    task: Optional[Task] = None,
    solo_mode: bool = False,
    read_log_allowlist: Optional[set] = None,
) -> Environment:
    """Build the environment for the configured synthesized world.

    Mirrors `banking_knowledge.environment.get_environment`; see that docstring
    for the shared arguments.
    """
    if solo_mode:
        raise ValueError("banking_synth domain does not support solo mode")

    root = configured_world_root()
    if _is_v2(root):
        from tau3.worldgen.v2.environment import get_environment as get_v2_environment

        return get_v2_environment(
            root,
            db=db,
            retrieval_variant=retrieval_variant,
            retrieval_kwargs=retrieval_kwargs,
            task=task,
            solo_mode=solo_mode,
        )

    world = get_world()
    if db is None:
        db = get_db(world)
    knowledge_base = get_knowledge_base(world)

    variant = resolve_variant(
        retrieval_variant or DEFAULT_RETRIEVAL_VARIANT, **(retrieval_kwargs or {})
    )
    tools = build_tools(
        variant, db, knowledge_base, read_log_allowlist=read_log_allowlist
    )
    user_tools = KnowledgeUserTools(db)
    tools, user_tools = alias_toolkits(tools, user_tools, world.alias_map)
    policy = build_policy(variant, knowledge_base, task)

    return Environment(
        domain_name=DOMAIN_NAME,
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """Load the world's tasks, optionally restricted to a named split."""
    root = configured_world_root()
    if _is_v2(root):
        from tau3.worldgen.v2.environment import load_spec
        from tau3.worldgen.world import allow_draft

        load_spec(root, allow_draft())
        splits = json.loads((root / "splits.json").read_text())
        split = task_split_name or "base"
        if split not in splits:
            raise ValueError(f"Unknown task split: {split}")
        return [
            Task.model_validate_json((root / f"tasks/{tid}.json").read_text())
            for tid in splits[split]
        ]
    world = get_world()
    tasks: list[Task] = []
    for task_file in sorted(Path(world.tasks_dir).glob("task_*.json")):
        tasks.append(Task.model_validate(json.loads(task_file.read_text())))

    # "base" is every task, matching how published task bundles define it.
    if task_split_name and task_split_name != "base":
        wanted = world.manifest.task_splits.get(task_split_name)
        if not wanted:
            raise ValueError(
                f"Unknown task split {task_split_name!r} for world {world.root}"
            )
        tasks = [task for task in tasks if task.id in set(wanted)]
    return tasks
