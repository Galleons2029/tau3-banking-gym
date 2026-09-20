# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau3.data_model.tasks import Task
from tau3.domains.movie.data_model import MovieDB
from tau3.domains.movie.tools import MovieTools
from tau3.domains.movie.utils import (
    MOVIE_DB_PATH,
    MOVIE_POLICY_PATH,
    MOVIE_TASK_SET_PATH,
)
from tau3.environment.environment import Environment
from tau3.utils import load_file


def get_environment(
    db: Optional[MovieDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("movie domain does not support solo mode")
    if db is None:
        db = MovieDB.load(MOVIE_DB_PATH)
    tools = MovieTools(db)
    with open(MOVIE_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
    return Environment(
        domain_name="movie",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """Load the movie task set.

    If ``task_split_name`` is None (the default) all tasks are returned. A
    split file (``split_tasks.json``) is optional; it is only consulted when a
    split name is explicitly requested.
    """
    tasks = load_file(MOVIE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(task_splits.keys())}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(MOVIE_TASK_SET_PATH).parent
        / f"split_{Path(MOVIE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
