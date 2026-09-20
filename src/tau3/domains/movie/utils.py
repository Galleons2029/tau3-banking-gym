# Copyright Sierra
"""Filesystem paths for the movie ticketing domain.

The path constants for the *data* files (db + policy) are defined in
``data_model.py`` so that ``data_model.py`` and ``tools.py`` stay self-contained
and runnable on their own. They are re-exported here (plus the task-set path) to
mirror the layout of the other domains, where ``utils.py`` is the canonical
place the environment imports paths from.
"""

from tau3.domains.movie.data_model import (
    MOVIE_DATA_DIR,
    MOVIE_DB_PATH,
    MOVIE_POLICY_PATH,
)

MOVIE_TASK_SET_PATH = MOVIE_DATA_DIR / "tasks.json"

__all__ = [
    "MOVIE_DATA_DIR",
    "MOVIE_DB_PATH",
    "MOVIE_POLICY_PATH",
    "MOVIE_TASK_SET_PATH",
]
