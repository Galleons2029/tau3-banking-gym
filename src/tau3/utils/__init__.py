"""Utility exports loaded on first use.

Keeping this package initializer lightweight avoids importing ``deepdiff`` (and
therefore pandas) for callers that only need the data directory constant.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "dump_file": ("tau3.utils.io_utils", "dump_file"),
    "load_file": ("tau3.utils.io_utils", "load_file"),
    "get_pydantic_hash": ("tau3.utils.pydantic_utils", "get_pydantic_hash"),
    "update_pydantic_model_with_dict": (
        "tau3.utils.pydantic_utils",
        "update_pydantic_model_with_dict",
    ),
    "DATA_DIR": ("tau3.utils.utils", "DATA_DIR"),
    "get_dict_hash": ("tau3.utils.utils", "get_dict_hash"),
    "show_dict_diff": ("tau3.utils.utils", "show_dict_diff"),
    "llm_utils": ("tau3.utils.llm_utils", None),
}


def __getattr__(name: str) -> Any:
    """Resolve a utility export without eagerly importing unrelated helpers."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "dump_file",
    "load_file",
    "get_pydantic_hash",
    "update_pydantic_model_with_dict",
    "DATA_DIR",
    "get_dict_hash",
    "show_dict_diff",
]
