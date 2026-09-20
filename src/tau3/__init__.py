"""Tau3 conversational agent benchmark framework.

Public objects are imported lazily so lightweight commands such as ``tau3 web`` do
not pay the import cost of the simulation and LLM stacks during startup.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "runner": ("tau3.runner", None),
    "LLMConfigMixin": ("tau3.agent.base.llm_config", "LLMConfigMixin"),
    "HalfDuplexAgent": ("tau3.agent.base_agent", "HalfDuplexAgent"),
    "LLMAgent": ("tau3.agent.llm_agent", "LLMAgent"),
    "LLMSoloAgent": ("tau3.agent.llm_agent", "LLMSoloAgent"),
    "BaseRunConfig": ("tau3.data_model.simulation", "BaseRunConfig"),
    "RunConfig": ("tau3.data_model.simulation", "RunConfig"),
    "SimulationRun": ("tau3.data_model.simulation", "SimulationRun"),
    "TextRunConfig": ("tau3.data_model.simulation", "TextRunConfig"),
    "Task": ("tau3.data_model.tasks", "Task"),
    "Environment": ("tau3.environment.environment", "Environment"),
    "EvaluationType": ("tau3.evaluator.evaluator", "EvaluationType"),
    "evaluate_simulation": ("tau3.evaluator.evaluator", "evaluate_simulation"),
    "Orchestrator": ("tau3.orchestrator.orchestrator", "Orchestrator"),
    "Registry": ("tau3.registry", "Registry"),
    "registry": ("tau3.registry", "registry"),
    "run_domain": ("tau3.run", "run_domain"),
    "UserSimulator": ("tau3.user.user_simulator", "UserSimulator"),
    "HalfDuplexUser": ("tau3.user.user_simulator_base", "HalfDuplexUser"),
    "ConsoleDisplay": ("tau3.utils.display", "ConsoleDisplay"),
    "MarkdownDisplay": ("tau3.utils.display", "MarkdownDisplay"),
}

_DEPRECATED_ALIASES = {
    "BaseAgent": "HalfDuplexAgent",
    "LocalAgent": "HalfDuplexAgent",
    "BaseUser": "HalfDuplexUser",
}


def _load_export(name: str) -> Any:
    module_name, attribute = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Resolve the stable public API only when a caller first accesses it."""

    if name in _DEPRECATED_ALIASES:
        replacement = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} is deprecated, use {replacement} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _load_export(replacement)
        globals()[name] = value
        return value
    if name in _EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""

    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Orchestrator",
    "LLMAgent",
    "LLMSoloAgent",
    "LLMConfigMixin",
    "UserSimulator",
    "HalfDuplexAgent",
    "HalfDuplexUser",
    "Environment",
    "Registry",
    "registry",
    "SimulationRun",
    "Task",
    "evaluate_simulation",
    "EvaluationType",
    "BaseRunConfig",
    "TextRunConfig",
    "RunConfig",
    "run_domain",
    "runner",
    "ConsoleDisplay",
    "MarkdownDisplay",
    "BaseAgent",
    "LocalAgent",
    "BaseUser",
]
