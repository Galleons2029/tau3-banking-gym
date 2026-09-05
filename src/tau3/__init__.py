"""
Tau3: Conversational Agent Benchmark Framework

Main exports for easy access to key components.
"""

import warnings

# Runner package: clean API for simulation execution
# - Layer 1: run_simulation (execute pre-built orchestrator)
# - Layer 2: build_* functions (construct instances from config/names)
# - Layer 3: run_domain, run_tasks, run_single_task (batch execution)
import tau3.runner as runner
from tau3.agent.base.llm_config import LLMConfigMixin
from tau3.agent.base_agent import HalfDuplexAgent
from tau3.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau3.data_model.simulation import (
    BaseRunConfig,
    RunConfig,
    SimulationRun,
    TextRunConfig,
)
from tau3.data_model.tasks import Task
from tau3.environment.environment import Environment
from tau3.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau3.orchestrator.orchestrator import Orchestrator
from tau3.registry import Registry, registry
from tau3.run import run_domain
from tau3.user.user_simulator import UserSimulator
from tau3.user.user_simulator_base import HalfDuplexUser
from tau3.utils.display import ConsoleDisplay, MarkdownDisplay

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================


def __getattr__(name: str):
    """Module-level __getattr__ for deprecation warnings."""
    deprecated_aliases = {
        "BaseAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "LocalAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "BaseUser": ("HalfDuplexUser", HalfDuplexUser),
    }

    if name in deprecated_aliases:
        new_name, new_class = deprecated_aliases[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct aliases for static analysis tools
BaseAgent = HalfDuplexAgent
LocalAgent = HalfDuplexAgent
BaseUser = HalfDuplexUser


__all__ = [
    # Core
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
    # Utils
    "ConsoleDisplay",
    "MarkdownDisplay",
    # Deprecated aliases (kept for backward compatibility)
    "BaseAgent",
    "LocalAgent",
    "BaseUser",
]
