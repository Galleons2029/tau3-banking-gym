"""Explicit synthesis rollout settings, shared by actors and semantic judging."""

import os
from pathlib import Path

from pydantic import Field

from tau3.config import WORLDGEN_ROLLOUT_API_BASE, WORLDGEN_ROLLOUT_MODELS
from tau3.worldgen.v2.specs import StrictModel


class RolloutSettings(StrictModel):
    """Gateway model names and bounded per-request settings; no GPT fallback."""

    agent_models: list[str] = Field(
        default_factory=lambda: list(WORLDGEN_ROLLOUT_MODELS), min_length=1
    )
    user_model: str = WORLDGEN_ROLLOUT_MODELS[0]
    judge_model: str = WORLDGEN_ROLLOUT_MODELS[1]
    simulation_max_steps: int = Field(default=40, ge=1, le=500)
    simulation_timeout: int = Field(default=300, ge=1, le=3600)
    audit_transport_attempts: int = Field(default=3, ge=1, le=5)
    agent_llm_args: dict = Field(default_factory=dict)
    user_llm_args: dict = Field(default_factory=dict)
    capture_retrieval: str = "bm25"
    llm_args: dict = Field(
        default_factory=lambda: {
            "api_base": WORLDGEN_ROLLOUT_API_BASE,
            "api_key": "not-required",
            "temperature": 0,
            "max_tokens": 8192,
            "timeout": 60,
            "num_retries": 0,
        }
    )


def load_settings(path: Path | None = None) -> RolloutSettings:
    """Read an explicitly selected local settings file or repository defaults."""
    import yaml

    configured = path or os.environ.get("TAU3_SYNTH_ROLLOUT_CONFIG")
    return (
        RolloutSettings.model_validate(yaml.safe_load(Path(configured).read_text()))
        if configured
        else RolloutSettings()
    )
