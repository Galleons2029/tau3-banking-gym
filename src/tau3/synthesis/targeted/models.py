"""Versioned contracts for reproducible failure-directed rounds."""

from typing import Literal

from pydantic import Field, model_validator

from tau3.worldgen.v2.settings import RolloutSettings
from tau3.worldgen.v2.specs import StrictModel

RECIPES = {
    "discovery": ["F1", "F3"],
    "policy": ["F2"],
    "handoff": ["F3b"],
    "restraint": ["F4"],
    "escalation": ["F5", "F6"],
    "signature": ["F7"],
    "boundary": ["F8"],
}
BASE_WEIGHTS = dict(zip(RECIPES, [30, 25, 15, 10, 10, 5, 5]))
DIFFICULTIES = {"1-4": 15, "5-9": 25, "10-14": 25, "15-19": 20, "20-29": 15}


class RoundConfig(StrictModel):
    """Explicit finite limits include audits, users, judges and teacher calls."""

    schema_version: Literal[1] = 1
    seed: int = 42000
    num_tasks: int = Field(default=3000, ge=20)
    pilot_tasks: Literal[20] = 20
    trials: Literal[4] = 4
    candidate_attempts: int = Field(default=3, ge=1, le=3)
    max_total_calls: int = Field(default=1000000, ge=1)
    max_total_tokens: int = Field(default=1000000000, ge=1)
    max_audit_calls: int = Field(default=750000, ge=1)
    max_rollouts: int = Field(default=12080, ge=1)
    workers: int = Field(default=128, ge=1, le=256)
    llm_concurrency: int = Field(default=128, ge=1, le=256)
    analysis_batch_bytes: int = Field(default=120000, ge=10000, le=500000)
    attribution_timeout: int = Field(default=1800, ge=1, le=3600)
    teacher_model: str = "openai/GLM-5.3-Flash"
    reviewer_model: str = "openai/gemini-3.5-flash"
    settings: RolloutSettings = Field(
        default_factory=lambda: RolloutSettings(
            agent_models=["openai/GLM-5.3-Flash", "openai/gemini-3.5-flash"],
            user_model="openai/gemini-3.5-flash",
            judge_model="openai/gemini-3.5-flash",
            simulation_max_steps=200,
            simulation_timeout=1800,
            audit_transport_attempts=1,
            capture_retrieval="bm25_grep",
            agent_llm_args={"temperature": 0.7},
            user_llm_args={"temperature": 0},
            llm_args={
                "api_base": "http://10.39.62.231:9091/v1",
                "api_key": "not-required",
                "temperature": 0,
                "max_tokens": 65536,
                "timeout": 300,
                "num_retries": 0,
                "extra_body": {"reasoning_effort": "high"},
            },
        )
    )

    @model_validator(mode="after")
    def roles(self):
        """Do not silently degrade independent validation to a single model."""
        if set(self.settings.agent_models) != {self.teacher_model, self.reviewer_model}:
            raise ValueError("Audit models must be the distinct teacher and reviewer")
        if self.teacher_model == self.reviewer_model:
            raise ValueError("Two distinct model IDs are required")
        if self.settings.capture_retrieval != "bm25_grep":
            raise ValueError("Targeted v1 teacher collection requires bm25_grep")
        return self


class Finding(StrictModel):
    """An interpretation is distinct from its observable, source-bound evidence."""

    label: str
    observation: str
    interpretation: str
    source: str
    quote: str
    confidence: Literal["high", "medium", "low"]
    severity: Literal["high", "medium", "low"]
    affected_task_count: int | None = Field(default=None, ge=0)
    mechanisms: list[str] = Field(default_factory=list)
    quarantined: bool = False


class FailureProfile(StrictModel):
    """Round input and source evidence, with unknown counts left unknown."""

    schema_version: Literal[1] = 1
    round_id: str
    base_model: str
    report_hash: str
    run_hash: str | None = None
    findings: list[Finding]
    task_observations: dict[str, dict] = Field(default_factory=dict)
    task_attributions: dict[str, list[dict]] = Field(default_factory=dict)
    quarantine: list[dict] = Field(default_factory=list)


class Slot(StrictModel):
    """Frozen business/difficulty allocation; teacher outcomes cannot retarget it."""

    index: int
    recipe: str
    difficulty: str
    origin: Literal["current", "replay", "explore"]
    secondary: str | None = None
    seed: int


class SynthesisPlan(StrictModel):
    """Complete immutable allocation with evidence and previous-round provenance."""

    schema_version: Literal[1] = 1
    profile_hash: str
    parent_hash: str | None = None
    proposal: dict
    weights: dict[str, float]
    slots: list[Slot]
    pilot: list[Slot]
    backlog: list[dict] = Field(default_factory=list)


class TaskProvenance(StrictModel):
    """A task belongs to one exact world and one strategy allocation."""

    schema_version: Literal[1] = 1
    task_id: str
    world: str
    world_hash: str
    task_hash: str
    plan_hash: str
    business_fingerprint: str
    slot: Slot
    candidate: int
    reference_actions: int
    business_operations: int
    checks: dict
    status: Literal["VALID", "INVALID", "INCONCLUSIVE"]
