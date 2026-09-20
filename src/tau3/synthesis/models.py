"""Versioned synthesis artifacts; exported tasks retain the benchmark schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau3.config import DEFAULT_LLM_SYNTHESIS
from tau3.data_model.tasks import Task

FAMILIES = ("selection", "cashback", "credit_limit", "ordering")
RETRIEVAL = "bm25_grep"
VERSION = 1


class SynthesisConfig(BaseModel):
    """Bounded generation and validation settings, persisted with every bundle."""

    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    concurrency: int = Field(default=4, ge=1, le=32)
    candidate_attempts: int = Field(default=10, ge=1, le=10)
    text_revisions: int = Field(default=2, ge=0, le=2)
    generator_model: str = DEFAULT_LLM_SYNTHESIS
    teacher_model: str = DEFAULT_LLM_SYNTHESIS
    user_model: str = DEFAULT_LLM_SYNTHESIS
    judge_model: str = DEFAULT_LLM_SYNTHESIS
    llm_args: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0,
            "timeout": 300,
            "num_retries": 2,
            "max_retries": 0,
        }
    )
    structured_attempts: int = Field(default=2, ge=1, le=3)
    structured_token_limit: int = Field(default=16384, ge=1024, le=65536)
    structured_timeout_limit: int = Field(default=1800, ge=60, le=3600)
    generator_llm_args: dict[str, Any] = Field(
        default_factory=lambda: {"timeout": 300, "max_tokens": 4096}
    )
    judge_llm_args: dict[str, Any] = Field(
        default_factory=lambda: {"timeout": 600, "max_tokens": 8192}
    )
    max_steps: int = Field(default=100, ge=10)
    retrieval_config: Literal["bm25_grep"] = RETRIEVAL
    top_k: Literal[10] = 10
    # Deterministic rendering is for offline development, never publishable.
    text_mode: Literal["llm", "template"] = "llm"


class Evidence(BaseModel):
    """Exact evidence from an immutable source document."""

    document_id: str
    sha256: str
    quote: str


class RuleCatalog(BaseModel):
    """Reviewed pilot rule subset with executable values and source evidence."""

    schema_version: int = VERSION
    environment_hash: str
    documents: dict[str, Evidence]
    products: dict[str, dict[str, Any]]
    rules: dict[str, Any]
    tool_signatures: dict[str, str]
    exclusions: list[str] = Field(default_factory=list)


class ScenarioSkeleton(BaseModel):
    """Separate user-visible inputs, business solution, and grouping identity."""

    family: Literal["selection", "cashback", "credit_limit", "ordering"]
    seed: int
    group_id: str
    facts: dict[str, Any]
    private: dict[str, Any]
    difficulty: str
    evidence_ids: list[str]


class GenerationRecord(BaseModel):
    """Checkpoint for one candidate; admission requires all validation stages."""

    task: Task
    skeleton: ScenarioSkeleton
    slot: int
    attempt: int
    text_mode: str
    text_checked: bool = False
    static_passed: bool = False
    accepted: bool = False
    checks: dict[str, Any] = Field(default_factory=dict)
    trials: dict[str, dict[str, Any]] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
