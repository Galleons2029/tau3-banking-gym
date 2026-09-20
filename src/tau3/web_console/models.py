"""Public API models for the local web console."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ArtifactKind = Literal[
    "run",
    "simulation",
    "bundle",
    "task",
    "trajectory",
    "document",
    "seed_table",
    "seed_record",
]


class ArtifactRef(BaseModel):
    """Logical reference to an indexed artifact; never a client-provided path."""

    kind: ArtifactKind
    id: str
    bundle_id: str | None = None
    package_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    trial: str | int | None = None


class ReasoningBlock(BaseModel):
    """A normalized reasoning block extracted from a provider response."""

    participant: Literal["agent", "user"]
    text: str | None = None
    source: str | None = None
    available: bool = False


class PatchOperation(BaseModel):
    """One RFC 6902-style mutation used by a calibration change set."""

    op: Literal["add", "replace", "remove"]
    path: str
    value: Any = None


class ChangeTarget(BaseModel):
    """Patch target and optimistic-concurrency base hash."""

    artifact: ArtifactRef
    base_sha256: str
    operations: list[PatchOperation] = Field(min_length=1)


class ChangeSetCreate(BaseModel):
    """Payload for creating a reviewable calibration change set."""

    rationale: str = Field(min_length=1)
    author: str | None = None
    targets: list[ChangeTarget] = Field(min_length=1)


class ChangeSet(BaseModel):
    """Persisted calibration change set."""

    id: str
    rationale: str
    author: str | None = None
    status: Literal["draft", "validated", "applied", "rejected"] = "draft"
    created_at: str
    updated_at: str
    targets: list[ChangeTarget]
    validation: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class HumanAttribution(BaseModel):
    """Human adjudication stored separately from immutable model reviews."""

    id: str | None = None
    artifact: ArtifactRef
    source: Literal["agent", "user", "system", "unknown"]
    turn_idx: int | None = None
    error_tags: list[str] = Field(default_factory=list)
    severity: str | None = None
    reasoning: str = Field(min_length=1)
    correct_behavior: str | None = None
    verdict: Literal["confirmed", "overridden", "additional"] = "additional"
    author: str | None = None
    created_at: str | None = None


class JobCreate(BaseModel):
    """A strictly allowlisted background operation."""

    type: Literal[
        "check_data",
        "bundle_validate",
        "bundle_export",
        "results_review",
        "trajectory_review",
    ]
    bundle_id: str | None = None
    run_id: str | None = None
    artifact: ArtifactRef | None = None
    offline: bool = False
    review_mode: Literal["full", "user"] = "full"
    review_model: str | None = None
    max_concurrency: int = Field(default=4, ge=1, le=32)
    task_ids: list[str] | None = None


class JobRecord(BaseModel):
    """Persisted status for an allowlisted subprocess."""

    id: str
    type: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]
    created_at: str
    updated_at: str
    pid: int | None = None
    return_code: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    target_key: str | None = None
    log_file: str
    output_file: str | None = None
    error: str | None = None
