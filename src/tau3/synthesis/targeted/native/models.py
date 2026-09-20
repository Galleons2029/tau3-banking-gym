"""Immutable native curricula, admission evidence and external evaluation gates."""

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from tau3.data_model.tasks import Task
from tau3.synthesis.targeted.models import FailureProfile, RoundConfig
from tau3.worldgen.v2.specs import StrictModel

FAMILIES = {
    "disputes": 25,
    "debit": 20,
    "accounts": 15,
    "optimization": 15,
    "identity": 10,
    "handoff": 10,
    "boundary": 5,
}
FAMILY_LABELS = {
    "disputes": ["F1", "F2", "F3", "F3b", "F4"],
    "debit": ["F1", "F2", "F3", "F4", "F8"],
    "accounts": ["F1", "F2", "F3", "F4", "F7"],
    "optimization": ["F2", "F3", "F8"],
    "identity": ["F1", "F7", "F8"],
    "handoff": ["F3", "F3b", "F7"],
    "boundary": ["F4", "F5", "F6", "F8"],
}

V03_FAMILIES = {
    "credit_limit": 25, "transaction_disputes": 20, "replacement_closure": 15,
    "accounts_funds": 12, "debit_security": 10, "optimization": 8,
    "handoff": 5, "escalation_boundary": 5,
}
V03_LABELS = {
    "credit_limit": ["F1", "F2", "F3", "F4", "F7"],
    "transaction_disputes": FAMILY_LABELS["disputes"],
    "replacement_closure": ["F1", "F2", "F3", "F4", "F7"],
    "accounts_funds": FAMILY_LABELS["accounts"],
    "debit_security": FAMILY_LABELS["debit"],
    "optimization": FAMILY_LABELS["optimization"],
    "handoff": FAMILY_LABELS["handoff"],
    "escalation_boundary": FAMILY_LABELS["boundary"],
}


class NativeConfig(RoundConfig):
    """A fresh round; all stages share one physical-request budget and pool."""

    schema_version: Literal[2] = 2
    backend: Literal["banking_native"] = "banking_native"
    num_tasks: Literal[3000] = 3000
    small_tasks: Literal[400] = 400
    expansion_target_rows: Literal[3000] | None = None
    user_role_guard: Literal["customer_role_v1", "customer_role_v2"] | None = None
    retained_sft_manifest: str | None = None
    validation_tasks: Literal[200] = 200
    workers: int = Field(default=256, ge=1, le=256)
    llm_concurrency: Literal[256] = 256
    max_total_tokens: int = Field(default=40_000_000_000, ge=1)
    max_blind_rollouts: int = Field(default=19320, ge=1)
    reasoning_policy: Literal["strict_preserve"] = "strict_preserve"
    training_contract: str | None = None
    structured_max_tokens: int = Field(default=16384, ge=1024, le=65536)
    structured_output_mode: Literal["prompt_json"] = "prompt_json"
    structured_reasoning_effort: Literal["low", "high", "max"] = "low"
    structured_reviewer_max_tokens: int = Field(default=32768, ge=16384, le=65536)
    structured_reviewer_timeout: int = Field(default=900, ge=300, le=1800)
    curriculum: Literal["native_v2", "v03_r10"] = "native_v2"
    clean_only: bool = False
    parent_round: str | None = None
    budget_parent_round: str | None = None
    reuse_parent_pilot: bool = False
    reuse_parent_splits: list[Literal["pilot", "train", "validation"]] = Field(default_factory=list)
    # Explicitly version any authorized synthetic runtime corrections.
    runtime_revision: Literal["official", "native_cli_approval_v1", "native_card_lifecycle_v2"] = "official"

    @model_validator(mode="after")
    def v03_contract(self):
        if self.retained_sft_manifest and not (self.user_role_guard and self.expansion_target_rows and self.parent_round):
            raise ValueError("Filtered retention requires a role-guarded expansion with a parent")
        if self.expansion_target_rows is not None and self.curriculum != "v03_r10":
            raise ValueError("Pre-evaluation expansion requires the v03 curriculum")
        if self.curriculum == "v03_r10" and not self.clean_only:
            raise ValueError("The approved v03 curriculum requires a clean-only main dataset")
        if self.reuse_parent_splits and not self.parent_round:
            raise ValueError("Explicit split reuse requires a declared parent round")
        return self


class NativeProfile(FailureProfile):
    """Comparable units and explicit model roles, without inventing statistics."""

    schema_version: Literal[2] = 2
    run_roles: dict[str, str] = Field(default_factory=dict)
    inference_equivalent: Literal[True] = True
    statistics_unit: Literal["task_with_trial_details"] = "task_with_trial_details"


class NativeSlot(StrictModel):
    """One business instance; four seeds never depend on teacher outcomes."""

    index: int
    runtime_revision: Literal["official", "native_cli_approval_v1", "native_card_lifecycle_v2"] = "official"
    split: Literal["pilot", "train", "validation"]
    family: str
    difficulty: str
    origin: Literal["current", "replay", "explore"]
    seed: int
    trial_seeds: list[int]
    evidence_ids: list[str]
    mechanism: Literal["card_selection", "referral", "savings_correction"] | None = None
    v03_labels: list[str] = Field(default_factory=list)
    target_operations: list[str] = Field(default_factory=list)
    branch_id: str | None = None
    pair_id: str | None = None
    pair_side: Literal["control", "boundary"] | None = None
    protocol_challenges: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def supported(self):
        if self.family not in FAMILIES and self.family not in V03_FAMILIES:
            raise ValueError("Unregistered native family")
        if len(self.trial_seeds) != 4 or len(set(self.trial_seeds)) != 4:
            raise ValueError("Exactly four distinct frozen seeds required")
        if not self.evidence_ids:
            raise ValueError("A slot must be bound to failure/retention evidence")
        if self.mechanism and self.family != "optimization":
            raise ValueError("Unsupported mechanism for family")
        return self


class GatePolicy(StrictModel):
    """Predeclared criteria; outcomes cannot change their thresholds."""

    trials: Literal[5] = 5
    minimum_gain: float = 0.03
    confidence: float = 0.95
    bootstrap_replicates: int = 10000
    protected_max_drop: float = 0.05
    protected_task_floor: float = 0.4
    v03_checks: bool = False
    credit_limit_gain: float = 0.10
    task_051_floor: float = 0.6
    protocol_error_reduction: float = 0.30
    unknown_tool_reduction: float = 0.30
    first_call_format_gain: float = 0.10


class OperationCoverageContract(StrictModel):
    """Exact runtime operation names, with task-level minimum coverage."""

    official_tasks_hash: str
    operations: dict[str, dict]
    small_minimum: int = 4
    full_minimum: int = 30
    credit_small_minimum: int = 30
    credit_full_minimum: int = 150
    minimum_multi_object_fraction: float = 0.4
    minimum_cross_entity_fraction: float = 0.25


class NativePlan(StrictModel):
    """400 tasks prefix a frozen 3000-task training allocation."""

    schema_version: Literal[2] = 2
    backend: Literal["banking_native"] = "banking_native"
    profile_hash: str
    snapshot_hash: str
    parent_hash: str | None = None
    slots: list[NativeSlot]
    pilot: list[NativeSlot]
    validation: list[NativeSlot]
    gate: GatePolicy = Field(default_factory=GatePolicy)
    proposal: dict = Field(default_factory=dict)
    operation_coverage: OperationCoverageContract | None = None
    branch_matrix: dict = Field(default_factory=dict)
    pair_matrix: dict = Field(default_factory=dict)


class NativeCandidate(StrictModel):
    """Private compiled obligations stay separate from actual agent context."""

    schema_version: Literal[2] = 2
    task: Task
    slot: NativeSlot
    candidate: int
    facts: dict
    goals: list[dict]
    graph: dict
    group_id: str
    business_fingerprint: str
    checks: dict = Field(default_factory=dict)


class NativeProvenance(StrictModel):
    """Evidence binds the exact initialized native environment and sampling slots."""

    schema_version: Literal[2] = 2
    task_id: str
    task_hash: str
    candidate_hash: str
    plan_hash: str
    snapshot_hash: str
    initial_state_hash: str
    slot: NativeSlot
    group_id: str
    business_fingerprint: str
    reference_actions: int
    business_operations: int
    checks: dict
    status: Literal["VALID", "INVALID", "INCONCLUSIVE"]


class NativeQuality(StrictModel):
    """Reasoning rejection and per-assistant erroneous-action masking are explicit."""

    reasoning_correct: StrictBool
    grounded_in_visible_history: StrictBool
    user_compliant: StrictBool
    handoff_correct: StrictBool
    final_answer_correct: StrictBool
    no_private_leakage: StrictBool
    erroneous_assistant_turns: list[StrictInt]
    unlocalizable_error: StrictBool
    explanation: str

    @property
    def passed(self):
        """Never infer approval from a missing or truthy string-valued judgment."""
        return all(
            getattr(self, field)
            for field in (
                "reasoning_correct", "grounded_in_visible_history", "user_compliant",
                "handoff_correct", "final_answer_correct", "no_private_leakage",
            )
        ) and not self.unlocalizable_error
