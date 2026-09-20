"""Versioned worldgen artifacts; published worlds keep the benchmark schemas.

A world bundle mirrors the directory shape of an installed domain, so the
synthesized knowledge base, database and tasks load through the same
`KnowledgeBase` / `TransactionalDB` / `Task` models the benchmark already uses.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = 1
DOMAIN = "banking_synth"
RETRIEVAL = "bm25_grep"

# Self-hosted OpenAI-compatible gateway. The endpoint does not check credentials,
# but LiteLLM's openai provider requires some key, hence the placeholder.
WORLDGEN_API_BASE = "http://10.39.62.231:9091/v1"
WORLDGEN_API_KEY = "not-required"
WORLDGEN_MODEL = "openai/gemini-3.5-flash"

# Document archetypes and their share of the corpus (docs/tau-banking-synth.md C1).
ARCHETYPES: dict[str, dict[str, float]] = {
    "product_overview": {"share": 0.10, "tokens": 420},
    "faq": {"share": 0.20, "tokens": 240},
    "internal_protocol": {"share": 0.12, "tokens": 520},
    "tool_doc": {"share": 0.09, "tokens": 330},
    "program_terms": {"share": 0.09, "tokens": 260},
    "howto_short": {"share": 0.25, "tokens": 150},
    "eligibility_matrix": {"share": 0.08, "tokens": 300},
    "promo_notice": {"share": 0.07, "tokens": 165},
}


class WorldConfig(BaseModel):
    """Bounded generation settings, persisted with every world bundle."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    # Every model-calling stage fans out to this width. The generator and the
    # renderer are one independent call per product and per document, so the
    # ceiling is what the endpoint will take rather than anything in the design.
    concurrency: int = Field(default=8, ge=1, le=128)
    # Re-rolled on every publication so tool suffixes cannot be memorized.
    suffix_salt: str = "tau3-world-v1"

    generator_model: str = WORLDGEN_MODEL
    teacher_model: str = WORLDGEN_MODEL
    user_model: str = WORLDGEN_MODEL
    judge_model: str = WORLDGEN_MODEL

    llm_args: dict[str, Any] = Field(
        default_factory=lambda: {
            "api_base": WORLDGEN_API_BASE,
            "api_key": WORLDGEN_API_KEY,
            "temperature": 0,
            "timeout": 600,
            "num_retries": 2,
            "max_retries": 0,
        }
    )
    # The configured model spends most of its completion budget on reasoning
    # tokens before emitting any JSON, so these are an order of magnitude above
    # what a non-reasoning model needs.
    generator_llm_args: dict[str, Any] = Field(
        default_factory=lambda: {"timeout": 600, "max_tokens": 16384}
    )
    judge_llm_args: dict[str, Any] = Field(
        default_factory=lambda: {"timeout": 900, "max_tokens": 16384}
    )
    structured_attempts: int = Field(default=3, ge=1, le=5)
    structured_token_limit: int = Field(default=32768, ge=1024, le=131072)
    structured_timeout_limit: int = Field(default=1800, ge=60, le=3600)

    retrieval_config: Literal["bm25_grep"] = RETRIEVAL
    top_k: Literal[10] = 10
    max_steps: int = Field(default=100, ge=10)
    # Deterministic rendering is for offline development, never publishable.
    text_mode: Literal["llm", "template"] = "llm"

    # Slice size. The mini slice proves the loop; the full run keeps the code and
    # changes only these numbers.
    num_categories: int = Field(default=4, ge=1)
    num_topics: int = Field(default=10, ge=1)
    num_documents: int = Field(default=80, ge=1)
    num_tasks: int = Field(default=10, ge=1)
    num_noise_users: int = Field(default=40, ge=0)
    # Attributes per product. Documents are shaped from the archetype mix, so a
    # product with too few facts forces the same fact into several documents and
    # drives the corpus more redundant than intended.
    variables_per_product: int = Field(default=16, ge=4, le=40)
    # Documents rotate between these so a retriever cannot separate them by
    # author. Only models the configured endpoint actually serves belong here.
    render_models: list[str] = Field(
        default_factory=lambda: ["openai/gemini-3.5-flash", "openai/GLM-5.3-Flash"]
    )
    # Fragmentation knob: how many task-critical variables one document may reveal.
    leak_divisor: int = Field(default=3, ge=1, le=10)


class Distribution(BaseModel):
    """Summary of a measured numeric distribution."""

    model_config = ConfigDict(extra="forbid")

    count: int
    total: float
    mean: float
    median: float
    minimum: float
    maximum: float
    p10: float
    p25: float
    p75: float
    p90: float


class CorpusTargets(BaseModel):
    """Baselines measured from an installed corpus, not copied from the paper.

    Every acceptance gate reads these. Paper values are carried in `reference`
    for provenance only.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = VERSION
    source: str
    environment_hash: str
    tokenizer: str

    documents: int
    document_tokens: Distribution
    document_words: Distribution
    topics: int
    documents_per_topic: Distribution
    # The corpus filename does not encode categories unambiguously: the count
    # depends on the split rule, so the rule name travels with the number and
    # the count is never used as a gate.
    category_rule: str
    categories: int

    tasks: int
    gold_documents: Distribution
    actions: Distribution
    distinct_gold_documents: int
    gold_document_reuse: float

    permanent_tools: list[str]
    discoverable_tools: list[str]
    suffixed_tools: list[str]

    reference: dict[str, Any] = Field(default_factory=dict)


class Variable(BaseModel):
    """One dimension of the product space, with the value the solver assigned."""

    model_config = ConfigDict(extra="forbid")

    id: str
    feature_id: str
    name: str
    type: Literal[
        "currency",
        "percent",
        "int_days",
        "int_count",
        "enum",
        "bool",
        "duration",
        "string",
    ]
    value: Any
    unit: str | None = None
    # Candidate values for an enum variable; empty for every other type.
    choices: list[str] = Field(default_factory=list)
    visibility: Literal["customer_facing", "internal_only"] = "customer_facing"
    volatility: Literal["static", "promotional"] = "static"
    window: dict[str, str] | None = None
    # Computed from other variables at the simulated date. Derived values prove
    # uniqueness to the linters but are never allocated to a document: stating
    # one would remove the need to work it out, which is the point of the task.
    derived: bool = False


class Feature(BaseModel):
    """A product or process; features carry no references to each other.

    Cross-feature interaction is declared separately and only when a task needs
    it, which is what keeps unintended collisions out of the world.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category_id: str
    entity_name: str
    entity_kind: Literal["account", "card", "program", "protocol"]
    doc_budget: int = Field(default=1, ge=1)
    variable_ids: list[str] = Field(default_factory=list)


class Category(BaseModel):
    """A knowledge category; `topic_budget` is how many features it holds."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    kind: Literal["product", "protocol", "program", "support"]
    audience: Literal["customer", "internal", "both"] = "both"
    topic_budget: int = Field(default=1, ge=1)


class Rule(BaseModel):
    """A policy statement that is knowledge but not a number.

    Eligibility and ordering constraints are what tasks actually turn on, and a
    task's required knowledge is incomplete if it only counts variables.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    feature_id: str | None = None
    enforced_by_tool: str | None = None


class Interaction(BaseModel):
    """A cross-feature dependency, valid only while a task requires it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    required_by: list[str] = Field(min_length=1)
    predicate: str
    surfaced_in_docs: list[str] = Field(default_factory=list)


class WorldSchema(BaseModel):
    """The structured layer every document is rendered from."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = VERSION
    categories: list[Category] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)

    def variable(self, variable_id: str) -> Variable:
        for variable in self.variables:
            if variable.id == variable_id:
                return variable
        raise KeyError(variable_id)

    def feature_variables(self, feature_id: str) -> list[Variable]:
        return [v for v in self.variables if v.feature_id == feature_id]


class DocPlan(BaseModel):
    """What one document is allowed to reveal, decided before any text exists."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    archetype: str
    feature_id: str | None = None
    variable_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    crossrefs: list[str] = Field(default_factory=list)
    target_tokens: int = Field(default=250, ge=1)

    @property
    def reveals(self) -> set[str]:
        """Knowledge elements a reader learns from this document."""
        return set(self.variable_ids) | set(self.tool_ids) | set(self.rule_ids)


class TaskSpec(BaseModel):
    """The knowledge a task turns on, kept next to the task it explains."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    archetype: Literal[
        "selection",
        "protocol",
        "ordering",
        "over_trust",
        "clarification",
        "grounding",
        "denial",
        "mid_conversation_change",
    ]
    requires_variables: list[str] = Field(default_factory=list)
    requires_tools: list[str] = Field(default_factory=list)
    requires_rules: list[str] = Field(default_factory=list)
    # Products the customer's stated constraints must select between, and the
    # single one that satisfies all of them.
    # Everything the requirements range over; the answer must be unique across
    # all of it, because the customer names no shortlist.
    candidate_features: list[str] = Field(default_factory=list)
    # The products deliberately placed one constraint away from qualifying.
    # Everything else in `candidate_features` is clearly out of contention.
    near_misses: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unique_answer: str | None = None
    # The catalogue tier the customer says they are shopping in. It is what
    # bounds `candidate_features` to one segment, so the answer is unique within
    # what the customer asked for rather than across the whole catalogue.
    scope: str = ""
    # Free-text argument values the DB evaluator compares verbatim. The customer
    # is the only source for these, so the user instructions have to state them
    # literally or the task is unwinnable however well the agent reasons.
    verbatim_arguments: list[str] = Field(default_factory=list)

    @property
    def requires(self) -> set[str]:
        """Req(t): the knowledge elements a solver cannot do without."""
        return (
            set(self.requires_variables)
            | set(self.requires_tools)
            | set(self.requires_rules)
        )


class WorldPlan(BaseModel):
    """Document allocation and task specifications for one world."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = VERSION
    documents: list[DocPlan] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)


class WorldManifest(BaseModel):
    """Publication record for one world bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = VERSION
    domain: Literal["banking_synth"] = DOMAIN
    status: Literal["draft", "published"] = "draft"
    name: str
    seed: int
    suffix_salt: str
    # alias -> the official tool the alias dispatches to.
    alias_map: dict[str, str] = Field(default_factory=dict)
    environment_hash: str
    config_hash: str
    targets_hash: str = ""
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    # Named task splits; "base" is implicit and always the whole task set.
    task_splits: dict[str, list[str]] = Field(default_factory=dict)
    stages: dict[str, str] = Field(default_factory=dict)
