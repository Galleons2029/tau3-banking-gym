"""Strict, extensible contracts for facts, storage, operations and scenarios."""

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau3.worldgen.v2.expressions import evaluate, parse, references


class StrictModel(BaseModel):
    """Reject unknown fields instead of silently dropping business semantics."""

    model_config = ConfigDict(extra="forbid")


class ScalarSpec(StrictModel):
    """Typed scalar with explicit bounds and optional enumeration."""

    type: Literal["string", "decimal", "integer", "boolean", "date"]
    unit: str = ""
    minimum: str | None = None
    maximum: str | None = None
    choices: list[str] = Field(default_factory=list)
    scale: int | None = Field(default=None, ge=0, le=12)

    def coerce(self, value):
        """Validate and return a runtime value without lossy coercions."""
        if self.type in {"decimal", "integer"}:
            if isinstance(value, bool):
                raise ValueError("Boolean is not a numeric quantity")
            number = Decimal(str(value))
            if not number.is_finite():
                raise ValueError("Non-finite number")
            if len(number.as_tuple().digits) > 28 or abs(number.adjusted()) > 28:
                raise ValueError("Numeric magnitude exceeds runtime precision")
            if self.scale is not None and number != number.quantize(
                Decimal(1).scaleb(-self.scale)
            ):
                raise ValueError("Too many decimal places")
            if self.type == "integer" and number != number.to_integral_value():
                raise ValueError("Fractional integer")
            if self.minimum is not None and number < Decimal(self.minimum):
                raise ValueError("Below lower bound")
            if self.maximum is not None and number > Decimal(self.maximum):
                raise ValueError("Above upper bound")
            return number
        if self.type == "boolean":
            if type(value) is not bool:
                raise ValueError("Expected boolean")
            return value
        if not isinstance(value, str):
            raise ValueError("Expected string")
        if self.type == "date":
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must use YYYY-MM-DD")
        if self.choices and value not in self.choices:
            raise ValueError(f"Not in enumeration: {value}")
        return value


class FactSpec(StrictModel):
    """A fact, including the complete inputs to a derived fact."""

    id: str
    label: str
    scalar: ScalarSpec
    value: Any = None
    expression: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    visible: bool = True


class ProductSpec(StrictModel):
    """One named offering; names are scoped by category, identifiers are global."""

    id: str
    name: str
    facts: list[FactSpec]

    def values(self, clock: str) -> dict:
        """Resolve the dependency graph, failing on cycles and missing facts."""
        result, pending = {}, {f.id: f for f in self.facts}
        if len(pending) != len(self.facts):
            raise ValueError("Duplicate fact id")
        while pending:
            progressed = False
            for key, fact in list(pending.items()):
                if fact.expression:
                    if references(fact.expression, "facts") != set(fact.depends_on):
                        raise ValueError(f"Incomplete dependencies: {key}")
                    if not set(fact.depends_on) <= result.keys():
                        continue
                    value = evaluate(fact.expression, {"facts": result, "clock": clock})
                else:
                    if fact.depends_on:
                        raise ValueError("Literal fact has dependencies")
                    value = fact.value
                result[key] = fact.scalar.coerce(value)
                del pending[key]
                progressed = True
            if not progressed:
                raise ValueError(
                    f"Cyclic or missing fact dependencies: {sorted(pending)}"
                )
        return result


class ForeignKey(StrictModel):
    """A field referencing a row id in another table."""

    field: str
    table: str


class EntitySpec(StrictModel):
    """Extensible database table; fields are strictly validated."""

    id: str
    fields: dict[str, ScalarSpec]
    unique: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)


class PolicySpec(StrictModel):
    """A documented rule with parameterized text and an executable predicate."""

    id: str
    statement: str
    predicate: str
    enforcement: Literal["runtime", "trajectory"] = "runtime"


class EffectSpec(StrictModel):
    """Atomic create/update effect; every value is a checked expression."""

    table: str
    mode: Literal["create", "update"]
    key: str
    values: dict[str, str]


class QuerySpec(StrictModel):
    """Customer-scoped, paginated read with explicit public field projection."""

    table: str
    fields: list[str] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    page_size: int = Field(default=3, ge=1, le=100)


class OperationSpec(StrictModel):
    """An executable capability, distinct from its discoverable public alias."""

    id: str
    description: str
    actor: Literal["assistant", "user"] = "assistant"
    parameters: dict[str, ScalarSpec]
    record_table: str | None = None
    rules: list[str] = Field(default_factory=list)
    kind: Literal["write", "read"] = "write"
    query: QuerySpec | None = None
    effects: list[EffectSpec] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_access_kind(self):
        """Reads have no write effects; existing operations remain writes."""
        if self.kind == "write" and (not self.effects or self.query is not None):
            raise ValueError("Write operations require effects and cannot have a query")
        if self.kind == "read" and (
            self.query is None
            or self.effects
            or self.postconditions
            or self.rules
            or self.actor != "assistant"
        ):
            raise ValueError("Read operations require a query and no write effects")
        return self


class StepSpec(StrictModel):
    """A workflow step uses arguments explicit in the scenario or prior state."""

    operation: str
    arguments: dict[str, Any]


class GoalSpec(StrictModel):
    """Independent state assertions, not a hash copied from reference replay."""

    table: str
    row_id: str
    fields: dict[str, Any]


class ScenarioSpec(StrictModel):
    """A typed scenario template, including independent outcomes and evidence."""

    id: str
    kind: Literal["selection", "service", "ordering", "denial", "grounding"]
    user_id: str
    request: str
    customer: dict[str, str] = Field(default_factory=dict)
    initial_rows: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    steps: list[StepSpec] = Field(default_factory=list)
    goals: list[GoalSpec] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    required_rules: list[str] = Field(default_factory=list)
    selection: str | None = None
    expected_product: str | None = None
    communication: list[str] = Field(default_factory=list)
    # Expected failing steps have a named rule, never an arbitrary tool error.
    expected_denial: str | None = None
    structural_family: str
    user_behavior: list[str] = Field(default_factory=list)


class CategorySpec(StrictModel):
    """Portable category package. No fixed list of category kinds or tables."""

    id: str
    version: str = "1"
    description: str
    dependencies: list[str] = Field(default_factory=list)
    validation_capabilities: list[str] = Field(
        default_factory=lambda: [
            "public_evidence",
            "state_projection",
            "decimal_arithmetic",
            "calendar_dates",
            "actor_permissions",
        ]
    )
    products: list[ProductSpec] = Field(min_length=1)
    entities: list[EntitySpec] = Field(min_length=1)
    policies: list[PolicySpec] = Field(default_factory=list)
    operations: list[OperationSpec] = Field(min_length=1)
    scenarios: list[ScenarioSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self):
        """Reject dangling references and unsupported expressions at admission."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError("Category id must be lowercase snake_case")
        for group in [
            self.products,
            self.entities,
            self.policies,
            self.operations,
            self.scenarios,
        ]:
            identifiers = [x.id for x in group]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("Duplicate identifier")
            if any(not re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in identifiers):
                raise ValueError("Identifiers must be lowercase snake_case")
        tables = {e.id: e for e in self.entities}
        rules = {p.id for p in self.policies}
        operations = {o.id: o for o in self.operations}
        for entity in self.entities:
            if not {"user_id", "product_id"} <= entity.fields.keys():
                raise ValueError("Business tables require user_id and product_id")
            if not set(entity.unique) <= entity.fields.keys():
                raise ValueError("Unknown unique field")
            for fk in entity.foreign_keys:
                if fk.field not in entity.fields or fk.table not in tables:
                    raise ValueError("Unknown foreign key")
            for invariant in entity.invariants:
                parse(invariant)
        for policy in self.policies:
            parse(policy.predicate)
        for op in self.operations:
            for condition in op.postconditions:
                parse(condition)
            if not {"user_id", "product_id"} <= op.parameters.keys():
                raise ValueError("Operations require user_id and product_id")
            if not set(op.rules) <= rules:
                raise ValueError("Unknown operation rule")
            if op.record_table and op.record_table not in tables:
                raise ValueError("Unknown record table")
            if op.query:
                query = op.query
                if query.table not in tables:
                    raise ValueError("Unknown query table")
                if not set(query.fields) <= tables[query.table].fields.keys():
                    raise ValueError("Unknown query projection")
                if not set(query.filters) <= tables[query.table].fields.keys():
                    raise ValueError("Unknown query filter")
                if not set(query.filters.values()) <= op.parameters.keys():
                    raise ValueError("Query filter requires a declared argument")
                if (
                    "offset" not in op.parameters
                    or op.parameters["offset"].type != "integer"
                ):
                    raise ValueError("Queries require an integer offset")
            for effect in op.effects:
                if effect.table not in tables:
                    raise ValueError("Unknown effect table")
                if not effect.values.keys() <= tables[effect.table].fields.keys():
                    raise ValueError("Unknown effect field")
                parse(effect.key)
                for expression in effect.values.values():
                    parse(expression)
        for scenario in self.scenarios:
            if not scenario.customer.keys() <= {"name", "email", "joined_on"}:
                raise ValueError("Unknown customer field")
            if len({(g.table, g.row_id) for g in scenario.goals}) != len(
                scenario.goals
            ):
                raise ValueError("Duplicate goal row")
            if scenario.expected_denial and scenario.expected_denial not in rules:
                raise ValueError("Unknown denial policy")
            if not scenario.communication and scenario.kind in {"denial", "grounding"}:
                raise ValueError("Non-mutating tasks require communication assertions")
            if not scenario.goals and scenario.kind not in {"denial", "grounding"}:
                raise ValueError("Missing independent state goal")
            if scenario.selection and not scenario.expected_product:
                raise ValueError("Selection requires independently declared answer")
            if scenario.selection:
                parse(scenario.selection)
            if not set(scenario.required_rules) <= rules:
                raise ValueError("Unknown evidence rule")
            for step in scenario.steps:
                if step.operation not in operations:
                    raise ValueError("Unknown workflow operation")
                operation = operations[step.operation]
                if step.arguments.keys() != operation.parameters.keys():
                    raise ValueError("Workflow argument schema mismatch")
                for key, value in step.arguments.items():
                    operation.parameters[key].coerce(value)
                if step.arguments.get("user_id") != scenario.user_id:
                    raise ValueError("Workflow belongs to another customer")
            for goal in scenario.goals:
                if (
                    goal.table not in tables
                    or not goal.fields.keys() <= tables[goal.table].fields.keys()
                ):
                    raise ValueError("Unknown goal table or field")
                for key, value in goal.fields.items():
                    tables[goal.table].fields[key].coerce(value)
        return self


class WorldSpec(StrictModel):
    """Immutable-by-convention generation context, independent of globals."""

    schema_version: Literal[2] = 2
    seed: int = 42
    clock: str = "2025-11-14"
    purpose: Literal[
        "banking_expansion", "paper_reproduction", "training_curriculum"
    ] = "banking_expansion"
    categories: list[CategorySpec] = Field(min_length=1)
    public_record_snapshot: bool = True
    public_document_catalog: bool = True
    split_policy: Literal["structural", "all_train"] = "structural"

    @model_validator(mode="after")
    def validate_world(self):
        """Validate composition, global namespaces and facts at the world date."""
        date.fromisoformat(self.clock)
        if date.fromisoformat(self.clock).isoformat() != self.clock:
            raise ValueError("World clock must use YYYY-MM-DD")
        ids = [c.id for c in self.categories]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate category")
        descriptions = [c.description.casefold().strip() for c in self.categories]
        if len(set(descriptions)) != len(descriptions):
            raise ValueError(
                "Category descriptions must expose distinct user-visible scopes"
            )
        names = [p.name.casefold().strip() for c in self.categories for p in c.products]
        if len(set(names)) != len(names):
            raise ValueError("Product names must be unambiguous across the world")
        for collection in [
            "products",
            "entities",
            "operations",
            "policies",
            "scenarios",
        ]:
            names = [x.id for c in self.categories for x in getattr(c, collection)]
            if len(set(names)) != len(names):
                raise ValueError(f"World namespace collision: {collection}")
        for category in self.categories:
            if not set(category.dependencies) <= set(ids) - {category.id}:
                raise ValueError("Missing category dependency")
            for product in category.products:
                product.values(self.clock)
        remaining = {c.id: set(c.dependencies) for c in self.categories}
        resolved = set()
        while remaining:
            ready = {key for key, deps in remaining.items() if deps <= resolved}
            if not ready:
                raise ValueError("Cyclic category dependencies")
            resolved |= ready
            remaining = {
                key: deps for key, deps in remaining.items() if key not in ready
            }
        return self


def load_categories(paths: list[Path]) -> list[CategorySpec]:
    """Read category packages from YAML or JSON, without importing code."""
    import yaml

    return [CategorySpec.model_validate(yaml.safe_load(p.read_text())) for p in paths]
