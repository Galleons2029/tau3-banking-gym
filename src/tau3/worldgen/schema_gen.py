"""Stage A1: ask a model for the shape of the product space, never for its values.

The generator sees variable names, types, units and plausible ranges only. Values
are assigned later by the solver, because a model asked for both invents numbers
that no downstream constraint can hold to, and the documents then contradict the
database. Every variable the scenario families turn on is required rather than
suggested: the rest is realistic filler that keeps the catalogue from looking
like a test fixture.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau3.worldgen.models import Variable, WorldConfig
from tau3.worldgen.slice import BANK, REQUIRED_VARIABLES, ProductSpec

# A generator may not produce free-form strings: a string attribute has no
# assignable value, and a document that states one would be inventing a fact. A
# categorical attribute is an enum with explicit candidates instead.
VARIABLE_TYPES = ("currency", "percent", "int_days", "int_count", "bool", "enum")
SNAKE = re.compile(r"[^a-z0-9]+")

# Token usage across this process's skeleton calls, retries included.
_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def usage_snapshot() -> dict:
    """What the skeleton calls have consumed so far."""
    return dict(_USAGE)


def reset_usage() -> None:
    """Start a fresh accounting window."""
    for key in _USAGE:
        _USAGE[key] = 0


class GeneratedVariable(BaseModel):
    """The response contract: what a generator is allowed to propose.

    Free-form strings are absent by construction. A string attribute has no
    assignable value, so a document stating one would be inventing a fact; a
    categorical attribute is an enum with explicit candidates instead.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["currency", "percent", "int_days", "int_count", "bool", "enum"]
    unit: str
    plausible_min: float
    plausible_max: float
    enum_values: list[str]
    visibility: Literal["customer_facing", "internal_only"]


class VariableSkeleton(BaseModel):
    """One attribute of a product, without a value.

    Wider than the response contract: the required class-name variables are
    strings injected by this module, not proposed by a generator.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal[
        "currency", "percent", "int_days", "int_count", "bool", "enum", "string"
    ]
    unit: str
    plausible_min: float
    plausible_max: float
    enum_values: list[str] = Field(default_factory=list)
    visibility: Literal["customer_facing", "internal_only"] = "customer_facing"


class FeatureSkeleton(BaseModel):
    """The attribute set of one product, as returned by a generator."""

    model_config = ConfigDict(extra="forbid")

    variables: list[GeneratedVariable] = Field(min_length=1)


SYSTEM = (
    f"You design the structured product catalogue for {BANK}, a fictional US "
    "retail bank. You output the SHAPE of a product's attributes: names, types, "
    "units and plausible ranges. You never output a concrete value for any "
    "attribute, because values are assigned later by a constraint solver.\n\n"
    "Rules:\n"
    "- Every name is snake_case.\n"
    "- plausible_min and plausible_max bound a realistic range for a US retail "
    "bank; they are a range, not a value.\n"
    "- Use enum for a categorical attribute and list its candidates in "
    "enum_values; every other type has an empty enum_values.\n"
    "- Never use a free-form string attribute: a value nobody can assign is a "
    "fact a document would have to invent.\n"
    "- Mark an attribute internal_only when a customer would not see it in "
    "marketing material.\n"
    "- Do not invent attributes that reference another product."
)


def snake_case(text: str) -> str:
    """Normalize a generated name; models return prose case regardless of asking."""
    return SNAKE.sub("_", text.strip().lower()).strip("_")


def is_synonym(candidate: str, required: str) -> bool:
    """Whether one name is a wordier restatement of another.

    Token containment either way: `monthly_maintenance_fee` restates
    `monthly_fee`, and `apy` does not restate `base_apy`.
    """
    left, right = set(candidate.split("_")), set(required.split("_"))
    return bool(left) and bool(right) and (left <= right or right <= left)


def prompt_for(spec: ProductSpec, category_name: str, extra: int) -> dict:
    """The Stage-1 request for one product."""
    required = REQUIRED_VARIABLES[spec.kind]
    return {
        "category": category_name,
        "product": spec.entity_name,
        "product_kind": spec.kind,
        "required_variables": [
            {"name": name, "type": kind, "unit": unit or ""}
            for name, (kind, unit) in required.items()
        ],
        "instruction": (
            f"Return the {len(required)} required variables exactly as named and "
            f"typed above, plus {extra} further attributes a real {spec.kind} "
            "product would document. Give every variable a plausible range, "
            "never a value."
        ),
    }


def merge_required(spec: ProductSpec, generated: list[GeneratedVariable]):
    """Keep the required variables authoritative, and the generated ones distinct.

    A generated variable that collides with a required name is dropped rather
    than merged: the required type and unit are what the solver and the scenario
    families rely on.
    """
    required = REQUIRED_VARIABLES[spec.kind]
    skeletons: list[VariableSkeleton] = []
    for name, (kind, unit) in required.items():
        match = next((v for v in generated if snake_case(v.name) == name), None)
        if match is None:
            # A generator that renames a required variable ("monthly maintenance
            # fee" for "monthly fee") would otherwise contribute a second,
            # separately valued copy of the same fact.
            match = next(
                (v for v in generated if is_synonym(snake_case(v.name), name)), None
            )
        skeletons.append(
            VariableSkeleton(
                name=name,
                type=kind,
                unit=unit or "",
                # The range is the generator's contribution; everything else is
                # fixed, so a family constraint always addresses the variable it
                # thinks it does.
                plausible_min=match.plausible_min if match else 0.0,
                plausible_max=match.plausible_max if match else 0.0,
                enum_values=[],
                visibility="customer_facing",
            )
        )
    seen = set(required)
    for variable in generated:
        name = snake_case(variable.name)
        if not name or name in seen:
            continue
        if any(is_synonym(name, required_name) for required_name in required):
            continue
        seen.add(name)
        skeletons.append(VariableSkeleton(**{**variable.model_dump(), "name": name}))
    return skeletons


def to_variables(
    spec: ProductSpec, skeletons: list[VariableSkeleton]
) -> list[Variable]:
    """Carry skeletons into schema variables, with values still unassigned."""
    variables = []
    for skeleton in skeletons:
        variables.append(
            Variable(
                id=f"var_{spec.feature_id.removeprefix('feat_')}_{skeleton.name}",
                feature_id=spec.feature_id,
                name=skeleton.name,
                type=skeleton.type,
                unit=skeleton.unit or None,
                choices=list(skeleton.enum_values),
                value=None,
                visibility=skeleton.visibility,
            )
        )
    return variables


def offline_skeletons(spec: ProductSpec) -> list[VariableSkeleton]:
    """Deterministic development stand-in; never publishable."""
    required = REQUIRED_VARIABLES[spec.kind]
    return [
        VariableSkeleton(
            name=name,
            type=kind,
            unit=unit or "",
            plausible_min=0.0,
            plausible_max=100.0 if kind == "percent" else 5000.0,
            enum_values=[],
            visibility="customer_facing",
        )
        for name, (kind, unit) in required.items()
    ]


def generate_skeleton(
    spec: ProductSpec, category_name: str, config: WorldConfig, extra: int = 4
) -> list[VariableSkeleton]:
    """Ask the generator for one product's attribute shape."""
    if config.text_mode == "template":
        return offline_skeletons(spec)

    from tau3.synthesis.llm import json_response, response_log

    records: list[dict] = []
    with response_log(records.append):
        payload, _evidence = json_response(
            config.generator_model,
            SYSTEM,
            prompt_for(spec, category_name, extra),
            config,
            "worldgen_skeleton",
            schema=FeatureSkeleton,
        )
    # Structured generation retries on a truncated response, so what a product
    # costs is the sum of its attempts rather than the one that succeeded.
    for record in records:
        usage = record.get("usage") or {}
        _USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
        _USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
        _USAGE["calls"] += 1
    generated = FeatureSkeleton.model_validate(payload).variables
    return merge_required(spec, generated)
