"""Stage A2: assign values so the tasks are well posed and the traps bite.

Values are constructed rather than searched. For a plain selection family the
answer satisfies every stated constraint and each rejected candidate fails
exactly one -- a candidate failing two can be discarded from a single document,
which removes the comparison the task is testing.

Trap families work the other way round: every candidate satisfies the surface
constraints, and the only thing separating them is a value that has to be worked
out. A total rate that the advertised promotion does not close, or a rate that
depends on today's date, is a discriminator an agent cannot reach by skimming
one headline. Construction is deterministic under the configured seed and cannot
fail to find a solution, which a rejection sampler over eight products and three
families can.
"""

import random
from dataclasses import dataclass, field

from tau3.worldgen.linters import CONSTRAINT
from tau3.worldgen.models import Variable, WorldSchema
from tau3.worldgen.slice import FamilySpec, ProductSpec

# How far past a bound a value is pushed, per variable type.
STEPS = {"currency": 1.0, "percent": 0.25, "int_days": 1.0, "int_count": 1.0}
# Comfortable margin on the satisfying side, so a value never sits on a bound by
# accident and read back as a near miss.
SLACK = {"currency": 250.0, "percent": 0.5, "int_days": 1.0, "int_count": 1.0}

SIM_DATE = "11/14/2025"
EXPIRED_WINDOW = {"start": "08/01/2025", "end": "10/31/2025"}
LIVE_WINDOW = {"start": "11/01/2025", "end": "12/31/2025"}


@dataclass
class SolvedFamily:
    """A family after assignment, with the constraints a task will state."""

    family_id: str
    kind: str
    candidates: list[str]
    answer: str
    constraints: list[str]
    trap: str
    near_misses: list[str] = field(default_factory=list)
    # The tier the customer states; families of one kind are disjoint across it.
    segment: str = ""


@dataclass
class Assignment:
    """What the solver did, so the report can show the traps actually landed."""

    seed: int
    families: list[SolvedFamily]
    relaxations: list[str]

    @property
    def traps(self) -> dict[str, str]:
        return {family.family_id: family.trap for family in self.families}


def _round(value: float, kind: str) -> float:
    """Values are stated in documents, so they must read like real figures."""
    if kind in {"int_days", "int_count", "currency"}:
        return float(int(round(value)))
    return round(value, 2)


def satisfying_value(operator: str, bound: float, kind: str) -> float:
    """A value comfortably on the satisfying side of one constraint."""
    slack = SLACK.get(kind, 1.0)
    step = STEPS.get(kind, 1.0)
    if operator == "==":
        return _round(bound, kind)
    if operator == "!=":
        return _round(bound + step, kind)
    if operator in {">=", ">"}:
        return _round(bound + slack, kind)
    return _round(max(bound - slack, 0.0), kind)


def violating_value(operator: str, bound: float, kind: str) -> float:
    """A value just past one constraint, so the candidate stays a near miss."""
    step = STEPS.get(kind, 1.0)
    if operator == "==":
        return _round(bound + step, kind)
    if operator == "!=":
        return _round(bound, kind)
    if operator in {">=", ">"}:
        return _round(max(bound - step, 0.0), kind)
    return _round(bound + step, kind)


def parse_constraints(constraints: list[str]):
    """Split constraint strings into (variable name, operator, bound)."""
    parsed = []
    for text in constraints:
        match = CONSTRAINT.match(text)
        if not match:
            raise ValueError(f"Unparsable constraint: {text!r}")
        parsed.append(
            (match.group("name"), match.group("op"), float(match.group("value")))
        )
    return parsed


def _variable(schema: WorldSchema, feature_id: str, name: str) -> Variable:
    for variable in schema.variables:
        if variable.feature_id == feature_id and variable.name == name:
            return variable
    raise KeyError(f"{feature_id} has no variable named {name}")


def _add_derived(
    schema: WorldSchema, feature_id: str, name: str, value: float
) -> Variable:
    """Record a computed value for the linters, never for a document."""
    variable = Variable(
        id=f"var_{feature_id.removeprefix('feat_')}_{name}",
        feature_id=feature_id,
        name=name,
        type="percent",
        unit="percent",
        value=_round(value, "percent"),
        visibility="internal_only",
        derived=True,
    )
    schema.variables.append(variable)
    return variable


def seed_baseline(
    schema: WorldSchema, ranges: dict[str, tuple[float, float]], seed: int
) -> None:
    """Fill every value from its plausible range before families constrain them.

    Each variable draws from its own stream, keyed on its identifier. Drawing
    sequentially from one shared stream would make a value depend on how many
    variables happened to precede it, so adding or removing a single variable --
    a derived one, say -- would silently move every value after it and
    invalidate the whole rendered corpus. Incremental refresh depends on the
    opposite being true.
    """
    for variable in sorted(schema.variables, key=lambda v: v.id):
        if variable.type == "string":
            continue
        rng = random.Random(f"{seed}|{variable.id}")
        if variable.type == "bool":
            variable.value = rng.random() < 0.5
            continue
        if variable.type == "enum":
            if not variable.choices:
                raise ValueError(f"Enum variable has no candidates: {variable.id}")
            variable.value = rng.choice(sorted(variable.choices))
            continue
        low, high = ranges.get(variable.id, (0.0, 1.0))
        if high <= low:
            high = low + 1.0
        variable.value = _round(rng.uniform(low, high), variable.type)


def satisfy_all(
    schema: WorldSchema, feature_ids: list[str], constraints: list[str]
) -> None:
    """Put every named candidate on the satisfying side of every constraint."""
    for name, operator, bound in parse_constraints(constraints):
        for feature_id in feature_ids:
            variable = _variable(schema, feature_id, name)
            variable.value = satisfying_value(operator, bound, variable.type)


def place_rest_out_of_contention(
    schema: WorldSchema, family: FamilySpec, spare: set[str]
) -> None:
    """Push every product that is neither the answer nor a near miss clearly out.

    A customer who states requirements rather than a shortlist is comparing the
    whole catalogue, so the answer has to be the only product that qualifies
    across all of it. Products outside the designated near misses fail two
    constraints, which keeps them obviously wrong without turning them into
    additional distractors the task never intended.
    """
    parsed = parse_constraints(family.constraints)
    if len(parsed) < 2:
        return
    for index, feature_id in enumerate(sorted(spare)):
        # Rotate which pair fails so the catalogue does not fail on one axis.
        first = index % len(parsed)
        second = (index + 1) % len(parsed)
        for position, (name, operator, bound) in enumerate(parsed):
            try:
                variable = _variable(schema, feature_id, name)
            except KeyError:
                continue
            variable.value = (
                violating_value(operator, bound, variable.type)
                if position in {first, second}
                else satisfying_value(operator, bound, variable.type)
            )


def apply_selection(schema: WorldSchema, family: FamilySpec) -> SolvedFamily:
    """One product qualifies; two miss by one thing; the rest are clearly out."""
    parsed = parse_constraints(family.constraints)
    satisfy_all(schema, [family.answer], family.constraints)

    near = family.near_misses or [c for c in family.candidates if c != family.answer]
    for index, feature_id in enumerate(near):
        # Rotate which constraint each distractor fails, so the same dimension is
        # not always the discriminating one.
        failing = index % len(parsed)
        for position, (name, operator, bound) in enumerate(parsed):
            variable = _variable(schema, feature_id, name)
            variable.value = (
                violating_value(operator, bound, variable.type)
                if position == failing
                else satisfying_value(operator, bound, variable.type)
            )

    spare = set(family.candidates) - {family.answer} - set(near)
    place_rest_out_of_contention(schema, family, spare)

    return SolvedFamily(
        family_id=family.family_id,
        kind=family.kind,
        candidates=list(family.candidates),
        answer=family.answer,
        segment=family.segment,
        near_misses=list(near),
        constraints=list(family.constraints),
        trap=(
            f"{len(near)} near misses failing exactly one constraint, "
            f"{len(spare)} further products clearly out"
        ),
    )


def apply_promotional_trap(schema: WorldSchema, family: FamilySpec) -> SolvedFamily:
    """Make the advertised boost smaller than the gap it appears to close.

    Every candidate passes the stated constraints -- including the base-rate
    floor -- so an agent that stops at the headline promotion picks a product
    whose total is still lower. The rivals are placed first and the answer is
    then lifted a full gap above them, because lifting the answer first would
    push the rivals below the floor and make them fail two constraints instead
    of the one that carries the trap.
    """
    gap, boost = 0.75, 0.5
    satisfy_all(schema, family.candidates, family.constraints)

    rivals = [c for c in family.candidates if c != family.answer]
    rival_base = max(
        float(_variable(schema, feature_id, "base_apy").value) for feature_id in rivals
    )
    for feature_id in rivals:
        _variable(schema, feature_id, "base_apy").value = _round(rival_base, "percent")
        rival_boost = _variable(schema, feature_id, "promotional_apy_boost")
        rival_boost.value = boost
        rival_boost.volatility = "promotional"
        rival_boost.window = dict(LIVE_WINDOW)

    answer_total = _round(rival_base + gap, "percent")
    _variable(schema, family.answer, "base_apy").value = answer_total
    _variable(schema, family.answer, "promotional_apy_boost").value = 0.0

    for feature_id in family.candidates:
        base = float(_variable(schema, feature_id, "base_apy").value)
        extra = float(_variable(schema, feature_id, "promotional_apy_boost").value)
        _add_derived(schema, feature_id, "effective_apy", base + extra)

    return SolvedFamily(
        family_id=family.family_id,
        kind=family.kind,
        candidates=list(family.candidates),
        answer=family.answer,
        segment=family.segment,
        near_misses=list(family.near_misses or rivals[:2]),
        constraints=[*family.constraints, f"effective_apy >= {answer_total}"],
        trap=(
            f"rivals advertise a {boost} point boost on a base rate {gap} points "
            f"lower, so every rival of {family.answer} totals "
            f"{_round(rival_base + boost, 'percent')} against {answer_total}"
        ),
    )


def apply_temporal_trap(schema: WorldSchema, family: FamilySpec) -> SolvedFamily:
    """Two promotional windows disagree; only the live one counts.

    The product with the higher headline promotional rate has an expired window,
    so an agent that never asks what day it is answers wrongly. Every other card
    is given an expired window too and an explicit effective rate, so the answer
    is the only one that qualifies across the catalogue rather than the only one
    that happens to have the derived value computed.
    """
    satisfy_all(schema, family.candidates, family.constraints)
    rivals = [c for c in family.candidates if c != family.answer]
    headline = rivals[0] if rivals else None

    winner_base, winner_promo_rate = 1.0, 5.0
    _configure_card(
        schema, family.answer, winner_base, winner_promo_rate, LIVE_WINDOW, True
    )
    if headline is not None:
        # The one a careless agent picks: the best advertised rate, already over.
        _configure_card(schema, headline, 1.5, 6.0, EXPIRED_WINDOW, False)
    for index, feature_id in enumerate(rivals[1:]):
        # Rivals must still clear the stated requirements: a card rejected for a
        # low base rate as well as an expired promotion is not a trap, it is an
        # ordinary near miss, and the derived rate stops being the only thing
        # that separates the catalogue.
        _configure_card(
            schema, feature_id, 1.0 + (index % 3) * 0.2, 4.0, EXPIRED_WINDOW, False
        )

    bound = _round((winner_promo_rate + 1.5) / 2, "percent")
    return SolvedFamily(
        family_id=family.family_id,
        kind=family.kind,
        candidates=list(family.candidates),
        answer=family.answer,
        segment=family.segment,
        near_misses=[headline] if headline else [],
        constraints=[*family.constraints, f"effective_cashback_rate >= {bound}"],
        trap=(
            f"{headline} advertises the higher promotional rate but its window "
            f"closed before {SIM_DATE}; {family.answer} wins only once the date "
            "is checked"
        ),
    )


def _configure_card(
    schema: WorldSchema,
    feature_id: str,
    base_rate: float,
    promo_rate: float,
    window: dict[str, str],
    live: bool,
) -> None:
    """Set one card's rates and the effective rate they imply on the fixed date."""
    base = _variable(schema, feature_id, "base_cashback_rate")
    promo = _variable(schema, feature_id, "promotional_cashback_rate")
    base.value = base_rate
    promo.value = promo_rate
    promo.volatility = "promotional"
    promo.window = dict(window)
    _add_derived(
        schema,
        feature_id,
        "effective_cashback_rate",
        promo_rate if live else base_rate,
    )


def set_identity_strings(schema: WorldSchema, products: list[ProductSpec]) -> None:
    """A product's class string and its tier are both documented values.

    The class string is what an opening tool expects verbatim. The tier is what
    a customer can state as the range they are shopping in, so it has to be
    written down somewhere the agent can retrieve it -- a tier nobody documents
    scopes nothing.
    """
    for spec in products:
        name = "card_class" if spec.kind == "card" else "account_class"
        _variable(schema, spec.feature_id, name).value = spec.class_name
        _variable(schema, spec.feature_id, "service_tier").value = spec.segment


def solve(
    schema: WorldSchema,
    families: list[FamilySpec],
    ranges: dict[str, tuple[float, float]],
    seed: int,
    products: list[ProductSpec],
) -> Assignment:
    """Assign every value, then impose each family's constraints and its trap."""
    # Stages are re-runnable, and derived variables are produced here rather than
    # read from the schema, so drop any left by a previous run instead of
    # appending a second copy.
    schema.variables = [v for v in schema.variables if not v.derived]
    seed_baseline(schema, ranges, seed)
    set_identity_strings(schema, products)

    solved = []
    for family in families:
        if family.kind == "promotional":
            solved.append(apply_promotional_trap(schema, family))
        elif family.kind == "temporal":
            solved.append(apply_temporal_trap(schema, family))
        else:
            solved.append(apply_selection(schema, family))
    return Assignment(seed=seed, families=solved, relaxations=[])
