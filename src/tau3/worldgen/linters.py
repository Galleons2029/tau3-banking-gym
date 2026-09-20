"""Stage E linters. M1 implements the two that decide whether tasks are well posed.

L4 and L5 are the "unit-style tests" the paper describes: L4 proves each task
has exactly one correct answer over the structured database, and L5 proves no
single document hands the agent enough to skip retrieval. Both run over the
whole task set, not task by task, because a uniqueness violation is usually a
collision between two tasks rather than a defect in either one.
"""

import math
import operator
import re
from dataclasses import dataclass, field

from tau3.worldgen.models import DocPlan, TaskSpec, WorldSchema

COMPARISONS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}
# `variable_name >= 2500`; the left side names a variable of the candidate
# feature, so one constraint reads the same against every candidate.
CONSTRAINT = re.compile(
    r"^\s*(?P<name>[a-z_][a-z0-9_]*)\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<value>\S+)\s*$"
)


@dataclass
class LintFinding:
    """One linter finding, named by the check that produced it."""

    check: str
    subject: str
    message: str
    # "error" blocks publication; "advice" is reported and tracked but does not.
    severity: str = "error"


@dataclass
class LintReport:
    """Findings across a whole world."""

    findings: list[LintFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def advice(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity != "error"]

    @property
    def passed(self) -> bool:
        """Advisory findings are reported without blocking a world."""
        return not self.errors

    def add(
        self, check: str, subject: str, message: str, severity: str = "error"
    ) -> None:
        self.findings.append(LintFinding(check, subject, message, severity))

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": len(self.errors),
            "advice": len(self.advice),
            "findings": [
                {
                    "check": f.check,
                    "subject": f.subject,
                    "message": f.message,
                    "severity": f.severity,
                }
                for f in self.findings
            ],
        }


def _coerce(text: str):
    """Read a constraint's right-hand side as a number or a bare string."""
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def satisfies(schema: WorldSchema, feature_id: str, constraint: str) -> bool:
    """Whether one candidate feature satisfies one constraint.

    A feature that does not define the variable fails the constraint rather than
    passing vacuously: an unstated fee is not a zero fee.
    """
    match = CONSTRAINT.match(constraint)
    if not match:
        raise ValueError(f"Unparsable constraint: {constraint!r}")
    wanted = _coerce(match.group("value"))
    for variable in schema.feature_variables(feature_id):
        if variable.name != match.group("name"):
            continue
        value = variable.value
        if isinstance(wanted, float):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
        return COMPARISONS[match.group("op")](value, wanted)
    return False


def solutions(schema: WorldSchema, spec: TaskSpec) -> list[str]:
    """Candidate features satisfying every constraint the customer states."""
    return [
        feature_id
        for feature_id in spec.candidate_features
        if all(satisfies(schema, feature_id, c) for c in spec.constraints)
    ]


def check_unique_answers(schema: WorldSchema, specs: list[TaskSpec]) -> LintReport:
    """L4: every constraint-bearing task resolves to exactly one product."""
    report = LintReport()
    for spec in specs:
        if not spec.constraints:
            continue
        if not spec.candidate_features:
            report.add("L4", spec.task_id, "Constraints given with no candidates")
            continue
        try:
            found = solutions(schema, spec)
        except ValueError as exc:
            report.add("L4", spec.task_id, str(exc))
            continue
        if len(found) != 1:
            report.add(
                "L4",
                spec.task_id,
                f"Constraints select {len(found)} products ({found}); exactly one required",
            )
        elif spec.unique_answer and found[0] != spec.unique_answer:
            report.add(
                "L4",
                spec.task_id,
                f"Constraints select {found[0]}, but the task expects {spec.unique_answer}",
            )
    return report


def check_near_misses(schema: WorldSchema, specs: list[TaskSpec]) -> LintReport:
    """Every rejected candidate must fail exactly one constraint.

    A candidate that fails several is not a distractor: the agent can discard it
    from any one document, and the task stops requiring the full comparison.
    """
    report = LintReport()
    for spec in specs:
        if len(spec.constraints) < 2:
            continue
        # Only the designated near misses; the rest of the catalogue is meant to
        # be clearly out, and holding it to the one-constraint rule would turn
        # every product into a distractor the task never intended.
        designated = spec.near_misses or [
            f for f in spec.candidate_features if f != spec.unique_answer
        ]
        for feature_id in designated:
            if feature_id == spec.unique_answer:
                continue
            failed = [
                c for c in spec.constraints if not satisfies(schema, feature_id, c)
            ]
            if len(failed) != 1:
                report.add(
                    "L4-near-miss",
                    f"{spec.task_id}:{feature_id}",
                    f"Distractor fails {len(failed)} constraints ({failed}); exactly one required",
                )
    return report


def check_scope_is_documented(schema: WorldSchema, specs: list[TaskSpec]) -> LintReport:
    """A stated tier must be one the answer is actually documented as being in.

    Layering the catalogue is what lets several scenario families of one product
    kind coexist, and the tier is the only thing separating their answers. If the
    tier a customer states is not the tier the answer's documents give it, the
    task asks for a product the evidence places somewhere else.
    """
    report = LintReport()
    for spec in specs:
        if not spec.scope or not spec.unique_answer:
            continue
        tier = next(
            (
                v
                for v in schema.variables
                if v.feature_id == spec.unique_answer and v.name == "service_tier"
            ),
            None,
        )
        if tier is None:
            report.add(
                "L12",
                spec.task_id,
                f"The customer asks for the {spec.scope} tier, but "
                f"{spec.unique_answer} documents no tier at all",
            )
        elif str(tier.value) != spec.scope:
            report.add(
                "L12",
                spec.task_id,
                f"The customer asks for the {spec.scope} tier, but the answer "
                f"{spec.unique_answer} is documented as {tier.value!r}",
            )
        outside = [
            feature.id
            for feature in schema.features
            if feature.id not in spec.candidate_features
            and all(satisfies(schema, feature.id, c) for c in spec.constraints)
        ]
        if outside:
            # Not a defect: it is what makes the stated tier load-bearing rather
            # than decorative. Reported so the difficulty it adds is visible.
            report.add(
                "L12",
                spec.task_id,
                f"{len(outside)} products outside the {spec.scope} tier meet "
                "every stated requirement; only the tier rules them out",
                severity="advice",
            )
    return report


def check_single_document_leakage(
    specs: list[TaskSpec], documents: list[DocPlan], leak_divisor: int = 3
) -> LintReport:
    """L5: no document reveals enough of one task to make retrieval optional."""
    report = LintReport()
    for spec in specs:
        required = spec.requires
        if not required:
            continue
        budget = math.ceil(len(required) / leak_divisor)
        for document in documents:
            shared = required & document.reveals
            if len(shared) > budget:
                report.add(
                    "L5",
                    f"{spec.task_id}:{document.doc_id}",
                    f"Reveals {len(shared)} of {len(required)} required elements "
                    f"(budget {budget}): {sorted(shared)}",
                )
    return report


def check_plan_references(schema: WorldSchema, documents: list[DocPlan]) -> LintReport:
    """Documents may only reveal variables and rules the schema defines."""
    report = LintReport()
    known_variables = {v.id for v in schema.variables}
    known_rules = {r.id for r in schema.rules}
    titles = {d.doc_id for d in documents}
    for document in documents:
        for variable_id in document.variable_ids:
            if variable_id not in known_variables:
                report.add(
                    "L1-plan", document.doc_id, f"Unknown variable {variable_id}"
                )
        for rule_id in document.rule_ids:
            if rule_id not in known_rules:
                report.add("L1-plan", document.doc_id, f"Unknown rule {rule_id}")
        for target in document.crossrefs:
            if target not in titles:
                report.add("L6", document.doc_id, f"Dangling cross-reference {target}")
    return report


def check_retrievable_subjects(
    schema: WorldSchema, documents: list[DocPlan], rendered: dict[str, str]
) -> LintReport:
    """Every document body must name the thing it is about.

    Retrieval indexes document content, not titles, so a product named only in a
    heading is unreachable by search or grep. A gold set containing such a
    document claims evidence the agent can never retrieve, which shows up as an
    unsolvable task rather than as a broken document.
    """
    report = LintReport()
    features = {feature.id: feature for feature in schema.features}
    for document in documents:
        feature = features.get(document.feature_id) if document.feature_id else None
        if feature is None:
            continue
        body = rendered.get(document.doc_id)
        if body is None:
            report.add("L11", document.doc_id, "Document was planned but not rendered")
            continue
        if feature.entity_name.lower() not in body.lower():
            report.add(
                "L11",
                document.doc_id,
                f"Body never names its subject {feature.entity_name!r}; "
                "the document cannot be retrieved",
            )
    return report


def check_official_name_collisions(schema: WorldSchema) -> LintReport:
    """L10: synthesized products must not reuse official corpus names.

    A shared name lets an agent answer from memory of the benchmark rather than
    from this world's documents, and makes the two corpora interfere if they are
    ever indexed together.
    """
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR
    from tau3.worldgen.targets import document_stem

    # A token shared by several official topics is a category word -- "saver",
    # "rewards", "account" -- and reusing it is what makes a synthesized
    # catalogue read like the real one. Only the distinctive part of a name,
    # appearing in one or two topics, identifies a product.
    topics: dict[str, set[str]] = {}
    for path in KNOWLEDGE_DOCUMENTS_DIR.glob("*.json"):
        stem = document_stem(path.stem)
        topics.setdefault(stem, set()).update(
            part for part in stem.split("_") if len(part) > 3
        )
    frequency: dict[str, int] = {}
    for tokens in topics.values():
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
    official = {token for token, count in frequency.items() if count <= 2}

    report = LintReport()
    for feature in schema.features:
        # Only product names carry the memorization risk. A process is called
        # what processes are called, and "Bank Account Closure" has to be
        # allowed to say "bank".
        if feature.entity_kind not in {"account", "card"}:
            continue
        tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", feature.entity_name.lower())
            if len(token) > 3
        }
        shared = sorted(tokens & official)
        if shared:
            report.add(
                "L10",
                feature.id,
                f"Entity name {feature.entity_name!r} reuses official corpus "
                f"terms {shared}",
            )
    return report


def lint_world(
    schema: WorldSchema,
    documents: list[DocPlan],
    specs: list[TaskSpec],
    leak_divisor: int = 3,
    rendered: dict[str, str] | None = None,
) -> LintReport:
    """Run the linters; the retrievability check needs rendered text."""
    report = LintReport()
    parts = [
        check_official_name_collisions(schema),
        check_plan_references(schema, documents),
        check_unique_answers(schema, specs),
        check_near_misses(schema, specs),
        check_scope_is_documented(schema, specs),
        check_single_document_leakage(specs, documents, leak_divisor),
    ]
    if rendered is not None:
        parts.append(check_retrievable_subjects(schema, documents, rendered))
    for part in parts:
        report.findings.extend(part.findings)
    return report
