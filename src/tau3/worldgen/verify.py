"""Stage G: is each task solvable, and is its gold set a minimal cover?

Two automated checks stand in for the paper's two independent human reviewers.
G1 executes the reference actions against the real environment: the official
tools enforce their own eligibility rules and report violations as `Error:`
text, so a reference sequence in the wrong order fails here rather than
producing an unsolvable task. G2 checks the gold document set is exactly a
minimal cover of the knowledge the task needs — too small and the task is
unsolvable from its own evidence, too large and retrieval scores mean nothing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from tau3.data_model.tasks import Task
from tau3.synthesis.validation import fresh_environment, replay
from tau3.worldgen.models import DocPlan, TaskSpec, WorldPlan

DOMAIN = "banking_synth"
# Tools the agent always has; they need no knowledge base evidence to be found.
PERMANENT_ROUTES = {
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
    "list_discoverable_agent_tools",
    "give_discoverable_user_tool",
    "call_discoverable_user_tool",
    "list_discoverable_user_tools",
}


@dataclass
class TaskVerdict:
    """Outcome of verifying one task."""

    task_id: str
    solvable: bool = False
    minimal: bool = False
    complete: bool = False
    errors: list[str] = field(default_factory=list)
    missing_knowledge: list[str] = field(default_factory=list)
    redundant_documents: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.solvable and self.complete and self.minimal and not self.errors


def discoverable_routes(task: Task) -> set[str]:
    """Discoverable tools the reference actions actually route through.

    Reference actions reach these tools through `unlock`/`call`, so the tool
    name lives in the arguments rather than in the action name.
    """
    used: set[str] = set()
    for action in task.evaluation_criteria.actions:
        if action.name in {
            "unlock_discoverable_agent_tool",
            "call_discoverable_agent_tool",
        }:
            name = action.arguments.get("agent_tool_name")
        elif action.name in {
            "give_discoverable_user_tool",
            "call_discoverable_user_tool",
        }:
            name = action.arguments.get("discoverable_tool_name")
        else:
            continue
        if name:
            used.add(name)
    return used


def verify_solvable(task: Task, verdict: TaskVerdict) -> None:
    """G1: the reference actions execute cleanly, in the order recorded."""
    try:
        replay(task, strict=True, domain=DOMAIN)
        verdict.solvable = True
    except Exception as exc:
        verdict.errors.append(f"G1 replay failed: {exc}")


def verify_gold_cover(
    spec: TaskSpec, plans: dict[str, DocPlan], gold: list[str], verdict: TaskVerdict
) -> None:
    """G2: gold(t) is a minimal cover of Req(t)."""
    unknown = [doc_id for doc_id in gold if doc_id not in plans]
    if unknown:
        verdict.errors.append(f"G2 gold references unplanned documents: {unknown}")
        return

    required = spec.requires
    covered: set[str] = set()
    for doc_id in gold:
        covered |= plans[doc_id].reveals

    missing = sorted(required - covered)
    verdict.missing_knowledge = missing
    verdict.complete = not missing

    redundant = []
    for doc_id in gold:
        without = set()
        for other in gold:
            if other != doc_id:
                without |= plans[other].reveals
        if required <= without:
            redundant.append(doc_id)
    verdict.redundant_documents = redundant
    verdict.minimal = not redundant

    if missing:
        verdict.errors.append(f"G2 gold set does not cover: {missing}")
    if redundant:
        verdict.errors.append(f"G2 gold set is not minimal, drop: {redundant}")


def verify_tool_evidence(
    task: Task, spec: TaskSpec, alias_map: dict[str, str], verdict: TaskVerdict
) -> None:
    """The declared tools must be the tools the reference actions really use.

    Plans name official tools, because a plan outlives any one world's suffixes;
    reference actions name this world's aliases, because that is the interface
    the agent is given. The comparison happens in the official namespace.
    """
    routed = {alias_map.get(name, name) for name in discoverable_routes(task)}
    declared = set(spec.requires_tools)
    if routed != declared:
        verdict.errors.append(
            f"Declared tools {sorted(declared)} do not match reference actions "
            f"{sorted(routed)}"
        )


def verify_verbatim_arguments(task: Task, spec: TaskSpec, verdict: TaskVerdict) -> None:
    """Values the evaluator compares verbatim must be stated by the customer.

    A closure reason is scored by exact string match but originates only in the
    conversation, so an agent that reasons perfectly still fails unless the
    customer said the exact words. Requiring them in the instructions -- and in
    the reference actions -- keeps the task winnable.
    """
    instructions = task.user_scenario.instructions or ""
    recorded = json.dumps(
        [action.arguments for action in task.evaluation_criteria.actions]
    )
    for value in spec.verbatim_arguments:
        if value not in instructions:
            verdict.errors.append(
                f"Verbatim argument {value!r} is never stated in the user instructions"
            )
        if value not in recorded:
            verdict.errors.append(
                f"Verbatim argument {value!r} appears in no reference action"
            )


def literal_arguments(task: Task) -> list[tuple[str, str]]:
    """Free-text argument values a reference action supplies.

    Identifiers and figures are excluded: an account id is read back from a
    tool, and a number is either stated by the customer or computed. What
    remains are the words an agent has to know, such as the exact product class
    an opening tool expects.
    """
    found: list[tuple[str, str]] = []
    for action in task.evaluation_criteria.actions:
        payloads = [action.arguments]
        inner = action.arguments.get("arguments")
        if isinstance(inner, str):
            try:
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                payloads.append(parsed)
        for payload in payloads:
            for name, value in payload.items():
                if name in {"arguments", "agent_tool_name", "discoverable_tool_name"}:
                    continue
                if not isinstance(value, str) or len(value) < 4:
                    continue
                if any(character.isdigit() for character in value):
                    continue
                if value.replace("_", "").isalnum() and value.islower():
                    continue
                found.append((name, value))
    return found


def verify_argument_grounding(task: Task, corpus: str, verdict: TaskVerdict) -> None:
    """Every word a reference action supplies must be knowable from somewhere.

    A gold set can cover a task's declared requirements and the task still be
    unsolvable: if an opening tool expects the exact string "Business Checking
    Account" and no document contains it, no amount of retrieval tells the agent
    what to type. Coverage is checked against what a task says it needs, so this
    checks what its actions actually use.
    """
    instructions = task.user_scenario.instructions or ""
    for name, value in literal_arguments(task):
        if value in corpus or value in instructions:
            continue
        verdict.errors.append(
            f"Reference action passes {name}={value!r}, which appears in no "
            f"document and in nothing the customer says"
        )


def verify_world(root: Path, alias_map: dict[str, str] | None = None) -> dict:
    """Run G1 and G2 over every task in a world bundle."""
    from tau3.worldgen.world import load_plan, load_tasks, read_world_manifest

    if alias_map is None:
        alias_map = read_world_manifest(root).alias_map
    plan: WorldPlan = load_plan(root)
    plans = {document.doc_id: document for document in plan.documents}
    specs = {spec.task_id: spec for spec in plan.tasks}
    tasks = load_tasks(root)
    corpus = "\n".join(
        json.loads(path.read_text()).get("content", "")
        for path in sorted((Path(root) / "documents").glob("*.json"))
    )

    verdicts: list[TaskVerdict] = []
    for task in tasks:
        verdict = TaskVerdict(task_id=task.id)
        spec = specs.get(task.id)
        if spec is None:
            verdict.errors.append("No task specification for this task")
            verdicts.append(verdict)
            continue
        verify_tool_evidence(task, spec, alias_map, verdict)
        verify_verbatim_arguments(task, spec, verdict)
        verify_argument_grounding(task, corpus, verdict)
        verify_gold_cover(spec, plans, task.required_documents or [], verdict)
        verify_solvable(task, verdict)
        verdicts.append(verdict)

    return {
        "world": str(root),
        "tasks": len(verdicts),
        "passed": sum(1 for v in verdicts if v.passed),
        "verdicts": [
            {
                "task_id": v.task_id,
                "solvable": v.solvable,
                "complete": v.complete,
                "minimal": v.minimal,
                "errors": v.errors,
                "missing_knowledge": v.missing_knowledge,
                "redundant_documents": v.redundant_documents,
            }
            for v in verdicts
        ],
    }


def environment_for(task: Task):
    """A fresh synth-world environment initialized for one task."""
    return fresh_environment(task, DOMAIN)
