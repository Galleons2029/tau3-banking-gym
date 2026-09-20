"""Stage I: change one value without regenerating the world.

A published world is expensive: seven hundred documents is seven hundred model
calls. But most changes are not new prose — a rate moves, a fee changes, a
policy is reworded. Those touch the values a document states, not the sentences
around them, and the two-pass renderer already separates the two: Pass 1's
template is prose with placeholders, Pass 2 substitutes values.

So a value change is a Pass 2 rerun over the documents that mention it, which is
milliseconds, plus re-verification of the tasks that depend on it. Only a change
to what a document is *about* needs the model again, and this module says which
case a change is rather than guessing.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from tau3.worldgen.models import DocPlan, WorldPlan, WorldSchema


@dataclass
class Impact:
    """What one change touches."""

    variables: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    needs_model: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "variables": self.variables,
            "documents": self.documents,
            "tasks": self.tasks,
            "needs_model": self.needs_model,
        }


def template_hash(document: DocPlan, style_model: str, seed: int) -> str:
    """Identify a document's prose by what it was written from.

    Values are deliberately absent: the same template serves any value the
    solver later assigns, which is what makes a value change cheap.
    """
    payload = "|".join(
        [
            document.doc_id,
            document.title,
            document.archetype,
            ",".join(sorted(document.variable_ids)),
            ",".join(sorted(document.tool_ids)),
            ",".join(sorted(document.rule_ids)),
            ",".join(sorted(document.crossrefs)),
            style_model,
            str(seed),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def variable_index(plan: WorldPlan) -> dict[str, list[str]]:
    """variable -> the documents that state it."""
    index: dict[str, list[str]] = {}
    for document in plan.documents:
        for variable_id in document.variable_ids:
            index.setdefault(variable_id, []).append(document.doc_id)
    return index


def task_index(plan: WorldPlan) -> dict[str, list[str]]:
    """knowledge element -> the tasks that need it."""
    index: dict[str, list[str]] = {}
    for spec in plan.tasks:
        for element in spec.requires:
            index.setdefault(element, []).append(spec.task_id)
    return index


def plan_hashes(
    plan: WorldPlan, style_models: dict[str, str], seed: int
) -> dict[str, str]:
    """The template identity of every planned document."""
    return {
        document.doc_id: template_hash(
            document, style_models.get(document.doc_id, ""), seed
        )
        for document in plan.documents
    }


def assess(
    plan: WorldPlan,
    schema: WorldSchema,
    changed: list[str],
    recorded_hashes: dict[str, str] | None = None,
    style_models: dict[str, str] | None = None,
    seed: int = 0,
) -> Impact:
    """Which documents and tasks a set of changed variables reaches.

    A document needs the model again only when its brief has changed -- a
    variable that left it, or arrived in it, or a retitling. That is what the
    template hash records, and it deliberately excludes values, so a rate that
    moves costs a substitution rather than a generation.
    """
    by_variable = variable_index(plan)
    by_task = task_index(plan)
    known = {v.id for v in schema.variables}

    impact = Impact(variables=sorted(changed))
    documents, tasks, regenerate = set(), set(), set()
    for variable_id in changed:
        documents.update(by_variable.get(variable_id, []))
        tasks.update(by_task.get(variable_id, []))
        if variable_id not in known:
            # The variable is gone: any document that stated it was written from
            # a brief that no longer holds.
            regenerate.update(by_variable.get(variable_id, []))

    if recorded_hashes is not None:
        current = plan_hashes(plan, style_models or {}, seed)
        for doc_id, digest in current.items():
            if recorded_hashes.get(doc_id) != digest:
                regenerate.add(doc_id)
                documents.add(doc_id)

    impact.documents = sorted(documents)
    impact.tasks = sorted(tasks)
    impact.needs_model = sorted(regenerate)
    return impact


def refill(
    root: Path,
    schema: WorldSchema,
    documents: list[DocPlan],
    aliases: dict[str, str],
) -> list[str]:
    """Rerun Pass 2 over named documents; no model is called.

    A template may only be refilled while the brief it was written from still
    holds. Refilling one whose allocation has since changed produces a document
    stating figures that belong to some other document -- which L1 then reports
    as invented values, from prose that was never wrong. The check lives here
    rather than in the caller because the caller is exactly who gets it wrong.
    """
    from tau3.synthesis.storage import write_json
    from tau3.worldgen.render import PLACEHOLDER, fill

    written = []
    for document in documents:
        template_path = Path(root) / "render" / f"{document.doc_id}.md"
        if not template_path.exists():
            continue
        allowed = set(document.variable_ids) | {
            f"tool:{tool_id}" for tool_id in document.tool_ids
        }
        stale = sorted(set(PLACEHOLDER.findall(template_path.read_text())) - allowed)
        if stale:
            raise ValueError(
                f"{document.doc_id} was written from a different allocation "
                f"(orphan placeholders {stale[:3]}); re-render it instead of "
                "refilling"
            )
        write_json(
            Path(root) / "documents" / f"{document.doc_id}.json",
            {
                "id": document.doc_id,
                "title": document.title,
                "content": fill(template_path.read_text(), document, schema, aliases),
            },
        )
        written.append(document.doc_id)
    return written
