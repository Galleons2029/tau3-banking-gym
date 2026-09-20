"""Stage D: write the documents in two passes, so no value is ever invented.

Pass 1 asks a model for prose given variable *names*, never values, and requires
every figure to appear as a `[[placeholder]]`. Pass 2 substitutes the values the
solver assigned. A model that never sees a number cannot get one wrong, which is
the single most effective guard against documents that contradict the database.

The gate between the passes is deterministic: any bare digit outside a short
whitelist means the model stated a fact of its own, and the document is written
again rather than repaired.
"""

import hashlib
import re
from dataclasses import dataclass

from tau3.worldgen.models import DocPlan, Variable, WorldConfig, WorldSchema

PLACEHOLDER = re.compile(r"\[\[(?P<name>[a-zA-Z0-9_:.]+)\]\]")
# Digits that are structure rather than fact: list numbering, four-digit years,
# and the numeric suffix of a tool name.
ALLOWED_DIGITS = [
    re.compile(r"(?m)^\s*\d+[.)]\s"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # snake_case identifiers: tool names and argument names such as
    # `last_4_digits` embed digits without stating a fact.
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
    re.compile(r"\[\[[^\]]*\]\]"),
    re.compile(r"(?m)^#{1,6}\s.*$"),
]
BARE_DIGIT = re.compile(r"\d")

PERSONAS = (
    "product marketing writer",
    "compliance officer",
    "support operations trainer",
    "help-centre writer",
)
REGISTERS = (
    "warm and customer-facing",
    "terse and customer-facing",
    "procedural and internal",
    "legalistic and internal",
)
INTERNAL_ARCHETYPES = {"internal_protocol", "tool_doc", "eligibility_matrix"}


@dataclass
class Style:
    """One document's assigned voice."""

    model: str
    persona: str
    register: str


class HallucinatedValue(RuntimeError):
    """Pass 1 stated a figure instead of leaving a placeholder."""


def style_for(document: DocPlan, models: list[str], salt: str) -> Style:
    """Rotate voice per document, keeping internal documents internal.

    Style is rotated so a retriever cannot separate documents by who wrote them:
    a corpus that clusters by author is easier to search than the one it is
    meant to imitate.
    """
    digest = int(hashlib.sha256(f"{salt}|{document.doc_id}".encode()).hexdigest(), 16)
    register_pool = (
        [r for r in REGISTERS if "internal" in r]
        if document.archetype in INTERNAL_ARCHETYPES
        else [r for r in REGISTERS if "customer" in r]
    )
    persona_pool = (
        [
            p
            for p in PERSONAS
            if p in {"compliance officer", "support operations trainer"}
        ]
        if document.archetype in INTERNAL_ARCHETYPES
        else list(PERSONAS)
    )
    return Style(
        model=models[digest % len(models)],
        persona=persona_pool[(digest // 7) % len(persona_pool)],
        register=register_pool[(digest // 13) % len(register_pool)],
    )


def format_value(variable: Variable) -> str:
    """Render a value the way a knowledge base states it."""
    value = variable.value
    if variable.type == "currency":
        return f"${float(value):,.0f}"
    if variable.type == "percent":
        text = f"{float(value):.2f}".rstrip("0").rstrip(".")
        return f"{text}%"
    if variable.type == "bool":
        return "Yes" if value else "No"
    if variable.type in {"int_days", "int_count"}:
        return f"{int(float(value)):,}"
    return str(value)


def unresolved_digits(text: str) -> list[str]:
    """Digits in Pass 1 output that are not structure and not placeholders."""
    masked = text
    for pattern in ALLOWED_DIGITS:
        masked = pattern.sub(" ", masked)
    return BARE_DIGIT.findall(masked)


def brief(
    document: DocPlan,
    schema: WorldSchema,
    tools: dict[str, dict],
    titles: dict[str, str],
) -> dict:
    """What Pass 1 is told: names, types and units, never a value."""
    variables = []
    for variable_id in document.variable_ids:
        variable = schema.variable(variable_id)
        variables.append(
            {
                "placeholder": f"[[{variable.id}]]",
                "means": variable.name.replace("_", " "),
                "type": variable.type,
                "unit": variable.unit or "",
            }
        )
    signatures = []
    for tool_id in document.tool_ids:
        binding = tools.get(tool_id, {})
        signatures.append(
            {
                "placeholder": f"[[tool:{tool_id}]]",
                "arguments": binding.get("arguments", []),
            }
        )
    rules = [
        rule.statement for rule in schema.rules if rule.id in set(document.rule_ids)
    ]
    subject = None
    if document.feature_id:
        subject = next(
            (f.entity_name for f in schema.features if f.id == document.feature_id),
            None,
        )
    return {
        "title": document.title,
        "subject": subject,
        "archetype": document.archetype,
        "target_tokens": document.target_tokens,
        "variables": variables,
        "tool_signatures": signatures,
        "policy_statements": rules,
        "cross_references": [titles[t] for t in document.crossrefs if t in titles],
    }


SYSTEM = """You write internal knowledge base documents for Rho-Bank, a fictional US retail bank.

You are given the facts a document must state as PLACEHOLDERS, never as values.

Absolute rules:
1. Every figure you state must be a placeholder copied exactly, e.g. [[var_copper_monthly_fee]].
   You must never write a number of your own: not a fee, not a rate, not a limit,
   not a day count. If you cannot express something without inventing a figure,
   leave it out.
2. Use every placeholder you are given, each at least once.
3. Name the subject of the document in the body itself, not only in the title:
   retrieval searches the body, and a product named only in a heading cannot be
   found at all.
4. Mention no product, fee or tool that is not in your brief.
5. Write Markdown, and prefer bullet lists. The corpus you are joining states
   most facts as short bullets under a heading; a table is for a genuine
   multi-column matrix, such as tiers against waiting periods, and is the
   exception rather than the house style. Internal procedures use numbered steps.
6. Output the document body only: no title line, no preamble, no commentary.

Numbered list markers, four-digit years, and the digits inside a tool name are
the only digits allowed to appear outside a placeholder."""


def instruction(document: DocPlan, style: Style, payload: dict) -> str:
    """The Pass 1 request for one document."""
    lines = [
        f"Title: {payload['title']}",
        f"Document type: {document.archetype}",
        f"Voice: you are a {style.persona}; the register is {style.register}.",
        f"Target length: about {payload['target_tokens']} tokens.",
        "",
        "Facts to state, as placeholders:",
    ]
    if payload.get("subject"):
        lines.insert(
            2, f"Subject: {payload['subject']} -- name it in the body, not only above."
        )
    for variable in payload["variables"]:
        unit = f" in {variable['unit']}" if variable["unit"] else ""
        lines.append(
            f"  - {variable['placeholder']} is the {variable['means']} "
            f"({variable['type']}{unit})"
        )
    if payload["tool_signatures"]:
        lines += ["", "Name these tools and list their arguments:"]
        for signature in payload["tool_signatures"]:
            lines.append(
                f"  - {signature['placeholder']} takes: "
                f"{', '.join(signature['arguments']) or 'no arguments'}"
            )
    if payload["policy_statements"]:
        lines += ["", "State this policy in your own words:"]
        lines += [f"  - {statement}" for statement in payload["policy_statements"]]
    if payload["cross_references"]:
        lines += ["", "Do not restate these; refer the reader to them by title:"]
        lines += [
            f'  - See the "{title}" document for details.'
            for title in payload["cross_references"]
        ]
    return "\n".join(lines)


def offline_body(document: DocPlan, payload: dict) -> str:
    """Deterministic development text; never publishable."""
    lines = [f"## {payload['title']}", ""]
    if payload["variables"]:
        lines += ["| Item | Value |", "|---|---|"]
        lines += [
            f"| {variable['means'].title()} | {variable['placeholder']} |"
            for variable in payload["variables"]
        ]
        lines.append("")
    for statement in payload["policy_statements"]:
        lines += [statement, ""]
    if payload["tool_signatures"]:
        lines.append("## Required Steps")
        lines.append("")
        for index, signature in enumerate(payload["tool_signatures"], start=1):
            arguments = ", ".join(signature["arguments"]) or "no arguments"
            lines.append(
                f"{index}. Use the {signature['placeholder']} tool ({arguments})."
            )
        lines.append("")
    for title in payload["cross_references"]:
        lines += [f'See the "{title}" document for details.', ""]
    return "\n".join(lines).strip() + "\n"


def fill(
    template: str, document: DocPlan, schema: WorldSchema, aliases: dict[str, str]
) -> str:
    """Pass 2: substitute assigned values and this world's tool names."""

    def resolve(match: re.Match) -> str:
        name = match.group("name")
        if name.startswith("tool:"):
            official = name.split(":", 1)[1]
            return aliases.get(official, official)
        return format_value(schema.variable(name))

    return PLACEHOLDER.sub(resolve, template)


def missing_placeholders(template: str, document: DocPlan) -> list[str]:
    """Placeholders the brief required but the draft never used."""
    used = set(PLACEHOLDER.findall(template))
    required = set(document.variable_ids) | {
        f"tool:{tool_id}" for tool_id in document.tool_ids
    }
    return sorted(required - used)


def unknown_placeholders(template: str, document: DocPlan) -> list[str]:
    """Placeholders the draft invented, which resolve to nothing."""
    allowed = set(document.variable_ids) | {
        f"tool:{tool_id}" for tool_id in document.tool_ids
    }
    return sorted(set(PLACEHOLDER.findall(template)) - allowed)


def draft(
    document: DocPlan,
    schema: WorldSchema,
    tools: dict[str, dict],
    titles: dict[str, str],
    style: Style,
    config: WorldConfig,
    attempts: int = 3,
) -> tuple[str, dict]:
    """Pass 1 with the hallucinated-figure gate; returns the template."""
    payload = brief(document, schema, tools, titles)
    if config.text_mode == "template":
        return offline_body(document, payload), {
            "model": "offline",
            "attempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    from tau3.data_model.message import SystemMessage, UserMessage
    from tau3.utils.llm_utils import generate

    problems: list[str] = []
    for attempt in range(attempts):
        note = ""
        if problems:
            note = (
                "\n\nYour previous draft was rejected: "
                + "; ".join(problems)
                + ". Write it again, obeying the rules exactly."
            )
        response = generate(
            model=style.model,
            messages=[
                SystemMessage(role="system", content=SYSTEM),
                UserMessage(
                    role="user", content=instruction(document, style, payload) + note
                ),
            ],
            call_name="worldgen_render",
            **{
                **config.llm_args,
                **config.generator_llm_args,
                "num_retries": 0,
                "max_retries": 0,
            },
        )
        usage = response.usage or {}
        body = (response.content or "").strip()
        body = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", body).strip()

        problems = []
        if not body:
            problems.append("it was empty")
        digits = unresolved_digits(body)
        if digits:
            problems.append(f"it stated {len(digits)} figures outside a placeholder")
        missing = missing_placeholders(body, document)
        if missing:
            problems.append(f"it never used {missing}")
        unknown = unknown_placeholders(body, document)
        if unknown:
            problems.append(f"it invented the placeholders {unknown}")
        subject = payload.get("subject")
        if subject and subject.lower() not in body.lower():
            problems.append(f"it never named its subject {subject!r} in the body")
        if not problems:
            return body, {
                "model": style.model,
                "attempts": attempt + 1,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

    raise HallucinatedValue(
        f"{document.doc_id}: {'; '.join(problems)} after {attempts} attempts"
    )
