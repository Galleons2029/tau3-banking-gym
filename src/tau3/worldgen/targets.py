"""Stage 0: measure the installed corpus so gates target it, not the paper.

docs/tau-banking-synth.md sets its acceptance targets from the paper's Table 1.
Several of those values do not describe the corpus actually installed here (gold
documents per task in particular), so every gate reads what this module measures
and treats the paper numbers as provenance.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Iterable

from tau3.synthesis.storage import environment_fingerprint
from tau3.worldgen.models import CorpusTargets, Distribution

DOC_ID_PATTERN = re.compile(r"^doc_(?P<stem>.+)_(?P<index>\d{3})$")
SUFFIXED_TOOL_PATTERN = re.compile(r"^(?P<base>.+)_(?P<suffix>\d{4})$")

# Paper Table 1. Kept for provenance; never used as a threshold.
PAPER_REFERENCE = {
    "documents": 698,
    "total_tokens_cl100k": 194562,
    "mean_tokens_per_document": 278.7,
    "categories": 21,
    "topics": 71,
    "discoverable_tools": 51,
    "permanent_tools": 14,
    "tasks": 97,
    "mean_gold_documents_per_task": 18.6,
    "mean_tool_calls_per_task": 9.52,
    "note": "Values from the paper. The installed corpus is measured directly; "
    "where the two disagree the measurement wins.",
}


def distribution(values: Iterable[float]) -> Distribution:
    """Summarize a distribution, tolerating samples too small for quantiles."""
    data = sorted(float(v) for v in values)
    if not data:
        raise ValueError("Cannot summarize an empty distribution")

    def percentile(p: float) -> float:
        if len(data) == 1:
            return data[0]
        position = (len(data) - 1) * p
        low = int(position)
        high = min(low + 1, len(data) - 1)
        return data[low] + (data[high] - data[low]) * (position - low)

    return Distribution(
        count=len(data),
        total=sum(data),
        mean=statistics.fmean(data),
        median=statistics.median(data),
        minimum=data[0],
        maximum=data[-1],
        p10=percentile(0.10),
        p25=percentile(0.25),
        p75=percentile(0.75),
        p90=percentile(0.90),
    )


def _encoder():
    """Return a cl100k token counter, or a word-count fallback that says so."""
    try:
        import tiktoken
    except ImportError:
        return "words-fallback", lambda text: len(text.split())
    encoding = tiktoken.get_encoding("cl100k_base")
    return "cl100k_base", lambda text: len(encoding.encode(text))


def document_stem(doc_id: str) -> str:
    """Topic stem of a document id, i.e. the id without its numeric suffix."""
    match = DOC_ID_PATTERN.match(doc_id)
    if not match:
        raise ValueError(f"Document id does not follow doc_<stem>_NNN: {doc_id}")
    return match.group("stem")


def category_of(stem: str, stems: set[str]) -> str:
    """Longest token prefix this stem shares with a sibling, else the whole stem.

    The corpus filename does not separate category from topic, so any category
    count is rule-dependent: this rule reproduces the paper's 21, while splitting
    on the first token alone yields 11. The rule name is recorded alongside the
    count and the count is never gated on.
    """
    tokens = stem.split("_")
    for size in range(len(tokens) - 1, 0, -1):
        prefix = "_".join(tokens[:size])
        if any(o != stem and o.startswith(prefix + "_") for o in stems):
            return prefix
    return stem


def tool_inventory() -> tuple[list[str], list[str], list[str]]:
    """Permanent, discoverable and suffix-bearing tools across both toolkits."""
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
    from tau3.environment.toolkit import DISCOVERABLE_ATTR, TOOL_ATTR

    permanent: set[str] = set()
    discoverable: set[str] = set()
    for toolkit in (KnowledgeTools, KnowledgeUserTools):
        for name in dir(toolkit):
            member = getattr(toolkit, name, None)
            if not callable(member) or not getattr(member, TOOL_ATTR, False):
                continue
            if getattr(member, DISCOVERABLE_ATTR, False):
                discoverable.add(name)
            else:
                permanent.add(name)
    suffixed = {n for n in discoverable | permanent if SUFFIXED_TOOL_PATTERN.match(n)}
    return sorted(permanent), sorted(discoverable), sorted(suffixed)


def measure_documents(documents_dir: Path) -> dict:
    """Token/word length, topic stems and the rule-dependent category count."""
    tokenizer, count_tokens = _encoder()
    tokens: list[int] = []
    words: list[int] = []
    stems: list[str] = []
    for path in sorted(documents_dir.glob("*.json")):
        document = json.loads(path.read_text())
        text = f"{document['title']}\n{document['content']}"
        tokens.append(count_tokens(text))
        words.append(len(text.split()))
        stems.append(document_stem(document["id"]))

    unique = set(stems)
    per_topic = [stems.count(stem) for stem in sorted(unique)]
    categories = {category_of(stem, unique) for stem in unique}
    return {
        "tokenizer": tokenizer,
        "documents": len(tokens),
        "document_tokens": distribution(tokens),
        "document_words": distribution(words),
        "topics": len(unique),
        "documents_per_topic": distribution(per_topic),
        "category_rule": "longest_shared_token_prefix",
        "categories": len(categories),
    }


def measure_tasks(tasks_dir: Path) -> dict:
    """Gold-document and reference-action distributions over a task set."""
    golds: list[int] = []
    actions: list[int] = []
    referenced: dict[str, int] = {}
    for path in sorted(tasks_dir.glob("task_*.json")):
        task = json.loads(path.read_text())
        documents = task.get("required_documents") or []
        golds.append(len(documents))
        for document_id in documents:
            referenced[document_id] = referenced.get(document_id, 0) + 1
        criteria = task.get("evaluation_criteria") or {}
        actions.append(len(criteria.get("actions") or []))

    return {
        "tasks": len(golds),
        "gold_documents": distribution(golds),
        "actions": distribution(actions),
        "distinct_gold_documents": len(referenced),
        "gold_document_reuse": (sum(referenced.values()) / len(referenced))
        if referenced
        else 0.0,
    }


def calibrate(
    documents_dir: Path | None = None, tasks_dir: Path | None = None
) -> CorpusTargets:
    """Measure the installed banking_knowledge corpus into acceptance targets."""
    from tau3.domains.banking_knowledge.utils import (
        KNOWLEDGE_DOCUMENTS_DIR,
        KNOWLEDGE_TASK_SET_PATH,
    )

    documents_dir = Path(documents_dir or KNOWLEDGE_DOCUMENTS_DIR)
    tasks_dir = Path(tasks_dir or KNOWLEDGE_TASK_SET_PATH)
    permanent, discoverable, suffixed = tool_inventory()
    return CorpusTargets(
        source=str(documents_dir.parent),
        environment_hash=environment_fingerprint(),
        permanent_tools=permanent,
        discoverable_tools=discoverable,
        suffixed_tools=suffixed,
        reference=PAPER_REFERENCE,
        **measure_documents(documents_dir),
        **measure_tasks(tasks_dir),
    )
