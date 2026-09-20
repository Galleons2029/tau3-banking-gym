"""Stage H: does the synthesized world behave like the corpus it imitates?

Three layers, of increasing cost and increasing authority. Text statistics say
the documents look alike; retrieval says they are about as hard to search; the
behavioural layer says an agent finds the tasks about as hard, and it is the one
that decides.

The behavioural comparison is a delta, not an absolute. The paper's numbers come
from models this deployment does not serve, so the only meaningful question is
whether the same agent, under the same retrieval configuration, scores similarly
on this world and on a matched sample of the official one.
"""

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tau3.worldgen.models import DocPlan


@dataclass
class Comparison:
    """One measurement taken on both corpora."""

    metric: str
    synthetic: float
    official: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.synthetic - self.official

    @property
    def within(self) -> bool:
        return abs(self.delta) <= self.tolerance


@dataclass
class GateReport:
    """What one acceptance layer found."""

    gate: str
    comparisons: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Layers whose thresholds are advisory rather than blocking.
    advisory: bool = False

    @property
    def passed(self) -> bool:
        return self.advisory or all(c.within for c in self.comparisons)

    def as_dict(self) -> dict:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "advisory": self.advisory,
            "notes": self.notes,
            "comparisons": [
                {**asdict(c), "delta": round(c.delta, 3), "within": c.within}
                for c in self.comparisons
            ],
        }


def _encoder():
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except ImportError:  # pragma: no cover - tiktoken ships with the extra
        return lambda text: len(text.split())


def _corpus_texts(documents_dir: Path) -> dict[str, str]:
    texts = {}
    for path in sorted(documents_dir.glob("*.json")):
        document = json.loads(path.read_text())
        texts[document["id"]] = f"{document['title']}\n{document['content']}"
    return texts


def _mtld(text: str, threshold: float = 0.72) -> float:
    """Lexical diversity that does not fall away with length, unlike a raw ratio."""
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) < 10:
        return 0.0
    factors, types, tokens = 0.0, set(), 0
    for word in words:
        types.add(word)
        tokens += 1
        if len(types) / tokens <= threshold:
            factors += 1
            types, tokens = set(), 0
    if tokens:
        factors += (1 - len(types) / tokens) / (1 - threshold)
    return len(words) / factors if factors else float(len(words))


def text_statistics(texts: dict[str, str]) -> dict[str, float]:
    """Shape of a corpus: length, diversity, and how much of it is structure."""
    count = _encoder()
    lengths = [count(text) for text in texts.values()]
    sentences = []
    tables = bullets = 0
    for text in texts.values():
        parts = [p for p in text.replace("\n", " ").split(".") if p.strip()]
        sentences.extend(len(p.split()) for p in parts)
        tables += sum(1 for line in text.splitlines() if line.strip().startswith("|"))
        bullets += sum(
            1 for line in text.splitlines() if line.strip().startswith(("-", "*"))
        )
    total_lines = sum(len(text.splitlines()) for text in texts.values()) or 1
    return {
        "documents": float(len(texts)),
        "mean_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "median_tokens": float(statistics.median(lengths)) if lengths else 0.0,
        "mean_sentence_words": statistics.fmean(sentences) if sentences else 0.0,
        "mtld": statistics.fmean([_mtld(t) for t in texts.values()]) if texts else 0.0,
        "table_line_share": tables / total_lines,
        "bullet_line_share": bullets / total_lines,
    }


def compare_text(world: Path, advisory: bool = True) -> GateReport:
    """L1: the synthesized documents read like the corpus being imitated.

    Advisory by default. Length in particular varies by several percent between
    renders of the same configuration, so a tight gate here rejects corpora that
    are fine; the numbers are worth tracking and steering, not blocking on.
    """
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR

    synthetic = text_statistics(_corpus_texts(Path(world) / "documents"))
    official = text_statistics(_corpus_texts(Path(KNOWLEDGE_DOCUMENTS_DIR)))
    tolerances = {
        "mean_tokens": 60.0,
        "median_tokens": 60.0,
        "mean_sentence_words": 6.0,
        "mtld": 40.0,
        "table_line_share": 0.12,
        "bullet_line_share": 0.12,
    }
    report = GateReport(gate="text", advisory=advisory)
    for metric, tolerance in tolerances.items():
        report.comparisons.append(
            Comparison(metric, synthetic[metric], official[metric], tolerance)
        )
    report.notes.append(
        "Style similarity is the goal; exact length agreement is not required."
    )
    return report


def gold_recall(world: Path, top_k: int = 10) -> tuple[float, float]:
    """How much of each task's evidence one search round can reach, both corpora."""
    from tau3.domains.banking_knowledge.data_model import KnowledgeBase
    from tau3.domains.banking_knowledge.retrieval import (
        _create_kb_pipeline,
        resolve_variant,
    )
    from tau3.domains.banking_knowledge.utils import (
        KNOWLEDGE_DOCUMENTS_DIR,
        KNOWLEDGE_TASK_SET_PATH,
    )

    variant = resolve_variant("bm25_grep", top_k=top_k)

    def measure(documents_dir: Path, tasks_dir: Path) -> float:
        knowledge = KnowledgeBase.load(str(documents_dir))
        pipeline = _create_kb_pipeline(variant.kb_search, knowledge)
        scores = []
        for path in sorted(tasks_dir.glob("task_*.json")):
            task = json.loads(path.read_text())
            gold = set(task.get("required_documents") or [])
            if not gold:
                continue
            query = task.get("user_scenario", {}).get("instructions", "")[:600]
            found = {doc_id for doc_id, _ in pipeline.retrieve(query)}
            scores.append(len(gold & found) / len(gold))
        return statistics.fmean(scores) if scores else 0.0

    world = Path(world)
    return (
        measure(world / "documents", world / "tasks"),
        measure(Path(KNOWLEDGE_DOCUMENTS_DIR), Path(KNOWLEDGE_TASK_SET_PATH)),
    )


def compare_retrieval(world: Path, tolerance: float = 0.20) -> GateReport:
    """L2: a task's evidence is about as reachable here as in the official corpus.

    Recall at a fixed cut-off depends on how much of the corpus that cut-off is,
    so the two corpus sizes travel with the number. A ten-document window over
    eighty documents is a far wider net than the same window over six hundred
    and ninety-eight, and a smaller world will look easier to search for that
    reason alone.
    """
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR

    synthetic, official = gold_recall(world)
    synthetic_size = len(list((Path(world) / "documents").glob("*.json")))
    official_size = len(list(Path(KNOWLEDGE_DOCUMENTS_DIR).glob("*.json")))

    report = GateReport(gate="retrieval")
    report.comparisons.append(
        Comparison("gold_recall_at_10", synthetic, official, tolerance)
    )
    report.notes.append(
        "Both measured with the same BM25 pipeline and the task's own opening "
        "instructions as the query."
    )
    report.notes.append(
        f"Corpus sizes: {synthetic_size} synthetic against {official_size} "
        f"official. A top-10 window covers "
        f"{10 / max(synthetic_size, 1):.1%} of this world and "
        f"{10 / max(official_size, 1):.1%} of the official corpus, so a recall "
        "gap of roughly that ratio is expected from size alone and this "
        "comparison only becomes meaningful at full scale."
    )
    return report


def official_control_sample(size: int, seed: int) -> list[str]:
    """A size-matched official sample, stratified by how much work a task is.

    Comparing against whichever official tasks happen to come first would
    compare difficulty with task length, so the sample is drawn across the range
    of reference-action counts.
    """
    from tau3.domains.banking_knowledge.utils import KNOWLEDGE_TASK_SET_PATH

    tasks = []
    for path in sorted(Path(KNOWLEDGE_TASK_SET_PATH).glob("task_*.json")):
        task = json.loads(path.read_text())
        actions = len((task.get("evaluation_criteria") or {}).get("actions") or [])
        tasks.append((actions, task["id"]))
    tasks.sort()
    if size >= len(tasks):
        return [task_id for _, task_id in tasks]

    rng = random.Random(seed)
    buckets = max(1, min(size, 4))
    per_bucket = size // buckets
    chosen: list[str] = []
    width = len(tasks) / buckets
    for index in range(buckets):
        window = tasks[int(index * width) : int((index + 1) * width)]
        take = per_bucket if index < buckets - 1 else size - len(chosen)
        chosen += [task_id for _, task_id in rng.sample(window, min(take, len(window)))]
    return chosen


def summarize_run(results_path: Path) -> dict:
    """Pass rate, alongside the trajectories that are not evidence about it.

    A simulation that died on a service error never attempted its task, and one
    that ran out of steps never finished it. Counting either as a failure would
    quietly move the pass rate, and because the two corpora are run separately,
    an outage on one side would surface as a difficulty difference.
    """
    results = json.loads(Path(results_path).read_text())
    simulations = results.get("simulations", [])
    scored = [s for s in simulations if s.get("reward_info") is not None]
    broken = [s for s in simulations if s.get("reward_info") is None]
    summary = {
        "tasks": len(simulations),
        "scored": len(scored),
        "pass_rate": 0.0,
        "mean_reward": 0.0,
        "max_steps": 0,
        "max_steps_share": 0.0,
        "infrastructure_errors": len(broken),
        "infrastructure_error_tasks": [s.get("task_id") for s in broken],
    }
    if not scored:
        return summary

    rewards = [s["reward_info"]["reward"] for s in scored]
    truncated = sum(1 for s in scored if s.get("termination_reason") == "max_steps")
    summary.update(
        pass_rate=statistics.fmean(1.0 if r >= 1.0 else 0.0 for r in rewards),
        mean_reward=statistics.fmean(rewards),
        max_steps=truncated,
        max_steps_share=truncated / len(scored),
    )
    return summary


def delta_uncertainty(synthetic: dict, official: dict) -> float:
    """Roughly how far the measured difference could be from the real one.

    Two independent pass rates over a handful of tasks each; the standard error
    of their difference is what decides whether a gate can tell a real gap from
    noise. Reported so a tolerance wider than this interval is visibly a
    formality rather than a test.
    """
    import math

    def variance(summary: dict) -> float:
        n = max(summary["scored"], 1)
        p = summary["pass_rate"]
        return p * (1 - p) / n

    return 1.96 * math.sqrt(variance(synthetic) + variance(official))


def compare_behaviour(
    synthetic_run: Path, official_run: Path, tolerance: float = 0.30
) -> GateReport:
    """L3: the same agent finds both worlds about as hard."""
    synthetic = summarize_run(synthetic_run)
    official = summarize_run(official_run)
    report = GateReport(gate="behaviour")
    report.comparisons.append(
        Comparison(
            "pass_rate", synthetic["pass_rate"], official["pass_rate"], tolerance
        )
    )
    report.notes.append(
        f"Scored {synthetic['scored']}/{synthetic['tasks']} synthetic and "
        f"{official['scored']}/{official['tasks']} official trajectories. "
        f"Truncated by step budget: {synthetic['max_steps']} synthetic, "
        f"{official['max_steps']} official; a truncated trajectory is not "
        "evidence about difficulty."
    )
    if synthetic["infrastructure_errors"] or official["infrastructure_errors"]:
        report.notes.append(
            f"Excluded from the pass rates: {synthetic['infrastructure_errors']} "
            f"synthetic and {official['infrastructure_errors']} official "
            f"trajectories that failed on a service error "
            f"({official.get('infrastructure_error_tasks', [])}). These never "
            "attempted their task, and counting them as failures would move a "
            "pass rate for reasons that have nothing to do with the corpus."
        )
    interval = delta_uncertainty(synthetic, official)
    comparison = report.comparisons[0]
    report.notes.append(
        f"The measured difference is {comparison.delta:+.2f} with a 95% interval "
        f"of about ±{interval:.2f} at these sample sizes."
        + (
            " That interval is wider than the tolerance, so this layer cannot "
            "yet distinguish a real difference from sampling noise: passing it "
            "is not evidence that the two corpora are equally hard."
            if interval > comparison.tolerance
            else ""
        )
    )
    if synthetic["max_steps_share"] > 0.25 or official["max_steps_share"] > 0.25:
        report.notes.append(
            "More than a quarter of one side was truncated; raise the step "
            "budget before reading the pass rates as difficulty."
        )
    return report


def document_archetypes(documents: list[DocPlan]) -> dict[str, int]:
    """Archetype mix of a plan, for the text layer's report."""
    counts: dict[str, int] = {}
    for document in documents:
        counts[document.archetype] = counts.get(document.archetype, 0) + 1
    return counts
