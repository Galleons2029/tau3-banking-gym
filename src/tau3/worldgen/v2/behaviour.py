"""Matched, complete behavioural comparisons with conservative equivalence intervals."""

import json
from math import sqrt
from pathlib import Path

from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash, write_json
from tau3.worldgen.v2.runtime import digest


def wilson(successes: int, count: int) -> tuple[float, float]:
    """Wilson 95% interval, including nonzero uncertainty at zero/one pass rates."""
    if count == 0:
        return 0.0, 1.0
    z, p = 1.96, successes / count
    divisor = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / divisor
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return centre - radius, centre + radius


def summarize(result, expected_tasks: dict) -> dict:
    """Every declared task/trial must be present, scored and normally terminated."""
    tasks = {t.id: t.model_dump(mode="json") for t in result.tasks}
    if not tasks or not tasks.keys() <= expected_tasks.keys():
        raise ValueError("Unknown or empty task set")
    if any(value != expected_tasks[key] for key, value in tasks.items()):
        raise ValueError("Result tasks differ from current source")
    trials = result.info.num_trials
    if trials < 1:
        raise ValueError("Invalid trial count")
    observed = {}
    for simulation in result.simulations:
        key = simulation.task_id, simulation.trial
        if key in observed or simulation.task_id not in tasks:
            raise ValueError("Duplicate or unexpected task/trial")
        if (
            simulation.reward_info is None
            or simulation.termination_reason.value not in {"agent_stop", "user_stop"}
        ):
            raise ValueError("Incomplete, unscored or truncated trajectory")
        if not simulation.messages:
            raise ValueError("Missing trajectory evidence")
        if any(
            c.justification.startswith("INCONCLUSIVE:")
            for c in simulation.reward_info.nl_assertions or []
        ):
            raise ValueError("Semantic judge did not produce a valid result")
        expected_basis = set(
            tasks[simulation.task_id]["evaluation_criteria"]["reward_basis"]
        )
        actual_basis = {b.value for b in simulation.reward_info.reward_basis or []}
        if not expected_basis <= actual_basis:
            raise ValueError("Result omits a required reward component")
        observed[key] = simulation.reward_info.reward == 1.0
    trial_ids = {trial for _, trial in observed}
    if (
        None in trial_ids
        or len(trial_ids) != trials
        or set(observed) != {(tid, trial) for tid in tasks for trial in trial_ids}
    ):
        raise ValueError("Incomplete task/trial grid")
    # Repeated trials are not independent sample units: use pass-all-trials/task.
    successes = sum(all(observed[tid, trial] for trial in trial_ids) for tid in tasks)
    return {
        "tasks": len(tasks),
        "trials": trials,
        "successes": successes,
        "rate": successes / len(tasks),
        "interval": wilson(successes, len(tasks)),
    }


def qualify(
    root: Path,
    synthetic_paths: list[Path],
    control_paths: list[Path],
    max_gap: float = 0.1,
    min_tasks: int = 50,
) -> dict:
    """Require matched models/settings, exact inputs and a contained gap interval."""
    from tau3.data_model.simulation import Results
    from tau3.domains.banking_knowledge.environment import get_tasks
    from tau3.worldgen.v2.settings import load_settings

    if not 0 < max_gap <= 0.1 or min_tasks < 50:
        raise ValueError(
            "Equivalence limits require gap <= 0.1 and at least 50 tasks per arm"
        )
    if len(synthetic_paths) != len(control_paths) or not synthetic_paths:
        raise ValueError("Provide paired synthetic/control results")
    expected = {
        p.stem: json.loads(p.read_text()) for p in (root / "tasks").glob("*.json")
    }
    controls = {t.id: t.model_dump(mode="json") for t in get_tasks()}
    identity = digest(artifact_hashes(root))
    comparisons, sources, models, variants = [], {}, set(), set()
    for synthetic_path, control_path in zip(
        synthetic_paths, control_paths, strict=True
    ):
        synth, control = Results.load(synthetic_path), Results.load(control_path)
        # Store only content identities; never export credentials in run settings.
        sources[str(synthetic_path.resolve())] = digest(synth.model_dump(mode="json"))
        sources[str(control_path.resolve())] = digest(control.model_dump(mode="json"))
        a, b = synth.info, control.info
        if (
            a.world_artifact_hash != identity
            or a.world_implementation_hash != implementation_hash()
            or a.world_evaluation_hash != digest(load_settings().model_dump())
        ):
            raise ValueError("Synthetic result belongs to stale or unidentified world")
        if (
            a.environment_info.domain_name != "banking_synth"
            or b.environment_info.domain_name != "banking_knowledge"
        ):
            raise ValueError("Wrong comparison domains")
        for field in [
            "agent_info",
            "user_info",
            "num_trials",
            "max_steps",
            "max_errors",
            "seed",
            "retrieval_config",
            "retrieval_config_kwargs",
        ]:
            if getattr(a, field) != getattr(b, field):
                raise ValueError(f"Unmatched experimental setting: {field}")
        left, right = summarize(synth, expected), summarize(control, controls)
        lo, hi = (
            left["interval"][0] - right["interval"][1],
            left["interval"][1] - right["interval"][0],
        )
        enough = min(left["tasks"], right["tasks"]) >= min_tasks
        status = (
            "PASS" if enough and -max_gap <= lo <= hi <= max_gap else "INCONCLUSIVE"
        )
        if enough and (lo > max_gap or hi < -max_gap):
            status = "FAIL"
        comparisons.append(
            {
                "model": a.agent_info.llm,
                "retrieval": a.retrieval_config,
                "synthetic": left,
                "control": right,
                "gap_interval": [lo, hi],
                "status": status,
            }
        )
        models.add(a.agent_info.llm)
        variants.add(a.retrieval_config)
    required_variants = {"no_knowledge", "golden_retrieval"}
    coverage = len(models) >= 2 and required_variants <= variants and len(variants) >= 3
    pairs = {(c["model"], c["retrieval"]) for c in comparisons}
    coverage = coverage and len(pairs) == len(models) * len(variants) == len(
        comparisons
    )
    report = {
        "status": "PASS"
        if coverage and all(c["status"] == "PASS" for c in comparisons)
        else "INCONCLUSIVE",
        "comparisons": comparisons,
        "complete_model_retrieval_grid": coverage,
        "world_artifact_hash": identity,
        "implementation": implementation_hash(),
        "sources": sources,
        "max_gap": max_gap,
        "scope": "Matched behavioural equivalence; not a claim of identical task distribution",
    }
    if any(c["status"] == "FAIL" for c in comparisons):
        report["status"] = "FAIL"
    write_json(root / "validation/behaviour.json", report)
    return report


def check_qualification(root: Path) -> dict:
    """Recheck all source runs and identities before trusting a previous comparison."""
    from tau3.data_model.simulation import Results

    path = root / "validation/behaviour.json"
    if not path.exists():
        raise ValueError("Missing matched behavioural certificate")
    report = json.loads(path.read_text())
    if (
        report["status"] != "PASS"
        or report["world_artifact_hash"] != digest(artifact_hashes(root))
        or report["implementation"] != implementation_hash()
    ):
        raise ValueError("Failed or stale matched behavioural certificate")
    for source, identity in report["sources"].items():
        if digest(Results.load(Path(source)).model_dump(mode="json")) != identity:
            raise ValueError("Behavioural source changed")
    return report
