"""Explicit, read-only reuse of a stopped stream after compatibility replay."""

import json
from pathlib import Path

from tau3.data_model.simulation import SimulationRun
from tau3.evaluator.evaluator_env import EnvironmentEvaluator
from tau3.worldgen.v2.audit import AuditIncomplete
from tau3.worldgen.v2.blind import (
    Witness,
    check_frozen_witness,
    public_problem,
    solve_public,
)
from tau3.worldgen.v2.json_output import parse_object
from tau3.worldgen.v2.pipeline import artifact_hashes, implementation_hash
from tau3.worldgen.v2.runtime import digest


def read(path):
    """Read one immutable JSON artifact."""
    return json.loads(Path(path).read_text())


def check_migration(output, previous, world, settings):
    """A migration grants reuse only for its exact world, code and source run."""
    report = read(output / "migration.json")
    expected = {
        "source": str(previous.resolve()),
        "source_manifest": digest(read(previous / "manifest.json")),
        "implementation": implementation_hash(),
        "artifacts": artifact_hashes(world),
        "settings": digest(settings.model_dump()),
    }
    if report.get("status") != "PASS" or any(
        report.get(k) != v for k, v in expected.items()
    ):
        raise AuditIncomplete("Missing or stale explicit stream migration")
    return report


def check_trial(directory, task, world, settings):
    """Replay saved tool outputs strictly; keep original model grades and identities."""
    from tau3.synthesis.world_sft import trial_sample
    from tau3.worldgen.v2.environment import get_environment

    result = read(directory / "result.json")
    capture = read(directory / "capture.json")
    if (
        result["identity"] != read(directory / "started.json")
        or result["identity"] != capture["identity"]
        or result["capture_hash"] != digest(capture)
        or result["identity"]["task"] != digest(task.model_dump(mode="json"))
        or result["identity"]["model"] not in settings.agent_models
    ):
        raise AuditIncomplete("Saved teacher binding changed")
    simulation = SimulationRun.model_validate(result["simulation"])
    replay = EnvironmentEvaluator.calculate_reward(
        lambda **kwargs: get_environment(world, retrieval_variant="bm25"),
        task,
        simulation.messages,
        strict_replay=True,
    )
    old = simulation.reward_info
    if (
        old is None
        or replay.env_assertions != old.env_assertions
        or replay.db_check != old.db_check
    ):
        raise AuditIncomplete("Saved teacher outcome changed under new runtime")
    # Negative grades remain negative. Export validation is only for successful
    # samples; failed attempts must not be re-sampled during migration.
    if old.reward == 1:
        trial_sample(directory)
    return result


def replay_blind(world, previous, record, settings):
    """Require identical public prompts and raw replies, then replay private checks."""
    task = record["task"]
    problem = public_problem(
        world, task["id"], instructions=task["user_scenario"]["instructions"]
    )

    def ask(model, system, payload):
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        response = read(previous / "audit/calls" / f"{digest(request)}.json")
        if (
            response["request"] != request
            or response.get("status") != "COMPLETE"
            or response.get("finish_reason") not in (None, "stop")
        ):
            raise AuditIncomplete("Frozen public response is unavailable")
        return parse_object(response["response"])

    directory = (
        previous / "slots" / f"{record['slot']:06d}" / f"attempt_{record['attempt']}"
    )
    for i, model in enumerate(settings.agent_models):
        solution, review, attempts = solve_public(
            problem, model, settings.agent_models[1 - i], ask
        )
        saved = read(directory / f"blind_{i}.json")
        if (solution, review, attempts) != (
            saved["solution"],
            saved["review"],
            saved["attempts"],
        ):
            raise AuditIncomplete("Frozen task proof changed")
        check_frozen_witness(
            world, record["anchor_id"], Witness.model_validate(solution)
        )
