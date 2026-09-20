"""Transport recovery keeps conclusive grades and admissions immutable."""

import hashlib
import json
import runpy
from pathlib import Path

from tau3.worldgen.v2.pipeline import write_json
from tau3.worldgen.v2.runtime import digest


def test_only_inconclusive_nonadmitted_current_trials_are_archived(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_world_stream_http.py"
    prepare = runpy.run_path(str(script))["prepare_recovery"]
    output, previous = tmp_path / "output", tmp_path / "previous"
    write_json(output / "manifest.json", {"identity": {"max_rollouts": 36}})
    write_json(output / "audit/calls/pending.json", {"status": "RESERVED"})
    write_json(
        output / "audit/calls/invalid.json",
        {"status": "COMPLETE", "response": "invalid content"},
    )
    write_json(output / "audit/budget.json", {"used": 9})

    def trial(index, termination=None, captured=True):
        root = output / "slots" / f"{index:06d}" / "attempt_0/online_0"
        write_json(root / "started.json", {"original_identity": index})
        if captured:
            write_json(root / "capture.json", {"raw": "immutable"})
        if termination:
            write_json(
                root / "result.json",
                {
                    "simulation": {
                        "termination_reason": termination,
                        "reward_info": {"reward": 0},
                    }
                },
            )
        return root

    timeout = trial(0, "timeout")
    admitted = trial(1, "timeout")
    write_json(output / "admitted/000001.json", {"status": "published"})
    negative = trial(2, "agent_stop")
    interrupted = trial(3, captured=False)
    grading = trial(4)
    legacy = trial(5, "timeout")
    write_json(previous / legacy.relative_to(output) / "started.json", {"old": True})
    untouched = {
        str(p): p.read_bytes()
        for root in [admitted, negative, grading, legacy]
        for p in root.glob("*.json")
    }
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in timeout.glob("*.json")
    }
    deployment = {"id": "repair", "resume_from": str(previous)}
    plan = prepare(output, deployment, write_json, digest)
    assert plan["status"] == "ARCHIVED" and len(plan["trials"]) == 2
    assert len(plan["audit_calls"]) == 1
    assert not (output / "audit/calls/pending.json").exists()
    assert (output / "audit/calls/invalid.json").exists()
    assert json.loads((output / "audit/budget.json").read_text())["used"] == 9
    assert plan["already_charged_rollouts"] == 7
    assert not timeout.exists() and not interrupted.exists()
    row = next(
        r for r in plan["trials"] if r["trial"] == str(timeout.relative_to(output))
    )
    assert row["files"] == before
    assert all(Path(p).read_bytes() == content for p, content in untouched.items())
    budget_path = output / "transport/repair/rollout-budget.json"
    budget = json.loads(budget_path.read_text())
    budget["used"] += 1
    write_json(budget_path, budget)
    assert prepare(output, deployment, write_json, digest) == plan
    assert json.loads(budget_path.read_text())["used"] == 8
