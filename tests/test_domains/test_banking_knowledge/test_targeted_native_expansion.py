"""Expansion freezes quotas and pairs without turning replicas into new evidence."""

from collections import Counter

import pytest

from tau3.synthesis.storage import read_json, write_json
from tau3.synthesis.targeted.models import DIFFICULTIES, Finding
from tau3.synthesis.targeted.native.expansion import (
    content_identity,
    evaluation_manifest,
)
from tau3.synthesis.targeted.native.models import NativeConfig, NativeProfile
from tau3.synthesis.targeted.native.planning import (
    allocation,
    compile_plan,
    stage_slots,
)
from tau3.synthesis.targeted.native.v03_planning import recipe_counts


@pytest.fixture(scope="module")
def plans():
    profile = NativeProfile(round_id="expansion-test", base_model="base", report_hash="report", findings=[
        Finding(label=label, observation="x", interpretation="x", source="report", quote="x", confidence="high", severity="high")
        for label in ["F1", "F2", "F3", "F3b", "F4", "F5", "F6", "F7", "F8"]])
    config = NativeConfig(curriculum="v03_r10", clean_only=True, runtime_revision="native_card_lifecycle_v2", seed=44000)
    old = compile_plan(config, profile, "snapshot")
    new = compile_plan(config.model_copy(update={"expansion_target_rows": 3000}), profile, "snapshot")
    return old, new


def test_existing_tasks_and_seeds_preserved(plans):
    old, new = plans
    assert old.slots[:400] == new.slots[:400]
    assert old.pilot == new.pilot
    assert old.validation == new.validation
    assert old.gate == new.gate
    assert old.operation_coverage == new.operation_coverage


def test_every_expansion_cohort_is_frozen_balanced_and_pair_complete(plans):
    old, new = plans
    for stage, count in new.proposal["expansion"]["stages"].items():
        slots = stage_slots(new, stage)
        assert len(slots) == count
        assert [s.index for s in slots] == list(range(count))
        assert Counter(s.difficulty for s in slots) == allocation(count, DIFFICULTIES)
        assert Counter(s.origin for s in slots) == allocation(count, {"current": 70, "replay": 20, "explore": 10})
        assert set(Counter(s.pair_id for s in slots if s.pair_id).values()) == {2}
        assert all(len(set(s.trial_seeds)) == 4 for s in slots)
        recipes = Counter(s.family + "/" + (s.branch_id or "default") for s in slots)
        expected = recipe_counts(3000) if count == 3000 else {k: v * (count // 400) for k, v in recipe_counts(400).items()}
        assert recipes == expected
    with pytest.raises(ValueError, match="Unregistered"):
        stage_slots(old, "expand1600")
    with pytest.raises(ValueError, match="Unregistered"):
        stage_slots(new, "expand9999")


def test_replicas_and_metadata_edits_do_not_count_as_unique():
    row = {"messages": [{"role": "assistant", "content": "done"}], "tools": [], "loss_mask": [1], "metadata": {"seed": 1}}
    assert content_identity(row) == content_identity({**row, "metadata": {"seed": 2}, "task_id": "invented"})
    assert content_identity(row) != content_identity({**row, "messages": [{"role": "assistant", "content": "different"}]})


def test_cross_mechanism_validation_overlap_is_removed_before_sampling(plans):
    from tau3.synthesis.targeted.native.scenarios import compile_candidate

    _, plan = plans
    groups = {compile_candidate(s).group_id for s in plan.validation}
    assert groups == set(plan.proposal["expansion"]["validation_groups"])
    changes = plan.proposal["expansion"]["precollection_seed_rebindings"]
    assert changes  # The 44000 seed previously overlapped default optimization.
    for change in changes:
        for index in change["indices"]:
            assert index >= 400
            for attempt in range(3):
                assert compile_candidate(plan.slots[index], attempt).group_id not in groups


def test_external_receipt_cannot_fall_back_to_old_small_package(tmp_path, plans):
    old, new = plans
    write_json(tmp_path / "small/training_manifest.json", {"status": "READY", "rows": 890})
    assert evaluation_manifest(tmp_path, old) == read_json(tmp_path / "small/training_manifest.json")
    with pytest.raises(FileNotFoundError):
        evaluation_manifest(tmp_path, new)


def test_config_requires_explicit_native_v03():
    with pytest.raises(ValueError, match="v03"):
        NativeConfig(expansion_target_rows=3000)


def test_full_gate_is_not_bypassed(tmp_path, monkeypatch, plans):
    from tau3.synthesis.targeted.native import workflow
    from tau3.worldgen.v2.audit import AuditIncomplete

    _, plan = plans
    monkeypatch.setattr(workflow, "load_plan", lambda root: plan)
    write_json(tmp_path / "pilot/report.json", {"status": "COMPLETE"})
    write_json(tmp_path / "pilot/training_manifest.json", {"status": "READY"})
    write_json(tmp_path / "pilot/budget_forecast.json", {"status": "PASS"})
    workflow.guard_stage(tmp_path, "expand1600", "train")
    with pytest.raises(AuditIncomplete, match="gate"):
        workflow.guard_stage(tmp_path, "full", "train")


@pytest.mark.parametrize("changed", [False, True])
def test_quality_reuse_preserves_negative_decision_and_refuses_changed_result(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace

    from tau3.synthesis.storage import digest
    from tau3.synthesis.targeted.native import reuse

    parent, root = tmp_path / "parent", tmp_path / "child"
    relative = "tasks/train/00000/trials/0"
    source, directory = parent / relative, root / relative
    candidate = SimpleNamespace(task=SimpleNamespace(user_scenario="same user"))
    monkeypatch.setattr(reuse, "parent_candidate", lambda *args: (parent, candidate))
    sample = {"messages": [{"role": "assistant", "content": "ok"}], "tools": []}
    capture = {"identity": "old", "sample": sample, "simulation": {"messages": sample["messages"]}}
    new_capture = {**capture, "identity": "new"}
    previous = {"identity": "old", "capture_hash": digest(capture), "status": "COMPLETE",
                "environment_success": True, "simulation": capture["simulation"]}
    result = {**previous, "identity": "new", "capture_hash": digest(new_capture)}
    if changed:
        result["obligation_failures"] = ["changed judgment"]
    proof = {"original": True}
    review = {"reasoning_correct": False, "explanation": "original rejection"}
    write_json(source / "result.json", previous)
    write_json(source / "capture.json", capture)
    write_json(source.parent.parent / "provenance.json", proof)
    write_json(source / "quality.json", {"identity": digest([previous, sample, proof]), "review": review})
    write_json(directory / "capture.json", new_capture)
    write_json(directory / "adopted-capture.json", {"source": str(source), "strict_replay": True,
               "source_result_hash": digest(previous), "source_capture_hash": digest(capture)})
    for path in (parent, root):
        write_json(path / "snapshot/manifest.json", {"hashes": {"runtime/evaluator.py": "unchanged"}})
    config = SimpleNamespace(expansion_target_rows=3000)
    accepted = reuse.reuse_quality_review(root, config, candidate, directory, result, {"new": True})
    assert accepted is not changed
    if accepted:
        assert read_json(directory / "quality.json")["review"] == review
    else:
        assert not (directory / "quality.json").exists()


def test_handoff_checks_actual_slots_and_refuses_replica_target(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from tau3.synthesis.storage import digest
    from tau3.synthesis.targeted.native.expansion import audit_package, handoff
    from tau3.synthesis.targeted.native.models import NativeSlot

    slot = NativeSlot(index=0, split="train", family="debit_security", difficulty="5-9",
                      origin="current", seed=1, trial_seeds=[10, 11, 12, 13], evidence_ids=["finding:0"])
    plan = SimpleNamespace(slots=[slot], proposal={"expansion": {"stages": {"expand1600": 1}, "target_unique_rows": 2}},
                           model_dump=lambda **kwargs: {"test_plan": True})
    folder = tmp_path / "expand1600"
    write_json(tmp_path / "identity.json", {"identity": 1})
    write_json(tmp_path / "config.json", {})
    write_json(tmp_path / "validation/index.json", {"group_ids": ["held-out"]})
    for name in ("operation_coverage", "pair_checks"):
        write_json(folder / (name + ".json"), {"status": "PASS"})
    write_json(folder / "report.json", {"status": "COMPLETE", "expected_slots": 4, "complete_slots": 4,
               "valid_tasks": 1, "expected_tasks": 1, "qualified_rows": 2})
    rows = [{"task_id": "task", "seed": seed, "messages": [{"role": "assistant", "content": text}],
             "tools": [], "loss_mask": [1], "metadata": {"dataset_partition": "clean",
             "provenance": {"slot": slot.model_dump(mode="json"), "group_id": "train"}}}
            for seed, text in ((10, "one"), (11, "two"))]
    for i, seed in enumerate(slot.trial_seeds):
        write_json(tmp_path / "tasks/train/00000/trials" / str(i) / "slot.json",
                   {"seed": seed, "status": "COMPLETE", "sft_qualified": i < 2})

    def save():
        raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (folder / "sft.jsonl").write_bytes(raw)
        (folder / "general_agent_reasoning.jsonl").write_bytes(b"adapter")
        write_json(folder / "training_manifest.json", {"status": "READY", "round_identity": {"identity": 1},
                   "rows": 2, "rows_index": [{"row_hash": digest(r)} for r in rows],
                   "sft_sha256": hashlib.sha256(raw).hexdigest(),
                   "general_agent_sha256": hashlib.sha256(b"adapter").hexdigest(),
                   "token_audits": [{}, {}], "training_failures": []})

    save()
    assert audit_package(tmp_path, plan, "expand1600")["status"] == "READY"
    handoff(tmp_path, plan, "expand1600")
    rows[1]["messages"] = rows[0]["messages"]
    save()
    assert audit_package(tmp_path, plan, "expand1600")["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="target"):
        handoff(tmp_path, plan, "expand1600")
    rows[1]["messages"] = [{"role": "assistant", "content": "two"}]
    save()
    write_json(tmp_path / "tasks/train/00000/trials/3/slot.json",
               {"seed": 13, "status": "INCONCLUSIVE", "sft_qualified": False})
    assert not audit_package(tmp_path, plan, "expand1600")["checks"]["actual_complete_slots"]
