"""Filtered historical retention cannot revive excluded rows or reroll old seeds."""

import json
from types import SimpleNamespace

import pytest

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.expansion import content_identity
from tau3.synthesis.targeted.native.retention import (
    audit_retention,
    bind_retention,
    inherited_role_review,
    sha256,
)
from tau3.synthesis.targeted.native.user_roles import role_context


@pytest.fixture
def filtered(tmp_path):
    root, package = tmp_path / "round", tmp_path / "filter"
    root.mkdir()
    package.mkdir()
    rows, index, files = [], [], {}
    for seed, status in enumerate(("KEEP", "DISCARD", "QUARANTINE")):
        text = "My own request" if status == "KEEP" else "Please verify your identity"
        messages = [{"role": "user", "content": text}]
        row = {"task_id": "task", "seed": seed, "messages": messages, "tools": [], "loss_mask": [0]}
        context = role_context(messages)
        reviews = []
        for model in ("glm", "gemini"):
            label = "customer" if status == "KEEP" else "bank_staff"
            if status == "QUARANTINE" and model == "gemini":
                label = "uncertain"
            reviews.append({"model": model, "status": "COMPLETE", "context_hash": digest(context),
                            "review": {"turns": [{"customer_index": 0, "role": label,
                            "quote": "" if label == "customer" else text, "reason": "evidence"}]}})
        index.append({"task_id": "task", "seed": seed, "source_line": seed + 1,
                      "source_line_sha256": "bound", "disposition": status, "reviews": reviews})
        path = package / (status + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        files[status] = {"path": str(path), "sha256": sha256(path)}
        rows.append(row)
    source = package / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(package / "sampling_index.json", index)
    manifest = package / "manifest.json"
    write_json(manifest, {"status": "COMPLETE", "source": str(source), "source_sha256": sha256(source),
                         "source_rows": 3, "mapping_sha256": sha256(package / "sampling_index.json"),
                         "counts": {k: 1 for k in files}, "files": files})
    config = SimpleNamespace(retained_sft_manifest=str(manifest), teacher_model="glm", reviewer_model="gemini")
    bind_retention(root, config)
    return root, config, rows, package


def test_only_approved_subset_satisfies_retention(filtered):
    root, _, rows, _ = filtered
    assert audit_retention(root, {content_identity(rows[0])})["passed"]
    assert not audit_retention(root, set())["passed"]
    assert not audit_retention(root, {content_identity(r) for r in rows})["passed"]


def test_retention_detects_mutated_partition(filtered):
    root, _, rows, package = filtered
    with (package / "KEEP.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="partition bytes"):
        audit_retention(root, {content_identity(rows[0])})


@pytest.mark.parametrize("seed,status", [(0, "KEEP"), (1, "DISCARD"), (2, "QUARANTINE")])
def test_reuse_is_for_exact_adopted_context_only(filtered, seed, status):
    root, config, rows, _ = filtered
    directory = root / "trial"
    directory.mkdir()
    result = {"identity": {"seed": seed}, "simulation": {"task_id": "task", "messages": rows[seed]["messages"]}}
    assert inherited_role_review(root, config, directory, result) is None
    write_json(directory / "adopted-capture.json", {"strict_replay": True})
    assert inherited_role_review(root, config, directory, result)["disposition"] == status
    result["simulation"]["messages"] = [{"role": "user", "content": "a different capture"}]
    with pytest.raises(ValueError, match="customer context"):
        inherited_role_review(root, config, directory, result)


def test_capture_lookup_skips_task_only_parent(tmp_path, monkeypatch):
    from tau3.synthesis.targeted.native import reuse

    root, middle, ancestor = [tmp_path / p for p in ("child", "task_only", "sampled")]
    relative = "tasks/train/00000/trials/0"
    candidate = SimpleNamespace(slot=SimpleNamespace(split="train", index=0))
    monkeypatch.setattr(reuse, "NativeCandidate", SimpleNamespace(model_validate=lambda _: candidate))
    monkeypatch.setattr(reuse, "same_business_task", lambda *_: True)
    config = SimpleNamespace(reuse_parent_splits=["train"], reuse_parent_pilot=False, parent_round=str(middle),
                             settings=SimpleNamespace(model_dump=lambda **_: {}))
    write_json(root / "budget-inheritance.json", {"parent": str(middle)})
    binding = {"parent": str(ancestor), "identity": {}, "budget": {}, "blind_budget": {}}
    binding["hash"] = digest(binding)
    write_json(middle / "budget-inheritance.json", binding)
    for name in ("identity", "budget", "blind-budget"):
        write_json(ancestor / (name + ".json"), {})
    write_json(ancestor / "supervisor.json", {"status": "STOPPED"})
    for parent in (middle, ancestor):
        write_json(parent / "tasks/train/00000/provenance.json", {"status": "VALID", "candidate_hash": digest({})})
        write_json(parent / "tasks/train/00000/candidate.json", {})
    for name in ("capture.json", "result.json"):
        write_json(ancestor / relative / name, {})
    write_json(ancestor / "config.json", {"settings": {}})
    manifest = {"hashes": {}, "public_tool_schema_hash": "tools", "policy_hash": "policy", "retrieval": "bm25_grep"}
    for directory in (root, ancestor):
        write_json(directory / "snapshot/manifest.json", manifest)
    assert reuse.parent_candidate(root, config, candidate, relative) == (ancestor, candidate)
    changed = read_json(root / "snapshot/manifest.json")
    changed["hashes"]["runtime/agent/llm_agent.py"] = "unexpected change"
    write_json(root / "snapshot/manifest.json", changed)
    assert reuse.parent_candidate(root, config, candidate, relative) is None


def test_supervisor_rejects_export_with_changed_role_evidence(filtered):
    import importlib.util
    from pathlib import Path

    root, _, rows, package = filtered
    script = Path(__file__).resolve().parents[3] / "scripts/run_targeted_native_expansion.py"
    spec = importlib.util.spec_from_file_location("role_expansion_driver", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_json(root / "config.json", {"user_role_guard": "customer_role_v2", "teacher_model": "glm", "reviewer_model": "gemini"})
    directory = root / "tasks/train/00000/trials/0"
    reviews = read_json(package / "sampling_index.json")[0]["reviews"]
    proof = {"result_hash": "result", "disposition": "KEEP", "reviews": reviews}
    write_json(directory / "customer-role-review.json", proof)
    write_json(directory / "slot.json", {"result_hash": "result"})
    row = {**rows[0], "metadata": {"evidence": str(directory), "customer_role_review_hash": digest(proof)}}
    (root / "pilot").mkdir()
    (root / "pilot/sft.jsonl").write_text(json.dumps(row) + "\n")
    assert module.audit_export_roles(root, "pilot")["status"] == "PASS"
    write_json(directory / "customer-role-review.json", {**proof, "disposition": "QUARANTINE"})
    with pytest.raises(ValueError, match="bound dual-model"):
        module.audit_export_roles(root, "pilot")
