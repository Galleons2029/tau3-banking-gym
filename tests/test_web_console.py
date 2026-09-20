import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tau3.web_console.app import create_app
from tau3.web_console.jobs import JobManager
from tau3.web_console.models import (
    ArtifactRef,
    ChangeSetCreate,
    ChangeTarget,
    JobCreate,
    JobRecord,
    PatchOperation,
)
from tau3.web_console.store import (
    ArtifactStore,
    apply_patch,
    digest,
    package_name_prefix,
    reasoning_blocks,
    task_id_prefix,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(value)}\n" for value in values))


def task(task_id="task_1"):
    return {
        "id": task_id,
        "description": {"purpose": "Test task"},
        "user_scenario": {"instructions": "Ask for help"},
        "initial_state": None,
        "evaluation_criteria": {"actions": [], "reward_basis": ["DB"]},
        "required_documents": ["doc_1"],
        "user_tools": [],
    }


def simulation(task_id="task_1"):
    return {
        "id": "sim_1",
        "task_id": task_id,
        "trial": 0,
        "reward_info": {"reward": 0},
        "termination_reason": "max_steps",
        "messages": [
            {
                "role": "user",
                "turn_idx": 1,
                "content": "hello",
                "raw_data": {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "private thought",
                                "provider_specific_fields": {
                                    "reasoning_content": "private thought"
                                },
                            }
                        }
                    ]
                },
            }
        ],
    }


def make_data(root: Path):
    data = root / "data"
    write_json(
        data / "simulations" / "run-a" / "results.json",
        {
            "timestamp": "2026-01-01",
            "info": {
                "environment_info": {"domain_name": "banking_knowledge"},
                "agent_info": {"llm": "agent-model"},
                "user_info": {"llm": "user-model"},
            },
            "tasks": [task()],
            "simulations": [simulation()],
        },
    )
    bundle = data / "synthetic" / "suite" / "bundle-a"
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": 1,
            "domain": "banking_knowledge",
            "status": "published",
            "environment_hash": "test-environment",
            "requested_tasks": 1,
            "published_tasks": 1,
        },
    )
    write_json(bundle / "tasks.json", [task()])
    write_json(bundle / "split_tasks.json", {"base": ["task_1"]})
    write_json(
        bundle / "candidates" / "0.json",
        {
            "task": task(),
            "skeleton": {"family": "selection"},
            "accepted": True,
            "static_passed": True,
            "text_checked": True,
            "checks": {"ok": True},
            "trials": {
                "bm25_grep::0": {
                    "trajectory": "trajectories/task_1/validation.json",
                    "reward": 0,
                    "review": {"fact_consistent": False, "explanation": "bad"},
                }
            },
            "errors": [],
        },
    )
    write_json(bundle / "trajectories" / "task_1" / "validation.json", simulation())
    docs = data / "tau3" / "domains" / "banking_knowledge" / "documents"
    write_json(docs / "doc_1.json", {"id": "doc_1", "title": "Old", "content": "body"})
    write_json(
        data / "tau3" / "domains" / "banking_knowledge" / "db.json",
        {"users": {"data": {"user_1": {"name": "Ada"}}, "notes": ""}},
    )
    return data


def make_synthesis_round(data: Path):
    round_root = data / "synthesis" / "temporary-round"
    write_json(
        data / "synthesis" / "active-native-round.json",
        {"round_dir": str(round_root), "pid": 123},
    )
    write_json(
        round_root / "config.json",
        {
            "backend": "banking_native",
            "pilot_tasks": 1,
            "small_tasks": 4,
            "validation_tasks": 2,
        },
    )
    write_json(
        round_root / "supervisor.json",
        {"status": "INCOMPLETE", "stage": ["pilot"]},
    )
    write_json(
        round_root / "pilot" / "report.json",
        {
            "status": "INCOMPLETE",
            "expected_tasks": 1,
            "valid_tasks": 1,
            "qualified_rows": 1,
        },
    )
    task_value = task("native_pilot_00000_fixture")
    task_value["description"]["purpose"] = "Temporary credit limit sample"
    slot_root = round_root / "tasks" / "pilot" / "00000"
    write_json(slot_root / "task.json", task_value)
    write_json(
        slot_root / "provenance.json",
        {
            "task_id": task_value["id"],
            "slot": {
                "family": "credit_limit",
                "difficulty": "1-4",
                "origin": "current",
                "target_operations": ["get_account", "update_limit"],
                "protocol_challenges": ["unlock_before_call"],
            },
        },
    )
    write_json(
        slot_root / "candidate.json",
        {"task": task_value, "checks": {"reference": "PASS"}},
    )
    row_id = "row_fixture_1"
    write_json(
        round_root / "pilot" / "training_manifest.json",
        {
            "rows": 1,
            "rows_index": [
                {
                    "row_id": row_id,
                    "row_hash": "row-hash",
                    "task_id": task_value["id"],
                    "weight": 1.0,
                    "family": "credit_limit",
                    "seed": 42,
                }
            ],
            "token_audits": [
                {
                    "row_id": row_id,
                    "tokens": 120,
                    "supervised_tokens": 20,
                    "reasoning_preserved": True,
                }
            ],
        },
    )
    sft = {
        "task_id": task_value["id"],
        "schema_version": 2,
        "retrieval_config": "bm25_grep",
        "teacher_model": "teacher-model",
        "messages": [
            {"role": "system", "content": "Follow policy"},
            {"role": "user", "content": "Increase my limit"},
            {
                "role": "assistant",
                "content": "I will check first",
                "reasoning_content": "Need to verify eligibility",
            },
        ],
        "tools": [],
        "loss_mask": [0, 0, 1],
        "seed": 42,
        "metadata": {"trial": 0},
    }
    write_jsonl(round_root / "pilot" / "sft.jsonl", [sft])
    write_jsonl(
        round_root / "pilot" / "general_agent_reasoning.jsonl",
        [{"tools": [], "messages": sft["messages"]}],
    )
    write_json(
        round_root / "requests" / "ignored.json",
        {"this": "large request payload must not be indexed"},
    )
    return round_root, task_value["id"], row_id


def build_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    return ArtifactStore(make_data(tmp_path), refresh_seconds=999)


def test_reasoning_is_normalized_and_deduplicated():
    blocks = reasoning_blocks(simulation()["messages"][0])
    assert len(blocks) == 1
    assert blocks[0]["participant"] == "user"
    assert blocks[0]["text"] == "private thought"


def test_json_patch_uses_rfc6902_pointer_and_replace_rules():
    value = apply_patch(
        {"a/b": {"~key": 1}},
        [PatchOperation(op="replace", path="/a~1b/~0key", value=2)],
    )
    assert value["a/b"]["~key"] == 2
    with pytest.raises(ValueError, match="does not exist"):
        apply_patch({}, [PatchOperation(op="replace", path="/missing", value=1)])


def test_indexes_and_links_supported_artifacts(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    overview = store.overview()
    assert overview["runs"] == 1
    assert overview["bundles"] == 1
    assert overview["documents"] == 1
    bundle = store.list_bundles("", 1, 10)["items"][0]
    detail = store.task_detail(bundle["id"], "task_1")
    assert detail["trajectories"][0]["summary"]["reward"] == 0
    trajectory = store.trajectory_detail(detail["trajectories"][0]["id"])
    reasoning = trajectory["simulation"]["messages"][0]["reasoning"]
    assert reasoning[0]["available"]


def test_indexes_temporary_synthesis_rounds_and_loads_tasks_on_demand(
    tmp_path, monkeypatch
):
    data = make_data(tmp_path)
    round_root, task_id, _ = make_synthesis_round(data)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=999)

    rounds = store.list_synthesis_rounds("temporary", 1, 10)
    assert rounds["total"] == 1
    round_item = rounds["items"][0]
    assert round_item["active"] is True
    assert round_item["reports"]["pilot"]["qualified_rows"] == 1

    tasks = store.list_synthesis_tasks(
        round_item["id"], "credit", "pilot", "credit_limit", 1, 10
    )
    assert tasks["total"] == 1
    assert tasks["items"][0]["id"] == task_id
    assert tasks["items"][0]["sample_count"] == 1
    assert tasks["summary"]["training_samples"] == 1

    pending_slot = round_root / "tasks" / "pilot" / "00001"
    pending_slot.mkdir(parents=True)
    assert (
        store.list_synthesis_tasks(round_item["id"], "", "pilot", "", 1, 10)["total"]
        == 1
    )
    write_json(pending_slot / "task.json", task("native_pilot_00001_late"))
    refreshed = store.list_synthesis_tasks(round_item["id"], "", "pilot", "", 1, 10)
    assert refreshed["total"] == 2

    detail = store.synthesis_task_detail(round_item["id"], task_id)
    assert detail["candidate"]["checks"]["reference"] == "PASS"
    assert detail["samples"][0]["token_audit"]["tokens"] == 120


def test_reads_one_temporary_synthesis_jsonl_row_by_manifest(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    _, task_id, row_id = make_synthesis_round(data)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=999)
    round_id = store.list_synthesis_rounds("", 1, 10)["items"][0]["id"]

    detail = store.synthesis_sample_detail(round_id, row_id)
    assert detail["simulation"]["task_id"] == task_id
    assert detail["simulation"]["reward_info"]["reward"] is None
    assistant = next(
        message
        for message in detail["simulation"]["messages"]
        if message["role"] == "assistant"
    )
    assert assistant["training_loss"] is True
    assert assistant["reasoning"][0]["text"] == "Need to verify eligibility"
    assert detail["training_sample"]["audits"][0]["tokens"] == 120


def test_temporary_synthesis_api_rejects_unknown_logical_ids(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    _, task_id, row_id = make_synthesis_round(data)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    client = TestClient(create_app(data, refresh_seconds=999))
    deadline = time.monotonic() + 2
    while client.app.state.store.index_status()["status"] == "indexing":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    round_id = client.get("/api/web/v1/synthesis-samples/rounds").json()["items"][0][
        "id"
    ]
    tasks = client.get(
        f"/api/web/v1/synthesis-samples/rounds/{round_id}/tasks",
        params={"stage": "pilot"},
    )
    assert tasks.status_code == 200
    assert tasks.json()["items"][0]["id"] == task_id
    sample = client.get(
        f"/api/web/v1/synthesis-samples/rounds/{round_id}/rows/{row_id}"
    )
    assert sample.status_code == 200
    assert sample.json()["training_sample"]["source_kind"] == "temporary_synthesis"
    assert (
        client.get("/api/web/v1/synthesis-samples/rounds/../../etc/tasks").status_code
        == 404
    )


def fixed_diagnosis_payload(
    *, task_id, package, artifact_id, trial, has_errors=True, turn_idx=1
):
    errors = []
    if has_errors:
        errors.append(
            {
                "source": "agent",
                "turn_idx": turn_idx,
                "severity": "critical",
                "error_tags": ["incorrect_tool"],
                "reasoning": "The agent used the wrong tool.",
                "correct_behavior": "Read the account first.",
            }
        )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "package": package,
        "diagnoses": [
            {
                "artifact_id": artifact_id,
                "trial": trial,
                "summary": "Fixed diagnosis",
                "has_errors": has_errors,
                "errors": errors,
            }
        ],
    }


def test_fixed_diagnoses_are_task_scoped_and_package_isolated(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    run = next(iter(store.runs.values()))
    simulation_ref = ArtifactRef(
        kind="simulation",
        id="sim_1",
        run_id=run["id"],
        task_id="task_1",
        trial=0,
    )
    simulation_missing = store.fixed_diagnosis(simulation_ref)
    assert simulation_missing["status"] == "file_missing"

    bundle = next(iter(store.bundles.values()))
    trajectory = next(
        item
        for item in store.trajectories.values()
        if item.get("bundle_id") == bundle["id"]
    )
    trajectory_ref = ArtifactRef(
        kind="trajectory",
        id=trajectory["id"],
        bundle_id=bundle["id"],
        task_id="task_1",
        trial=trajectory["trial"],
    )
    trajectory_missing = store.fixed_diagnosis(trajectory_ref)
    assert trajectory_missing["status"] == "file_missing"
    assert simulation_missing["expected_file"] != trajectory_missing["expected_file"]

    simulation_path = store.data_dir / simulation_missing["expected_file"]
    write_json(
        simulation_path,
        fixed_diagnosis_payload(
            task_id="task_1",
            package=simulation_missing["package"],
            artifact_id="sim_1",
            trial=0,
            has_errors=False,
        ),
    )
    trajectory_path = store.data_dir / trajectory_missing["expected_file"]
    write_json(
        trajectory_path,
        fixed_diagnosis_payload(
            task_id="task_1",
            package=trajectory_missing["package"],
            artifact_id=trajectory["id"],
            trial=trajectory["trial"],
        ),
    )

    assert store.fixed_diagnosis(simulation_ref)["diagnosis"]["has_errors"] is False
    selected = store.fixed_diagnosis(trajectory_ref)
    assert selected["diagnosis"]["errors"][0]["turn_idx"] == 1


def test_fixed_diagnosis_prefers_artifact_then_unique_trial(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    run = next(iter(store.runs.values()))
    artifact = ArtifactRef(kind="simulation", id="sim_1", run_id=run["id"])
    missing = store.fixed_diagnosis(artifact)
    payload = fixed_diagnosis_payload(
        task_id="task_1",
        package=missing["package"],
        artifact_id="another-artifact",
        trial=0,
        has_errors=False,
    )
    write_json(store.data_dir / missing["expected_file"], payload)
    assert store.fixed_diagnosis(artifact)["diagnosis"]["artifact_id"] == (
        "another-artifact"
    )

    payload["diagnoses"].append(
        payload["diagnoses"][0]
        | {"artifact_id": "sim_1", "summary": "Exact artifact wins"}
    )
    write_json(store.data_dir / missing["expected_file"], payload)
    assert store.fixed_diagnosis(artifact)["diagnosis"]["summary"] == (
        "Exact artifact wins"
    )


def test_fixed_diagnosis_reports_missing_and_rejects_invalid_files(
    tmp_path, monkeypatch
):
    store = build_store(tmp_path, monkeypatch)
    run = next(iter(store.runs.values()))
    artifact = ArtifactRef(kind="simulation", id="sim_1", run_id=run["id"])
    missing = store.fixed_diagnosis(artifact)
    path = store.data_dir / missing["expected_file"]
    payload = fixed_diagnosis_payload(
        task_id="task_1",
        package=missing["package"],
        artifact_id="not-current",
        trial=9,
    )
    write_json(path, payload)
    assert store.fixed_diagnosis(artifact)["status"] == "artifact_missing"

    payload["diagnoses"][0]["errors"][0]["turn_idx"] = 999
    write_json(path, payload)
    with pytest.raises(ValueError, match="turn_idx does not exist"):
        store.fixed_diagnosis(artifact)

    payload["diagnoses"][0]["errors"][0]["turn_idx"] = 1
    payload["diagnoses"].append(payload["diagnoses"][0].copy())
    write_json(path, payload)
    with pytest.raises(ValueError, match="duplicate artifact_id"):
        store.fixed_diagnosis(artifact)

    filename = store._diagnosis_task_filename("../../outside")
    assert filename.startswith("task-")
    assert "/" not in filename


def test_fixed_diagnosis_api_accepts_only_logical_artifacts(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    client = TestClient(create_app(data, refresh_seconds=999))
    run = client.get("/api/web/v1/runs").json()["items"][0]
    response = client.post(
        "/api/web/v1/fixed-diagnoses/resolve",
        json={"kind": "simulation", "id": "sim_1", "run_id": run["id"]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "file_missing"
    invalid = client.post(
        "/api/web/v1/fixed-diagnoses/resolve",
        json={"kind": "simulation", "id": "sim_1", "run_id": "../../etc"},
    )
    assert invalid.status_code == 404


def test_normalizes_heterogeneous_synthetic_packages(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    world = data / "synthetic" / "worldgen" / "natural"
    write_json(
        world / "manifest.json",
        {
            "schema_version": 2,
            "status": "published",
            "domain": "banking_knowledge",
            "assurance": "high",
        },
    )
    write_json(world / "spec.json", {"categories": 1})
    write_json(world / "build.json", {"status": "complete"})
    world_task = task("world_task_1")
    world_task["description"]["purpose"] = "World task"
    write_json(world / "tasks" / "world_task_1.json", world_task)
    write_json(world / "splits.json", {"test": ["world_task_1"]})
    write_json(world / "validation" / "certificate.json", {"status": "PASS"})

    sft = data / "synthetic" / "exports" / "sft"
    write_json(
        sft / "manifest.json",
        {"status": "PASS", "format": "jsonl", "files": {}},
    )
    job = data / "synthetic" / "maintenance"
    write_json(job / "manifest.json", {"status": "running", "identity": {}})
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=999)

    packages = store.list_synthetic_packages("", 1, 20)
    assert {item["format"] for item in packages["items"]} == {
        "tau3_aa",
        "world_package",
    }
    assert packages["summary"]["packages"] == 2
    world_package = next(
        item for item in packages["items"] if item["format"] == "world_package"
    )
    children = store.list_synthetic_package_tasks(world_package["id"], "", 1, 20)
    assert children["items"][0]["id"] == "world_task_1"
    assert children["items"][0]["splits"] == ["test"]
    detail = store.synthetic_package_task_detail(world_package["id"], "world_task_1")
    assert detail["task"]["description"]["purpose"] == "World task"
    assert detail["validation"]["package"]["status"] == "PASS"
    assert detail["package"]["editable"] is False


def test_normalizes_targeted_round_slots_and_teacher_results(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    root = data / "synthetic" / "targeted-v1-round-000"
    write_json(root / "config.json", {"schema_version": 1, "num_tasks": 1})
    write_json(root / "plan.json", {"schema_version": 1, "slots": [{"index": 7}]})
    write_json(root / "run-state.json", {"status": "INCOMPLETE", "stage": "finished"})
    write_json(
        root / "formal" / "report.json",
        {"status": "INCOMPLETE", "valid_tasks": 1},
    )
    slot = root / "formal" / "slots" / "000007"
    write_json(
        slot / "task.json",
        {
            "task_id": "targeted_task_1",
            "candidate": 0,
            "slot": {"index": 7, "recipe": "boundary"},
            "checks": {"mechanisms": {"status": "PASS"}},
            "status": "VALID",
        },
    )
    targeted_task = task("targeted_task_1")
    write_json(
        slot / "candidate_0" / "world" / "tasks" / "targeted_task_1.json",
        targeted_task,
    )
    write_json(
        slot / "candidate_0" / "audit" / "blind" / "targeted_task_1.json",
        targeted_task,
    )
    write_json(
        slot / "trials.json",
        [
            {
                "trial": 0,
                "status": "COMPLETE",
                "environment_success": False,
                "task_id": "targeted_task_1",
            }
        ],
    )
    write_json(
        slot / "teacher" / "0" / "result.json",
        {
            "identity": {"agent_model": "teacher-model"},
            "status": "GRADED",
            "simulation": simulation("targeted_task_1"),
        },
    )
    write_json(
        slot / "teacher" / "0" / "capture.json",
        {"visible_sample": {"messages": []}},
    )
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )

    store = ArtifactStore(data, refresh_seconds=999)
    packages = store.list_synthetic_packages("targeted-v1", 1, 20)
    assert packages["total"] == 1
    package = packages["items"][0]
    assert package["format"] == "targeted_round"
    assert package["task_count"] == 1
    assert package["trajectory_count"] == 0

    trajectories = store.list_synthetic_trajectories(
        "",
        1,
        20,
        bundle_id=package["id"],
        package_prefix="targeted-v1",
    )
    assert trajectories["total"] == 1
    assert trajectories["items"][0]["task_id"] == "targeted_task_1"
    assert trajectories["items"][0]["package_prefix"] == "targeted-v1"
    assert trajectories["items"][0]["task_prefix"] == "targeted_task_1"

    children = store.list_synthetic_package_tasks(package["id"], "000007", 1, 20)
    assert children["total"] == 1
    assert (
        children["items"][0]
        | {
            "stage": "formal",
            "slot_id": "000007",
            "candidate_id": "0",
        }
        == children["items"][0]
    )
    assert children["items"][0]["trajectory_count"] == 1
    assert store.synthetic_packages[package["id"]]["trajectory_count"] == 1

    detail = store.synthetic_package_task_detail(package["id"], "targeted_task_1")
    assert detail["task"]["id"] == "targeted_task_1"
    assert detail["targeted_round"]["slot_id"] == "000007"
    assert len(detail["trajectories"]) == 1
    trajectory = store.trajectory_detail(detail["trajectories"][0]["id"])
    assert trajectory["artifact"]["package_id"] == package["id"]
    assert trajectory["simulation"]["task_id"] == "targeted_task_1"

    store.refresh(force=True)
    refreshed = store.synthetic_packages[package["id"]]
    assert refreshed["targeted_loaded"] is True
    assert len(refreshed["tasks"]) == 1
    assert len(refreshed["tasks"]["targeted_task_1"]["trajectory_ids"]) == 1


def test_indexes_training_dataset_jsonl_without_splitting_source(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    root = data / "synthetic" / "targeted-v1-data-revision-ready"
    write_json(
        root / "report.json",
        {
            "status": "PASS",
            "worlds": 1,
            "rows": 1,
            "focus_rows": 1,
            "balanced_rows": 2,
        },
    )
    write_json(root / "delivery_check.json", {"status": "PASS"})
    training_task = task("task_case_12345678_review")
    write_json(
        root / "shards" / "000000" / "world_view.json",
        {"tasks": {f"{training_task['id']}.json": training_task}},
    )

    def sft_row(trial, content):
        return {
            "schema_version": 2,
            "task_id": training_task["id"],
            "retrieval_config": "bm25_grep",
            "metadata": {"trial": trial, "teacher_model": "teacher-model"},
            "messages": [
                {"role": "system", "content": "saved system"},
                {"role": "user", "content": "request"},
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call-{trial}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"id": "1"}',
                            },
                        }
                    ],
                },
            ],
            "loss_mask": [0, 0, 1],
            "tools": [],
        }

    def general_row(content):
        return {
            "messages": [
                {"role": "user", "content": "request", "loss": False},
                {"role": "assistant", "content": content, "loss": True},
            ],
            "tools": [],
        }

    full = sft_row(0, "full answer")
    focus = sft_row(1, "focus answer")
    write_jsonl(root / "sft.jsonl", [full])
    write_jsonl(root / "general_agent.jsonl", [general_row("full answer")])
    write_jsonl(root / "focus.sft.jsonl", [focus])
    write_jsonl(root / "focus.general_agent.jsonl", [general_row("focus answer")])
    write_jsonl(root / "balanced.sft.jsonl", [full, focus])
    write_jsonl(
        root / "balanced.general_agent.jsonl",
        [general_row("full answer"), general_row("focus answer")],
    )
    write_jsonl(
        root / "sampling.jsonl",
        [
            {
                "line": 1,
                "slot": "000000",
                "source_kind": "full",
                "shard_line": 1,
                "sample_hash": "hash-full",
            },
            {
                "line": 2,
                "slot": "000000",
                "source_kind": "focus",
                "shard_line": 1,
                "sample_hash": "hash-focus",
            },
        ],
    )
    write_jsonl(
        root / "audit.jsonl",
        [
            {
                "task_id": training_task["id"],
                "trial": 0,
                "replay": {"status": "PASS"},
            }
        ],
    )
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )

    store = ArtifactStore(data, refresh_seconds=999)
    packages = store.list_synthetic_packages("targeted-v1-data-revision-ready", 1, 20)
    package = packages["items"][0]
    assert package["format"] == "training_dataset"
    assert package["task_count"] == 1
    assert package["trajectory_count"] == 4
    assert package["training_summary"] == {
        "full": 1,
        "focus": 1,
        "balanced": 2,
        "unique_tasks": 1,
    }

    samples = store.list_synthetic_trajectories(
        "", 1, 20, bundle_id=package["id"], status="training"
    )
    assert samples["total"] == 4
    assert {item["dataset_variant"] for item in samples["items"]} == {
        "full",
        "focus",
        "balanced",
    }
    assert {item["task_prefix"] for item in samples["items"]} == {"task_case_review"}

    children = store.list_synthetic_package_tasks(package["id"], "", 1, 20)
    assert children["items"][0]["sample_counts"] == {
        "full": 1,
        "focus": 1,
        "balanced": 2,
    }
    task_detail = store.synthetic_package_task_detail(
        package["id"], training_task["id"]
    )
    assert task_detail["task"]["id"] == training_task["id"]
    assert task_detail["training_dataset"]["slot_ids"] == ["000000"]

    full_sample = next(
        item for item in samples["items"] if item["dataset_variant"] == "full"
    )
    detail = store.trajectory_detail(full_sample["id"])
    assert detail["training_sample"]["dataset_variant"] == "full"
    assert detail["training_sample"]["audits"][0]["replay"]["status"] == "PASS"
    assert detail["simulation"]["messages"][1]["training_loss"] is True
    assert detail["simulation"]["messages"][1]["tool_calls"][0]["name"] == "lookup"
    assert detail["verifier_diagnostics"] is None

    store.refresh(force=True)
    assert store.synthetic_packages[package["id"]]["training_loaded"] is True
    assert detail["artifact"]["id"] in store.trajectories


def test_synthetic_package_api_uses_namespaced_logical_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    app = create_app(make_data(tmp_path), refresh_seconds=999)
    assert app.state.store._refresh_thread is not None
    app.state.store._refresh_thread.join(timeout=5)
    client = TestClient(app)

    packages = client.get("/api/web/v1/synthetic-packages").json()["items"]
    package_id = packages[0]["id"]
    tasks = client.get(f"/api/web/v1/synthetic-packages/{package_id}/tasks").json()[
        "items"
    ]
    response = client.get(
        f"/api/web/v1/synthetic-packages/{package_id}/tasks/{tasks[0]['id']}"
    )
    assert response.status_code == 200
    assert response.json()["package"]["id"] == package_id
    assert (
        client.get("/api/web/v1/synthetic-packages/not-real/tasks").status_code == 404
    )


def test_lists_synthetic_trajectories_from_lightweight_trial_index(
    tmp_path, monkeypatch
):
    store = build_store(tmp_path, monkeypatch)
    indexed = next(iter(store.trajectories.values()))
    indexed["summary"]["termination_reason"] = "max_steps"
    indexed["summary"]["model"] = "teacher-model"
    result = store.list_synthetic_trajectories("task_1", 1, 10, status="failed")

    assert result["total"] == 1
    assert result["summary"] == {
        "trajectories": 1,
        "successful": 0,
        "failed": 1,
        "errors": 0,
        "training": 0,
    }
    trajectory = result["items"][0]
    assert trajectory["bundle_name"] == "bundle-a"
    assert trajectory["task_id"] == "task_1"
    assert trajectory["candidate_id"] == "0"
    assert trajectory["trial"] == "bm25_grep::0"
    assert trajectory["review_summary"] == "bad"
    assert [
        {"id": option["id"], "name": option["name"]}
        for option in result["bundle_options"]
    ] == [{"id": trajectory["bundle_id"], "name": "bundle-a"}]
    assert result["package_groups"] == [
        {
            "prefix": "bundle",
            "packages": 1,
            "tasks": 1,
            "trajectories": 1,
        }
    ]
    assert result["task_prefixes"] == ["task_1"]
    assert "task_1" in result["filter_options"]["bundle_task"]
    assert "bm25_grep::0" in result["filter_options"]["candidate_trial"]
    assert result["filter_options"]["termination"] == ["max_steps"]
    assert result["filter_options"]["model"] == ["teacher-model"]

    filtered = store.list_synthetic_trajectories(
        "",
        1,
        10,
        bundle_task="task_1",
        candidate_trial="0 bm25_grep",
        termination="max_step",
        model="teacher",
        review_status="reviewed",
    )
    assert filtered["total"] == 1
    assert (
        store.list_synthetic_trajectories("", 1, 10, review_status="unreviewed")[
            "total"
        ]
        == 0
    )
    empty = store.list_synthetic_trajectories("does-not-exist", 1, 10)
    assert empty["total"] == 0
    assert empty["filter_options"]["model"] == ["teacher-model"]

    for index in range(300):
        store.trajectories[f"extra_{index}"] = {
            **indexed,
            "id": f"extra_{index}",
            "task_id": f"task_{index:03d}",
        }
    limited = store.list_synthetic_trajectories("", 1, 10)
    assert len(limited["filter_options"]["bundle_task"]) == 250
    assert limited["filter_options_truncated"]["bundle_task"] is True


def test_synthetic_prefixes_preserve_semantic_task_suffixes():
    assert (
        package_name_prefix("targeted-v1-round-000-independent-formal-budget")
        == "targeted-v1"
    )
    assert package_name_prefix("pilot-20260918") == "pilot"
    assert task_id_prefix("task_case_17fde1234a_review") == "task_case_review"
    assert task_id_prefix("synth_cashback_32c8ab9012") == "synth_cashback"


def test_verifier_diagnostics_explain_argument_mismatch_and_extra_call():
    expected = {
        "action_id": "transfer-1",
        "requestor": "assistant",
        "name": "make_transfer",
        "arguments": {"amount": 100, "currency": "CNY"},
        "compare_args": ["amount", "currency"],
    }
    task_value = task()
    task_value["evaluation_criteria"] = {
        "actions": [expected],
        "reward_basis": ["ACTION"],
    }
    simulation_value = simulation()
    simulation_value["reward_info"] = {
        "reward": 0,
        "action_checks": [
            {
                "action": expected,
                "action_match": False,
                "action_reward": 0,
                "tool_type": "agent",
            }
        ],
    }
    simulation_value["messages"] = [
        {
            "role": "assistant",
            "turn_idx": 2,
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "make_transfer",
                    "requestor": "assistant",
                    "arguments": {"amount": 90, "currency": "CNY"},
                }
            ],
        },
        {
            "role": "assistant",
            "turn_idx": 3,
            "tool_calls": [
                {
                    "id": "call-2",
                    "name": "look_up_account",
                    "requestor": "assistant",
                    "arguments": {"account_id": "acct-1"},
                }
            ],
        },
    ]

    diagnostics = ArtifactStore._verifier_diagnostics(simulation_value, task_value)

    action = diagnostics["actions"][0]
    assert action["status"] == "argument_mismatch"
    assert action["observed_candidates"][0]["turn_idx"] == 2
    assert action["observed_candidates"][0]["argument_diffs"] == [
        {
            "field": "amount",
            "expected": 100,
            "actual": 90,
            "expected_present": True,
            "actual_present": True,
        }
    ]
    assert diagnostics["summary"] == {
        "expected": 1,
        "matched": 0,
        "argument_mismatch": 1,
        "missing": 0,
        "not_evaluated": 0,
        "extra": 2,
    }
    assert diagnostics["failed_turns"] == [2, 3]


def test_synthetic_trajectory_api_supports_status_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    app = create_app(make_data(tmp_path), refresh_seconds=999)
    assert app.state.store._refresh_thread is not None
    app.state.store._refresh_thread.join(timeout=5)
    client = TestClient(app)

    response = client.get(
        "/api/web/v1/synthetic-trajectories",
        params={"status": "failed", "q": "bundle-a"},
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["task_id"] == "task_1"
    assert (
        client.get(
            "/api/web/v1/synthetic-trajectories", params={"status": "not-valid"}
        ).status_code
        == 422
    )


def test_background_index_does_not_block_overview(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_scan(_store):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(ArtifactStore, "_scan_runs", slow_scan)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=0.01, background_refresh=True)
    assert started.wait(timeout=1)

    before = time.monotonic()
    overview = store.overview()
    elapsed = time.monotonic() - before
    assert elapsed < 0.5
    assert overview["index"]["status"] == "indexing"

    release.set()
    assert store._refresh_thread is not None
    store._refresh_thread.join(timeout=5)
    assert store.index_status()["status"] == "ready"


def test_manifest_discovery_skips_large_non_bundle_artifacts(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    base = data / "synthetic" / "raw" / "pilot" / "slots" / "000001"
    valid = base / "candidate_0" / "bundle" / "manifest.json"
    ignored = base / "candidate_0" / "audit" / "calls" / "manifest.json"
    world = base / "candidate_0" / "world" / "manifest.json"
    write_json(valid, {"status": "draft", "domain": "banking_knowledge"})
    write_json(valid.parent / "tasks.json", [])
    write_json(ignored, {"status": "draft", "domain": "banking_knowledge"})
    write_json(world, {"status": "published", "domain": "banking_synth"})
    monkeypatch.setattr("tau3.web_console.store.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )

    store = ArtifactStore(data, refresh_seconds=999)
    discovered = store._discover_manifests(data / "synthetic")
    assert valid.resolve() in discovered
    assert ignored.resolve() not in discovered
    assert world.resolve() not in discovered


def test_directory_results_loads_trajectory_on_demand(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    run = data / "simulations" / "run-a"
    metadata = json.loads((run / "results.json").read_text())
    metadata["simulations"] = []
    metadata["simulation_index"] = [
        {
            "id": "sim_1",
            "task_id": "task_1",
            "trial": 0,
            "reward": 0,
            "termination_reason": "max_steps",
        }
    ]
    write_json(run / "results.json", metadata)
    write_json(run / "simulations" / "sim_1.json", simulation())
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=999)
    indexed = next(iter(store.runs.values()))["simulations"][0]
    detail = store.simulation_detail(next(iter(store.runs)), indexed["id"])
    assert detail["simulation"]["messages"][0]["content"] == "hello"


def test_classic_view_prefers_reviewed_results_and_matches_terminal_columns(
    tmp_path, monkeypatch
):
    data = make_data(tmp_path)
    run = data / "simulations" / "run-a"
    reviewed = json.loads((run / "results.json").read_text())
    reviewed_simulation = reviewed["simulations"][0]
    reviewed_simulation["reward_info"] = {
        "reward": 0,
        "db_check": {"db_match": False},
        "action_checks": [
            {"action_match": True, "tool_type": "read"},
            {"action_match": False, "tool_type": "read"},
            {"action_match": False, "tool_type": "write"},
        ],
    }
    reviewed_simulation["auth_classification"] = {"status": "failed"}
    reviewed_simulation["info"] = {"had_unresponsive_period": True}
    reviewed_simulation["review"] = {
        "errors": [
            {
                "source": "agent",
                "severity": "critical",
                "error_tags": ["hallucination"],
            },
            {
                "source": "user",
                "severity": "minor",
                "error_tags": ["inconsistent_behavior"],
            },
        ]
    }
    write_json(run / "results_reviewed.json", reviewed)
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    store = ArtifactStore(data, refresh_seconds=999)

    classic_runs = store.list_classic_runs()
    assert len(classic_runs) == 1
    assert classic_runs[0]["file"] == "results_reviewed.json"
    rows = store.list_classic_simulations(
        classic_runs[0]["id"], "failed", "task_1", 1, 10
    )
    assert rows["total"] == 1
    assert rows["failed_task_count"] == 1
    row = rows["items"][0]
    assert row["db_match"] is False
    assert row["read_actions"] == {"correct": 1, "count": 2}
    assert row["write_actions"] == {"correct": 0, "count": 1}
    assert row["auth_status"] == "failed"
    assert row["had_unresponsive_period"] is True
    assert row["agent_critical"] is True
    assert row["agent_tags"] == ["hallucination"]
    assert row["user_tags"] == ["inconsistent_behavior"]
    assert store.list_classic_tasks(classic_runs[0]["id"])[0]["id"] == "task_1"


def test_classic_view_api_uses_logical_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    client = TestClient(create_app(make_data(tmp_path), refresh_seconds=999))
    runs = client.get("/api/web/v1/classic-view/runs").json()["items"]
    run_id = runs[0]["id"]
    response = client.get(
        f"/api/web/v1/classic-view/runs/{run_id}/simulations",
        params={"mode": "all_failed"},
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["task_id"] == "task_1"
    assert client.get("/api/web/v1/classic-view/runs/not-real/tasks").status_code == 404


def test_document_change_set_validates_and_applies_with_backup(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    original = {"id": "doc_1", "title": "Old", "content": "body"}
    change = store.create_change_set(
        ChangeSetCreate(
            rationale="correct title",
            targets=[
                ChangeTarget(
                    artifact=ArtifactRef(kind="document", id="doc_1"),
                    base_sha256=digest(original),
                    operations=[
                        PatchOperation(op="replace", path="/title", value="New")
                    ],
                )
            ],
        )
    )
    assert store.validate_change_set(change.id).status == "validated"
    applied = store.apply_change_set(change.id)
    assert applied.status == "applied"
    assert store.document_detail("doc_1")["document"]["title"] == "New"
    assert list((store.calibration_dir / "backups" / change.id).rglob("doc_1.json"))


def test_seed_table_change_preserves_record_ids(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    table = store.db_table_detail("users")
    changed = table["value"] | {"notes": "calibrated"}
    change = store.create_change_set(
        ChangeSetCreate(
            rationale="document table semantics",
            targets=[
                ChangeTarget(
                    artifact=ArtifactRef(kind="seed_table", id="users"),
                    base_sha256=table["sha256"],
                    operations=[PatchOperation(op="replace", path="", value=changed)],
                )
            ],
        )
    )
    assert store.apply_change_set(change.id).status == "applied"
    assert store.db_table_detail("users")["value"]["notes"] == "calibrated"


def test_published_task_edit_creates_draft_fork(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    bundle = next(iter(store.bundles.values()))
    original = bundle["tasks"]["task_1"]["task"]
    change = store.create_change_set(
        ChangeSetCreate(
            rationale="calibrate task",
            targets=[
                ChangeTarget(
                    artifact=ArtifactRef(
                        kind="task",
                        id="task_1",
                        bundle_id=bundle["id"],
                        task_id="task_1",
                    ),
                    base_sha256=digest(original),
                    operations=[
                        PatchOperation(
                            op="replace",
                            path="/description/purpose",
                            value="Calibrated",
                        )
                    ],
                )
            ],
        )
    )
    applied = store.apply_change_set(change.id)
    fork = store.data_dir / applied.result["bundle"]
    assert fork != bundle["root"]
    source_manifest = json.loads((bundle["root"] / "manifest.json").read_text())
    assert source_manifest["status"] == "published"
    assert json.loads((fork / "manifest.json").read_text())["status"] == "draft"
    candidate = json.loads((fork / "candidates" / "0.json").read_text())
    assert candidate["task"]["description"]["purpose"] == "Calibrated"
    assert candidate["trials"] == {}


def test_job_commands_are_allowlisted_and_review_outputs_are_unique(
    tmp_path, monkeypatch
):
    store = build_store(tmp_path, monkeypatch)
    manager = JobManager(store)
    run_id = next(iter(store.runs))
    command_a, target, output_a = manager._command(
        JobCreate(type="results_review", run_id=run_id, task_ids=["task_1"])
    )
    command_b, _, output_b = manager._command(
        JobCreate(type="results_review", run_id=run_id)
    )
    assert command_a[:3] == [command_a[0], "-m", "tau3.cli"]
    assert "review" in command_a
    assert target == f"run:{run_id}"
    assert output_a != output_b


def test_single_trajectory_review_materializes_only_the_requested_simulation(
    tmp_path, monkeypatch
):
    store = build_store(tmp_path, monkeypatch)
    manager = JobManager(store)
    run_id = next(iter(store.runs))
    artifact = ArtifactRef(
        kind="simulation", id="sim_1", run_id=run_id, task_id="task_1"
    )
    command, target, output = manager._command(
        JobCreate(type="trajectory_review", artifact=artifact)
    )

    input_path = Path(command[4])
    snapshot = json.loads(input_path.read_text())
    assert command[3] == "review"
    assert command[command.index("--max-concurrency") + 1] == "1"
    assert target == f"trajectory-review:simulation:{run_id}:sim_1"
    assert output.startswith(str(store.calibration_dir / "reviews"))
    assert [item["id"] for item in snapshot["simulations"]] == ["sim_1"]
    assert [item["id"] for item in snapshot["tasks"]] == ["task_1"]


def test_synthetic_trajectory_review_builds_results_context(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    manager = JobManager(store)
    trajectory_id = next(iter(store.trajectories))
    artifact = ArtifactRef(
        kind="trajectory", id=trajectory_id, bundle_id=next(iter(store.bundles))
    )
    command, target, _ = manager._command(
        JobCreate(type="trajectory_review", artifact=artifact, review_mode="user")
    )

    snapshot = json.loads(Path(command[4]).read_text())
    assert target == (
        f"trajectory-review:trajectory:{artifact.bundle_id}:{trajectory_id}"
    )
    assert snapshot["simulations"][0]["trial"] == 0
    assert snapshot["tasks"][0]["id"] == "task_1"
    assert snapshot["info"]["environment_info"]["domain_name"] == ("banking_knowledge")


def test_trajectory_review_result_and_mutex_are_persisted(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    manager = JobManager(store)
    monkeypatch.setattr(manager, "_run", lambda *_: None)
    run_id = next(iter(store.runs))
    artifact = ArtifactRef(kind="simulation", id="sim_1", run_id=run_id)
    payload = JobCreate(type="trajectory_review", artifact=artifact)
    job = manager.create(payload)
    with pytest.raises(ValueError, match="writer is already active"):
        manager.create(payload)

    review = {
        "summary": "Agent used the wrong tool",
        "has_errors": True,
        "agent_error": True,
        "user_error": False,
        "errors": [
            {
                "source": "agent",
                "severity": "critical",
                "turn_idx": 2,
                "error_tags": ["irrelevant_tool_call"],
                "reasoning": "Wrong tool",
            }
        ],
        "cost": 0.01,
    }
    write_json(Path(job.output_file), {"simulations": [{"review": review}]})
    job.status = "succeeded"
    manager._save(job)
    result = manager.result(job.id)
    assert result["review"]["errors"][0]["turn_idx"] == 2
    assert manager.list_for_artifact("simulation", "sim_1", run_id=run_id)[0].id == (
        job.id
    )


def test_trajectory_review_api_filters_jobs_and_returns_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    app = create_app(make_data(tmp_path), refresh_seconds=999)
    manager = app.state.jobs
    review_file = manager.store.calibration_dir / "reviews" / "reviewed.json"
    write_json(
        review_file,
        {
            "simulations": [
                {
                    "review": {
                        "summary": "Wrong tool",
                        "has_errors": True,
                        "errors": [{"turn_idx": 2}],
                    }
                }
            ]
        },
    )
    record = JobRecord(
        id="job_review",
        type="trajectory_review",
        status="succeeded",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        params={
            "artifact": {
                "kind": "simulation",
                "id": "sim_1",
                "run_id": "run_a",
            },
            "review_mode": "full",
        },
        log_file="job_review.log",
        output_file=str(review_file),
    )
    manager._jobs[record.id] = record
    client = TestClient(app)

    jobs = client.get(
        "/api/web/v1/jobs?artifact_kind=simulation&artifact_id=sim_1&run_id=run_a"
    )
    assert jobs.status_code == 200
    assert [item["id"] for item in jobs.json()["items"]] == ["job_review"]
    other_run = client.get(
        "/api/web/v1/jobs?artifact_kind=simulation&artifact_id=sim_1&run_id=run_b"
    )
    assert other_run.json()["items"] == []
    result = client.get("/api/web/v1/jobs/job_review/result")
    assert result.status_code == 200
    assert result.json()["review"]["summary"] == "Wrong tool"


def test_restarted_running_job_becomes_orphaned(tmp_path, monkeypatch):
    store = build_store(tmp_path, monkeypatch)
    directory = store.calibration_dir / "jobs"
    record = JobRecord(
        id="job_old",
        type="check_data",
        status="running",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        log_file="job_old.log",
    )
    write_json(directory / "job_old.json", record.model_dump(mode="json"))
    assert JobManager(store).get("job_old").status == "orphaned"


def test_api_rejects_unknown_logical_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    app = create_app(make_data(tmp_path), refresh_seconds=999)
    client = TestClient(app)
    assert client.get("/api/web/v1/overview").status_code == 200
    response = client.get("/api/web/v1/trajectories/not-a-real-id")
    assert response.status_code == 404


def test_unknown_api_returns_json_instead_of_spa_html(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau3.web_console.store.environment_fingerprint", lambda: "test-environment"
    )
    client = TestClient(create_app(make_data(tmp_path), refresh_seconds=999))
    response = client.get("/api/web/v1/not-a-real-endpoint")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"].startswith("Unknown API endpoint")
