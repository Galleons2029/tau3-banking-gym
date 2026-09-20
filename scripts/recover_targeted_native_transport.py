"""Version a transport repair while preserving seeds, captures and audit history."""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.models import NativeConfig, NativePlan
from tau3.synthesis.targeted.native.quality import bind_reasoning, qualify
from tau3.synthesis.targeted.native.runner import physical_record
from tau3.synthesis.targeted.native.workflow import initialize


def recover(parent, root, resume_copy=False, config_path=None):
    """Rebind unchanged public episodes; never turn an unresolved result into success."""
    if root.exists() and not resume_copy:
        raise ValueError("Recovery destination must be new")
    if (root / "plan.json").exists() or (root / "launch.json").exists():
        raise ValueError("Cannot overwrite an already rebound recovery")
    state = read_json(parent / "supervisor.json")
    if state["status"] != "INCOMPLETE":
        raise ValueError("Parent controller must have stopped")
    old_config = NativeConfig.model_validate(read_json(parent / "config.json"))
    config = NativeConfig.model_validate(read_json(config_path)) if config_path else old_config
    assert config.settings.llm_args == old_config.settings.llm_args
    assert config.settings.simulation_max_steps == old_config.settings.simulation_max_steps
    root.mkdir(parents=True, exist_ok=resume_copy)
    manifest = initialize(root, config)
    old_manifest = read_json(parent / "snapshot/manifest.json")
    write_json(root / "evidence-parent.json", {"root": str(parent), "snapshot_hash": old_manifest["snapshot_hash"]})
    for key in ("db_snapshot_hash", "public_tool_schema_hash", "policy_hash", "retrieval"):
        assert manifest[key] == old_manifest[key], key
    with gzip.open(parent / "snapshot/sources.json.gz", "rt") as handle:
        old_sources = json.load(handle)
    with gzip.open(root / "snapshot/sources.json.gz", "rt") as handle:
        new_sources = json.load(handle)
    changed = sorted(k for k in old_sources.keys() | new_sources.keys() if old_sources.get(k) != new_sources.get(k))
    assert changed and set(changed) <= {"runtime/synthesis/targeted/budget.py", "runtime/synthesis/targeted/native/runner.py", "runtime/synthesis/targeted/native/workflow.py"}, changed
    for name in ("budget.json", "blind-budget.json", "profile.json", "report-source.json", "analysis-backlog.json", "analysis-sections.json", "implementation-checks.json"):
        if (parent / name).exists():
            shutil.copyfile(parent / name, root / name)
    created_directories = set()
    directory_lock = threading.Lock()

    def copy_file(source):
        target = root / source.relative_to(parent)
        with directory_lock:
            if target.parent not in created_directories:
                target.parent.mkdir(parents=True, exist_ok=True)
                created_directories.add(target.parent)
        # Every mutable evidence JSON is replaced atomically by the runtime.
        # Shared original inodes remain untouched; outputs are regenerated.
        try:
            os.link(source, target)
        except FileExistsError:
            if not os.path.samefile(source, target):
                raise ValueError(f"Recovery copy collision: {target}")

    partitions = [parent / "checks"]
    partitions += [Path(task.path) for split in os.scandir(parent / "tasks") if split.is_dir()
                   for task in os.scandir(split.path) if task.is_dir()]

    def scan(partition):
        paths = []
        for directory, subdirs, filenames in os.walk(partition):
            location = Path(directory)
            if "result.json" in filenames and read_json(location / "result.json")["status"] == "COMPLETE":
                subdirs[:] = [name for name in subdirs if name not in {"responses", "grading-responses"}]
            paths += [location / filename for filename in filenames if not filename.endswith((".tmp", ".lock"))]
        return paths

    with ThreadPoolExecutor(max_workers=32) as executor:
        paths = [path for group in executor.map(scan, partitions) for path in group]
    with ThreadPoolExecutor(max_workers=32) as executor:
        for index, _ in enumerate(executor.map(copy_file, paths), 1):
            if index % 1000 == 0 or index == len(paths):
                write_json(root / "recovery-progress.json", {"pid": os.getpid(),
                           "status": "COPYING_EVIDENCE", "completed": index,
                           "expected": len(paths), "updated_at": time.time()})
    write_json(root / "recovery-progress.json", {"pid": os.getpid(), "status": "REBINDING_EVIDENCE", "updated_at": time.time()})
    plan = NativePlan.model_validate(read_json(parent / "plan.json"))
    plan.parent_hash = digest(plan.model_dump(mode="json"))
    plan.snapshot_hash = manifest["snapshot_hash"]
    write_json(root / "plan.json", plan.model_dump(mode="json"))
    write_json(root / "plan-binding.json", {"hash": digest(plan.model_dump(mode="json"))})
    pilot_budget = read_json(parent / "pilot/report.json")["budget"]
    write_json(root / "pilot/budget-measurement.json", {"budget": pilot_budget,
               "budget_hash": digest(pilot_budget), "source": str(parent / "pilot/report.json")})
    records, clocks = [], []

    def rebind(directory, binding):
        marker = directory / "started.json"
        if not marker.exists():
            return
        before = read_json(marker)
        assert before["config_hash"] == digest(old_config.model_dump(mode="json"))
        after = {**before, "binding": binding, "config_hash": digest(config.model_dump(mode="json"))}
        write_json(marker, after)
        record = {"directory": str(directory.relative_to(root)), "old_identity": before,
                  "new_identity": after, "source": str(parent / directory.relative_to(root))}
        capture_path, result_path = directory / "capture.json", directory / "result.json"
        prior_elapsed = 0
        if result_path.exists() and not before["blind"]:
            old_result = read_json(result_path)
            if old_result["status"] == "INCONCLUSIVE" and old_result["simulation"]["termination_reason"] == "timeout":
                prior_elapsed = read_json(capture_path)["elapsed_seconds"]
                capture_path.replace(directory / "pre-extension-capture.json")
                result_path.replace(directory / "pre-extension-result.json")
                record["resumed_captured_timeout"] = True
        if capture_path.exists():
            capture = read_json(capture_path)
            assert capture["identity"] == before
            record["original_capture_hash"] = digest(capture)
            record["unchanged_payload_hash"] = digest({k: v for k, v in capture.items() if k != "identity"})
            capture["identity"] = after
            write_json(capture_path, capture)
            if result_path.exists():
                result = read_json(result_path)
                assert result["identity"] == before
                result.update(identity=after, capture_hash=digest(capture))
                write_json(result_path, result)
        else:
            calls = []
            for path in (directory / "responses").glob("*.json"):
                entry = read_json(path)
                for physical_id in [*entry.get("history", []), entry["physical_id"]]:
                    calls.append(physical_record(root, physical_id))
            if any(c["status"] == "RESERVED" for c in calls):
                raise ValueError("Unknown in-flight transmission cannot be resumed")
            # A versioned repair excludes the offline outage, while charging the
            # entire original observed execution interval, including failed calls.
            consumed = max(prior_elapsed, max(c["finished_at"] for c in calls) - min(c["reserved_at"] for c in calls))
            remaining = config.settings.simulation_timeout - consumed
            if remaining <= 0:
                raise ValueError("Active trajectory time budget was exhausted")
            clocks.append((directory, remaining, read_json(directory / "clock.json"), consumed))
            record["remaining_active_seconds"] = remaining
        records.append(record)

    authorized_requests = []
    for directory in sorted(root.glob("tasks/*/*")):
        # A rejected paraphrase is not a defective business instance. Restore
        # the byte-identical source request, retaining all rejected outputs.
        if directory.name == "00190" and directory.parent.name == "train":
            work = directory / "candidates/0"
            rejection = read_json(work / "invalid.json")
            if rejection["reason"] != "Narrative changed customer intent or leaked an answer":
                raise ValueError("Canonical narrative recovery evidence differs")
            candidate = read_json(work / "candidate.json")
            assert candidate["task"]["user_scenario"]["instructions"]["reason_for_call"] == candidate["facts"]["goal"]
            (work / "invalid.json").replace(work / "rejected-paraphrase.json")
            candidate["checks"]["text_review"] = {"equivalent": True, "no_answer_added": True,
                "explanation": "Canonical source text, byte-identical; rejected paraphrases retained. Full independent admission still required."}
            write_json(work / "candidate.json", candidate)
        for work in sorted((directory / "candidates").iterdir()):
            if (work / "invalid.json").exists():
                continue
            for trial in (work / "blind").glob("*"):
                if (trial / "started.json").exists():
                    binding = read_json(trial / "started.json")["binding"]
                    rebind(trial, {**binding, "snapshot": plan.snapshot_hash})
        proof_path = directory / "provenance.json"
        if not proof_path.exists():
            continue
        candidate = read_json(directory / "candidate.json")
        proof = read_json(proof_path)
        for model, blind in candidate["checks"]["blind"].items():
            matches = [p for p in directory.glob(f"candidates/*/blind/{digest(model)[:12]}/result.json")
                       if not (p.parents[2] / "invalid.json").exists()]
            assert len(matches) == 1
            blind["result_hash"] = digest(read_json(matches[0]))
        proof.update(snapshot_hash=plan.snapshot_hash, plan_hash=digest(plan.model_dump(mode="json")),
                     candidate_hash=digest(candidate), checks=candidate["checks"])
        write_json(directory / "candidate.json", candidate)
        write_json(proof_path, proof)
        for trial in (directory / "trials").glob("*"):
            if not trial.is_dir():
                continue
            rebind(trial, proof)
            if (trial / "quality.json").exists():
                from tau3.synthesis.targeted.native.models import NativeQuality

                try:
                    quality = NativeQuality.model_validate(read_json(trial / "quality.json")["review"])
                    count = sum(m["role"] == "assistant" for m in read_json(trial / "capture.json")["sample"]["messages"])
                    if any(not 0 <= i < count for i in quality.erroneous_assistant_turns):
                        raise ValueError("Invalid audit index")
                except ValueError:
                    (trial / "quality.json").replace(trial / "pre-recovery-quality.json")
            if (trial / "quality.json").exists():
                capture = read_json(trial / "capture.json")
                sample = bind_reasoning(capture["sample"], capture["simulation"])
                quality = read_json(trial / "quality.json")
                quality["identity"] = digest([read_json(trial / "result.json"), sample, proof])
                write_json(trial / "quality.json", quality)
                from tau3.synthesis.targeted.native.models import NativeCandidate

                row = qualify(root, config.model_copy(update={"training_contract": None}),
                              NativeCandidate.model_validate(candidate), trial, proof)
                if row is not None and (trial / "training-audit.json").exists():
                    audit = read_json(trial / "training-audit.json")
                    old_part = parent / trial.relative_to(root) / "sft-part.json"
                    if old_part.exists():
                        old_row = read_json(old_part)
                        assert {k: v for k, v in old_row.items() if k != "metadata"} == {k: v for k, v in row.items() if k != "metadata"}
                    audit["identity"] = digest([row, read_json(root / "training_contract.json")])
                    write_json(trial / "training-audit.json", audit)
                if (trial / "sft-part.json").exists():
                    assert row is not None
                    write_json(trial / "sft-part.json", row)
            if (trial / "slot.json").exists() and (trial / "result.json").exists():
                slot = read_json(trial / "slot.json")
                result = read_json(trial / "result.json")
                slot.update(result_hash=digest(result), environment_success=result["environment_success"])
                write_json(trial / "slot.json", slot)
    paused = []
    for directory, remaining, old_clock, consumed in clocks:
        record = {"old_clock": old_clock, "consumed_active_seconds": consumed,
                  "remaining_active_seconds": remaining, "pending": True,
                  "identity": read_json(directory / "started.json"),
                  "total_active_budget": config.settings.simulation_timeout,
                  "previous_budget": old_config.settings.simulation_timeout,
                  "policy": "Versioned extended-budget continuation; same seed and prefix; previous consumed time deducted"}
        write_json(directory / "clock-recovery.json", record)
        paused.append({"directory": str(directory.relative_to(root)), **record})
        for path in (directory / "responses").glob("*.json"):
            entry = read_json(path)
            physical = physical_record(root, entry["physical_id"])
            if physical["status"] == "INCONCLUSIVE" and physical.get("error") in {"APIError", "AuthenticationError"}:
                authorized_requests.append(physical["id"])
    probe = read_json(parent / "small/recovery-endpoint-probe.json") if authorized_requests else []
    if authorized_requests and (len(probe) != 2 or any(p["status"] != "PASS" for p in probe)):
        raise ValueError("Backend failures require successful endpoint recovery probes")
    write_json(root / "transport-recovery-authorization.json", {"physical_ids": authorized_requests,
               "health_evidence": probe, "parent": str(parent), "created_at": time.time()})
    write_json(root / "lineage.json", {"parent": str(parent), "changed_sources": changed,
               "config_unchanged": config == old_config, "previous_config": old_config.model_dump(mode="json"),
               "new_config": config.model_dump(mode="json"), "seeds_and_task_payloads_unchanged": True,
               "rebound_episodes": records, "paused_clocks": paused,
               "inherited_budget": read_json(parent / "budget.json"), "created_at": time.time()})
    command = read_json(parent / "launch.json")["command"]
    command[command.index("--round-dir") + 1] = str(root)
    if config_path:
        command[command.index("--config") + 1] = str(config_path.resolve())
        command += ["--start-index", "2"]
    with (root / "supervisor.log").open("a") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "LOGURU_LEVEL": "ERROR"})
    write_json(root / "launch.json", {"pid": process.pid, "command": command, "started_at": time.time()})
    write_json(root.parent / "active-native-round.json", {"round_dir": str(root), "pid": process.pid, "updated_at": time.time()})
    write_json(root / "recovery-progress.json", {"status": "COMPLETE", "supervisor_pid": process.pid,
               "rebound_episodes": len(records), "continued_slots": len(paused), "updated_at": time.time()})
    print(json.dumps({"status": "RESUMED", "round_dir": str(root), "pid": process.pid, "rebound_episodes": len(records), "paused_slots": len(paused)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--resume-copy", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    recover(args.parent.resolve(), args.round_dir.resolve(), args.resume_copy, args.config)
