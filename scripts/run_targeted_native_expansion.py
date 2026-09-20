"""Durable fixed-cohort collection until 3000 unique qualified SFT rows exist."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.expansion import audit_package, handoff
from tau3.synthesis.targeted.native.workflow import load_plan
from tau3.utils.llm_concurrency import file_lock


def audit_export_roles(root, stage):
    """Bind every exported row to its complete two-model customer-role decision."""
    from tau3.synthesis.targeted.native.user_roles import role_disposition

    config = read_json(root / "config.json")
    if not config.get("user_role_guard"):
        return {"status": "NOT_APPLICABLE"}
    models = {config["teacher_model"], config["reviewer_model"]}
    rows = 0
    with (root / stage / "sft.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            directory = Path(row["metadata"]["evidence"])
            directory.resolve().relative_to(root.resolve())
            proof = read_json(directory / "customer-role-review.json")
            slot = read_json(directory / "slot.json")
            if (digest(proof) != row["metadata"].get("customer_role_review_hash")
                    or proof["result_hash"] != slot["result_hash"]
                    or proof["disposition"] != "KEEP"
                    or {r["model"] for r in proof["reviews"]} != models
                    or role_disposition(proof["reviews"]) != "KEEP"):
                raise ValueError("Exported row lacks bound dual-model customer-role approval")
            rows += 1
    result = {"status": "PASS", "rows": rows, "user_role_guard": config["user_role_guard"]}
    write_json(root / stage / "customer-role-export-audit.json", result)
    return result


def main():
    """Keep incomplete evidence visible; never call full or fabricate a gate pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.round_dir.resolve()
    common = ["--round-dir", str(root), "--config", str(args.config.resolve()), "--resume"]
    root.mkdir(parents=True, exist_ok=True)

    def run(command):
        name = "-".join(command)
        log = root / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        state = {"pid": os.getpid(), "status": "RUNNING", "stage": command,
                 "started_at": time.time(), "log": str(log)}
        with log.open("a") as handle:
            child = subprocess.Popen(
                [sys.executable, "-m", "tau3.cli", "synthesize", "targeted", *command, *common],
                stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "LOGURU_LEVEL": "ERROR"},
            )
            write_json(root / "supervisor.json", {**state, "child_pid": child.pid})
            code = child.wait()
        write_json(root / "execution" / (name + ".json"), {**state, "exit_code": code, "finished_at": time.time()})
        return code

    def incomplete(reason):
        write_json(root / "supervisor.json", {"pid": os.getpid(), "status": "INCOMPLETE",
                   "reason": reason, "finished_at": time.time()})
        return 2

    with file_lock(root / "supervisor.lock"):
        try:
            for command in (["plan"], ["pilot"], ["generate", "--split", "validation"]):
                if run(command):
                    return incomplete(f"Unresolved prerequisite: {command}")
                if command == ["pilot"]:
                    audit_export_roles(root, "pilot")
            plan = load_plan(root)
            policy = plan.proposal.get("expansion")
            if not policy:
                return incomplete("No frozen expansion policy")
            for stage in policy["stages"]:
                # Still collect and export admitted tasks if another task has
                # an objection. No extra seed or silent task replacement.
                codes = [run([command, "--stage", stage]) for command in ("generate", "collect", "export")]
                audit = audit_package(root, plan, stage)
                audit_export_roles(root, stage)
                if any(codes) or audit["status"] == "INCOMPLETE":
                    return incomplete(f"Unresolved cohort {stage}; see stage logs and expansion-audit.json")
                if audit["status"] == "READY":
                    binding = handoff(root, plan, stage)
                    write_json(root / "supervisor.json", {
                        "pid": os.getpid(), "status": "WAITING_EXTERNAL_EVALUATION",
                        "finished_at": time.time(), "training_manifest": str(root / stage / "training_manifest.json"),
                        "unique_rows": audit["unique_rows"], "handoff": binding,
                        "validation_bundle": str(root / "validation/bundle"),
                        "next_action": "Train and evaluate the bound unique-trajectory package before large-scale expansion.",
                    })
                    return 0
            return incomplete("Frozen 3000-task ceiling reached before qualified unique-row target")
        except Exception as exc:
            incomplete(f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
