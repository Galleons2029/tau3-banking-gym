"""Reuse explicitly selected unchanged captures after source binding and strict replay."""

from copy import deepcopy
from pathlib import Path

from tau3.data_model.simulation import SimulationRun
from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.models import NativeCandidate
from tau3.synthesis.validation import fresh_environment

PLANNING_ONLY_CHANGES = {
    "runtime/synthesis/targeted/native/models.py",
    "runtime/synthesis/targeted/native/planning.py",
    "runtime/synthesis/targeted/native/v03_planning.py",
    "runtime/synthesis/targeted/native/expansion.py",
    "runtime/synthesis/targeted/native/gate.py",
    "runtime/synthesis/targeted/cli.py",
}

ROLE_UPGRADE_CHANGES = {
    "runtime/user/user_simulator.py", "runtime/user/role_guard.py",
    "runtime/synthesis/targeted/native/runner.py",
    "runtime/synthesis/targeted/native/quality.py",
    "runtime/synthesis/targeted/native/user_roles.py",
    "runtime/synthesis/targeted/native/retention.py",
}


def same_business_task(left, right):
    """Allow only private runtime revision and reviewed opening wording to differ."""
    if left.facts != right.facts or left.goals != right.goals:
        return False
    def normalized(candidate):
        task = candidate.task.model_dump(mode="json")
        data = task["initial_state"]["initialization_data"]["agent_data"]
        data.get("task_config", {}).get("data", {}).get("native_runtime", {}).pop("revision", None)
        task["user_scenario"]["instructions"]["reason_for_call"] = candidate.facts["goal"]
        return task
    return normalized(left) == normalized(right) and left.slot.trial_seeds == right.slot.trial_seeds


def parent_candidate(root, config, candidate, capture_relative=None):
    """Return an admitted parent only for an explicitly selected, budget-bound split."""
    enabled = set(config.reuse_parent_splits)
    if config.reuse_parent_pilot:
        enabled.add("pilot")
    if candidate.slot.split not in enabled:
        return None
    parent = Path(config.parent_round).resolve()
    origin = read_json(root / "budget-inheritance.json")
    if origin["parent"] != str(parent):
        raise ValueError("Reuse parent is not bound to cumulative budget")
    # A repair may stop in its pilot before reaching formal tasks. Recover from
    # its immutable, budget-bound ancestry instead of resampling old captures.
    seen = {root.resolve()}
    while True:
        if parent in seen:
            raise ValueError("Cycle in capture reuse ancestry")
        seen.add(parent)
        directory = parent / "tasks" / candidate.slot.split / f"{candidate.slot.index:05d}"
        complete_capture = (capture_relative is None or all(
            (parent / capture_relative / name).exists() for name in ("capture.json", "result.json")))
        if (directory / "provenance.json").exists() and complete_capture:
            break
        path = parent / "budget-inheritance.json"
        if not path.exists():
            return None
        binding = read_json(path)
        if binding.get("hash") != digest({k: v for k, v in binding.items() if k != "hash"}):
            raise ValueError("Capture ancestry binding changed")
        ancestor = Path(binding["parent"]).resolve()
        for name, filename in (("identity", "identity.json"), ("budget", "budget.json"),
                               ("blind_budget", "blind-budget.json")):
            if binding[name] != read_json(ancestor / filename):
                raise ValueError("Capture ancestor identity or budget changed")
        if read_json(ancestor / "supervisor.json")["status"] == "RUNNING":
            raise ValueError("Cannot reuse a running ancestor")
        parent = ancestor
    raw = read_json(directory / "candidate.json")
    proof = read_json(directory / "provenance.json")
    if proof["status"] != "VALID" or proof["candidate_hash"] != digest(raw):
        raise ValueError("Parent candidate evidence changed")
    old = NativeCandidate.model_validate(raw)
    if not same_business_task(old, candidate):
        return None
    parent_config = read_json(parent / "config.json")
    if parent_config["settings"] != config.settings.model_dump(mode="json"):
        return None
    previous = read_json(parent / "snapshot/manifest.json")
    current = read_json(root / "snapshot/manifest.json")
    for key in ("public_tool_schema_hash", "policy_hash", "retrieval"):
        if previous[key] != current[key]:
            return None
    allowed = ("runtime/synthesis/targeted/native/", "runtime/synthesis/targeted/budget.py",
               "runtime/domains/banking_knowledge/tools.py")
    role_changes = ROLE_UPGRADE_CHANGES if getattr(config, "user_role_guard", None) and getattr(config, "retained_sft_manifest", None) else set()
    for key in previous["hashes"].keys() | current["hashes"].keys():
        if (previous["hashes"].get(key) != current["hashes"].get(key)
                and not key.startswith(allowed) and key not in PLANNING_ONLY_CHANGES | role_changes):
            return None
    return parent, old


def reuse_narrative(root, config, candidate):
    """Retain the parent's audited opening when the business problem is unchanged."""
    found = parent_candidate(root, config, candidate)
    if found is None:
        return
    parent, old = found
    candidate.task.user_scenario.instructions.reason_for_call = old.task.user_scenario.instructions.reason_for_call
    candidate.checks["text_review"] = deepcopy(old.checks["text_review"])
    candidate.checks["narrative_origin"] = {
        "parent": str(parent), "candidate_hash": digest(old.model_dump(mode="json"))
    }


def reuse_admission_review(root, config, candidate, work, model, result):
    """Reuse a bound positive certificate only after identical public replay evidence."""
    found = parent_candidate(root, config, candidate)
    if found is None:
        return None
    parent, old = found
    previous = read_json(parent / "snapshot/manifest.json")["hashes"]
    current = read_json(root / "snapshot/manifest.json")["hashes"]
    # Generator-only edits cannot affect an identical materialized business task.
    # Public obligation checks run before reuse; checker/runtime edits still
    # require fresh independent review.
    allowed = {"runtime/synthesis/targeted/native/workflow.py", "runtime/synthesis/targeted/native/reuse.py",
               "runtime/synthesis/targeted/native/v03_scenarios.py", "runtime/synthesis/targeted/native/v03_supplements.py"}
    if config.expansion_target_rows is not None:
        allowed |= PLANNING_ONLY_CHANGES
    if getattr(config, "user_role_guard", None) and getattr(config, "retained_sft_manifest", None):
        allowed |= ROLE_UPGRADE_CHANGES
    if any(previous.get(key) != current.get(key) and key not in allowed
           for key in previous.keys() | current.keys()):
        return None
    if digest(candidate.checks.get("static")) != digest(old.checks.get("static")):
        return None
    directory = work / "blind" / digest(model)[:12]
    path = directory / "adopted-capture.json"
    if not path.exists():
        return None
    origin = read_json(path)
    source = parent / directory.relative_to(root)
    if origin["source"] != str(source) or origin.get("strict_replay") is not True:
        raise ValueError("Admission reuse lacks its bound strict replay")
    capture = read_json(directory / "capture.json")
    source_result = read_json(source / "result.json")
    if (origin["source_result_hash"] != digest(source_result)
            or origin["source_capture_hash"] != source_result["capture_hash"]
            or result["capture_hash"] != digest(capture)
            or digest({k: v for k, v in capture.items() if k != "identity"}) != origin["payload_hash"]):
        raise ValueError("Admission reuse evidence changed")
    blind = old.checks.get("blind", {}).get(model, {})
    if blind.get("result_hash") != origin["source_result_hash"]:
        raise ValueError("Admission decision is not bound to the source blind result")
    if any(result.get(key) != source_result.get(key)
           for key in ("status", "environment_success", "obligation_failures")):
        return None
    decision = old.checks.get("independent_review", {}).get(model)
    if not decision or any(decision.get(key) is not True
                           for key in ("valid", "publicly_solvable", "goals_correct", "no_leak")):
        return None
    candidate.checks.setdefault("admission_review_origins", {})[model] = {
        "parent": str(parent), "candidate_hash": digest(old.model_dump(mode="json")),
        "decision_hash": digest(decision), "source_result_hash": origin["source_result_hash"],
        "source_capture_hash": origin["source_capture_hash"], "strict_replay": True,
    }
    return deepcopy(decision)


def reuse_quality_review(root, config, candidate, directory, result, provenance):
    """Rebind unchanged quality evidence after exact capture and evaluator checks."""
    if config.expansion_target_rows is None or (directory / "quality.json").exists():
        return False
    if result["status"] != "COMPLETE" or not result["environment_success"]:
        return False
    origin_path = directory / "adopted-capture.json"
    if not origin_path.exists():
        return False
    found = parent_candidate(root, config, candidate, directory.relative_to(root))
    if found is None:
        return False
    parent, old = found
    before = read_json(parent / "snapshot/manifest.json")["hashes"]
    after = read_json(root / "snapshot/manifest.json")["hashes"]
    allowed = PLANNING_ONLY_CHANGES | {
        "runtime/synthesis/targeted/native/reuse.py",
        "runtime/synthesis/targeted/native/workflow.py",
    }
    if getattr(config, "user_role_guard", None) and getattr(config, "retained_sft_manifest", None):
        allowed |= ROLE_UPGRADE_CHANGES
    if any(before.get(k) != after.get(k) and k not in allowed for k in before.keys() | after.keys()):
        return False
    origin = read_json(origin_path)
    source = parent / directory.relative_to(root)
    if origin["source"] != str(source) or origin.get("strict_replay") is not True:
        raise ValueError("Quality reuse lacks bound strict replay")
    if not (source / "quality.json").exists():
        return False
    original = read_json(source / "result.json")
    capture = read_json(source / "capture.json")
    current_capture = read_json(directory / "capture.json")
    if (digest(original) != origin["source_result_hash"]
            or digest(capture) != origin["source_capture_hash"]
            or digest(current_capture) != result["capture_hash"]
            or {k: v for k, v in original.items() if k not in {"identity", "capture_hash"}}
            != {k: v for k, v in result.items() if k not in {"identity", "capture_hash"}}
            or {k: v for k, v in capture.items() if k != "identity"}
            != {k: v for k, v in current_capture.items() if k != "identity"}
            or old.task.user_scenario != candidate.task.user_scenario):
        return False
    from tau3.synthesis.targeted.native.quality import bind_reasoning

    source_sample = bind_reasoning(capture["sample"], capture["simulation"])
    sample = bind_reasoning(current_capture["sample"], current_capture["simulation"])
    previous = read_json(source / "quality.json")
    source_proof = read_json(source.parent.parent / "provenance.json")
    if previous["identity"] != digest([original, source_sample, source_proof]):
        raise ValueError("Parent quality evidence identity changed")
    # Negative quality decisions are preserved too; no reroll of an old refusal.
    write_json(directory / "quality.json", {
        **previous, "identity": digest([result, sample, provenance]),
        "reuse_origin": {"source": str(source / "quality.json"), "source_hash": digest(previous),
                         "strict_replay": True, "identical_evaluation": True},
    })
    return True


def reuse_capture(root, config, candidate, directory, identity):
    """Import raw successful AND failed histories; always grade/review anew."""
    found = parent_candidate(root, config, candidate, directory.relative_to(root))
    if found is None:
        return False
    parent, old = found
    if candidate.task.user_scenario != old.task.user_scenario:
        return False
    source = parent / directory.relative_to(root)
    if not (source / "capture.json").exists() or not (source / "result.json").exists():
        return False
    capture, result = read_json(source / "capture.json"), read_json(source / "result.json")
    if result["capture_hash"] != digest(capture) or read_json(source / "started.json") != capture["identity"]:
        raise ValueError("Parent capture lost its original binding")
    if any(capture["identity"][key] != identity[key] for key in ("model", "seed", "blind")):
        return False
    runtime_key = "runtime/domains/banking_knowledge/tools.py"
    before = read_json(parent / "snapshot/manifest.json")["hashes"].get(runtime_key)
    after = read_json(root / "snapshot/manifest.json")["hashes"].get(runtime_key)
    changed_operations = set()
    if before is None or before != after or old.slot.runtime_revision != candidate.slot.runtime_revision:
        changed_operations = {"order_replacement_credit_card_7291", "apply_statement_credit_8472"}
    for message in capture["simulation"]["messages"]:
        for call in message.get("tool_calls") or []:
            if (call["name"] in changed_operations or
                    call.get("arguments", {}).get("agent_tool_name") in changed_operations):
                return False  # Changed card state or transaction IDs cannot be reused.
    simulation = SimulationRun.model_validate(capture["simulation"])
    env = fresh_environment(candidate.task)
    env.set_state(initialization_data=candidate.task.initial_state.initialization_data,
                  initialization_actions=candidate.task.initial_state.initialization_actions,
                  message_history=simulation.messages, strict=True)
    imported = deepcopy(capture)
    imported["identity"] = identity
    proof = {"source": str(source), "source_capture_hash": digest(capture),
             "source_result_hash": digest(result), "source_candidate_hash": digest(old.model_dump(mode="json")),
             "source_identity": capture["identity"], "target_identity": identity,
             "strict_replay": True, "payload_hash": digest({k: v for k, v in capture.items() if k != "identity"})}
    write_json(directory / "adopted-capture.json", proof)
    write_json(directory / "capture.json", imported)
    write_json(directory / "started.json", identity)
    return True
