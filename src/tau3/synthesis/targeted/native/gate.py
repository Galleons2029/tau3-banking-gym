"""Fail-closed acceptance of externally trained checkpoints and paired evaluation."""

import inspect
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native.validation import operations
from tau3.synthesis.targeted.planning import operation_kind, unpack


def operation_identity(actor, name, arguments):
    """Normalize JSON numbers and genuine signature defaults before comparing writes."""
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    owner = KnowledgeUserTools if actor == "user" else KnowledgeTools
    method = getattr(owner, name, None)
    if method is not None:
        try:
            bound = inspect.signature(method).bind(None, **arguments)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
        except TypeError:
            pass
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(v) for key, v in value.items()}
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if type(value) in {int, float}:
            return {"json_number": str(Decimal(str(value)).normalize())}
        return value
    return actor, name, digest(canonical(arguments))


def paired_statistics(baseline, candidate, policy, seed=42000):
    """Bootstrap tasks, not individual correlated trials."""
    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("Evaluation task identities differ or are empty")
    differences = []
    for task in sorted(baseline):
        left, right = baseline[task], candidate[task]
        if len(left) != policy.trials or len(right) != policy.trials or left.keys() != right.keys():
            raise ValueError("Need five complete paired seeds per task")
        differences.append(sum(right.values()) / policy.trials - sum(left.values()) / policy.trials)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                     for _ in range(policy.bootstrap_replicates))
    tail = (1 - policy.confidence) / 2
    return {"tasks": len(differences), "gain": sum(differences) / len(differences),
            "ci_low": samples[int(tail * len(samples))], "ci_high": samples[min(len(samples) - 1, int((1 - tail) * len(samples)))]}


def load_evaluation(root, expected_tasks):
    """Read actual trial files; a stale simulation_index cannot alter the denominator."""
    scores, errors = defaultdict(dict), []
    details = {}
    paths = sorted((Path(root) / "simulations").glob("*.json"))
    for path in paths:
        raw = read_json(path)
        tid = raw["task_id"]
        if tid not in expected_tasks:
            continue
        reward = raw.get("reward_info") or {}
        if reward.get("reward") is None or raw.get("termination_reason") in {"infrastructure_error", "timeout", "unexpected_error"}:
            errors.append(str(path))
            continue
        seed = str(raw.get("seed", raw.get("trial")))
        if seed in scores[tid]:
            raise ValueError("Duplicate scored trial seed")
        scores[tid][seed] = int(reward["reward"] == 1)
        details[f"{tid}:{seed}"] = {"source": str(path), "hash": digest(raw)}
    if set(scores) != set(expected_tasks) or errors:
        raise ValueError("Evaluation contains missing tasks or unresolved attempts")
    return dict(scores), details


def evaluation_metadata(directory, strict=False):
    """Compare settings recorded by the runner, excluding credentials and model identity."""
    raw = read_json(Path(directory) / "results.json")
    info = raw["info"]
    def sanitize(value):
        if isinstance(value, dict):
            excluded = {"api_key", "headers", "extra_headers"}
            if not strict:
                excluded |= {"base_url", "api_base", "reasoning_effort"}
            return {key: digest(v) if key in {"base_url", "api_base"} else sanitize(v)
                    for key, v in value.items() if key not in excluded}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value
    settings = sanitize({key: info.get(key) for key in (
        "max_steps", "max_errors", "user_info", "seed", "retrieval_config", "retrieval_config_kwargs")})
    agent = sanitize(info["agent_info"])
    agent.pop("llm", None)
    # A gateway's ineffective reasoning_effort hint does not create a new setting.
    if agent.get("llm_args", {}).get("extra_body") == {}:
        agent["llm_args"].pop("extra_body")
    settings["agent"] = agent
    return {t["id"]: t for t in raw["tasks"]}, settings


def diagnostics(tasks, scores, details):
    """Compute errors from actual successful writes, not wrapper string comparisons."""
    from tau3.synthesis.targeted.native.protocol import (
        audit_protocol,
        is_protocol_misuse,
    )

    unknown_tasks, extra_tasks, protocol_tasks, wrong_entity_tasks = set(), set(), set(), set()
    unknown_events, first_correct, first_total = 0, 0, 0
    for key, evidence in details.items():
        raw = read_json(Path(evidence["source"]))
        tid = raw["task_id"]
        protocol = audit_protocol(raw["messages"])
        protocol_issues = {"invalid_selector_or_wrapper_signature", "nested_arguments_not_string",
                           "invalid_nested_json", "nested_arguments_not_object", "call_without_prior_unlock",
                           "argument_signature", "argument_type", "hidden_tool_bypass"}
        if any(is_protocol_misuse(e) for e in protocol["errors"]):
            protocol_tasks.add(tid)
        unknown = sum(e["issue"] == "unknown_tool" for e in protocol["errors"])
        unknown_events += unknown
        if unknown:
            unknown_tasks.add(tid)
        first = next((c for c in protocol["calls"] if c["wrapper"] == "call_discoverable_agent_tool"), None)
        # Every trial stays in the denominator; avoiding all calls earns no format success.
        first_total += 1
        first_correct += bool(first and not (set(first["issues"]) & (protocol_issues | {"unknown_tool"})))
        expected = Counter()
        expected_ids = defaultdict(set)
        for action in tasks[tid]["evaluation_criteria"]["actions"]:
            name, arguments = unpack(action["name"], action["arguments"])
            actor = action.get("requestor", "assistant")
            if operation_kind(name, actor) == "write":
                expected[operation_identity(actor, name, arguments)] += 1
                for field, value in arguments.items():
                    if field.endswith("_id") and isinstance(value, str):
                        expected_ids[name, field].add(value)
        actual = Counter()
        for call in operations(raw["messages"]):
            kind = operation_kind(call["name"], call["actor"])
            if kind == "unknown":
                unknown_tasks.add(tid)
            if kind == "write" and call["succeeded"]:
                actual[operation_identity(call["actor"], call["name"], call["arguments"])] += 1
                if any((call["name"], field) in expected_ids and value not in expected_ids[call["name"], field]
                       for field, value in call["arguments"].items() if isinstance(value, str)):
                    wrong_entity_tasks.add(tid)
        if actual - expected:
            extra_tasks.add(tid)
    medium = [tid for tid in scores if 5 <= len(tasks[tid]["evaluation_criteria"]["actions"]) <= 9]
    if not medium:
        raise ValueError("Official 5-9 action bucket is empty")
    return {"unknown_tool_task_rate": len(unknown_tasks) / len(scores),
            "unknown_tool_events": unknown_events,
            "protocol_error_task_rate": len(protocol_tasks) / len(scores),
            "first_discoverable_format_success": first_correct / first_total if first_total else 0,
            "first_discoverable_trials": first_total,
            "wrong_entity_write_task_rate": len(wrong_entity_tasks) / len(scores),
            "extra_write_task_rate": len(extra_tasks) / len(scores),
            "steps_5_9": sum(sum(scores[t].values()) / len(scores[t]) for t in medium) / len(medium),
            "unknown_tool_task_ids": sorted(unknown_tasks), "extra_write_task_ids": sorted(extra_tasks)}


def accept(root, plan, receipt_path):
    """No full generation without bound training inputs, complete trials and all gates."""
    receipt = read_json(receipt_path)
    training = receipt["training"]
    required = {"base_checkpoint", "candidate_checkpoint", "data_manifest_hash", "tokenizer_hashes",
                "chat_template_hash", "adapter_hash", "max_length", "hyperparameters", "mask_verified"}
    if not required <= training.keys() or training["mask_verified"] is not True:
        raise ValueError("Incomplete training receipt or unverified token mask")
    from tau3.synthesis.targeted.native.expansion import evaluation_manifest

    manifest = evaluation_manifest(root, plan)
    if manifest["status"] != "READY" or digest(manifest) != training["data_manifest_hash"]:
        raise ValueError("External training did not bind the ready training package")
    identity = manifest["round_identity"]
    if (training["tokenizer_hashes"] != identity["tokenizer_hashes"]
            or training["chat_template_hash"] != identity["tokenizer_hashes"]["chat_template.jinja"]
            or not manifest["token_audits"]
            or training["adapter_hash"] != manifest["token_audits"][0]["adapter_hash"]
            or training["max_length"] != read_json(root / "training_contract.json")["max_length"]):
        raise ValueError("External training tokenizer, adapter or sequence contract differs")
    checks, statistics = {}, {}
    for split, count in (("official", 97), ("validation", 200)):
        run = receipt[split]
        task_ids = run["task_ids"]
        if len(set(task_ids)) != count:
            raise ValueError(f"Expected {count} {split} tasks")
        base_tasks, base_settings = evaluation_metadata(run["baseline_dir"], strict=plan.gate.v03_checks)
        new_tasks, new_settings = evaluation_metadata(run["candidate_dir"], strict=plan.gate.v03_checks)
        if base_settings != new_settings:
            raise ValueError("Actual recorded inference settings differ")
        if any(base_tasks.get(t) != new_tasks.get(t) for t in task_ids):
            raise ValueError("Actual evaluation tasks differ")
        if split == "validation":
            frozen = read_json(root / "validation" / "index.json")
            if set(task_ids) != set(frozen["task_ids"]):
                raise ValueError("Validation task identities differ from frozen set")
            bundle = {t["id"]: t for t in read_json(root / "validation/bundle/tasks.json")}
            if any(new_tasks.get(t) != bundle[t] for t in task_ids):
                raise ValueError("Validation task content differs from admitted bundle")
        baseline, base_details = load_evaluation(run["baseline_dir"], task_ids)
        candidate, new_details = load_evaluation(run["candidate_dir"], task_ids)
        stats = paired_statistics(baseline, candidate, plan.gate)
        statistics[split] = stats
        checks[split + "_gain"] = stats["gain"] >= plan.gate.minimum_gain
        checks[split + "_confidence"] = stats["ci_low"] > 0
        if split == "official":
            protected = [t for t in baseline if sum(baseline[t].values()) >= 4]
            protected_deltas = [(sum(candidate[t].values()) - sum(baseline[t].values())) / 5 for t in protected]
            checks["protected_aggregate"] = bool(protected) and sum(protected_deltas) / len(protected) >= -plan.gate.protected_max_drop
            checks["protected_individual"] = all(sum(candidate[t].values()) / 5 > plan.gate.protected_task_floor for t in protected)
            left_stats = diagnostics(base_tasks, baseline, base_details)
            right_stats = diagnostics(new_tasks, candidate, new_details)
            statistics["diagnostics"] = {"baseline": left_stats, "candidate": right_stats,
                "source_hashes": {"baseline": digest(base_details), "candidate": digest(new_details)}}
            for key in ("steps_5_9", "unknown_tool_task_rate", "extra_write_task_rate"):
                left, right = left_stats[key], right_stats[key]
                checks[key] = right >= left if key == "steps_5_9" else right <= left
            if plan.gate.v03_checks:
                checks.update(v03_diagnostic_checks(left_stats, right_stats, baseline, candidate, plan.gate))
    record = {"receipt_hash": digest(receipt), "plan_hash": digest(plan.model_dump(mode="json")),
              "checks": checks, "statistics": statistics, "status": "PASS" if all(checks.values()) else "FAIL"}
    write_json(root / "evaluation_gate.json", record)
    return record


def v03_diagnostic_checks(left, right, baseline, candidate, policy):
    """Apply the predeclared v03 gate without silently relaxing ceiling cases."""
    cli_ids = [f"task_{i:03d}" for i in range(50, 55)]
    if not set(cli_ids) <= baseline.keys() or not set(cli_ids) <= candidate.keys():
        raise ValueError("Missing task_050–054 credit-limit gate evidence")
    def rate(values):
        return sum(values.values()) / policy.trials
    cli_gain = sum(rate(candidate[t]) - rate(baseline[t]) for t in cli_ids) / len(cli_ids)
    unknown_factor = 1 if left["unknown_tool_events"] < 10 else 1 - policy.unknown_tool_reduction
    return {
        "credit_limit_050_054_gain": cli_gain >= policy.credit_limit_gain - 1e-12,
        "task_051_floor": rate(candidate["task_051"]) >= policy.task_051_floor,
        "protocol_error_reduction": right["protocol_error_task_rate"] <= left["protocol_error_task_rate"] * (1 - policy.protocol_error_reduction) + 1e-12,
        "unknown_tool_reduction": right["unknown_tool_task_rate"] <= left["unknown_tool_task_rate"] * unknown_factor + 1e-12,
        "first_discoverable_format_gain": right["first_discoverable_format_success"] - left["first_discoverable_format_success"] >= policy.first_call_format_gain - 1e-12,
        "wrong_entity_nonincrease": right["wrong_entity_write_task_rate"] <= left["wrong_entity_write_task_rate"],
    }
