"""Freeze exact official operation contracts and measure admitted business coverage."""

import inspect
from collections import Counter

from tau3.synthesis.storage import digest
from tau3.synthesis.targeted.native.models import OperationCoverageContract
from tau3.synthesis.targeted.planning import unpack

CREDIT_CORE = {
    "submit_credit_limit_increase_request_7392", "get_credit_limit_increase_history_4829",
    "get_payment_history_6183", "approve_credit_limit_increase_5847",
    "deny_credit_limit_increase_5848",
}


def official_contract():
    """Read names/signatures only; official customer states and answers are not copied."""
    from tau3.domains.banking_knowledge.environment import get_tasks
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    tasks = get_tasks()
    if len(tasks) != 97:
        raise ValueError("Operation contract requires the installed 97-task official set")
    records = {}
    for task in tasks:
        for action in task.evaluation_criteria.actions:
            if action.name in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}:
                continue
            name, _ = unpack(action.name, action.arguments)
            owner = KnowledgeUserTools if action.requestor == "user" else KnowledgeTools
            method = getattr(owner, name, None)
            if method is None:
                raise ValueError(f"Official action missing from runtime: {name}")
            record = records.setdefault(name, {"signature": str(inspect.signature(method)),
                                               "actor": action.requestor, "official_task_ids": []})
            if task.id not in record["official_task_ids"]:
                record["official_task_ids"].append(task.id)
    return OperationCoverageContract(
        official_tasks_hash=digest([t.model_dump(mode="json") for t in tasks]),
        operations=dict(sorted(records.items())),
    )


def coverage_report(candidates, contract, stage):
    """Count tasks, not repeated calls; a coverage label without an action earns zero."""
    counts, branches, matrices = Counter(), Counter(), {}
    multi, cross = 0, 0
    for candidate in candidates:
        operations = {unpack(a.name, a.arguments)[0] for a in candidate.task.evaluation_criteria.actions
                      if a.name not in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}}
        counts.update(operations)
        branch = candidate.slot.branch_id or candidate.slot.family
        branches[branch] += 1
        matrices[candidate.task.id] = sorted(operations)
        data = candidate.task.initial_state.initialization_data.agent_data
        multi += any(len(data.get(table, {}).get("data", {})) >= 2 for table in
                     ("accounts", "credit_card_accounts", "debit_cards", "credit_card_transaction_history", "bank_account_transaction_history"))
        # Explicit dependency edges, not the presence of several unrelated tables.
        cross += bool(candidate.graph.get("dependencies"))
    minima = {}
    for name in contract.operations:
        if stage == "pilot":
            minimum = 0
        elif name in CREDIT_CORE:
            minimum = contract.credit_full_minimum if stage == "full" else contract.credit_small_minimum
        else:
            minimum = contract.full_minimum if stage == "full" else contract.small_minimum
        minima[name] = minimum
    gaps = {name: minimum - counts[name] for name, minimum in minima.items() if counts[name] < minimum}
    size = len(candidates)
    multi_fraction, cross_fraction = multi / size if size else 0, cross / size if size else 0
    checks = {"operation_minima": not gaps,
              "multi_object": stage == "pilot" or multi_fraction >= contract.minimum_multi_object_fraction,
              "cross_entity": stage == "pilot" or cross_fraction >= contract.minimum_cross_entity_fraction}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "exact_operation_counts": dict(counts), "minima": minima, "gaps": gaps,
            "branch_counts": dict(branches), "task_operations": matrices,
            "multi_object_fraction": multi_fraction, "cross_entity_fraction": cross_fraction,
            "contract_hash": digest(contract.model_dump(mode="json"))}


def pair_report(candidates, slots):
    """Require paired business states to differ in exactly one primitive condition."""
    pairs = {}
    for slot in slots:
        if slot.pair_id:
            pairs.setdefault(slot.pair_id, []).append(slot.index)
    by_index = {c.slot.index: c for c in candidates}
    checks = {}

    def flatten(value, prefix=()):
        if isinstance(value, dict):
            return {path: leaf for key, child in value.items() for path, leaf in flatten(child, (*prefix, key)).items()}
        return {prefix: value}

    for pid, indices in pairs.items():
        if len(indices) != 2 or any(index not in by_index for index in indices):
            checks[pid] = {"valid": False, "reason": "incomplete_pair", "slots": indices}
            continue
        left, right = (by_index[index] for index in indices)
        a, b = (flatten(c.task.initial_state.initialization_data.agent_data) for c in (left, right))
        differences = [list(key) for key in a.keys() | b.keys() if a.get(key) != b.get(key)]
        valid = (len(differences) == 1 and differences[0][-1] == "current_balance"
                 and left.group_id == right.group_id and left.slot.branch_id != right.slot.branch_id)
        checks[pid] = {"valid": valid, "changed_fields": differences, "slots": indices,
                       "fingerprints": [left.business_fingerprint, right.business_fingerprint]}
    credit = sum(s.family == "credit_limit" for s in slots)
    fraction = sum(len(indices) for indices in pairs.values()) / credit if credit else 1
    # Pilot has five credit tasks and must cover mixed/no-submit as well.
    required = 0 if len(slots) == 20 else 0.5
    return {"status": "PASS" if all(c["valid"] for c in checks.values()) and fraction >= required else "FAIL",
            "credit_paired_fraction": fraction, "required_fraction": required, "pairs": checks}
