"""Freeze the approved v03 quotas before any teacher outcome is observed."""

import random
from collections import Counter
from functools import lru_cache

from tau3.synthesis.storage import digest
from tau3.synthesis.targeted.models import DIFFICULTIES
from tau3.synthesis.targeted.native.coverage import official_contract
from tau3.synthesis.targeted.native.models import (
    V03_FAMILIES,
    V03_LABELS,
    GatePolicy,
    NativePlan,
    NativeSlot,
)
from tau3.synthesis.targeted.native.planning import allocation, joint_allocation
from tau3.synthesis.targeted.native.scenarios import compile_candidate
from tau3.synthesis.targeted.native.v03_scenarios import DENIALS
from tau3.synthesis.targeted.planning import unpack

CHALLENGES = ["unlock_before_call", "exact_tool_name", "json_string_arguments",
              "actual_signature", "correct_entity", "no_extra_writes"]


def recipe_counts(total, pilot=False, validation=False):
    """Keep exact family counts and preserve every denial reason at formal scale."""
    families = allocation(total, V03_FAMILIES)
    if pilot:
        families = dict(zip(V03_FAMILIES, [5, 4, 3, 2, 2, 2, 1, 1], strict=True))
    recipes = {}
    supplements = {
        "transaction_disputes": {"rewards": 8},
        "replacement_closure": {"retention_credit": 4, "retention_waiver": 4},
        "accounts_funds": {"email": 4, "checking_fee": 4, "open": 4},
        "debit_security": {"activate": 4, "clear_block": 4, "temporary_limit": 4,
                           "unfreeze": 4, "replacement": 4},
        "handoff": {"referral_link": 4, "purchase": 4},
        "escalation_boundary": {"payment_incident": 4, "decline_incident": 4, "emergency": 4},
    }
    for family, size in families.items():
        if not pilot and not validation:
            for branch, small_size in supplements.get(family, {}).items():
                n = total * small_size // 400
                if n:
                    recipes[family + "/" + branch] = n
                    size -= n
        if family == "credit_limit":
            credit = allocation(size, {"approve": 250, "deny": 300, "pre_submit": 100, "mixed": 100})
            if pilot:
                credit = {"approve": 1, "deny": 2, "pre_submit": 1, "mixed": 1}
            recipes[family + "/approve"] = credit["approve"]
            paired_denials = min(credit["approve"], credit["deny"])
            recipes[family + "/deny:high_utilization"] = paired_denials
            for reason, n in allocation(credit["deny"] - paired_denials, {r: 1 for r in DENIALS if r != "high_utilization"}).items():
                if n:
                    recipes[family + "/deny:" + reason] = n
            recipes[family + "/overlimit"] = (credit["pre_submit"] + 1) // 2
            if credit["pre_submit"] // 2:
                recipes[family + "/missing_amount"] = credit["pre_submit"] // 2
            recipes[family + "/mixed"] = credit["mixed"]
        elif family == "optimization" and not pilot and not validation:
            recipes.update({family + "/" + branch: n for branch, n in allocation(size, {
                "card_selection": 3, "referral": 3, "savings_correction": 2}).items()})
        elif family in {"transaction_disputes", "replacement_closure"}:
            branches = ["credit", "debit"] if family == "transaction_disputes" else ["replace", "close"]
            recipes.update({family + "/" + branch: n for branch, n in allocation(size, dict.fromkeys(branches, 1)).items()})
        else:
            recipes[family + "/default"] = size
    return {key: n for key, n in recipes.items() if n}


@lru_cache(maxsize=512)
def supported(recipe, difficulty, split):
    """Probe deterministic structural support, never teacher success."""
    family, branch = recipe.split("/", 1)
    slot = NativeSlot(index=0, split=split, family=family, difficulty=difficulty,
                      origin="current", seed=43000, trial_seeds=[1, 2, 3, 4],
                      evidence_ids=["compiler-probe"], branch_id=None if branch == "default" else branch,
                      v03_labels=["A", "B", "C", "D", "E", "F"])
    try:
        compile_candidate(slot)
        return True
    except ValueError:
        return False


def compile_v03_plan(config, profile, snapshot_hash, parent_hash=None):
    """Build exact frozen strata with evidence IDs and an executable coverage contract."""
    rng = random.Random(config.seed)
    evidence = {family: [f"finding:{i}" for i, finding in enumerate(profile.findings)
                         if not finding.quarantined and (finding.label in V03_LABELS[family] or family in finding.mechanisms)]
                for family in V03_FAMILIES}
    if any(not value for value in evidence.values()):
        raise ValueError("The v03 curriculum has families without bound evidence")

    def make(total, split, offset=0, pilot=False, recipes_override=None):
        recipes = recipe_counts(total, pilot, validation=split == "validation") if recipes_override is None else recipes_override
        difficulties = allocation(total, DIFFICULTIES)
        if pilot:
            difficulties = {"1-4": 3, "5-9": 5, "10-14": 4, "15-19": 4, "20-29": 4}
        supported_cells = {(recipe, d) for recipe in recipes for d in DIFFICULTIES
                           if supported(recipe, d, split)}
        pairs = joint_allocation(recipes, difficulties, supported_cells, rng)
        origins = [k for k, n in allocation(total, {"current": 70, "replay": 20, "explore": 10}).items() for _ in range(n)]
        rng.shuffle(origins)
        result = []
        for local, ((recipe, difficulty), origin) in enumerate(zip(pairs, origins, strict=True)):
            family, branch = recipe.split("/", 1)
            index = offset + local
            seed = int(digest([config.seed, split, index])[:8], 16)
            result.append(NativeSlot(index=index, split=split, family=family, difficulty=difficulty,
                runtime_revision=config.runtime_revision,
                origin=origin, seed=seed, trial_seeds=[int(digest([seed, "teacher", i])[:8], 16) for i in range(4)],
                evidence_ids=evidence[family], branch_id=None if branch == "default" else branch,
                v03_labels=["A", "B", "C", "D", "E", "F"], protocol_challenges=CHALLENGES))
            if family == "optimization" and branch != "default":
                result[-1].mechanism = branch
        # Pair only equal action buckets, preserving all difficulty marginals.
        for difficulty in DIFFICULTIES:
            left = [s for s in result if s.branch_id == "approve" and s.difficulty == difficulty]
            right = [s for s in result if s.branch_id == "deny:high_utilization" and s.difficulty == difficulty]
            for control, boundary in zip(left, right):
                pair_id = digest([config.seed, split, offset, control.index, boundary.index])
                for slot, side in ((control, "control"), (boundary, "boundary")):
                    slot.pair_id, slot.pair_side = pair_id, side
                    slot.seed = control.seed
        return result

    small = make(400, "train")
    # Allocate the remainder by subtraction, avoiding rounded-credit-branch drift.
    full_counts, small_counts = recipe_counts(3000), recipe_counts(400)
    rest = make(2600, "train", 400, recipes_override={k: n - small_counts.get(k, 0) for k, n in full_counts.items()})
    pilot = make(20, "pilot", pilot=True)
    validation = make(200, "validation")
    isolation_changes = []
    validation_groups = set()
    if config.expansion_target_rows is not None:
        # Preserve the original pilot, validation and first 400 exactly. Only
        # unused training slots change; freeze every additional cohort now.
        rest = []
        for offset in range(400, 2800, 400):
            rest.extend(make(400, "train", offset))
        remainder = {k: n - 7 * small_counts.get(k, 0) for k, n in full_counts.items()}
        if any(n < 0 for n in remainder.values()) or sum(remainder.values()) != 200:
            raise ValueError("Expansion cohorts cannot preserve full recipe quotas")
        rest.extend(make(200, "train", 2800, recipes_override={k: n for k, n in remainder.items() if n}))
        # Per-mechanism holdouts can overlap the validation default mechanism.
        # Check the ACTUAL fixed validation groups for every bounded content
        # attempt. Resolve this before any model outcomes, preserving quotas.
        validation_groups = {compile_candidate(s).group_id for s in validation}
        units = {}
        for slot in rest:
            units.setdefault(slot.pair_id or str(slot.index), []).append(slot)
        for unit in units.values():
            original = {s.index: s.seed for s in unit}
            for alternative in range(101):
                if alternative:
                    seed = int(digest([config.seed, "expansion-isolation", unit[0].index, alternative])[:8], 16)
                    for slot in unit:
                        slot.seed = seed
                        slot.trial_seeds = [int(digest([seed, slot.index, "teacher", i])[:8], 16) for i in range(4)]
                if all(compile_candidate(s, attempt).group_id not in validation_groups
                       for s in unit for attempt in range(config.candidate_attempts)):
                    if alternative:
                        isolation_changes.append({"indices": [s.index for s in unit],
                                                  "old_seeds": original, "new_seed": unit[0].seed})
                    break
            else:
                raise ValueError(f"No validation-disjoint business variant for expansion slots {[s.index for s in unit]}")
    slots = small + rest
    # Bind actual emitted operations. Rebuilds must preserve them; a narrative
    # label is never accepted as proof that an operation was synthesized.
    for slot in [*pilot, *slots, *validation]:
        candidate = compile_candidate(slot)
        slot.target_operations = sorted({unpack(action.name, action.arguments)[0]
            for action in candidate.task.evaluation_criteria.actions
            if action.name not in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}})
    plan = NativePlan(profile_hash=digest(profile.model_dump(mode="json")), snapshot_hash=snapshot_hash,
        parent_hash=parent_hash, slots=slots, pilot=pilot, validation=validation,
        gate=GatePolicy(v03_checks=True), operation_coverage=official_contract(),
        branch_matrix={split: dict(Counter(s.family + "/" + (s.branch_id or "default") for s in selected))
                       for split, selected in (("pilot", pilot), ("small", small), ("full", slots), ("validation", validation))},
        pair_matrix={split: {pid: [s.index for s in selected if s.pair_id == pid]
                            for pid in sorted({s.pair_id for s in selected if s.pair_id})}
                     for split, selected in (("train", slots), ("validation", validation), ("pilot", pilot))},
        proposal={"approved_recipe": "v03_r10", "family_quotas": V03_FAMILIES,
                  "difficulty_quotas": DIFFICULTIES, "origin_quotas": {"current": 70, "replay": 20, "explore": 10},
                  "parent_round": config.parent_round, "clean_only": True,
                  "approval_runtime_revision": config.runtime_revision})
    if config.expansion_target_rows is not None:
        plan.proposal["expansion"] = {
            "version": 1, "target_unique_rows": config.expansion_target_rows,
            "minimum_handoff_tasks": 1600,
            "stages": {f"expand{n}": n for n in (800, 1200, 1600, 2000, 2400, 2800, 3000)},
            "validation_groups": sorted(validation_groups),
            "precollection_seed_rebindings": isolation_changes,
            "stop_rule": "Finish all four slots for the entire frozen cohort before checking unique qualified rows",
        }
    return plan
