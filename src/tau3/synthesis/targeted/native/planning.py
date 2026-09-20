"""Deterministic marginal quotas and disjoint, immutable collection stages."""

import random
from collections import Counter
from functools import lru_cache

from tau3.synthesis.storage import digest
from tau3.synthesis.targeted.models import DIFFICULTIES
from tau3.synthesis.targeted.native.models import (
    FAMILIES,
    FAMILY_LABELS,
    NativePlan,
    NativeSlot,
)


def allocation(total, weights):
    """Largest-remainder allocation with stable key-order tie breaking."""
    denominator = sum(weights.values())
    counts = {key: total * weight // denominator for key, weight in weights.items()}
    ranked = sorted(weights, key=lambda key: -(total * weights[key] % denominator))
    for key in ranked[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _spread(counts, rng):
    items = [key for key, count in counts.items() for _ in range(count)]
    rng.shuffle(items)
    return items


@lru_cache(maxsize=2)
def feasible_buckets(split):
    """Certify compiler support before assigning quotas, without teacher outcomes."""
    from tau3.synthesis.targeted.native.scenarios import compile_candidate

    supported = set()
    for family in FAMILIES:
        for difficulty in DIFFICULTIES:
            slot = NativeSlot(index=0, split=split, family=family, difficulty=difficulty,
                              origin="current", seed=43000, trial_seeds=[1, 2, 3, 4], evidence_ids=["compiler-probe"])
            try:
                compile_candidate(slot)
            except ValueError:
                continue
            supported.add((family, difficulty))
    return supported


def joint_allocation(families, difficulties, supported, rng):
    """Integer max flow preserves both marginals and never assigns an unsupported cell."""
    residual, adjacency = {}, {}
    def edge(a, b, capacity):
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        residual[a, b] = capacity
        residual[b, a] = 0
    for family, n in families.items():
        edge("source", "f:" + family, n)
    pairs = sorted(supported)
    rng.shuffle(pairs)
    for family, difficulty in pairs:
        edge("f:" + family, "d:" + difficulty, sum(families.values()))
    for difficulty, n in difficulties.items():
        edge("d:" + difficulty, "sink", n)
    total = 0
    while True:
        queue, parents = ["source"], {"source": None}
        for node in queue:
            for target in adjacency[node]:
                if target not in parents and residual[node, target] > 0:
                    parents[target] = node
                    queue.append(target)
            if "sink" in parents:
                break
        if "sink" not in parents:
            break
        node, amount = "sink", sum(families.values())
        while parents[node] is not None:
            amount = min(amount, residual[parents[node], node])
            node = parents[node]
        node = "sink"
        while parents[node] is not None:
            previous = parents[node]
            residual[previous, node] -= amount
            residual[node, previous] += amount
            node = previous
        total += amount
    if total != sum(families.values()):
        raise ValueError("Registered graphs cannot satisfy frozen quota marginals")
    result = [(f, d) for f, d in pairs for _ in range(residual["d:" + d, "f:" + f])]
    rng.shuffle(result)
    return result


def compile_plan(config, profile, snapshot_hash, parent_hash=None):
    """Freeze all marginals before sampling; the first 400 have exact quotas."""
    if config.curriculum == "v03_r10":
        from tau3.synthesis.targeted.native.v03_planning import compile_v03_plan

        return compile_v03_plan(config, profile, snapshot_hash, parent_hash)
    rng = random.Random(config.seed)
    findings = {}
    mechanism_evidence = {}
    for index, finding in enumerate(profile.findings):
        if not finding.quarantined:
            findings.setdefault(finding.label, []).append(f"finding:{index}")
            for mechanism in finding.mechanisms:
                if mechanism in FAMILIES:
                    mechanism_evidence.setdefault(mechanism, []).append(f"finding:{index}")
    if not findings:
        raise ValueError("No non-quarantined failure evidence")

    def make(total, split, offset=0):
        family_counts = allocation(total, FAMILIES)
        difficulty_counts = allocation(total, DIFFICULTIES)
        if split == "pilot":
            # Eight genuinely long tasks, independently of whether teachers solve them.
            difficulty_counts = {"1-4": 3, "5-9": 5, "10-14": 4, "15-19": 4, "20-29": 4}
        pairs = joint_allocation(family_counts, difficulty_counts,
                                 feasible_buckets("validation" if split == "validation" else "pilot"), rng)
        origins = _spread(allocation(total, {"current": 70, "replay": 20, "explore": 10}), rng)
        result = []
        for local, ((family, difficulty), origin) in enumerate(zip(pairs, origins, strict=True)):
            index = offset + local
            seed = int(digest([config.seed, split, index])[:8], 16)
            evidence = sorted({e for label in FAMILY_LABELS[family] for e in findings.get(label, [])})
            evidence = sorted(set(evidence) | set(mechanism_evidence.get(family, [])))
            if not evidence:
                raise ValueError(f"No evidence for native family {family}")
            result.append(NativeSlot(
                index=index, split=split, family=family, difficulty=difficulty,
                origin=origin, seed=seed,
                trial_seeds=[int(digest([seed, "teacher", i])[:8], 16) for i in range(4)],
                evidence_ids=evidence,
            ))
        return result

    small = make(config.small_tasks, "train")
    rest = make(config.num_tasks - config.small_tasks, "train", config.small_tasks)
    pilot = make(20, "pilot")
    optimized = [s for s in pilot if s.family == "optimization"]
    for slot, mechanism in zip(optimized, ("referral", "savings_correction", "card_selection"), strict=True):
        slot.mechanism = mechanism
    return NativePlan(
        profile_hash=digest(profile.model_dump(mode="json")), snapshot_hash=snapshot_hash,
        parent_hash=parent_hash, slots=small + rest, pilot=pilot,
        validation=make(200, "validation"),
        proposal={"approved_recipe": "official-native-20260918", "family_quotas": FAMILIES,
                  "difficulty_quotas": DIFFICULTIES, "origin_quotas": {"current": 70, "replay": 20, "explore": 10},
                  "small_family_counts": dict(Counter(s.family for s in small))},
    )


def stage_slots(plan, stage, split="train"):
    """Return stable slot identities; full includes already completed small slots."""
    if split == "validation":
        return plan.validation
    if stage == "pilot":
        return plan.pilot
    if stage == "small":
        return plan.slots[:400]
    if stage == "full":
        return plan.slots
    expansion = plan.proposal.get("expansion", {})
    if stage in expansion.get("stages", {}):
        return plan.slots[:expansion["stages"][stage]]
    raise ValueError(f"Unregistered native stage: {stage}")
