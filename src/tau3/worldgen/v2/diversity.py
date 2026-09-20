"""Structural fingerprints that do not count product/category renaming as novelty."""

import re

from tau3.worldgen.v2.runtime import digest


def structural_fingerprint(category, scenario) -> str:
    """Canonicalize workflow shape and rule expressions, excluding names and values."""
    replacements = {}
    for i, entity in enumerate(category.entities):
        replacements[entity.id] = f"table_{i}"
        for j, name in enumerate(entity.fields):
            replacements.setdefault(name, f"field_{j}")

    def expression(value):
        return re.sub(
            r"\b[a-zA-Z_][a-zA-Z_0-9]*\b", lambda m: replacements.get(m[0], m[0]), value
        )

    ops = {o.id: o for o in category.operations}
    policies = {p.id: p for p in category.policies}
    steps = []
    for step in scenario.steps:
        op = ops[step.operation]
        steps.append(
            {
                **(
                    {
                        "access": "read",
                        "query": {
                            "table": replacements[op.query.table],
                            "fields": [replacements.get(k, k) for k in op.query.fields],
                            "filters": {
                                replacements.get(k, k): v
                                for k, v in op.query.filters.items()
                            },
                        },
                    }
                    if op.query
                    else {}
                ),
                "actor": op.actor,
                "parameters": [p.type for p in op.parameters.values()],
                "rules": [
                    (expression(policies[r].predicate), policies[r].enforcement)
                    for r in op.rules
                ],
                "postconditions": [expression(p) for p in op.postconditions],
                "effects": [
                    {
                        "table": replacements[e.table],
                        "mode": e.mode,
                        "values": {
                            replacements.get(k, k): expression(v)
                            for k, v in e.values.items()
                        },
                    }
                    for e in op.effects
                ],
            }
        )
    return digest(
        {
            "kind": scenario.kind,
            "steps": steps,
            "selection": expression(scenario.selection or ""),
            "goals": [
                [replacements.get(k, k) for k in g.fields] for g in scenario.goals
            ],
            "communication": bool(scenario.communication),
        }
    )


def diversity_report(spec) -> dict:
    """Report structural duplicates explicitly, without arbitrary novelty scores."""
    families = {}
    for category in spec.categories:
        for scenario in category.scenarios:
            families.setdefault(structural_fingerprint(category, scenario), []).append(
                scenario.id
            )
    return {
        "unique_structures": len(families),
        "families": families,
        "duplicate_scenarios": sum(len(v) - 1 for v in families.values()),
    }
