"""Deterministic business compilers and independent, mechanism-specific checks.

These are fictional case-processing contracts, not claims about real bank law.
Generation and the Decimal/date oracle intentionally use different code paths.
"""

import json
from datetime import date, timedelta
from decimal import Decimal

from tau3.synthesis.targeted.models import RECIPES
from tau3.worldgen.v2.documents import build_articles
from tau3.worldgen.v2.pipeline import initial_database, task_payload, validate_scenario
from tau3.worldgen.v2.runtime import OperationRuntime, digest, goal_errors
from tau3.worldgen.v2.specs import WorldSpec

STRING = {"type": "string"}
MONEY = {"type": "decimal", "scale": 2, "minimum": "0"}
INTEGER = {"type": "integer", "minimum": "0"}
BOOL = {"type": "boolean"}
COUNTS = {"1-4": 1, "5-9": 2, "10-14": 3, "15-19": 4, "20-29": 7}
PUBLIC_SERVICES = {
    "discovery": "transaction_review",
    "policy": "benefit_adjustment",
    "handoff": "customer_claim",
    "restraint": "statement_correction",
    "escalation": "specialist_triage",
    "signature": "card_service",
    "boundary": "receipt_review",
}


def expected(row, recipe, threshold, rate, bonus, clock):
    """Independent literal oracle: no expression interpreter or runtime replay."""
    age = (date.fromisoformat(clock) - date.fromisoformat(row["reported_on"])).days
    liability = Decimal("25") if age <= threshold else Decimal("250")
    amount = (Decimal(row["gross"]) * Decimal(rate) / 100 + Decimal(bonus)).quantize(
        Decimal(".01")
    )
    route = "self_service"
    if recipe == "escalation":
        route = {"ownership": "ownership_review", "security": "security_review"}.get(
            row["issue"], "self_service"
        )
    elif recipe == "boundary":
        route = (
            "awaiting_information" if not row["receipt_available"] else "self_service"
        )
    return {
        "processed": True,
        "adjustment": str(amount),
        "liability": str(liability),
        "route": route,
        "delivery": "PRIORITY" if row["premium"] else "STANDARD",
    }


def compile_world(slot, namespace, attempt=0):
    """Instantiate a distinct business state, preserving the frozen difficulty."""
    if slot.recipe not in RECIPES or (slot.secondary and slot.secondary not in RECIPES):
        raise ValueError("Unknown recipe")
    prefix = "case_" + digest([namespace, slot.index, attempt])[:12]
    uid, pid = prefix + "_customer", prefix + "_product"
    clock = "2025-11-14"
    count, deep = COUNTS[slot.difficulty], slot.difficulty not in {"1-4", "5-9"}
    naccounts = 2 if count >= 4 else 1
    variant_seed = slot.seed + attempt * 104729
    threshold = 2 + (variant_seed // 2) % 4
    rate, bonus = str(1 + (variant_seed // 2) % 4), str((variant_seed // 2) % 3)
    tables = {name: prefix + "_" + name for name in ("accounts", "cards", "cases")}
    common = {"user_id": STRING, "product_id": STRING}
    fields = {
        **common,
        "account_id": STRING,
        "card_id": STRING,
        "eligible": BOOL,
        "gross": MONEY,
        "reported_on": {"type": "date"},
        "premium": BOOL,
        "receipt_available": BOOL,
        "issue": STRING,
        "processed": BOOL,
        "adjustment": MONEY,
        "liability": MONEY,
        "route": STRING,
        "delivery": STRING,
        "service_kind": STRING,
    }
    entities = [
        {"id": tables["accounts"], "fields": {**common, "name": STRING}},
        {
            "id": tables["cards"],
            "fields": {**common, "account_id": STRING},
            "foreign_keys": [{"field": "account_id", "table": tables["accounts"]}],
        },
        {
            "id": tables["cases"],
            "fields": fields,
            "foreign_keys": [
                {"field": "account_id", "table": tables["accounts"]},
                {"field": "card_id", "table": tables["cards"]},
            ],
        },
    ]
    rows = {t: {} for t in tables.values()}
    accounts, cards = [], []
    for i in range(naccounts):
        aid, cid = prefix + f"_account_{i}", prefix + f"_card_{i}"
        accounts.append(aid)
        cards.append(cid)
        rows[tables["accounts"]][aid] = {
            "user_id": uid,
            "product_id": pid,
            "name": f"Service account {i}",
        }
        rows[tables["cards"]][cid] = {
            "user_id": uid,
            "product_id": pid,
            "account_id": aid,
        }
    recipes = [
        slot.recipe if not slot.secondary or i % 2 == 0 else slot.secondary
        for i in range(count)
    ]
    # For a short exploratory task, combine a second mechanism on the same case.
    # Every task also exercises policy arithmetic, exact signatures and discovery.
    if slot.secondary and count == 1:
        recipes[0] = (
            slot.secondary
            if slot.recipe in {"discovery", "policy", "signature"}
            else slot.recipe
        )
    for i in range(count + 1):
        rid = prefix + f"_record_{i}"
        recipe = recipes[min(i, count - 1)]
        # Adjacent seeds straddle the policy date boundary with otherwise equal facts.
        age = threshold + (variant_seed % 2)
        rows[tables["cases"]][rid] = {
            "user_id": uid,
            "product_id": pid,
            "account_id": accounts[i % naccounts],
            "card_id": cards[i % naccounts],
            "eligible": i < count,
            "gross": str(100 + (variant_seed // 2) % 997 + i * 17),
            "reported_on": (
                date.fromisoformat(clock) - timedelta(days=age)
            ).isoformat(),
            "premium": (i + variant_seed // 2) % 2 == 0,
            "receipt_available": recipe != "boundary"
            or (i + variant_seed // 2) % 2 == 0,
            "issue": "ownership"
            if (variant_seed // 2 + i) % 3 == 0
            else "security"
            if (variant_seed // 2 + i) % 3 == 1
            else "billing",
            "processed": i == count,
            "adjustment": "0",
            "liability": "0",
            "route": "unreviewed",
            "delivery": "STANDARD",
            "service_kind": PUBLIC_SERVICES[recipe],
        }
    facts = [
        {
            "id": "report_window",
            "label": "Timely reporting calendar-day window",
            "scalar": INTEGER,
            "value": threshold,
        },
        {
            "id": "rate",
            "label": "Case adjustment percentage",
            "scalar": MONEY,
            "value": rate,
        },
        {
            "id": "bonus",
            "label": "Case adjustment additive amount",
            "scalar": MONEY,
            "value": bonus,
        },
    ]
    policies, operations = [], []
    params = {
        **common,
        "record_id": STRING,
        "amount": MONEY,
        "liability": MONEY,
        "route": {
            "type": "string",
            "choices": [
                "self_service",
                "ownership_review",
                "security_review",
                "awaiting_information",
            ],
        },
        "delivery": {"type": "string", "choices": ["STANDARD", "PRIORITY"]},
    }
    for recipe in sorted(
        set(recipes) | {slot.recipe} | ({slot.secondary} if slot.secondary else set())
    ):
        route_expr = "'self_service'"
        if recipe == "escalation":
            route_expr = "'ownership_review' if record.issue == 'ownership' else ('security_review' if record.issue == 'security' else 'self_service')"
        elif recipe == "boundary":
            route_expr = (
                "'self_service' if record.receipt_available else 'awaiting_information'"
            )
        service = PUBLIC_SERVICES[recipe]
        rule_id = prefix + "_" + service + "_rule"
        predicate = (
            f"record.service_kind == '{service}' and "
            "record.eligible and not record.processed and "
            "args.amount == money(record.gross * facts.rate / 100 + facts.bonus) and "
            "args.liability == (25 if days(clock, record.reported_on) <= facts.report_window else 250) and "
            f"args.route == ({route_expr}) and "
            "args.delivery == ('PRIORITY' if record.premium else 'STANDARD')"
        )
        policies.append(
            {
                "id": rule_id,
                "statement": "Apply the case-specific servicing policy.",
                "predicate": predicate,
                "enforcement": "trajectory",
            }
        )
        recipe_params = dict(params)
        if recipe == "signature":
            recipe_params["service_code"] = {
                "type": "string",
                "choices": ["CASE_REVIEW"],
            }
        op_id = prefix + "_resolve_" + service
        operations.append(
            {
                "id": op_id,
                "description": f"Review {service.replace('_', ' ')} cases. Record the calculated adjustment and liability as case estimates, "
                "not an actual payout or bank transfer. Mark the case reviewed and route it according to its policy. "
                "A review awaiting information or a specialist is not a final resolution. Do not reprocess completed cases.",
                "actor": "user" if recipe == "handoff" else "assistant",
                "parameters": recipe_params,
                "record_table": tables["cases"],
                "rules": [rule_id],
                "effects": [
                    {
                        "table": tables["cases"],
                        "mode": "update",
                        "key": "args.record_id",
                        "values": {
                            "processed": "True",
                            "adjustment": "args.amount",
                            "liability": "args.liability",
                            "route": "args.route",
                            "delivery": "args.delivery",
                        },
                    }
                ],
            }
        )
    # Similar name, intentionally different signature; not a solution shortcut.
    decoy = json.loads(json.dumps(operations[0]))
    decoy["id"] = prefix + "_archive_case"
    decoy["description"] = (
        "Archive a case ONLY when the customer explicitly requests archival. Does not perform a review."
    )
    decoy["parameters"] = {**common, "record_id": STRING, "archive_reason": STRING}
    decoy["rules"] = []
    decoy["actor"] = "assistant"
    decoy["effects"][0]["values"] = {"route": "'archived'"}
    operations.append(decoy)
    queries = [
        ("accounts", {}, {**common, "offset": INTEGER}),
        (
            "cards",
            {"account_id": "account_id"},
            {**common, "offset": INTEGER, "account_id": STRING},
        ),
        (
            "cases",
            {"card_id": "card_id"} if deep else {},
            {**common, "offset": INTEGER, **({"card_id": STRING} if deep else {})},
        ),
    ]
    for name, filters, query_params in queries if deep else queries[-1:]:
        entity = next(e for e in entities if e["id"] == tables[name])
        operations.append(
            {
                "id": prefix + "_list_" + name,
                "description": f"List this customer's {name} and their identifiers.",
                "kind": "read",
                "parameters": query_params,
                "query": {
                    "table": tables[name],
                    "fields": list(entity["fields"]),
                    "filters": filters,
                    "page_size": 3,
                },
            }
        )
    steps = []
    base = {"user_id": uid, "product_id": pid}
    if deep:
        steps.append(
            {"operation": prefix + "_list_accounts", "arguments": {**base, "offset": 0}}
        )
        for aid in accounts:
            steps.append(
                {
                    "operation": prefix + "_list_cards",
                    "arguments": {**base, "offset": 0, "account_id": aid},
                }
            )
    for cid in cards if deep else [None]:
        nrows = sum(
            not deep or r["card_id"] == cid for r in rows[tables["cases"]].values()
        )
        for offset in range(0, nrows, 3):
            steps.append(
                {
                    "operation": prefix + "_list_cases",
                    "arguments": {
                        **base,
                        "offset": offset,
                        **({"card_id": cid} if deep else {}),
                    },
                }
            )
    goals = []
    for i, recipe in enumerate(recipes):
        rid = prefix + f"_record_{i}"
        outcome = expected(
            rows[tables["cases"]][rid], recipe, threshold, rate, bonus, clock
        )
        arguments = {
            **base,
            "record_id": rid,
            "amount": outcome["adjustment"],
            "liability": outcome["liability"],
            "route": outcome["route"],
            "delivery": outcome["delivery"],
        }
        if recipe == "signature":
            arguments["service_code"] = "CASE_REVIEW"
        steps.append(
            {
                "operation": prefix + "_resolve_" + PUBLIC_SERVICES[recipe],
                "arguments": arguments,
            }
        )
        goals.append({"table": tables["cases"], "row_id": rid, "fields": outcome})
    # The customer requests categories of service, never a solved tool/parameter list.
    request = (
        f"Please review every eligible, not-yet-reviewed case in my {prefix} Service Plan under its documented policies. "
        "Include all my accounts and cards. Leave already reviewed and ineligible records alone. "
        "Record the policy-based estimates and required service route; do not pay or transfer money. "
        "If customer execution is required, I consent to performing it after you give me the exact instructions. "
        "Explain any need for information or specialist review without claiming the underlying issue is resolved. "
    )
    # Single-recipe tasks require finding objects in records; mixed workflows need public routing facts.
    request += "Use the service kind shown on each case record. "
    behavior = [
        "Do not invent case identifiers, bank policies or parameter values. Follow the agent's exact handed-off arguments; do not correct them from private knowledge."
    ]
    if "restraint" in recipes:
        behavior.append(
            "After the first completed review, repeat that you want all eligible cases handled; this is a reminder, not authorization to repeat an operation."
        )
    if "escalation" in recipes:
        behavior.append(
            "If asked about escalation, request the route prescribed by the documented business issue, not a generic transfer because you sound frustrated."
        )
    if "boundary" in recipes:
        behavior.append(
            "If asked for a missing receipt, say you cannot provide it now. Accept a pending-information review when the policy requires it."
        )
    category = {
        "id": prefix,
        "description": f"{prefix} customer case servicing",
        "products": [
            {"id": pid, "name": prefix + " Service Plan", "facts": facts},
            {
                "id": prefix + "_other_product",
                "name": prefix + " Business Service Plan",
                "facts": [
                    {**f, "value": str(9) if f["id"] == "rate" else f["value"]}
                    for f in facts
                ],
            },
        ],
        "entities": entities,
        "policies": policies,
        "operations": operations,
        "scenarios": [
            {
                "id": prefix + "_review",
                "kind": "service",
                "user_id": uid,
                "request": request,
                "user_behavior": behavior,
                "initial_rows": rows,
                "steps": steps,
                "goals": goals,
                "structural_family": slot.recipe,
                "communication": [
                    "The assistant accurately describes the review outcome and any outstanding information or specialist handling; estimates are not described as money paid."
                ],
            }
        ],
    }
    spec = WorldSpec.model_validate(
        {
            "purpose": "training_curriculum",
            "seed": slot.seed,
            "clock": clock,
            "categories": [category],
            "public_record_snapshot": False,
            "public_document_catalog": False,
            "split_policy": "all_train",
        }
    )
    return spec


def verify_mechanisms(spec, slot):
    """Check all targets independently and exercise valid-but-wrong operations."""
    category, scenario = spec.categories[0], spec.categories[0].scenarios[0]
    initial = initial_database(spec)
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    aliases = {op: alias for alias, op in runtime.aliases.items()}
    articles = build_articles(spec, runtime.aliases)
    generic = validate_scenario(spec, category, scenario, initial, articles)
    errors = list(generic["errors"])
    facts = {f.id: f.value for f in category.products[0].facts}
    ops = {o.id: o for o in category.operations}
    writes = [s for s in scenario.steps if ops[s.operation].kind == "write"]
    reads = [s for s in scenario.steps if ops[s.operation].kind == "read"]
    covered = set()
    for step in reads:
        result = runtime.execute(aliases[step.operation], step.arguments)
        covered.update(
            (ops[step.operation].query.table, r["record_id"]) for r in result["records"]
        )
    if runtime.db != initial:
        errors.append("Queries mutated the initial state")
    for step in writes:
        op = ops[step.operation]
        recipe = {v: k for k, v in PUBLIC_SERVICES.items()}[
            op.id.split("_resolve_")[-1]
        ]
        row = initial.tables[op.record_table][step.arguments["record_id"]]
        answer = expected(
            row,
            recipe,
            facts["report_window"],
            facts["rate"],
            facts["bonus"],
            spec.clock,
        )
        goal = next(
            g for g in scenario.goals if g.row_id == step.arguments["record_id"]
        )
        if answer != goal.fields:
            errors.append("Independent Decimal/date oracle differs from goal")
        if (op.record_table, goal.row_id) not in covered:
            errors.append("Target cannot be reached through declared public queries")
        runtime.execute(aliases[step.operation], step.arguments, op.actor)
    errors.extend(
        goal_errors(initial, runtime.db, scenario.goals, scenario.user_id, spec)
    )

    def rejects(changes):
        probe = OperationRuntime(spec, initial.model_copy(deep=True))
        try:
            for step in changes:
                probe.execute(
                    aliases[step.operation], step.arguments, ops[step.operation].actor
                )
        except ValueError:
            return True
        return bool(
            goal_errors(initial, probe.db, scenario.goals, scenario.user_id, spec)
        )

    counterexamples = {
        "empty": rejects([]),
        "missing_object": rejects(writes[:-1]),
        "duplicate_write": rejects([*writes, writes[0]]),
    }
    for field, value in {
        "amount": "0.01",
        "liability": "999",
        "route": "security_review",
        "delivery": "STANDARD",
        "invented_parameter": False,
    }.items():
        source = next((s for s in writes if s.arguments.get(field) != value), writes[0])
        if source.arguments.get(field) == value:
            value = "self_service" if field == "route" else "PRIORITY"
        wrong = source.model_copy(deep=True)
        wrong.arguments[field] = value
        counterexamples["wrong_" + field] = rejects(
            [wrong if s is source else s for s in writes]
        )
    decoy = next(o for o in category.operations if o.id.endswith("_archive_case"))
    from tau3.worldgen.v2.specs import StepSpec

    extra = StepSpec(
        operation=decoy.id,
        arguments={
            "user_id": scenario.user_id,
            "product_id": category.products[0].id,
            "record_id": writes[0].arguments["record_id"],
            "archive_reason": "unsolicited",
        },
    )
    counterexamples["extra_write"] = rejects([*writes, extra])
    if any(ops[s.operation].actor == "user" for s in writes):
        probe = OperationRuntime(spec, initial.model_copy(deep=True))
        step = next(s for s in writes if ops[s.operation].actor == "user")
        try:
            probe.execute(aliases[step.operation], step.arguments, "assistant")
            counterexamples["wrong_handoff_actor"] = False
        except ValueError:
            counterexamples["wrong_handoff_actor"] = True
    if not all(counterexamples.values()):
        errors.append("A targeted counterexample passed")
    payload = task_payload(spec, scenario, generic["gold"], runtime.aliases, initial)
    actions = len(payload["evaluation_criteria"]["actions"])
    lo, hi = map(int, slot.difficulty.split("-"))
    if not lo <= actions <= hi:
        errors.append(
            f"Reference action count {actions} outside frozen bucket {slot.difficulty}"
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counterexamples": counterexamples,
        "reference_actions": actions,
        "business_operations": len(writes),
        "public_targets": len(covered),
        "business_fingerprint": digest(
            json.dumps(
                [scenario.initial_rows, [g.model_dump() for g in scenario.goals]],
                sort_keys=True,
            ).replace(category.id, "CASE")
        ),
    }
