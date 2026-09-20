"""Bootstrap category packages; loaded packages are ordinary data, not plugins."""

from tau3.worldgen.v2.specs import CategorySpec

PROFILES = {
    "checking": (
        "personal checking accounts",
        "withdraw",
        "withdraw funds",
        "balance",
        "balance",
    ),
    "savings": (
        "personal savings accounts",
        "withdraw",
        "withdraw savings",
        "balance",
        "balance",
    ),
    "term_deposit": (
        "fixed-term deposits",
        "redeem",
        "redeem principal",
        "principal",
        "principal",
    ),
    "credit_card": (
        "rewards credit cards",
        "pay",
        "make a card payment",
        "balance",
        "outstanding balance",
    ),
    "installment": (
        "installment payment plans",
        "repay",
        "repay principal",
        "principal",
        "remaining principal",
    ),
    "merchant": (
        "merchant settlement accounts",
        "settle",
        "settle available proceeds",
        "proceeds",
        "unsettled proceeds",
    ),
    "rewards": (
        "cash reward programmes",
        "redeem",
        "redeem earned cash rewards",
        "rewards",
        "earned rewards",
    ),
    "escrow": (
        "escrow holding accounts",
        "release",
        "release held funds",
        "held_funds",
        "held funds",
    ),
}


def scaffold(
    category_id: str, profile: str, ordinal: int = 0, clock: str = "2025-11-14"
) -> CategorySpec:
    """Create an executable starting package with several distinct workflow branches.

    This is a bootstrap, not a novelty claim. New category definitions may replace
    any table, rule, operation and scenario using the same contract language.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile; choose one of {sorted(PROFILES)}")
    description, verb, action, balance, balance_label = PROFILES[profile]
    # A copied profile must still expose an unambiguous market/category scope
    # in the user's request; private category IDs alone are insufficient.
    description = f"{category_id.replace('_', ' ').title()} {description}"
    prefix, table = category_id, f"{category_id}_records"
    string = {"type": "string"}
    amount = {"type": "decimal", "unit": "USD", "minimum": "0", "scale": 2}
    common = {"user_id": string, "product_id": string}
    parameters = {**common, "record_id": string}
    products = []
    for i, name in enumerate(["Harbor", "Meadow", "Summit"]):

        def fact(fid, label, typ, value, unit=""):
            return {
                "id": fid,
                "label": label,
                "scalar": {"type": typ, "unit": unit},
                "value": value,
            }

        facts = [
            fact(
                "monthly_fee",
                "Monthly service fee",
                "decimal",
                ["0", "5", "0"][i],
                "USD",
            ),
            fact(
                "base_rate",
                "Base annual rate",
                "decimal",
                ["3.5", "3.5", "2.5"][i],
                "percent",
            ),
            fact(
                "minimum_tenure",
                "Minimum customer tenure",
                "integer",
                14 + ordinal % 3,
                "days",
            ),
            fact("online_access", "Online access", "boolean", i != 2),
            fact("promotion_start", "Promotion start date", "date", "2025-11-01"),
            fact("promotion_end", "Promotion end date", "date", "2025-12-31"),
            fact(
                "promotion_boost",
                "Promotional annual-rate increase",
                "decimal",
                "0.5",
                "percentage points",
            ),
            {
                "id": "effective_rate",
                "label": "Effective annual rate",
                "scalar": {"type": "decimal", "unit": "percent"},
                "expression": "facts.base_rate + (facts.promotion_boost if facts.promotion_start <= clock <= facts.promotion_end else 0)",
                "depends_on": [
                    "base_rate",
                    "promotion_boost",
                    "promotion_start",
                    "promotion_end",
                ],
            },
        ]
        products.append(
            {
                "id": f"{prefix}_{name.lower()}",
                "name": f"{name} {category_id.replace('_', ' ').title()}",
                "facts": facts,
            }
        )
    product_id = products[0]["id"]
    rules = [
        {
            "id": f"{prefix}_tenure",
            "statement": "Opening requires at least {minimum_tenure} days since the customer joined the bank.",
            "predicate": "days(clock, user.joined_on) >= facts.minimum_tenure",
        },
        {
            "id": f"{prefix}_active",
            "statement": f"To {action}, the record must be OPEN and the amount must be positive and no greater than the {balance_label}.",
            "predicate": f"record.status == 'OPEN' and args.amount > 0 and args.amount <= record.{balance}",
        },
        {
            "id": f"{prefix}_confirmable",
            "statement": f"Closure confirmation requires an OPEN record with zero {balance_label}.",
            "predicate": f"record.status == 'OPEN' and record.{balance} == 0",
        },
        {
            "id": f"{prefix}_closable",
            "statement": f"Close only after the customer has used the closure-confirmation tool; the {balance_label} must be zero.",
            "predicate": f"record.status == 'CLOSURE_REQUESTED' and record.{balance} == 0",
        },
    ]
    operations = [
        {
            "id": f"{prefix}_open",
            "description": f"Open a record for {description}",
            "parameters": parameters,
            "rules": [rules[0]["id"]],
            "effects": [
                {
                    "table": table,
                    "mode": "create",
                    "key": "args.record_id",
                    "values": {
                        "user_id": "args.user_id",
                        "product_id": "args.product_id",
                        "status": "'OPEN'",
                        balance: "'0'",
                        "opened_on": "clock",
                    },
                }
            ],
        },
        {
            "id": f"{prefix}_{verb}",
            "description": action.capitalize(),
            "parameters": {**parameters, "amount": amount},
            "record_table": table,
            "rules": [rules[1]["id"]],
            "effects": [
                {
                    "table": table,
                    "mode": "update",
                    "key": "args.record_id",
                    "values": {balance: f"money(record.{balance} - args.amount)"},
                }
            ],
        },
        {
            "id": f"{prefix}_confirm_close",
            "description": f"Customer confirmation to close {description}",
            "actor": "user",
            "parameters": parameters,
            "record_table": table,
            "rules": [rules[2]["id"]],
            "effects": [
                {
                    "table": table,
                    "mode": "update",
                    "key": "args.record_id",
                    "values": {"status": "'CLOSURE_REQUESTED'"},
                }
            ],
        },
        {
            "id": f"{prefix}_close",
            "description": f"Close {description}",
            "parameters": parameters,
            "record_table": table,
            "rules": [rules[3]["id"]],
            "effects": [
                {
                    "table": table,
                    "mode": "update",
                    "key": "args.record_id",
                    "values": {"status": "'CLOSED'"},
                }
            ],
        },
    ]
    scenarios = []
    for index, kind in enumerate(
        ["selection", "service", "ordering", "denial", "grounding"]
    ):
        uid, rid = f"{prefix}_customer_{index}", f"{prefix}_record_{index}"
        args = {"user_id": uid, "product_id": product_id, "record_id": rid}
        row = {
            "user_id": uid,
            "product_id": product_id,
            "status": "OPEN",
            balance: "50.00",
            "opened_on": "2025-01-01",
        }
        scenario = {
            "id": f"{prefix}_{kind}",
            "kind": kind,
            "user_id": uid,
            "request": "",
            "structural_family": f"{profile}:{kind}",
            "required_facts": [f"{product_id}.identity"],
        }
        if kind == "selection":
            scenario.update(
                request=f"Please find and open the one product in {description} whose monthly fee is zero and whose effective annual rate is at least 4 percent. I do not know the product names. Use record identifier {rid} for my new record.",
                selection="facts.monthly_fee == 0 and facts.effective_rate >= 4",
                expected_product=product_id,
                steps=[{"operation": f"{prefix}_open", "arguments": args}],
                goals=[
                    {
                        "table": table,
                        "row_id": rid,
                        "fields": {**row, balance: "0", "opened_on": "2025-11-14"},
                    }
                ],
            )
        elif kind == "service":
            scenario.update(
                request=f"Please {action} for 20 USD from my record {rid}.",
                initial_rows={table: {rid: row}},
                steps=[
                    {
                        "operation": f"{prefix}_{verb}",
                        "arguments": {**args, "amount": "20"},
                    }
                ],
                goals=[
                    {
                        "table": table,
                        "row_id": rid,
                        "fields": {balance: "30.00", "status": "OPEN"},
                    }
                ],
            )
        elif kind == "ordering":
            scenario.update(
                request=f"Please close my record {rid} and {action} for all its remaining funds. I prefer closure first, but accept a different order if necessary. If you give me a closure-confirmation tool, I will execute it after the funds have been processed.",
                initial_rows={table: {rid: row}},
                steps=[
                    {
                        "operation": f"{prefix}_{verb}",
                        "arguments": {**args, "amount": "50"},
                    },
                    {"operation": f"{prefix}_confirm_close", "arguments": args},
                    {"operation": f"{prefix}_close", "arguments": args},
                ],
                goals=[
                    {
                        "table": table,
                        "row_id": rid,
                        "fields": {balance: "0.00", "status": "CLOSED"},
                    }
                ],
            )
        elif kind == "denial":
            scenario.update(
                request=f"I joined the bank yesterday. Please open {products[0]['name']} for me using record identifier {rid}. If I am not eligible, explain why and when I can apply.",
                steps=[{"operation": f"{prefix}_open", "arguments": args}],
                expected_denial=rules[0]["id"],
                communication=[
                    f"The assistant explains that this customer cannot open the product yet because the minimum tenure is {14 + ordinal % 3} days, and does not claim an account was opened."
                ],
            )
        else:
            scenario.update(
                request=f"Does {products[0]['name']} guarantee that next year's annual rate will double? If you cannot establish that from the documentation, say so.",
                communication=[
                    "The assistant clearly says the documents do not establish a guaranteed doubling of next year's annual rate and does not invent a guaranteed future rate."
                ],
            )
            scenario["required_facts"].append(f"{product_id}.effective_rate")
        scenarios.append(scenario)
    for scenario in scenarios:
        for goal in scenario.get("goals", []):
            if "opened_on" in goal["fields"]:
                goal["fields"]["opened_on"] = clock
    if profile == "credit_card":
        for product in products:
            labels = {
                "base_rate": "Base cash-back rate",
                "effective_rate": "Effective cash-back rate",
                "promotion_boost": "Promotional cash-back increase",
            }
            for fact_spec in product["facts"]:
                if fact_spec["id"] in labels:
                    fact_spec["label"] = labels[fact_spec["id"]]
        for scenario in scenarios:
            scenario["request"] = scenario["request"].replace(
                "annual rate", "cash-back rate"
            )
            scenario["communication"] = [
                s.replace("annual rate", "cash-back rate").replace(
                    "future rate", "future cash-back rate"
                )
                for s in scenario.get("communication", [])
            ]
    if profile == "installment":
        for i, product in enumerate(products):
            for fact_spec in product["facts"]:
                if fact_spec["id"] == "base_rate":
                    fact_spec.update(
                        label="Base annual borrowing rate",
                        value=["3.5", "4.5", "4.5"][i],
                    )
                elif fact_spec["id"] == "promotion_boost":
                    fact_spec["label"] = "Promotional borrowing-rate discount"
                elif fact_spec["id"] == "effective_rate":
                    fact_spec["label"] = "Effective annual borrowing rate"
                    fact_spec["expression"] = fact_spec["expression"].replace(
                        " + ", " - "
                    )
        scenarios[0]["selection"] = (
            "facts.monthly_fee == 0 and facts.effective_rate <= 3"
        )
        scenarios[0]["request"] = scenarios[0]["request"].replace(
            "effective annual rate is at least 4",
            "effective annual borrowing rate is at most 3",
        )
    entities = [
        {
            "id": table,
            "fields": {
                **common,
                "status": {
                    "type": "string",
                    "choices": ["OPEN", "CLOSURE_REQUESTED", "CLOSED"],
                },
                balance: amount,
                "opened_on": {"type": "date"},
            },
            "invariants": [f"row.{balance} >= 0", "row.opened_on <= clock"],
        }
    ]
    # A settlement posting is a real second table/effect, not a renamed balance.
    if profile in {"merchant", "credit_card", "installment"}:
        journal = f"{prefix}_postings"
        entities.append(
            {
                "id": journal,
                "fields": {
                    **common,
                    "source_record": string,
                    "amount": amount,
                    "posted_on": {"type": "date"},
                },
                "foreign_keys": [{"field": "source_record", "table": table}],
                "invariants": ["row.amount > 0", "row.posted_on <= clock"],
            }
        )
        service = operations[1]
        service["parameters"]["transaction_id"] = string
        service["effects"].append(
            {
                "table": journal,
                "mode": "create",
                "key": "args.transaction_id",
                "values": {
                    "user_id": "args.user_id",
                    "product_id": "args.product_id",
                    "source_record": "args.record_id",
                    "amount": "money(args.amount)",
                    "posted_on": "clock",
                },
            }
        )
        service["postconditions"] = [
            f"record.{balance} - after.{table}[args.record_id].{balance} == args.amount",
            f"after.{journal}[args.transaction_id].amount == args.amount",
        ]
        for scenario in scenarios:
            for step in scenario.get("steps", []):
                if step["operation"] == service["id"]:
                    transaction = scenario["id"] + "_posting"
                    step["arguments"]["transaction_id"] = transaction
                    scenario["request"] += (
                        f" Use transaction identifier {transaction} for the posting."
                    )
                    scenario["goals"].append(
                        {
                            "table": journal,
                            "row_id": transaction,
                            "fields": {
                                "user_id": scenario["user_id"],
                                "product_id": product_id,
                                "source_record": step["arguments"]["record_id"],
                                "amount": step["arguments"]["amount"],
                                "posted_on": clock,
                            },
                        }
                    )
    if profile == "escrow":
        entities[0]["fields"]["release_authorized"] = {"type": "boolean"}
        operations[0]["effects"][0]["values"]["release_authorized"] = "False"
        rules.append(
            {
                "id": f"{prefix}_authorized",
                "statement": "Customer authorization is required before release.",
                "predicate": "record.release_authorized == True",
            }
        )
        operations[1]["rules"].append(rules[-1]["id"])
        authorize = {
            "id": f"{prefix}_authorize",
            "description": "Customer authorizes release of escrow funds",
            "actor": "user",
            "parameters": parameters,
            "record_table": table,
            "effects": [
                {
                    "table": table,
                    "mode": "update",
                    "key": "args.record_id",
                    "values": {"release_authorized": "True"},
                }
            ],
        }
        operations.append(authorize)
        for scenario in scenarios:
            for rows in scenario.get("initial_rows", {}).values():
                for row in rows.values():
                    row["release_authorized"] = False
            for goal in scenario.get("goals", []):
                goal["fields"]["release_authorized"] = scenario["kind"] != "selection"
            if scenario["kind"] in {"service", "ordering"}:
                args = {k: scenario["steps"][0]["arguments"][k] for k in parameters}
                scenario["steps"].insert(
                    0, {"operation": authorize["id"], "arguments": args}
                )
                scenario["request"] += (
                    " Give me the release-authorization tool when needed; I will execute it."
                )
    return CategorySpec.model_validate(
        {
            "id": prefix,
            "description": description,
            "products": products,
            "entities": entities,
            "policies": rules,
            "operations": operations,
            "scenarios": scenarios,
        }
    )
