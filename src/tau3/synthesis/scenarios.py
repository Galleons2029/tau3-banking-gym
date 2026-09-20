"""Programmatic business samplers and reference actions, without benchmark seeds."""

import json
import random
from datetime import datetime, timedelta
from decimal import Decimal

from tau3.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    InitializationData,
    InitialState,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau3.domains.banking_knowledge.utils import get_now, get_today
from tau3.synthesis.catalog import BANK, CARDS, LOGISTICS, PRODUCT_DOCS
from tau3.synthesis.models import (
    FAMILIES,
    GenerationRecord,
    RuleCatalog,
    ScenarioSkeleton,
)
from tau3.synthesis.storage import digest


def customer_age(date_of_birth: str) -> int:
    """Compute completed years at the fixed environment date."""
    born = datetime.strptime(date_of_birth, "%m/%d/%Y").date()
    today = get_today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def solve_selection(products: dict, constraints: dict, *, age: int) -> str:
    """Choose the unique cheapest listed product satisfying every hard constraint."""
    candidates = {
        name: Decimal(p["monthly_fee"])
        for name, p in products.items()
        if Decimal(p["mobile_deposit_limit"])
        >= Decimal(str(constraints["mobile_check_deposit_per_day"]))
        and p["early_days"] >= constraints["early_days"]
        and Decimal(p["monthly_fee"]) <= Decimal(str(constraints["fee_cap"]))
        and p.get("min_age", 0) <= age <= p.get("max_age", 200)
    }
    if not candidates:
        raise ValueError("No eligible product")
    best = min(candidates.values())
    winners = [name for name, cost in candidates.items() if cost == best]
    if len(winners) != 1:
        raise ValueError("Ambiguous product selection")
    return winners[0]


def cli_decision(state: dict, rules: dict) -> str:
    """Apply the documented age, utilization, payment and account requirements."""
    if state["age"] < rules["age"]:
        return "insufficient_account_age"
    if state["pending_disputes"]:
        return "pending_disputes"
    if state["pending_replacement"]:
        return "pending_replacement_card"
    if state["past_due"]:
        return "past_due_balance"
    if (
        Decimal(state["balance"]) * 100
        >= Decimal(state["limit"]) * rules["utilization"]
    ):
        return "high_utilization"
    if state["on_time_months"] < rules["months"]:
        return "insufficient_payment_history"
    if (
        Decimal(state["increase"]) * 100
        > Decimal(state["limit"]) * rules["max_increase_percent"]
    ):
        raise ValueError("Requested amount must be adjusted before submission")
    return "approved"


def reward_points(amount: str, category: str, card: str, catalog: RuleCatalog) -> int:
    """Calculate whole-point pilot purchases; reject unspecified fractional rounding."""
    rule = catalog.rules["rewards"][card]
    rate = rule["enhanced"] if category in rule["categories"] else rule["default"]
    points = Decimal(amount) * Decimal(rate)
    if points != points.to_integral_value():
        raise ValueError(
            "Fractional points require a separately verified rounding rule"
        )
    return int(points)


class Actions:
    """Build correctly routed calls while unlocking each discoverable tool once."""

    def __init__(self):
        self.items = []
        self.unlocked = set()

    def add(self, name, arguments, requestor="assistant"):
        self.items.append(
            Action(
                action_id=f"a{len(self.items):03d}",
                name=name,
                arguments=arguments,
                requestor=requestor,
            )
        )

    def call(self, name, **arguments):
        if name not in self.unlocked:
            self.add("unlock_discoverable_agent_tool", {"agent_tool_name": name})
            self.unlocked.add(name)
        self.add(
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        )


def sample_candidate(
    catalog: RuleCatalog, seed: int, slot: int, attempt: int = 0
) -> GenerationRecord:
    """Sample a fresh customer and compile one supported business scenario."""
    family = FAMILIES[slot % len(FAMILIES)]
    rng = random.Random(seed)
    key = digest(["fixed-tools-v3", family, seed])[:12]
    user_id = "syn_" + key
    user = {
        "user_id": user_id,
        "name": f"{rng.choice(['Alex', 'Jordan', 'Morgan', 'Casey', 'Robin', 'Taylor'])} {rng.choice(['Chen', 'Patel', 'Rivera', 'Kim', 'Reed', 'Brooks'])} {key[:4]}",
        "date_of_birth": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.choice([2000, 2001, 2005, 1980]) if family == 'selection' else rng.randint(1970, 1998)}",
        "address": f"{rng.randint(100, 9999)} Cedar Lane, Austin, TX 78701",
        "email": f"customer.{key}@example.com",
        "phone_number": f"512-555-{rng.randint(1000, 9999)}",
    }
    data = {"users": {"data": {user_id: user}}}
    actions = Actions()
    actions.add(
        "log_verification",
        {**user, "time_verified": get_now().strftime("%Y-%m-%d %H:%M:%S EST")},
    )
    facts = {
        "identity": user,
        "funding": "Decline optional extra products and referrals.",
    }
    private = {"user_id": user_id}
    docs = set()
    user_tools = []
    reveal = rng.choice(["upfront", "when_asked"])

    def put(table, record_id, row):
        data.setdefault(table, {"data": {}})["data"][record_id] = row

    def account(
        suffix, kind="checking", product="Light Blue Account", balance="0.00", age=400
    ):
        aid = f"acct_{key}_{suffix}"
        put(
            "accounts",
            aid,
            {
                "account_id": aid,
                "user_id": user_id,
                "class": kind,
                "level": product,
                "account_type": kind,
                "account_class": product,
                "date_opened": (get_today() - timedelta(days=age)).strftime("%m/%d/%Y"),
                "status": "OPEN",
                "current_holdings": balance,
            },
        )
        return aid

    if family == "selection":
        bucket = rng.randrange(3)
        constraints = {
            "mobile_check_deposit_per_day": rng.choice(
                [
                    [100, 250, 499, 500],
                    [501, 1200, 2499, 2500],
                    [2501, 2800, 2999, 3000],
                ][bucket]
            ),
            "early_days": rng.choice([0, 1]) if bucket else 0,
            "fee_cap": rng.choice(["22.50", "25.00", "30.00"]),
        }
        age = customer_age(user["date_of_birth"])
        chosen = solve_selection(catalog.products, constraints, age=age)
        facts.update(
            {
                "goal": "Compare only my three shortlisted checking accounts, recommend the cheapest monthly maintenance fee option meeting my needs, and open it after I confirm.",
                "shortlist": list(catalog.products),
                "preferences": constraints,
                "preference_timing": reveal,
                "funding": "I cannot guarantee a minimum daily balance for any fee waiver. Compare standard monthly fees. I want electronic statements. I will fund externally later; do not transfer money or order cards today.",
            }
        )
        actions.call(
            "open_bank_account_4821",
            user_id=user_id,
            account_type="checking",
            account_class=chosen,
        )
        private.update({"constraints": constraints, "selected_product": chosen})
        docs.update(PRODUCT_DOCS.values())
        if constraints["mobile_check_deposit_per_day"] <= 500:
            docs.add(catalog.products["Light Green Account"]["age_document"])
        docs.add(BANK + "001")
        shape = [family, bucket, constraints["early_days"], reveal]
        if bucket == 0:
            shape.append(13 <= age <= 24)

    elif family == "cashback":
        card = rng.choice(list(catalog.rules["rewards"]))
        cid = f"cc_{key}"
        count = rng.randint(2, 5)
        auto = rng.choice([True, False])
        put(
            "credit_card_accounts",
            cid,
            {
                "account_id": cid,
                "user_id": user_id,
                "card_type": card,
                "status": "ACTIVE",
                "account_status": "CURRENT",
                "date_of_account_open": "01/15/2024",
                "current_balance": "$500.00",
            },
        )
        put("task_config", "dispute_settings", {"auto_resolve_disputes": auto})
        txns = []
        for i in range(count):
            category = rng.choice(["Travel", "Software", "Groceries"])
            amount = str(rng.randint(10, 700))
            points = reward_points(amount, category, card, catalog)
            wrong = i < count - 1  # At least one correct distractor.
            actual = max(0, points - rng.randint(1, points)) if wrong else points
            tid = f"txn_{key}_{i}"
            put(
                "credit_card_transaction_history",
                tid,
                {
                    "transaction_id": tid,
                    "user_id": user_id,
                    "credit_card_account_id": cid,
                    "credit_card_type": card,
                    "merchant_name": f"{category} Merchant {i + 1}",
                    "transaction_amount": f"${Decimal(amount):.2f}",
                    "transaction_date": "10/15/2025",
                    "category": category,
                    "status": "COMPLETED",
                    "rewards_earned": f"{actual} points",
                    "promotion_applied": False,
                },
            )
            txns.append(
                {"id": tid, "amount": amount, "category": category, "actual": actual}
            )
        facts.update(
            {
                "goal": "Investigate my cash back statement, identify any incorrect rewards, and help me dispute only the incorrect transactions.",
                "card": card,
                "knowledge": "I do not know transaction IDs, the affected purchases or reward rates. Purchases were made directly with merchants; no returns, promotional enrollment or third-party wallets. Please inspect my records.",
                "interaction": "Use a supplied customer capability only as directed by the agent. Report its actual status. If it reports approval and resolution, ask the agent to correct the rewards; otherwise accept that review is pending. Never invent a resolution notification.",
            }
        )
        actions.add(
            "give_discoverable_user_tool",
            {"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        for txn in txns[:-1]:
            actions.add(
                "call_discoverable_user_tool",
                {
                    "discoverable_tool_name": "submit_cash_back_dispute_0589",
                    "arguments": json.dumps(
                        {"user_id": user_id, "transaction_id": txn["id"]}
                    ),
                },
                "user",
            )
        if auto:
            for txn in txns[:-1]:
                points = reward_points(txn["amount"], txn["category"], card, catalog)
                actions.call(
                    "update_transaction_rewards_3847",
                    transaction_id=txn["id"],
                    new_rewards_earned=f"{points} points",
                )
        private.update({"card": card, "transactions": txns, "auto_resolve": auto})
        user_tools = ["call_discoverable_user_tool"]
        docs.update([CARDS + "003", CARDS + "004"])
        docs.update(
            d
            for d in catalog.documents
            if (
                "silver_rewards_card_00" in d
                if card.startswith("Silver")
                else "bronze_rewards_card_002" in d
            )
        )
        shape = [family, card, count, auto, sorted(t["category"] for t in txns)]

    elif family == "credit_limit":
        tier = rng.choice(["entry", "mid", "premium"])
        rule = catalog.rules["cli_tiers"][tier]
        mode = rng.choice(
            [
                "age",
                "utilization",
                "payment",
                "dispute",
                "replacement",
                "past_due",
            ]
        )
        cid = f"cc_{key}"
        card = {
            "entry": "Bronze Rewards Card",
            "mid": "Silver Rewards Card",
            "premium": "Gold Rewards Card",
        }[tier]
        limit = rng.choice([4000, 8000, 12000, 20000])
        state = {
            "age": rule["age"] - 1
            if mode == "age"
            else rule["age"] + rng.choice([0, 1, 60, 180]),
            "balance": str(
                limit * (rule["utilization"] if mode == "utilization" else 20) // 100
            ),
            "limit": str(limit),
            "increase": str(limit * rng.choice([10, 20, 25]) // 100),
            "on_time_months": rule["months"] - 1
            if mode == "payment"
            else rule["months"],
            "pending_disputes": mode == "dispute",
            "pending_replacement": mode == "replacement",
            "past_due": mode == "past_due",
        }
        decision = cli_decision(state, rule)
        put(
            "credit_card_accounts",
            cid,
            {
                "account_id": cid,
                "credit_card_account_id": cid,
                "user_id": user_id,
                "card_type": card,
                "card_tier": tier,
                "status": "ACTIVE",
                "account_status": "PAST_DUE" if state["past_due"] else "CURRENT",
                "date_of_account_open": (
                    get_today() - timedelta(days=state["age"])
                ).strftime("%m/%d/%Y"),
                "credit_limit": f"${limit:.2f}",
                "current_balance": f"${Decimal(state['balance']):.2f}",
                "past_due_balance": "$100.00" if state["past_due"] else "$0.00",
            },
        )
        for i in range(rule["months"]):
            pid = f"payment_{key}_{i}"
            put(
                "payment_history",
                pid,
                {
                    "payment_id": pid,
                    "credit_card_account_id": cid,
                    "user_id": user_id,
                    "payment_date": f"2025-{10 - i:02d}-15",
                    "amount": "$300.00",
                    "status": "ON_TIME" if i < state["on_time_months"] else "LATE",
                },
            )
        if state["pending_disputes"]:
            put(
                "transaction_disputes",
                f"dispute_{key}",
                {
                    "user_id": user_id,
                    "credit_card_account_id": cid,
                    "status": "PENDING",
                },
            )
        if state["pending_replacement"]:
            put(
                "credit_card_orders",
                f"order_{key}",
                {
                    "user_id": user_id,
                    "credit_card_account_id": cid,
                    "status": "PENDING",
                },
            )
        facts.update(
            {
                "goal": f"Request a ${state['increase']} increase, not a new total limit, on my {card} for planned household spending.",
                "interaction": "Authorize submitting this request. Let the agent inspect eligibility, accept either an approval or a documented denial, and do not request payments, replacement cards, reapplications or other changes today. Do not invent account dates, eligibility thresholds or payment records.",
            }
        )
        actions.call(
            "submit_credit_limit_increase_request_7392",
            credit_card_account_id=cid,
            user_id=user_id,
            requested_increase_amount=int(state["increase"]),
        )
        actions.call(
            "get_credit_limit_increase_history_4829", credit_card_account_id=cid
        )
        actions.call("get_user_dispute_history_7291", user_id=user_id)
        actions.call("get_pending_replacement_orders_5765", credit_card_account_id=cid)
        actions.call(
            "get_payment_history_6183",
            credit_card_account_id=cid,
            months=rule["months"],
        )
        if decision == "approved":
            actions.call(
                "approve_credit_limit_increase_5847",
                credit_card_account_id=cid,
                user_id=user_id,
                new_credit_limit=limit + int(state["increase"]),
            )
        else:
            actions.call(
                "deny_credit_limit_increase_5848",
                credit_card_account_id=cid,
                user_id=user_id,
                denial_reason=decision,
            )
        private.update(
            {"state": state, "tier": tier, "card_id": cid, "decision": decision}
        )
        docs.update([LOGISTICS + n for n in ("005", "006", "007")])
        shape = [family, tier, mode, state["age"] == rule["age"]]

    else:
        # Never depend on unsupported notice periods or early-closure fee waivers.
        count = rng.choice([1, 2, 3])
        open_business = rng.choice([True, False])
        # Keep combined opening/closure coverage across candidate replacements.
        if (slot // len(FAMILIES)) % 5 == 2:
            open_business = True
        balances = [str(rng.randint(5, 50) * 100) for _ in range(count)]
        sources = [
            account(
                str(i),
                balance=amount,
                product=rng.choice(["Light Blue Account", "Light Green Account"]),
            )
            for i, amount in enumerate(balances)
        ]
        destination = account("keep", product="Blue Account", balance="1000.00")
        facts.update(
            {
                "goal": "Consolidate my old entry-tier checking accounts into my existing Blue Account and close the emptied old accounts. Record my closure reason exactly as 'Customer requested closure'."
                + (
                    " Also open a Navy Blue business checking account for my established company."
                    if open_business
                    else ""
                ),
                "interaction": "I initially ask for closures first, but accept a different order when the agent explains the prerequisites. Authorize moving every dollar from the old entry-tier accounts to the existing Blue Account before closure. Keep the Blue Account. Proactively ask the agent to record the closure reason exactly as 'Customer requested closure'; this wording is my explicit requirement. Decline optional funding transfers to any new account, cards and referrals. End only after all requested changes are done.",
                "old_accounts": sources,
                "destination": destination,
            }
        )
        if open_business:
            facts["interaction"] += (
                " State the business-account opening request in your first message, "
                "together with the consolidation and closure requests, before "
                "authorizing any account changes. You may prefer closures first, "
                "but disclose all existing goals up front; do not introduce the "
                "business-account request only after accounts have been closed."
            )
            actions.call(
                "open_bank_account_4821",
                user_id=user_id,
                account_type="business_checking",
                account_class="Navy Blue",
            )
        for aid, balance in zip(sources, balances):
            actions.call(
                "transfer_funds_between_bank_accounts_7291",
                source_account_id=aid,
                destination_account_id=destination,
                amount=float(balance),
            )
            actions.call(
                "close_bank_account_7392",
                account_id=aid,
                reason="Customer requested closure",
            )
        private.update(
            {
                "sources": sources,
                "balances": balances,
                "destination": destination,
                "open_business": open_business,
            }
        )
        docs.update([BANK + "005", BANK + "010"])
        if open_business:
            docs.update([BANK + "003", "doc_business_checking_accounts_navy_blue_001"])
        shape = [
            family,
            count,
            open_business,
            sorted(data["accounts"]["data"][s]["level"] for s in sources),
        ]

    # Include tool discovery evidence but never place it in user instructions.
    for action in actions.items:
        tool = action.arguments.get("agent_tool_name", action.name)
        if tool in catalog.tool_signatures and not any(
            tool in catalog.documents[d].quote for d in docs
        ):
            matches = [d for d, e in catalog.documents.items() if tool in e.quote]
            docs.add(min(matches, key=lambda d: len(catalog.documents[d].quote)))
    skeleton = ScenarioSkeleton(
        family=family,
        seed=seed,
        group_id=digest(shape),
        facts=facts,
        private=private,
        difficulty="multi_step"
        if len(actions.items) > 10
        else "boundary"
        if family == "credit_limit"
        else "basic",
        evidence_ids=sorted(docs),
    )
    instructions = StructuredUserInstructions(
        domain="banking_knowledge",
        reason_for_call=facts["goal"],
        known_info=json.dumps(facts, ensure_ascii=False),
        unknown_info="Do not invent bank policy, correct answers, internal tools, account records or unreported approvals.",
        task_instructions="Provide identity fields when asked. Follow the stated preferences and interaction constraints. State numerical requirements exactly; do not round or weaken minimums, maximums, or requested amounts. Ask the agent to research bank rules. Confirm a suitable recommendation, accept a justified refusal, and end once the requested outcome is complete.",
    )
    task = Task(
        id=f"synth_{family}_{key}",
        description=Description(purpose=f"Synthetic {family}"),
        user_scenario=UserScenario(instructions=instructions),
        initial_state=InitialState(
            initialization_data=InitializationData(agent_data=data)
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions.items, reward_basis=["DB"]
        ),
        required_documents=sorted(docs),
        user_tools=user_tools,
    )
    return GenerationRecord(
        task=task, skeleton=skeleton, slot=slot, attempt=attempt, text_mode="template"
    )
