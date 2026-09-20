"""Versioned native business branches addressing the v03 failure mechanisms."""

import hashlib
from datetime import timedelta
from decimal import Decimal
from functools import lru_cache

from tau3.domains.banking_knowledge.utils import get_today
from tau3.synthesis.catalog import BANK, CARDS, LOGISTICS, build_catalog

DENIALS = (
    "insufficient_account_age", "cooldown_period_active", "pending_disputes",
    "pending_replacement_card", "past_due_balance", "high_utilization",
    "insufficient_payment_history",
)


@lru_cache(maxsize=1)
def cli_rules():
    """Extract numeric thresholds from the installed public KB."""
    return build_catalog().rules["cli_tiers"]


def credit_card(b, suffix, card_type="Silver Rewards Card", balance=0, limit=8000):
    """Create an owned credit account with explicit public-queryable fields."""
    cid = f"cc_{b.key}_{suffix}"
    return b.put("credit_card_accounts", cid, {
        "account_id": cid, "credit_card_account_id": cid, "user_id": b.uid,
        "card_type": card_type, "status": "ACTIVE", "account_status": "CURRENT",
        "date_of_account_open": "01/15/2023", "current_balance": f"${balance:.2f}",
        "credit_limit": f"${limit:.2f}", "past_due_balance": "$0.00",
        "last_4_digits": "".join(c for c in hashlib.sha256(f"card_last4:{cid}".encode()).hexdigest() if c.isdigit())[:4].ljust(4, "0"),
    })


def credit_limit(b, count, variant):
    """Submit, inspect all eligibility conditions, then record the justified decision."""
    branch = b.slot.branch_id or "deny:high_utilization"
    b.verify()
    if b.discovery:
        b.actions.add("get_credit_card_accounts_by_user", {"user_id": b.uid})
    tiers = ["entry", "mid", "premium"]
    products = ["Bronze Rewards Card", "Silver Rewards Card", "Gold Rewards Card"]
    tier = tiers[variant % 3]
    rule = cli_rules()[tier]
    if branch == "deny:insufficient_account_age" and tier != "mid":
        # Three completed calendar-month payments can fit inside the mid-tier
        # age boundary. Entry/premium boundaries cannot isolate this failure.
        raise ValueError("Account-age boundary requires a feasible payment history")
    payment_dates = []
    month_start = get_today().replace(day=1)
    for _ in range(rule["months"]):
        month_end = month_start - timedelta(days=1)
        payment_dates.append(month_end)
        month_start = month_end.replace(day=1)
    desired_count = min(count, 3) if branch == "mixed" else 1
    requests = []
    for i in range(desired_count):
        decision = branch.split(":", 1)[1] if branch.startswith("deny:") else branch
        if branch == "mixed":
            if count < 2:
                raise ValueError("Mixed branch requires multiple target cards")
            decision = "approve" if i == 0 else "high_utilization"
        limit = [8000, 12000, 20000][(variant // 3) % 3]
        increase = limit * rule["max_increase_percent"] // 100
        if variant >= 24:
            increase //= 2
        cid = credit_card(b, i, products[(variant + i) % 3] if branch == "mixed" else products[variant % 3], limit=limit)
        # Mixed accounts use the same tier rules, with unique last-four identifiers.
        b.data["credit_card_accounts"]["data"][cid]["card_type"] = products[variant % 3]
        row = b.data["credit_card_accounts"]["data"][cid]
        row["card_tier"] = tier
        age = (rule["age"] - 1 if decision == "insufficient_account_age" else
               max(rule["age"], (get_today() - payment_dates[-1]).days + 1,
                   rule["cooldown"]))
        row["date_of_account_open"] = (get_today() - timedelta(days=age)).strftime("%m/%d/%Y")
        utilization = Decimal(rule["utilization"] * limit) / 100
        row["current_balance"] = f"${utilization if decision == 'high_utilization' else utilization - 1:.2f}"
        if decision == "past_due_balance":
            row.update(account_status="PAST_DUE", past_due_balance="$100.00")
        if decision == "cooldown_period_active":
            previous = f"prior_cli_{b.key}_{i}"
            b.put("credit_limit_increase_requests", previous, {
                "request_id": previous, "user_id": b.uid, "credit_card_account_id": cid,
                "submitted_at": (get_today() - timedelta(days=rule["cooldown"] - 1)).strftime("%m/%d/%Y"),
                "status": "APPROVED", "requested_increase_amount": 500,
            })
        if decision == "pending_disputes":
            b.put("transaction_disputes", f"prior_dispute_{b.key}_{i}", {
                "user_id": b.uid, "credit_card_account_id": cid, "status": "PENDING",
            })
        if decision == "pending_replacement_card":
            b.put("credit_card_orders", f"prior_order_{b.key}_{i}", {
                "user_id": b.uid, "credit_card_account_id": cid, "status": "PENDING",
            })
        for month in range(rule["months"]):
            pid = f"payment_{b.key}_{i}_{month}"
            b.put("payment_history", pid, {
                "payment_id": pid, "user_id": b.uid, "credit_card_account_id": cid,
                "payment_date": payment_dates[month].isoformat(),
                "amount": "$300.00", "status": "LATE" if decision == "insufficient_payment_history" and month == 0 else "ON_TIME",
            })
        if decision == "overlimit":
            increase = limit * rule["max_increase_percent"] // 100 + 1
        requests.append({"last_four": row["last_4_digits"], "product": row["card_type"],
                         "increase_amount": None if decision == "missing_amount" else increase})
        if decision in {"overlimit", "missing_amount"}:
            b.goals.append({"kind": "cli_decision", "account_id": cid, "requested_amount": None if decision == "missing_amount" else increase})
            continue
        b.actions.call("submit_credit_limit_increase_request_7392", credit_card_account_id=cid,
                       user_id=b.uid, requested_increase_amount=increase)
        b.actions.call("get_credit_limit_increase_history_4829", credit_card_account_id=cid)
        b.actions.call("get_user_dispute_history_7291", user_id=b.uid)
        b.actions.call("get_pending_replacement_orders_5765", credit_card_account_id=cid)
        b.actions.call("get_payment_history_6183", credit_card_account_id=cid, months=rule["months"])
        if decision == "approve":
            b.actions.call("approve_credit_limit_increase_5847", credit_card_account_id=cid,
                           user_id=b.uid, new_credit_limit=limit + increase)
        else:
            b.actions.call("deny_credit_limit_increase_5848", credit_card_account_id=cid,
                           user_id=b.uid, denial_reason=decision)
        b.goals.append({"kind": "cli_decision", "account_id": cid, "requested_amount": increase})
    # Another real owned account must be preserved, even when it has a similar name.
    distractor = credit_card(b, "untouched", products[(variant + 1) % 3])
    b.field("credit_card_accounts", distractor, credit_limit="$8000.00", status="ACTIVE")
    b.facts["credit_increase_requests"] = requests
    b.intents.append("Process the credit limit increase requests for the cards I list, after checking the bank's rules. The amounts mean additional credit, not the new total. Do not pay down balances, change another card, or request replacement cards.")
    if branch in {"overlimit", "missing_amount"}:
        b.facts["request_constraint"] = "If my amount is above the allowed maximum, I decline reducing it today. If I have not given an amount, ask me, but I need time to decide and do not authorize any amount yet. Leave the request unsubmitted in either case."
    b.docs.update(LOGISTICS + n for n in ("005", "006", "007"))
    b.branches.append(["cli", tier, branch, desired_count, (variant // 3) % 3, "at_max" if variant < 24 else "half_max"])


def replacement_closure(b, count, variant):
    """Replace damaged credit cards or close and pay off selected owned accounts."""
    b.verify()
    b.actions.add("get_credit_card_accounts_by_user", {"user_id": b.uid})
    close = b.slot.branch_id == "close" or (b.slot.branch_id is None and variant % 2 == 0)
    requests = []
    for i in range(min(count, 3)):
        balance = (i + 1) * 100 if close and variant % 3 else 0
        product = ["Bronze Rewards Card", "Silver Rewards Card", "Gold Rewards Card"][variant % 3]
        cid = credit_card(b, i, card_type=product, balance=balance)
        requests.append(b.data["credit_card_accounts"]["data"][cid]["last_4_digits"])
        # Disputes block account closure, not replacement of a damaged physical
        # card. Scored reference reads must not invent a closure prerequisite.
        b.actions.call("get_user_dispute_history_7291", user_id=b.uid) if close and i == 0 else None
        b.actions.call("get_pending_replacement_orders_5765", credit_card_account_id=cid)
        if close:
            if balance:
                aid = b.account(f"payoff{i}", "Blue Account", balance="1000.00")
                nickname = f"Payoff for card ending {requests[-1]}"
                b.data["accounts"]["data"][aid]["nickname"] = nickname
                b.facts.setdefault("repayment_account_nicknames", []).append({
                    "credit_card_last_four": requests[-1], "checking_nickname": nickname})
                b.actions.call("pay_credit_card_from_checking_9182", user_id=b.uid,
                               checking_account_id=aid, credit_card_account_id=cid, amount=balance)
                b.goals.append({"kind": "balance", "account_id": aid, "expected": str(1000 - balance)})
                b.field("credit_card_accounts", cid, current_balance="$0.00")
            b.actions.call("get_closure_reason_history_8293", credit_card_account_id=cid)
            b.actions.call("log_credit_card_closure_reason_4521", credit_card_account_id=cid,
                           user_id=b.uid, closure_reason="simplifying_finances")
            b.actions.call("close_credit_card_account_7834", credit_card_account_id=cid, user_id=b.uid)
        else:
            b.actions.call("order_replacement_credit_card_7291", credit_card_account_id=cid,
                           user_id=b.uid, shipping_address=b.user["address"], reason="damaged", expedited_shipping=False)
            b.goals.append({"kind": "row_match", "table": "credit_card_orders", "selector": {"credit_card_account_id": cid},
                           "expected": {"reason": "damaged", "old_card_cancelled": True, "expedited_shipping": False}})
        status = "ACTIVE" if not close and b.slot.runtime_revision == "native_card_lifecycle_v2" else "CLOSED"
        b.field("credit_card_accounts", cid, status=status)
    untouched = credit_card(b, "keep")
    b.field("credit_card_accounts", untouched, status="ACTIVE")
    b.facts["selected_card_last_four"] = requests
    payoff_instruction = (
        "Pay any remaining balance using the funded Blue checking account whose nickname I provide for that card before closure; I authorize that exact payoff. "
        if b.facts.get("repayment_account_nicknames") else
        "Pay any remaining balance using its matching funded Blue checking account before closure; I authorize that exact payoff. ")
    b.intents.append("For each credit card whose last four digits I list, " + (
        "close the account because I am simplifying my finances. I decline all retention offers. " + payoff_instruction + "Keep other cards and all checking accounts."
        if close else "replace the damaged physical card, with standard shipping to my address on file. I still want the account and do not want an extra account closure or a retention offer. Keep other cards unchanged."))
    b.branches.append(["credit_closure" if close else "credit_replacement", min(count, 3), bool(variant % 3), product])
    b.docs.update(LOGISTICS + n for n in ("001", "002", "003"))


def transaction_disputes(b, count, variant):
    """Merchant duplicate disputes with fresh transactions and unambiguous consent."""
    b.verify()
    debit = b.slot.branch_id == "debit" or (b.slot.branch_id is None and variant % 2 == 0)
    today = get_today().strftime("%m/%d/%Y")
    date = (get_today() - timedelta(days=7)).strftime("%m/%d/%Y")
    if debit:
        aid = b.account("dispute", "Blue Account", "2500.00", age=900)
        cid = b.card(aid, "dispute")
        b.query_accounts()
        b.actions.call("get_debit_cards_by_account_id_7823", account_id=aid)
        b.actions.call("get_bank_account_transactions_9173", account_id=aid)
        b.actions.call("get_debit_dispute_status_7483", user_id=b.uid)
    else:
        cid = credit_card(b, "dispute")
        b.actions.add("get_credit_card_transactions_by_user", {"user_id": b.uid})
        b.actions.call("get_user_dispute_history_7291", user_id=b.uid)
        b.actions.call("get_pending_replacement_orders_5765", credit_card_account_id=cid)
        b.grant("get_card_last_4_digits", {"credit_card_account_id": cid})
    problems = []
    for i in range(min(count, 3 if debit else 4)):
        amount = b.rng.randint(40, 250)
        merchant = f"Cedar Books {i}"
        original, duplicate = f"purchase_{b.key}_{i}", f"duplicate_{b.key}_{i}"
        for key in (original, duplicate):
            if debit:
                b.put("bank_account_transaction_history", key, {"transaction_id": key, "account_id": aid,
                    "date": date if key == original else (get_today() - timedelta(days=6)).strftime("%m/%d/%Y"),
                    "description": merchant, "amount": -amount, "type": "purchase", "status": "posted"})
            else:
                b.put("credit_card_transaction_history", key, {"transaction_id": key, "user_id": b.uid,
                    "credit_card_account_id": cid, "credit_card_type": "Silver Rewards Card", "merchant_name": merchant,
                    "transaction_amount": f"${amount:.2f}", "transaction_date": date,
                    "category": "Books", "status": "COMPLETED", "rewards_earned": f"{amount} points"})
        problems.append({"merchant": merchant, "amount": amount, "date": date, "discovered": today,
                         "duplicate_reference": duplicate, "original_reference": original})
        selected = original if debit else duplicate
        if debit:
            args = dict(transaction_id=selected, account_id=aid, card_id=cid, user_id=b.uid,
                        dispute_category="duplicate_charge", transaction_date=date, discovery_date=today,
                        disputed_amount=amount, transaction_type="online_purchase", card_in_possession=True,
                        pin_compromised="no", contacted_merchant=True, police_report_filed=False,
                        written_statement_provided=True, provisional_credit_eligible=True,
                        customer_max_liability_amount=min(50, amount), card_action="keep_active")
            b.actions.call("file_debit_card_transaction_dispute_6281", **args)
            table = "debit_card_disputes"
        else:
            # The private customer has the physical card; no private policy answer is supplied.
            args = dict(transaction_id=duplicate, card_action="keep_active",
                        card_last_4_digits=b.data["credit_card_accounts"]["data"][cid]["last_4_digits"],
                        full_name=b.user["name"], user_id=b.uid, phone=b.user["phone_number"], email=b.user["email"],
                        address=b.user["address"], contacted_merchant=True, purchase_date=date,
                        issue_noticed_date=today, dispute_reason="duplicate_charge", resolution_requested="full_refund",
                        eligible_for_provisional_credit=i <= 2)
            b.actions.call("file_credit_card_transaction_dispute_4829", **args)
            table = "transaction_disputes"
        b.goals.append({"kind": "row_match", "table": table, "selector": {"transaction_id": selected},
                       "expected": args})
    b.facts["statement_duplicates"] = problems
    b.facts["merchant_contact"] = "I bought once at each listed merchant, and each merchant confirms the second listed charge is a duplicate but has not refunded it. I contacted them yesterday. I still have the physical card, never shared my PIN, and recognize the original purchases. I have provided a signed written statement and can confirm those facts. No charge is pending."
    b.facts["physical_card_last_four"] = b.data["debit_cards" if debit else "credit_card_accounts"]["data"][cid]["last_4_digits"]
    b.facts["purchase_channel"] = "All listed purchases were online orders, with no in-store PIN or signature."
    b.facts["statement_date"] = today
    b.facts["police_report"] = "I have not filed a police report. These are recognized purchases with merchant duplicate charges, not suspected fraud."
    b.facts["requested_resolution"] = "I want a full refund for one charge in each duplicate pair."
    if not debit:
        b.facts["selected_dispute_entries"] = [p["duplicate_reference"] for p in problems]
        b.intents.append("For my credit card, dispute the statement entries I identify as the duplicate references, requesting full refunds. Please handle the merchants in the order I list them.")
    b.intents.append("For each duplicated purchase in my statement, file one formal dispute for one of the two charges according to the bank's duplicate-charge procedure, so I only pay once. Keep my card active. I can identify both statement references. I do not request replacing the card or any unrelated account changes.")
    b.docs.update([CARDS + "014", CARDS + "015", BANK + "031", BANK + "032"])
    b.branches.append(["transaction_disputes", "debit" if debit else "credit", min(count, 4)])
    if count > 4:
        from tau3.synthesis.targeted.native.scenarios import accounts

        accounts(b, count - 4, variant)


def install_generators():
    """Return versioned generators without changing legacy family semantics."""
    from tau3.synthesis.targeted.native.scenarios import (
        accounts,
        boundary,
        debit,
        disputes,
        handoff,
        optimization,
    )
    from tau3.synthesis.targeted.native.v03_supplements import (
        account_service,
        card_service,
        incident,
        retention,
        user_action,
    )

    def dispatch(default, mapping):
        def generate(builder, count, variant):
            return mapping.get(builder.slot.branch_id, default)(builder, count, variant)
        return generate

    def account_workflow(builder, count, variant):
        # Opening with eligibility discovery needs at least five real actions.
        # Short account-service slots use an authorized email change instead.
        if builder.slot.v03_labels and builder.slot.difficulty == "1-4":
            return account_service(builder, count, variant)
        return accounts(builder, count, variant)

    def rewards(builder, count, variant):
        # Freeze approved-resolution cases to cover the actual reward correction.
        disputes(builder, count, variant if variant % 3 else variant + 1)

    def optimize(builder, count, variant):
        mechanism = "savings_correction" if variant % 6 == 4 else "referral" if variant % 2 else "card_selection"
        if builder.slot.branch_id in {"savings_correction", "referral", "card_selection"} and mechanism != builder.slot.branch_id:
            raise ValueError("Variant does not implement the frozen optimization mechanism")
        optimization(builder, count, variant)
        if any(goal["kind"] == "referral" for goal in builder.goals):
            builder.actions.add("get_referrals_by_user", {"user_id": builder.uid})
            query = builder.actions.items.pop()
            index = next(i for i, a in enumerate(builder.actions.items) if a.name == "submit_referral")
            builder.actions.items.insert(index, query)

    return {"credit_limit": credit_limit,
            "transaction_disputes": dispatch(transaction_disputes, {"rewards": rewards}),
            "replacement_closure": dispatch(replacement_closure, {"retention_credit": retention, "retention_waiver": retention}),
            "accounts_funds": dispatch(account_workflow, {"email": account_service, "checking_fee": account_service, "open": account_service}),
            "debit_security": dispatch(debit, {name: card_service for name in ("activate", "clear_block", "temporary_limit", "unfreeze", "replacement")}),
            "handoff": dispatch(handoff, {"referral_link": user_action, "purchase": user_action}),
            "optimization": optimize,
            "escalation_boundary": dispatch(boundary, {"payment_incident": incident, "decline_incident": incident, "emergency": incident})}
