"""Independent policy predicates over initial records and actual resulting state."""

import json
import re
from datetime import datetime
from decimal import Decimal

from tau3.domains.banking_knowledge.utils import get_today
from tau3.synthesis.catalog import BANK, LOGISTICS
from tau3.synthesis.targeted.native.catalog import CARDS, documents
from tau3.synthesis.validation import money


def check_initial_consistency(candidate):
    """Reject impossible dated records and dispute answers hidden from the user."""
    check_incident_transfer_reason(candidate)
    data = candidate.task.initial_state.initialization_data.agent_data
    cards = data.get("credit_card_accounts", {}).get("data", {})
    for payment in data.get("payment_history", {}).get("data", {}).values():
        card = cards.get(payment.get("credit_card_account_id"))
        if card is None:
            raise ValueError("Payment history references an unknown credit account")
        opened = datetime.strptime(card["date_of_account_open"], "%m/%d/%Y").date()
        paid = datetime.strptime(payment["payment_date"], "%Y-%m-%d").date()
        if not opened <= paid <= get_today():
            raise ValueError("Payment history must be between account opening and today")
    for request in data.get("credit_limit_increase_requests", {}).get("data", {}).values():
        card = cards.get(request.get("credit_card_account_id"))
        if card is not None and request.get("submitted_at"):
            opened = datetime.strptime(card["date_of_account_open"], "%m/%d/%Y").date()
            submitted = datetime.strptime(request["submitted_at"], "%m/%d/%Y").date()
            if not opened <= submitted <= get_today():
                raise ValueError("Prior request must follow account opening")
    if candidate.slot.family != "transaction_disputes" or "statement_duplicates" not in candidate.facts:
        return
    known = json.loads(candidate.task.user_scenario.instructions.known_info)
    for key in ("purchase_channel", "statement_date", "police_report", "requested_resolution"):
        if key not in known or known[key] != candidate.facts.get(key):
            raise ValueError(f"Dispute fact is not available to the user: {key}")
    for goal in candidate.goals:
        if goal["kind"] != "row_match" or goal["table"] != "debit_card_disputes":
            continue
        args = goal["expected"]
        if "online" not in known["purchase_channel"].lower() or args["transaction_type"] != "online_purchase":
            raise ValueError("Dispute channel is inconsistent with customer facts")
        # This recipe reports on the statement day. The public 031 policy gives
        # a $50 timing cap, bounded by the disputed amount (actual tool contract).
        statement = datetime.strptime(known["statement_date"], "%m/%d/%Y").date()
        timing_rule = re.search(r"within 2 business days of statement: Maximum liability \$(\d+)",
                                documents()[BANK + "031"].content)
        if timing_rule is None:
            raise ValueError("Cannot independently derive debit reporting liability")
        cap = int(timing_rule[1])
        if statement != get_today() or args["customer_max_liability_amount"] != min(cap, args["disputed_amount"]):
            raise ValueError("Dispute liability is inconsistent with reporting timing")


def check_incident_transfer_reason(candidate):
    """Check the public incident's cause independently of generator branch labels."""
    from tau3.synthesis.targeted.planning import unpack

    if not candidate.slot.v03_labels:
        return
    actions = [unpack(action.name, action.arguments) for action in candidate.task.evaluation_criteria.actions]
    if not any(name == "emergency_credit_bureau_incident_transfer_1114" for name, _ in actions):
        return
    incident = documents()[CARDS + "012"].content
    reasons = documents()[BANK + "042"].content
    if "backend batch processing error" not in incident or not re.search(
        r"\| technical_system_error \| System error or outage", reasons
    ):
        raise ValueError("Cannot independently derive bureau incident transfer reason")
    required = "technical_system_error"
    goals = [goal for goal in candidate.goals if goal["kind"] == "transfer"]
    transfers = [args for name, args in actions if name == "transfer_to_human_agents"]
    if len(goals) != 1 or len(transfers) != 1 or any(
        record.get("reason") != required for record in [*goals, *transfers]
    ):
        raise ValueError("Bureau incident requires the public system-error transfer reason")


def cli_obligation(candidate, goal, env, calls):
    """Re-derive the outcome from records; do not trust the generator's branch label."""
    data = candidate.task.initial_state.initialization_data.agent_data
    cid = goal["account_id"]
    initial = data["credit_card_accounts"]["data"][cid]
    uid = initial["user_id"]
    tier = {"Bronze Rewards Card": "Entry", "Silver Rewards Card": "Mid", "Gold Rewards Card": "Premium"}[initial["card_type"]]
    age_doc, amount_doc = documents()[LOGISTICS + "005"].content, documents()[LOGISTICS + "006"].content
    row = re.search(rf"\| {tier}-tier\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)%", age_doc)
    if row is None:
        raise ValueError("Cannot independently derive CLI tier policy")
    minimum_age, cooldown, utilization = map(int, row.groups())
    months = int(re.search(rf"{tier}-tier cards: Requires (\d+)", amount_doc)[1])
    percent = int(re.search(rf"{tier}-tier cards: Maximum increase of (\d+)%", amount_doc)[1])
    limit, balance = money(initial["credit_limit"]), money(initial["current_balance"])
    requested = goal["requested_amount"]
    initial_requests = data.get("credit_limit_increase_requests", {}).get("data", {})
    actual_requests = [r for key, r in env.tools.db.credit_limit_increase_requests.data.items()
                       if key not in initial_requests and r.get("credit_card_account_id") == cid]
    final = env.tools.db.credit_card_accounts.data[cid]
    if requested is None or Decimal(requested) * 100 > limit * percent:
        return not actual_requests and money(final["credit_limit"]) == limit
    def parse(value):
        return datetime.strptime(value, "%m/%d/%Y").date()
    decision = None
    if (get_today() - parse(initial["date_of_account_open"])).days < minimum_age:
        decision = "insufficient_account_age"
    elif any(r.get("status") == "APPROVED" and r.get("credit_card_account_id") == cid
             and (get_today() - parse(r["submitted_at"])).days < cooldown for r in initial_requests.values()):
        decision = "cooldown_period_active"
    elif any(r.get("user_id") == uid and r.get("status") in {"PENDING", "SUBMITTED", "UNDER_REVIEW"}
             for r in data.get("transaction_disputes", {}).get("data", {}).values()):
        decision = "pending_disputes"
    elif any(r.get("credit_card_account_id") == cid and r.get("status") in {"PENDING", "PROCESSING", "SHIPPED"}
             for r in data.get("credit_card_orders", {}).get("data", {}).values()):
        decision = "pending_replacement_card"
    elif money(initial.get("past_due_balance", 0)) > 0 or initial["account_status"] != "CURRENT":
        decision = "past_due_balance"
    elif balance * 100 >= limit * utilization:
        decision = "high_utilization"
    else:
        payments = sorted((r for r in data.get("payment_history", {}).get("data", {}).values()
                           if r.get("credit_card_account_id") == cid), key=lambda r: r["payment_date"], reverse=True)[:months]
        if len(payments) < months or any(r["status"] != "ON_TIME" for r in payments):
            decision = "insufficient_payment_history"
    relevant = [c for c in calls if c.get("succeeded") and
                (c["arguments"].get("credit_card_account_id") == cid or c["name"] == "get_user_dispute_history_7291")]
    submitted = [c for c in relevant if c["name"] == "submit_credit_limit_increase_request_7392"]
    terminal = "deny_credit_limit_increase_5848" if decision else "approve_credit_limit_increase_5847"
    decisions = [c for c in relevant if c["name"] == terminal]
    if len(submitted) != 1 or len(decisions) != 1:
        return False
    if submitted[0]["arguments"].get("requested_increase_amount") != requested:
        return False
    start, end = submitted[0]["position"], decisions[0]["position"]
    required = {"get_credit_limit_increase_history_4829", "get_payment_history_6183",
                "get_user_dispute_history_7291", "get_pending_replacement_orders_5765"}
    if not required <= {c["name"] for c in relevant if start < c["position"] < end}:
        return False
    if any(c["name"] == "pay_credit_card_from_checking_9182" for c in calls):
        return False
    if decision:
        return (money(final["credit_limit"]) == limit and len(actual_requests) == 2
                and sum(r.get("status") == "DENIED" and r.get("denial_reason") == decision for r in actual_requests) == 1)
    # Detect the installed official submit→approve collision instead of accepting
    # a changed limit with a stale pending request as a valid synthetic target.
    return (money(final["credit_limit"]) == limit + requested and len(actual_requests) == 1
            and actual_requests[0].get("status") == "APPROVED")
