"""Policy-bound minority branches that prevent aggregate family quotas hiding gaps."""

from datetime import timedelta

from tau3.domains.banking_knowledge.utils import get_today
from tau3.synthesis.catalog import BANK, CARDS, LOGISTICS
from tau3.synthesis.targeted.native.v03_scenarios import credit_card


def account_service(b, count, variant):
    """Correct a documented checking fee or perform an explicitly requested email change."""
    b.verify()
    if b.slot.branch_id == "open":
        # Eligibility is public account evidence, even when no accounts exist.
        b.actions.call("get_all_user_accounts_by_user_id_3847", user_id=b.uid)
        products = ["Light Blue Account", "Blue Account", "Green Account (checking)"]
        for product in products[:min(count, 3)]:
            b.actions.call("open_bank_account_4821", user_id=b.uid, account_type="checking", account_class=product)
            b.goals.append({"kind": "opened_account", "user_id": b.uid, "account_type": "checking", "account_class": product})
        b.intents.append("Open one personal checking account of each of these types: " + ", ".join(products[:min(count, 3)]) + ". I will fund them externally later. No debit card or referral is requested.")
        b.branches.append(["open_products", min(count, 3)])
        return
    if b.slot.branch_id == "email" or (b.slot.branch_id is None and b.slot.difficulty == "1-4"):
        email = f"updated-{b.key}@example.com"
        b.facts["new_email"] = email
        b.actions.add("change_user_email", {"user_id": b.uid, "new_email": email})
        b.field("users", b.uid, email=email)
        b.intents.append("Update only my email address to the new address I provide.")
    else:
        aid = b.account("fee", "Green Fee-Free Account", "900.00", age=800)
        amount = 10 + variant % 3 * 5
        tid = f"wire_fee_{b.key}"
        b.put("bank_account_transaction_history", tid, {"transaction_id": tid, "account_id": aid,
              "date": get_today().strftime("%m/%d/%Y"), "amount": -(12.50 + amount),
              "description": "Incoming domestic wire fee", "type": "fee", "status": "posted"})
        b.query_accounts()
        b.actions.call("get_bank_account_transactions_9173", account_id=aid)
        b.actions.call("apply_checking_account_credit_5829", account_id=aid, amount=amount, credit_type="fee_refund")
        b.goals.append({"kind": "balance", "account_id": aid, "expected": str(900 + amount)})
        b.intents.append("Check the incoming domestic wire fee on my Green Fee-Free checking account against its fee schedule and correct any overcharge. I have not received any credit or refund in the last 14 days.")
        b.docs.update([BANK + "017", "doc_checking_accounts_green_fee-free_account_001"])
    service = "email" if "new_email" in b.facts else "checking_fee"
    # Email spelling is not a distinct policy structure for held-out validation.
    b.branches.append(["account_service", service, None if service == "email" else variant % 3])
    if count > 1 and b.slot.difficulty != "1-4":
        from tau3.synthesis.targeted.native.scenarios import accounts

        accounts(b, count - 1, variant)


def card_service(b, count, variant):
    """Use distinct activation signatures or clear a customer-resolvable velocity block."""
    b.verify()
    for i in range(min(count, 4)):
        aid = b.account(f"service{i}", "Blue Account", "1500.00")
        cid = b.card(aid, f"service{i}", status="PENDING" if b.slot.branch_id == "activate" else "ACTIVE")
        row = b.data["debit_cards"]["data"][cid]
        b.query_accounts()
        b.actions.call("get_debit_cards_by_account_id_7823", account_id=aid)
        if b.slot.branch_id == "unfreeze":
            row.update(status="FROZEN")
            b.actions.call("unfreeze_debit_card_3893", card_id=cid)
            b.field("debit_cards", cid, status="ACTIVE")
            b.facts.setdefault("affected_cards", []).append(row["last_4_digits"])
            b.docs.add(BANK + "027")
        elif b.slot.branch_id == "replacement":
            b.actions.call("close_debit_card_4721", card_id=cid, reason="stolen")
            b.actions.call("order_debit_card_5739", account_id=aid, user_id=b.uid,
                           delivery_option="STANDARD", delivery_fee=0, card_design="CLASSIC",
                           design_fee=0, shipping_address=b.user["address"])
            b.field("debit_cards", cid, status="CLOSED", closure_reason="stolen")
            b.goals.append({"kind": "replacement", "account_id": aid, "delivery_option": "STANDARD", "delivery_fee": 0,
                            "card_design": "CLASSIC", "design_fee": 0})
            b.facts.setdefault("affected_cards", []).append(row["last_4_digits"])
            b.docs.add(BANK + "029")
        elif b.slot.branch_id == "temporary_limit":
            amount = [700, 725, 750][variant % 3]
            b.actions.call("request_temporary_debit_card_limit_increase_8374", card_id=cid, limit_type="atm", new_limit=amount)
            b.field("debit_cards", cid, daily_atm_limit=amount)
            b.facts.setdefault("atm_limit_requests", []).append({"last_four": row["last_4_digits"], "amount": amount})
            b.docs.add(BANK + "040")
        elif b.slot.branch_id == "activate":
            row.update(issue_reason=["lost", "stolen", "fraud"][variant % 3], cvv="582")
            b.actions.call("activate_debit_card_8292", card_id=cid, last_4_digits=row["last_4_digits"],
                           expiration_date=row["expiration_date"], cvv="582", pin="5837")
            b.field("debit_cards", cid, status="ACTIVE")
            b.facts.setdefault("received_cards", []).append({"last_four": row["last_4_digits"],
                "expiration_date": row["expiration_date"], "cvv": "582", "chosen_pin": "5837"})
            b.docs.add(BANK + "024")
        else:
            row.update(velocity_blocked=True)
            b.actions.call("clear_debit_card_fraud_alert_4892", card_id=cid, reason="velocity_clear")
            b.field("debit_cards", cid, velocity_blocked=False)
            b.facts.setdefault("blocked_cards", []).append(row["last_4_digits"])
            b.docs.add(BANK + "042")
    if b.slot.branch_id == "unfreeze":
        b.intents.append("I found the listed cards that I temporarily froze myself after misplacing them. Please unfreeze each one. I recognize all recent transactions and do not want replacements or PIN changes.")
    elif b.slot.branch_id == "replacement":
        b.intents.append("My listed debit cards were stolen. Close each stolen card and order a replacement with classic design and standard shipping to my address on file. I recognize all posted transactions and am not disputing any today.")
    elif b.slot.branch_id == "temporary_limit":
        b.intents.append("Raise the ATM withdrawal limit on each listed card to the amount I specify for 24 hours if policy allows. I have not requested any temporary increase today.")
    else:
        b.intents.append("Activate each replacement debit card I have received. I have the physical cards and recognize all recent purchases." if b.slot.branch_id == "activate" else "My listed cards stopped working after several purchases that I recognize. Please inspect each restriction and restore use if your policy allows it. I made all these purchases myself and have the cards.")
    b.branches.append(["card_service", b.slot.branch_id, min(count, 4), variant % 3])


def retention(b, count, variant):
    """An accepted documented retention offer must leave the credit account open."""
    b.verify()
    for i in range(min(count, 3)):
        waiver = b.slot.branch_id == "retention_waiver"
        cid = credit_card(b, f"retention{i}", "Platinum Rewards Card" if waiver else "Silver Rewards Card")
        b.actions.add("get_credit_card_accounts_by_user", {"user_id": b.uid}) if i == 0 else None
        b.actions.call("get_user_dispute_history_7291", user_id=b.uid) if i == 0 else None
        b.actions.call("get_pending_replacement_orders_5765", credit_card_account_id=cid)
        b.actions.call("get_closure_reason_history_8293", credit_card_account_id=cid)
        b.actions.call("log_credit_card_closure_reason_4521", credit_card_account_id=cid, user_id=b.uid,
                       closure_reason="annual_fee" if waiver else "simplifying_finances")
        if waiver:
            expiry = (get_today() + timedelta(days=365)).strftime("%m/%d/%Y")
            b.actions.call("apply_credit_card_account_flag_6147", credit_card_account_id=cid, user_id=b.uid,
                           flag_type="annual_fee_waived", expiration_date=expiry, reason="loyalty_benefit")
            b.goals.append({"kind": "required_operation", "name": "apply_credit_card_account_flag_6147",
                           "arguments": {"credit_card_account_id": cid, "flag_type": "annual_fee_waived", "expiration_date": expiry, "reason": "loyalty_benefit"}})
        else:
            b.actions.call("apply_statement_credit_8472", user_id=b.uid, credit_card_account_id=cid, amount=20, reason="retention_offer")
            b.goals.append({"kind": "required_operation", "name": "apply_statement_credit_8472",
                           "arguments": {"credit_card_account_id": cid, "amount": 20, "reason": "retention_offer"}})
        b.field("credit_card_accounts", cid, status="ACTIVE")
        b.facts.setdefault("retention_cards", []).append(b.data["credit_card_accounts"]["data"][cid]["last_4_digits"])
    b.intents.append("I am considering closing the listed cards because of their annual fees. I would keep each if you can waive its annual fee for the next year under the bank's policy." if b.slot.branch_id == "retention_waiver" else "I am considering closing the listed cards to simplify my finances. General benefit reminders do not change my mind, but I will keep each if you offer the standard monetary statement-credit retention incentive for its tier. I prefer cash credit over reward points.")
    b.docs.update(LOGISTICS + n for n in ("001", "002", "003"))
    b.branches.append(["retention", b.slot.branch_id, min(count, 3)])


def user_action(b, count, variant):
    """Execute an authorized customer action using actual user tools."""
    b.verify()
    cid = credit_card(b, "user", "Silver Rewards Card")
    if b.slot.branch_id == "referral_link":
        b.grant("get_referral_link", {"user_id": b.uid, "card_name": "Silver Rewards Card"})
        b.goals.append({"kind": "user_execution", "tool": "get_referral_link", "arguments": {"user_id": b.uid, "card_name": "Silver Rewards Card"}})
        b.intents.append("Help me generate my personal referral link for my existing Silver Rewards Card to share with a friend. I am not requesting a new application or purchase.")
    else:
        amount = 80 + variant
        args = {"user_id": b.uid, "credit_card_type": "Silver Rewards Card", "merchant_name": "Cedar Books",
                "amount": amount, "category": "Shopping"}
        b.user_tools.add("submit_transaction")
        b.actions.add("submit_transaction", args, "user")
        b.goals.append({"kind": "user_execution", "tool": "submit_transaction", "arguments": args})
        b.facts["planned_purchase"] = args
        b.intents.append(f"Help me complete my planned ${amount} purchase at Cedar Books on my Silver Rewards Card and confirm the posted purchase. I will actually make the purchase using my app when guided; do not claim it happened beforehand.")
    b.field("credit_card_accounts", cid, status="ACTIVE")
    b.branches.append(["user_action", b.slot.branch_id, variant % 3])
    if count > 1:
        from tau3.synthesis.targeted.native.scenarios import handoff

        handoff(b, count - 1, variant)


def incident(b, count, variant):
    """Actual repeated customer requests drive the public incident escalation sequence."""
    if b.slot.branch_id == "emergency":
        b.actions.call("emergency_credit_bureau_incident_transfer_1114")
        b.actions.add("transfer_to_human_agents", {"summary": "Credit monitoring alert during the bureau reporting incident.", "reason": "technical_system_error"})
        b.goals.extend([{"kind": "transfer", "reason": "technical_system_error"},
                        {"kind": "forbidden", "tools": ["log_verification"]},
                        {"kind": "required_operation", "name": "emergency_credit_bureau_incident_transfer_1114", "arguments": {}}])
        b.intents.append("My credit monitoring service reported a sudden large score drop today even though my card account is current. I need urgent help with the credit bureau reporting problem.")
        b.docs.add(CARDS + "012")
        b.branches.append(["bureau_incident"])
        return
    b.user_tools.add("request_human_agent_transfer")
    if b.slot.branch_id == "payment_incident":
        names = ["initial_transfer_to_human_agent_1822", "initial_transfer_to_human_agent_0218"]
        b.intents.append("My credit card payment was deducted from my checking account but is not reflected in my card statement balance. I want a human specialist to fix this.")
        b.docs.add(CARDS + "011")
        b.goals.append({"kind": "forbidden", "tools": ["log_verification"]})
    else:
        b.verify()
        credit_card(b, "declining", balance=100, limit=8000)
        names = ["initial_transfer_to_human_agent_0218"] * 3
        b.intents.append("My credit card purchases keep being declined even though I have available credit. I am frustrated by the repeated declines and insist on a human agent after any troubleshooting.")
        b.docs.add(CARDS + "010")
    for name in names:
        b.actions.add("request_human_agent_transfer", {}, "user")
        b.actions.call(name)
    b.actions.add("request_human_agent_transfer", {}, "user")
    reason = "technical_system_error" if b.slot.branch_id == "payment_incident" else "customer_frustrated_demands_human"
    b.actions.add("transfer_to_human_agents", {"summary": "Customer continues requesting specialist assistance for the reported incident.", "reason": reason})
    b.goals.append({"kind": "transfer", "reason": reason})
    b.goals.append({"kind": "operation_sequence", "names": names})
    b.facts["persistence"] = "Request a human through your available request tool on your first turn. If the agent says the transfer has not completed, insist again and execute another request. Continue when the agent says the transfer is still pending, unsuccessful, or merely planned. When the agent clearly reports that the transfer succeeded or has been completed, acknowledge the completed handoff and end the conversation. Do not require the human specialist to join this chat or answer here; that occurs after the handoff. A future promise to attempt transfer is not a completion notice."
    b.branches.append(["incident", b.slot.branch_id])
