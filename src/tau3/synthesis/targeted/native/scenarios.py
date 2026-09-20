"""New customer states and executable banking workflows, before paraphrasing."""

import json
import random
from datetime import timedelta
from decimal import Decimal
from functools import lru_cache

from tau3.data_model.tasks import (
    Description,
    EvaluationCriteria,
    InitializationData,
    InitialState,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau3.domains.banking_knowledge.utils import get_now, get_today
from tau3.synthesis.catalog import BANK, CARDS
from tau3.synthesis.scenarios import Actions
from tau3.synthesis.storage import digest
from tau3.synthesis.targeted.native.catalog import (
    BASE_DOCS,
    DISPUTE_DOCS,
    PERSONAL_CARDS,
    REFERRALS,
    evidence_for,
    product_evidence,
)
from tau3.synthesis.targeted.native.models import NativeCandidate
from tau3.synthesis.targeted.planning import unpack


class Builder:
    """Compile fresh tables, public customer facts and private business obligations."""

    def __init__(self, slot, attempt, variant):
        self.slot, self.attempt, self.variant = slot, attempt, variant
        self.key = digest(["banking-native-v2", slot.pair_id or slot.seed, attempt, variant])[:14]
        self.rng = random.Random(int(self.key, 16))
        self.uid = "native_" + self.key
        self.user = {
            "user_id": self.uid, "name": f"Morgan Reed {self.key[:6]}",
            "date_of_birth": "06/18/1985", "address": f"{self.rng.randint(100, 9999)} Cedar Lane, Austin, TX 78701",
            "email": f"{self.key}@example.com", "phone_number": f"512-555-{self.rng.randint(1000, 9999)}",
        }
        self.data = {"users": {"data": {self.uid: self.user}}}
        if slot.runtime_revision != "official":
            self.put("task_config", "native_runtime", {"revision": slot.runtime_revision})
        self.actions, self.goals, self.intents = Actions(), [], []
        self.facts = {"identity": self.user, "interaction": "Disclose all goals initially. Provide your own identity details when asked; confirm suitable proposed actions. Decline unrelated products, extra referrals and other changes. Never claim that a tool succeeded unless its result says so."}
        self.docs = set(BASE_DOCS)
        self.user_tools = set()
        self.branches = []
        self.verified = False
        self.discovery = slot.difficulty not in {"1-4"}

    def put(self, table, key, row):
        self.data.setdefault(table, {"data": {}})["data"][key] = row
        return key

    def verify(self):
        if not self.verified:
            self.actions.add("log_verification", {**self.user, "time_verified": get_now().strftime("%Y-%m-%d %H:%M:%S EST")})
            self.goals.append({"kind": "verification", "user_id": self.uid})
            self.verified = True

    def account(self, suffix, product="Light Blue Account", balance="1000.00", kind="checking", age=None):
        age = getattr(self, "account_age", 300) if age is None else age
        key = f"acct_{self.key}_{suffix}"
        return self.put("accounts", key, {
            "account_id": key, "user_id": self.uid, "class": kind, "account_type": kind,
            "level": product, "account_class": product, "status": "OPEN",
            "date_opened": (get_today() - timedelta(days=age)).strftime("%m/%d/%Y"),
            "current_holdings": str(balance),
        })

    def card(self, account, suffix, status="ACTIVE", locked=False):
        key = f"dc_{self.key}_{suffix}"
        return self.put("debit_cards", key, {
            "card_id": key, "user_id": self.uid, "account_id": account, "status": status,
            "last_4_digits": f"{self.rng.randint(1000, 9999)}", "date_issued": "01/15/2025",
            "issue_reason": "first_card", "expiration_date": "01/29", "pin_locked": locked,
            "pin_attempts_remaining": 0 if locked else 3, "daily_atm_limit": 500,
            "daily_purchase_limit": 1000, "pending_transactions": False, "pending_refunds": False,
        })

    def query_accounts(self):
        name = "get_all_user_accounts_by_user_id_3847"
        if self.discovery and name not in self.actions.unlocked:
            self.actions.call(name, user_id=self.uid)

    def grant(self, name, arguments):
        self.user_tools.add("call_discoverable_user_tool")
        self.actions.add("give_discoverable_user_tool", {"discoverable_tool_name": name, "arguments": json.dumps(arguments, sort_keys=True)})
        self.actions.add("call_discoverable_user_tool", {"discoverable_tool_name": name, "arguments": json.dumps(arguments, sort_keys=True)}, "user")

    def field(self, table, key, **expected):
        self.goals.append({"kind": "fields", "table": table, "id": key, "expected": expected})

    def finish(self):
        names = [unpack(a.name, a.arguments)[0] for a in self.actions.items
                 if a.name not in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}]
        self.docs = set(evidence_for(names, self.docs))
        shape, group = self.shape_and_group()
        dependencies = []
        for action in self.actions.items:
            name, args = unpack(action.name, action.arguments)
            if name in {"pay_credit_card_from_checking_9182", "transfer_funds_between_bank_accounts_7291",
                        "file_debit_card_transaction_dispute_6281"}:
                dependencies.append({"operation": name, "entity_fields": sorted(k for k in args if k.endswith("_id"))})
        shape["dependencies"] = dependencies
        goal = " ".join(self.intents)
        self.facts["goal"] = goal
        task = Task(
            id=f"native_{self.slot.split}_{self.slot.index:05d}_{self.key}",
            description=Description(purpose=f"Native banking {self.slot.family}"),
            initial_state=InitialState(initialization_data=InitializationData(agent_data=self.data)),
            user_scenario=UserScenario(instructions=StructuredUserInstructions(
                domain="banking_knowledge", reason_for_call=goal,
                known_info=json.dumps(self.facts, ensure_ascii=False),
                unknown_info="You do not know internal tools, policies, eligibility thresholds, correct rewards or undisclosed transaction details. Account IDs explicitly listed in your own check instructions are known; other internal IDs are unknown until the agent supplies them. Do not invent them. Use only your private customer facts and tools actually granted to you.",
                task_instructions="Follow your stated goals and constraints. Answer identity and factual questions accurately. Preserve exact amounts. Use only tools you actually have; for a granted tool, follow the agent's actual instructions and report its actual result. Do not infer an approval or completed operation from a promise. End after all requests have been handled, or a justified refusal/transfer is complete.",
            )),
            evaluation_criteria=EvaluationCriteria(actions=self.actions.items, reward_basis=["DB"]),
            required_documents=sorted(self.docs), user_tools=sorted(self.user_tools),
        )
        return NativeCandidate(
            task=task, slot=self.slot, candidate=self.attempt, facts=self.facts, goals=self.goals,
            graph=shape, group_id=group,
            business_fingerprint=digest({"initial_state": task.initial_state.model_dump(mode="json"), "goals": self.goals}),
        )

    def shape_and_group(self):
        """Group executable policy structures, without relying on instance IDs."""
        shape = {"family": self.slot.family, "branches": self.branches,
                 "operations": [(a.requestor, unpack(a.name, a.arguments)[0]) for a in self.actions.items]}
        grouping = shape
        if self.slot.pair_id:
            grouping = {"family": self.slot.family, "pair_mechanism": "utilization_threshold",
                        "tier": self.variant % 3, "limit_band": (self.variant // 3) % 3}
        return shape, digest(grouping)


def disputes(b, count, variant):
    """Actual user disputes precede approved agent adjustments, with correct distractors."""
    b.verify()
    card_type = "Silver Rewards Card" if variant % 2 else "Bronze Rewards Card"
    auto = bool(variant % 3)
    cid = f"cc_{b.key}"
    b.put("credit_card_accounts", cid, {"account_id": cid, "user_id": b.uid,
        "card_type": card_type, "status": "ACTIVE", "account_status": "CURRENT",
        "date_of_account_open": "01/15/2024", "current_balance": "$500.00"})
    b.put("task_config", "dispute_settings", {"auto_resolve_disputes": auto})
    if b.discovery:
        b.actions.add("get_credit_card_transactions_by_user", {"user_id": b.uid})
    for i in range(count + 1):
        category = ["Travel", "Software", "Groceries"][(i + variant) % 3]
        amount = b.rng.randint(10, 300)
        rate = 4 if card_type.startswith("Silver") and category in {"Travel", "Software"} else 1
        correct = amount * rate
        earned = correct if i == count else max(0, correct - b.rng.randint(1, correct))
        tid = f"txn_{b.key}_{i}"
        b.put("credit_card_transaction_history", tid, {"transaction_id": tid, "user_id": b.uid,
            "credit_card_account_id": cid, "credit_card_type": card_type,
            "merchant_name": f"{category} Merchant {i}", "transaction_amount": f"${amount:.2f}",
            "transaction_date": "10/15/2025", "category": category, "status": "COMPLETED",
            "rewards_earned": f"{earned} points", "promotion_applied": False})
        if i < count:
            b.grant("submit_cash_back_dispute_0589", {"user_id": b.uid, "transaction_id": tid})
            if auto:
                b.actions.call("update_transaction_rewards_3847", transaction_id=tid, new_rewards_earned=f"{correct} points")
        b.goals.append({"kind": "cashback", "transaction_id": tid, "auto": auto,
                        "affected": i < count})
    b.docs.update(DISPUTE_DOCS)
    b.branches.append(["disputes", card_type, auto, count])
    b.intents.append("Please inspect all purchases on my cash back statement, identify every incorrectly credited purchase, and help me dispute only those purchases. Correct the rewards only after an actual approved resolution.")
    b.facts["purchase_context"] = "All purchases were direct with merchants; no returns, promotions or third-party wallets. I do not know which transactions or reward rates are wrong. I can follow instructions in the mobile app. If the app says pending, accept review; do not invent a resolution."


def accounts(b, count, variant):
    """Card closure and balance consolidation are prerequisites to account closure."""
    b.verify()
    if b.slot.difficulty == "1-4":
        if b.slot.v03_labels:
            raise ValueError("Opening eligibility discovery cannot fit the short action bucket")
        product = ["Light Blue Account", "Blue Account", "Green Account (checking)"][variant % 3]
        b.actions.call("open_bank_account_4821", user_id=b.uid, account_type="checking", account_class=product)
        b.goals.append({"kind": "opened_account", "user_id": b.uid, "account_type": "checking", "account_class": product})
        b.intents.append(f"Open one {product} personal checking account after checking eligibility. I will fund it externally later; no card or referral is requested.")
        b.branches.append(["open", product])
        return
    relationship = f"acct_{b.key}_relationship"
    existing = b.data.get("accounts", {}).get("data", {})
    blue = [key for key, row in existing.items() if row.get("level") == "Blue Account"]
    if len(blue) > 1:
        raise ValueError("Ambiguous retained Blue Account")
    if relationship in existing:
        keep = relationship
        b.data["accounts"]["data"][keep].update(level="Blue Account", account_class="Blue Account")
    else:
        keep = blue[0] if blue else b.account("keep", "Blue Account", "1000.00")
    b.query_accounts()
    total = Decimal(b.data["accounts"]["data"][keep]["current_holdings"])
    for i in range(min(count, 3)):
        balance = Decimal(b.rng.randint(2, 20) * 100)
        aid = b.account(f"old{i}", balance=str(balance))
        if variant & (1 << i):
            card = b.card(aid, f"old{i}")
            if b.discovery:
                b.actions.call("get_debit_cards_by_account_id_7823", account_id=aid)
            b.actions.call("close_debit_card_4721", card_id=card, reason="account_closing")
            b.field("debit_cards", card, status="CLOSED", closure_reason="account_closing")
        b.actions.call("transfer_funds_between_bank_accounts_7291", source_account_id=aid,
                       destination_account_id=keep, amount=float(balance))
        b.actions.call("close_bank_account_7392", account_id=aid, reason="Customer requested closure")
        b.goals.append({"kind": "closed_account", "account_id": aid})
        total += balance
    if count > 3:
        from tau3.domains.banking_knowledge.tools import _deterministic_id

        product = "Silver Plus Account"
        if any(row.get("level") == product for row in b.data["accounts"]["data"].values()):
            raise ValueError("Composition must not request a duplicate existing savings product")
        new_id = _deterministic_id(f"account:{b.uid}:savings:{product}")
        b.actions.call("open_bank_account_4821", user_id=b.uid, account_type="savings", account_class=product)
        b.actions.call("transfer_funds_between_bank_accounts_7291", source_account_id=keep, destination_account_id=new_id, amount=1000)
        total -= 1000
        b.goals.append({"kind": "opened_account", "user_id": b.uid, "account_type": "savings", "account_class": product})
        b.goals.append({"kind": "balance", "account_id": new_id, "expected": "1000"})
        b.docs.update([BANK + "002", "doc_savings_accounts_silver_plus_account_001"])
        b.intents.append("After consolidation, open one Silver Plus savings account and immediately fund it with $1,000 from my retained Blue Account.")
    b.goals.append({"kind": "balance", "account_id": keep, "expected": str(total)})
    b.docs.update([BANK + "005", BANK + "025"])
    b.branches.append(["consolidation", min(count, 3), variant & ((1 << min(count, 3)) - 1), count > 3])
    b.intents.append("Move all money from my old Light Blue accounts into my existing Blue Account, close associated old debit cards and then the emptied old accounts. Keep my Blue Account. Record the account closure reason exactly as 'Customer requested closure'.")


def debit(b, count, variant):
    """Select different real card workflows from customer facts and database evidence."""
    b.verify()
    b.query_accounts()
    customer_cards = []
    security_review = False
    for i in range(min(count, 4)):
        mode = (variant + i) % 6
        age = 59 + variant % 2 if mode == 4 else 900
        aid = b.account(f"debit{i}", ["Light Blue Account", "Blue Account", "Green Account (checking)", "Evergreen Account"][i], "1500.00", age=age)
        cid = b.card(aid, i, status="FROZEN" if mode == 1 else "ACTIVE", locked=mode == 5)
        card = b.data["debit_cards"]["data"][cid]
        if b.discovery:
            b.actions.call("get_debit_cards_by_account_id_7823", account_id=aid)
            if count > 4 or mode in {4, 5}:
                merchant = f"Cedar Market {i}"
                txn_id = f"banktxn_{b.key}_{i}"
                b.put("bank_account_transaction_history", txn_id, {"transaction_id": txn_id,
                    "account_id": aid, "date": "11/12/2025", "description": merchant,
                    "amount": -23.5, "type": "purchase", "status": "posted"})
                b.actions.call("get_bank_account_transactions_9173", account_id=aid)
                if count > 4:
                    b.goals.append({"kind": "communicate", "contains": merchant})
        fact = {"last_four": card["last_4_digits"], "new_pin": "5837"}
        if mode in {0, 1}:
            fact["problem"] = "I forgot my PIN. I froze the card myself and have now found it; unfreeze it and help reset the PIN." if mode else "I forgot my PIN but have not tried entering it. Please help reset it so I can use my card again."
            if mode == 1:
                b.actions.call("unfreeze_debit_card_3893", card_id=cid)
            b.actions.call("reset_debit_card_pin_6284", card_id=cid, last_4_digits=card["last_4_digits"], new_pin="5837")
            b.field("debit_cards", cid, status="ACTIVE", pin_locked=False, pin_attempts_remaining=3)
        elif mode == 2:
            fact["problem"] = "I misplaced this card at home and am still looking for it. Temporarily prevent spending; do not cancel or replace it."
            b.actions.call("freeze_debit_card_3892", card_id=cid)
            b.field("debit_cards", cid, status="FROZEN")
        elif mode == 3:
            fact["problem"] = "This card was stolen. Permanently deactivate it and order a replacement with standard delivery and classic design to my address on file. There are no transactions I dispute today."
            b.actions.call("close_debit_card_4721", card_id=cid, reason="stolen")
            b.field("debit_cards", cid, status="CLOSED", closure_reason="stolen")
            if b.data["accounts"]["data"][aid]["level"] == "Light Blue Account":
                # Public doc 029 forbids ordering an entry-tier replacement on
                # the same call as closure, even though the tool accepts it.
                b.goals.append({"kind": "replacement_wait", "account_id": aid, "hours": 48})
                b.branches.append(["replacement_wait", "entry", 48])
            else:
                b.actions.call("order_debit_card_5739", account_id=aid, user_id=b.uid, delivery_option="STANDARD", delivery_fee=0,
                               card_design="CLASSIC", design_fee=0, shipping_address=b.user["address"])
                b.goals.append({"kind": "replacement", "account_id": aid, "delivery_option": "STANDARD", "delivery_fee": 0,
                                "card_design": "CLASSIC", "design_fee": 0})
            b.docs.add(BANK + "029")
        elif mode == 4:
            requested = 750 + int(variant % 4 == 0)
            allowed = age >= 60 and requested <= 750
            fact["problem"] = f"Please raise my ATM limit to ${requested} for the next 24 hours if policy permits it. Otherwise explain why and leave the limit unchanged. I have made no previous increase request today."
            if allowed:
                b.actions.call("request_temporary_debit_card_limit_increase_8374", card_id=cid, limit_type="atm", new_limit=requested)
            b.field("debit_cards", cid, daily_atm_limit=requested if allowed else 500)
            b.docs.add(BANK + "040")
            b.branches.append(["temporary_limit_boundary", age >= 60, requested <= 750])
        else:
            card.update(pin_lock_reason="security_hold", pin_locks_last_90_days=0)
            fact["problem"] = "My card stopped accepting PIN transactions. I do not know why. Please check its records, do what is permitted, and connect me to the security team if its restrictions require specialist help. I recognize my recent posted purchases."
            security_review = True
            b.field("debit_cards", cid, status="ACTIVE", pin_locked=True, pin_lock_reason="security_hold")
            b.docs.update([BANK + "041", BANK + "042"])
        customer_cards.append(fact)
    b.facts["my_cards"] = customer_cards
    b.docs.update([BANK + n for n in ("023", "025", "026", "027")])
    if security_review:
        b.actions.add("transfer_to_human_agents", {"summary": "Security hold requires specialist handling; completed other permitted card requests.", "reason": "fraud_or_security_concern"})
        b.goals.append({"kind": "transfer", "reason": "fraud_or_security_concern", "after_operations": True})
    b.branches.append(["debit", [((variant + i) % 6) for i in range(min(count, 4))]])
    b.intents.append("Help with each of my cards according to the problems I describe. Explain and complete the appropriate changes for every affected card, without changing cards that do not need it.")
    if count > 4:
        b.intents.append("Also look up and identify the merchant of the most recent posted purchase on every linked checking account; I recognize those purchases and am not disputing them.")
        b.branches.append(["statement_review", min(count, 4)])


def handoff(b, count, variant):
    """Actual mobile deposits fund subsequent agent transfers, never fictitious handoffs."""
    b.verify()
    destination = b.account("deposit_destination", "Silver Plus Account", "3000.00", "savings")
    b.query_accounts()
    checks = []
    destination_balance = Decimal("3000")
    for i in range(min(count, 3)):
        aid = b.account(f"deposit{i}", "Light Blue Account", "100.00")
        card = b.card(aid, f"deposit{i}") if variant % 4 == 3 else None
        if card and b.discovery:
            b.actions.call("get_debit_cards_by_account_id_7823", account_id=aid)
        amount = b.rng.randint(10, 90)
        checks.append({"account_id": aid, "check_amount": amount, "check_label": f"paycheck-{i}"})
        b.grant("deposit_check_3847", {"account_id": aid, "check_amount": amount})
        if variant % 2:
            if card:
                b.actions.call("close_debit_card_4721", card_id=card, reason="account_closing")
                b.field("debit_cards", card, status="CLOSED", closure_reason="account_closing")
            b.actions.call("transfer_funds_between_bank_accounts_7291", source_account_id=aid,
                           destination_account_id=destination, amount=100 + amount)
            b.actions.call("close_bank_account_7392", account_id=aid, reason="Customer requested closure")
            b.goals.append({"kind": "closed_account", "account_id": aid})
            destination_balance += 100 + amount
        else:
            b.goals.append({"kind": "balance", "account_id": aid, "expected": str(100 + amount)})
        b.goals.append({"kind": "user_execution", "tool": "deposit_check_3847", "arguments": {"account_id": aid, "check_amount": amount}})
    b.goals.append({"kind": "balance", "account_id": destination, "expected": str(destination_balance)})
    b.facts["checks_in_my_possession"] = checks
    b.docs.update(["doc_checking_accounts_light_blue_account_001", BANK + "005", BANK + "025"])
    b.branches.append(["handoff", min(count, 3), bool(variant % 2)])
    b.intents.append("Guide me through depositing each listed check into its specified account using the mobile app. I have the checks with me; do not treat instructions as a completed deposit." + (" After each deposit succeeds, close its linked debit card if any, move the checking account's entire resulting balance into my existing Silver Plus savings account, and close that emptied checking account. Record the account closure reason exactly as 'Customer requested closure'. Keep the savings account." if variant % 2 else " Leave the deposited money in the specified checking accounts."))


def optimization(b, count, variant):
    """Compare all eligible personal products using financial objectives and constraints."""
    b.docs.update(product_evidence())
    if variant % 6 == 4:
        savings_correction(b, variant)
    elif variant % 2:
        # The user has no other checking history or completed/pending referrals.
        tenure = [30, 45, 60, 90][(variant // 2) % 4]
        b.account_age = tenure
        deposit = [300, 500, 1000, 2000][(variant // 3) % 4]
        b.account("relationship", "Light Blue Account", "1500.00", age=tenure)
        age = 65 if variant % 5 == 0 else 30
        eligible = {name: values[0] + values[1] for name, values in REFERRALS.items()
                    if deposit >= values[2] and tenure >= values[3] and values[4] <= age <= values[5]}
        if not eligible:
            raise ValueError("No eligible referral")
        selected = max(eligible, key=eligible.get)
        if list(eligible.values()).count(eligible[selected]) != 1:
            raise ValueError("Ambiguous referral optimization")
        b.verify()
        b.query_accounts()
        b.user_tools.add("submit_referral")
        b.actions.add("submit_referral", {"user_id": b.uid, "account_type": selected}, "user")
        b.facts["referral_constraints"] = {"friend_age": age, "friend_deposit": deposit,
            "friend_deposit_days": 30, "objective": "maximize the sum of my referral bonus and my friend's welcome bonus",
            "annual_referrals": 0, "other_checking_history": "none"}
        b.goals.append({"kind": "referral", "user_id": b.uid, "deposit": deposit, "tenure": tenure, "age": age})
        b.intents.append("Compare every checking referral program my friend and I qualify for, maximize our combined cash bonuses, and help me actually submit the referral for the best option. My friend's funding and age limits are firm.")
        b.branches.append(["referral", selected, tenure in (30, 45, 60), age >= 62])
    else:
        monthly = [150, 300, 500, 1200, 8000][(variant // 2) % 5]
        travel = variant % 3 == 0
        subscription = variant % 4 == 0
        score = [680, 720, 750, 800][(variant // 3) % 4]
        offers = {}
        for name, (minimum, fee, ordinary, enhanced, required) in PERSONAL_CARDS.items():
            if score < minimum or (required and not subscription) or name == "Diamond Elite Card":
                continue
            rate = Decimal(enhanced if travel else ordinary)
            benefit = Decimal(monthly * 12) * rate / 100 - Decimal(fee)
            if name == "Platinum Rewards Card" and monthly >= 7500:
                benefit += 150
            offers[name] = benefit
        selected = max(offers, key=offers.get)
        if list(offers.values()).count(offers[selected]) != 1:
            raise ValueError("Ambiguous card optimization")
        b.user_tools.add("apply_for_credit_card")
        income = b.rng.randint(80, 180) * 1000
        b.actions.add("apply_for_credit_card", {"card_type": selected, "customer_name": b.user["name"],
                      "annual_income": income, "rho_bank_subscription": subscription}, "user")
        b.facts["card_preferences"] = {"monthly_spend": monthly, "category": "Travel" if travel else "Groceries",
            "credit_score": score, "annual_income": income, "rho_bank_subscription": subscription,
            "diamond_invitation": False, "business_owner": False,
            "objective": "maximize second-year annual cash-equivalent rewards minus annual fees; same eligible spend every month; exclude one-time welcome bonuses; pay in full; domestic transactions only",
            "occupation": "environmental science researcher", "additional_constraint": "Do not subscribe to a paid plan I do not already have; do not prioritize environmental branding over cash value."}
        b.goals.append({"kind": "card_choice", "customer_name": b.user["name"], "income": income,
                       "monthly": monthly, "travel": travel, "subscription": subscription, "score": score})
        b.intents.append("Compare all eligible personal credit cards on the stated second-year financial objective, then help me submit one application for the best choice. My profession should not change the financial objective.")
        b.branches.append(["card_choice", selected, travel, subscription, monthly >= 7500])
    if count > 1:
        # A second, genuine customer goal creates a multi-business workflow.
        accounts(b, count - 1, variant)


def savings_correction(b, variant):
    """Correct a full-year APY shortfall, then submit the actual backend report."""
    b.verify()
    principal = Decimal(12000 if variant % 4 else 18000)
    base = Decimal("3.0") if principal < 15000 else Decimal("4.5")
    expected = base + Decimal("0.35") + Decimal("0.025")
    actual = base
    paid = principal * actual / 100
    correction = principal * (expected - actual) / 100
    b.account("apy_linked", "Blue Account", "1800.00", age=800)
    aid = b.account("apy_savings", "Silver Plus Account", str(principal + paid), "savings", age=800)
    b.data["accounts"]["data"][aid].update(
        interest_principal=str(principal), interest_period_years=1,
        interest_period_deposits=0, interest_period_withdrawals=0,
        relationship_bonus_eligible=True, relationship_eligible_entire_period=True,
        linked_checking_entire_period="Blue Account", credited_apy=str(actual))
    tid = f"interest_{b.key}"
    b.put("bank_account_transaction_history", tid, {"transaction_id": tid, "account_id": aid,
        "date": get_today().strftime("%m/%d/%Y"), "description": f"Annual interest statement: principal ${principal}, full one-year holding period; no additions or withdrawals; applied APY {actual}%.",
        "amount": float(paid), "type": "interest", "status": "posted"})
    b.query_accounts()
    b.actions.call("get_bank_account_transactions_9173", account_id=aid)
    b.actions.call("apply_savings_account_credit_6831", account_id=aid, amount=float(correction), credit_type="interest_correction")
    b.actions.call("submit_interest_discrepancy_report_7294", account_id=aid, user_id=b.uid,
                   expected_apy=float(expected), actual_apy=float(actual), amount_difference=float(correction))
    b.goals.append({"kind": "savings_correction", "account_id": aid, "user_id": b.uid, "interest_transaction": tid})
    b.facts["interest_concern"] = {"product": "Silver Plus Account", "statement_principal": str(principal),
        "period": "full one-year APY holding period, no new deposits or withdrawals",
        "relationship": "I held my Blue Account throughout the whole period and my statement marks me eligible for the relationship bonus throughout. I hold no credit cards."}
    b.intents.append("Please verify the interest on my Silver Plus annual statement using all benefits I qualified for throughout that year. If it was underpaid, correct the missing interest and file the required investigation report. Do not change my account products.")
    b.docs.update([BANK + "012", BANK + "043", BANK + "044",
                   "doc_checking_accounts_blue_account_003", "doc_savings_accounts_silver_plus_account_001",
                   "doc_savings_accounts_silver_plus_account_002", "doc_savings_accounts_silver_plus_account_009"])
    b.branches.append(["savings_correction", principal >= 15000, "linked_and_relationship"])


def identity(b, count, variant):
    """Necessary verification followed by a real, authorized bank operation."""
    b.verify()
    b.facts["identity_disclosure"] = "Give your full name initially; supply requested identity fields accurately when asked. Do not tell the agent which database record to write."
    if variant % 2:
        debit(b, count, variant)
    else:
        accounts(b, count, variant)
    b.branches.append(["identity", "verify_before_write"])


def boundary(b, count, variant):
    """Actual escalation obligations include exception cases requiring no verification."""
    if count == 1:
        reason = ["legal_or_regulatory_matter", "deceased_account_holder", "fraud_or_security_concern"][variant % 3]
        goals = {"legal_or_regulatory_matter": "I received a court garnishment notice and need a specialist to handle the legal order; please connect me.",
                 "deceased_account_holder": "I need to report a deceased account holder and discuss estate processing with a specialist.",
                 "fraud_or_security_concern": "I received a credit monitoring alert for a large score drop today even though my credit card is current; I need urgent help."}
        if reason == "fraud_or_security_concern":
            b.actions.call("emergency_credit_bureau_incident_transfer_1114")
            b.docs.add(CARDS + "012")
            b.goals.append({"kind": "forbidden", "tools": ["log_verification"]})
        # Keep the graph's stable incident identity across the correction. The
        # customer's symptom denotes the documented bank system error, not fraud.
        actual_reason = "technical_system_error" if reason == "fraud_or_security_concern" and b.slot.v03_labels else reason
        b.actions.add("transfer_to_human_agents", {"summary": goals[reason], "reason": actual_reason})
        b.goals.append({"kind": "transfer", "reason": actual_reason})
        b.intents.append(goals[reason])
        b.branches.append(["boundary", reason])
    else:
        # Complete an available self-service request, then honor a distinct followup.
        debit(b, count, variant)
        if any(g["kind"] == "transfer" for g in b.goals):
            return
        reason = "request_completed_customer_wants_human_followup"
        b.intents.append("Once these card requests are finished, I still want to speak with a person for follow-up questions; please complete the changes before transferring me.")
        b.actions.add("transfer_to_human_agents", {"summary": "Completed card requests; customer requests followup.", "reason": reason})
        b.goals.append({"kind": "transfer", "reason": reason, "after_operations": True})
        b.branches.append(["boundary", "complete_then_transfer"])


GENERATORS = {"disputes": disputes, "debit": debit, "accounts": accounts,
              "optimization": optimization, "identity": identity, "handoff": handoff, "boundary": boundary}


@lru_cache(maxsize=512)
def v03_holdout_groups(family, branch, difficulty, paired, mechanism):
    """Reserve structural groups deterministically, avoiding empty hash-modulo strata."""
    from tau3.synthesis.targeted.native.models import NativeSlot
    from tau3.synthesis.targeted.native.v03_scenarios import install_generators

    slot = NativeSlot(index=0, split="pilot", family=family, difficulty=difficulty,
        origin="current", seed=43000, trial_seeds=[1, 2, 3, 4], evidence_ids=["partition-probe"],
        branch_id=branch, pair_id="partition-probe" if paired else None, mechanism=mechanism,
        v03_labels=["A"])
    generator = {**GENERATORS, **install_generators()}[family]
    lower, upper = map(int, difficulty.split("-"))
    groups = set()
    for variant in range(48):
        for count in range(1, 9):
            builder = Builder(slot, 0, variant)
            try:
                generator(builder, count, variant)
            except ValueError:
                continue
            if lower <= len(builder.actions.items) <= upper:
                groups.add(builder.shape_and_group()[1])
    # A single structure is training-only. Validation cannot claim independence
    # by assigning a new customer ID to an identical execution graph.
    return frozenset(sorted(groups)[:max(1, len(groups) // 5)]) if len(groups) > 1 else frozenset()


def compile_candidate(slot, attempt=0):
    """Choose a feasible graph in the requested bucket without using teacher outcomes."""
    lower, upper = map(int, slot.difficulty.split("-"))
    # These are deterministic compiler alternatives, not extra model/candidate retries.
    variants = list(range(48))
    random.Random(slot.seed + attempt).shuffle(variants)
    from tau3.synthesis.targeted.native.v03_scenarios import install_generators

    generators = {**GENERATORS, **install_generators()} if slot.v03_labels or slot.family not in GENERATORS else GENERATORS
    for variant in variants:
        if slot.mechanism:
            mechanism = "savings_correction" if variant % 6 == 4 else "referral" if variant % 2 else "card_selection"
            if mechanism != slot.mechanism:
                continue
        for count in range(1, 9):
            b = Builder(slot, attempt, variant)
            try:
                generators[slot.family](b, count, variant)
            except ValueError:
                continue
            if not lower <= len(b.actions.items) <= upper:
                continue
            candidate = b.finish()
            actual = {unpack(a.name, a.arguments)[0] for a in b.actions.items}
            if not set(slot.target_operations) <= actual:
                continue
            if slot.v03_labels:
                held_out = candidate.group_id in v03_holdout_groups(
                    slot.family, slot.branch_id, slot.difficulty, bool(slot.pair_id), slot.mechanism)
            else:
                held_out = int(candidate.group_id[:8], 16) % 5 == 0
            if held_out != (slot.split == "validation"):
                continue
            return candidate
    raise ValueError(f"No supported graph for {slot.family}/{slot.difficulty}/{slot.split}; quota remains unfilled")
