"""Independent business predicates, protocol obligations and executable negatives."""

import inspect
import json
import re
from collections import Counter
from decimal import Decimal

from tau3.data_model.tasks import Action
from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
from tau3.domains.banking_knowledge.utils import get_now
from tau3.environment.toolkit import DISCOVERABLE_ATTR
from tau3.synthesis.storage import digest
from tau3.synthesis.targeted.native.catalog import PERSONAL_CARDS, REFERRALS, documents
from tau3.synthesis.targeted.planning import operation_kind, unpack
from tau3.synthesis.validation import fresh_environment, hashes, money, replay


def operations(messages):
    """Unwrap actual operations, preserving actor and call position."""
    messages = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages]
    responses = {m.get("id", m.get("tool_call_id")): m for m in messages if m.get("role") == "tool"}
    result = []
    for position, message in enumerate(messages):
        m = message
        for call in m.get("tool_calls") or []:
            name, args = unpack(call["name"], call.get("arguments"))
            response = responses.get(call.get("id"))
            succeeded = response is not None and not response.get("error") and not re.search(
                r"(?im)^\s*(?:error:|failed to)", response.get("content") or "")
            result.append({"name": name, "arguments": args, "actor": m["role"],
                           "wrapper": call["name"], "position": position, "succeeded": bool(succeeded)})
    return result


def reference_operations(candidate):
    """Represent reference obligations independently of serialized argument strings."""
    return [{"name": unpack(a.name, a.arguments)[0], "arguments": unpack(a.name, a.arguments)[1],
             "actor": a.requestor, "wrapper": a.name, "position": i, "succeeded": True}
            for i, a in enumerate(candidate.task.evaluation_criteria.actions)]


def check_goals(candidate, env, calls, messages=None):
    """Check true outcomes; no equality test against a list of gold wrapper strings."""
    db = env.tools.db
    errors = []
    initial = candidate.task.initial_state.initialization_data.agent_data

    def rows(table):
        return getattr(db, table).data

    for goal in candidate.goals:
        kind = goal["kind"]
        good = False
        if kind == "fields":
            row = rows(goal["table"]).get(goal["id"], {})
            good = all(row.get(k) == v for k, v in goal["expected"].items())
        elif kind == "row_match":
            found = [r for r in rows(goal["table"]).values()
                     if all(r.get(k) == v for k, v in goal["selector"].items())]
            good = len(found) == 1 and all(found[0].get(k) == v for k, v in goal["expected"].items())
        elif kind == "cli_decision":
            from tau3.synthesis.targeted.native.business_checks import cli_obligation

            good = cli_obligation(candidate, goal, env, calls)
        elif kind == "required_operation":
            found = [c for c in calls if c.get("succeeded") and c["name"] == goal["name"]
                     and all(c["arguments"].get(k) == v for k, v in goal["arguments"].items())]
            good = len(found) == 1
        elif kind == "operation_sequence":
            selected = [c for c in calls if c.get("succeeded") and c["name"] in goal["names"]]
            good = [c["name"] for c in selected] == goal["names"]
            previous = -1
            for call in selected:
                good &= any(c.get("succeeded") and c["name"] == "request_human_agent_transfer"
                            and c["actor"] == "user" and previous < c["position"] < call["position"] for c in calls)
                previous = call["position"]
        elif kind in {"balance", "closed_account"}:
            row = rows("accounts").get(goal["account_id"], {})
            expected = Decimal(goal.get("expected", "0"))
            good = money(row.get("current_holdings", "-999")) == expected
            if kind == "closed_account":
                good &= row.get("status") == "CLOSED"
        elif kind == "opened_account":
            found = [r for r in rows("accounts").values() if all(r.get(k) == goal[k] for k in ("user_id", "account_type", "account_class"))]
            good = len(found) == 1 and found[0].get("status") == "OPEN"
            if candidate.slot.v03_labels:
                opened = [c for c in calls if c.get("succeeded") and c["actor"] == "assistant"
                          and c["name"] == "open_bank_account_4821"
                          and all(c["arguments"].get(k) == goal[k] for k in ("user_id", "account_type", "account_class"))]
                checked = [c for c in calls if c.get("succeeded") and c["actor"] == "assistant"
                           and c["name"] == "get_all_user_accounts_by_user_id_3847"
                           and c["arguments"].get("user_id") == goal["user_id"]]
                good &= len(opened) == 1 and any(c["position"] < opened[0]["position"] for c in checked)
        elif kind == "verification":
            expected = {**candidate.facts["identity"], "time_verified": get_now().strftime("%Y-%m-%d %H:%M:%S EST")}
            found = [r for r in rows("verification_history").values() if r.get("user_id") == goal["user_id"]]
            good = len(found) == 1 and all(found[0].get(k) == v for k, v in expected.items())
            verification = [c for c in calls if c.get("succeeded") and c["name"] == "log_verification"
                            and c["arguments"].get("user_id") == goal["user_id"]]
            protected = [c for c in calls if c.get("succeeded") and c["actor"] == "assistant"
                         and c["name"] != "log_verification"
                         and (operation_kind(c["name"], c["actor"]) == "write"
                              or c["wrapper"] == "give_discoverable_user_tool")]
            good &= len(verification) == 1
            if good:
                good &= all(c["position"] > verification[0]["position"] for c in protected)
        elif kind == "cashback":
            original = initial["credit_card_transaction_history"]["data"][goal["transaction_id"]]
            row = rows("credit_card_transaction_history")[goal["transaction_id"]]
            # Independently read source transactions and policy rates, not generator answers.
            rate = Decimal("4") if (original["credit_card_type"] == "Silver Rewards Card" and original["category"] in ("Travel", "Software")) else Decimal("1")
            expected = f"{int(money(original['transaction_amount']) * rate)} points"
            disputes = [r for r in rows("cash_back_disputes").values() if r.get("transaction_id") == goal["transaction_id"]]
            good = row["rewards_earned"] == (expected if goal["affected"] and goal["auto"] else original["rewards_earned"])
            good &= len(disputes) == int(goal["affected"])
            if disputes:
                good &= disputes[0].get("status") == ("RESOLVED" if goal["auto"] else "SUBMITTED")
        elif kind == "replacement":
            found = [r for r in rows("debit_card_orders").values() if r.get("account_id") == goal["account_id"]]
            good = len(found) == 1 and all(found[0].get(k) == v for k, v in goal.items() if k not in {"kind", "account_id"})
        elif kind == "replacement_wait":
            # A policy-compliant deferral must not create an order. The final
            # explanation is checked by the independent trajectory reviewer.
            good = not any(r.get("account_id") == goal["account_id"] for r in rows("debit_card_orders").values())
        elif kind == "card_choice":
            scored = []
            for name, (minimum, annual_fee, ordinary, enhanced, subscription) in PERSONAL_CARDS.items():
                if goal["score"] < minimum or (subscription and not goal["subscription"]) or name == "Diamond Elite Card":
                    continue
                gross = Decimal(goal["monthly"]) * 12 * Decimal(enhanced if goal["travel"] else ordinary) / Decimal(100)
                fee = Decimal(annual_fee)
                if name == "Platinum Rewards Card" and goal["monthly"] >= 7500:
                    fee -= Decimal(150)
                scored.append((gross - fee, name))
            scored.sort(reverse=True)
            good = len(scored) == 1 or scored[0][0] > scored[1][0]
            found = [r for r in rows("credit_card_applications").values() if r.get("customer_name") == goal["customer_name"]]
            good &= len(found) == 1 and found[0].get("card_type") == scored[0][1]
            if found:
                good &= found[0].get("annual_income") == goal["income"] and found[0].get("rho_bank_subscription") is goal["subscription"]
        elif kind == "referral":
            eligible = []
            for name, (mine, theirs, deposit, days, low, high) in REFERRALS.items():
                if goal["deposit"] >= deposit and goal["tenure"] >= days and low <= goal["age"] <= high:
                    eligible.append((mine + theirs, name))
            eligible.sort(reverse=True)
            found = [r for r in rows("referrals").values() if r.get("referrer_id") == goal["user_id"]]
            good = bool(eligible) and len(found) == 1 and found[0].get("referred_account_type") == eligible[0][1]
        elif kind == "savings_correction":
            original = initial["accounts"]["data"][goal["account_id"]]
            principal = Decimal(original["interest_principal"])
            # Independent oracle reads source records and all applicable public components.
            rate = Decimal("4.5") if principal >= Decimal(15000) else Decimal("3.0")
            linked = [r for r in initial["accounts"]["data"].values()
                      if r.get("user_id") == goal["user_id"] and r.get("level") == "Blue Account"]
            if linked and original["linked_checking_entire_period"] == "Blue Account":
                rate += Decimal("0.35")
            if original["relationship_bonus_eligible"] and original["relationship_eligible_entire_period"]:
                rate += Decimal("0.025")
            transaction = initial["bank_account_transaction_history"]["data"][goal["interest_transaction"]]
            correction = (principal * rate / 100 - money(transaction["amount"])).quantize(Decimal("0.01"))
            good = money(rows("accounts")[goal["account_id"]]["current_holdings"]) == money(original["current_holdings"]) + correction
            credits = [c for c in calls if c.get("succeeded") and c["name"] == "apply_savings_account_credit_6831" and c["arguments"].get("account_id") == goal["account_id"]]
            reports = [c for c in calls if c.get("succeeded") and c["name"] == "submit_interest_discrepancy_report_7294" and c["arguments"].get("account_id") == goal["account_id"]]
            good &= len(credits) == len(reports) == 1
            if good:
                expected_args = {"account_id": goal["account_id"], "user_id": goal["user_id"],
                    "expected_apy": float(rate), "actual_apy": float(money(transaction["amount"]) * 100 / principal), "amount_difference": float(correction)}
                good &= reports[0]["arguments"] == expected_args and credits[0]["position"] < reports[0]["position"]
        elif kind == "user_execution":
            good = any(c.get("succeeded") and c["name"] == goal["tool"] and c["actor"] == "user" and c["arguments"] == goal["arguments"] for c in calls)
        elif kind == "transfer":
            found = [c for c in calls if c.get("succeeded") and c["name"] == "transfer_to_human_agents" and c["actor"] == "assistant"]
            good = len(found) == 1 and found[0]["arguments"].get("reason") == goal["reason"]
            if good and goal.get("after_operations"):
                good &= all(c["position"] < found[0]["position"] for c in calls if operation_kind(c["name"], c["actor"]) == "write")
        elif kind == "forbidden":
            good = not any(c["name"] in goal["tools"] for c in calls)
        elif kind == "communicate":
            # Static admission proves this fact is retrievable; actual speech is audited.
            good = messages is None or any(
                m.get("role") == "assistant" and goal["contains"].lower() in (m.get("content") or "").lower()
                for m in messages
            )
        if not good:
            errors.append(f"Unsatisfied {kind}: {digest(goal)[:12]}")
    return errors


def protocol_errors(calls):
    """Require real unlock and handoff; never auto-correct the model's invocation."""
    unlocked, granted, errors = set(), set(), []
    for c in calls:
        if not c.get("succeeded"):
            continue
        name, args, wrapper, actor = c["name"], c["arguments"], c["wrapper"], c["actor"]
        if wrapper == "unlock_discoverable_agent_tool":
            unlocked.add(args.get("agent_tool_name"))
        elif wrapper == "give_discoverable_user_tool":
            granted.add(args.get("discoverable_tool_name"))
        elif wrapper == "call_discoverable_agent_tool" and (actor != "assistant" or name not in unlocked):
            errors.append("Agent call without prior unlock")
        elif wrapper == "call_discoverable_user_tool" and (actor != "user" or name not in granted):
            errors.append("User execution without actual grant")
        owner = KnowledgeUserTools if actor == "user" else KnowledgeTools
        method = getattr(owner, name, None)
        if method is not None and getattr(method, DISCOVERABLE_ATTR, False) and wrapper == name:
            errors.append("Hidden operation bypassed discoverable wrapper")
    return errors


def replacement_policy_errors(candidate, calls):
    """Independently enforce public doc 029 even when the backend permits a write."""
    initial = candidate.task.initial_state.initialization_data.agent_data
    accounts = initial.get("accounts", {}).get("data", {})
    cards = initial.get("debit_cards", {}).get("data", {})
    newly_closed = {cards[c["arguments"]["card_id"]]["account_id"]
                    for c in calls if c.get("succeeded") and c["name"] == "close_debit_card_4721"
                    and c["arguments"].get("card_id") in cards}
    return ["Entry-tier replacement ordered before the 48-hour waiting period"
            for c in calls if c.get("succeeded") and c["name"] == "order_debit_card_5739"
            and c["arguments"].get("account_id") in newly_closed
            and accounts.get(c["arguments"]["account_id"], {}).get("level") == "Light Blue Account"]


def validate_static(candidate):
    """Replay positive, omission, wrong-parameter and unauthorized-write cases."""
    if candidate.slot.v03_labels:
        from tau3.synthesis.targeted.native.business_checks import (
            check_initial_consistency,
        )

        check_initial_consistency(candidate)
    task = candidate.task
    lower, upper = map(int, candidate.slot.difficulty.split("-"))
    if not lower <= len(task.evaluation_criteria.actions) <= upper:
        raise ValueError("Compiled action bucket differs from frozen slot")
    public = documents()
    if not task.required_documents or not set(task.required_documents) <= public.keys():
        raise ValueError("Missing public evidence")
    initial = task.initial_state.initialization_data.agent_data
    env0 = fresh_environment(task)
    for table, payload in initial.items():
        if table == "task_config":
            continue
        for key, row in payload["data"].items():
            if row.get("user_id", candidate.facts["identity"]["user_id"]) != candidate.facts["identity"]["user_id"]:
                raise ValueError(f"Foreign owner in {table}/{key}")
            for field, target in {"account_id": "accounts", "credit_card_account_id": "credit_card_accounts", "user_id": "users"}.items():
                if field in row and field != {"accounts": "account_id", "credit_card_accounts": "account_id"}.get(table):
                    if field == "account_id" and table == "credit_card_accounts":
                        continue
                    if row[field] not in getattr(env0.tools.db, target).data:
                        raise ValueError(f"Broken foreign key {table}/{field}")
    for action in task.evaluation_criteria.actions:
        owner = KnowledgeUserTools if action.requestor == "user" else KnowledgeTools
        inspect.signature(getattr(owner, action.name)).bind(None, **action.arguments)
    env, messages = replay(task)
    calls = reference_operations(candidate)
    errors = check_goals(candidate, env, calls) + protocol_errors(calls) + replacement_policy_errors(candidate, calls)
    if errors:
        raise ValueError("Invalid reference: " + "; ".join(errors))
    target = hashes(env)
    actions = task.evaluation_criteria.actions
    negatives = {"empty": []}
    for i, call in enumerate(calls):
        mandated = {name for goal in candidate.goals if goal["kind"] == "operation_sequence" for name in goal["names"]}
        if any(goal["kind"] == "cli_decision" for goal in candidate.goals):
            mandated |= {"get_credit_limit_increase_history_4829", "get_payment_history_6183",
                         "get_user_dispute_history_7291", "get_pending_replacement_orders_5765"}
        if candidate.slot.v03_labels and any(g["kind"] == "opened_account" for g in candidate.goals):
            mandated.add("get_all_user_accounts_by_user_id_3847")
        mandated |= {goal["name"] for goal in candidate.goals if goal["kind"] == "required_operation"}
        if operation_kind(call["name"], call["actor"]) == "write" or call["name"] == "transfer_to_human_agents" or call["name"] in mandated:
            negatives[f"omit_{i}_{call['name']}"] = actions[:i] + actions[i + 1:]
            changed = actions[i].model_copy(deep=True)
            name, args = unpack(changed.name, changed.arguments)
            wrong = dict(args)
            for key in ("amount", "check_amount", "requested_increase_amount", "new_credit_limit",
                        "denial_reason", "disputed_amount", "eligible_for_provisional_credit",
                        "provisional_credit_eligible", "new_rewards_earned", "reason", "card_type", "account_class", "account_type"):
                if key not in wrong:
                    continue
                value = wrong[key]
                wrong[key] = not value if type(value) is bool else value + 1 if isinstance(value, (int, float)) else {
                    "reason": "other", "card_type": "EcoCard", "account_class": "Purple Account",
                    "account_type": "Purple Account", "new_rewards_earned": "0 points",
                    "denial_reason": "other",
                }.get(key, "wrong")
                if wrong[key] == value:
                    wrong[key] = "invalid_alternative"
                if changed.name.startswith("call_discoverable_"):
                    changed.arguments["arguments"] = json.dumps(wrong)
                else:
                    changed.arguments = wrong
                negatives[f"wrong_{i}_{key}"] = actions[:i] + [changed] + actions[i + 1:]
                break
    for index, call in enumerate(calls):
        if call["name"] == "deny_credit_limit_increase_5848":
            cid = call["arguments"]["credit_card_account_id"]
            original = initial["credit_card_accounts"]["data"][cid]
            approval = Action(action_id="wrong_approval", name="call_discoverable_agent_tool", arguments={
                "agent_tool_name": "approve_credit_limit_increase_5847", "arguments": json.dumps({
                    "credit_card_account_id": cid, "user_id": candidate.facts["identity"]["user_id"],
                    "new_credit_limit": int(money(original["credit_limit"])) + 1000})})
            unlock = Action(action_id="wrong_approval_unlock", name="unlock_discoverable_agent_tool",
                            arguments={"agent_tool_name": "approve_credit_limit_increase_5847"})
            negatives["approve_instead_of_deny"] = [*actions[:index], unlock, approval, *actions[index + 1:]]
            break
    extra = Action(action_id="extra_write", name="change_user_email", arguments={
        "user_id": candidate.facts["identity"]["user_id"], "new_email": "unauthorized@example.com"})
    negatives["extra_write"] = [*actions, extra]
    if candidate.slot.v03_labels:
        from tau3.synthesis.targeted.native.protocol import audit_protocol

        for i, action in enumerate(actions):
            if action.name == "unlock_discoverable_agent_tool":
                negatives["missing_unlock"] = actions[:i] + actions[i + 1:]
                break
        for i, action in enumerate(actions):
            if action.name != "call_discoverable_agent_tool":
                continue
            for label, change in (
                ("wrong_suffix", {"agent_tool_name": action.arguments["agent_tool_name"] + "_wrong"}),
                ("nested_object", {"arguments": {"invalid": True}}),
                ("invalid_json", {"arguments": "{"}),
                ("extra_wrapper_parameter", {"unexpected": "value"}),
                ("missing_parameters", {"arguments": "{}"}),
            ):
                if label == "missing_parameters" and not calls[i]["arguments"]:
                    # {} is the valid signature of incident routing tools.
                    continue
                changed = action.model_copy(deep=True)
                changed.arguments.update(change)
                negatives[label] = actions[:i] + [changed] + actions[i + 1:]
            changed = action.model_copy(deep=True)
            changed.arguments["capability"] = changed.arguments.pop("agent_tool_name")
            negatives["wrong_selector"] = actions[:i] + [changed] + actions[i + 1:]
            hidden = action.model_copy(update={"name": calls[i]["name"], "arguments": calls[i]["arguments"]})
            negatives["hidden_bypass"] = actions[:i] + [hidden] + actions[i + 1:]
            break
    for goal in candidate.goals:
        if goal["kind"] == "replacement_wait":
            unlock = Action(action_id="premature_unlock", name="unlock_discoverable_agent_tool", arguments={"agent_tool_name": "order_debit_card_5739"})
            order = Action(action_id="premature_order", name="call_discoverable_agent_tool", arguments={
                "agent_tool_name": "order_debit_card_5739", "arguments": json.dumps({
                    "account_id": goal["account_id"], "user_id": candidate.facts["identity"]["user_id"],
                    "delivery_option": "STANDARD", "delivery_fee": 0, "card_design": "CLASSIC",
                    "design_fee": 0, "shipping_address": candidate.facts["identity"]["address"]})})
            negatives["premature_entry_replacement"] = [*actions, unlock, order]
    # Valid, successful tools can still implement an entirely wrong business path.
    for cid, card in initial.get("debit_cards", {}).get("data", {}).items():
        if card.get("pin_lock_reason") == "security_hold":
            unlock = Action(action_id="wrong_unlock", name="unlock_discoverable_agent_tool", arguments={"agent_tool_name": "reset_debit_card_pin_6284"})
            reset = Action(action_id="wrong_reset", name="call_discoverable_agent_tool", arguments={"agent_tool_name": "reset_debit_card_pin_6284", "arguments": json.dumps({"card_id": cid, "last_4_digits": card["last_4_digits"], "new_pin": "5837"})})
            negatives["wrong_successful_security_workflow"] = [*actions, unlock, reset]
            break
    for index, call in enumerate(calls):
        if call["name"] in {"apply_savings_account_credit_6831", "deposit_check_3847"}:
            negatives["duplicate_credit"] = [*actions[:index + 1], actions[index].model_copy(update={"action_id": "duplicate_credit"}), *actions[index + 1:]]
            break
    if candidate.slot.v03_labels and any(g["kind"] == "opened_account" for g in candidate.goals):
        for index, call in enumerate(calls):
            if call["name"] == "get_all_user_accounts_by_user_id_3847":
                negatives["late_eligibility_check"] = [*actions[:index], *actions[index + 1:], actions[index]]
                changed = actions[index].model_copy(deep=True)
                changed.arguments["arguments"] = json.dumps({"user_id": "another_customer"})
                negatives["wrong_customer_eligibility_check"] = [*actions[:index], changed, *actions[index + 1:]]
                break
    outcomes = {}
    for label, sequence in negatives.items():
        negative, native_messages = replay(task, sequence, strict=False)
        failures = (check_goals(candidate, negative, operations(native_messages))
                    + protocol_errors(operations(native_messages))
                    + replacement_policy_errors(candidate, operations(native_messages)))
        if candidate.slot.v03_labels:
            failures += audit_protocol(native_messages)["errors"]
        rejected = hashes(negative) != target or bool(failures)
        if not rejected:
            raise ValueError(f"Negative not rejected: {label}")
        outcomes[label] = True
    # The same real read may be repeated or moved without inventing a required write.
    extra_read = Action(action_id="extra_query", name="get_user_information_by_id", arguments={"user_id": candidate.facts["identity"]["user_id"]})
    alternative, alternative_messages = replay(task, [extra_read, *actions])
    if hashes(alternative) != target or check_goals(candidate, alternative, operations(alternative_messages)):
        raise ValueError("Legal additional query was rejected")
    state_diff = {}
    if candidate.slot.v03_labels:
        before, after = env0.tools.db.model_dump(mode="json"), env.tools.db.model_dump(mode="json")
        for table, payload in after.items():
            if not isinstance(payload, dict) or "data" not in payload:
                continue
            left, right = before.get(table, {}).get("data", {}), payload["data"]
            changes = {key: {"before": left.get(key), "after": right.get(key)}
                       for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
            if changes:
                state_diff[table] = changes
    return {"positive": True, "counterexamples": outcomes, "alternative_query": True,
            **({"expected_db_diff": state_diff} if candidate.slot.v03_labels else {}),
            "reference_db_hashes": target, "business_operations": sum(operation_kind(c["name"], c["actor"]) == "write" for c in calls),
            "operation_counts": dict(Counter(c["name"] for c in calls))}
