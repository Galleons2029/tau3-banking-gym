"""Fail-closed replay, independent business checks and scoring counterexamples."""

import inspect
import json
import re
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache

from tau3.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau3.data_model.tasks import Action, Task
from tau3.domains.knowledge_domains import corpus_identity, environment_factory_for
from tau3.runner.build import build_env_kwargs
from tau3.synthesis.models import GenerationRecord, RuleCatalog
from tau3.synthesis.scenarios import (
    cli_decision,
    customer_age,
    reward_points,
    solve_selection,
)


@lru_cache(maxsize=8)
def _replay_template(domain: str, corpus: str):
    """One unmodified environment per corpus.

    Keyed on the corpus as well as the domain because `banking_synth` serves a
    different world per bundle, and because rebuilding the retrieval index for
    every replayed task would dominate validation time.
    """
    del corpus  # part of the cache key only
    return environment_factory_for(domain)(
        retrieval_variant="bm25_grep",
        retrieval_kwargs={"top_k": 10},
        read_log_allowlist=set(),
    )


def replay_environment_constructor(domain: str = "banking_knowledge", **kwargs):
    """Clone an unmodified real environment; never share mutable episode state."""
    if (
        kwargs.get("solo_mode")
        or kwargs.get("retrieval_variant", "bm25_grep") != "bm25_grep"
    ):
        raise ValueError("Replay validator only supports bm25_grep half-duplex")
    env = deepcopy(_replay_template(domain, corpus_identity(domain)))
    env.tools.set_read_log_allowlist(kwargs.get("read_log_allowlist", set()))
    return env


def fresh_environment(task: Task, domain: str = "banking_knowledge"):
    """Construct the same initialized environment used by runner and Gym."""
    kwargs = build_env_kwargs(domain, task, "bm25_grep", {"top_k": 10})
    env = replay_environment_constructor(domain, **kwargs)
    state = task.initial_state
    env.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=state.initialization_actions if state else None,
        message_history=state.message_history or [] if state else [],
    )
    return env


def replay(
    task: Task,
    actions: list[Action] | None = None,
    strict=True,
    domain: str = "banking_knowledge",
):
    """Replay routed tool calls, rejecting textual errors as well as exceptions."""
    env = fresh_environment(task, domain)
    messages = []
    sequence = task.evaluation_criteria.actions if actions is None else actions
    for action in sequence:
        call = ToolCall(
            id=action.action_id,
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        cls = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(cls(role=action.requestor, tool_calls=[call]))
        response = env.get_response(call)
        messages.append(response)
        if strict and (
            response.error
            or re.search(
                r"(?im)^\s*(?:error:|failed to|note:.*already)", response.content or ""
            )
        ):
            raise ValueError(
                f"Reference action failed: {action.action_id}: {response.content}"
            )
    return env, messages


def hashes(env):
    """Both databases participate in benchmark DB scoring."""
    return env.get_db_hash(), env.get_user_db_hash()


def money(value):
    """Parse the domain's dollar-formatted and plain balance representations."""
    return Decimal(str(value).replace("$", "").replace(",", ""))


def validate_references(record: GenerationRecord, catalog: RuleCatalog):
    """Validate table links, source documents, signatures and routing permissions."""
    if (
        record.skeleton.family == "credit_limit"
        and record.skeleton.private.get("decision") == "approved"
    ):
        raise ValueError(
            "Quarantined: official submit/approve request ID collision prevents consistent approval state"
        )
    task = Task.model_validate(record.task.model_dump())
    if not task.evaluation_criteria or task.evaluation_criteria.reward_basis != ["DB"]:
        raise ValueError("Synthesis requires explicit DB scoring")
    if not task.evaluation_criteria.actions:
        raise ValueError("Empty reference actions")
    if not task.required_documents or any(
        d not in catalog.documents for d in task.required_documents
    ):
        raise ValueError("Missing source evidence")
    if (
        record.skeleton.family == "selection"
        and record.skeleton.private["constraints"]["mobile_check_deposit_per_day"]
        <= 500
    ):
        if (
            catalog.products["Light Green Account"]["age_document"]
            not in task.required_documents
        ):
            raise ValueError("Missing age eligibility evidence")
    env = fresh_environment(task)
    patch = task.initial_state.initialization_data.agent_data
    private = record.skeleton.private
    uid = private["user_id"]
    if patch["users"]["data"][uid] != record.skeleton.facts["identity"]:
        raise ValueError("User instructions contradict initial identity")
    if (
        record.skeleton.family == "selection"
        and private["constraints"] != record.skeleton.facts["preferences"]
    ):
        raise ValueError("Selection solver contradicts user preferences")
    if record.skeleton.family == "credit_limit":
        from datetime import datetime

        from tau3.domains.banking_knowledge.utils import get_today

        row = patch["credit_card_accounts"]["data"][private["card_id"]]
        age = (
            get_today()
            - datetime.strptime(row["date_of_account_open"], "%m/%d/%Y").date()
        ).days
        state = private["state"]
        if (
            age != state["age"]
            or money(row["current_balance"]) != money(state["balance"])
            or money(row["credit_limit"]) != money(state["limit"])
        ):
            raise ValueError("CLI solver facts contradict initial database")
        pending_disputes = any(
            r.get("user_id") == uid and r.get("status") == "PENDING"
            for r in env.tools.db.transaction_disputes.data.values()
        )
        pending_replacement = any(
            r.get("credit_card_account_id") == private["card_id"]
            and r.get("status") == "PENDING"
            for r in env.tools.db.credit_card_orders.data.values()
        )
        payments = [
            r
            for r in env.tools.db.payment_history.data.values()
            if r.get("credit_card_account_id") == private["card_id"]
        ]
        on_time = 0
        for payment in sorted(payments, key=lambda r: r["payment_date"], reverse=True):
            if payment["status"] != "ON_TIME":
                break
            on_time += 1
        if (
            pending_disputes != state["pending_disputes"]
            or pending_replacement != state["pending_replacement"]
            or (row["account_status"] == "PAST_DUE") != state["past_due"]
            or on_time != state["on_time_months"]
        ):
            raise ValueError("CLI eligibility facts contradict supporting records")
    if record.skeleton.family == "cashback":
        actual_auto = patch["task_config"]["data"]["dispute_settings"][
            "auto_resolve_disputes"
        ]
        if actual_auto != private["auto_resolve"]:
            raise ValueError("Dispute configuration contradicts solver")
        for txn in private["transactions"]:
            row = patch["credit_card_transaction_history"]["data"][txn["id"]]
            if (
                money(row["transaction_amount"]) != money(txn["amount"])
                or row["category"] != txn["category"]
                or row["rewards_earned"] != f"{txn['actual']} points"
            ):
                raise ValueError("Reward solver facts contradict initial database")
    from tau3.domains.banking_knowledge.environment import get_db

    baseline = get_db()
    foreign_keys = {
        "user_id": "users",
        "credit_card_account_id": "credit_card_accounts",
        "transaction_id": "credit_card_transaction_history",
    }
    for table, payload in patch.items():
        if table == "task_config":
            continue
        for rid, row in payload["data"].items():
            if rid in getattr(baseline, table).data:
                raise ValueError(f"Baseline record collision: {table}/{rid}")
            if "user_id" in row and row["user_id"] != uid:
                raise ValueError(f"Broken reference or ownership: {table}/{rid}")
            for field, target in foreign_keys.items():
                if (
                    field in row
                    and row[field] not in getattr(env.tools.db, target).data
                ):
                    raise ValueError(f"Broken reference: {table}/{rid}/{field}")
    visible_agent_tools = {tool.name for tool in env.get_tools()}
    seen = set()
    for action in task.evaluation_criteria.actions:
        if action.action_id in seen:
            raise ValueError("Duplicate action ID")
        seen.add(action.action_id)
        if action.requestor == "assistant" and action.name not in visible_agent_tools:
            raise ValueError(f"Reference directly invokes hidden tool: {action.name}")
        owner = env.tools if action.requestor == "assistant" else env.user_tools
        if action.requestor == "user" and action.name not in (task.user_tools or []):
            raise ValueError("User tool not enabled")
        inspect.signature(getattr(owner, action.name)).bind(**action.arguments)
        if action.name in (
            "call_discoverable_agent_tool",
            "call_discoverable_user_tool",
        ):
            name = action.arguments.get(
                "agent_tool_name", action.arguments.get("discoverable_tool_name")
            )
            args = json.loads(action.arguments["arguments"])
            inspect.signature(getattr(owner, name)).bind(**args)


def check_business(record: GenerationRecord, env, catalog: RuleCatalog):
    """Check requested outcomes from facts and rules, independently of gold actions."""
    skeleton = record.skeleton
    p = skeleton.private
    db = env.tools.db
    uid = p["user_id"]
    if skeleton.family == "selection":
        chosen = solve_selection(
            catalog.products,
            p["constraints"],
            age=customer_age(skeleton.facts["identity"]["date_of_birth"]),
        )
        rows = [r for r in db.accounts.data.values() if r.get("user_id") == uid]
        if (
            len(rows) != 1
            or rows[0].get("account_class") != chosen
            or rows[0].get("status") != "OPEN"
        ):
            raise ValueError("Wrong selected product or account state")
    elif skeleton.family == "cashback":
        wrong = [
            t
            for t in p["transactions"]
            if t["actual"]
            != reward_points(t["amount"], t["category"], p["card"], catalog)
        ]
        disputes = [
            r for r in db.cash_back_disputes.data.values() if r.get("user_id") == uid
        ]
        if {r["transaction_id"] for r in disputes} != {t["id"] for t in wrong}:
            raise ValueError("Incorrect set of disputed purchases")
        status = "RESOLVED" if p["auto_resolve"] else "SUBMITTED"
        if any(r.get("status") != status for r in disputes):
            raise ValueError("Incorrect dispute status")
        for t in p["transactions"]:
            expected = (
                reward_points(t["amount"], t["category"], p["card"], catalog)
                if p["auto_resolve"]
                else t["actual"]
            )
            if (
                db.credit_card_transaction_history.data[t["id"]]["rewards_earned"]
                != f"{expected} points"
            ):
                raise ValueError("Incorrect rewards")
    elif skeleton.family == "credit_limit":
        decision = cli_decision(p["state"], catalog.rules["cli_tiers"][p["tier"]])
        records = [
            r
            for r in db.credit_limit_increase_requests.data.values()
            if r.get("user_id") == uid
        ]
        approvals = [r for r in records if r.get("status") == "APPROVED"]
        denials = [r for r in records if r.get("status") == "DENIED"]
        expected_limit = Decimal(p["state"]["limit"])
        if decision == "approved":
            expected_limit += Decimal(p["state"]["increase"])
            if len(approvals) != 1 or denials:
                raise ValueError("Missing or inconsistent approval")
        elif (
            len(denials) != 1
            or denials[0].get("denial_reason") != decision
            or approvals
        ):
            raise ValueError("Missing or incorrect denial")
        actual = Decimal(
            db.credit_card_accounts.data[p["card_id"]]["credit_limit"]
            .replace("$", "")
            .replace(",", "")
        )
        if actual != expected_limit:
            raise ValueError("Incorrect approved credit limit")
    else:
        for aid in p["sources"]:
            row = db.accounts.data[aid]
            if row["status"] != "CLOSED" or money(row["current_holdings"]) != 0:
                raise ValueError("Source account not drained and closed")
        destination = db.accounts.data[p["destination"]]
        if money(destination["current_holdings"]) != Decimal(1000) + sum(
            map(Decimal, p["balances"])
        ):
            raise ValueError("Incorrect consolidation balance")
        new = [
            r
            for r in db.accounts.data.values()
            if r.get("user_id") == uid and r.get("account_type") == "business_checking"
        ]
        if len(new) != int(p["open_business"]) or any(
            r.get("account_class") != "Navy Blue" for r in new
        ):
            raise ValueError("Incorrect business account outcome")


def counterexamples(record: GenerationRecord) -> dict[str, list[Action]]:
    """Construct semantic mistakes, including errors accepted by permissive tools."""
    actions = record.task.evaluation_criteria.actions
    cases = {"empty": [], "missing_final_action": actions[:-1]}
    extra = Action(
        action_id="extra_write",
        name="change_user_email",
        arguments={
            "user_id": record.skeleton.private["user_id"],
            "new_email": "unauthorized@example.com",
        },
    )
    cases["extra_write"] = [*actions, extra]
    if record.skeleton.family == "selection":
        changed = actions[-1].model_copy(deep=True)
        arguments = json.loads(changed.arguments["arguments"])
        arguments["account_class"] = next(
            p
            for p in record.skeleton.facts["shortlist"]
            if p != record.skeleton.private["selected_product"]
        )
        changed.arguments["arguments"] = json.dumps(arguments)
        cases["wrong_valid_product"] = [*actions[:-1], changed]
    for i, action in enumerate(actions):
        if action.name != "call_discoverable_agent_tool":
            continue
        inner = json.loads(action.arguments["arguments"])
        for key in (
            "account_class",
            "new_rewards_earned",
            "new_credit_limit",
            "amount",
            "denial_reason",
        ):
            if key not in inner:
                continue
            wrong = dict(inner)
            wrong[key] = {
                "account_class": "Nonexistent Account",
                "new_rewards_earned": "999999 points",
                "denial_reason": "other",
            }.get(key, 1)
            changed = action.model_copy(deep=True)
            changed.arguments["arguments"] = json.dumps(wrong)
            cases[f"wrong_{key}_{i}"] = [*actions[:i], changed, *actions[i + 1 :]]
    return cases


def validate_static(record: GenerationRecord, catalog: RuleCatalog) -> dict:
    """Run every inexpensive gate; raise instead of admitting a faulty task."""
    validate_references(record, catalog)
    env, messages = replay(record.task)
    check_business(record, env, catalog)
    second, _ = replay(record.task)
    target = hashes(env)
    if target != hashes(second):
        raise ValueError("Nondeterministic reference state")
    from tau3.evaluator.evaluator_env import EnvironmentEvaluator

    kwargs = build_env_kwargs(
        "banking_knowledge", record.task, "bm25_grep", {"top_k": 10}
    )
    result = EnvironmentEvaluator.calculate_reward(
        replay_environment_constructor, record.task, messages, env_kwargs=kwargs
    )
    if result.reward != 1:
        raise ValueError("Benchmark evaluator rejects reference trajectory")
    outcomes = {}
    for name, actions in counterexamples(record).items():
        wrong_env, _ = replay(record.task, actions, strict=False)
        if hashes(wrong_env) == target:
            raise ValueError(f"Counterexample receives full DB reward: {name}")
        outcomes[name] = "rejected"
    names = {t.name for t in env.get_tools()}
    if (
        not {"KB_search", "grep"}.issubset(names)
        or {"shell", "KB_search_dense"} & names
    ):
        raise ValueError("Unexpected retrieval tool surface")
    return {
        "db_hashes": target,
        "counterexamples": outcomes,
        "reference_actions": len(messages) // 2,
        "required_documents": len(record.task.required_documents),
    }
