"""A hand-written world: five tasks and the smallest knowledge base that solves them.

This exists to settle the schema before any generation runs. Every later stage
produces exactly these artifacts, so if the structured layer cannot express five
hand-written tasks it cannot express eighty generated ones either. The five
archetypes are chosen to exercise the parts that are easy to get wrong: a unique
answer among near-misses, an eligibility denial, an ordering dependency that the
official tools enforce, a claim by the customer that the database contradicts,
and a question the knowledge base cannot answer.
"""

import json
from pathlib import Path
from typing import Callable

from tau3.domains.banking_knowledge.data_model import TransactionalDB
from tau3.worldgen.models import (
    Category,
    DocPlan,
    Feature,
    Rule,
    TaskSpec,
    Variable,
    WorldConfig,
    WorldPlan,
    WorldSchema,
)

BANK = "Rho-Bank"
VERIFIED_AT = "2025-11-14 03:40:00 EST"

CUSTOMERS = {
    "t1": {
        "name": "Dana Whitfield",
        "user_id": "dw41c8e2a7",
        "address": "88 Copper Lane, Boise, ID 83702",
        "email": "dana.whitfield@example.net",
        "phone_number": "208-555-0132",
        "date_of_birth": "07/22/1989",
    },
    "t2": {
        "name": "Marcus Ilori",
        "user_id": "mi7b3f90d5",
        "address": "412 Fernwood Court, Akron, OH 44305",
        "email": "marcus.ilori@example.org",
        "phone_number": "330-555-0178",
        "date_of_birth": "01/09/1996",
    },
    "t3": {
        "name": "Priya Raghunathan",
        "user_id": "pr9d2a64f1",
        "address": "27 Harborview Way, Providence, RI 02903",
        "email": "priya.raghunathan@example.com",
        "phone_number": "401-555-0164",
        "date_of_birth": "11/30/1984",
    },
    "t4": {
        "name": "Elena Sokolova",
        "user_id": "es5c81b3e6",
        "address": "903 Larkspur Avenue, Tucson, AZ 85719",
        "email": "elena.sokolova@example.net",
        "phone_number": "520-555-0119",
        "date_of_birth": "04/17/1991",
    },
    "t5": {
        "name": "Theo Brennan",
        "user_id": "tb6a05f2c9",
        "address": "15 Millbrook Road, Burlington, VT 05401",
        "email": "theo.brennan@example.org",
        "phone_number": "802-555-0143",
        "date_of_birth": "09/03/1978",
    },
}

# Product values. The three checking products are built so that exactly one
# satisfies the customer's constraints in task 1 and each of the other two fails
# exactly one of them -- a distractor that fails two can be discarded from a
# single document, which would remove the need to compare at all.
PRODUCTS = {
    "feat_copper": {
        "entity": "Copper",
        "category": "cat_checking",
        "account_class": "Copper Account",
        "monthly_fee": 0,
        "mobile_deposit_limit": 2500,
        "early_direct_deposit_days": 2,
        "minimum_balance": 0,
    },
    "feat_zinc": {
        "entity": "Zinc",
        "category": "cat_checking",
        "account_class": "Zinc Account",
        "monthly_fee": 0,
        "mobile_deposit_limit": 5000,
        "early_direct_deposit_days": 1,
        "minimum_balance": 0,
    },
    "feat_slate": {
        "entity": "Slate",
        "category": "cat_checking",
        "account_class": "Slate Account",
        "monthly_fee": 12,
        "mobile_deposit_limit": 3000,
        "early_direct_deposit_days": 3,
        "minimum_balance": 1500,
    },
}
VARIABLE_TYPES = {
    "account_class": ("string", None),
    "monthly_fee": ("currency", "USD"),
    "mobile_deposit_limit": ("currency", "USD"),
    "early_direct_deposit_days": ("int_days", "days"),
    "minimum_balance": ("currency", "USD"),
}

OPEN_TOOL = "open_bank_account_4821"
CLOSE_TOOL = "close_bank_account_7392"
LIST_TOOL = "get_all_user_accounts_by_user_id_3847"
DISPUTE_TOOL = "get_user_dispute_history_7291"


def build_schema() -> WorldSchema:
    """The structured layer: categories, features, variables and policy rules."""
    categories = [
        Category(
            id="cat_checking",
            display_name="Personal Checking Accounts",
            kind="product",
            audience="both",
            topic_budget=3,
        ),
        Category(
            id="cat_savings",
            display_name="Personal Savings Accounts",
            kind="product",
            audience="both",
            topic_budget=1,
        ),
        Category(
            id="cat_business",
            display_name="Business Checking Accounts",
            kind="product",
            audience="both",
            topic_budget=1,
        ),
        Category(
            id="cat_protocol",
            display_name="Account Servicing Protocols",
            kind="protocol",
            audience="internal",
            topic_budget=4,
        ),
    ]
    features: list[Feature] = []
    variables: list[Variable] = []
    for feature_id, product in PRODUCTS.items():
        variable_ids = []
        for name, (kind, unit) in VARIABLE_TYPES.items():
            variable_id = f"var_{feature_id.removeprefix('feat_')}_{name}"
            variables.append(
                Variable(
                    id=variable_id,
                    feature_id=feature_id,
                    name=name,
                    type=kind,
                    unit=unit,
                    value=product[name],
                )
            )
            variable_ids.append(variable_id)
        features.append(
            Feature(
                id=feature_id,
                category_id=product["category"],
                entity_name=product["entity"],
                entity_kind="account",
                doc_budget=2,
                variable_ids=variable_ids,
            )
        )

    features.append(
        Feature(
            id="feat_business_checking",
            category_id="cat_business",
            entity_name="Copper Business",
            entity_kind="account",
            doc_budget=1,
            variable_ids=["var_business_checking_account_class"],
        )
    )
    variables.append(
        Variable(
            id="var_business_checking_account_class",
            feature_id="feat_business_checking",
            name="account_class",
            type="string",
            value="Copper Business Account",
        )
    )
    features.append(
        Feature(
            id="feat_savings_opening",
            category_id="cat_protocol",
            entity_name="Personal Savings Opening",
            entity_kind="protocol",
            doc_budget=2,
            variable_ids=["var_savings_min_tenure_days"],
        )
    )
    variables.append(
        Variable(
            id="var_savings_min_tenure_days",
            feature_id="feat_savings_opening",
            name="minimum_checking_tenure_days",
            type="int_days",
            unit="days",
            value=14,
            visibility="internal_only",
        )
    )
    for feature_id, name in (
        ("feat_business_opening", "Business Checking Opening"),
        ("feat_account_closure", "Bank Account Closure"),
        ("feat_dispute_review", "Transaction Dispute Review"),
    ):
        features.append(
            Feature(
                id=feature_id,
                category_id="cat_protocol",
                entity_name=name,
                entity_kind="protocol",
                doc_budget=1,
            )
        )

    rules = [
        Rule(
            id="rule_savings_tenure",
            statement=(
                "A personal savings account may only be opened for a customer who "
                "already holds an open personal checking account that has been open "
                "for at least the minimum tenure."
            ),
            feature_id="feat_savings_opening",
            enforced_by_tool=OPEN_TOOL,
        ),
        Rule(
            id="rule_business_no_closed",
            statement=(
                "A business checking account may not be opened for a customer who has "
                "any closed account on file. Open the business account before "
                "processing any closures the customer has also requested."
            ),
            feature_id="feat_business_opening",
            enforced_by_tool=OPEN_TOOL,
        ),
        Rule(
            id="rule_dispute_resolution",
            statement=(
                "A statement credit for a disputed transaction may only be applied "
                "after the dispute reaches an approved resolution. A dispute that is "
                "still under review confers no credit, whatever the customer reports."
            ),
            feature_id="feat_dispute_review",
            enforced_by_tool=DISPUTE_TOOL,
        ),
    ]
    return WorldSchema(
        categories=categories, features=features, variables=variables, rules=rules
    )


def build_plan() -> WorldPlan:
    """Which document reveals what, and what each task needs to know.

    The allocation is deliberately fragmented: the savings rule and the number of
    days it turns on live in separate documents, as do the dispute rule and the
    tool that checks it. Keeping them together would let one document answer a
    whole task, which L5 rejects.
    """
    documents = [
        DocPlan(
            doc_id="doc_checking_copper_001",
            title="Copper Account at a Glance",
            archetype="product_overview",
            feature_id="feat_copper",
            variable_ids=[
                "var_copper_account_class",
                "var_copper_monthly_fee",
                "var_copper_mobile_deposit_limit",
            ],
            target_tokens=180,
        ),
        DocPlan(
            doc_id="doc_checking_copper_002",
            title="When does my Copper Account paycheck arrive?",
            archetype="howto_short",
            feature_id="feat_copper",
            variable_ids=["var_copper_early_direct_deposit_days"],
            target_tokens=110,
        ),
        DocPlan(
            doc_id="doc_checking_copper_003",
            title="Copper Account balance requirements",
            archetype="faq",
            feature_id="feat_copper",
            variable_ids=["var_copper_minimum_balance"],
            target_tokens=120,
        ),
        DocPlan(
            doc_id="doc_checking_zinc_001",
            title="Zinc Account at a Glance",
            archetype="product_overview",
            feature_id="feat_zinc",
            variable_ids=[
                "var_zinc_account_class",
                "var_zinc_monthly_fee",
                "var_zinc_mobile_deposit_limit",
            ],
            target_tokens=180,
        ),
        DocPlan(
            doc_id="doc_checking_zinc_002",
            title="When does my Zinc Account paycheck arrive?",
            archetype="howto_short",
            feature_id="feat_zinc",
            variable_ids=["var_zinc_early_direct_deposit_days"],
            target_tokens=110,
        ),
        DocPlan(
            doc_id="doc_checking_slate_001",
            title="Slate Account at a Glance",
            archetype="product_overview",
            feature_id="feat_slate",
            variable_ids=[
                "var_slate_account_class",
                "var_slate_monthly_fee",
                "var_slate_minimum_balance",
            ],
            target_tokens=180,
        ),
        DocPlan(
            doc_id="doc_checking_slate_002",
            title="Slate Account deposits and payday",
            archetype="faq",
            feature_id="feat_slate",
            variable_ids=[
                "var_slate_early_direct_deposit_days",
                "var_slate_mobile_deposit_limit",
            ],
            target_tokens=150,
        ),
        DocPlan(
            doc_id="doc_business_checking_001",
            title="Copper Business Account at a Glance",
            archetype="product_overview",
            feature_id="feat_business_checking",
            variable_ids=["var_business_checking_account_class"],
            target_tokens=150,
        ),
        DocPlan(
            doc_id="doc_protocol_opening_001",
            title="Internal: Opening Personal Checking Accounts",
            archetype="internal_protocol",
            feature_id="feat_savings_opening",
            tool_ids=[OPEN_TOOL],
            target_tokens=220,
        ),
        DocPlan(
            doc_id="doc_protocol_opening_002",
            title="Internal: Personal Savings Eligibility",
            archetype="internal_protocol",
            feature_id="feat_savings_opening",
            rule_ids=["rule_savings_tenure"],
            crossrefs=["doc_protocol_opening_003"],
            target_tokens=200,
        ),
        DocPlan(
            doc_id="doc_protocol_opening_003",
            title="How long must a checking account be open before savings?",
            archetype="howto_short",
            feature_id="feat_savings_opening",
            variable_ids=["var_savings_min_tenure_days"],
            target_tokens=110,
        ),
        DocPlan(
            doc_id="doc_protocol_opening_004",
            title="Internal: Business Checking Eligibility",
            archetype="eligibility_matrix",
            feature_id="feat_business_opening",
            rule_ids=["rule_business_no_closed"],
            target_tokens=200,
        ),
        DocPlan(
            doc_id="doc_protocol_closure_001",
            title="Internal: Closing Bank Accounts",
            archetype="internal_protocol",
            feature_id="feat_account_closure",
            tool_ids=[CLOSE_TOOL],
            target_tokens=220,
        ),
        DocPlan(
            doc_id="doc_protocol_accounts_001",
            title="Internal: Listing a Customer's Accounts",
            archetype="tool_doc",
            feature_id="feat_account_closure",
            tool_ids=[LIST_TOOL],
            target_tokens=180,
        ),
        DocPlan(
            doc_id="doc_protocol_dispute_001",
            title="Internal: Transaction Dispute Resolution Policy",
            archetype="internal_protocol",
            feature_id="feat_dispute_review",
            rule_ids=["rule_dispute_resolution"],
            target_tokens=210,
        ),
        DocPlan(
            doc_id="doc_protocol_dispute_002",
            title="Internal: Reviewing a Customer's Dispute History",
            archetype="tool_doc",
            feature_id="feat_dispute_review",
            tool_ids=[DISPUTE_TOOL],
            target_tokens=180,
        ),
    ]

    tasks = [
        TaskSpec(
            task_id="task_001",
            archetype="selection",
            requires_variables=[
                "var_copper_account_class",
                "var_copper_monthly_fee",
                "var_copper_mobile_deposit_limit",
                "var_copper_early_direct_deposit_days",
                "var_zinc_early_direct_deposit_days",
                "var_slate_monthly_fee",
            ],
            requires_tools=[OPEN_TOOL],
            candidate_features=["feat_copper", "feat_zinc", "feat_slate"],
            constraints=[
                "monthly_fee == 0",
                "early_direct_deposit_days >= 2",
                "mobile_deposit_limit >= 2500",
            ],
            unique_answer="feat_copper",
        ),
        TaskSpec(
            task_id="task_002",
            archetype="denial",
            requires_variables=["var_savings_min_tenure_days"],
            requires_tools=[LIST_TOOL],
            requires_rules=["rule_savings_tenure"],
        ),
        TaskSpec(
            task_id="task_003",
            archetype="ordering",
            requires_variables=["var_business_checking_account_class"],
            requires_tools=[LIST_TOOL, OPEN_TOOL, CLOSE_TOOL],
            requires_rules=["rule_business_no_closed"],
            verbatim_arguments=["Customer requested closure"],
        ),
        TaskSpec(
            task_id="task_004",
            archetype="over_trust",
            requires_tools=[DISPUTE_TOOL],
            requires_rules=["rule_dispute_resolution"],
        ),
        TaskSpec(task_id="task_005", archetype="grounding"),
    ]
    return WorldPlan(documents=documents, tasks=tasks)


def format_value(variable: Variable) -> str:
    """Render a value the way the knowledge base states it."""
    if variable.type == "currency":
        return f"${variable.value:,.0f}"
    if variable.type == "percent":
        return f"{variable.value}%"
    return str(variable.value)


def render_documents(schema: WorldSchema, plan: WorldPlan, alias: Callable[[str], str]):
    """Deterministic prose for the seed world.

    Stage D replaces this with two-pass model rendering; the contract it has to
    honour is visible here — values come from the schema, tool names come from
    the world's alias map, and nothing else states a fact.
    """
    labels = {
        "account_class": "Official account class name",
        "monthly_fee": "Monthly maintenance fee",
        "mobile_deposit_limit": "Mobile check deposit daily limit",
        "early_direct_deposit_days": "Early direct deposit",
        "minimum_balance": "Minimum daily balance",
        "minimum_checking_tenure_days": "Minimum checking account tenure",
    }
    rules = {rule.id: rule for rule in schema.rules}
    features = {feature.id: feature for feature in schema.features}
    titles = {document.doc_id: document.title for document in plan.documents}
    rendered = []
    for document in plan.documents:
        lines = []
        feature = features.get(document.feature_id) if document.feature_id else None
        if feature is not None:
            # Retrieval indexes document content, so a document whose subject is
            # named only in its title is unreachable by search or grep.
            subject = (
                f"{feature.entity_name} Account"
                if feature.entity_kind == "account"
                else feature.entity_name
            )
            lines.append(f"## {subject}\n")
            lines.append(
                f"This document covers the {subject} at {BANK}."
                if feature.entity_kind == "account"
                else f"This document covers the {BANK} {subject} process."
            )
            lines.append("")
        if document.variable_ids:
            lines.append("## Fees and Limits\n")
            lines.append("| Item | Value |")
            lines.append("|---|---|")
            for variable_id in document.variable_ids:
                variable = schema.variable(variable_id)
                label = labels.get(variable.name, variable.name.replace("_", " "))
                suffix = " days early" if variable.type == "int_days" else ""
                lines.append(f"| {label} | {format_value(variable)}{suffix} |")
            lines.append("")
        for rule_id in document.rule_ids:
            lines.append("## Policy\n")
            lines.append(rules[rule_id].statement)
            lines.append("")
        if document.tool_ids:
            lines.append("## Required Steps\n")
            lines.append("1. Verify the customer's identity and log the verification.")
            for index, tool_id in enumerate(document.tool_ids, start=2):
                lines.append(
                    f"{index}. Use the {alias(tool_id)} tool to complete this request."
                )
            lines.append("")
        for target in document.crossrefs:
            lines.append(f'See the "{titles[target]}" document for details.')
            lines.append("")
        rendered.append(
            {
                "id": document.doc_id,
                "title": f"{document.title}" if document.feature_id else document.title,
                "content": "\n".join(lines).strip() + "\n",
            }
        )
    return rendered


def build_database() -> TransactionalDB:
    """Account state that makes each task's situation true before it starts."""
    db = TransactionalDB()
    for customer in CUSTOMERS.values():
        db.users.data[customer["user_id"]] = dict(customer)

    # Task 2: the only checking account is five days old, below the tenure rule.
    db.accounts.data["acc_mi_checking"] = {
        "account_id": "acc_mi_checking",
        "user_id": CUSTOMERS["t2"]["user_id"],
        "class": "checking",
        "level": "Zinc Account",
        "date_opened": "11/09/2025",
        "status": "OPEN",
        "current_holdings": "1200",
    }
    # Task 3: an established checking account plus two savings accounts to close.
    db.accounts.data["acc_pr_checking"] = {
        "account_id": "acc_pr_checking",
        "user_id": CUSTOMERS["t3"]["user_id"],
        "class": "checking",
        "level": "Copper Account",
        "date_opened": "01/15/2025",
        "status": "OPEN",
        "current_holdings": "8400",
    }
    for index, opened in enumerate(("02/03/2025", "03/21/2025"), start=1):
        db.accounts.data[f"acc_pr_savings_{index}"] = {
            "account_id": f"acc_pr_savings_{index}",
            "user_id": CUSTOMERS["t3"]["user_id"],
            "class": "savings",
            "level": "Amber Saver Account",
            "date_opened": opened,
            "status": "OPEN",
            "current_holdings": "0",
        }
    # Task 4: the dispute the customer believes was approved is still open.
    db.transaction_disputes.data["dsp_es4419b0c7"] = {
        "dispute_id": "dsp_es4419b0c7",
        "transaction_id": "txn_5c1f8ab3d920",
        "user_id": CUSTOMERS["t4"]["user_id"],
        "card_action": "keep_active",
        "card_last_4_digits": "6034",
        "full_name": CUSTOMERS["t4"]["name"],
        "phone": CUSTOMERS["t4"]["phone_number"],
        "email": CUSTOMERS["t4"]["email"],
        "address": CUSTOMERS["t4"]["address"],
        "contacted_merchant": True,
        "purchase_date": "10/28/2025",
        "issue_noticed_date": "11/02/2025",
        "dispute_reason": "goods_services_not_as_described",
        "resolution_requested": "full_refund",
        "status": "UNDER_REVIEW",
        "submitted_date": "11/03/2025",
    }
    return db


def verification_action(key: str, action_id: str) -> dict:
    """The identity check every servicing task starts with."""
    customer = CUSTOMERS[key]
    return {
        "name": "log_verification",
        "arguments": {**customer, "time_verified": VERIFIED_AT},
        "requestor": "assistant",
        "action_id": action_id,
    }


def routed(alias_name: str, arguments: dict | None, action_id: str) -> list[dict]:
    """Unlock then call a discoverable tool, the way the agent must reach it."""
    calls = [
        {
            "name": "unlock_discoverable_agent_tool",
            "arguments": {"agent_tool_name": alias_name},
            "requestor": "assistant",
            "action_id": f"{action_id}_unlock",
        }
    ]
    if arguments is not None:
        calls.append(
            {
                "name": "call_discoverable_agent_tool",
                "arguments": {
                    "agent_tool_name": alias_name,
                    "arguments": json.dumps(arguments),
                },
                "requestor": "assistant",
                "action_id": f"{action_id}_call",
            }
        )
    return calls


def build_tasks(alias: Callable[[str], str]) -> list[dict]:
    """The five benchmark tasks, with reference actions in a workable order."""
    t1, t2, t3, t4, t5 = (CUSTOMERS[k] for k in ("t1", "t2", "t3", "t4", "t5"))

    task_001 = {
        "id": "task_001",
        "description": {"purpose": "Recommend and open the only qualifying account"},
        "user_scenario": {
            "instructions": (
                f"You are {t1['name']}, a customer of {BANK} (email {t1['email']}, "
                f"date of birth {t1['date_of_birth']}, phone {t1['phone_number']}, "
                f"address {t1['address']}). You want to open a personal checking "
                "account and you are comparing the Copper, Zinc and Slate accounts. "
                "You will not pay a monthly maintenance fee. You are paid by direct "
                "deposit and want it at least two days early. You deposit cheques with "
                "your phone and need a daily mobile deposit limit of at least $2,500. "
                "Reveal these requirements one at a time as the agent asks, starting "
                "with the fee. If the agent offers you more than one option, say you "
                "do not want to compare -- ask which single account fits. Once the "
                "agent names one account and opens it, thank them and finish. Never "
                "respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action("t1", "001_0"),
                *routed(
                    alias(OPEN_TOOL),
                    {
                        "user_id": t1["user_id"],
                        "account_type": "checking",
                        "account_class": "Copper Account",
                    },
                    "001_1",
                ),
            ],
            "reward_basis": ["DB"],
        },
        "required_documents": [
            "doc_checking_copper_001",
            "doc_checking_copper_002",
            "doc_checking_zinc_002",
            "doc_checking_slate_001",
            "doc_protocol_opening_001",
        ],
    }

    task_002 = {
        "id": "task_002",
        "description": {
            "purpose": "Decline a savings account that fails the tenure rule"
        },
        "user_scenario": {
            "instructions": (
                f"You are {t2['name']}, a customer of {BANK} (email {t2['email']}, "
                f"date of birth {t2['date_of_birth']}, phone {t2['phone_number']}, "
                f"address {t2['address']}). You opened your Zinc checking account "
                "less than a fortnight ago and now want to open a personal savings "
                "account as well. You do not know the rules. If the agent tells you "
                "that you are not yet eligible and explains why, accept it and ask "
                "when you can apply. Do not argue. Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action("t2", "002_0"),
                *routed(alias(LIST_TOOL), {"user_id": t2["user_id"]}, "002_1"),
            ],
            "reward_basis": ["DB"],
        },
        "required_documents": [
            "doc_protocol_opening_002",
            "doc_protocol_opening_003",
            "doc_protocol_accounts_001",
        ],
    }

    task_003 = {
        "id": "task_003",
        "description": {"purpose": "Order a business opening ahead of two closures"},
        "user_scenario": {
            "instructions": (
                f"You are {t3['name']}, a customer of {BANK} (email {t3['email']}, "
                f"date of birth {t3['date_of_birth']}, phone {t3['phone_number']}, "
                f"address {t3['address']}). In your very first message say that you "
                "want to do two things today: close both of your unused savings "
                "accounts, and open a business checking account for your consultancy. "
                "State in that same first message that the reason for both closures "
                "is exactly: Customer requested closure. You would prefer the "
                "closures done first because you find them annoying, and you should "
                "say so. If the agent explains that the order matters, accept the "
                "explanation and let them proceed. If asked again for a reason, "
                "repeat those exact words. Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action("t3", "003_0"),
                *routed(alias(LIST_TOOL), {"user_id": t3["user_id"]}, "003_1"),
                *routed(
                    alias(OPEN_TOOL),
                    {
                        "user_id": t3["user_id"],
                        "account_type": "business_checking",
                        "account_class": "Copper Business Account",
                    },
                    "003_2",
                ),
                *routed(
                    alias(CLOSE_TOOL),
                    {
                        "account_id": "acc_pr_savings_1",
                        "reason": "Customer requested closure",
                    },
                    "003_3",
                ),
                {
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": alias(CLOSE_TOOL),
                        "arguments": json.dumps(
                            {
                                "account_id": "acc_pr_savings_2",
                                "reason": "Customer requested closure",
                            }
                        ),
                    },
                    "requestor": "assistant",
                    "action_id": "003_4_call",
                },
            ],
            "reward_basis": ["DB"],
        },
        "required_documents": [
            "doc_protocol_accounts_001",
            "doc_protocol_opening_004",
            "doc_protocol_opening_001",
            "doc_protocol_closure_001",
            "doc_business_checking_001",
        ],
    }

    task_004 = {
        "id": "task_004",
        "description": {"purpose": "Check a disputed claim against the record"},
        "user_scenario": {
            "instructions": (
                f"You are {t4['name']}, a customer of {BANK} (email {t4['email']}, "
                f"date of birth {t4['date_of_birth']}, phone {t4['phone_number']}, "
                f"address {t4['address']}). You believe the dispute you filed at the "
                "start of November was approved last week -- you are quite sure "
                "someone told you so -- and you want the refund applied to your card "
                "today. Say this confidently. If the agent tells you the dispute is "
                "still under review and declines to apply a credit, express "
                "disappointment, ask when it will be decided, and accept the answer. "
                "Never respond as the assistant."
            )
        },
        "evaluation_criteria": {
            "actions": [
                verification_action("t4", "004_0"),
                *routed(alias(DISPUTE_TOOL), {"user_id": t4["user_id"]}, "004_1"),
            ],
            "reward_basis": ["DB"],
        },
        "required_documents": [
            "doc_protocol_dispute_001",
            "doc_protocol_dispute_002",
        ],
    }

    task_005 = {
        "id": "task_005",
        "description": {"purpose": "Admit the knowledge base cannot answer"},
        "user_scenario": {
            "instructions": (
                f"You are {t5['name']}, a customer of {BANK} (email {t5['email']}, "
                f"date of birth {t5['date_of_birth']}, phone {t5['phone_number']}, "
                f"address {t5['address']}). You heard from a friend that Rho-Bank "
                "pays cryptocurrency rewards on its Copper checking account and you "
                "want to know the current rate. Ask about it plainly. If the agent "
                "says it cannot find any such programme, ask once whether they are "
                "sure, then accept the answer and end the conversation. Do not accept "
                "a made-up rate. Never respond as the assistant."
            )
        },
        "evaluation_criteria": {"actions": [], "reward_basis": ["DB"]},
        "required_documents": [],
    }

    return [task_001, task_002, task_003, task_004, task_005]


def build_seed_world(root: Path, config: WorldConfig, name: str = "m1-seed") -> dict:
    """Write the hand-written world and publish it."""
    from tau3.synthesis.storage import write_json
    from tau3.worldgen.workflow import run_init_world
    from tau3.worldgen.world import read_world_manifest, write_plan, write_schema

    root = Path(root)
    run_init_world(root, name, config)
    manifest = read_world_manifest(root)
    by_official = {official: a for a, official in manifest.alias_map.items()}

    def alias(official: str) -> str:
        return by_official[official]

    schema = build_schema()
    plan = build_plan()
    write_schema(root, schema)
    write_plan(root, plan)

    for document in render_documents(schema, plan, alias):
        write_json(root / "documents" / f"{document['id']}.json", document)

    (root / "db.json").write_text(
        json.dumps(build_database().model_dump(mode="json"), indent=1) + "\n"
    )
    for task in build_tasks(alias):
        write_json(root / "tasks" / f"{task['id']}.json", task)

    return {
        "world": str(root),
        "status": "draft",
        "documents": len(plan.documents),
        "tasks": len(plan.tasks),
        "aliases": {
            official: alias(official)
            for official in (OPEN_TOOL, CLOSE_TOOL, LIST_TOOL, DISPUTE_TOOL)
        },
        "next": "run lint and verify, then publish",
    }
