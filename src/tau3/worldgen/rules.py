"""Policy rules taken from the environment, not invented.

The synthesized world reuses the official tools, so its policy layer is not free
to say whatever reads well: a rule the tools do not enforce produces documents
that contradict the environment and tasks that cannot be solved. Each rule here
names the tool that enforces it and is admitted only after the environment has
been observed refusing the violating case and accepting the satisfying one.

The wording belongs to this world; the condition belongs to the environment.
"""

from dataclasses import dataclass, field
from typing import Any

from tau3.domains.banking_knowledge.data_model import TransactionalDB

OPEN_TOOL = "open_bank_account_4821"
CUSTOMER = "probe_user_0001"
ELIGIBILITY_ERROR = "Error: Account eligibility requirements not met."


@dataclass
class RuleSpec:
    """A policy rule and the environment observation that proves it."""

    id: str
    statement: str
    tool: str
    # Account rows that violate the rule, and rows that satisfy it.
    violating_accounts: list[dict[str, Any]]
    satisfying_accounts: list[dict[str, Any]]
    call: dict[str, Any] = field(default_factory=dict)


def _account(account_id: str, kind: str, opened: str, status="OPEN", holdings="1000"):
    return {
        "account_id": account_id,
        "user_id": CUSTOMER,
        "class": kind,
        "account_type": kind,
        "level": "Probe Account",
        "date_opened": opened,
        "status": status,
        "current_holdings": holdings,
    }


# Dates are relative to the domain's fixed today, 11/14/2025.
RECENT = "11/09/2025"  # five days old
ESTABLISHED = "01/15/2025"
OLD_BUSINESS = "02/01/2025"  # over thirty days old

RULES = [
    RuleSpec(
        id="rule_savings_tenure",
        statement=(
            "A personal savings account may only be opened for a customer who "
            "already holds an open personal checking account, and that checking "
            "account must have been open for at least the minimum tenure."
        ),
        tool=OPEN_TOOL,
        violating_accounts=[_account("probe_chk", "checking", RECENT)],
        satisfying_accounts=[_account("probe_chk", "checking", ESTABLISHED)],
        call={"account_type": "savings", "account_class": "Probe Savings Account"},
    ),
    RuleSpec(
        id="rule_business_no_closed",
        statement=(
            "A business checking account may not be opened for a customer who has "
            "any closed account on file. Where a customer asks for both a business "
            "opening and a closure, the opening is processed first."
        ),
        tool=OPEN_TOOL,
        violating_accounts=[
            _account("probe_chk", "checking", ESTABLISHED),
            _account("probe_old", "savings", ESTABLISHED, status="CLOSED"),
        ],
        satisfying_accounts=[_account("probe_chk", "checking", ESTABLISHED)],
        call={
            "account_type": "business_checking",
            "account_class": "Probe Business Account",
        },
    ),
    RuleSpec(
        id="rule_business_savings_tenure",
        statement=(
            "A business savings account requires an open business checking account "
            "of at least the minimum business tenure, and no account may be "
            "overdrawn at the time of the request."
        ),
        tool=OPEN_TOOL,
        violating_accounts=[
            _account("probe_bchk", "business_checking", RECENT),
        ],
        satisfying_accounts=[
            _account("probe_bchk", "business_checking", OLD_BUSINESS),
        ],
        call={
            "account_type": "business_savings",
            "account_class": "Probe Business Savings Account",
        },
    ),
]


def _probe(accounts: list[dict], call: dict, tool: str) -> str:
    """Run one tool call against a database built for this observation."""
    from tau3.domains.banking_knowledge.tools import KnowledgeTools

    db = TransactionalDB()
    db.users.data[CUSTOMER] = {
        "name": "Probe Customer",
        "user_id": CUSTOMER,
        "address": "1 Probe Street",
        "email": "probe@example.invalid",
        "phone_number": "000-555-0000",
        "date_of_birth": "01/01/1990",
    }
    for account in accounts:
        db.accounts.data[account["account_id"]] = dict(account)
    tools = KnowledgeTools(db)
    return getattr(tools, tool)(user_id=CUSTOMER, **call)


def verify_rules(specs: list[RuleSpec] | None = None) -> list[str]:
    """Return the ids of rules the environment does not actually enforce.

    Fails closed in the caller: a world whose policy layer has drifted from the
    tools would document rules that nothing applies, and every task resting on
    one of them would be unsolvable for reasons no reviewer could see.
    """
    broken = []
    for spec in specs or RULES:
        refused = _probe(spec.violating_accounts, spec.call, spec.tool)
        allowed = _probe(spec.satisfying_accounts, spec.call, spec.tool)
        if not refused.startswith("Error:"):
            broken.append(f"{spec.id}: violating case was not refused ({refused[:60]})")
        elif allowed.startswith("Error:"):
            broken.append(f"{spec.id}: satisfying case was refused ({allowed[:60]})")
    return broken


def environment_rules() -> list:
    """The verified rules, as schema objects."""
    from tau3.worldgen.models import Rule

    broken = verify_rules()
    if broken:
        raise ValueError(f"Environment no longer enforces these rules: {broken}")
    return [
        Rule(id=spec.id, statement=spec.statement, enforced_by_tool=spec.tool)
        for spec in RULES
    ]
