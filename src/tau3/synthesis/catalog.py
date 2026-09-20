"""Compile a small, auditable rule catalog from the installed banking KB."""

import hashlib
import inspect
import re

from tau3.synthesis.models import Evidence, RuleCatalog
from tau3.synthesis.storage import environment_fingerprint

BANK = "doc_bank_accounts_bank_accounts_(general)_"
LOGISTICS = "doc_credit_cards_credit_card_account_logistics_"
CARDS = "doc_credit_cards_credit_cards_(general)_"
PRODUCT_DOCS = {
    "Light Green Account": "doc_checking_accounts_light_green_account_001",
    "Blue Account": "doc_checking_accounts_blue_account_001",
    "Green Account (checking)": "doc_checking_accounts_green_account_(checking)_001",
}


def build_catalog() -> RuleCatalog:
    """Extract supported rules; fail closed when evidence no longer matches."""
    from tau3.domains.banking_knowledge.environment import get_knowledge_base
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    documents = get_knowledge_base().documents
    evidence = {}

    def use(doc_id):
        document = documents[doc_id]
        evidence[doc_id] = Evidence(
            document_id=doc_id,
            sha256=hashlib.sha256(document.content.encode()).hexdigest(),
            quote=document.content,
        )
        return document.content

    def number(doc_id, pattern):
        text = use(doc_id)
        match = re.search(pattern, text, re.I)
        if not match:
            raise ValueError(f"Unsupported or changed rule: {doc_id}: {pattern}")
        return match.group(1).replace(",", "")

    products = {}
    for name, doc_id in PRODUCT_DOCS.items():
        # Three explicitly named alternatives, not a claim of whole-KB optimality.
        text = use(doc_id)
        fee = number(doc_id, r"Monthly maintenance fee\s*[:|]\s*\$([\d,.]+)")
        deposit = number(
            doc_id,
            r"(?:Mobile check deposit daily limit|Daily mobile check deposit limit)\s*[:|]\s*\$([\d,.]+)",
        )
        early = number(doc_id, r"Early direct deposit\s*[:|]\s*(?:Up to )?(\d+)")
        products[name] = {
            "monthly_fee": fee,
            "mobile_deposit_limit": deposit,
            "early_days": int(early),
            "doc_id": doc_id,
        }
        assert "overdraft" in text.lower()

    age_doc = "doc_checking_accounts_light_green_account_002"
    age_match = re.search(r"between (\d+) and (\d+) years old", use(age_doc))
    if not age_match:
        raise ValueError("Missing Light Green age eligibility evidence")
    products["Light Green Account"].update(
        min_age=int(age_match[1]), max_age=int(age_match[2]), age_document=age_doc
    )

    tiers = {}
    for tier in ("Entry", "Mid", "Premium"):
        row = re.search(
            rf"\| {tier}-tier\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)%",
            use(LOGISTICS + "005"),
        )
        if not row:
            raise ValueError(f"Missing CLI tier evidence: {tier}")
        tiers[tier.lower()] = {
            "age": int(row[1]),
            "cooldown": int(row[2]),
            "utilization": int(row[3]),
            "months": int(
                number(LOGISTICS + "006", rf"{tier}-tier cards: Requires (\d+)")
            ),
            "max_increase_percent": int(
                number(
                    LOGISTICS + "006", rf"{tier}-tier cards: Maximum increase of (\d+)%"
                )
            ),
        }
    rewards = {
        "Silver Rewards Card": {
            "default": number(
                "doc_credit_cards_silver_rewards_card_001",
                r"outside top categories earn ([\d.]+)%",
            ),
            "enhanced": number(
                "doc_credit_cards_silver_rewards_card_002",
                r"transactions to earn ([\d.]+)%",
            ),
            "categories": ["Travel", "Software"],
        },
        "Bronze Rewards Card": {
            "default": number(
                "doc_credit_cards_bronze_rewards_card_002", r"You earn ([\d.]+)%"
            ),
            "categories": [],
        },
    }
    tools = [
        "open_bank_account_4821",
        "close_bank_account_7392",
        "transfer_funds_between_bank_accounts_7291",
        "get_all_user_accounts_by_user_id_3847",
        "submit_credit_limit_increase_request_7392",
        "get_credit_limit_increase_history_4829",
        "get_user_dispute_history_7291",
        "get_pending_replacement_orders_5765",
        "get_payment_history_6183",
        "approve_credit_limit_increase_5847",
        "deny_credit_limit_increase_5848",
        "update_transaction_rewards_3847",
        "submit_cash_back_dispute_0589",
    ]
    signatures = {}
    for tool in tools:
        owner = KnowledgeTools if hasattr(KnowledgeTools, tool) else KnowledgeUserTools
        signatures[tool] = str(inspect.signature(getattr(owner, tool)))
        matches = [d.id for d in documents.values() if tool in d.content]
        if not matches:
            raise ValueError(f"Tool lacks KB evidence: {tool}")
        # Keep all mentions: policies often spread prerequisites over several docs.
        for doc_id in matches:
            use(doc_id)
    for doc_id in (
        BANK + "001",
        BANK + "002",
        BANK + "003",
        BANK + "005",
        BANK + "010",
        LOGISTICS + "007",
        CARDS + "003",
        CARDS + "004",
        "doc_business_checking_accounts_navy_blue_001",
    ):
        use(doc_id)
    return RuleCatalog(
        environment_hash=environment_fingerprint(),
        documents=evidence,
        products=products,
        tool_signatures=signatures,
        rules={
            "cli_tiers": tiers,
            "rewards": rewards,
            "savings_min_age": int(number(BANK + "002", r"at least (\d+) days")),
            "business_checking_min_balance": number(
                BANK + "003", r"balance of at least \$([\d,]+)"
            ),
        },
        exclusions=[
            "CLI approvals after submission are quarantined: fixed official tool re-inserts the same request ID, leaves PENDING status while changing the limit, and returns success. CLI generation covers documented denial branches only; approval coverage is unavailable without an official environment fix.",
            "Business savings opening: document 004 says savings but runtime requires business_savings.",
            "CLI cooldown conflicts: pilot contains no earlier requests; current pending request is not an approved request.",
            "Fractional reward rounding and promotional purchases excluded from pilot until independently specified.",
            "Product optimization is explicitly limited to the three customer-selected alternatives.",
        ],
    )
