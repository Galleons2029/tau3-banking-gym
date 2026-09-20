"""Public evidence and a reviewed, finite set of official banking mechanisms."""

from functools import lru_cache

from tau3.synthesis.catalog import BANK, CARDS

# Rates are percentages; EcoCard points convert at $0.01/point (document 005).
# Business cards are outside these personal-customer objectives.
PERSONAL_CARDS = {
    "Bronze Rewards Card": (640, "0", "1", "1", False),
    "Silver Rewards Card": (680, "0", "1", "4", False),
    "Gold Rewards Card": (720, "0", "2.5", "2.5", True),
    "Platinum Rewards Card": (750, "200", "10", "10", False),
    "Diamond Elite Card": (780, "495", "5", "5", False),
    "EcoCard": (0, "50", "1", "1", False),
    "Crypto-Cash Back": (660, "75", "2", "2", False),
}
REFERRALS = {
    # referrer bonus, invitee bonus, deposit, tenure, invitee minimum/maximum age
    "Light Green Account": (15, 25, 100, 14, 13, 24),
    "Light Blue Account": (30, 20, 500, 30, 18, 200),
    "Blue Account": (35, 30, 500, 30, 18, 200),
    "Green Account (checking)": (20, 30, 500, 30, 18, 200),
    "Green Fee-Free Account": (20, 35, 300, 30, 18, 200),
    "Purple Account": (45, 35, 1000, 45, 18, 200),
    "Dark Green Account": (40, 30, 1000, 45, 18, 200),
    "Evergreen Account": (35, 25, 750, 45, 18, 200),
    "Bluest Account": (75, 50, 2000, 60, 18, 200),
    "Gold Years Account": (50, 75, 1000, 30, 62, 200),
}


@lru_cache(maxsize=1)
def documents():
    """Load only public documents, never official tasks or golden trajectories."""
    from tau3.domains.banking_knowledge.environment import get_knowledge_base

    return get_knowledge_base().documents


def evidence_for(names, extra=()):
    """Bind every hidden business operation to its public discovery source."""
    return list(_evidence_for(tuple(sorted(set(names))), tuple(sorted(set(extra)))))


@lru_cache(maxsize=1024)
def _evidence_for(names, extra):
    """The frozen public corpus permits reusing identical operation/doc lookups."""
    selected = set(extra)
    public = documents()
    for name in names:
        matches = [key for key, doc in public.items() if name in doc.content]
        if not matches and name not in {
            "log_verification", "get_current_time", "get_user_information_by_id",
            "get_credit_card_transactions_by_user", "apply_for_credit_card", "submit_referral",
            "get_credit_card_accounts_by_user", "get_referrals_by_user", "change_user_email",
            "submit_transaction", "request_human_agent_transfer",
        }:
            raise ValueError(f"No public evidence for {name}")
        selected.update(matches)
    if not selected <= public.keys():
        raise ValueError("Missing policy document")
    return tuple(sorted(selected))


def product_evidence():
    """All personal card and checking referral documents participate in review."""
    return [
        key for key, doc in documents().items()
        if (key.startswith("doc_credit_cards_") and any(
            key.startswith("doc_credit_cards_" + slug + "_")
            for slug in ("bronze_rewards_card", "silver_rewards_card", "gold_rewards_card",
                         "platinum_rewards_card", "diamond_elite_card", "ecocard", "crypto-cash_back")
        )) or (key.startswith("doc_checking_accounts_") and "referral" in doc.content.lower())
    ]


BASE_DOCS = [BANK + "001", BANK + "009", BANK + "010", BANK + "042"]
DISPUTE_DOCS = [CARDS + "003", CARDS + "004", "doc_credit_cards_silver_rewards_card_001",
                "doc_credit_cards_silver_rewards_card_002", "doc_credit_cards_bronze_rewards_card_002"]
