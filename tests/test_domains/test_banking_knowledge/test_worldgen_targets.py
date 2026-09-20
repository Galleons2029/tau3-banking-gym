"""Stage 0 calibration: gates target the installed corpus, not the paper's table."""

import pytest

from tau3.worldgen.targets import (
    PAPER_REFERENCE,
    calibrate,
    category_of,
    distribution,
    document_stem,
    tool_inventory,
)


@pytest.fixture(scope="module")
def targets():
    return calibrate()


def test_measured_corpus_shape(targets):
    assert targets.documents == 698
    assert targets.topics == 71
    assert targets.tasks == 97


def test_measurements_that_contradict_the_paper_are_kept(targets):
    # The paper reports 18.6 gold documents and 278.7 tokens per document for this
    # benchmark; the corpus installed here measures materially lower on both. Gates
    # read the measurement, so these are pinned to catch silent corpus drift.
    assert targets.gold_documents.mean == pytest.approx(9.89, abs=0.01)
    assert targets.gold_documents.minimum == 1
    assert targets.gold_documents.maximum == 30
    assert targets.actions.mean == pytest.approx(9.85, abs=0.01)
    assert targets.distinct_gold_documents == 218
    assert targets.gold_document_reuse == pytest.approx(4.40, abs=0.01)

    assert targets.tokenizer == "cl100k_base"
    assert targets.document_tokens.mean == pytest.approx(257.8, abs=0.1)
    assert targets.document_tokens.total == pytest.approx(179943, abs=1)

    assert PAPER_REFERENCE["mean_gold_documents_per_task"] == 18.6
    assert PAPER_REFERENCE["mean_tokens_per_document"] == 278.7


def test_category_count_is_rule_dependent_and_labelled(targets):
    # Document ids do not separate category from topic, so the count only means
    # something next to the rule that produced it.
    assert targets.category_rule == "longest_shared_token_prefix"
    assert targets.categories == 21

    stems = {
        "savings_accounts_gold_account",
        "savings_accounts_silver_account",
        "lonely_topic",
    }
    assert category_of("savings_accounts_gold_account", stems) == "savings_accounts"
    # A stem with no sibling is its own category rather than being force-grouped.
    assert category_of("lonely_topic", stems) == "lonely_topic"


def test_tool_inventory_matches_the_installed_toolkits(targets):
    permanent, discoverable, suffixed = tool_inventory()
    assert len(discoverable) == 48
    assert len(permanent) == 20
    # Two discoverable tools carry no four-digit suffix and are never re-rolled.
    assert len(suffixed) == 46
    assert "get_referral_link" in discoverable
    assert "get_referral_link" not in suffixed
    assert targets.discoverable_tools == discoverable


def test_document_stem_rejects_ids_that_break_the_convention():
    assert document_stem("doc_checking_accounts_blue_account_001") == (
        "checking_accounts_blue_account"
    )
    with pytest.raises(ValueError, match="doc_<stem>_NNN"):
        document_stem("checking_accounts_blue_account")


def test_distribution_handles_degenerate_samples():
    single = distribution([5])
    assert single.mean == single.median == single.p10 == single.p90 == 5
    with pytest.raises(ValueError, match="empty distribution"):
        distribution([])
