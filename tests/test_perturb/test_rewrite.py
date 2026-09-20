"""The rewrite engine's safety properties, pinned."""

import json

import pytest

from tau3.perturb.rewrite import CollisionError, Replacement, Rewriter, build_rewriter

BRAND = {
    "Rho-Bank+": "Vela-Trust+",
    "Rho-Bank": "Vela-Trust",
    "RHO-BANK": "VELA-TRUST",
    "rho_bank": "vela_trust",
    "rhobank": "velatrust",
    "Rho": "Vela",
    "RHO": "VELA",
}


def test_longest_match_wins_over_prefix_forms():
    rewriter = build_rewriter(BRAND)
    text = "Rho-Bank+ and Rho-Bank and RHO-BANK and rhobank and Rho alone"
    result, hits = rewriter.apply(text)
    assert result == (
        "Vela-Trust+ and Vela-Trust and VELA-TRUST and velatrust and Vela alone"
    )
    assert hits["Rho-Bank+"] == 1
    assert hits["Rho-Bank"] == 1
    assert hits["Rho"] == 1


def test_word_boundaries_protect_embedded_occurrences():
    rewriter = build_rewriter(BRAND)
    # "Rhode" and "Rhoda" merely start with the brand's short form.
    result, hits = rewriter.apply("Rhode Island, Rhoda, and prho_bank_x")
    assert result == "Rhode Island, Rhoda, and prho_bank_x"
    assert hits == {}


def test_punctuation_edged_form_still_matches_next_to_text():
    rewriter = build_rewriter({"Rho-Bank+": "Vela-Trust+"})
    result, _ = rewriter.apply("The Rho-Bank+ subscription.")
    assert result == "The Vela-Trust+ subscription."


def test_single_pass_prevents_cascade():
    """A swap is the sharpest case: two passes would collapse it, one cannot.

    Plans forbid a target that is also a source (see the test below), but the
    engine must still be order-independent so that a plan-builder bug cannot
    silently corrupt data.
    """
    rewriter = build_rewriter(
        {"Blue Account": "Green Account", "Green Account": "Blue Account"},
        allow_shared_symbols=True,
    )
    result, hits = rewriter.apply("Blue Account and Green Account")
    assert result == "Green Account and Blue Account"
    assert hits == {"Blue Account": 1, "Green Account": 1}


def test_single_pass_chain_does_not_run_forward():
    # Blue -> Green while Green -> Amber: a two-pass implementation would turn
    # every Blue into Amber. One pass must not.
    rewriter = build_rewriter(
        {"Blue Account": "Green Account", "Green Account": "Amber Account"},
        allow_shared_symbols=True,
    )
    result, _ = rewriter.apply("Blue Account and Green Account")
    assert result == "Green Account and Amber Account"


def test_counts_are_reported_per_source_form():
    rewriter = build_rewriter(BRAND)
    _, hits = rewriter.apply("Rho-Bank Rho-Bank rhobank")
    assert hits == {"Rho-Bank": 2, "rhobank": 1}


def test_duplicate_source_rejected():
    with pytest.raises(CollisionError, match="Duplicate source"):
        Rewriter([Replacement("Rho", "Vela"), Replacement("Rho", "Nova")])


def test_two_sources_to_one_target_rejected():
    with pytest.raises(CollisionError, match="map to"):
        Rewriter([Replacement("Rho", "Vela"), Replacement("Sigma", "Vela")])


def test_target_that_is_also_a_source_rejected_by_default():
    # A variant must not reuse a canonical symbol, or the leak scan cannot tell
    # a missed rename from an intentional new name.
    with pytest.raises(CollisionError, match="canonical symbols"):
        Rewriter([Replacement("Blue", "Green"), Replacement("Green", "Amber")])


def test_identity_replacement_rejected():
    with pytest.raises(CollisionError, match="Identity"):
        Rewriter([Replacement("Rho", "Rho")])


def test_empty_source_or_target_rejected():
    with pytest.raises(CollisionError):
        Replacement("", "Vela")
    with pytest.raises(CollisionError):
        Replacement("Rho", "")


def test_json_values_rewritten_but_keys_left_alone_by_default():
    rewriter = build_rewriter({"rho_bank_subscription": "vela_plus_subscription"})
    payload = {"rho_bank_subscription": "rho_bank_subscription is set"}
    result, hits = rewriter.apply_json(payload)
    assert result == {"rho_bank_subscription": "vela_plus_subscription is set"}
    assert hits == {"rho_bank_subscription": 1}


def test_json_keys_rewritten_only_by_the_key_rewriter():
    values = build_rewriter({"rho_bank_subscription": "vela_plus_subscription"})
    keys = build_rewriter({"rho_bank_subscription": "vela_plus_subscription"})
    payload = {"rho_bank_subscription": True, "note": "rho_bank_subscription"}
    result, hits = values.apply_json(payload, keys=keys)
    assert result == {"vela_plus_subscription": True, "note": "vela_plus_subscription"}
    assert hits["rho_bank_subscription"] == 2


def test_embedded_json_carried_as_a_string_is_reached():
    # Task files store discoverable-tool arguments as an opaque JSON string.
    rewriter = build_rewriter(
        {
            "submit_cash_back_dispute_0589": "file_reward_claim_3312",
            "rho_bank_subscription": "vela_plus_subscription",
        }
    )
    payload = {
        "name": "call_discoverable_agent_tool",
        "arguments": json.dumps(
            {"tool_name": "submit_cash_back_dispute_0589", "rho_bank_subscription": True},
            sort_keys=True,
        ),
    }
    result, hits = rewriter.apply_json(payload)
    inner = json.loads(result["arguments"])
    assert inner["tool_name"] == "file_reward_claim_3312"
    assert "vela_plus_subscription" in inner
    assert hits["submit_cash_back_dispute_0589"] == 1


def test_nested_containers_and_non_string_scalars_survive():
    rewriter = build_rewriter({"Rho": "Vela"})
    payload = {"a": [1, None, True, {"b": "Rho"}], "c": 2.5}
    result, _ = rewriter.apply_json(payload)
    assert result == {"a": [1, None, True, {"b": "Vela"}], "c": 2.5}


def test_empty_rewriter_is_a_noop():
    rewriter = Rewriter([])
    assert rewriter.apply("Rho-Bank") == ("Rho-Bank", {})


def test_sources_are_ordered_longest_first_and_deterministic():
    first = build_rewriter(BRAND).sources
    second = build_rewriter(dict(reversed(list(BRAND.items())))).sources
    assert first == second
    assert list(first) == sorted(first, key=lambda s: (-len(s), s))


def test_form_counter_matches_rewriter_semantics():
    """Scan counts become materialize assertions, so they must agree exactly."""
    from tau3.perturb.rewrite import FormCounter

    text = (
        "Rho-Bank+ Rho-Bank RHO-BANK rhobank Rho Rhode "
        "Dark Green Account and Green Account and Light Green Account"
    )
    pairs = {
        **BRAND,
        "Dark Green Account": "Dark Amber Account",
        "Light Green Account": "Light Amber Account",
        "Green Account": "Amber Account",
    }
    rewriter = build_rewriter(pairs)
    counter = FormCounter(pairs)
    assert counter.count(text) == rewriter.count(text)


def test_longer_product_phrase_shields_its_substring():
    """"Dark Green Account" must not be rewritten as "Dark " + Green Account."""
    rewriter = build_rewriter(
        {
            "Dark Green Account": "Dark Amber Account",
            "Light Green Account": "Light Amber Account",
            "Green Account": "Teal Account",
        }
    )
    result, hits = rewriter.apply("Dark Green Account, Light Green Account, Green Account")
    assert result == "Dark Amber Account, Light Amber Account, Teal Account"
    assert hits["Green Account"] == 1
