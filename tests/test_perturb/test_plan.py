"""Plan construction: consistency rules that protect a build."""

import pytest

from tau3.perturb import naming, plan as planner
from tau3.perturb.models import ConceptMap, FormMap
from tau3.perturb.plan import PlanError
from tau3.perturb.rewrite import CollisionError


def concept(concept_id, root, new_root, *, kind="product", base=None, forms=None):
    forms = forms if forms is not None else [
        FormMap(source=root, target=new_root, style="literal", count=1)
    ]
    return ConceptMap(
        concept_id=concept_id, kind=kind, root=root, new_root=new_root,
        base=base, forms=forms,
    )


class _Inventory:
    """Minimal stand-in; _validate only needs the concepts it is handed."""

    concepts = ()


def test_rename_root_applies_the_token_map_consistently():
    token_map = {"Gold": "Zircon", "Silver": "Granite"}
    brand = ("Rho Bank", "Kestrel Union")
    rename = lambda root: planner._rename_root(root, token_map, brand)

    # One token, one replacement, everywhere it appears.
    assert rename("Gold Account") == "Zircon Account"
    assert rename("Gold Plus Account") == "Zircon Plus Account"
    assert rename("Gold Rewards Card") == "Zircon Rewards Card"
    assert rename("BNPL Gold") == "BNPL Zircon"
    # Structural words and generic modifiers are left alone.
    assert rename("Business Silver Rewards Card") == "Business Granite Rewards Card"
    # A product named after the bank follows the brand.
    assert rename("Rho Bank Plus") == "Kestrel Union Plus"
    # Untouched tokens survive.
    assert rename("Scheduled Payments") == "Scheduled Payments"


def test_composed_name_is_derivable_from_its_base():
    """This is the property that stops one product getting two names."""
    token_map = {"Silver": "Granite"}
    brand = ("Rho Bank", "Kestrel Union")
    base = planner._rename_root("Silver Rewards Card", token_map, brand)
    composed = planner._rename_root("Business Silver Rewards Card", token_map, brand)
    assert composed == f"Business {base}"


def test_validate_rejects_an_inconsistent_composed_name():
    concepts = [
        concept("product.silver_rewards_card", "Silver Rewards Card", "Granite Rewards Card"),
        concept(
            "product.business_silver_rewards_card",
            "Business Silver Rewards Card",
            "Business Onyx Rewards Card",  # does not follow its base
            base="product.silver_rewards_card",
        ),
    ]
    with pytest.raises(PlanError, match="Composed name is inconsistent"):
        planner._validate(concepts, _Inventory())


def test_validate_rejects_two_products_sharing_a_new_name():
    concepts = [
        concept("product.a", "Gold Account", "Zircon Account"),
        concept("product.b", "Silver Account", "Zircon Account"),
    ]
    with pytest.raises((PlanError, CollisionError)):
        planner._validate(concepts, _Inventory())


def test_validate_rejects_a_new_name_that_is_a_canonical_name():
    """A variant must never reuse a canonical symbol; the leak scan relies on it."""
    concepts = [
        concept("product.a", "Gold Account", "Silver Account"),
        concept("product.b", "Silver Account", "Zircon Account"),
    ]
    with pytest.raises(CollisionError, match="canonical symbols"):
        planner._validate(concepts, _Inventory())


def test_document_id_rename_follows_the_product_slug():
    slug_map = {"light_green_account": "light_alder_account"}
    renamed = planner._rename_document(
        "doc_checking_accounts_light_green_account_002", slug_map
    )
    assert renamed == "doc_checking_accounts_light_alder_account_002"


def test_document_id_for_a_generic_segment_is_untouched():
    # Category words are ordinary domain vocabulary and are never renamed.
    assert (
        planner._rename_document("doc_bank_accounts_bank_accounts_(general)_001", {})
        == "doc_bank_accounts_bank_accounts_(general)_001"
    )


def test_tool_rename_changes_verb_and_suffix_but_keeps_the_noun():
    import random

    renamed = planner._rename_tool(
        "submit_cash_back_dispute_0589", random.Random(1), set(), set()
    )
    assert renamed != "submit_cash_back_dispute_0589"
    assert "cash_back_dispute" in renamed, "the docstring still describes this tool"
    assert renamed.split("_")[-1].isdigit()


def test_tool_rename_keeps_near_identical_tools_distinct():
    import random

    rng = random.Random(7)
    used_names: set[str] = set()
    used_suffixes: set[str] = set()
    renamed = [
        planner._rename_tool(name, rng, used_names, used_suffixes)
        for name in ("activate_debit_card_8291", "activate_debit_card_8292",
                     "activate_debit_card_8293")
    ]
    assert len(set(renamed)) == 3


def test_draw_is_deterministic_and_respects_exclusions():
    import random

    pool = ("Quartz", "Onyx", "Slate", "Basalt")
    first = planner._draw(random.Random(3), pool, 2, set())
    second = planner._draw(random.Random(3), pool, 2, set())
    assert first == second

    excluded = set(first)
    third = planner._draw(random.Random(3), pool, 2, excluded)
    assert not set(third) & set(first), "a sibling variant must not reuse names"


def test_draw_fails_loudly_when_the_pool_runs_out():
    import random

    with pytest.raises(PlanError, match="Pool exhausted"):
        planner._draw(random.Random(0), ("Quartz",), 2, set())


def test_target_form_renders_each_style_from_the_new_root():
    assert planner._target_form("Kestrel Union", "Rho-Bank", "title_hyphen") == "Kestrel-Union"
    assert planner._target_form("Kestrel Union", "RHOBANK", "upperconcat") == "KESTRELUNION"
    assert planner._target_form("Alder Account", "green_account", "slug") == "alder_account"
    assert planner._target_form("EcoCard X", "EcoCard", "literal") == "EcoCard X"
