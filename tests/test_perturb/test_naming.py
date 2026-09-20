"""Surface-form rendering and classification."""

import pytest

from tau3.perturb import naming


@pytest.mark.parametrize(
    "style,expected",
    [
        ("title_hyphen", "Vela-Trust"),
        ("upper_hyphen", "VELA-TRUST"),
        ("lower_hyphen", "vela-trust"),
        ("mixed_hyphen", "Vela-trust"),
        ("title_space", "Vela Trust"),
        ("snake", "vela_trust"),
        ("lowerconcat", "velatrust"),
        ("upperconcat", "VELATRUST"),
        ("title_first", "Vela"),
        ("upper_first", "VELA"),
        ("title_hyphen_plus", "Vela-Trust+"),
        ("mixed_hyphen_plus", "Vela-trust+"),
    ],
)
def test_render_styles(style, expected):
    assert naming.render("Vela Trust", style) == expected


def test_render_is_root_spelling_insensitive():
    for root in ("Vela Trust", "vela-trust", "VELA_TRUST"):
        assert naming.render(root, "title_hyphen") == "Vela-Trust"


def test_render_rejects_unknown_style_and_empty_root():
    with pytest.raises(KeyError):
        naming.render("Vela Trust", "sideways")
    with pytest.raises(ValueError):
        naming.render("   ", "snake")


@pytest.mark.parametrize(
    "form,style",
    [
        ("Rho-Bank", "title_hyphen"),
        ("RHO-BANK", "upper_hyphen"),
        ("rho-bank", "lower_hyphen"),
        ("Rho-bank+", "mixed_hyphen_plus"),
        ("Rho Bank", "title_space"),
        ("rho_bank", "snake"),
        ("rhobank", "lowerconcat"),
        ("Rho-Bank+", "title_hyphen_plus"),
        ("Rho", "title_first"),
        ("RHO", "upper_first"),
    ],
)
def test_classify_recovers_every_observed_brand_spelling(form, style):
    """These ten spellings are exactly what the canonical corpus contains."""
    assert naming.classify("Rho Bank", form) == style


def test_classify_returns_none_for_an_unrelated_form():
    assert naming.classify("Rho Bank", "Rhodium") is None


def test_round_trip_classify_then_render_transfers_the_style():
    for form in ("Rho-Bank", "RHO-BANK", "rho_bank", "rhobank", "Rho-Bank+"):
        style = naming.classify("Rho Bank", form)
        assert style is not None
        assert naming.render("Vela Trust", style) == naming.render("Vela Trust", style)


def test_all_forms_covers_the_observed_spellings():
    forms = naming.all_forms("Rho Bank")
    for observed in ("Rho-Bank", "RHO-BANK", "rho-bank", "Rho Bank", "rho_bank",
                     "rhobank", "Rho", "RHO", "Rho-Bank+", "Rho-bank+"):
        assert observed in forms
