"""Render a concept's surface forms from its root words.

A concept -- a brand, a product, a tool -- appears in the corpus under several
spellings.  The banking brand alone appears as ``Rho-Bank``, ``RHO-BANK``,
``rho_bank``, ``rhobank``, ``Rho Bank``, ``Rho-Bank+`` and bare ``Rho``.  Those
are irregular enough that a generic case-preserving replacer would guess wrong,
so each observed spelling is classified as a *style* instead, and the
replacement is re-derived from the new root through the same style function.

The consequence worth keeping in mind: a spelling nobody classified is a
spelling nobody rewrites.  The leak scan, not this module, is what guarantees
completeness.
"""

import re
from typing import Callable, Optional

#: A "+"-suffixed style is written as ``<base>_plus`` (e.g. ``title_hyphen_plus``).
PLUS = "_plus"


def split_root(root: str) -> tuple[str, ...]:
    """Split a root phrase into words on spaces, hyphens and underscores."""
    return tuple(part for part in re.split(r"[\s_\-]+", root.strip()) if part)


def _cap(word: str) -> str:
    """Title-case a single word, normalising the tail so styles are total."""
    return word[:1].upper() + word[1:].lower()


STYLES: dict[str, Callable[[tuple[str, ...]], str]] = {
    "title_hyphen": lambda w: "-".join(_cap(p) for p in w),
    "upper_hyphen": lambda w: "-".join(p.upper() for p in w),
    "lower_hyphen": lambda w: "-".join(p.lower() for p in w),
    "title_space": lambda w: " ".join(_cap(p) for p in w),
    "upper_space": lambda w: " ".join(p.upper() for p in w),
    "lower_space": lambda w: " ".join(p.lower() for p in w),
    # "Rho-bank": leading word title-cased, the rest lowered.
    "mixed_hyphen": lambda w: "-".join([_cap(w[0]), *(p.lower() for p in w[1:])]),
    "snake": lambda w: "_".join(p.lower() for p in w),
    "upper_snake": lambda w: "_".join(p.upper() for p in w),
    "lowerconcat": lambda w: "".join(p.lower() for p in w),
    "upperconcat": lambda w: "".join(p.upper() for p in w),
    "titleconcat": lambda w: "".join(_cap(p) for p in w),
    "title_first": lambda w: _cap(w[0]),
    "upper_first": lambda w: w[0].upper(),
    "lower_first": lambda w: w[0].lower(),
}

#: Classification order: longer, more specific renderings win ties so that a
#: single-word root does not get reported as ``title_first`` when the corpus
#: really shows the full ``title_space`` form.
_CLASSIFY_ORDER: tuple[str, ...] = (
    "title_hyphen",
    "upper_hyphen",
    "lower_hyphen",
    "mixed_hyphen",
    "title_space",
    "upper_space",
    "lower_space",
    "snake",
    "upper_snake",
    "titleconcat",
    "lowerconcat",
    "upperconcat",
    "title_first",
    "upper_first",
    "lower_first",
)


def render(root: str, style: str) -> str:
    """Render ``root`` in ``style``.

    Args:
        root: The concept's root phrase, e.g. ``"Vela Trust"``.
        style: A key of :data:`STYLES`, optionally suffixed with ``_plus``.

    Returns:
        The rendered surface form.

    Raises:
        KeyError: If ``style`` is not a known style.
        ValueError: If ``root`` has no words.
    """
    base, plus = (style[: -len(PLUS)], True) if style.endswith(PLUS) else (style, False)
    if base not in STYLES:
        raise KeyError(f"Unknown naming style: {style}")
    words = split_root(root)
    if not words:
        raise ValueError(f"Empty root phrase: {root!r}")
    rendered = STYLES[base](words)
    return rendered + "+" if plus else rendered


def classify(root: str, form: str) -> Optional[str]:
    """Return the style rendering ``root`` as ``form``, or None if none does."""
    for style in _CLASSIFY_ORDER:
        for candidate in (style, style + PLUS):
            if render(root, candidate) == form:
                return candidate
    return None


def all_forms(root: str, styles: tuple[str, ...] = _CLASSIFY_ORDER) -> dict[str, str]:
    """Render ``root`` in every style, deduplicated by rendered form.

    Used by the leak scan to enumerate spellings a variant must never contain,
    including ones the canonical corpus happens not to use.
    """
    forms: dict[str, str] = {}
    for style in styles:
        for candidate in (style, style + PLUS):
            forms.setdefault(render(root, candidate), candidate)
    return forms
