"""One-pass, longest-match-first, boundary-aware symbol replacement.

The safety property this module exists to provide: **no replacement can feed
another replacement**.  Every source form is compiled into a single regex
alternation and substituted in one ``re.sub`` pass, so freshly written targets
are never rescanned.  That makes a swap like ``Blue -> Green`` alongside
``Green -> Amber`` well defined instead of order-dependent.

Two more properties fall out of the construction:

* **Longest match first.**  Alternatives are ordered by descending length, so
  ``Rho-Bank+`` wins over ``Rho-Bank``, which wins over bare ``Rho``.
* **Boundary awareness.**  Each alternative carries lookarounds derived from its
  own edge characters, so ``Rho`` does not match inside ``Rhode`` while
  ``Rho-Bank+`` still matches despite ending in punctuation.

Every application returns a :class:`collections.Counter` of per-source hits.
Callers compare those counts against the plan, which is what turns "a rename
was missed" into a hard failure rather than a silently degraded variant.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Characters that make a boundary lookaround meaningful.
_WORD = r"[A-Za-z0-9_]"


class CollisionError(ValueError):
    """A replacement set is ambiguous or would allow a rename to cascade."""


@dataclass(frozen=True)
class Replacement:
    """One surface form and what it becomes."""

    source: str
    target: str
    concept_id: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise CollisionError("Replacement source must not be empty")
        if not self.target:
            raise CollisionError(f"Replacement target must not be empty: {self.source!r}")


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _alternative(source: str) -> str:
    """Escape ``source`` with boundary lookarounds implied by its own edges."""
    prefix = f"(?<!{_WORD})" if _is_word_char(source[0]) else ""
    suffix = f"(?!{_WORD})" if _is_word_char(source[-1]) else ""
    return f"(?:{prefix}{re.escape(source)}{suffix})"


class Rewriter:
    """Apply a validated replacement set to text, JSON and source code."""

    def __init__(
        self,
        replacements: Sequence[Replacement],
        *,
        allow_shared_symbols: bool = False,
    ) -> None:
        """Build a rewriter.

        Args:
            replacements: The validated replacement set.
            allow_shared_symbols: Permit a target that is also another
                replacement's source.  Substitution stays well defined either
                way -- the single pass guarantees that -- but a plan must not
                do it, because the leak scan could then not tell a leftover
                canonical token from an intentional new one.  Only the engine's
                own tests relax this.
        """
        self._validate(replacements, allow_shared_symbols=allow_shared_symbols)
        self._replacements = tuple(replacements)
        self._map = {r.source: r.target for r in replacements}
        self._concepts = {r.source: r.concept_id for r in replacements}
        # Longest first so a prefix form never shadows a longer one; the
        # secondary key keeps the compiled pattern stable across runs.
        ordered = sorted(self._map, key=lambda s: (-len(s), s))
        self._ordered = tuple(ordered)
        self._pattern = (
            re.compile("|".join(_alternative(s) for s in ordered)) if ordered else None
        )

    @staticmethod
    def _validate(
        replacements: Sequence[Replacement], *, allow_shared_symbols: bool
    ) -> None:
        sources: set[str] = set()
        for replacement in replacements:
            if replacement.source in sources:
                raise CollisionError(f"Duplicate source form: {replacement.source!r}")
            if replacement.target == replacement.source:
                raise CollisionError(f"Identity replacement: {replacement.source!r}")
            sources.add(replacement.source)

        targets: dict[str, str] = {}
        for replacement in replacements:
            clash = targets.get(replacement.target)
            if clash is not None:
                raise CollisionError(
                    f"Two sources map to {replacement.target!r}: "
                    f"{clash!r} and {replacement.source!r}"
                )
            targets[replacement.target] = replacement.source

        if allow_shared_symbols:
            return
        for replacement in replacements:
            if replacement.target in sources:
                raise CollisionError(
                    f"Target {replacement.target!r} is also a source form; "
                    "a variant must not reuse canonical symbols"
                )

    # -- application ------------------------------------------------------

    @property
    def sources(self) -> tuple[str, ...]:
        """Source forms, longest first -- the compiled alternation order."""
        return self._ordered

    def concept_of(self, source: str) -> str:
        """Return the concept id that owns ``source``."""
        return self._concepts[source]

    def apply(self, text: str) -> tuple[str, Counter]:
        """Rewrite every source form in ``text``.

        Returns:
            The rewritten text and a Counter of hits keyed by source form.
        """
        hits: Counter = Counter()
        if self._pattern is None or not text:
            return text, hits

        def substitute(match: re.Match) -> str:
            found = match.group(0)
            hits[found] += 1
            return self._map[found]

        return self._pattern.sub(substitute, text), hits

    def count(self, text: str) -> Counter:
        """Count source-form occurrences without rewriting."""
        return self.apply(text)[1]

    def apply_json(self, value: Any, *, keys: "Rewriter | None" = None) -> tuple[Any, Counter]:
        """Rewrite a decoded JSON document.

        String *values* are rewritten by this rewriter.  Dict *keys* are
        rewritten only by ``keys``, so a concept can be renamed in prose
        without disturbing the schema -- and ``rho_bank_subscription``, which
        is genuinely both, is simply present in both rewriters.

        Task files carry nested JSON as opaque strings (``"arguments":
        "{\\"rho_bank_subscription\\": true}"``).  Those are string values, so
        the value rewriter reaches the embedded keys with no special case.
        """
        hits: Counter = Counter()

        def walk(node: Any) -> Any:
            if isinstance(node, str):
                rewritten, found = self.apply(node)
                hits.update(found)
                return rewritten
            if isinstance(node, dict):
                result = {}
                for key, child in node.items():
                    new_key = key
                    if keys is not None and isinstance(key, str):
                        new_key, found = keys.apply(key)
                        hits.update(found)
                    result[new_key] = walk(child)
                return result
            if isinstance(node, list):
                return [walk(item) for item in node]
            return node

        return walk(value), hits

    def apply_json_text(
        self, text: str, *, keys: "Rewriter | None" = None, indent: int = 2
    ) -> tuple[str, Counter]:
        """Rewrite a JSON document given as text, preserving pretty-printing."""
        rewritten, hits = self.apply_json(json.loads(text), keys=keys)
        return json.dumps(rewritten, indent=indent, ensure_ascii=False) + "\n", hits


def build_rewriter(
    pairs: Iterable[tuple[str, str]] | Mapping[str, str],
    *,
    allow_shared_symbols: bool = False,
) -> Rewriter:
    """Convenience constructor from ``(source, target)`` pairs."""
    items = pairs.items() if isinstance(pairs, Mapping) else pairs
    return Rewriter(
        [Replacement(source=s, target=t) for s, t in items],
        allow_shared_symbols=allow_shared_symbols,
    )


class FormCounter:
    """Count source forms with exactly the rewriter's matching semantics.

    The scanner runs before targets are assigned, but its counts become
    assertions the materializer checks against actual replacement counts.  If
    the two used different matching rules those assertions would be noise, so
    both go through the same alternation, ordering and boundary construction.
    """

    def __init__(self, forms: Iterable[str]) -> None:
        ordered = sorted({f for f in forms if f}, key=lambda s: (-len(s), s))
        self._ordered = tuple(ordered)
        self._pattern = (
            re.compile("|".join(_alternative(s) for s in ordered)) if ordered else None
        )

    @property
    def forms(self) -> tuple[str, ...]:
        """Forms, longest first."""
        return self._ordered

    def count(self, text: str) -> Counter:
        """Count non-overlapping occurrences, longest form winning."""
        if self._pattern is None or not text:
            return Counter()
        return Counter(match.group(0) for match in self._pattern.finditer(text))
