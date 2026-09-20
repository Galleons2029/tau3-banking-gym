"""Suffix aliasing for synthesized worlds.

Official tool implementations are reused verbatim; only the four-digit suffix in
their names is re-rolled per world. The suffix exists to cut parametric memory —
an agent cannot guess `8293`, it has to retrieve the document that names it — so
a synthesized world that kept the official suffixes would inherit the official
world's memorization. Aliasing happens at instance construction: no official
tool code is edited, and the alias is what the agent sees, unlocks, calls and
what lands in the `agent_discoverable_tools` audit records.
"""

import functools
import hashlib
import re
from typing import Any, Callable, Iterable

from tau3.environment.toolkit import DISCOVERABLE_ATTR, TOOL_ATTR, ToolKitType

SUFFIXED = re.compile(r"^(?P<base>.+)_(?P<suffix>\d{4})$")


def suffix_alias_map(names: Iterable[str], salt: str) -> dict[str, str]:
    """Map re-rolled tool names to the official names they dispatch to.

    Keyed on the full official name, so tools sharing a base (three
    `activate_debit_card_*` variants) still get distinct aliases. Suffixes may
    collide across different bases — the official corpus does that too, and it is
    what stops a suffix from identifying a tool family.
    """
    aliases: dict[str, str] = {}
    for name in sorted(names):
        match = SUFFIXED.match(name)
        if not match:
            continue
        base = match.group("base")
        for attempt in range(1000):
            seed = f"{salt}|{name}|{attempt}".encode()
            suffix = 1000 + int(hashlib.sha256(seed).hexdigest(), 16) % 9000
            alias = f"{base}_{suffix}"
            if alias not in aliases:
                aliases[alias] = name
                break
        else:  # pragma: no cover - 9000 slots, 46 tools
            raise RuntimeError(f"Could not allocate a unique alias for {name}")
    return aliases


def _rewriter(aliases: dict[str, str]) -> Callable[[str], str] | None:
    """Rewrite official tool names in tool output into this world's names.

    Twenty-three tools echo their own name back to the agent ("Executed:
    get_all_user_accounts_by_user_id_3847"). Left alone, the first successful
    call would hand the agent the official suffix that the re-roll exists to
    withhold, and synthesized transcripts would name tools the world does not
    have. Longest names are matched first so one tool name cannot match inside
    another's.
    """
    by_official = {official: alias for alias, official in aliases.items()}
    if not by_official:
        return None
    pattern = re.compile(
        "|".join(sorted((re.escape(n) for n in by_official), key=len, reverse=True))
    )
    return lambda text: pattern.sub(lambda match: by_official[match.group(0)], text)


def _renamed(
    function: Callable, alias: str, rewrite: Callable[[str], str] | None = None
) -> Callable:
    """Copy a tool method under a new public name.

    `functools.wraps` carries the `@is_tool` / `@is_discoverable_tool` markers
    (they live in the function's `__dict__`) and the docstring that
    `parse_discoverable_tool_docstring` reads. It also sets `__wrapped__`, so
    `inspect.signature` still reports the real parameters — which
    `give_discoverable_user_tool` relies on to validate arguments.
    """

    @functools.wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any):
        result = function(self, *args, **kwargs)
        if rewrite is not None and isinstance(result, str):
            return rewrite(result)
        return result

    wrapper.__name__ = alias
    wrapper.__qualname__ = alias
    return wrapper


class _Superseded:
    """Stands in for an official tool name this world renamed.

    Subclassing cannot remove an inherited attribute, and
    `give_discoverable_user_tool` resolves tools with `hasattr`/`getattr` on the
    class rather than through the filtered `tools` mapping. Shadowing the old
    name with a value that is neither callable nor marked as a tool makes the
    official lookup reject it, with a repr that says why.
    """

    __slots__ = ("alias",)

    def __init__(self, alias: str) -> None:
        self.alias = alias

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<tool renamed to {self.alias!r} in this world>"


def _aliased_class(base: type, aliases: dict[str, str], **namespace: Any) -> type:
    """Build a subclass exposing `aliases` and hiding the names they replace."""
    superseded = set(aliases.values())
    rewrite = _rewriter(aliases)
    attributes: dict[str, Any] = dict(namespace)
    for alias, original in aliases.items():
        attributes[alias] = _renamed(getattr(base, original), alias, rewrite)
        attributes[original] = _Superseded(alias)

    def tools(self) -> dict[str, Callable]:
        # The metaclass merges this class's tools with every parent's, so the
        # official names would still resolve unless they are dropped here.
        return {
            name: getattr(self, name)
            for name in self._func_tools
            if name not in superseded
        }

    attributes["tools"] = property(tools)
    attributes["__doc__"] = f"{base.__name__} with world-specific tool names."
    return ToolKitType(f"Aliased{base.__name__}", (base,), attributes)


def _tool_names(toolkit: type) -> set[str]:
    """Names this toolkit exposes as tools, discoverable or not."""
    names = set()
    for name in dir(toolkit):
        member = getattr(toolkit, name, None)
        if callable(member) and getattr(member, TOOL_ATTR, False):
            names.add(name)
    return names


def official_tool_names() -> set[str]:
    """Every tool name across both official toolkits."""
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    return _tool_names(KnowledgeTools) | _tool_names(KnowledgeUserTools)


def discoverable_tool_names() -> set[str]:
    """Discoverable tool names across both official toolkits."""
    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    names = set()
    for toolkit in (KnowledgeTools, KnowledgeUserTools):
        for name in dir(toolkit):
            member = getattr(toolkit, name, None)
            if callable(member) and getattr(member, DISCOVERABLE_ATTR, False):
                names.add(name)
    return names


def alias_toolkits(agent_tools, user_tools, alias_map: dict[str, str]):
    """Rebind both toolkits onto alias-named subclasses, in place.

    Reassigning `__class__` avoids reconstructing the agent toolkit, which owns
    already-built retrieval pipelines. The agent class is given the aliased user
    class so `give_discoverable_user_tool` resolves the names the synthesized
    knowledge base documents.
    """
    if not alias_map:
        return agent_tools, user_tools

    user_available = _tool_names(type(user_tools))
    agent_available = _tool_names(type(agent_tools))
    unknown = set(alias_map.values()) - user_available - agent_available
    if unknown:
        raise ValueError(f"Alias map references unknown tools: {sorted(unknown)}")

    user_aliases = {a: o for a, o in alias_map.items() if o in user_available}
    agent_aliases = {a: o for a, o in alias_map.items() if o in agent_available}

    user_tools.__class__ = _aliased_class(type(user_tools), user_aliases)
    agent_tools.__class__ = _aliased_class(
        type(agent_tools), agent_aliases, user_tool_class=type(user_tools)
    )
    return agent_tools, user_tools
