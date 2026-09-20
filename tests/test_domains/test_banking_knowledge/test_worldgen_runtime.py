"""Suffix aliasing: what the agent sees, unlocks, calls and leaves in the audit log."""

import pytest

from tau3.domains.banking_knowledge.environment import get_db
from tau3.domains.banking_knowledge.tools import (
    KnowledgeTools,
    KnowledgeUserTools,
    parse_discoverable_tool_docstring,
)
from tau3.worldgen.runtime import (
    alias_toolkits,
    discoverable_tool_names,
    official_tool_names,
    suffix_alias_map,
)

SALT = "test-salt"


@pytest.fixture
def toolkits():
    db = get_db()
    alias_map = suffix_alias_map(discoverable_tool_names(), SALT)
    agent, user = alias_toolkits(KnowledgeTools(db), KnowledgeUserTools(db), alias_map)
    return agent, user, alias_map


def alias_for(alias_map, official):
    return next(a for a, o in alias_map.items() if o == official)


def test_alias_map_is_deterministic_and_disjoint_from_official_names():
    names = discoverable_tool_names()
    first = suffix_alias_map(names, SALT)
    assert first == suffix_alias_map(names, SALT)
    assert first != suffix_alias_map(names, "other-salt")
    # Re-rolled names must not resolve against memorized official ones.
    assert not set(first) & official_tool_names()
    assert len(set(first.values())) == len(first)
    # Only suffixed tools are aliased; the two unsuffixed ones are left alone.
    assert set(first.values()) == {n for n in names if n[-4:].isdigit()}


def test_tools_sharing_a_base_get_distinct_aliases():
    # Keying on the base alone would collapse these three onto one name.
    variants = {
        "activate_debit_card_8291",
        "activate_debit_card_8292",
        "activate_debit_card_8293",
    }
    alias_map = suffix_alias_map(variants, SALT)
    assert len(alias_map) == 3


def test_agent_reaches_tools_only_through_the_alias(toolkits):
    agent, _, alias_map = toolkits
    alias = alias_for(alias_map, "close_bank_account_7392")

    assert agent.has_discoverable_tool(alias)
    assert not agent.has_discoverable_tool("close_bank_account_7392")
    assert "Tool unlocked" in agent.unlock_discoverable_agent_tool(alias)
    assert "Unknown agent tool" in agent.unlock_discoverable_agent_tool(
        "close_bank_account_7392"
    )


def test_unlocked_tool_is_advertised_under_its_alias_with_real_parameters(toolkits):
    agent, _, alias_map = toolkits
    alias = alias_for(alias_map, "close_bank_account_7392")
    info = parse_discoverable_tool_docstring(agent.get_discoverable_tools()[alias])

    assert info["name"] == alias
    # The docstring-derived schema must survive renaming, or the agent is told to
    # call a tool with the wrong arguments.
    assert "account_id" in info["parameters"]
    assert info["description"]


def test_permanent_tools_keep_their_names_and_schemas(toolkits):
    agent, _, _ = toolkits
    official = KnowledgeTools(get_db())
    assert set(agent.get_tools()) == set(official.get_tools())


def test_user_side_discoverable_tools_route_through_the_alias(toolkits):
    agent, user, alias_map = toolkits
    alias = alias_for(alias_map, "deposit_check_3847")

    given = agent.give_discoverable_user_tool(alias, "{}")
    assert f"Tool given to user: {alias}" in given
    assert "Unknown discoverable tool" in agent.give_discoverable_user_tool(
        "deposit_check_3847", "{}"
    )
    assert alias in agent.get_user_discoverable_tools_state()


def test_audit_log_records_the_alias_not_the_official_name(toolkits):
    agent, _, alias_map = toolkits
    alias = alias_for(alias_map, "close_bank_account_7392")
    agent.unlock_discoverable_agent_tool(alias)
    agent.call_discoverable_agent_tool(alias, '{"account_id": "01", "reason": "test"}')

    recorded = {
        row["tool_name"] for row in agent.db.agent_discoverable_tools.data.values()
    }
    assert alias in recorded
    assert "close_bank_account_7392" not in recorded


def test_aliasing_rejects_unknown_tools():
    db = get_db()
    with pytest.raises(ValueError, match="unknown tools"):
        alias_toolkits(
            KnowledgeTools(db), KnowledgeUserTools(db), {"made_up_1234": "made_up_0000"}
        )


def test_empty_alias_map_leaves_toolkits_untouched():
    db = get_db()
    agent, user = alias_toolkits(KnowledgeTools(db), KnowledgeUserTools(db), {})
    assert agent.has_discoverable_tool("close_bank_account_7392")
    assert type(agent) is KnowledgeTools
    assert type(user) is KnowledgeUserTools


def test_tool_output_never_leaks_the_official_name(toolkits):
    agent, _, alias_map = toolkits
    alias = alias_for(alias_map, "get_all_user_accounts_by_user_id_3847")
    agent.unlock_discoverable_agent_tool(alias)
    result = agent.call_discoverable_agent_tool(alias, '{"user_id": "123"}')

    # This tool echoes its own name; without the rewrite the first successful
    # call would hand the agent the official suffix the re-roll withholds.
    assert alias in result
    assert "get_all_user_accounts_by_user_id_3847" not in result


def test_official_names_are_rewritten_across_sibling_tools(toolkits):
    agent, _, alias_map = toolkits
    officials = set(alias_map.values())
    alias = alias_for(alias_map, "get_user_dispute_history_7291")
    agent.unlock_discoverable_agent_tool(alias)
    result = agent.call_discoverable_agent_tool(alias, '{"user_id": "123"}')
    assert not [name for name in officials if name in result]
