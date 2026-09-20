"""Regression coverage for evidence-preserving offline data revision."""

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import revise_targeted_sft as revision  # noqa: E402


def identities():
    return [
        {
            "old": {
                "user_id": "case_customer",
                "name": "Case Customer",
                "email": "case_customer@example.invalid",
            },
            "new": {
                "user_id": "cust_100001",
                "name": "Alice Smith",
                "email": "alice.smith100001@gmail.com",
            },
        }
    ]


def test_identity_mapping_covers_keys_embedded_json_and_has_exact_inverse():
    mapper = revision.IdentityMap(identities())
    original = {
        "case_customer": {
            "name": "Case Customer",
            "email": "case_customer@example.invalid",
            "arguments": '{"user_id":"case_customer","account_id":"acct_001"}',
        }
    }
    result = mapper(original)
    assert result["cust_100001"]["email"] == "alice.smith100001@gmail.com"
    assert "acct_001" in result["cust_100001"]["arguments"]
    assert revision.IdentityMap(identities(), reverse=True)(result) == original
    assert mapper.text("case_customer@example.invalid") == "alice.smith100001@gmail.com"


def test_numeric_normalization_requires_declared_contract_and_preserves_id():
    prop = SimpleNamespace(type="decimal", coerce=lambda value: Decimal(value))
    operation = SimpleNamespace(
        parameters={"amount": prop, "account_id": SimpleNamespace(type="string")}
    )
    old = {"arguments": '{"amount":"30.00","account_id":"001234"}'}
    converted, count = revision.normalize_numbers(old, operation)
    assert count == 1
    assert json.loads(converted["arguments"]) == {
        "amount": 30.0,
        "account_id": "001234",
    }
    assert old["arguments"].startswith('{"amount":"30.00"')
    with pytest.raises(ValueError, match="precision"):
        revision.normalize_numbers(
            {"arguments": '{"amount":"12345678901234567890.01"}'}, operation
        )


def test_lookup_correction_does_not_claim_authentication_or_hide_negative_claims():
    text, edits, risky = revision.clean_text(
        "Thanks, you're verified. I found your profile.", True
    )
    assert "profile has been located" in text and edits == 1 and not risky
    assert revision.clean_text(
        "The policy was verified. Your issue is not resolved.", True
    ) == ("The policy was verified. Your issue is not resolved.", 0, False)
    assert revision.clean_text("I've transferred you to a human.", True)[2]
    assert not revision.clean_text("I haven't transferred you to a human.", True)[2]


def test_rendered_contract_preserves_choices_and_numeric_bounds():
    rendered = revision.render_contract(
        {
            "name": "write_42",
            "description": "Update a case.",
            "parameters": {
                "amount": {
                    "type": "decimal",
                    "minimum": "0",
                    "maximum": "200",
                    "scale": 2,
                },
                "route": {
                    "type": "string",
                    "choices": ["self_service", "awaiting_information"],
                },
            },
        }
    )
    assert "amount: number (required)" in rendered
    assert "minimum: 0; maximum: 200; scale: 2" in rendered
    assert "self_service, awaiting_information" in rendered
    assert "agent_tool_name='write_42'" in rendered


def test_numeric_equivalence_does_not_normalize_identifiers():
    assert revision.numeric_view(
        {"amount": "25.00"}, {"amount"}
    ) == revision.numeric_view({"amount": 25}, {"amount"})
    assert revision.numeric_view(
        {"user_id": "001"}, {"amount"}
    ) != revision.numeric_view({"user_id": "1"}, {"amount"})


def test_focus_keeps_past_masks_and_does_not_include_future_assistant_answer():
    row = {
        "schema_version": 2,
        "metadata": {},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "unlock_discoverable_agent_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"agent_tool_name": {"type": "string"}},
                        "required": ["agent_tool_name"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "Help with this case."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {
                            "name": "unlock_discoverable_agent_tool",
                            "arguments": '{"agent_tool_name":"write_42"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "tool signature"},
            {"role": "assistant", "content": "Future information must not appear."},
        ],
        "loss_mask": [0, 0, 1, 0, 1],
    }
    focus = revision.focus_candidates(row)["discovery"]
    assert len(focus["messages"]) == 4
    assert focus["loss_mask"] == [0, 0, 1, 0]
    assert "Future information" not in json.dumps(focus)
    assert row["loss_mask"] == [0, 0, 1, 0, 1]


def test_general_training_bridge_preserves_selective_loss_and_inner_json():
    from audit_targeted_revision_training import general_training_sample

    public = {
        "tools": [],
        "messages": [
            {"role": "system", "content": "rules", "loss": False},
            {"role": "assistant", "content": "failed attempt", "loss": False},
            {"role": "assistant", "content": "correct recovery", "loss": True},
        ],
    }
    prepared = general_training_sample(public)
    assert prepared["schema_version"] == 2
    assert prepared["loss_mask"] == [0, 0, 1]
    assert all("loss" not in message for message in prepared["messages"])
    assert public["messages"][1]["loss"] is False
    public["messages"][0]["loss"] = True
    with pytest.raises(ValueError, match="Non-assistant"):
        general_training_sample(public)


def test_final_polish_corrects_future_lookup_and_masks_premature_assertion():
    from finalize_targeted_revision import polish

    row = {
        "tools": [],
        "metadata": {},
        "messages": [
            {"role": "system", "content": "Use profile lookup."},
            {"role": "user", "content": "Please help."},
            {
                "role": "assistant",
                "content": "Once you're verified, I will list your cases.",
            },
            {"role": "assistant", "content": "Thanks, you're verified."},
            {"role": "assistant", "content": "Please provide your email address."},
        ],
        "loss_mask": [0, 0, 1, 1, 1],
    }
    revised, audit = polish(row)
    assert (
        revised["messages"][2]["content"]
        == "Once your customer profile is located, I will list your cases."
    )
    assert revised["messages"][3]["content"] == row["messages"][3]["content"]
    assert revised["loss_mask"] == [0, 0, 1, 0, 1]
    assert audit["counts"]["conditional_lookup_edits"] == 1
    assert audit["counts"]["premature_lookup_turns_masked"] == 1


def lookup_row(first_email, second_email=None):
    """Build actual query/result pairs for email-grounding checks."""
    messages = [
        {"role": "system", "content": "Locate the customer."},
        {"role": "user", "content": "My email is " + first_email},
    ]
    masks = [0, 0]
    for index, email in enumerate(
        [first_email] + ([second_email] if second_email else [])
    ):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": str(index),
                            "type": "function",
                            "function": {
                                "name": "find_customer",
                                "arguments": json.dumps({"email": email}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(index), "content": "[]"},
            ]
        )
        masks.extend([1, 0])
    messages.append(
        {"role": "assistant", "content": "Please check your email address."}
    )
    masks.append(1)
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "find_customer",
                    "parameters": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                },
            }
        ],
        "metadata": {},
        "messages": messages,
        "loss_mask": masks,
    }


def test_customer_supplied_typo_is_preserved_but_unsupported_lookup_is_masked():
    from finalize_targeted_revision import polish

    original = lookup_row("typo@example.invalid", "guess@example.invalid")
    revised, audit = polish(original)
    assert revised["loss_mask"] == [0, 0, 1, 0, 0, 0, 1]
    assert revised["messages"][4]["tool_calls"] == original["messages"][4]["tool_calls"]
    assert revised["messages"][1] == original["messages"][1]
    assert audit["counts"]["unsupported_lookup_inputs_masked"] == 1


def test_display_case_and_abbreviated_domain_follow_known_profile(tmp_path):
    from finalize_targeted_revision import polish

    world = tmp_path / "world.json"
    world.write_text(json.dumps({"identity_map": identities()}))
    row = lookup_row("alice.smith100001@gmail.com")
    row["metadata"]["data_revision"] = {"world_view": str(world)}
    row["messages"][3]["content"] = json.dumps(
        [{"name": "Alice Smith", "email": "alice.smith100001@gmail.com"}]
    )
    row["messages"][-1]["content"] = (
        "Verified as Alice Smith (email ending @example.invalid). Name: Case CuStOmEr."
    )
    revised, audit = polish(row)
    assert "Case CuStOmEr" not in revised["messages"][-1]["content"]
    assert "@gmail.com" in revised["messages"][-1]["content"]
    assert "Verified as" not in revised["messages"][-1]["content"]
    assert audit["counts"]["display_name_case_edits"] == 1
    assert audit["counts"]["abbreviated_email_domain_edits"] == 1


def test_preexisting_missing_email_failure_is_preserved_without_repair():
    from finalize_targeted_revision import polish

    row = lookup_row("known@gmail.com")
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{}"
    row["loss_mask"][2] = 0
    row["messages"][3]["content"] = "Error: missing required email"
    revised, _ = polish(row)
    assert revised["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert revised["loss_mask"][2] == 0
