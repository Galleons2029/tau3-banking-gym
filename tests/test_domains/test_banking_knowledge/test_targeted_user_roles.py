"""Regression checks for role boundaries and evidence-complete SFT filtering."""

from types import SimpleNamespace

import pytest

from tau3.synthesis.targeted.native.user_roles import (
    role_context,
    role_disposition,
    validate_role_review,
)
from tau3.user.role_guard import (
    CUSTOMER_ROLE_INSTRUCTION,
    SHORT_CUSTOMER_ROLE_INSTRUCTION,
)
from tau3.user.user_simulator import UserSimulator


def test_prompt_guard_preserves_default_and_applies_to_tool_users():
    for tools in (None, []):
        original = UserSimulator(llm="unused", instructions="My name is Morgan", tools=tools)
        guarded = UserSimulator(llm="unused", instructions="My name is Morgan", tools=tools,
                                customer_role_guard=True)
        assert guarded.system_prompt == original.system_prompt + "\n\n" + CUSTOMER_ROLE_INSTRUCTION
        assert len(guarded.get_init_state().system_messages) == 1
        assert CUSTOMER_ROLE_INSTRUCTION not in original.system_prompt
        guarded.customer_role_guard = "customer_role_v2"
        assert guarded.system_prompt == original.system_prompt + "\n\n" + SHORT_CUSTOMER_ROLE_INSTRUCTION


def test_role_context_excludes_bank_private_data_and_retains_customer_tools():
    context = role_context([
        {"role": "system", "content": "bank private policy"},
        {"role": "assistant", "content": "Hello", "raw_data": "private reasoning"},
        {"role": "tool", "requestor": "assistant", "content": "bank internal"},
        {"role": "user", "content": "I deposited my check", "tool_calls": [{"name": "deposit"}]},
        {"role": "tool", "requestor": "user", "content": "success", "name": "deposit"},
    ])
    assert len(context) == 3
    assert context[1]["customer_index"] == 0
    assert context[2]["speaker"] == "CUSTOMER_TOOL"
    assert "private" not in str(context)


@pytest.mark.parametrize("turns", [[], [0, 0], [1], [0, 2]])
def test_review_cannot_skip_or_duplicate_turns(turns):
    context = [{"customer_index": 0, "content": "Hello"}, {"customer_index": 1, "content": "Thanks"}]
    raw = {"turns": [{"customer_index": i, "role": "customer", "quote": "", "reason": "ok"} for i in turns]}
    with pytest.raises(ValueError):
        validate_role_review(raw, context)


def test_review_rejects_invented_evidence():
    raw = {"turns": [{"customer_index": 0, "role": "bank_staff", "quote": "verify identity", "reason": "staff"}]}
    with pytest.raises(ValueError, match="original"):
        validate_role_review(raw, [{"customer_index": 0, "content": "Hello"}])


def test_later_role_recovery_never_erases_early_error():
    reviews = [{"model": model, "review": {"turns": [{"role": "bank_staff"}, {"role": "customer"}]}}
               for model in ("glm", "gemini")]
    assert role_disposition(reviews) == "DISCARD"
    reviews[1]["review"]["turns"][0]["role"] = "customer"
    assert role_disposition(reviews) == "QUARANTINE"
    assert role_disposition(reviews[:1]) == "QUARANTINE"
    reviews[0]["review"]["turns"][0]["role"] = "customer"
    assert role_disposition(reviews) == "KEEP"


def test_role_failure_blocks_export_before_existing_quality_approval(tmp_path, monkeypatch):
    from tau3.synthesis.storage import read_json, write_json
    from tau3.synthesis.targeted.native import user_roles
    from tau3.synthesis.targeted.native.quality import qualify

    write_json(tmp_path / "result.json", {
        "status": "COMPLETE", "environment_success": True,
        "simulation": {"messages": [{"role": "user", "content": "Let me verify your identity."}]},
    })
    write_json(tmp_path / "quality.json", {"review": {"user_compliant": True}})
    def judge(root, config, context, model, *, scope):
        return {"model": model, "review": {"turns": [{"role": "bank_staff"}]}}
    monkeypatch.setattr(user_roles, "review_customer_roles", judge)
    config = SimpleNamespace(user_role_guard="customer_role_v1", teacher_model="glm", reviewer_model="gemini")
    candidate = SimpleNamespace(task=SimpleNamespace(user_scenario=SimpleNamespace(model_dump=lambda **_: {})))
    assert qualify(tmp_path, config, candidate, tmp_path, {}) is None
    assert read_json(tmp_path / "customer-role-review.json")["disposition"] == "DISCARD"
