"""Independent coverage regressions for the repair8 candidate and review changes."""
from copy import deepcopy

import pytest

from tau3.synthesis.targeted.native.models import NativeSlot
from tau3.synthesis.targeted.native.scenarios import compile_candidate
from tau3.synthesis.targeted.native.validation import (
    check_goals,
    operations,
    validate_static,
)
from tau3.synthesis.targeted.native.workflow import (
    ContentInvalid,
    preserve_original_request,
    public_account_obligations,
)
from tau3.synthesis.validation import replay


def candidate(index, revised=False):
    # Fixed seeds reproduce the affected business conditions without runtime artifacts.
    family, difficulty, seed, branch = {
        10: ("accounts_funds", "5-9", 2875050625, None),
        29: ("replacement_closure", "15-19", 3996315464, "close"),
        238: ("escalation_boundary", "5-9", 369958086, "payment_incident"),
    }[index]
    task = compile_candidate(NativeSlot(
        index=index, split="train", family=family, difficulty=difficulty,
        origin="current", seed=seed, trial_seeds=[1, 2, 3, 4],
        evidence_ids=["finding:0"], v03_labels=["A", "B", "C", "D", "E", "F"],
        runtime_revision="native_card_lifecycle_v2", branch_id=branch,
    ))
    if index == 29 and not revised:
        # Reproduce the previous ambiguous public assignment, retaining reference writes.
        task.facts.pop("repayment_account_nicknames")
        for row in task.task.initial_state.initialization_data.agent_data["accounts"]["data"].values():
            row.pop("nickname", None)
    return task


def test_empty_linked_card_set_is_publicly_provable_but_new_unmodeled_card_is_rejected():
    task = candidate(10)
    original = task.model_dump(mode='json')
    proof = public_account_obligations(task)
    assert proof['zero_linked_cards'] and proof['business_state_unchanged']
    assert len(proof['public_reads']) == len(proof['closed_account_ids']) + 1
    assert task.model_dump(mode='json') == original
    account = proof['closed_account_ids'][0]
    task.task.initial_state.initialization_data.agent_data.setdefault('debit_cards', {'data': {}})['data']['extra_card'] = {
        'card_id': 'extra_card', 'account_id': account, 'user_id': task.facts['identity']['user_id'], 'status': 'ACTIVE'}
    with pytest.raises(ContentInvalid, match='existing linked debit card'):
        public_account_obligations(task)


def test_payoff_aliases_are_returned_publicly_and_wrong_assignment_fails():
    task = candidate(29, revised=True)
    assert validate_static(task)['positive']
    proof = public_account_obligations(task)
    assert proof['payoff_mapping_count'] == 2
    assert all(m['checking_nickname'] in proof['public_reads'][0]['query_reply']
               for m in task.facts['repayment_account_nicknames'])
    broken = deepcopy(task)
    broken.facts['repayment_account_nicknames'][0]['checking_nickname'] = broken.facts['repayment_account_nicknames'][1]['checking_nickname']
    with pytest.raises(ContentInvalid, match='public nickname assignment'):
        public_account_obligations(broken)


def test_unidentified_original_payoff_is_not_admitted_by_new_proof():
    with pytest.raises(ContentInvalid, match='nickname assignment'):
        public_account_obligations(candidate(29))


def test_rejected_paraphrase_retains_original_intent_and_negative_review():
    task = candidate(10)
    review = {'equivalent': False, 'no_answer_added': False, 'explanation': 'Changes target and adds policy answer'}
    text, proof = preserve_original_request(task, 'Invented instruction: close every account and waive fees.', review)
    assert text == task.facts['goal']
    assert proof['rejected_review'] == review and proof['method'] == 'exact_source_identity'
    assert 'waive fees' not in text


def test_handoff_user_completion_does_not_make_wrong_transfer_reason_successful():
    task = candidate(238, revised=True)
    assert validate_static(task)['positive']
    actions = deepcopy(task.task.evaluation_criteria.actions)
    actions[-1].arguments['reason'] = 'customer_frustrated_demands_human'
    env, messages = replay(task.task, actions)
    raw = [m.model_dump(mode='json') for m in messages]
    assert check_goals(task, env, operations(raw), raw)
