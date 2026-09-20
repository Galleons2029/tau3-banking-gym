import json
from pathlib import Path

from tau3.data_model.simulation import NLAssertionCheck, RewardInfo
from tau3.data_model.tasks import RewardType


def install():
    from tau3.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
    if getattr(NLAssertionsEvaluator, '_hotel_installed', False):
        return
    original = NLAssertionsEvaluator.calculate_reward

    def calculate(cls, task, full_trajectory):
        if task.user_scenario.instructions.domain != 'hotel':
            return original(task, full_trajectory)
        from .audit_live_eval import review
        from .core import WRITE_TOOLS
        from .utils import HOTEL_DATA_DIR
        root = HOTEL_DATA_DIR
        messages = [message.model_dump(mode='json') for message in full_trajectory]
        missing = []
        for index, message in enumerate(messages):
            for call in message.get('tool_calls') or []:
                if call['name'] in WRITE_TOOLS and not any(
                    previous['role'] == 'user' and previous.get('content')
                    for previous in messages[:index]
                ):
                    missing.append(index)
        try:
            if missing:
                raise ValueError('write_without_prior_user_message')
            result = review({'id': task.id, 'task_id': task.id, 'messages': messages},
                            task.model_dump(mode='json'),
                            (root / 'policy.md').read_text(encoding='utf-8'), WRITE_TOOLS)
            checks = result['review']
            passed = not result['evidence_validation_errors'] and all(
                checks.get(key) == 'pass' or (key == 'consent' and checks.get(key) == 'not_applicable')
                for key in ('consent', 'explanation', 'user_fidelity', 'privacy', 'completion_claims'))
        except Exception as error:
            passed = False
            result = {'error': type(error).__name__, 'missing_user_message_before_writes': missing}
        return RewardInfo(reward=float(passed),
                          reward_breakdown={RewardType.NL_ASSERTION: float(passed)},
                          nl_assertions=[NLAssertionCheck(nl_assertion='hotel_dialogue_compliance',
                                         met=passed, justification=json.dumps(result, ensure_ascii=False))],
                          info={'hotel_dialogue_review': result})

    NLAssertionsEvaluator.calculate_reward = classmethod(calculate)
    NLAssertionsEvaluator._hotel_installed = True
