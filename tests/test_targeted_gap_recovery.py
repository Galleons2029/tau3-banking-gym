"""Recovery selection must never retry a conclusive rejection."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'gap_recovery', Path(__file__).parents[1] / 'scripts/recover_targeted_gaps.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pending_category_preserves_negative_verdicts():
    assert MODULE.pending_category([{'status': 'PASS'}, {'status': 'INCONCLUSIVE'}])
    assert not MODULE.pending_category([{'status': 'FAIL'}, {'status': 'INCONCLUSIVE'}])
    assert not MODULE.pending_category([{'status': 'PASS'}, {'status': 'PASS'}])
    assert not MODULE.pending_category([])
