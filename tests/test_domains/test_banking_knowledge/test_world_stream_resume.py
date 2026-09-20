"""A CLI addition must not grant a blanket exception to producer version pins."""

import runpy
import shutil
from pathlib import Path

import pytest

from tau3.synthesis import world_tasks


@pytest.fixture
def adapter(tmp_path):
    source = Path(world_tasks.__file__).parents[1]
    for name in (
        "synthesis/world_tasks.py",
        "synthesis/bundle.py",
        "synthesis/cli.py",
        "synthesis/concurrent_audit.py",
        "utils/llm_concurrency.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return tmp_path


def check(expected, root):
    script = Path(__file__).resolve().parents[3] / "scripts/resume_world_streaming.py"
    return runpy.run_path(str(script))["check_adapter_compatibility"](expected, root)


ORIGINAL = "44de23d6cdb22a3d4f3df8609fda42645bdc6c26b488f1cb0127db8d3ed70cf1"


def test_exact_targeted_additions_reconstruct_original_hash(adapter):
    result = check(ORIGINAL, adapter)
    assert result["original"] == ORIGINAL
    assert result["current"] != ORIGINAL
    assert len(result["allowed_additions"]) == 2
    assert check(result["current"], adapter)["allowed_additions"] == []


@pytest.mark.parametrize(
    "name", ["synthesis/world_tasks.py", "synthesis/cli.py", "utils/llm_concurrency.py"]
)
def test_other_changes_are_not_accepted(adapter, name):
    path = adapter / name
    path.write_text(path.read_text() + "\n# Another change\n")
    with pytest.raises(ValueError, match="beyond additive"):
        check(ORIGINAL, adapter)
