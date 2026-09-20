"""Stage G: solvability replay and minimal gold cover, including what they reject."""

import json

import pytest

from tau3.data_model.tasks import Task
from tau3.synthesis.validation import replay
from tau3.worldgen.seed import build_seed_world
from tau3.worldgen.verify import (
    TaskVerdict,
    discoverable_routes,
    verify_gold_cover,
    verify_tool_evidence,
    verify_world,
)
from tau3.worldgen.workflow import read_config
from tau3.worldgen.world import (
    DRAFT_ENV_VAR,
    WORLD_ENV_VAR,
    load_plan,
    read_world_manifest,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("worlds") / "seed"
    build_seed_world(root, read_config(), "test-seed")
    return root


@pytest.fixture(scope="module")
def configured(world):
    import os

    previous = {n: os.environ.get(n) for n in (WORLD_ENV_VAR, DRAFT_ENV_VAR)}
    os.environ[WORLD_ENV_VAR] = str(world)
    os.environ[DRAFT_ENV_VAR] = "1"
    yield world
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def task_from(world, task_id) -> Task:
    return Task.model_validate(
        json.loads((world / "tasks" / f"{task_id}.json").read_text())
    )


def test_every_seed_task_is_solvable_with_a_minimal_gold_set(configured):
    report = verify_world(configured)
    failures = {
        verdict["task_id"]: verdict["errors"]
        for verdict in report["verdicts"]
        if verdict["errors"]
    }
    assert not failures, failures
    assert report["passed"] == report["tasks"] == 5


def test_closing_before_opening_makes_the_ordering_task_unsolvable(configured):
    # The official tool refuses a business account for a customer with any closed
    # account, so the customer's preferred order is the one that cannot work.
    task = task_from(configured, "task_003")
    actions = task.evaluation_criteria.actions
    opening = [a for a in actions if "business_checking" in json.dumps(a.arguments)]
    closures = [a for a in actions if "acc_pr_savings" in json.dumps(a.arguments)]
    others = [a for a in actions if a not in opening and a not in closures]
    reordered = others + closures + opening

    with pytest.raises(ValueError, match="Reference action failed"):
        replay(task, reordered, strict=True, domain="banking_synth")
    # The recorded order still replays cleanly.
    replay(task, strict=True, domain="banking_synth")


def test_an_incomplete_gold_set_is_rejected(configured):
    plan = load_plan(configured)
    plans = {d.doc_id: d for d in plan.documents}
    spec = next(s for s in plan.tasks if s.task_id == "task_001")
    gold = task_from(configured, "task_001").required_documents

    verdict = TaskVerdict(task_id="task_001")
    verify_gold_cover(
        spec, plans, [g for g in gold if g != "doc_checking_zinc_002"], verdict
    )
    assert not verdict.complete
    assert verdict.missing_knowledge == ["var_zinc_early_direct_deposit_days"]


def test_a_redundant_gold_document_is_rejected(configured):
    plan = load_plan(configured)
    plans = {d.doc_id: d for d in plan.documents}
    spec = next(s for s in plan.tasks if s.task_id == "task_001")
    gold = list(task_from(configured, "task_001").required_documents)

    verdict = TaskVerdict(task_id="task_001")
    verify_gold_cover(spec, plans, gold + ["doc_checking_copper_003"], verdict)
    assert verdict.complete
    assert not verdict.minimal
    assert verdict.redundant_documents == ["doc_checking_copper_003"]


def test_gold_referencing_an_unplanned_document_is_rejected(configured):
    plan = load_plan(configured)
    plans = {d.doc_id: d for d in plan.documents}
    spec = next(s for s in plan.tasks if s.task_id == "task_001")
    verdict = TaskVerdict(task_id="task_001")
    verify_gold_cover(spec, plans, ["doc_not_in_plan"], verdict)
    assert "unplanned documents" in verdict.errors[0]


def test_declared_tools_are_compared_in_the_official_namespace(configured):
    manifest = read_world_manifest(configured)
    task = task_from(configured, "task_002")
    spec = next(s for s in load_plan(configured).tasks if s.task_id == "task_002")

    # Reference actions name this world's aliases.
    assert discoverable_routes(task) == {
        next(
            a
            for a, o in manifest.alias_map.items()
            if o == "get_all_user_accounts_by_user_id_3847"
        )
    }
    verdict = TaskVerdict(task_id="task_002")
    verify_tool_evidence(task, spec, manifest.alias_map, verdict)
    assert not verdict.errors

    # Without the alias map the two namespaces cannot agree, and the check says so.
    stale = TaskVerdict(task_id="task_002")
    verify_tool_evidence(task, spec, {}, stale)
    assert stale.errors


def test_a_task_with_no_specification_is_not_silently_passed(configured, tmp_path):
    import shutil

    copy = tmp_path / "world"
    shutil.copytree(configured, copy)
    plan = load_plan(copy)
    plan.tasks = [s for s in plan.tasks if s.task_id != "task_004"]
    from tau3.worldgen.world import write_plan

    write_plan(copy, plan)
    report = verify_world(copy)
    orphan = next(v for v in report["verdicts"] if v["task_id"] == "task_004")
    assert orphan["errors"] == ["No task specification for this task"]


def test_a_verbatim_argument_the_customer_never_states_is_rejected(configured):
    # The DB evaluator compares the closure reason by exact string, and the
    # customer is its only source, so an unstated reason makes the task
    # unwinnable however well the agent reasons.
    from tau3.worldgen.verify import verify_verbatim_arguments

    task = task_from(configured, "task_003")
    spec = next(s for s in load_plan(configured).tasks if s.task_id == "task_003")
    assert spec.verbatim_arguments == ["Customer requested closure"]

    verdict = TaskVerdict(task_id="task_003")
    verify_verbatim_arguments(task, spec, verdict)
    assert not verdict.errors

    silent = task.model_copy(deep=True)
    silent.user_scenario.instructions = "You want to close your savings accounts."
    quiet = TaskVerdict(task_id="task_003")
    verify_verbatim_arguments(silent, spec, quiet)
    assert "never stated in the user instructions" in quiet.errors[0]
