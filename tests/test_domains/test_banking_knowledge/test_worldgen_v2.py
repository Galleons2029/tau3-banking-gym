"""Adversarial checks for generated contracts, corpus certificates and execution."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tau3.worldgen.v2.catalog import scaffold
from tau3.worldgen.v2.diversity import structural_fingerprint
from tau3.worldgen.v2.documents import build_articles, evidence_closure, minimal_cover
from tau3.worldgen.v2.environment import get_environment, load_spec
from tau3.worldgen.v2.expressions import evaluate
from tau3.worldgen.v2.pipeline import (
    build,
    check_certificate,
    initial_database,
    publish,
    validate_world,
)
from tau3.worldgen.v2.runtime import OperationRuntime, goal_errors
from tau3.worldgen.v2.specs import WorldSpec


@pytest.fixture
def spec():
    return WorldSpec(categories=[scaffold("term_deposits", "term_deposit")])


@pytest.fixture
def bundle(tmp_path, spec):
    root = tmp_path / "world"
    report = build(root, spec)
    assert report["status"] == "PASS", report
    publish(root)
    return root


def test_build_revalidate_resume_and_publish_are_deterministic(bundle, spec):
    before = check_certificate(bundle)
    assert validate_world(bundle)["status"] == "PASS"
    assert check_certificate(bundle) == before
    assert build(bundle, spec)["status"] == "PASS"
    assert load_spec(bundle) == spec


@pytest.mark.parametrize(
    "artifact", ["documents", "tasks", "db.json", "spec.json", "build.json"]
)
def test_every_artifact_mutation_invalidates_publication(bundle, artifact):
    path = bundle / artifact
    if path.is_dir():
        path = next(path.glob("*.json"))
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Stale"):
        publish(bundle)
    with pytest.raises(ValueError, match="Stale"):
        load_spec(bundle)


def test_unreviewed_prose_cannot_be_recertified(bundle):
    public = next((bundle / "documents").glob("*.json"))
    private = bundle / "private/articles" / public.name
    for path in [public, private]:
        value = json.loads(path.read_text())
        value["content"] += "\nThis product guarantees a 99 percent crypto return."
        path.write_text(json.dumps(value))
    assert validate_world(bundle)["status"] == "FAIL"
    with pytest.raises(ValueError, match="did not pass"):
        publish(bundle)


def test_derived_evidence_includes_dates_and_competitors(spec):
    category = spec.categories[0]
    required = evidence_closure(spec, category, category.scenarios[0])
    for product in category.products:
        for fact in [
            "effective_rate",
            "base_rate",
            "promotion_boost",
            "promotion_start",
            "promotion_end",
        ]:
            assert f"{product.id}.{fact}" in required
    runtime = OperationRuntime(spec, initial_database(spec))
    articles = build_articles(spec, runtime.aliases)
    assert minimal_cover(required, articles) == minimal_cover(
        required, list(reversed(articles))
    )


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "facts.__class__",
        "[x for x in facts]",
        "2 ** 99999",
        "money()",
    ],
)
def test_expression_interpreter_rejects_unsupported_capabilities(expression):
    with pytest.raises(ValueError):
        evaluate(expression, {"facts": {}})


def test_decimal_dates_and_missing_dependencies(spec):
    assert evaluate("money(0.1 + 0.2)", {}) == Decimal("0.30")
    assert evaluate("days('2025-11-14', '2025-11-01')", {}) == 13
    raw = spec.model_dump()
    raw["categories"][0]["products"][0]["facts"][-1]["depends_on"] = ["base_rate"]
    with pytest.raises(ValueError, match="dependencies"):
        WorldSpec.model_validate(raw)


def test_reference_workflow_and_alternative_amount_partition(bundle, spec):
    scenario = spec.categories[0].scenarios[2]
    initial = initial_database(spec)
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    aliases = {o: a for a, o in runtime.aliases.items()}
    step = scenario.steps[0]
    for amount in ["20", "30"]:
        runtime.execute(aliases[step.operation], {**step.arguments, "amount": amount})
    for step in scenario.steps[1:]:
        op = next(o for o in spec.categories[0].operations if o.id == step.operation)
        runtime.execute(aliases[step.operation], step.arguments, op.actor)
    assert not goal_errors(initial, runtime.db, scenario.goals, scenario.user_id, spec)
    env = get_environment(bundle, retrieval_variant="no_knowledge")
    assert "check_scenario_outcome" not in env.tools.tools
    assert "call_discoverable_user_tool" in env.user_tools.tools
    assert not env.tools.check_scenario_outcome(scenario.id)


def test_full_tau_reference_replay(bundle, monkeypatch):
    from tau3.domains.banking_synth.environment import get_tasks

    monkeypatch.setenv("TAU3_SYNTH_WORLD", str(bundle))
    for task in get_tasks():
        env = get_environment(bundle, retrieval_variant="no_knowledge", task=task)
        for action in task.evaluation_criteria.actions:
            result = env.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
            assert "error" not in str(result).lower(), result
        assertion = task.evaluation_criteria.env_assertions[0]
        assert env.run_env_assertion(assertion, raise_assertion_error=True)


def test_rejected_write_is_atomic_and_worlds_are_isolated(spec):
    initial = initial_database(spec)
    first = OperationRuntime(spec, initial.model_copy(deep=True))
    second = OperationRuntime(spec, initial.model_copy(deep=True))
    step = spec.categories[0].scenarios[1].steps[0]
    alias = next(a for a, o in first.aliases.items() if o == step.operation)
    with pytest.raises(ValueError):
        first.execute(alias, {**step.arguments, "amount": "999999"})
    assert first.db == initial
    first.execute(alias, step.arguments)
    assert first.db != second.db == initial


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"results":[]}',
        '{"results":[{"index":0,"met":"true","reason":"yes"}]}',
        "not json",
    ],
)
def test_empty_or_malformed_semantic_judgment_never_passes(monkeypatch, response):
    from tau3.worldgen.v2 import semantic

    monkeypatch.setattr(
        semantic, "generate", lambda **kwargs: SimpleNamespace(content=response)
    )
    checks = semantic.StrictSemanticEvaluator.evaluate_nl_assertions(
        [], ["Explain refusal"]
    )
    assert len(checks) == 1 and checks[0].met is False


def test_names_do_not_create_structural_novelty():
    first, second = scaffold("first", "checking"), scaffold("second", "savings")
    assert [structural_fingerprint(first, s) for s in first.scenarios] == [
        structural_fingerprint(second, s) for s in second.scenarios
    ]


def test_cloned_profiles_expose_distinct_user_visible_selection_scopes():
    first = scaffold("aurora", "checking")
    second = scaffold("meadow", "checking")
    spec = WorldSpec(categories=[first, second])
    for category in spec.categories:
        assert category.description in category.scenarios[0].request
    assert first.description != second.description
    raw = spec.model_dump()
    raw["categories"][1]["description"] = raw["categories"][0]["description"]
    with pytest.raises(ValueError, match="user-visible scopes"):
        WorldSpec.model_validate(raw)


def test_opening_tenure_is_explicitly_scoped_away_from_withdrawals(spec):
    from tau3.worldgen.v2.documents import claim_catalog

    runtime = OperationRuntime(spec, initial_database(spec))
    aliases = {op: alias for alias, op in runtime.aliases.items()}
    claims = claim_catalog(spec, runtime.aliases)
    tenure = claims["rule:term_deposits_tenure:term_deposits_harbor"]
    assert "This rule applies only to" in tenure
    assert aliases["term_deposits_open"] in tenure
    assert aliases["term_deposits_redeem"] not in tenure
    operation = claims["tool:term_deposits_redeem"]
    assert "term_deposits_active" in operation
    assert "term_deposits_tenure" not in operation


def test_missing_semantic_models_and_budget_fail_closed(tmp_path, spec):
    with pytest.raises(ValueError, match="two distinct"):
        build(
            tmp_path / "missing",
            spec,
            text_mode="llm",
            render_model="one",
            review_models=["one"],
        )
    root = tmp_path / "budget"
    assert (
        build(
            root,
            spec,
            text_mode="llm",
            render_model="one",
            review_models=["one", "two"],
            max_model_calls=0,
        )["status"]
        == "INCONCLUSIVE"
    )
    with pytest.raises(ValueError, match="Missing validation"):
        publish(root)


def test_structural_pass_does_not_claim_paper_reproduction(tmp_path, spec):
    spec = spec.model_copy(update={"purpose": "paper_reproduction"})
    root = tmp_path / "paper"
    assert build(root, spec)["status"] == "PASS"
    with pytest.raises(ValueError, match="behavioural"):
        publish(root)


@pytest.mark.parametrize(
    "profile",
    [
        "checking",
        "savings",
        "term_deposit",
        "credit_card",
        "installment",
        "merchant",
        "rewards",
        "escrow",
    ],
)
def test_all_bootstrap_contracts_pass(tmp_path, profile):
    result = build(
        tmp_path / profile, WorldSpec(categories=[scaffold("demo_" + profile, profile)])
    )
    assert result["status"] == "PASS", result


def test_new_structure_admission_and_duplicate_quarantine(bundle, tmp_path, spec):
    from tau3.worldgen.v2.expansion import expand
    from tau3.worldgen.v2.pipeline import write_json

    package = tmp_path / "category.json"
    write_json(package, scaffold("merchant", "merchant").model_dump(mode="json"))
    assert (
        expand(bundle, tmp_path / "expanded", [package], foundation_only=True)["status"]
        == "PASS"
    )
    assert load_spec(bundle) == spec
    write_json(package, scaffold("renamed", "term_deposit").model_dump(mode="json"))
    with pytest.raises(ValueError, match="No new workflow"):
        expand(bundle, tmp_path / "duplicate", [package], foundation_only=True)


def test_tenure_and_promotion_boundaries(spec):
    from datetime import date, timedelta

    initial = initial_database(spec)
    step = spec.categories[0].scenarios[0].steps[0]
    for age, passes in [(13, False), (14, True), (15, True)]:
        db = initial.model_copy(deep=True)
        db.users[step.arguments["user_id"]]["joined_on"] = (
            date.fromisoformat(spec.clock) - timedelta(days=age)
        ).isoformat()
        runtime = OperationRuntime(spec, db)
        alias = next(a for a, o in runtime.aliases.items() if o == step.operation)
        if passes:
            runtime.execute(alias, step.arguments)
        else:
            with pytest.raises(ValueError, match="POLICY_DENIED"):
                runtime.execute(alias, step.arguments)
    product = spec.categories[0].products[0]
    for clock, rate in [
        ("2025-10-31", "3.5"),
        ("2025-11-01", "4"),
        ("2025-12-31", "4"),
        ("2026-01-01", "3.5"),
    ]:
        assert product.values(clock)["effective_rate"] == Decimal(rate)


def test_two_table_failure_rolls_back_every_effect():
    spec = WorldSpec(categories=[scaffold("merchant", "merchant")])
    initial = initial_database(spec)
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    scenario = spec.categories[0].scenarios[1]
    step = scenario.steps[0]
    alias = next(a for a, o in runtime.aliases.items() if o == step.operation)
    runtime.execute(alias, step.arguments)
    before = runtime.db.model_copy(deep=True)
    # A duplicate posting fails AFTER constructing the proposed balance change.
    with pytest.raises(ValueError, match="already exists"):
        runtime.execute(alias, step.arguments)
    assert runtime.db == before


def test_trajectory_obligation_cannot_be_bypassed_with_right_final_state(spec):
    raw = spec.model_dump()
    raw["categories"][0]["policies"][0]["enforcement"] = "trajectory"
    spec = WorldSpec.model_validate(raw)
    initial = initial_database(spec)
    scenario = spec.categories[0].scenarios[0]
    initial.users[scenario.user_id]["joined_on"] = spec.clock
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    step = scenario.steps[0]
    alias = next(a for a, o in runtime.aliases.items() if o == step.operation)
    runtime.execute(alias, step.arguments)
    assert any(
        "policy" in e
        for e in goal_errors(
            initial, runtime.db, scenario.goals, scenario.user_id, spec
        )
    )


def test_equivalence_intervals_retain_uncertainty_at_extreme_rates():
    from tau3.worldgen.v2.behaviour import wilson

    assert wilson(0, 10)[1] > 0.2
    assert wilson(10, 10)[0] < 0.8
    assert wilson(0, 0) == (0, 1)


def test_incomplete_trials_are_never_comparison_evidence():
    from tau3.worldgen.v2.behaviour import summarize

    task = SimpleNamespace(
        id="task_a",
        model_dump=lambda **kwargs: {
            "evaluation_criteria": {"reward_basis": ["ENV_ASSERTION"]}
        },
    )
    result = SimpleNamespace(
        tasks=[task], info=SimpleNamespace(num_trials=2), simulations=[]
    )
    with pytest.raises(ValueError, match="Incomplete task/trial"):
        summarize(result, {task.id: task.model_dump()})


def test_retrieval_exposes_only_public_corpus(bundle):
    env = get_environment(bundle, retrieval_variant="bm25")
    assert "KB_search" in env.tools.tools
    text = str(env.tools.tools["KB_search"](query="Harbor"))
    assert "Harbor" in text
    assert "check_scenario_outcome" not in text


def test_false_grounding_answer_scores_zero_in_full_evaluator(bundle, monkeypatch):
    from tau3.data_model.message import AssistantMessage
    from tau3.data_model.simulation import SimulationRun, TerminationReason
    from tau3.data_model.tasks import Task
    from tau3.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau3.registry import registry
    from tau3.worldgen.v2 import semantic

    monkeypatch.setenv("TAU3_SYNTH_WORLD", str(bundle))
    monkeypatch.setattr(
        registry,
        "get_env_constructor",
        lambda domain: lambda **kwargs: get_environment(
            bundle, retrieval_variant="no_knowledge"
        ),
    )
    monkeypatch.setattr(
        semantic,
        "generate",
        lambda **kwargs: SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "index": 0,
                            "met": False,
                            "reason": "Assistant invented a guarantee",
                        }
                    ]
                }
            )
        ),
    )
    task = Task.model_validate_json(
        (bundle / "tasks/task_term_deposits_grounding.json").read_text()
    )
    simulation = SimulationRun(
        id="false_answer",
        task_id=task.id,
        start_time="2025-11-14T00:00:00",
        end_time="2025-11-14T00:00:01",
        duration=1,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[
            AssistantMessage(
                role="assistant",
                content="Yes, your crypto return is guaranteed to double.",
            )
        ],
    )
    result = evaluate_simulation(
        simulation, task, EvaluationType.ALL, False, "banking_synth"
    )
    assert result.reward == 0
    assert result.nl_assertions[0].met is False
    with pytest.raises(ValueError, match="complete reward-basis"):
        evaluate_simulation(
            simulation, task, EvaluationType.ENV, False, "banking_synth"
        )


def test_false_posting_amount_cannot_satisfy_conservation():
    raw = scaffold("merchant", "merchant").model_dump()
    raw["operations"][1]["effects"][1]["values"]["amount"] = "money(args.amount + 1)"
    spec = WorldSpec.model_validate({"categories": [raw]})
    initial = initial_database(spec)
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    step = spec.categories[0].scenarios[1].steps[0]
    alias = next(a for a, o in runtime.aliases.items() if o == step.operation)
    with pytest.raises(ValueError, match="postcondition"):
        runtime.execute(alias, step.arguments)
    assert runtime.db == initial


def test_cli_routes_v2_commands():
    import argparse

    from tau3.worldgen.cli import add_worldgen_parser
    from tau3.worldgen.v2.cli import run

    parser = argparse.ArgumentParser()
    add_worldgen_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            "worldgen",
            "build-v2",
            "--world",
            "/tmp/example",
            "--category",
            "/tmp/category",
        ]
    )
    assert args.func is run


def test_successful_review_cache_restores_missing_public_document(
    tmp_path, spec, monkeypatch
):
    from tau3.worldgen.v2 import pipeline
    from tau3.worldgen.v2.runtime import digest

    calls = []

    def rewrite(article, model, llm_args, issues):
        calls.append(article.id)
        return article.model_copy(update={"style": "llm"})

    def review(article, catalog, model, llm_args):
        return {
            "status": "PASS",
            "model": model,
            "article_hash": digest(article.model_dump()),
        }

    monkeypatch.setattr(pipeline, "rewrite_article", rewrite)
    monkeypatch.setattr(pipeline, "review_article", review)
    root = tmp_path / "reviewed"
    kwargs = {
        "text_mode": "llm",
        "render_model": "one",
        "review_models": ["one", "two"],
    }
    assert build(root, spec, **kwargs)["status"] == "PASS"
    count = len(calls)
    path = next((root / "documents").glob("*.json"))
    path.unlink()
    assert build(root, spec, **kwargs)["status"] == "PASS"
    assert path.exists() and len(calls) == count


def test_unreviewed_task_goal_mismatch_cannot_be_published(tmp_path, spec):
    raw = spec.model_dump()
    raw["categories"][0]["scenarios"][0]["request"] = "Please close all my accounts."
    changed = WorldSpec.model_validate(raw)
    result = build(tmp_path / "mismatch", changed)
    assert result["status"] == "INCONCLUSIVE"
    assert "task/goal" in result["reason"]
    with pytest.raises(ValueError, match="Missing validation"):
        publish(tmp_path / "mismatch")


def test_category_review_requires_every_scenario(spec):
    from tau3.worldgen.v2.category_review import admitted, review_identity

    category = spec.categories[0].model_copy(deep=True)
    category.description += " with a proposed variant"
    initial = initial_database(spec)
    reviews = [
        {
            "model": model,
            "status": "PASS",
            "category_hash": review_identity(category, spec.clock, initial),
            "scenarios": [],
        }
        for model in ["one", "two"]
    ]
    assert not admitted(category, spec.clock, initial, reviews, ["one", "two"])


def test_semantic_judge_uses_selected_gateway_and_glm(monkeypatch):
    from tau3.worldgen.v2 import semantic
    from tau3.worldgen.v2.settings import load_settings

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content='{"results":[{"index":0,"met":false,"reason":"No supporting statement"}]}'
        )

    monkeypatch.delenv("TAU3_SYNTH_ROLLOUT_CONFIG", raising=False)
    monkeypatch.setattr(semantic, "generate", generate)
    semantic.StrictSemanticEvaluator.evaluate_nl_assertions([], ["State the rule"])
    assert calls[0]["model"] == "openai/GLM-5.3-Flash"
    assert calls[0]["api_base"] == "http://10.39.62.231:9091/v1"
    assert load_settings().agent_models == [
        "openai/gemini-3.5-flash",
        "openai/GLM-5.3-Flash",
    ]
