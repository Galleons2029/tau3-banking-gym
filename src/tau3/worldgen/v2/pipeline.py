"""Resumable construction, counterexample validation and certificate publication."""

import json
import os
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from tau3.worldgen.v2.category_review import admitted, review_category
from tau3.worldgen.v2.diversity import diversity_report, structural_fingerprint
from tau3.worldgen.v2.documents import (
    Article,
    build_articles,
    claim_catalog,
    evidence_closure,
    minimal_cover,
    review_article,
    rewrite_article,
)
from tau3.worldgen.v2.expressions import evaluate
from tau3.worldgen.v2.runtime import (
    OperationRuntime,
    WorldDB,
    digest,
    goal_errors,
    serializable,
    validate_db,
)
from tau3.worldgen.v2.specs import WorldSpec

ARTIFACTS = (
    "spec.json",
    "db.json",
    "documents",
    "tasks",
    "private",
    "splits.json",
    "build.json",
)


def write_json(path: Path, value) -> None:
    """Atomic artifact write; interrupted writes cannot masquerade as completed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(serializable(value), indent=2, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def implementation_hash() -> str:
    """Pin both generation and execution code, including the domain adapter."""
    tau = Path(__file__).resolve().parents[2]
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *[(tau / "synthesis/targeted/controls.py")],
        *[
            tau / p
            for p in (
                "domains/banking_synth/environment.py",
                "evaluator/evaluator.py",
                "evaluator/evaluator_env.py",
                "evaluator/evaluator_nl_assertions.py",
                "environment/environment.py",
                "environment/toolkit.py",
                "environment/db.py",
                "config.py",
                "utils/llm_utils.py",
                "runner/helpers.py",
                "data_model/simulation.py",
            )
        ],
    ]
    return digest({str(p.relative_to(tau)): p.read_text() for p in sorted(paths)})


def artifact_hashes(root: Path) -> dict[str, str]:
    """Hash exact artifact membership and bytes; additions invalidate certificates."""
    hashes = {}
    for name in ARTIFACTS:
        path = root / name
        if path.is_file():
            hashes[name] = digest(path.read_text())
        elif path.is_dir():
            hashes[name] = digest(
                {
                    str(p.relative_to(path)): p.read_text()
                    for p in sorted(path.rglob("*"))
                    if p.is_file()
                }
            )
        else:
            raise ValueError(f"Missing required artifact: {name}")
    return hashes


def initial_database(spec: WorldSpec) -> WorldDB:
    """Build valid scenario records and unique identities in one isolated seed DB."""
    db = WorldDB(tables={e.id: {} for c in spec.categories for e in c.entities})
    for category in spec.categories:
        for scenario in category.scenarios:
            uid = scenario.user_id
            joined = (
                date.fromisoformat(spec.clock)
                - timedelta(days=1 if scenario.kind == "denial" else 365)
            ).isoformat()
            user = {
                "name": uid.replace("_", " ").title(),
                "email": f"{uid}@example.invalid",
                "joined_on": joined,
            }
            user.update(scenario.customer)
            if uid in db.users and db.users[uid] != user:
                raise ValueError("Inconsistent customer history across scenarios")
            db.users[uid] = user
            for table, rows in scenario.initial_rows.items():
                if table not in db.tables:
                    raise ValueError("Unknown scenario table")
                for rid, row in rows.items():
                    if rid in db.tables[table] and db.tables[table][rid] != row:
                        raise ValueError("Inconsistent shared initial row")
                    db.tables[table][rid] = deepcopy(row)
    # Explicit synthetic opening snapshots, not fabricated real transaction logs.
    db.events = [
        {
            "operation": "seed_snapshot",
            "actor": "generator",
            "clock": row.get("opened_on", spec.clock),
            "policy_violations": [],
            "changes": [
                {"table": table, "row_id": rid, "before": None, "after": deepcopy(row)}
            ],
        }
        for table, rows in sorted(db.tables.items())
        for rid, row in sorted(rows.items())
    ]
    validate_db(spec, db)
    return db


def task_payload(spec, scenario, gold, aliases, initial):
    """Export the standard Tau task schema; goals and plans remain private."""
    from tau3.data_model.tasks import Task

    actions = []
    for i, step in enumerate(scenario.steps if not scenario.expected_denial else []):
        alias = next(a for a, op in aliases.items() if op == step.operation)
        operation = next(
            o for c in spec.categories for o in c.operations if o.id == step.operation
        )
        if operation.actor == "user":
            actions.extend(
                [
                    {
                        "action_id": f"{scenario.id}_{i}_give",
                        "name": "give_discoverable_user_tool",
                        "arguments": {"capability": alias},
                        "requestor": "assistant",
                    },
                    {
                        "action_id": f"{scenario.id}_{i}_call",
                        "name": "call_discoverable_user_tool",
                        "arguments": {
                            "capability": alias,
                            "arguments": json.dumps(step.arguments),
                        },
                        "requestor": "user",
                    },
                ]
            )
        else:
            actions.extend(
                [
                    {
                        "action_id": f"{scenario.id}_{i}_unlock",
                        "name": "unlock_discoverable_agent_tool",
                        "arguments": {"capability": alias},
                        "requestor": "assistant",
                    },
                    {
                        "action_id": f"{scenario.id}_{i}_call",
                        "name": "call_discoverable_agent_tool",
                        "arguments": {
                            "capability": alias,
                            "arguments": json.dumps(step.arguments),
                        },
                        "requestor": "assistant",
                    },
                ]
            )
    # Outcome assertion calls inspect private goals in the evaluator environment;
    # they are not decorated as agent tools.
    criteria = {
        "actions": actions,
        "reward_basis": ["ENV_ASSERTION"],
        "env_assertions": [
            {
                "env_type": "assistant",
                "func_name": "check_scenario_outcome",
                "arguments": {"scenario_id": scenario.id},
                "assert_value": True,
            }
        ],
    }
    from tau3.worldgen.v2.semantic import COMMUNICATION_INTEGRITY

    criteria["nl_assertions"] = [*scenario.communication, COMMUNICATION_INTEGRITY]
    criteria["reward_basis"].append("NL_ASSERTION")
    user = initial.users[scenario.user_id]
    payload = {
        "id": f"task_{scenario.id}",
        "description": {"purpose": scenario.kind},
        "user_scenario": {
            "instructions": f"You are {user['name']}, email {user['email']}, customer identifier {scenario.user_id}. {scenario.request} Provide identity when asked. Do not invent policy or product names you were not given. Wait for the agent to complete the requested operation or explain a refusal. Use a user tool only after the agent gives it to you."
            + (" " + " ".join(scenario.user_behavior) if scenario.user_behavior else "")
        },
        "evaluation_criteria": criteria,
        "required_documents": gold,
    }
    return Task.model_validate(payload).model_dump(mode="json")


def validate_scenario(spec, category, scenario, initial, articles) -> dict:
    """Verify independent outcomes, complete evidence and adversarial trajectories."""
    errors, counterexamples = [], []
    runtime = OperationRuntime(spec, initial.model_copy(deep=True))
    aliases = {o: a for a, o in runtime.aliases.items()}
    if scenario.selection:
        if category.description.casefold() not in scenario.request.casefold():
            errors.append(
                "Selection request must explicitly identify its category scope"
            )
        answers = [
            p.id
            for p in category.products
            if evaluate(
                scenario.selection, {"facts": p.values(spec.clock), "clock": spec.clock}
            )
            is True
        ]
        if answers != [scenario.expected_product]:
            errors.append(f"Non-unique or incorrect product selection: {answers}")
        if any(
            s.arguments.get("product_id") != scenario.expected_product
            for s in scenario.steps
        ):
            errors.append("Reference operation disagrees with selected product")
    required = evidence_closure(spec, category, scenario)
    gold = minimal_cover(required, articles)
    by_id = {a.id: a for a in articles}
    covered = set().union(*(by_id[d].claims.keys() for d in gold))
    if not required <= covered:
        errors.append("Incomplete evidence")
    # For controlled prose, independently check the actual text against the
    # authoritative claim catalog, not just DocPlan's declared coverage.
    catalog = claim_catalog(spec, runtime.aliases)
    for doc_id in gold:
        article = by_id[doc_id]
        if article.style == "controlled" and any(
            catalog[k] not in article.content for k in article.claims
        ):
            errors.append(f"Evidence missing from rendered body: {doc_id}")
    for doc_id in gold:
        without = set().union(*(by_id[d].claims.keys() for d in gold if d != doc_id))
        if required <= without:
            errors.append(f"Redundant gold document: {doc_id}")

    def run(steps):
        env = OperationRuntime(spec, initial.model_copy(deep=True))
        failures = []
        for step in steps:
            op = next(o for o in category.operations if o.id == step.operation)
            try:
                env.execute(aliases[step.operation], step.arguments, op.actor)
            except ValueError as exc:
                failures.append(str(exc))
                break
        return env.db, failures

    actual, failures = run(scenario.steps)
    if scenario.expected_denial:
        if failures != [f"POLICY_DENIED:{scenario.expected_denial}"]:
            errors.append(f"Expected policy refusal, received: {failures}")
        if actual != initial:
            errors.append("Refused request mutated state")
    elif failures:
        errors.extend(failures)
    else:
        errors.extend(
            goal_errors(initial, actual, scenario.goals, scenario.user_id, spec)
        )
    if scenario.goals:
        mutations = [("no_operation", [])]
        mutations += [
            (f"omit_step_{i}", scenario.steps[:i] + scenario.steps[i + 1 :])
            for i in range(len(scenario.steps))
            if next(
                o for o in category.operations if o.id == scenario.steps[i].operation
            ).kind
            == "write"
        ]
        if len(scenario.steps) > 1 and scenario.kind == "ordering":
            mutations.append(("reverse_order", list(reversed(scenario.steps))))
        for label, steps in mutations:
            result, failures = run(steps)
            rejected = bool(
                failures
                or goal_errors(initial, result, scenario.goals, scenario.user_id, spec)
            )
            counterexamples.append({"case": label, "rejected": rejected})
            if not rejected:
                errors.append(f"Unnecessary workflow step or scoring shortcut: {label}")
        for i, step in enumerate(scenario.steps):
            operation = next(o for o in category.operations if o.id == step.operation)
            if operation.kind == "read":
                probe = OperationRuntime(spec, initial.model_copy(deep=True))
                before = probe.db.model_copy(deep=True)
                probe.execute(aliases[step.operation], step.arguments)
                if probe.db != before:
                    errors.append("Read operation changed business state")
                continue
            wrong = step.model_copy(deep=True)
            wrong.arguments["user_id"] = "nonexistent_customer"
            result, failures = run(
                scenario.steps[:i] + [wrong] + scenario.steps[i + 1 :]
            )
            if not failures:
                errors.append("Wrong-customer operation accepted")
            counterexamples.append(
                {"case": f"wrong_customer_{i}", "rejected": bool(failures)}
            )
            # Test a real other customer, not only a missing primary key.
            other = next(
                (uid for uid in initial.users if uid != scenario.user_id), None
            )
            candidates = [("wrong_product", "product_id", "unknown_product")]
            if other:
                candidates.append(("other_customer", "user_id", other))
            if "amount" in step.arguments:
                candidates.extend(
                    (f"invalid_amount_{j}", "amount", amount)
                    for j, amount in enumerate(["-1", "0", "999999999999", "NaN"])
                )
            for label, key, value in candidates:
                wrong = step.model_copy(deep=True)
                wrong.arguments[key] = value
                result, failures = run(
                    scenario.steps[:i] + [wrong] + scenario.steps[i + 1 :]
                )
                rejected = bool(
                    failures
                    or goal_errors(
                        initial, result, scenario.goals, scenario.user_id, spec
                    )
                )
                counterexamples.append({"case": f"{label}_{i}", "rejected": rejected})
                if not rejected:
                    errors.append(f"Counterexample accepted: {label}_{i}")
    if scenario.kind in {"grounding", "denial"} and not scenario.communication:
        errors.append("Missing communication scoring")
    return {
        "status": "FAIL" if errors else "PASS",
        "scenario": scenario.id,
        "errors": errors,
        "gold": gold,
        "requires": sorted(required),
        "minimality": "inclusion-minimal under declared proof",
        "counterexamples": counterexamples,
        "communication": "requires_runtime_semantic_judge"
        if scenario.communication
        else "not_applicable",
    }


def validate_world(root: Path) -> dict:
    """Validate the exact persisted bundle; never trusts an old stage flag."""
    spec = WorldSpec.model_validate_json((root / "spec.json").read_text())
    initial = WorldDB.model_validate_json((root / "db.json").read_text())
    validate_db(spec, initial)
    # A tampered initial state must not redefine the task while retaining its goal.
    if initial != initial_database(spec):
        raise ValueError("Seed DB differs from declared scenario state")
    articles = [
        Article.model_validate_json(p.read_text())
        for p in sorted((root / "private/articles").glob("*.json"))
    ]
    runtime = OperationRuntime(spec, initial)
    expected = {a.id: a for a in build_articles(spec, runtime.aliases)}
    if {a.id for a in articles} != set(expected):
        raise ValueError("Missing or unexpected articles")
    errors, reports = [], []
    config = json.loads((root / "build.json").read_text())
    for category in spec.categories:
        path = root / f"private/category_reviews/{category.id}.json"
        reviews = json.loads(path.read_text()) if path.exists() else []
        if not admitted(
            category, spec.clock, initial, reviews, config["review_models"]
        ):
            errors.append(f"Missing independent task/goal review: {category.id}")
    for article in articles:
        public = json.loads((root / f"documents/{article.id}.json").read_text())
        if public != {
            "id": article.id,
            "title": article.title,
            "content": article.content,
        }:
            errors.append(f"Public/private document mismatch: {article.id}")
        canonical = expected[article.id]
        if article.style != config["text_mode"]:
            errors.append(f"Text mode bypass: {article.id}")
        if (
            article.claims != canonical.claims
            or article.crossrefs != canonical.crossrefs
        ):
            errors.append(f"Claim provenance changed: {article.id}")
        if article.style == "controlled":
            if article != canonical:
                errors.append(f"Unreviewed controlled prose mutation: {article.id}")
        else:
            audit_path = root / f"private/reviews/{article.id}.json"
            reviews = json.loads(audit_path.read_text()) if audit_path.exists() else []
            models = {r.get("model") for r in reviews}
            if (
                models != set(config["review_models"])
                or len(models) < 2
                or any(
                    r.get("status") != "PASS"
                    or r.get("article_hash") != digest(article.model_dump())
                    for r in reviews
                )
            ):
                errors.append(
                    f"Missing, failed or stale semantic reviews: {article.id}"
                )
    ids = set(expected)
    if {p.stem for p in (root / "documents").glob("*.json")} != ids:
        errors.append("Unexpected public documents")
    task_ids = set()
    for category in spec.categories:
        for scenario in category.scenarios:
            result = validate_scenario(spec, category, scenario, initial, articles)
            reports.append(result)
            task_id = f"task_{scenario.id}"
            task_ids.add(task_id)
            actual = json.loads((root / f"tasks/{task_id}.json").read_text())
            if actual != task_payload(
                spec, scenario, result["gold"], runtime.aliases, initial
            ):
                errors.append(f"Task differs from verified scenario: {task_id}")
    if {p.stem for p in (root / "tasks").glob("*.json")} != task_ids:
        errors.append("Unexpected task membership")
    splits = json.loads((root / "splits.json").read_text())
    if set(splits.get("base", [])) != task_ids:
        errors.append("Incomplete base split")
    train, test = set(splits.get("train", [])), set(splits.get("test", []))
    if spec.split_policy == "all_train" and (test or train != task_ids):
        errors.append("All-train world has a non-training partition")
    if train & test or train | test != task_ids:
        errors.append("Invalid train/test split")
    families = {}
    for category in spec.categories:
        for scenario in category.scenarios:
            family = structural_fingerprint(category, scenario)
            partition = "train" if f"task_{scenario.id}" in train else "test"
            if family in families and families[family] != partition:
                errors.append("Structural family leaks across splits")
            families[family] = partition
    report = {
        "status": "PASS"
        if not errors and all(r["status"] == "PASS" for r in reports)
        else "FAIL",
        "errors": errors,
        "scenarios": reports,
        "counts": {
            "categories": len(spec.categories),
            "products": sum(len(c.products) for c in spec.categories),
            "documents": len(articles),
            "tasks": len(reports),
            "structural_families": len(families),
        },
        "task_kinds": dict(
            Counter(s.kind for c in spec.categories for s in c.scenarios)
        ),
        "text_mode": config["text_mode"],
        "purpose": spec.purpose,
        "difficulty": "INCONCLUSIVE: no matched behavioural experiment attached",
    }
    certificate = {
        "schema_version": 2,
        "status": report["status"],
        "artifacts": artifact_hashes(root),
        "implementation": implementation_hash(),
        "report_hash": digest(report),
        "scope": "structural_and_controlled_text"
        if config["text_mode"] == "controlled"
        else "structural_and_model_reviewed_text",
    }
    report["diversity"] = diversity_report(spec)
    certificate["report_hash"] = digest(report)
    write_json(root / "validation/report.json", report)
    write_json(root / "validation/certificate.json", certificate)
    return report


def check_certificate(root: Path) -> dict:
    """Fail closed on incomplete checks, code drift or changed artifacts."""
    path = root / "validation/certificate.json"
    if not path.exists():
        raise ValueError("Missing validation certificate; run validate-v2")
    certificate = json.loads(path.read_text())
    report = json.loads((root / "validation/report.json").read_text())
    if certificate.get("status") != "PASS" or report.get("status") != "PASS":
        raise ValueError("Validation did not pass")
    if (
        certificate.get("artifacts") != artifact_hashes(root)
        or certificate.get("implementation") != implementation_hash()
        or certificate.get("report_hash") != digest(report)
    ):
        raise ValueError("Stale validation certificate")
    return certificate


def publish(root: Path) -> dict:
    """Publish a validated immutable snapshot with explicit assurance scope."""
    certificate = check_certificate(root)
    spec = WorldSpec.model_validate_json((root / "spec.json").read_text())
    if spec.purpose == "paper_reproduction":
        from tau3.worldgen.v2.behaviour import check_qualification

        qualification = check_qualification(root)
        if certificate["scope"] != "structural_and_model_reviewed_text":
            raise ValueError("Paper reproduction requires model-reviewed natural prose")
    manifest = {
        "schema_version": 2,
        "status": "published",
        "domain": "banking_synth",
        "purpose": spec.purpose,
        "certificate_hash": digest(certificate),
        "spec_hash": digest(spec.model_dump()),
        "assurance": "foundation_only; expansion readiness requires a separate current readiness report",
    }
    if spec.purpose == "paper_reproduction":
        manifest["behaviour_hash"] = digest(qualification)
    write_json(root / "manifest.json", manifest)
    return manifest


def build(
    root: Path,
    spec: WorldSpec,
    text_mode: str = "controlled",
    render_model: str | None = None,
    review_models: list[str] | None = None,
    llm_args: dict | None = None,
    max_repairs: int = 3,
    max_model_calls: int = 10000,
) -> dict:
    """Build or resume one exact configuration; unresolved articles are isolated."""
    review_models, llm_args = review_models or [], llm_args or {}
    if text_mode not in {"controlled", "llm"}:
        raise ValueError("Unknown text mode")
    if text_mode == "llm" and (
        not render_model
        or len(set(review_models)) < 2
        or not any(m != render_model for m in review_models)
    ):
        raise ValueError(
            "LLM text needs a renderer and two distinct reviewer models, including another model"
        )
    if not 1 <= max_repairs <= 5 or max_model_calls < 0:
        raise ValueError("Invalid bounded repair/call budget")
    config = {
        "text_mode": text_mode,
        "render_model": render_model,
        "review_models": sorted(set(review_models)),
        "llm_args_hash": digest(llm_args),
        "max_repairs": max_repairs,
        "max_model_calls": max_model_calls,
        "spec_hash": digest(spec.model_dump()),
    }
    if (root / "build.json").exists():
        if json.loads((root / "build.json").read_text()) != config:
            raise ValueError("Resume configuration differs; use a new world directory")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite existing world; use a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if (
        manifest.exists()
        and json.loads(manifest.read_text()).get("status") == "published"
    ):
        check_certificate(root)
        return json.loads((root / "validation/report.json").read_text())
    write_json(root / "build.json", config)
    write_json(root / "manifest.json", {"schema_version": 2, "status": "draft"})
    write_json(root / "spec.json", spec.model_dump(mode="json"))
    initial = initial_database(spec)
    write_json(root / "db.json", initial.model_dump(mode="json"))
    runtime = OperationRuntime(spec, initial)
    articles = build_articles(spec, runtime.aliases)
    catalog = claim_catalog(spec, runtime.aliases)
    progress_path = root / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"model_calls_reserved": 0, "quarantined": []}
    )
    for category in spec.categories:
        path = root / f"private/category_reviews/{category.id}.json"
        reviews = json.loads(path.read_text()) if path.exists() else []
        if admitted(category, spec.clock, initial, reviews, review_models):
            continue
        if (
            len(set(review_models)) < 2
            or progress["model_calls_reserved"] + len(review_models) > max_model_calls
        ):
            return {
                "status": "INCONCLUSIVE",
                "reason": "Novel categories require two independent task/goal reviews within budget",
            }
        progress["model_calls_reserved"] += len(review_models)
        write_json(progress_path, progress)
        reviews = [
            review_category(category, spec.clock, initial, model, llm_args)
            for model in review_models
        ]
        write_json(path, reviews)
        if not admitted(category, spec.clock, initial, reviews, review_models):
            progress["quarantined"] = sorted(
                set(progress["quarantined"]) | {category.id}
            )
            write_json(progress_path, progress)
            return {
                "status": "INCONCLUSIVE",
                "reason": "Independent task/goal review failed",
                "progress": progress,
            }
    for index, article in enumerate(articles):
        cached = root / f"private/articles/{article.id}.json"
        review_path = root / f"private/reviews/{article.id}.json"
        if text_mode == "llm":
            reviews = (
                json.loads(review_path.read_text()) if review_path.exists() else []
            )
            if cached.exists() and reviews:
                candidate = Article.model_validate_json(cached.read_text())
                if (
                    candidate.claims == article.claims
                    and {r.get("model") for r in reviews} == set(review_models)
                    and all(
                        r.get("status") == "PASS"
                        and r.get("article_hash") == digest(candidate.model_dump())
                        for r in reviews
                    )
                ):
                    articles[index] = candidate
                    write_json(
                        root / f"documents/{candidate.id}.json",
                        {
                            "id": candidate.id,
                            "title": candidate.title,
                            "content": candidate.content,
                        },
                    )
                    continue
            accepted = False
            issues = []
            for attempt in range(max_repairs):
                cost = 1 + 2 * len(review_models)
                if progress["model_calls_reserved"] + cost > max_model_calls:
                    break
                progress["model_calls_reserved"] += cost
                write_json(progress_path, progress)
                try:
                    candidate = rewrite_article(article, render_model, llm_args, issues)
                    reviews = [
                        review_article(candidate, catalog, model, llm_args)
                        for model in review_models
                    ]
                    write_json(
                        root
                        / f"attempts/{article.id}_{attempt}_{progress['model_calls_reserved']}.json",
                        {"article": candidate.model_dump(), "reviews": reviews},
                    )
                    if all(r["status"] == "PASS" for r in reviews):
                        article = candidate
                        write_json(review_path, reviews)
                        accepted = True
                        break
                    issues = [
                        str(issue)
                        for review in reviews
                        for issue in review.get("verdict", {}).get("issues", [])
                    ]
                except Exception as exc:
                    write_json(
                        root / f"attempts/{article.id}_error.json",
                        {"error": type(exc).__name__},
                    )
            if not accepted:
                progress["quarantined"] = sorted(
                    set(progress["quarantined"]) | {article.id}
                )
                write_json(progress_path, progress)
                return {
                    "status": "INCONCLUSIVE",
                    "reason": "Semantic validation failed or budget exhausted",
                    "progress": progress,
                }
        articles[index] = article
        write_json(cached, article.model_dump(mode="json"))
        write_json(
            root / f"documents/{article.id}.json",
            {"id": article.id, "title": article.title, "content": article.content},
        )
    splits = {"base": [], "train": [], "test": []}
    for category in spec.categories:
        for scenario in category.scenarios:
            gold = minimal_cover(evidence_closure(spec, category, scenario), articles)
            task = task_payload(spec, scenario, gold, runtime.aliases, initial)
            write_json(root / f"tasks/{task['id']}.json", task)
            splits["base"].append(task["id"])
            partition = (
                "test"
                if spec.split_policy != "all_train"
                and int(structural_fingerprint(category, scenario)[:8], 16) % 5 == 0
                else "train"
            )
            splits[partition].append(task["id"])
    write_json(root / "splits.json", splits)
    progress["quarantined"] = []
    write_json(progress_path, progress)
    return validate_world(root)
