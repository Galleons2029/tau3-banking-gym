"""Stage drivers for world synthesis.

Each stage is independently resumable and writes atomic artifacts, so an
interrupted run continues rather than restarting.
"""

import json
import time
from pathlib import Path
from typing import Optional

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.worldgen.models import WorldConfig
from tau3.worldgen.runtime import discoverable_tool_names, suffix_alias_map
from tau3.worldgen.targets import calibrate
from tau3.worldgen.world import (
    TARGETS,
    new_manifest,
    read_world_manifest,
    write_world_manifest,
)


def apply_scale(config: WorldConfig) -> None:
    """Resize the slice to this configuration before any stage reads it.

    Products and scenario families are derived from the topic budget, so every
    stage that imports them has to see the same size. Scale is a setting.
    """
    from tau3.worldgen.slice import configure

    configure(config.num_topics)


def read_config(path: Optional[str] = None) -> WorldConfig:
    """Read JSON or YAML configuration; defaults come from the model."""
    if path is None:
        return WorldConfig()
    import yaml

    return WorldConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def run_calibrate(output: Path) -> dict:
    """Stage 0: measure the installed corpus into acceptance targets."""
    targets = calibrate()
    payload = targets.model_dump(mode="json")
    write_json(output, payload)
    return {
        "targets": str(output),
        "tokenizer": targets.tokenizer,
        "documents": targets.documents,
        "mean_document_tokens": round(targets.document_tokens.mean, 2),
        "topics": targets.topics,
        "categories": targets.categories,
        "category_rule": targets.category_rule,
        "tasks": targets.tasks,
        "mean_gold_documents": round(targets.gold_documents.mean, 2),
        "mean_actions": round(targets.actions.mean, 2),
        "discoverable_tools": len(targets.discoverable_tools),
        "aliasable_tools": len(targets.suffixed_tools),
    }


def run_init_world(root: Path, name: str, config: WorldConfig) -> dict:
    """Create a draft bundle: directory skeleton, alias map and manifest."""
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError(f"World bundle already exists: {root}")
    for directory in ("documents", "tasks", "schema", "registry", "plan", "render"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    targets = calibrate()
    targets_payload = targets.model_dump(mode="json")
    write_json(root / TARGETS, targets_payload)

    alias_map = suffix_alias_map(discoverable_tool_names(), config.suffix_salt)
    manifest = new_manifest(name, config, alias_map, digest(targets_payload))
    manifest.stages["init"] = "done"
    write_world_manifest(root, manifest)
    return {
        "world": str(root),
        "status": manifest.status,
        "aliased_tools": len(alias_map),
        "suffix_salt": config.suffix_salt,
        "targets": str(root / TARGETS),
    }


def run_publish(root: Path) -> dict:
    """Reject uncertified legacy publication; retain actionable integrity errors."""
    root = Path(root)
    read_world_manifest(root)
    documents = list((root / "documents").glob("*.json"))
    tasks = list((root / "tasks").glob("task_*.json"))
    if not documents:
        raise ValueError(f"World {root} has no documents to publish")
    if not tasks:
        raise ValueError(f"World {root} has no tasks to publish")
    if not (root / "db.json").exists():
        raise ValueError(f"World {root} has no db.json to publish")
    raise ValueError(
        "Legacy worlds lack semantic validation certificates. "
        "Use build-v2 / validate-v2 / publish-v2; legacy artifacts are development-only."
    )


def run_seed_world(root: Path, config: WorldConfig, name: str = "m1-seed") -> dict:
    """Write the hand-written world used to settle the schema."""
    from tau3.worldgen.seed import build_seed_world

    return build_seed_world(root, config, name)


def run_lint(root: Path, config: WorldConfig) -> dict:
    """Stage E: structural linters that need no rendered text."""
    from tau3.worldgen.linters import lint_world
    from tau3.worldgen.world import load_plan, load_schema

    root = Path(root)
    plan = load_plan(root)
    rendered = {
        path.stem: read_json(path).get("content", "")
        for path in sorted((root / "documents").glob("*.json"))
    }
    schema = load_schema(root)
    report = lint_world(
        schema,
        plan.documents,
        plan.tasks,
        config.leak_divisor,
        rendered,
    )
    if rendered:
        from tau3.worldgen.text_linters import lint_rendered

        tools_path = root / "registry" / "tools.json"
        tools = read_json(tools_path) if tools_path.exists() else []
        targets = read_json(root / TARGETS)
        report.findings.extend(
            lint_rendered(
                schema,
                plan.documents,
                rendered,
                tools,
                targets["document_tokens"]["mean"],
            ).findings
        )
    report = report.as_dict()
    write_json(root / "lint" / "report.json", report)
    report["report"] = str(root / "lint" / "report.json")
    return report


def run_verify(root: Path) -> dict:
    """Stage G: solvability replay and minimal gold cover."""
    import os

    from tau3.worldgen.verify import verify_world

    root = Path(root)
    # The synth environment reads its world from the environment, and verifying a
    # bundle should not depend on the caller having exported it.
    from tau3.worldgen.world import DRAFT_ENV_VAR, WORLD_ENV_VAR

    previous = {name: os.environ.get(name) for name in (WORLD_ENV_VAR, DRAFT_ENV_VAR)}
    os.environ[WORLD_ENV_VAR] = str(root)
    # Verification is the gate a world passes to earn publication, so it has to
    # be able to load one that has not passed it yet.
    os.environ[DRAFT_ENV_VAR] = "1"
    try:
        report = verify_world(root)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    write_json(root / "verify" / "report.json", report)
    return report


def run_build_schema(root: Path, config: WorldConfig) -> dict:
    """Stage A1: generate the product space's shape, without any values."""
    apply_scale(config)
    from tau3.worldgen.models import Category, Feature, WorldSchema
    from tau3.worldgen.schema_gen import generate_skeleton, to_variables
    from tau3.worldgen.slice import (
        CATEGORIES,
        PRODUCTS,
        PROTOCOLS,
        REQUIRED_VARIABLES,
    )
    from tau3.worldgen.world import write_schema

    root = Path(root)
    categories = [
        Category(
            id=spec.category_id,
            display_name=spec.display_name,
            kind=spec.kind,
            audience=spec.audience,
            topic_budget=sum(1 for p in PRODUCTS if p.category_id == spec.category_id)
            or len(PROTOCOLS),
        )
        for spec in CATEGORIES
    ]
    display = {spec.category_id: spec.display_name for spec in CATEGORIES}

    from concurrent.futures import ThreadPoolExecutor

    from tau3.worldgen.schema_gen import reset_usage, usage_snapshot

    reset_usage()
    started = time.monotonic()

    def skeleton_for(spec):
        extra = max(
            1, config.variables_per_product - len(REQUIRED_VARIABLES[spec.kind])
        )
        return spec, generate_skeleton(spec, display[spec.category_id], config, extra)

    # One generation call per product, run together: serially this is the
    # slowest stage in the pipeline and the configured concurrency went unused.
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        generated = list(pool.map(skeleton_for, PRODUCTS))

    features, variables = [], []
    ranges: dict[str, list[float]] = {}
    for spec, skeletons in generated:
        product_variables = to_variables(spec, skeletons)
        for skeleton, variable in zip(skeletons, product_variables, strict=True):
            ranges[variable.id] = [skeleton.plausible_min, skeleton.plausible_max]
        variables.extend(product_variables)
        features.append(
            Feature(
                id=spec.feature_id,
                category_id=spec.category_id,
                entity_name=spec.entity_name,
                entity_kind="card" if spec.kind == "card" else "account",
                doc_budget=max(2, len(product_variables) // 2),
                variable_ids=[v.id for v in product_variables],
            )
        )
    for feature_id, name in PROTOCOLS:
        features.append(
            Feature(
                id=feature_id,
                category_id="cat_protocol",
                entity_name=name,
                entity_kind="protocol",
                doc_budget=2,
            )
        )

    # Policy rules come from the environment, verified, not from the generator:
    # a rule the official tools do not enforce makes every task resting on it
    # unsolvable.
    from tau3.worldgen.rules import environment_rules

    rules = environment_rules()
    schema = WorldSchema(
        categories=categories, features=features, variables=variables, rules=rules
    )
    usage = usage_snapshot()
    usage["elapsed_seconds"] = round(max(time.monotonic() - started, 1e-6), 1)
    total = usage["prompt_tokens"] + usage["completion_tokens"]
    usage["tokens_per_minute"] = round(total / usage["elapsed_seconds"] * 60)
    usage["tokens_per_product"] = round(total / max(len(PRODUCTS), 1))
    write_json(root / "schema" / "usage.json", usage)
    write_schema(root, schema)
    write_json(root / "schema" / "ranges.json", ranges)
    return {
        "world": str(root),
        "categories": len(categories),
        "features": len(features),
        "variables": len(variables),
        "rules": [rule.id for rule in rules],
        "usage": usage,
        "unassigned": sum(1 for v in variables if v.value is None),
        "text_mode": config.text_mode,
    }


def run_assign(root: Path, config: WorldConfig) -> dict:
    """Stage A2: assign values, impose family constraints and inject the traps."""
    apply_scale(config)
    from dataclasses import asdict

    from tau3.worldgen.slice import FAMILIES, PRODUCTS
    from tau3.worldgen.solver import solve
    from tau3.worldgen.world import load_schema, write_schema

    root = Path(root)
    schema = load_schema(root)
    raw = read_json(root / "schema" / "ranges.json")
    ranges = {key: (float(low), float(high)) for key, (low, high) in raw.items()}

    assignment = solve(schema, FAMILIES, ranges, config.seed, PRODUCTS)
    write_schema(root, schema)
    # The resolved constraints, including any derived discriminator, are what the
    # linters and Stage F must use -- the surface constraints alone no longer
    # single out the answer in a trap family.
    write_json(
        root / "plan" / "families.json",
        [asdict(family) for family in assignment.families],
    )
    unassigned = [v.id for v in schema.variables if v.value is None]
    if unassigned:
        # A document cannot state a value that does not exist, so an unassigned
        # variable is a broken world rather than a warning to carry forward.
        raise ValueError(f"Solver left variables unassigned: {unassigned}")
    # A family constraint outranks the generator's suggested range, but a value
    # that left the range is worth seeing: a narrow suggestion can make a
    # perfectly ordinary figure look invented.
    excursions = []
    for variable in schema.variables:
        bounds = ranges.get(variable.id)
        numeric = {"currency", "percent", "int_days", "int_count"}
        if bounds is None or variable.derived or variable.type not in numeric:
            continue
        low, high = bounds
        if high > low and not low <= float(variable.value) <= high:
            excursions.append(
                {"variable": variable.id, "value": variable.value, "range": [low, high]}
            )
    return {
        "world": str(root),
        "seed": assignment.seed,
        "range_excursions": excursions,
        "variables": len(schema.variables),
        "derived": sum(1 for v in schema.variables if v.derived),
        "traps": assignment.traps,
        "relaxations": assignment.relaxations,
    }


def run_map_tools(root: Path, config: WorldConfig) -> dict:
    """Stage B: bind every discoverable tool to the document that will name it."""
    import hashlib
    import inspect

    from tau3.worldgen.runtime import discoverable_tool_names
    from tau3.worldgen.world import read_world_manifest

    root = Path(root)
    manifest = read_world_manifest(root)
    by_official = {official: alias for alias, official in manifest.alias_map.items()}

    from tau3.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools

    bindings = []
    for official in sorted(discoverable_tool_names()):
        owner = (
            KnowledgeTools if hasattr(KnowledgeTools, official) else KnowledgeUserTools
        )
        method = getattr(owner, official)
        # Roughly three in five signatures belong in a numbered procedure and the
        # rest in a dedicated tool document, matching the official corpus.
        digest_value = hashlib.sha256(f"{config.suffix_salt}|{official}".encode())
        archetype = (
            "internal_protocol"
            if int(digest_value.hexdigest(), 16) % 5 < 3
            else "tool_doc"
        )
        bindings.append(
            {
                "official_name": official,
                "alias": by_official.get(official, official),
                "owner": "user" if owner is KnowledgeUserTools else "agent",
                "signature": str(inspect.signature(method)),
                "arguments": [
                    name
                    for name in inspect.signature(method).parameters
                    if name != "self"
                ],
                "document_archetype": archetype,
            }
        )
    write_json(root / "registry" / "tools.json", bindings)
    return {
        "world": str(root),
        "tools": len(bindings),
        "aliased": sum(1 for b in bindings if b["alias"] != b["official_name"]),
        "internal_protocol": sum(
            1 for b in bindings if b["document_archetype"] == "internal_protocol"
        ),
        "tool_doc": sum(1 for b in bindings if b["document_archetype"] == "tool_doc"),
    }


def run_plan_docs(root: Path, config: WorldConfig) -> dict:
    """Stage C: allocate variables, rules and tool signatures to documents."""
    apply_scale(config)
    from dataclasses import asdict

    from tau3.worldgen.allocate import allocate
    from tau3.worldgen.models import WorldPlan
    from tau3.worldgen.world import load_plan, load_schema, write_plan

    root = Path(root)
    schema = load_schema(root)
    tools = read_json(root / "registry" / "tools.json")
    families = read_json(root / "plan" / "families.json")

    # What the anti-aggregation bound protects is exactly what the tasks will
    # need, computed by the same function Stage F uses so the two cannot drift.
    # Protecting only the family constraints left one internal document holding
    # most of an ordering task, which L5 then rejected after rendering.
    from tau3.worldgen.tasks_gen import prospective_requirements

    protected = prospective_requirements(schema, families)

    targets = read_json(root / TARGETS)
    corpus_mean = targets["document_tokens"]["mean"]
    # Models write a little longer than they are asked to. Measure the overshoot
    # from whatever this world last rendered and ask for correspondingly less,
    # rather than carrying a hand-tuned constant that is only right for one model.
    overshoot = _measured_overshoot(root)
    if overshoot:
        # Damped: applying the whole measured correction each round makes the
        # loop oscillate, because how far a model overshoots varies per document
        # and the next render moves the measurement again. Half the correction
        # converges instead of ringing.
        corpus_mean = corpus_mean / (1 + (overshoot - 1) * 0.5)
    documents, report = allocate(
        schema,
        tools,
        config.num_documents,
        config.suffix_salt,
        protected,
        config.leak_divisor,
        corpus_mean,
    )
    try:
        existing = load_plan(root)
        tasks = existing.tasks
    except FileNotFoundError:
        tasks = []
    write_plan(root, WorldPlan(documents=documents, tasks=tasks))
    return {"world": str(root), "length_overshoot": overshoot, **asdict(report)}


def _measured_overshoot(root: Path) -> float | None:
    """How much longer the last render came out than it was asked for."""
    from tau3.worldgen.world import load_plan

    documents_dir = root / "documents"
    if not documents_dir.exists():
        return None
    try:
        previous = {d.doc_id: d for d in load_plan(root).documents}
    except FileNotFoundError:
        return None
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(encoding.encode(text))
    except ImportError:
        return None

    asked = realized = 0
    for path in documents_dir.glob("*.json"):
        plan = previous.get(path.stem)
        if plan is None:
            continue
        document = read_json(path)
        realized += count(f"{document['title']}\n{document['content']}")
        asked += plan.target_tokens
    if not asked or not realized:
        return None
    return realized / asked


def run_render(root: Path, config: WorldConfig, resume: bool = True) -> dict:
    """Stage D: draft every document with placeholders, then fill in the values."""
    from concurrent.futures import ThreadPoolExecutor

    from tau3.worldgen.render import HallucinatedValue, draft, fill, style_for
    from tau3.worldgen.world import load_plan, load_schema, read_world_manifest

    root = Path(root)
    schema = load_schema(root)
    plan = load_plan(root)
    manifest = read_world_manifest(root)
    aliases = {official: alias for alias, official in manifest.alias_map.items()}
    tools = {t["official_name"]: t for t in read_json(root / "registry" / "tools.json")}
    titles = {d.doc_id: d.title for d in plan.documents}

    render_dir = root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    models_used: dict[str, int] = {}
    attempts_used: dict[str, int] = {}
    styles: dict[str, str] = {}
    # Throughput is endpoint-bound rather than concurrency-bound, so what a run
    # actually consumed is worth recording rather than re-measuring later.
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    started = time.monotonic()

    def one(document):
        template_path = render_dir / f"{document.doc_id}.md"
        style = style_for(document, config.render_models, config.suffix_salt)
        styles[document.doc_id] = style.model
        fingerprint = digest(
            [
                document.model_dump(mode="json"),
                style.model,
                style.persona,
                style.register,
            ]
        )
        stamp_path = render_dir / f"{document.doc_id}.brief"
        cached = (
            resume
            and template_path.exists()
            and stamp_path.exists()
            and stamp_path.read_text() == fingerprint
        )
        if cached:
            template = template_path.read_text()
            info = {"model": "cached", "attempts": 0}
        else:
            try:
                template, info = draft(document, schema, tools, titles, style, config)
            except (HallucinatedValue, Exception) as exc:  # noqa: BLE001
                failures.append(f"{document.doc_id}: {type(exc).__name__}: {exc}")
                return
            template_path.write_text(template)
            stamp_path.write_text(fingerprint)
        models_used[info["model"]] = models_used.get(info["model"], 0) + 1
        attempts_used[document.doc_id] = info["attempts"]
        if info.get("attempts"):
            usage["prompt_tokens"] += info.get("prompt_tokens", 0)
            usage["completion_tokens"] += info.get("completion_tokens", 0)
            usage["calls"] += info["attempts"]
        write_json(
            root / "documents" / f"{document.doc_id}.json",
            {
                "id": document.doc_id,
                "title": document.title,
                "content": fill(template, document, schema, aliases),
            },
        )

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(one, plan.documents))

    # Record what each document was written from, so a later change can tell a
    # value that moved from a brief that changed.
    from tau3.worldgen.refresh import plan_hashes

    manifest = read_world_manifest(root)
    manifest.stages["template_hashes"] = "render"
    write_json(
        root / "render" / "hashes.json",
        plan_hashes(plan, styles, config.seed),
    )

    rendered = len(list((root / "documents").glob("*.json")))
    elapsed = max(time.monotonic() - started, 1e-6)
    total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    usage["elapsed_seconds"] = round(elapsed, 1)
    usage["tokens_per_minute"] = round(total_tokens / elapsed * 60)
    usage["documents_per_minute"] = round(rendered / elapsed * 60, 1)
    write_json(root / "render" / "usage.json", usage)
    return {
        "world": str(root),
        "planned": len(plan.documents),
        "rendered": rendered,
        "models": models_used,
        "usage": usage,
        "retried": sum(1 for v in attempts_used.values() if v > 1),
        "failures": failures,
        "text_mode": config.text_mode,
    }


def run_build_tasks(root: Path, config: WorldConfig) -> dict:
    """Stage F: write the task set and the database state it assumes."""
    apply_scale(config)
    from tau3.worldgen.models import WorldPlan
    from tau3.worldgen.tasks_gen import build_tasks, gold_statistics
    from tau3.worldgen.world import (
        load_plan,
        load_schema,
        read_world_manifest,
        write_plan,
    )

    root = Path(root)
    schema = load_schema(root)
    plan = load_plan(root)
    manifest = read_world_manifest(root)
    alias = {official: name for name, official in manifest.alias_map.items()}
    families = read_json(root / "plan" / "families.json")

    generated, db = build_tasks(
        schema,
        plan.documents,
        families,
        alias,
        config.num_tasks,
        config.seed,
        config.num_noise_users,
    )

    tasks_dir = root / "tasks"
    for stale in tasks_dir.glob("task_*.json"):
        stale.unlink()
    for item in generated:
        write_json(tasks_dir / f"{item.task['id']}.json", item.task)
    (root / "db.json").write_text(
        json.dumps(db.model_dump(mode="json"), indent=1) + "\n"
    )
    write_plan(
        root, WorldPlan(documents=plan.documents, tasks=[g.spec for g in generated])
    )

    targets = read_json(root / TARGETS)
    statistics = gold_statistics(generated)
    return {
        "world": str(root),
        "tasks": len(generated),
        "users": len(db.users.data),
        "accounts": len(db.accounts.data),
        **statistics,
        "corpus_mean_gold_documents": targets["gold_documents"]["mean"],
    }


def run_accept(
    root: Path,
    config: WorldConfig,
    gate: str = "all",
    reference_agent: str | None = None,
    control_sample: int = 10,
    trials: int = 1,
    max_steps: int = 200,
) -> dict:
    """Stage H: compare this world with the official corpus, layer by layer."""
    from tau3.worldgen.accept import (
        compare_behaviour,
        compare_retrieval,
        compare_text,
        official_control_sample,
    )

    root = Path(root)
    reports = []
    if gate in {"all", "text"}:
        reports.append(compare_text(root).as_dict())
    if gate in {"all", "retrieval"}:
        reports.append(compare_retrieval(root).as_dict())

    behaviour = None
    if gate in {"all", "behaviour"}:
        agent = reference_agent or config.render_models[0]
        control = official_control_sample(control_sample, config.seed)
        runs = _behaviour_runs(
            root, agent, control, trials, max_steps, config.concurrency
        )
        behaviour = compare_behaviour(runs["synthetic"], runs["official"]).as_dict()
        behaviour["reference_agent"] = agent
        behaviour["official_control_tasks"] = control
        reports.append(behaviour)

    payload = {
        "world": str(root),
        "gates": reports,
        # Only blocking layers decide publication; the text layer is advisory.
        "passed": all(r["passed"] for r in reports),
    }
    write_json(root / "accept" / f"{gate}.json", payload)
    return payload


def _behaviour_runs(
    root: Path,
    agent: str,
    control: list[str],
    trials: int,
    max_steps: int,
    concurrency: int = 8,
) -> dict[str, Path]:
    """Run the same agent on this world and on the official control sample."""
    import os
    import subprocess

    from tau3.worldgen.world import DRAFT_ENV_VAR, WORLD_ENV_VAR

    output = Path(root) / "accept" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    plans = {
        "synthetic": ["--domain", "banking_synth"],
        "official": ["--domain", "banking_knowledge", "--task-ids", *control],
    }
    produced = {}
    for name, arguments in plans.items():
        destination = output / name
        results = destination / "results.json"
        if results.exists():
            # Running twenty conversations is the most expensive thing this
            # pipeline does; a finished run is reused rather than repeated.
            produced[name] = results
            continue
        # A world is a draft until these gates pass, and the gates are what
        # decides whether it may be published, so this is the one place a draft
        # is meant to be loaded.
        environment = {
            **os.environ,
            WORLD_ENV_VAR: str(root),
            DRAFT_ENV_VAR: "1",
        }
        completed = subprocess.run(
            [
                "tau3",
                "run",
                *arguments,
                "--agent-llm",
                agent,
                "--user-llm",
                agent,
                "--num-trials",
                str(trials),
                "--max-steps",
                str(max_steps),
                "--max-concurrency",
                str(concurrency),
                "--save-to",
                str(destination),
            ],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            # The output of a failed batch is where the reason is; swallowing it
            # leaves a CalledProcessError that says only that something exited 1.
            tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"The {name} behaviour run failed (exit {completed.returncode}):\n{tail}"
            )
        produced[name] = destination / "results.json"
    return produced


def run_refresh(root: Path, config: WorldConfig, variables: list[str]) -> dict:
    """Stage I: re-state changed values without regenerating any prose."""
    from tau3.worldgen.refresh import assess, refill
    from tau3.worldgen.world import load_plan, load_schema, read_world_manifest

    root = Path(root)
    apply_scale(config)
    schema = load_schema(root)
    plan = load_plan(root)
    manifest = read_world_manifest(root)
    aliases = {official: alias for alias, official in manifest.alias_map.items()}

    hashes_path = root / "render" / "hashes.json"
    recorded = read_json(hashes_path) if hashes_path.exists() else None
    styles = {}
    if recorded is not None:
        from tau3.worldgen.render import style_for

        styles = {
            d.doc_id: style_for(d, config.render_models, config.suffix_salt).model
            for d in plan.documents
        }

    from tau3.worldgen.world import write_world_manifest

    impact = assess(plan, schema, variables, recorded, styles, config.seed)
    touched = [d for d in plan.documents if d.doc_id in set(impact.documents)]
    # Only documents whose brief still holds can be refilled; the rest need the
    # renderer, and saying so is the point of the assessment.
    refillable = [d for d in touched if d.doc_id not in set(impact.needs_model)]
    written = refill(root, schema, refillable, aliases)

    republish = False
    if written and manifest.status == "published":
        # The artifacts a published world was hashed against have changed, so it
        # goes back to draft rather than staying published against documents it
        # no longer describes. Verification is what earns publication back.
        manifest.status = "draft"
        manifest.artifact_hashes = {}
        manifest.stages["refresh"] = f"refilled {len(written)} documents"
        write_world_manifest(root, manifest)
        republish = True

    return {
        "world": str(root),
        **impact.as_dict(),
        "refilled": written,
        "model_calls": 0,
        "returned_to_draft": republish,
        "note": (
            "Documents listed under needs_model were written from a brief that "
            "has changed; rerun render for those. Refilled documents need "
            "verify and publish again before the world can be loaded."
        ),
    }
