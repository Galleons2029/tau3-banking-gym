"""CLI for portable category packages and certified V2 world bundles."""

import json
from pathlib import Path

from tau3.worldgen.v2.pipeline import build, publish, validate_world, write_json
from tau3.worldgen.v2.specs import WorldSpec, load_categories


def add_parsers(stages):
    """Register V2 alongside legacy inspection and development stages."""
    scaffold = stages.add_parser(
        "scaffold-category", help="Write a declarative category package"
    )
    scaffold.add_argument("--id", required=True)
    scaffold.add_argument("--profile", required=True)
    scaffold.add_argument("--output", required=True)
    scaffold.set_defaults(func=run)
    expansion = stages.add_parser(
        "expand-v2", help="Validate category additions in a new world"
    )
    expansion.add_argument("--source", required=True)
    expansion.add_argument("--world", required=True)
    expansion.add_argument("--category", action="append", required=True)
    expansion.add_argument("--allow-structural-duplicates", action="store_true")
    expansion.add_argument("--review-model", action="append", default=[])
    expansion.add_argument("--llm-config")
    expansion.add_argument("--readiness")
    expansion.add_argument("--foundation-only", action="store_true")
    expansion.add_argument("--verification-config")
    expansion.set_defaults(func=run)
    proposal = stages.add_parser(
        "propose-category", help="Automatically propose and validate a novel category"
    )
    proposal.add_argument("--source", required=True)
    proposal.add_argument("--output", required=True)
    proposal.add_argument("--concept", required=True)
    proposal.add_argument("--model", required=True)
    proposal.add_argument("--review-model", action="append", required=True)
    proposal.add_argument("--llm-config")
    proposal.add_argument("--max-attempts", type=int, default=3)
    proposal.add_argument("--readiness")
    proposal.add_argument("--foundation-only", action="store_true")
    proposal.add_argument("--verification-config")
    proposal.set_defaults(func=run)
    qualification = stages.add_parser(
        "qualify-v2", help="Check matched complete behavioural experiments"
    )
    qualification.add_argument("--world", required=True)
    qualification.add_argument("--synthetic", action="append", required=True)
    qualification.add_argument("--control", action="append", required=True)
    qualification.add_argument("--max-gap", type=float, default=0.1)
    qualification.add_argument("--min-tasks", type=int, default=50)
    qualification.set_defaults(func=run)
    rollout = stages.add_parser(
        "rollout-v2", help="Run explicitly configured Gemini/GLM rollout matrix"
    )
    rollout.add_argument("--world", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--retrieval", default="bm25")
    rollout.add_argument("--num-tasks", type=int, default=5)
    rollout.add_argument("--num-trials", type=int, default=1)
    rollout.add_argument("--max-steps", type=int)
    rollout.add_argument("--config")
    rollout.add_argument("--task-id", action="append")
    rollout.add_argument("--all-tasks", action="store_true")
    rollout.set_defaults(func=run)
    for name in ("blind-v2", "calibrate-v2", "matrix-v2", "readiness-v2"):
        audit = stages.add_parser(
            name, help="Bounded independent validation and admission"
        )
        audit.add_argument("--world", required=True)
        audit.add_argument("--output", required=True)
        audit.add_argument("--config")
        if name in {"blind-v2", "calibrate-v2"}:
            audit.add_argument("--max-calls", type=int, default=1000)
        if name in {"blind-v2", "matrix-v2"}:
            audit.add_argument("--task-id", action="append")
        if name == "matrix-v2":
            audit.add_argument("--num-trials", type=int, default=1)
            audit.add_argument("--max-steps", type=int)
            audit.add_argument("--max-rollouts", type=int, default=500)
        if name == "readiness-v2":
            for evidence in ("blind", "calibration", "online"):
                audit.add_argument(f"--{evidence}", required=True)
        audit.set_defaults(func=run)
    builder = stages.add_parser(
        "build-v2", help="Build and validate a resumable V2 world"
    )
    builder.add_argument("--world", required=True)
    builder.add_argument("--category", action="append", required=True)
    builder.add_argument("--seed", type=int, default=42)
    builder.add_argument("--clock", default="2025-11-14")
    builder.add_argument(
        "--purpose",
        choices=["banking_expansion", "paper_reproduction", "training_curriculum"],
        default="banking_expansion",
    )
    builder.add_argument(
        "--text-mode", choices=["controlled", "llm"], default="controlled"
    )
    builder.add_argument("--render-model")
    builder.add_argument("--review-model", action="append", default=[])
    builder.add_argument(
        "--llm-config",
        help="Local JSON/YAML request settings; credentials are never exported",
    )
    builder.add_argument("--max-repairs", type=int, default=3)
    builder.add_argument("--max-model-calls", type=int, default=10000)
    builder.set_defaults(func=run)
    for command, help_text in [
        ("validate-v2", "Revalidate exact persisted artifacts"),
        ("publish-v2", "Publish only a current passing certificate"),
    ]:
        parser = stages.add_parser(command, help=help_text)
        parser.add_argument("--world", required=True)
        parser.set_defaults(func=run)


def run(args):
    """Run a V2 command and print an explicit PASS/FAIL/INCONCLUSIVE result."""
    if args.worldgen_stage in {"blind-v2", "calibrate-v2", "matrix-v2", "readiness-v2"}:
        from tau3.worldgen.v2.blind import run_blind
        from tau3.worldgen.v2.calibration import run_calibration
        from tau3.worldgen.v2.readiness import assess
        from tau3.worldgen.v2.rollouts import run_matrix

        root, output = Path(args.world), Path(args.output)
        settings = Path(args.config) if args.config else None
        if args.worldgen_stage == "blind-v2":
            result = run_blind(root, output, args.task_id, settings, args.max_calls)
        elif args.worldgen_stage == "calibrate-v2":
            result = run_calibration(root, output, settings, args.max_calls)
        elif args.worldgen_stage == "matrix-v2":
            result = run_matrix(
                root,
                output,
                args.task_id,
                args.num_trials,
                args.max_steps,
                settings,
                args.max_rollouts,
            )
        else:
            result = assess(
                root,
                Path(args.blind),
                Path(args.calibration),
                Path(args.online),
                output,
                settings,
            )
    elif args.worldgen_stage == "scaffold-category":
        from tau3.worldgen.v2.catalog import scaffold

        path = Path(args.output)
        if path.exists():
            raise ValueError("Refusing to overwrite existing category")
        package = scaffold(args.id, args.profile)
        write_json(path, package.model_dump(mode="json"))
        result = {"status": "PASS", "category": str(path)}
    elif args.worldgen_stage == "rollout-v2":
        from tau3.worldgen.v2.rollouts import run_rollouts

        result = run_rollouts(
            Path(args.world),
            Path(args.output),
            args.retrieval,
            args.num_tasks,
            args.num_trials,
            args.max_steps,
            Path(args.config) if args.config else None,
            task_ids=args.task_id,
            all_tasks=args.all_tasks,
        )
    elif args.worldgen_stage == "qualify-v2":
        from tau3.worldgen.v2.behaviour import qualify

        result = qualify(
            Path(args.world),
            [Path(p) for p in args.synthetic],
            [Path(p) for p in args.control],
            args.max_gap,
            args.min_tasks,
        )
    elif args.worldgen_stage == "expand-v2":
        import yaml

        from tau3.worldgen.v2.expansion import expand

        llm_args = (
            yaml.safe_load(Path(args.llm_config).read_text()) if args.llm_config else {}
        )
        result = expand(
            Path(args.source),
            Path(args.world),
            [Path(p) for p in args.category],
            require_novel=not args.allow_structural_duplicates,
            review_models=args.review_model,
            llm_args=llm_args,
            readiness_path=Path(args.readiness) if args.readiness else None,
            foundation_only=args.foundation_only,
            verification_settings=Path(args.verification_config)
            if args.verification_config
            else None,
        )
    elif args.worldgen_stage == "propose-category":
        import yaml

        from tau3.worldgen.v2.expansion import propose

        llm_args = (
            yaml.safe_load(Path(args.llm_config).read_text()) if args.llm_config else {}
        )
        result = propose(
            Path(args.source),
            Path(args.output),
            args.concept,
            args.model,
            llm_args,
            args.max_attempts,
            args.review_model,
            readiness_path=Path(args.readiness) if args.readiness else None,
            foundation_only=args.foundation_only,
            verification_settings=Path(args.verification_config)
            if args.verification_config
            else None,
        )
    elif args.worldgen_stage == "build-v2":
        import yaml

        llm_args = (
            yaml.safe_load(Path(args.llm_config).read_text()) if args.llm_config else {}
        )
        spec = WorldSpec(
            seed=args.seed,
            clock=args.clock,
            purpose=args.purpose,
            categories=load_categories([Path(p) for p in args.category]),
        )
        result = build(
            Path(args.world),
            spec,
            args.text_mode,
            args.render_model,
            args.review_model,
            llm_args,
            args.max_repairs,
            args.max_model_calls,
        )
    elif args.worldgen_stage == "validate-v2":
        result = validate_world(Path(args.world))
    else:
        result = publish(Path(args.world))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") in {"FAIL", "INCONCLUSIVE"}:
        raise SystemExit(2)
    return result
