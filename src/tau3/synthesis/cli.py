"""Argparse entry points for the independently resumable synthesis stages."""

import json
from pathlib import Path


def add_synthesis_parser(subparsers):
    """Register tau3 synthesize without loading models or making API calls."""
    parser = subparsers.add_parser(
        "synthesize", help="Build and validate tau3-AA training task bundles"
    )
    stages = parser.add_subparsers(dest="synthesis_stage", required=True)
    from tau3.synthesis.targeted.cli import add_parser

    add_parser(stages)
    catalog = stages.add_parser(
        "build-catalog", help="Extract evidence-backed pilot rules"
    )
    catalog.add_argument("--config")
    catalog.add_argument("--output", default="data/synthetic/tau3-aa/catalog.json")
    generation = stages.add_parser(
        "generate", help="Generate candidate tasks (not yet published)"
    )
    generation.add_argument("--config")
    generation.add_argument("--num-tasks", type=int, default=200)
    generation.add_argument("--output", required=True)
    generation.add_argument("--resume", action="store_true")
    generation.add_argument(
        "--offline",
        action="store_true",
        help="Developer-only deterministic text; cannot publish these tasks",
    )
    validation = stages.add_parser(
        "validate", help="Strict replay and bm25_grep dialogue acceptance"
    )
    validation.add_argument("--bundle", required=True)
    validation.add_argument("--resume", action="store_true")
    validation.add_argument(
        "--offline",
        action="store_true",
        help="Run local gates only; cannot admit tasks",
    )
    export = stages.add_parser("export", help="Publish tasks that passed every gate")
    export.add_argument("--bundle", required=True)
    sft = stages.add_parser(
        "collect-sft", help="Sample reviewed bm25_grep teacher trajectories"
    )
    sft.add_argument("--bundle", required=True)
    sft.add_argument("--split", choices=["train"], default="train")
    repair = stages.add_parser(
        "repair-bundle",
        help="Fork an old draft with fixed tools and strict structured generation",
    )
    repair.add_argument("--source", required=True)
    repair.add_argument("--output", required=True)
    repair.add_argument("--config", required=True)
    world = stages.add_parser(
        "world-tasks",
        help="Generate independently validated task expressions from admitted V2 seeds",
    )
    for name in ("world", "readiness", "config", "output"):
        world.add_argument(f"--{name}", required=True)
    world.add_argument("--num-tasks", type=int, required=True)
    world.add_argument("--max-calls", type=int, required=True)
    world.add_argument("--resume", action="store_true")
    world.add_argument("--local-world", help="Optional identical local seed mirror")
    world.add_argument(
        "--pilot-bundle", help="Required published 20-task pilot for larger batches"
    )
    parser.set_defaults(func=run_synthesis)


def run_synthesis(args):
    """Dispatch one stage; errors leave atomic checkpoints available for resumption."""
    if args.synthesis_stage == "targeted":
        from tau3.synthesis.targeted.cli import run

        return run(args)
    from tau3.synthesis.catalog import build_catalog
    from tau3.synthesis.storage import write_json
    from tau3.synthesis.workflow import (
        collect_sft,
        export_bundle,
        fork_repaired_bundle,
        generate_bundle,
        read_config,
        validate_bundle,
    )

    stage = args.synthesis_stage
    if stage == "world-tasks":
        from tau3.synthesis.world_tasks import synthesize_world_tasks

        result = synthesize_world_tasks(
            Path(args.world),
            Path(args.readiness),
            Path(args.output),
            args.num_tasks,
            Path(args.config),
            args.max_calls,
            args.resume,
            Path(args.local_world) if args.local_world else None,
            Path(args.pilot_bundle) if args.pilot_bundle else None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    if stage == "build-catalog":
        read_config(args.config)
        catalog = build_catalog()
        write_json(args.output, catalog.model_dump(mode="json"))
        result = {
            "catalog": args.output,
            "evidence_documents": len(catalog.documents),
            "exclusions": catalog.exclusions,
        }
    elif stage == "repair-bundle":
        result = fork_repaired_bundle(
            Path(args.source), Path(args.output), read_config(args.config)
        )
    elif stage == "generate":
        config = read_config(args.config)
        if args.offline:
            config.text_mode = "template"
        result = generate_bundle(Path(args.output), args.num_tasks, config, args.resume)
    elif stage == "validate":
        result = validate_bundle(Path(args.bundle), args.resume, args.offline)
    elif stage == "export":
        result = export_bundle(Path(args.bundle))
    else:
        result = collect_sft(Path(args.bundle), args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
