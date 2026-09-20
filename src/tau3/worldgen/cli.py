"""Argparse entry points for the independently resumable worldgen stages."""

import json
from pathlib import Path


def add_worldgen_parser(subparsers):
    """Register tau3 worldgen without loading models or making API calls."""
    parser = subparsers.add_parser(
        "worldgen", help="Synthesize a same-distribution banking world"
    )
    stages = parser.add_subparsers(dest="worldgen_stage", required=True)

    from tau3.worldgen.v2.cli import add_parsers

    add_parsers(stages)

    calibrate = stages.add_parser(
        "calibrate", help="Measure the installed corpus into acceptance targets"
    )
    calibrate.add_argument("--output", default="data/synthetic/tau3-world/targets.json")

    init = stages.add_parser(
        "init-world", help="Create a draft world bundle with a fresh tool alias map"
    )
    init.add_argument("--world", required=True)
    init.add_argument("--name", default=None, help="Defaults to the bundle directory")
    init.add_argument("--config")

    seed = stages.add_parser(
        "seed-world", help="Write the hand-written world that settles the schema"
    )
    seed.add_argument("--world", required=True)
    seed.add_argument("--name", default="m1-seed")
    seed.add_argument("--config")

    build = stages.add_parser(
        "build-schema", help="Generate the product space's shape (no values)"
    )
    build.add_argument("--world", required=True)
    build.add_argument("--config")
    build.add_argument(
        "--offline",
        action="store_true",
        help="Developer-only deterministic skeletons; cannot publish these",
    )

    assign = stages.add_parser(
        "assign", help="Assign values, impose constraints and inject traps"
    )
    assign.add_argument("--world", required=True)
    assign.add_argument("--config")

    maptools = stages.add_parser(
        "map-tools", help="Bind discoverable tools to the documents that name them"
    )
    maptools.add_argument("--world", required=True)
    maptools.add_argument("--config")

    plandocs = stages.add_parser(
        "plan-docs", help="Allocate variables and tool signatures to documents"
    )
    plandocs.add_argument("--world", required=True)
    plandocs.add_argument("--config")

    render = stages.add_parser(
        "render", help="Draft documents with placeholders and fill in the values"
    )
    render.add_argument("--world", required=True)
    render.add_argument("--config")
    render.add_argument("--no-resume", action="store_true")
    render.add_argument(
        "--offline",
        action="store_true",
        help="Developer-only deterministic text; cannot publish these",
    )

    buildtasks = stages.add_parser(
        "build-tasks", help="Write the task set and the database state it assumes"
    )
    buildtasks.add_argument("--world", required=True)
    buildtasks.add_argument("--config")

    lint = stages.add_parser("lint", help="Run the structural linters (L4/L5/L6)")
    lint.add_argument("--world", required=True)
    lint.add_argument("--config")

    verify = stages.add_parser(
        "verify", help="Replay reference actions and check gold minimality"
    )
    verify.add_argument("--world", required=True)

    accept = stages.add_parser(
        "accept", help="Compare this world with the official corpus"
    )
    accept.add_argument("--world", required=True)
    accept.add_argument("--config")
    accept.add_argument(
        "--gate", choices=["all", "text", "retrieval", "behaviour"], default="all"
    )
    accept.add_argument("--reference-agent")
    accept.add_argument("--control-sample", type=int, default=10)
    accept.add_argument("--trials", type=int, default=1)
    accept.add_argument("--max-steps", type=int, default=200)

    refresh = stages.add_parser(
        "refresh", help="Re-state changed values without regenerating prose"
    )
    refresh.add_argument("--world", required=True)
    refresh.add_argument("--config")
    refresh.add_argument(
        "--variable", action="append", default=[], help="Repeatable variable id"
    )

    publish = stages.add_parser(
        "publish", help="Hash artifacts and mark a validated world published"
    )
    publish.add_argument("--world", required=True)

    parser.set_defaults(func=run_worldgen)


def run_worldgen(args):
    """Dispatch one stage; errors leave atomic artifacts available for resumption."""
    from tau3.worldgen.workflow import (
        read_config,
        run_accept,
        run_assign,
        run_build_schema,
        run_build_tasks,
        run_calibrate,
        run_init_world,
        run_lint,
        run_map_tools,
        run_plan_docs,
        run_publish,
        run_refresh,
        run_render,
        run_seed_world,
        run_verify,
    )

    stage = args.worldgen_stage
    if stage == "calibrate":
        result = run_calibrate(Path(args.output))
    elif stage == "init-world":
        root = Path(args.world)
        result = run_init_world(root, args.name or root.name, read_config(args.config))
    elif stage == "seed-world":
        result = run_seed_world(Path(args.world), read_config(args.config), args.name)
    elif stage == "build-schema":
        config = read_config(args.config)
        if args.offline:
            config.text_mode = "template"
        result = run_build_schema(Path(args.world), config)
    elif stage == "assign":
        result = run_assign(Path(args.world), read_config(args.config))
    elif stage == "map-tools":
        result = run_map_tools(Path(args.world), read_config(args.config))
    elif stage == "plan-docs":
        result = run_plan_docs(Path(args.world), read_config(args.config))
    elif stage == "render":
        config = read_config(args.config)
        if args.offline:
            config.text_mode = "template"
        result = run_render(Path(args.world), config, not args.no_resume)
    elif stage == "build-tasks":
        result = run_build_tasks(Path(args.world), read_config(args.config))
    elif stage == "accept":
        result = run_accept(
            Path(args.world),
            read_config(args.config),
            args.gate,
            args.reference_agent,
            args.control_sample,
            args.trials,
            args.max_steps,
        )
    elif stage == "refresh":
        result = run_refresh(Path(args.world), read_config(args.config), args.variable)
    elif stage == "lint":
        result = run_lint(Path(args.world), read_config(args.config))
    elif stage == "verify":
        result = run_verify(Path(args.world))
    else:
        result = run_publish(Path(args.world))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
