"""Command-line stages for one immutable failure-directed experiment."""

import json
from pathlib import Path

from tau3.synthesis.targeted.models import RoundConfig
from tau3.utils.llm_concurrency import file_lock
from tau3.worldgen.v2.audit import AuditIncomplete


def add_parser(stages):
    """Register isolated stages without making any model requests."""
    parser = stages.add_parser(
        "targeted", help="Failure-directed V2 business curricula and fixed-four SFT"
    )
    commands = parser.add_subparsers(dest="targeted_stage", required=True)
    for name in ("analyze", "plan", "pilot", "generate", "collect", "export", "report", "accept-evaluation"):
        command = commands.add_parser(name)
        command.add_argument("--round-dir", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--stage", dest="native_stage", choices=["pilot", "small", "full", "expand800", "expand1200", "expand1600", "expand2000", "expand2400", "expand2800", "expand3000"], default="small")
        if name == "generate":
            command.add_argument("--split", choices=["train", "validation"], default="train")
        if name == "accept-evaluation":
            command.add_argument("--receipt", type=Path, required=True)
        if name == "analyze":
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--additional-report", type=Path, action="append", default=[])
            command.add_argument("--run", type=Path)
            command.add_argument("--round-id", required=True)
            command.add_argument("--base-model", required=True)
        if name == "plan":
            command.add_argument("--previous-plan", type=Path)
            command.add_argument("--parent-round", type=Path)
        if name in {"collect", "export", "report"}:
            command.add_argument(
                "--phase", choices=["pilot", "formal"], default="formal"
            )
    return parser


def run(args):
    """Hold a round writer lock and return nonzero on every unfinished stage."""
    import yaml

    from tau3.synthesis.targeted import planning, workflow
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest

    root = args.round_dir.resolve()
    raw = yaml.safe_load(args.config.read_text()) or {}
    if raw.get("schema_version") == 2:
        from tau3.synthesis.targeted.native.cli import run_native

        return run_native(args, raw)
    config = RoundConfig.model_validate(raw)
    if args.targeted_stage == "accept-evaluation":
        raise ValueError("External evaluation gates require a schema v2 native round")
    stage = args.targeted_stage
    with file_lock(root / "round.lock"):
        workflow.initialize(root, config)
        phase_suffix = "-" + args.phase if hasattr(args, "phase") else ""
        status = root / "stages" / f"{stage}{phase_suffix}.json"
        if status.exists() and not args.resume and stage != "report":
            raise ValueError(
                "Stage already started; use --resume with identical inputs"
            )
        identity = {
            "stage": stage,
            "config": digest(config.model_dump()),
            "phase": getattr(args, "phase", None),
        }
        if stage == "analyze":
            identity.update(
                report=digest(args.report.read_text()),
                round_id=args.round_id,
                base_model=args.base_model,
                run=str(args.run.resolve()) if args.run else None,
            )
        if stage == "plan":
            identity["previous_plan"] = (
                digest(args.previous_plan.read_text()) if args.previous_plan else None
            )
        if status.exists() and json.loads(status.read_text())["identity"] != identity:
            raise ValueError("Stage inputs changed")
        write_json(status, {"identity": identity, "status": "RUNNING"})
        try:
            if stage == "analyze":
                if args.resume and (root / "profile.json").exists():
                    result = json.loads((root / "profile.json").read_text())
                else:
                    result = planning.analyze(
                        root,
                        config,
                        args.report,
                        args.round_id,
                        args.base_model,
                        args.run,
                    ).model_dump(mode="json")
            elif stage == "plan":
                if args.resume and (root / "plan.json").exists():
                    result = workflow.load_plan(root).model_dump(mode="json")
                else:
                    result = planning.make_plan(
                        root, config, args.previous_plan
                    ).model_dump(mode="json")
            elif stage == "pilot":
                workflow.generation(root, config, "pilot")
                workflow.collection(root, config, "pilot")
                result = workflow.export(root, config, "pilot")
            elif stage == "generate":
                result = workflow.generation(root, config)
            elif stage == "collect":
                result = workflow.collection(root, config, args.phase)
            elif stage == "export":
                result = workflow.export(root, config, args.phase)
            else:
                result = workflow.report(root, config, args.phase)
            complete = result.get("status", "COMPLETE") == "COMPLETE"
            if stage == "collect":
                complete = (
                    result["four_trial_complete_tasks"]
                    == result["expected_tasks"]
                    == result["valid_tasks"]
                )
            write_json(
                status,
                {
                    "identity": identity,
                    "status": "COMPLETE" if complete else "INCOMPLETE",
                },
            )
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "status": "COMPLETE" if complete else "INCOMPLETE",
                        "round_dir": str(root),
                    },
                    ensure_ascii=False,
                )
            )
            if not complete:
                raise SystemExit(2)
            return result
        except Exception as exc:
            write_json(
                status,
                {"identity": identity, "status": "INCONCLUSIVE", "reason": str(exc)},
            )
            if isinstance(exc, AuditIncomplete):
                print(json.dumps({"status": "INCONCLUSIVE", "reason": str(exc)}))
                raise SystemExit(2) from exc
            raise
