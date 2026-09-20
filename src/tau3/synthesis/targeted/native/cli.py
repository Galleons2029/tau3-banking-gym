"""CLI dispatch preserves the v1 implementation and immutable native stage identities."""

import json

from tau3.synthesis.storage import digest, read_json, write_json
from tau3.synthesis.targeted.native import workflow
from tau3.synthesis.targeted.native.gate import accept
from tau3.synthesis.targeted.native.models import NativeConfig, NativeProfile
from tau3.synthesis.targeted.native.planning import compile_plan
from tau3.utils.llm_concurrency import file_lock


def run_native(args, raw):
    """Execute only the authorized stage; full is guarded by the external receipt."""
    config, root = NativeConfig.model_validate(raw), args.round_dir.resolve()
    stage = "pilot" if args.targeted_stage == "pilot" else args.native_stage
    command = args.targeted_stage
    with file_lock(root / "round.lock"):
        snapshot = workflow.initialize(root, config)
        workflow._worker_init()
        state = root / "stages" / f"{command}-{stage}-{getattr(args, 'split', 'train')}.json"
        identity = {"command": command, "stage": stage, "config_hash": digest(raw), "split": getattr(args, "split", "train")}
        if command == "analyze":
            identity.update(report=digest(args.report.read_text()), round_id=args.round_id, base_model=args.base_model)
            if getattr(args, "additional_report", None):
                identity["additional_reports"] = {str(p.resolve()): digest(p.read_text()) for p in args.additional_report}
            identity["run_hash"] = digest({p.name: digest(read_json(p)) for p in sorted((args.run / "simulations").glob("*.json"))}) if args.run else None
        if command == "plan":
            if getattr(args, "parent_round", None):
                if args.previous_plan:
                    raise ValueError("Use either --parent-round or --previous-plan")
                args.previous_plan = args.parent_round / "plan.json"
            elif config.parent_round and not args.previous_plan:
                from pathlib import Path

                args.previous_plan = Path(config.parent_round) / "plan.json"
            identity["previous_plan_hash"] = digest(read_json(args.previous_plan)) if args.previous_plan else None
        if command == "accept-evaluation":
            identity["receipt_hash"] = digest(read_json(args.receipt))
        if state.exists():
            if not args.resume and command != "report":
                raise ValueError("Stage already started; pass --resume")
            if read_json(state)["identity"] != identity:
                raise ValueError("Native stage inputs changed")
        write_json(state, {"identity": identity, "status": "RUNNING"})
        try:
            if command == "analyze":
                from tau3.synthesis.targeted.native.analysis import analyze

                if not (root / "profile.json").exists():
                    # The corrected report supplies causes; raw observations remain phenomena.
                    profile = analyze(root, config, args.report, args.round_id, args.base_model, args.run,
                                      additional_reports=getattr(args, "additional_report", []))
                    payload = profile.model_dump(mode="json")
                else:
                    payload = read_json(root / "profile.json")
                profile = NativeProfile.model_validate(payload)
                write_json(root / "profile.json", profile.model_dump(mode="json"))
                result = {"status": "COMPLETE"}
            elif command == "plan":
                if not (root / "plan.json").exists():
                    profile = NativeProfile.model_validate(read_json(root / "profile.json"))
                    previous = digest(read_json(args.previous_plan)) if args.previous_plan else None
                    plan = compile_plan(config, profile, snapshot["snapshot_hash"], previous)
                    write_json(root / "plan.json", plan.model_dump(mode="json"))
                    write_json(root / "plan-binding.json", {"hash": digest(plan.model_dump(mode="json"))})
                else:
                    workflow.load_plan(root)
                result = {"status": "COMPLETE"}
            elif command == "pilot":
                result = workflow.generate(root, config, "pilot")
                # Valid tasks still get all four samples when another task has a quota gap.
                workflow.collect(root, config, "pilot")
                result = workflow.export(root, config, "pilot")
            elif command == "generate":
                result = workflow.generate(root, config, stage, args.split)
            elif command == "collect":
                result = workflow.collect(root, config, stage)
            elif command == "export":
                result = workflow.export(root, config, stage)
            elif command == "accept-evaluation":
                result = accept(root, workflow.load_plan(root), args.receipt)
            else:
                result = workflow.report(root, config, stage)
            status = result.get("status", "INCOMPLETE")
            write_json(state, {"identity": identity, "status": status})
            print(json.dumps({"command": command, "stage": stage, "round_dir": str(root), **result}, ensure_ascii=False))
            if status not in {"COMPLETE", "PASS"}:
                raise SystemExit(2)
            return result
        except Exception as exc:
            write_json(state, {"identity": identity, "status": "INCONCLUSIVE", "reason": f"{type(exc).__name__}: {exc}"})
            raise
