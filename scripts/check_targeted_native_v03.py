"""Offline compiler preflight; never starts LLM calls or certifies public admission."""

import argparse
from pathlib import Path

from tau3.synthesis.storage import read_json, write_json
from tau3.synthesis.targeted.models import Finding
from tau3.synthesis.targeted.native.business_checks import check_initial_consistency
from tau3.synthesis.targeted.native.coverage import coverage_report, pair_report
from tau3.synthesis.targeted.native.models import NativeConfig, NativeProfile
from tau3.synthesis.targeted.native.planning import compile_plan
from tau3.synthesis.targeted.native.scenarios import compile_candidate
from tau3.synthesis.targeted.native.validation import validate_static


def main():
    """Record quota, compiler and policy gaps before paying for narrative or teachers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static", action="store_true", help="Replay each pilot reference and its counterexamples")
    parser.add_argument("--full", action="store_true", help="Also check the frozen 3000-task allocation offline")
    args = parser.parse_args()
    config = NativeConfig.model_validate(read_json(args.config))
    profile = NativeProfile(round_id="offline-compiler-probe", base_model="not-an-evaluation",
        report_hash="probe-only", findings=[Finding(label=label, observation="compiler probe, not evidence",
        interpretation="not a production profile", source="probe", quote="probe", confidence="low", severity="low")
        for label in ("F1", "F2", "F3", "F3b", "F4", "F5", "F6", "F7", "F8")])
    plan = compile_plan(config, profile, "offline-probe-only")
    write_json(args.output / "probe-plan.json", plan.model_dump(mode="json"))
    summaries = {}
    stages = [("pilot", plan.pilot), ("small", plan.slots[:400]), ("validation", plan.validation)]
    if args.full:
        stages.append(("full", plan.slots))
    for stage, slots in stages:
        candidates, gaps, static = [], [], {}
        for slot in slots:
            try:
                candidate = compile_candidate(slot)
                check_initial_consistency(candidate)
                candidates.append(candidate)
                if args.static and stage == "pilot":
                    static[str(slot.index)] = validate_static(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                gaps.append({"slot": slot.model_dump(mode="json"), "reason": f"{type(exc).__name__}: {exc}"})
        coverage = coverage_report(candidates, plan.operation_coverage, "pilot" if stage == "validation" else stage)
        pairing = pair_report(candidates, slots)
        record = {"production_admission": False, "stage": stage, "compiled": len(candidates),
                  "expected": len(slots), "gaps": gaps, "static": static, "coverage": coverage, "pairs": pairing}
        record["status"] = "PASS" if not gaps and coverage["status"] == pairing["status"] == "PASS" else "FAIL"
        write_json(args.output / f"{stage}.json", record)
        summaries[stage] = {"compiled": len(candidates), "gaps": len(gaps), "status": record["status"],
                            "operation_gaps": coverage["gaps"], "pair_status": pairing["status"]}
        print(stage, summaries[stage], flush=True)
    write_json(args.output / "summary.json", summaries)
    return 0 if all(s["status"] == "PASS" for s in summaries.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
