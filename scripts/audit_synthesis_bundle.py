"""Read-only cross-artifact audit for the RL-to-SFT delivery gate."""

import argparse
import json
from collections import Counter
from pathlib import Path


def check_sft_sample(sample, captured, task_id, native_messages=()):
    """Reject altered context, private results and incorrect loss masks."""
    errors = []
    if sample != captured:
        errors.append("SFT differs from the saved actual agent-visible context")
    if sample.get("task_id") != task_id:
        errors.append("SFT task ID mismatch")
    if sample.get("retrieval_config") != "bm25_grep":
        errors.append("SFT retrieval mode is not bm25_grep")
    messages = sample.get("messages", [])
    private_ids = {
        m["id"]
        for m in native_messages
        if m["role"] == "tool" and m.get("requestor") == "user"
    }
    if any(
        m["role"] == "tool" and m.get("tool_call_id") in private_ids for m in messages
    ):
        errors.append("Native private user tool result appears in agent context")
    if sample.get("loss_mask") != [int(m["role"] == "assistant") for m in messages]:
        errors.append(
            "Loss mask includes non-assistant content or omits assistant content"
        )
    if any(m.get("weight") != 1 for m in messages if m["role"] == "assistant"):
        errors.append("Assistant loss weight is missing or incorrect")
    names = {t.get("function", {}).get("name") for t in sample.get("tools", [])}
    if not {"KB_search", "grep"} <= names:
        errors.append("Actual visible tool definitions lack KB_search or grep")
    return errors


def check_sft_attempts(started, trials, scenario_seed):
    """Require durable, resolved start markers for at most four actual attempts."""
    errors = []
    expected = {f"sft:{i}": scenario_seed + 100 + i for i in range(4)}
    if set(started) - set(expected):
        errors.append("SFT start markers exceed four allowed attempt slots")
    if set(trials) != set(started):
        errors.append("SFT start markers and completed trial records differ")
    for key, marker in started.items():
        if marker.get("seed") != expected.get(key):
            errors.append("SFT start marker seed differs from its assigned slot")
        if key in trials and trials[key].get("seed") != marker.get("seed"):
            errors.append("SFT trial seed differs from its start marker")
    return errors


def allowed_models(manifest, config):
    """Return the configured model plus any explicitly recorded prior runtime model."""
    models = {
        config.generator_model,
        config.teacher_model,
        config.user_model,
        config.judge_model,
    }
    for switch in manifest.get("runtime_switches", []):
        if switch.get("previous_model"):
            models.add(switch["previous_model"])
    return models


def audit(root, expected=200, require_sft=True):
    """Check published artifacts against native runs and visible captures.

    This verifies stored validation evidence; it does not replace strict replay
    or the independent business checks performed during task admission.
    """
    from tau3.synthesis.bundle import load_task_bundle
    from tau3.synthesis.models import FAMILIES, SynthesisConfig
    from tau3.synthesis.storage import digest, read_json, retrieval_settings
    from tau3.synthesis.workflow import records, trial_config

    errors = []
    tasks = load_task_bundle(root)
    manifest = read_json(root / "manifest.json")
    config = SynthesisConfig.model_validate(manifest["config"])
    splits = read_json(root / "split_tasks.json")
    items = {r.task.id: r for r in records(root)}
    family_counts = Counter(items[t.id].skeleton.family for t in tasks)
    combined = sum(
        bool(items[t.id].skeleton.private.get("open_business"))
        for t in tasks
        if items[t.id].skeleton.family == "ordering"
    )
    if combined < expected // 20:
        errors.append("Insufficient combined opening/closure coverage")
    if len(tasks) != expected:
        errors.append(f"Published {len(tasks)}/{expected} tasks")
    if family_counts != Counter({f: expected // 4 for f in FAMILIES}):
        errors.append("Family counts differ from the requested balanced allocation")
    if len(splits["validation"]) != expected // 5:
        errors.append("Validation split differs from the requested 20% allocation")
    if manifest["retrieval"] != retrieval_settings():
        errors.append("Resolved retrieval settings differ from the fixed environment")
    models = allowed_models(manifest, config)
    for role in ("generator", "teacher", "user", "judge"):
        if getattr(config, role + "_model") not in models:
            errors.append(f"Unexpected {role} endpoint")

    def inspect_trial(record, trial):
        prefix = f"{record.task.id}/{trial.get('seed')}"
        if trial.get("mode") != "bm25_grep":
            raise ValueError(prefix + ": non-primary retrieval mode")
        native = read_json(root / trial["trajectory"])
        capture = read_json((root / trial["trajectory"]).with_suffix(".capture.json"))
        expected_hashes = {
            digest(
                [
                    record.task.model_dump(mode="json"),
                    trial_config(
                        config.model_copy(
                            update={"teacher_model": model, "user_model": model}
                        ),
                        "bm25_grep",
                        trial["seed"],
                    ).model_dump(mode="json"),
                ]
            )
            for model in models
        }
        if capture["input_hash"] not in expected_hashes:
            raise ValueError(prefix + ": captured run inputs differ")
        if native["task_id"] != record.task.id or native["seed"] != trial["seed"]:
            raise ValueError(prefix + ": native task or seed mismatch")
        context_errors = check_sft_sample(
            capture["visible_sample"],
            capture["visible_sample"],
            record.task.id,
            native["messages"],
        )
        if context_errors:
            raise ValueError(prefix + ": " + "; ".join(context_errors))
        if trial.get("passed"):
            flags = trial.get("review") or {}
            if (
                trial.get("reward") != 1
                or native.get("reward_info", {}).get("reward") != 1
                or trial.get("infrastructure_error")
                or not all(
                    flags.get(k) is True
                    for k in (
                        "fact_consistent",
                        "no_answer_leak",
                        "follows_user_constraints",
                    )
                )
            ):
                raise ValueError(prefix + ": success lacks DB or quality evidence")
        return capture["visible_sample"]

    eligible_sft = {}
    for task in tasks:
        record = items[task.id]
        try:
            if record.task != task:
                raise ValueError("Candidate differs from published task")
            if not (
                record.accepted
                and record.static_passed
                and record.text_checked
                and record.text_mode == "llm"
            ):
                raise ValueError("Candidate has not passed all admission stages")
            static = record.checks["static"]
            hashes = static["db_hashes"]
            if len(hashes) != 2 or hashes[0] != hashes[1]:
                raise ValueError("Missing deterministic replay evidence")
            if not static["counterexamples"] or any(
                value != "rejected" for value in static["counterexamples"].values()
            ):
                raise ValueError("Counterexample was not rejected")
            if record.checks["reviewed_text_hash"] != digest(
                task.user_scenario.model_dump(mode="json")
            ):
                raise ValueError("Text review predates the user scenario")
            if record.checks["validated_content_hash"] != digest(
                [task.model_dump(mode="json"), record.skeleton.model_dump(mode="json")]
            ):
                raise ValueError("Admission evidence predates task contents")
            primary = [record.trials[f"bm25_grep:{i}"] for i in range(2)]
            if not any(t.get("passed") for t in primary):
                raise ValueError("No successful primary trial")
            if len({t["seed"] for t in primary}) != 2:
                raise ValueError("Primary seeds are not distinct")
            for trial in primary:
                inspect_trial(record, trial)
            sft_trials = {
                k: v for k, v in record.trials.items() if k.startswith("sft:")
            }
            attempt_errors = check_sft_attempts(
                record.checks.get("sft_attempts_started", {}),
                sft_trials,
                record.skeleton.seed,
            )
            if attempt_errors:
                raise ValueError("; ".join(attempt_errors))
            if set(sft_trials) - {f"sft:{i}" for i in range(4)}:
                raise ValueError("More than four SFT attempt slots")
            retained = [t for t in sft_trials.values() if t.get("passed")]
            if retained and task.id not in splits["train"]:
                raise ValueError("SFT sampled outside training split")
            if len(retained) > 2:
                raise ValueError("More than two successful SFT samples retained")
            for trial in retained:
                captured = inspect_trial(record, trial)
                sample = read_json(root / trial["sft_path"])
                problems = check_sft_sample(sample, captured, task.id)
                if problems:
                    raise ValueError("; ".join(problems))
                eligible_sft[digest(sample)] = sample
            fallback = record.checks.get("sft_primary_fallback")
            if fallback:
                if retained:
                    raise ValueError(
                        "Primary fallback exists beside retained SFT samples"
                    )
                source_key = fallback.get("source_key")
                source = record.trials.get(source_key, {})
                if not source.get("passed") or not source_key.startswith("bm25_grep:"):
                    raise ValueError(
                        "Primary fallback source is not a passing primary trial"
                    )
                captured = inspect_trial(record, source)
                sample = read_json(root / fallback["sft_path"])
                problems = check_sft_sample(sample, captured, task.id)
                if problems:
                    raise ValueError("; ".join(problems))
                eligible_sft[digest(sample)] = sample
        except (KeyError, ValueError, OSError, TypeError) as exc:
            errors.append(f"{task.id}: {exc}")
    exported = []
    if require_sft:
        path = root / "sft.jsonl"
        if path.exists():
            exported = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            errors.append("SFT export is missing")
        signatures = [digest(sample) for sample in exported]
        if len(signatures) != len(set(signatures)):
            errors.append("Duplicate exported SFT samples")
        if set(signatures) != set(eligible_sft):
            errors.append("SFT export differs from reviewed native/capture artifacts")
        missing = set(splits["train"]) - {s["task_id"] for s in exported}
        if missing:
            errors.append(f"{len(missing)} training tasks have no retained SFT success")
    return {
        "artifact_checks_passed": not errors,
        "requested_tasks": expected,
        "published_tasks": len(tasks),
        "families": dict(family_counts),
        "combined_ordering_tasks": combined,
        "splits": {k: len(v) for k, v in splits.items()},
        "sft_required": require_sft,
        "sft_samples": len(exported),
        "coverage_exclusions": read_json(root / "catalog.json").get("exclusions", []),
        "errors": errors,
    }


def main():
    """Print an audit report without modifying any bundle or environment files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=200)
    parser.add_argument("--rl-only", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(args.bundle, args.expected, not args.rl_only)
    except (ValueError, OSError, KeyError) as exc:
        report = {"artifact_checks_passed": False, "errors": [str(exc)]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["artifact_checks_passed"] else 1)


if __name__ == "__main__":
    main()
