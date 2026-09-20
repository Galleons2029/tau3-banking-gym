"""Qualify frozen seed snapshots, then admit a pilot and prepare a bounded batch."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def reuse_blind_responses(source, output, world, config):
    """Import exact raw public responses with origin metadata, never PASS flags."""
    from tau3.worldgen.v2.audit import AuditSession
    from tau3.worldgen.v2.pipeline import artifact_hashes, write_json
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.settings import load_settings

    original = json.loads((source / "audit.json").read_text())
    settings = load_settings(config)
    if (
        original["artifacts"] != artifact_hashes(world)
        or original["settings"] != digest(settings.model_dump())
        or original["kind"] != "blind-v1"
    ):
        raise ValueError(
            "Response reuse requires identical public artifacts/settings and blind audit kind"
        )
    AuditSession(world, output, settings, original["max_calls"], "blind-v1")
    imported, excluded = [], []
    for path in sorted((source / "calls").glob("*.json")):
        record = json.loads(path.read_text())
        if path.stem != digest(record["request"]):
            raise ValueError("Original response request hash changed")
        if record["status"] != "COMPLETE" or record.get("finish_reason") not in (
            None,
            "stop",
        ):
            excluded.append(
                {
                    "file": path.name,
                    "status": record["status"],
                    "reason": "No complete normally terminated response",
                }
            )
            continue
        imported_record = {
            **record,
            "origin": {
                "directory": str(source.resolve()),
                "audit_identity": original,
                "record_hash": digest(record),
            },
        }
        target = output / "calls" / path.name
        if target.exists() and json.loads(target.read_text()) != imported_record:
            raise ValueError("Destination response already differs")
        write_json(target, imported_record)
        imported.append(path.name)
    result = {
        "scope": "Raw responses only; new checker must reproduce all inputs and revalidate every result. Invalid returned content is imported too.",
        "source": str(source.resolve()),
        "source_audit": original,
        "imported": imported,
        "excluded": excluded,
    }
    write_json(output / "response-provenance.json", result)
    return result


def resume_full(root: Path, config: Path):
    """Resume exact checkpoints; the native full driver rechecks all smoke gates."""
    from tau3.worldgen.v2.pipeline import write_json

    mirrors_path = root / "local-worlds.json"
    mirrors = json.loads(mirrors_path.read_text())
    driver = [
        sys.executable,
        "scripts/validate_worldgen_v2_independent.py",
        "--output",
        str(root),
        "--config",
        str(config),
        "--local-worlds",
        str(mirrors_path),
    ]

    def state(phase, **fields):
        write_json(root / "stage-state.json", {"phase": phase, **fields})

    def run(command, log):
        with (root / log).open("a") as handle:
            return subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)

    state("full", full_started=True)
    code = run(driver + ["--stage", "full", "--parallel-online"], "driver.log")
    if code:
        state("full_incomplete", full_started=True, full_exit_code=code)
        raise SystemExit(code)
    state("task_pilot", full_started=True, full_exit_code=0)
    code = run(
        [
            sys.executable,
            "scripts/run_world_task_synthesis.py",
            "--world",
            str(root / "scale"),
            "--readiness",
            str(root / "full/scale/readiness.json"),
            "--config",
            str(config),
            "--local-world",
            mirrors["scale"],
            "--pilot",
            str(root / "task-pilot-20"),
            "--output",
            str(root / "task-batch-200"),
            "--prepare-only",
        ],
        "task-pilot.log",
    )
    state(
        "READY" if code == 0 else "task_pilot_incomplete",
        full_started=True,
        full_exit_code=0,
        pilot_exit_code=code,
    )
    raise SystemExit(code)


def main():
    """Persist every stage and stop on incomplete evidence without resampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reuse-blind-from", type=Path)
    parser.add_argument("--worker", choices=["blind", "calibration", "online"])
    parser.add_argument(
        "--resume-full",
        action="store_true",
        help="Resume full validation from existing frozen smoke evidence",
    )
    args = parser.parse_args()
    from tau3.utils.llm_concurrency import install_request_limit
    from tau3.worldgen.v2.pipeline import write_json

    install_request_limit()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if args.resume_full:
        if args.worker or args.reuse_blind_from:
            parser.error("--resume-full cannot import responses or run smoke workers")
        from tau3.utils.llm_concurrency import file_lock

        with file_lock(root / "qualification.lock"):
            return resume_full(root, args.config)
    mirrors_path = root / "local-worlds.json"
    if args.worker:
        from tau3.worldgen.v2.blind import run_blind
        from tau3.worldgen.v2.calibration import run_calibration
        from tau3.worldgen.v2.rollouts import run_matrix
        from tau3.worldgen.v2.specs import WorldSpec

        world = Path(json.loads(mirrors_path.read_text())["world"])
        spec = WorldSpec.model_validate_json((world / "spec.json").read_text())
        ids = [
            f"task_{s.id}"
            for c in (spec.categories[0], spec.categories[-1])
            for s in c.scenarios
        ]
        output = root / "smoke" / args.worker
        kwargs = {"settings_path": args.config}
        if args.worker == "blind":
            report = run_blind(world, output, ids, max_calls=48, **kwargs)
        elif args.worker == "calibration":
            report = run_calibration(world, output, max_calls=200, **kwargs)
        else:
            report = run_matrix(world, output, ids, max_rollouts=24, **kwargs)
        raise SystemExit(0 if report["status"] == "PASS" else 2)

    from validate_worldgen_v2_independent import refresh

    os.environ["TAU3_SYNTH_ROLLOUT_CONFIG"] = str(args.config.resolve())

    def state(phase, **fields):
        write_json(root / "stage-state.json", {"phase": phase, **fields})

    state("prepare", full_started=False)
    for name in ("world", "scale", "natural"):
        refresh(args.source / name, root / name)
    if not mirrors_path.exists():
        local = Path(tempfile.mkdtemp(prefix="worldgen-qualified-"))
        mirrors = {}
        for name in ("world", "scale", "natural"):
            shutil.copytree(root / name, local / name)
            mirrors[name] = str(local / name)
        write_json(mirrors_path, mirrors)
    mirrors = json.loads(mirrors_path.read_text())
    if args.reuse_blind_from:
        reuse = {}
        pairs = [("smoke/blind", "world")]
        for name in ("scale", "natural"):
            pairs.extend(
                (str(p.relative_to(args.reuse_blind_from)), name)
                for p in sorted(
                    (args.reuse_blind_from / "full" / name / "blind-shards").glob(
                        "[0-9]*"
                    )
                )
                if (p / "audit.json").exists()
            )
        for relative, name in pairs:
            reuse[relative] = reuse_blind_responses(
                args.reuse_blind_from / relative,
                root / relative,
                Path(mirrors[name]),
                args.config,
            )
        write_json(root / "blind-response-reuse.json", reuse)
    base = [
        sys.executable,
        __file__,
        "--output",
        str(root),
        "--source",
        str(args.source),
        "--config",
        str(args.config),
    ]

    def run(command, log):
        with (root / log).open("a") as handle:
            return subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)

    state("smoke_blind", full_started=False)
    if run(base + ["--worker", "blind"], "blind.log"):
        state("smoke_blind_incomplete", full_started=False)
        raise SystemExit(2)
    state("smoke_calibration_online", full_started=False)
    children = []
    for worker in ("calibration", "online"):
        with (root / f"{worker}.log").open("a") as handle:
            children.append(
                subprocess.Popen(
                    base + ["--worker", worker], stdout=handle, stderr=subprocess.STDOUT
                )
            )
    codes = [child.wait() for child in children]
    driver = [
        sys.executable,
        "scripts/validate_worldgen_v2_independent.py",
        "--output",
        str(root),
        "--config",
        str(args.config),
        "--local-worlds",
        str(mirrors_path),
    ]
    code = run(driver + ["--stage", "smoke", "--summarize-only"], "driver.log")
    if any(codes) or code:
        state("smoke_incomplete", full_started=False, worker_exit_codes=codes)
        raise SystemExit(2)
    return resume_full(root, args.config)


if __name__ == "__main__":
    main()
