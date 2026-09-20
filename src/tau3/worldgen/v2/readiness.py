"""Separate, fail-closed admission for further controlled corpus expansion."""

import json
from pathlib import Path

from tau3.worldgen.v2.pipeline import (
    artifact_hashes,
    check_certificate,
    implementation_hash,
    write_json,
)
from tau3.worldgen.v2.runtime import digest
from tau3.worldgen.v2.settings import load_settings


def evidence_files(directory: Path) -> dict:
    """Bind all audit inputs, frozen solutions and outputs, not only a PASS flag."""
    return {
        str(p.relative_to(directory)): digest(p.read_text())
        for p in sorted(directory.rglob("*.json"))
        if p.is_file()
    }


def assess(
    root: Path,
    blind: Path,
    calibration: Path,
    online: Path,
    output: Path,
    settings_path: Path | None = None,
    *,
    admission_mode: str = "expansion",
) -> dict:
    """Missing, stale, incomplete or failing evidence never grants expansion access."""
    certificate = check_certificate(root)
    if admission_mode not in {"expansion", "targeted_training"}:
        raise ValueError("Unknown admission mode")
    if admission_mode == "targeted_training":
        from tau3.worldgen.v2.specs import WorldSpec

        spec = WorldSpec.model_validate_json((root / "spec.json").read_text())
        if spec.purpose != "training_curriculum" or spec.public_record_snapshot:
            raise ValueError("Independent training admission requires a targeted world")
    settings = load_settings(settings_path)
    identity = {
        "artifacts": artifact_hashes(root),
        "implementation": implementation_hash(),
        "settings": digest(settings.model_dump()),
    }
    ids = {p.stem for p in (root / "tasks").glob("*.json")}
    models = set(settings.agent_models)
    checks, sources = {}, {}
    reports = {}
    for name, directory in [
        ("blind", blind),
        ("calibration", calibration),
        ("online", online),
    ]:
        if name == "online" and admission_mode == "targeted_training":
            continue
        try:
            report = json.loads((directory / "report.json").read_text())
            sources[name] = {
                "directory": str(directory.resolve()),
                "files": evidence_files(directory),
            }
            if any(report.get("identity", {}).get(k) != v for k, v in identity.items()):
                raise ValueError("Stale or mismatched evidence identity")
            reports[name] = report
            checks[name] = report["status"]
        except (OSError, ValueError, KeyError) as exc:
            checks[name] = "INCONCLUSIVE"
            reports[name] = {"reason": str(exc)}
    b = reports["blind"]
    rows = b.get("results", [])
    pairs = [(r.get("task_id"), r.get("model")) for r in rows]
    if (
        set(b.get("task_ids", [])) != ids
        or len(pairs) != len(ids) * len(models)
        or set(pairs) != {(tid, model) for tid in ids for model in models}
        or any(r.get("status") != "PASS" for r in rows)
    ):
        checks["blind"] = "INCONCLUSIVE"
    elif checks["blind"] == "PASS":
        from tau3.worldgen.v2.blind import verify_saved_blind

        try:
            for index, model in enumerate(settings.agent_models):
                for task_id in sorted(ids):
                    verify_saved_blind(
                        root, blind, task_id, model, settings.agent_models[1 - index]
                    )
        except (OSError, ValueError, KeyError):
            checks["blind"] = "INCONCLUSIVE"
    c = reports["calibration"]
    from tau3.worldgen.v2.calibration import controls

    try:
        suite = controls(root)
        pairs = [(r.get("case_id"), r.get("model")) for r in c.get("results", [])]
        if (
            c.get("suite_hash") != digest(suite)
            or len(pairs) != len(suite) * len(models)
            or set(pairs) != {(case["id"], model) for case in suite for model in models}
            or any(r.get("status") != "PASS" for r in c.get("results", []))
        ):
            checks["calibration"] = "INCONCLUSIVE"
    except ValueError:
        checks["calibration"] = "INCONCLUSIVE"
    o = reports.get("online", {})
    cells = o.get("cells", [])
    pairs = [
        (r.get("task_id"), r.get("model"), r.get("retrieval"), r.get("trial"))
        for r in cells
    ]
    trials = {r.get("trial") for r in cells}
    expected = {
        (tid, model, retrieval, trial)
        for tid in ids
        for model in models
        for retrieval in ("full_kb", "bm25")
        for trial in trials
    }
    if (
        not trials
        or None in trials
        or set(o.get("task_ids", [])) != ids
        or set(pairs) != expected
        or len(pairs) != len(expected)
        or any(
            r.get("judge_inconclusive")
            or r.get("reward") is None
            or r.get("termination") not in {"agent_stop", "user_stop"}
            for r in cells
        )
    ):
        checks["online"] = "INCONCLUSIVE"
    # At least one observed supported success for every task, in addition to the
    # two frozen blind solutions. A weak model's failure does not prove bad data.
    checks["online_solvability"] = (
        "PASS"
        if all(
            any(
                r.get("task_id") == tid
                and r.get("retrieval") == "full_kb"
                and r.get("reward") == 1
                and not r.get("judge_inconclusive")
                for r in cells
            )
            for tid in ids
        )
        else "INCONCLUSIVE"
    )
    if admission_mode == "targeted_training":
        # Two independently verified public solutions establish solvability.
        # Online success is measured by fixed-four collection, not task validity.
        del checks["online"]
        del checks["online_solvability"]
    report = {
        "status": "PASS"
        if all(v == "PASS" for v in checks.values())
        else "INCONCLUSIVE",
        "identity": identity,
        "admission_mode": admission_mode,
        "checks": checks,
        "sources": sources,
        "text_scope": certificate["scope"],
        "assurance": "Admission for controlled expansion only; no claim of production-scale truth or paper equivalence",
    }
    write_json(output, report)
    return report


def check_readiness(root: Path, path: Path, settings_path: Path | None = None) -> dict:
    """Recheck source hashes and the actual gates before using an admission report."""
    report = json.loads(path.read_text())
    if report.get("status") != "PASS":
        raise ValueError("Controlled expansion readiness has not passed")
    for source in report["sources"].values():
        if evidence_files(Path(source["directory"])) != source["files"]:
            raise ValueError("Readiness evidence changed")
    sources = report["sources"]
    # Recompute without overwriting the original certificate on failure.
    check_path = path.with_name(path.stem + ".recheck.json")
    fresh = assess(
        root,
        Path(sources["blind"]["directory"]),
        Path(sources["calibration"]["directory"]),
        Path(sources.get("online", {"directory": path.parent / "online"})["directory"]),
        check_path,
        settings_path,
        admission_mode=report.get("admission_mode", "expansion"),
    )
    if fresh != report:
        raise ValueError("Stale readiness report")
    return report
