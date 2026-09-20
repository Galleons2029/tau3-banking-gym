"""Incremental, evidence-bound SFT shards with isolated offline validation."""

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

VERSION = 1


def sha(text):
    """Hash exact serialized artifact text."""
    return hashlib.sha256(text.encode()).hexdigest()


def stamp(path):
    """Detect replacements and in-place edits, including preserved mtimes."""
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


@contextmanager
def observed_reads():
    """Read each unchanged input once per isolated validation, recording hashes."""
    original = Path.read_text
    observed, memo = {}, {}

    def read(path, *args, **kwargs):
        key = str(path.absolute())
        before = stamp(path)
        if key in memo and observed[key]["stat"] == before:
            return memo[key]
        text = original(path, *args, **kwargs)
        after = stamp(path)
        if before != after:
            raise ValueError(f"Input changed during validation: {key}")
        observed[key] = {"stat": after, "sha256": sha(text)}
        memo[key] = text
        return text

    Path.read_text = read
    try:
        yield observed
    finally:
        Path.read_text = original


def inventory(roots):
    """Detect additions/removals without recursively reading file contents."""
    found = []
    for root in roots:
        for directory, _, files in os.walk(root):
            found.extend(str(Path(directory) / f) for f in files if f.endswith(".json"))
    return sorted(found)


def valid_cache(cache, identity, roots, full=False):
    """Reuse only identical input bytes; final validation also replays all gates."""
    if cache.get("identity") != identity or inventory(roots) != cache.get("inventory"):
        return False
    if (
        sha(json.dumps(cache["rows"], ensure_ascii=False, sort_keys=True))
        != cache["rows_hash"]
    ):
        return False
    for name, bound in cache["inputs"].items():
        try:
            # FUSE can report unchanged timestamps for same-size in-place edits.
            # Every reused input is content-hashed, even during live checkpoints.
            if sha(Path(name).read_text()) != bound["sha256"]:
                return False
        except OSError:
            return False
    return True


def task_rows(root, config, slot, phase, plan_hash):
    """Apply the legacy exporter checks, retaining exactly its training rows."""
    from tau3.synthesis.targeted import workflow as w
    from tau3.synthesis.world_sft import trial_sample
    from tau3.worldgen.v2.json_output import parse_object
    from tau3.worldgen.v2.runtime import digest

    d = root / phase / "slots" / f"{slot.index:06d}"
    record = w.admitted_record(root, d / "task.json", plan_hash)
    rows = []
    for trial in range(4):
        t = d / "teacher" / str(trial)
        if not (t / "quality.json").exists():
            continue
        quality = json.loads((t / "quality.json").read_text())
        if not quality.get("sft_qualified"):
            continue
        captured = trial_sample(t)
        native = json.loads((t / "result.json").read_text())
        capture = json.loads((t / "capture.json").read_text())
        sample = json.loads((t / "sample.json").read_text())
        raw = json.loads(Path(quality["quality_evidence"]["path"]).read_text())
        if (
            quality["native_hash"] != digest(native)
            or quality["capture_hash"] != digest(capture)
            or quality["sample_hash"] != digest(sample)
            or w.clean_sample(captured) != sample
            or not w.QualityReview.model_validate(quality["quality"]).passed
            or digest(raw) != quality["quality_evidence"]["hash"]
            or parse_object(raw["response"]) != quality["quality"]
        ):
            raise ValueError("SFT evidence changed")
        rows.append(
            {
                **sample,
                "metadata": {
                    "round": str(root.resolve()),
                    "phase": phase,
                    "teacher_model": config.teacher_model,
                    "seed": quality["seed"],
                    "trial": trial,
                    "recipe": slot.recipe,
                    "secondary": slot.secondary,
                    "difficulty": slot.difficulty,
                    "world_hash": record.world_hash,
                    "plan_hash": record.plan_hash,
                    "business_fingerprint": record.business_fingerprint,
                    "evidence": str(t.resolve()),
                },
            }
        )
    return rows


def validate_task(root_string, config_raw, slot_raw, phase, plan_hash, full=False):
    """Validate one task and atomically cache a recoverable shard."""
    from tau3.synthesis.targeted.models import RoundConfig, Slot
    from tau3.worldgen.v2.pipeline import write_json

    root = Path(root_string)
    config, slot = RoundConfig.model_validate(config_raw), Slot.model_validate(slot_raw)
    directory = root / phase / "slots" / f"{slot.index:06d}"
    cache_path = root / phase / "export-shards" / f"{slot.index:06d}.json"
    started = time.monotonic()
    task = json.loads((directory / "task.json").read_text())
    world = Path(task["world"])
    readiness = world.parent / "audit/readiness.json"
    admission = json.loads(readiness.read_text())
    roots = [str(world), *(s["directory"] for s in admission["sources"].values())]
    # Only completed trial directories are stable. New quality markers invalidate
    # the identity without treating an in-flight capture as exportable evidence.
    quality = {
        str(i): json.loads(p.read_text())
        for i in range(4)
        if (p := directory / "teacher" / str(i) / "quality.json").exists()
    }
    roots += [str(directory / "teacher" / i) for i in quality]
    identity = {
        "version": VERSION,
        "exporter": sha(Path(__file__).read_text()),
        "plan": plan_hash,
        "task": task,
        "quality": quality,
        "settings": json.loads((root / "settings.json").read_text()),
        "teacher_model": config.teacher_model,
    }
    complete = len(quality) == 4 and all(
        q.get("status") == "COMPLETE" for q in quality.values()
    )
    if cache_path.exists() and not full:
        cached = json.loads(cache_path.read_text())
        if valid_cache(cached, identity, roots, full):
            return {
                "slot": slot.index,
                "cache": str(cache_path),
                "hit": True,
                "four_complete": complete,
                "rows": len(cached["rows"]),
                "rows_hash": cached["rows_hash"],
                "seconds": time.monotonic() - started,
            }
    before = inventory(roots)
    with observed_reads() as inputs:
        # Bind the admission report itself as well as all of its source files.
        readiness.read_text()
        rows = task_rows(root, config, slot, phase, plan_hash)
    after = inventory(roots)
    if before != after:
        raise ValueError("Input file membership changed during export")
    if any(sha(Path(p).read_text()) != item["sha256"] for p, item in inputs.items()):
        raise ValueError("Input changed after validation")
    cache = {
        "identity": identity,
        "inventory": after,
        "inputs": inputs,
        "rows": rows,
        "rows_hash": sha(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
    }
    write_json(cache_path, cache)
    return {
        "slot": slot.index,
        "cache": str(cache_path),
        "hit": False,
        "four_complete": complete,
        "rows": len(rows),
        "rows_hash": cache["rows_hash"],
        "seconds": time.monotonic() - started,
    }


def merge_rows(groups):
    """Idempotently merge task shards; conflicting duplicate identities fail."""
    result = {}
    for rows in groups:
        for row in rows:
            m = row["metadata"]
            key = (row["task_id"], m["trial"], m["seed"])
            if key in result and result[key] != row:
                raise ValueError("Conflicting duplicate SFT identity")
            result[key] = row
    return sorted(
        result.values(),
        key=lambda row: (
            int(Path(row["metadata"]["evidence"]).parents[1].name),
            row["metadata"]["trial"],
        ),
    )


def atomic_json(path, value):
    """Persist a publication manifest without an in-place overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def text_digest(text):
    """Match the legacy digest of JSONL text exactly."""
    return sha(json.dumps(text, ensure_ascii=False))


def read_baseline(root, phase, plan_hash):
    """Recover an interrupted JSONL/manifest publication using its write-ahead record."""
    directory = root / phase
    target, manifest = directory / "sft.jsonl", directory / "export.json"
    if not target.exists():
        return []
    text = target.read_text()
    bound = json.loads(manifest.read_text()) if manifest.exists() else {}
    if bound.get("hash") != text_digest(text):
        pending_path = directory / "export.pending.json"
        pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}
        if (
            pending.get("hash") != text_digest(text)
            or pending.get("plan_hash") != plan_hash
            or pending.get("phase") != phase
        ):
            raise ValueError("Existing export binding changed")
        atomic_json(manifest, pending)
        bound = pending
    if bound.get("plan_hash") != plan_hash or bound.get("phase") != phase:
        raise ValueError("Existing export belongs to another plan or phase")
    rows = [json.loads(line) for line in text.splitlines() if line]
    if len(rows) != bound.get("rows"):
        raise ValueError("Existing export row count changed")
    return rows


def publish(root, phase, plan_hash, rows, extra):
    """Publish complete JSONL atomically with a recoverable manifest transaction."""
    target = root / phase / "sft.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    bound = {
        "rows": len(rows),
        "hash": text_digest(text),
        "plan_hash": plan_hash,
        "phase": phase,
        **extra,
    }
    atomic_json(root / phase / "export.pending.json", bound)
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(target)
    atomic_json(root / phase / "export.json", bound)


def export_incremental(
    root, config, phase="formal", workers=8, full=False, indexes=None
):
    """Process changed shards in parallel; publish progress without shrinking SFT."""
    from tau3.synthesis.targeted import workflow as w
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest

    root = Path(root)
    with file_lock(root / "incremental-export.lock"):
        started = time.monotonic()
        plan = w.load_plan(root)
        plan_hash = digest(plan.model_dump())
        slots = plan.pilot if phase == "pilot" else plan.slots
        existing = {p.name for p in (root / phase / "slots").iterdir() if p.is_dir()}
        selected = [
            s
            for s in slots
            if f"{s.index:06d}" in existing
            and (indexes is None or s.index in indexes)
            and (root / phase / "slots" / f"{s.index:06d}" / "task.json").exists()
        ]
        baseline = read_baseline(root, phase, plan_hash)
        published_slots = {
            int(Path(r["metadata"]["evidence"]).parents[1].name) for r in baseline
        }
        # Deliver the already-collected backlog first, while preserving old rows.
        selected.sort(key=lambda slot: (slot.index in published_slots, slot.index))
        groups, failures, results = {}, {}, []
        last_publish = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    validate_task,
                    str(root),
                    config.model_dump(),
                    s.model_dump(),
                    phase,
                    plan_hash,
                    full,
                ): s.index
                for s in selected
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    cached = json.loads(Path(result["cache"]).read_text())
                    if (
                        sha(
                            json.dumps(
                                cached["rows"], ensure_ascii=False, sort_keys=True
                            )
                        )
                        != result["rows_hash"]
                    ):
                        raise ValueError("Task shard changed after validation")
                    groups[index] = cached["rows"]
                    results.append(result)
                except Exception as exc:
                    failures[index] = f"{type(exc).__name__}: {exc}"
                progress = {
                    "status": "RUNNING",
                    "selected": len(selected),
                    "processed": len(results) + len(failures),
                    "failures": failures,
                    "cache_hits": sum(r["hit"] for r in results),
                    "elapsed_seconds": time.monotonic() - started,
                    "workers": workers,
                }
                write_json(root / phase / "incremental-export-progress.json", progress)
                if not failures and time.monotonic() - last_publish >= 30:
                    retained = [
                        r
                        for r in baseline
                        if int(Path(r["metadata"]["evidence"]).parents[1].name)
                        not in groups
                    ]
                    publish(
                        root,
                        phase,
                        plan_hash,
                        merge_rows([retained, *groups.values()]),
                        {"incremental": True, "validation_complete": False},
                    )
                    last_publish = time.monotonic()
        if failures:
            write_json(
                root / phase / "incremental-export-progress.json",
                {**progress, "status": "INCOMPLETE"},
            )
            raise ValueError(f"Export evidence validation failed: {failures}")
        retained = [
            r
            for r in baseline
            if int(Path(r["metadata"]["evidence"]).parents[1].name) not in groups
        ]
        rows = merge_rows([retained, *groups.values()])
        publish(
            root,
            phase,
            plan_hash,
            rows,
            {
                "incremental": True,
                "validation_complete": indexes is None,
                "full_hash_audit": full,
            },
        )
        summary = {
            "status": "PASS",
            "tasks": len(selected),
            "rows": len(rows),
            "cache_hits": sum(r["hit"] for r in results),
            "elapsed_seconds": time.monotonic() - started,
            "full_hash_audit": full,
            "results": results,
        }
        write_json(root / phase / "incremental-export-progress.json", summary)
        if indexes is None:
            if (
                len(selected) == len(slots)
                and all(r["four_complete"] for r in results)
                and not full
            ):
                write_json(
                    root / phase / "full-export-audit-required.json", {"required": True}
                )
                raise ValueError(
                    "All slots finished; rerun export with full content-hash audit"
                )
            w.report(root, config, phase)
        return summary
