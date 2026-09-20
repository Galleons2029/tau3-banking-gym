"""Fast pure regressions for cache integrity and idempotent merging."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "incremental_export_test",
    Path(__file__).parents[1] / "src/tau3/synthesis/targeted/incremental_export.py",
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def cache_for(path):
    rows = []
    return {
        "identity": "test",
        "inventory": m.inventory([str(path.parent)]),
        "rows": rows,
        "rows_hash": m.sha(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
        "inputs": {
            str(path): {"stat": m.stamp(path), "sha256": m.sha(path.read_text())}
        },
    }


def test_cache_reuses_only_unchanged_evidence(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("good")
    c = cache_for(p)
    assert m.valid_cache(c, "test", [str(tmp_path)])
    before = p.stat()
    p.write_text("evil")
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not m.valid_cache(c, "test", [str(tmp_path)])
    c["inputs"][str(p)]["stat"] = m.stamp(p)
    assert not m.valid_cache(c, "test", [str(tmp_path)], full=True)


def test_cache_detects_new_files_and_corrupt_shard(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("good")
    c = cache_for(p)
    extra = tmp_path / "new.json"
    extra.write_text("{}")
    assert not m.valid_cache(c, "test", [str(tmp_path)])
    extra.unlink()
    c["rows"] = [{"unexpected": "row"}]
    assert not m.valid_cache(c, "test", [str(tmp_path)])


def test_read_memo_is_restored_and_tracks_changed_sources(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("first")
    original = Path.read_text
    with pytest.raises(RuntimeError), m.observed_reads() as evidence:
        assert p.read_text() == p.read_text() == "first"
        p.write_text("second")
        assert p.read_text() == "second"
        assert evidence[str(p)]["sha256"] == m.sha("second")
        raise RuntimeError("stop")
    assert Path.read_text is original


def test_merge_resume_is_idempotent_and_conflicts_fail():
    row = {
        "task_id": "task_a",
        "messages": [],
        "metadata": {"trial": 0, "seed": 12, "evidence": "/slots/000012/teacher/0"},
    }
    assert m.merge_rows([[row], [row]]) == [row]
    changed = {**row, "messages": ["changed"]}
    with pytest.raises(ValueError, match="Conflicting"):
        m.merge_rows([[row], [changed]])


def test_interrupted_publication_recovers_without_duplicate_rows(tmp_path, monkeypatch):
    rows = [{"task_id": "task_a"}]
    original = m.atomic_json

    def fail_manifest(path, value):
        if path.name == "export.json":
            raise OSError("simulated interruption after JSONL rename")
        return original(path, value)

    monkeypatch.setattr(m, "atomic_json", fail_manifest)
    with pytest.raises(OSError):
        m.publish(tmp_path, "formal", "plan", rows, {})
    monkeypatch.setattr(m, "atomic_json", original)
    assert m.read_baseline(tmp_path, "formal", "plan") == rows
    assert m.read_baseline(tmp_path, "formal", "plan") == rows
    (tmp_path / "formal/sft.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="binding changed"):
        m.read_baseline(tmp_path, "formal", "plan")
