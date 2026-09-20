"""Regression checks for export isolation and completion accounting."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location(
    "targeted_continuous",
    Path(__file__).parents[1] / "scripts/run_targeted_continuous.py",
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_slow_export_never_waits_or_launches_a_second_process(tmp_path, monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    exporter = driver.ExportProcess(["export"], tmp_path / "export.log", interval=0)
    exporter.tick(0)
    for revision in range(1, 100):
        exporter.tick(revision)
    assert launch.call_count == 1
    process.wait.assert_not_called()
    assert exporter.finished_revision == -1
    process.poll.return_value = 0
    exporter.tick(100, final=True)
    assert exporter.finished_revision == 0  # Old export cannot cover new work.
    monkeypatch.setattr(driver.time, "monotonic", lambda: exporter.last_start + 31)
    exporter.tick(100, final=True)
    assert launch.call_count == 2
    assert exporter.started_revision == 100


def test_failed_export_does_not_mark_revision_complete(tmp_path, monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = 2
    monkeypatch.setattr(driver.subprocess, "Popen", Mock(return_value=process))
    exporter = driver.ExportProcess(["export"], tmp_path / "export.log", interval=0)
    exporter.tick(5)
    exporter.tick(5, final=True)
    assert exporter.finished_revision == -1
    assert exporter.exit_code == 2
    assert exporter.process is None
