"""Filesystem-backed index and calibration store for the web console."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tau3.synthesis.storage import environment_fingerprint
from tau3.web_console.models import (
    ArtifactRef,
    ChangeSet,
    ChangeSetCreate,
    HumanAttribution,
    PatchOperation,
    ReasoningBlock,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    """Hash canonical JSON data."""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def opaque_id(prefix: str, relative: str) -> str:
    """Build a stable opaque identifier from a trusted relative locator."""

    return f"{prefix}_{hashlib.sha256(relative.encode()).hexdigest()[:16]}"


def package_name_prefix(name: str) -> str:
    """Return a stable family prefix for heterogeneous synthesis package names."""

    for pattern in (r"^(targeted-v\d+)", r"^(pilot)(?:-|$)", r"^(train)(?:-|$)"):
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return name.split("-", 1)[0].lower()


def task_id_prefix(task_id: str) -> str:
    """Remove generated hash components while retaining semantic task suffixes."""

    parts = task_id.split("_")
    semantic = [part for part in parts if not re.fullmatch(r"[0-9a-f]{8,}", part)]
    return "_".join(semantic) or task_id


def read_json_retry(path: Path) -> Any:
    """Read JSON, tolerating a file being replaced while it is observed."""

    error: Exception | None = None
    for _ in range(2):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(0.05)
    assert error is not None
    raise error


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def paginate(
    items: list[dict],
    page: int,
    page_size: int,
    sort_by: str = "",
    order: str = "asc",
) -> dict[str, Any]:
    """Return a deterministic page envelope."""

    if sort_by:
        if not any(sort_by in item for item in items):
            sort_by = ""

    if sort_by:

        def sort_key(item: dict) -> tuple[int, float | str]:
            value = item.get(sort_by)
            if value is None:
                return (2, "")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return (0, value)
            return (1, str(value).casefold())

        items = sorted(items, key=sort_key, reverse=order == "desc")
    page = max(1, page)
    page_size = min(100, max(1, page_size))
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": len(items),
    }


def _reasoning_candidates(raw: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def visit(value: Any, path: str = "raw_data") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                lowered = key.lower()
                if (
                    isinstance(child, str)
                    and child.strip()
                    and lowered in {"reasoning_content", "reasoning", "thinking"}
                ):
                    found.append((child_path, child.strip()))
                elif isinstance(child, (dict, list)):
                    visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(raw)
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, text in found:
        if text not in seen:
            seen.add(text)
            unique.append((source, text))
    return unique


def reasoning_blocks(message: dict) -> list[dict]:
    """Normalize provider-specific chain-of-thought fields without inference."""

    participant = "user" if message.get("role") == "user" else "agent"
    found = _reasoning_candidates(message)
    if not found:
        return [ReasoningBlock(participant=participant).model_dump()]
    return [
        ReasoningBlock(
            participant=participant, text=text, source=source, available=True
        ).model_dump()
        for source, text in found
    ]


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("JSON Patch path must start with '/'")
    return [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]


def apply_patch(document: Any, operations: list[PatchOperation]) -> Any:
    """Apply the supported RFC 6902 operations to a copy of a JSON value."""

    result = copy.deepcopy(document)
    for operation in operations:
        parts = _pointer_parts(operation.path)
        if not parts:
            if operation.op == "remove":
                raise ValueError("The artifact root cannot be removed")
            result = copy.deepcopy(operation.value)
            continue
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        leaf = parts[-1]
        if isinstance(parent, list):
            if operation.op == "add" and leaf == "-":
                parent.append(copy.deepcopy(operation.value))
            else:
                index = int(leaf)
                if operation.op == "add":
                    if not 0 <= index <= len(parent):
                        raise ValueError("JSON Patch list index is out of range")
                    parent.insert(index, copy.deepcopy(operation.value))
                elif not 0 <= index < len(parent):
                    raise ValueError("JSON Patch list index is out of range")
                elif operation.op == "remove":
                    parent.pop(index)
                else:
                    parent[index] = copy.deepcopy(operation.value)
        elif operation.op == "remove":
            if leaf not in parent:
                raise ValueError("JSON Patch remove target does not exist")
            del parent[leaf]
        else:
            if operation.op == "replace" and leaf not in parent:
                raise ValueError("JSON Patch replace target does not exist")
            parent[leaf] = copy.deepcopy(operation.value)
    return result


class ArtifactStore:
    """Indexes supported artifacts and performs guarded calibration writes."""

    def __init__(
        self,
        data_dir: Path,
        refresh_seconds: float = 5,
        background_refresh: bool = False,
    ):
        self.data_dir = data_dir.resolve()
        self.refresh_seconds = refresh_seconds
        self.calibration_dir = self.data_dir / "calibrations"
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._background_refresh = background_refresh
        self._refresh_thread: threading.Thread | None = None
        self._last_refresh = 0.0
        self._index_state = "starting"
        self._index_started_at: str | None = None
        self._index_completed_at: str | None = None
        self._index_error: str | None = None
        self._index_revision = 0
        self._json_cache: dict[Path, tuple[int, int, Any]] = {}
        self.runs: dict[str, dict] = {}
        self.bundles: dict[str, dict] = {}
        self.synthetic_packages: dict[str, dict] = {}
        self.trajectories: dict[str, dict] = {}
        self.synthesis_rounds: dict[str, dict] = {}
        self._synthesis_task_cache: dict[str, tuple[Any, dict[str, dict]]] = {}
        self._synthesis_offset_cache: dict[
            Path, tuple[int, int, list[tuple[int, int]]]
        ] = {}
        self.documents: dict[str, dict] = {}
        self._document_paths: dict[str, Path] = {}
        self._document_signatures: dict[Path, tuple[int, int]] = {}
        self._change_sets: dict[str, ChangeSet] = {}
        self._manifest_paths = self._load_manifest_catalog()
        self._last_manifest_discovery = (
            time.monotonic() if self._manifest_paths else 0.0
        )
        self.refresh(force=True)

    @property
    def fixed_attribution_dir(self) -> Path:
        """Root for read-only, task-scoped failure attribution files."""

        return self.calibration_dir / "failure-attributions"

    def _read_cached(self, path: Path) -> Any:
        """Return read-only JSON cached by mtime and size."""

        resolved = path.resolve()
        stat = resolved.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._json_cache.get(resolved)
        if cached and cached[:2] == signature:
            return cached[2]
        value = read_json_retry(resolved)
        final_stat = resolved.stat()
        self._json_cache[resolved] = (
            final_stat.st_mtime_ns,
            final_stat.st_size,
            value,
        )
        return value

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.data_dir):
            raise ValueError("Artifact resolves outside the configured data directory")
        return resolved.relative_to(self.data_dir).as_posix()

    def index_status(self) -> dict[str, Any]:
        """Return non-blocking metadata about the background index."""

        with self._lock:
            return {
                "status": self._index_state,
                "started_at": self._index_started_at,
                "completed_at": self._index_completed_at,
                "error": self._index_error,
                "revision": self._index_revision,
                "runs": len(self.runs),
                "bundles": len(self.bundles),
                "synthetic_packages": len(self.synthetic_packages),
                "trajectories": len(self.trajectories),
                "synthesis_rounds": len(self.synthesis_rounds),
                "documents": len(self.documents),
            }

    def refresh(self, force: bool = False) -> None:
        """Refresh indexes, scheduling non-blocking work in web-server mode."""

        refresh_interval = (
            max(300.0, self.refresh_seconds)
            if self._background_refresh
            else self.refresh_seconds
        )
        if not force and time.monotonic() - self._last_refresh < refresh_interval:
            return
        if not self._background_refresh:
            self._refresh_now(force=force)
            return
        with self._lock:
            if self._refresh_thread and self._refresh_thread.is_alive():
                return
            self._index_state = "indexing"
            self._index_started_at = utc_now()
            self._index_error = None
            thread = threading.Thread(
                target=self._refresh_worker,
                args=(force,),
                daemon=True,
                name="tau3-web-index",
            )
            self._refresh_thread = thread
            thread.start()

    def _refresh_worker(self, force: bool) -> None:
        try:
            self._refresh_now(force=force)
        except Exception as exc:
            with self._lock:
                self._index_state = "failed"
                self._index_error = f"{type(exc).__name__}: {exc}"

    def _refresh_now(self, force: bool = False) -> None:
        with self._refresh_lock:
            if (
                not force
                and time.monotonic() - self._last_refresh < self.refresh_seconds
            ):
                return
            with self._lock:
                self._index_state = "indexing"
                self._index_started_at = utc_now()
                self._index_error = None
            try:
                self._scan_runs()
                self._scan_bundles()
                self._scan_synthetic_packages()
                self._scan_synthesis_rounds()
                self._scan_documents()
                self._load_change_sets()
            except Exception as exc:
                with self._lock:
                    self._index_state = "failed"
                    self._index_error = f"{type(exc).__name__}: {exc}"
                raise
            with self._lock:
                self._last_refresh = time.monotonic()
                self._index_revision += 1
                self._index_state = "ready"
                self._index_completed_at = utc_now()

    _DISCOVERY_SKIP_PARTS = frozenset(
        {
            "audit",
            "business",
            "calls",
            "candidates",
            "documents",
            "llm_calls",
            "llm_pool",
            "physical_calls",
            "requests",
            "teacher",
            "trajectories",
            "world",
        }
    )

    def _manifest_catalog_path(self) -> Path:
        return self.calibration_dir / "web-manifest-index.json"

    def _supported_manifest(self, path: Path, base: Path) -> bool:
        try:
            candidate = path if path.is_absolute() else (Path.cwd() / path)
            relative = candidate.relative_to(base)
        except ValueError:
            return False
        return (
            relative.name == "manifest.json"
            and not self._DISCOVERY_SKIP_PARTS.intersection(relative.parts[:-1])
        )

    def _load_manifest_catalog(self) -> list[Path]:
        base = self.data_dir / "synthetic"
        path = self._manifest_catalog_path()
        if not path.exists() or not base.exists():
            return []
        try:
            payload = read_json_retry(path)
            values = payload.get("manifests") if isinstance(payload, dict) else []
            result = []
            for value in values or []:
                candidate = base / str(value)
                if self._supported_manifest(candidate, base):
                    result.append(candidate)
            return sorted(set(result))
        except Exception:
            return []

    def _save_manifest_catalog(self, base: Path, paths: list[Path]) -> None:
        write_json_atomic(
            self._manifest_catalog_path(),
            {
                "updated_at": utc_now(),
                "manifests": [path.relative_to(base).as_posix() for path in paths],
            },
        )

    def _discover_manifests(self, base: Path) -> list[Path]:
        """Find bundle manifests without Python-walking large LLM call trees."""

        ripgrep = shutil.which("rg")
        if ripgrep:
            try:
                completed = subprocess.run(
                    [
                        ripgrep,
                        "--files",
                        "--hidden",
                        "--no-ignore",
                        "--glob",
                        "manifest.json",
                        *[
                            value
                            for part in sorted(self._DISCOVERY_SKIP_PARTS)
                            for value in ("--glob", f"!**/{part}/**")
                        ],
                        str(base),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if completed.returncode in {0, 1}:
                    paths = [Path(line) for line in completed.stdout.splitlines()]
                    return sorted(
                        {path for path in paths if self._supported_manifest(path, base)}
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass

        result = []
        for current, directory_names, filenames in os.walk(base):
            directory_names[:] = [
                name
                for name in directory_names
                if name not in self._DISCOVERY_SKIP_PARTS
            ]
            if "manifest.json" in filenames:
                result.append(Path(current) / "manifest.json")
                directory_names.clear()
        return sorted(result)

    def _scan_runs(self) -> None:
        runs: dict[str, dict] = {}
        base = self.data_dir / "simulations"
        if not base.exists():
            self.runs = runs
            return
        result_paths = []
        for current, directory_names, filenames in os.walk(base):
            matches = [
                name
                for name in filenames
                if name.startswith("results") and name.endswith(".json")
            ]
            result_paths.extend(Path(current) / name for name in matches)
            if matches:
                directory_names.clear()
        for path in sorted(result_paths):
            rel = self._relative(path)
            run_id = opaque_id("run", rel)
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size)
            previous = self.runs.get(run_id)
            if previous and previous.get("signature") == signature:
                runs[run_id] = previous
                continue
            try:
                raw = self._read_cached(path)
                info = raw.get("info", {})
                simulations = raw.get("simulations") or []
                index = raw.get("simulation_index") or []
                summaries = []
                for sim in simulations or index:
                    reward_info = sim.get("reward_info") or {}
                    reward = sim.get("reward", reward_info.get("reward"))
                    summaries.append(
                        {
                            "id": sim.get("id"),
                            "task_id": str(sim.get("task_id")),
                            "trial": sim.get("trial"),
                            "reward": reward,
                            "termination_reason": sim.get("termination_reason"),
                            "duration": sim.get("duration"),
                            "has_review": bool(
                                sim.get("review") or sim.get("user_only_review")
                            ),
                        }
                    )
                runs[run_id] = {
                    "id": run_id,
                    "name": path.parent.name,
                    "file": path.name,
                    "path": path,
                    "relative_path": rel,
                    "mtime": stat.st_mtime,
                    "signature": signature,
                    "timestamp": raw.get("timestamp"),
                    "domain": (info.get("environment_info") or {}).get("domain_name"),
                    "task_bundle": info.get("task_bundle"),
                    "agent_model": (info.get("agent_info") or {}).get("llm"),
                    "user_model": (info.get("user_info") or {}).get("llm"),
                    "tasks": raw.get("tasks") or [],
                    "simulations": summaries,
                    "error": None,
                }
            except Exception as exc:
                runs[run_id] = {
                    "id": run_id,
                    "name": path.parent.name,
                    "file": path.name,
                    "path": path,
                    "relative_path": rel,
                    "mtime": stat.st_mtime,
                    "signature": signature,
                    "simulations": [],
                    "tasks": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        self.runs = runs

    def _scan_bundles(self) -> None:
        bundles: dict[str, dict] = {}
        trajectories: dict[str, dict] = {
            key: value
            for key, value in self.trajectories.items()
            if value.get("package_id")
        }
        base = self.data_dir / "synthetic"
        if not base.exists():
            self.bundles, self.trajectories = bundles, trajectories
            return
        current_environment_hash = environment_fingerprint()
        discovery_interval = max(300.0, self.refresh_seconds)
        if (
            not self._manifest_paths
            or time.monotonic() - self._last_manifest_discovery >= discovery_interval
        ):
            manifest_paths = self._discover_manifests(base)
            self._manifest_paths = manifest_paths
            self._last_manifest_discovery = time.monotonic()
            self._save_manifest_catalog(base, manifest_paths)
        else:
            manifest_paths = [path for path in self._manifest_paths if path.exists()]
        for manifest_path in sorted(manifest_paths):
            root = manifest_path.parent
            rel = self._relative(root)
            bundle_id = opaque_id("bundle", rel)
            tracked = [
                manifest_path,
                root / "tasks.json",
                root / "split_tasks.json",
                root / "candidates",
            ]
            try:
                signature = tuple(
                    (
                        path.relative_to(root).as_posix(),
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                    )
                    for path in tracked
                    if path.exists()
                )
            except OSError:
                signature = ()
            previous = self.bundles.get(bundle_id)
            if previous and previous.get("signature") == signature:
                bundles[bundle_id] = previous | {
                    "stale": previous["manifest"].get("environment_hash")
                    != current_environment_hash
                }
                trajectories.update(
                    {
                        key: value
                        for key, value in self.trajectories.items()
                        if value["bundle_id"] == bundle_id
                    }
                )
                continue
            try:
                manifest = read_json_retry(manifest_path)
                tasks_path = root / "tasks.json"
                candidate_dir = root / "candidates"
                if (
                    not tasks_path.exists()
                    and not candidate_dir.exists()
                    and manifest.get("benchmark") != "tau3-AA"
                ):
                    continue
                records: dict[str, dict] = {}
                candidate_paths: dict[str, Path] = {}
                if tasks_path.exists():
                    for task in read_json_retry(tasks_path):
                        records[str(task["id"])] = {
                            "task": task,
                            "candidate": None,
                            "source": "published",
                        }
                if candidate_dir.exists():
                    for candidate_path in sorted(candidate_dir.glob("*.json")):
                        try:
                            candidate = read_json_retry(candidate_path)
                        except Exception:
                            continue
                        task = candidate.get("task") or {}
                        task_id = str(task.get("id", ""))
                        if not task_id:
                            continue
                        candidate_paths[task_id] = candidate_path
                        record = records.setdefault(
                            task_id,
                            {
                                "task": task,
                                "candidate": candidate,
                                "source": "candidate",
                            },
                        )
                        record["candidate"] = candidate
                        for trial_key, trial in (candidate.get("trials") or {}).items():
                            trajectory_rel = trial.get("trajectory")
                            if not trajectory_rel:
                                continue
                            trajectory_path = (root / trajectory_rel).resolve()
                            if not trajectory_path.is_relative_to(root.resolve()):
                                continue
                            trajectory_id = opaque_id(
                                "trajectory", f"{rel}/{trajectory_rel}"
                            )
                            trajectories[trajectory_id] = {
                                "id": trajectory_id,
                                "bundle_id": bundle_id,
                                "task_id": task_id,
                                "candidate_id": candidate_path.stem,
                                "trial": trial_key,
                                "path": trajectory_path,
                                "relative_path": trajectory_rel,
                                "summary": trial,
                            }
                splits = {}
                split_path = root / "split_tasks.json"
                if split_path.exists():
                    try:
                        splits = read_json_retry(split_path)
                    except Exception:
                        pass
                split_by_task: dict[str, list[str]] = {}
                for name, ids in splits.items():
                    for task_id in ids:
                        split_by_task.setdefault(str(task_id), []).append(name)
                for task_id, record in records.items():
                    record["splits"] = split_by_task.get(task_id, [])
                bundles[bundle_id] = {
                    "id": bundle_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": rel,
                    "manifest": manifest,
                    "tasks": records,
                    "candidate_paths": candidate_paths,
                    "stale": manifest.get("environment_hash")
                    != current_environment_hash,
                    "error": None,
                    "signature": signature,
                }
            except Exception as exc:
                bundles[bundle_id] = {
                    "id": bundle_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": rel,
                    "manifest": {},
                    "tasks": {},
                    "candidate_paths": {},
                    "stale": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "signature": signature,
                }
        self.bundles, self.trajectories = bundles, trajectories

    @staticmethod
    def _synthetic_package_format(root: Path, manifest: dict) -> str | None:
        """Classify a manifest-backed synthetic artifact by positive markers."""

        if manifest.get("format") or (
            "schema_document" in manifest and "files" in manifest
        ):
            return None
        if manifest.get("benchmark") == "tau3-AA" or (root / "candidates").is_dir():
            return "tau3_aa"
        if (root / "tasks").is_dir() or (
            (root / "spec.json").exists() and (root / "build.json").exists()
        ):
            return "world_package"
        if (root / "tasks.json").exists():
            return "task_batch"
        if (root / "admitted").is_dir() or (
            manifest.get("bundle_id") and "task-batch" in root.name
        ):
            return "streaming_batch"
        return None

    @staticmethod
    def _split_index(root: Path) -> dict[str, list[str]]:
        """Return task-to-split membership for either supported split filename."""

        for name in ("split_tasks.json", "splits.json"):
            path = root / name
            if not path.exists():
                continue
            try:
                raw = read_json_retry(path)
            except Exception:
                return {}
            result: dict[str, list[str]] = {}
            if not isinstance(raw, dict):
                return result
            for split, values in raw.items():
                if not isinstance(values, list):
                    continue
                for task_id in values:
                    result.setdefault(str(task_id), []).append(str(split))
            return result
        return {}

    @staticmethod
    def _package_validation(root: Path, manifest: dict) -> dict[str, Any]:
        """Summarize package-level validation without treating it as a trajectory."""

        files = []
        status = None
        for relative in (
            "validation/certificate.json",
            "validation/report.json",
            "certificate.json",
            "report.json",
        ):
            path = root / relative
            if not path.exists():
                continue
            files.append(relative)
            if status is None:
                try:
                    payload = read_json_retry(path)
                    if isinstance(payload, dict):
                        status = payload.get("status")
                except Exception:
                    pass
        assurance = manifest.get("assurance")
        assurance_status = (
            assurance.get("status") if isinstance(assurance, dict) else assurance
        )
        return {
            "status": status or assurance_status or manifest.get("validation_status"),
            "files": files,
        }

    def _scan_synthetic_packages(self) -> None:
        """Normalize heterogeneous synthetic artifacts into package/task records."""

        packages: dict[str, dict] = {}
        bundle_by_root = {bundle["root"]: bundle for bundle in self.bundles.values()}
        for manifest_path in self._manifest_paths:
            root = manifest_path.parent
            try:
                manifest = read_json_retry(manifest_path)
                package_format = self._synthetic_package_format(root, manifest)
                if package_format is None:
                    continue
                relative_path = self._relative(root)
                package_id = opaque_id("package", relative_path)
                split_by_task = self._split_index(root)
                source_bundle = bundle_by_root.get(root)
                records: dict[str, dict] = {}
                if source_bundle is not None:
                    for task_id, record in source_bundle["tasks"].items():
                        records[task_id] = {
                            **record,
                            "task_path": None,
                            "splits": record.get(
                                "splits", split_by_task.get(task_id, [])
                            ),
                        }
                elif package_format == "world_package":
                    task_directory = root / "tasks"
                    if task_directory.exists():
                        for path in sorted(task_directory.glob("*.json")):
                            task_id = path.stem
                            records[task_id] = {
                                "task": None,
                                "task_path": path,
                                "candidate": None,
                                "source": "world_task",
                                "splits": split_by_task.get(task_id, []),
                            }
                elif package_format == "streaming_batch":
                    admitted_directory = root / "admitted"
                    if admitted_directory.exists():
                        for path in sorted(admitted_directory.glob("*.json")):
                            try:
                                admitted = read_json_retry(path)
                                task = admitted.get("task") or {}
                                task_id = str(task.get("id") or "")
                                if task_id:
                                    records[task_id] = {
                                        "task": task,
                                        "task_path": path,
                                        "candidate": admitted,
                                        "source": "admitted",
                                        "splits": split_by_task.get(task_id, []),
                                    }
                            except Exception:
                                continue
                packages[package_id] = {
                    "id": package_id,
                    "name": manifest.get("bundle_id") or root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": package_format,
                    "status": manifest.get("status") or "unknown",
                    "domain": manifest.get("domain"),
                    "manifest": manifest,
                    "tasks": records,
                    "source_bundle_id": source_bundle["id"] if source_bundle else None,
                    "editable": package_format == "tau3_aa",
                    "validation": self._package_validation(root, manifest),
                    "error": None,
                }
            except Exception as exc:
                relative_path = self._relative(root)
                package_id = opaque_id("package", relative_path)
                packages[package_id] = {
                    "id": package_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": "unknown",
                    "status": "unavailable",
                    "domain": None,
                    "manifest": {},
                    "tasks": {},
                    "source_bundle_id": None,
                    "editable": False,
                    "validation": {"status": None, "files": []},
                    "error": f"{type(exc).__name__}: {exc}",
                }
        self._scan_targeted_rounds(packages)
        self._scan_training_datasets(packages)
        loaded_package_ids = {
            package_id
            for package_id, package in packages.items()
            if package.get("targeted_loaded") or package.get("training_loaded")
        }
        self.trajectories = {
            trajectory_id: trajectory
            for trajectory_id, trajectory in self.trajectories.items()
            if not trajectory.get("package_id")
            or trajectory.get("package_id") in loaded_package_ids
        }
        self.synthetic_packages = packages

    def _scan_training_datasets(self, packages: dict[str, dict]) -> None:
        """Discover line-oriented training deliveries without reading their rows."""

        base = self.data_dir / "synthetic"
        if not base.exists():
            return
        required = (
            "report.json",
            "balanced.sft.jsonl",
            "balanced.general_agent.jsonl",
            "sampling.jsonl",
        )
        for root in sorted(path for path in base.iterdir() if path.is_dir()):
            if not all((root / name).is_file() for name in required):
                continue
            relative_path = self._relative(root)
            package_id = opaque_id("package", relative_path)
            signature = self._training_dataset_signature(root)
            previous = self.synthetic_packages.get(package_id)
            if (
                previous
                and previous.get("training_loaded")
                and previous.get("training_signature") == signature
            ):
                packages[package_id] = previous
                continue
            try:
                report = read_json_retry(root / "report.json")
                status = str(report.get("status") or "unknown")
                packages[package_id] = {
                    "id": package_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": "training_dataset",
                    "status": status,
                    "domain": "banking_synth",
                    "manifest": {"status": status, "report": report},
                    "tasks": {},
                    "source_bundle_id": None,
                    "editable": False,
                    "validation": {
                        "status": status,
                        "files": ["report.json", "delivery_check.json"],
                        "report": report,
                    },
                    "trajectory_count": sum(
                        int(report.get(key) or 0)
                        for key in ("rows", "focus_rows", "balanced_rows")
                    ),
                    "task_count": int(report.get("worlds") or 0),
                    "training_loaded": False,
                    "training_signature": signature,
                    "training_summary": {
                        "full": int(report.get("rows") or 0),
                        "focus": int(report.get("focus_rows") or 0),
                        "balanced": int(report.get("balanced_rows") or 0),
                        "unique_tasks": int(report.get("worlds") or 0),
                    },
                    "error": None,
                }
            except Exception as exc:
                packages[package_id] = {
                    "id": package_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": "training_dataset",
                    "status": "unavailable",
                    "domain": "banking_synth",
                    "manifest": {},
                    "tasks": {},
                    "source_bundle_id": None,
                    "editable": False,
                    "validation": {"status": None, "files": []},
                    "trajectory_count": 0,
                    "task_count": 0,
                    "training_loaded": False,
                    "training_signature": signature,
                    "training_summary": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def _scan_targeted_rounds(self, packages: dict[str, dict]) -> None:
        """Index round/phase/slot synthesis output without walking audit trees."""

        base = self.data_dir / "synthetic"
        if not base.exists():
            return
        roots = [
            root
            for root in sorted(path for path in base.iterdir() if path.is_dir())
            if (root / "config.json").exists()
            and (root / "plan.json").exists()
            and any((root / phase / "slots").is_dir() for phase in ("formal", "pilot"))
        ]
        for root in roots:
            relative_path = self._relative(root)
            package_id = opaque_id("package", relative_path)
            signature = self._targeted_round_signature([root])
            previous = self.synthetic_packages.get(package_id)
            if (
                previous
                and previous.get("targeted_loaded")
                and previous.get("targeted_signature") == signature
            ):
                packages[package_id] = previous
                continue
            records: dict[str, dict] = {}
            trajectory_count = 0
            errors: list[str] = []
            reports: dict[str, dict] = {}
            try:
                run_state_path = root / "run-state.json"
                run_state = (
                    read_json_retry(run_state_path) if run_state_path.exists() else {}
                )
                for phase in ("pilot", "formal"):
                    report_path = root / phase / "report.json"
                    if report_path.exists():
                        try:
                            reports[phase] = read_json_retry(report_path)
                        except Exception as exc:
                            errors.append(f"{phase}/report.json: {type(exc).__name__}")
                formal_report = reports.get("formal") or {}
                pilot_report = reports.get("pilot") or {}
                status = (
                    run_state.get("status")
                    or formal_report.get("status")
                    or pilot_report.get("status")
                    or "unknown"
                )
                packages[package_id] = {
                    "id": package_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": "targeted_round",
                    "status": status,
                    "domain": "banking_synth",
                    "manifest": {
                        "status": status,
                        "domain": "banking_synth",
                        "run_state": run_state,
                    },
                    "tasks": records,
                    "source_bundle_id": None,
                    "editable": False,
                    "validation": {
                        "status": formal_report.get("status")
                        or pilot_report.get("status"),
                        "files": [f"{phase}/report.json" for phase in reports],
                        "reports": reports,
                    },
                    "trajectory_count": trajectory_count,
                    "task_count": sum(
                        int(report.get("valid_tasks") or 0)
                        for report in reports.values()
                    ),
                    "targeted_loaded": False,
                    "targeted_signature": signature,
                    "error": "; ".join(errors[:10]) if errors else None,
                }
            except Exception as exc:
                packages[package_id] = {
                    "id": package_id,
                    "name": root.name,
                    "root": root,
                    "relative_path": relative_path,
                    "format": "targeted_round",
                    "status": "unavailable",
                    "domain": "banking_synth",
                    "manifest": {},
                    "tasks": records,
                    "source_bundle_id": None,
                    "editable": False,
                    "validation": {"status": None, "files": []},
                    "trajectory_count": trajectory_count,
                    "task_count": 0,
                    "targeted_loaded": False,
                    "targeted_signature": signature,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def _ensure_targeted_package(self, package: dict) -> None:
        """Load one targeted round's task and trajectory index on first access."""

        if package.get("targeted_loaded"):
            return
        root = package["root"]
        artifact_paths = self._targeted_round_artifact_paths([root])[root]
        records: dict[str, dict] = {}
        task_by_slot: dict[tuple[str, str], str] = {}
        for task_path in artifact_paths["tasks"]:
            parts = task_path.relative_to(root).parts
            if len(parts) != 7:
                continue
            phase, _, slot_id, candidate_name, _, _, _ = parts
            if phase not in {"pilot", "formal"} or not candidate_name.startswith(
                "candidate_"
            ):
                continue
            task_id = task_path.stem
            candidate_id = candidate_name.removeprefix("candidate_")
            records[task_id] = {
                "task": None,
                "task_path": task_path,
                "candidate": {"status": "indexed", "accepted": None},
                "source": "targeted_round",
                "splits": [phase],
                "stage": phase,
                "slot_id": slot_id,
                "candidate_id": candidate_id,
                "index_path": root / phase / "slots" / slot_id / "task.json",
                "trajectory_ids": [],
            }
            task_by_slot[(phase, slot_id)] = task_id
        trajectory_count = 0
        for result_path in artifact_paths["results"]:
            parts = result_path.relative_to(root).parts
            if len(parts) != 6:
                continue
            phase, _, slot_id, teacher, trial_number, _ = parts
            task_id = task_by_slot.get((phase, slot_id))
            if teacher != "teacher" or task_id is None:
                continue
            record = records[task_id]
            trajectory_rel = result_path.relative_to(self.data_dir).as_posix()
            trajectory_id = opaque_id("trajectory", trajectory_rel)
            self.trajectories[trajectory_id] = {
                "id": trajectory_id,
                "bundle_id": None,
                "package_id": package["id"],
                "task_id": task_id,
                "candidate_id": record["candidate_id"],
                "trial": trial_number,
                "path": result_path,
                "relative_path": trajectory_rel,
                "summary": {"status": "indexed"},
            }
            record["trajectory_ids"].append(trajectory_id)
            trajectory_count += 1
        package["tasks"] = records
        package["task_count"] = len(records)
        package["trajectory_count"] = trajectory_count
        package["targeted_loaded"] = True
        self._index_revision += 1

    @staticmethod
    def _training_dataset_signature(root: Path) -> list[list[Any]]:
        """Return cheap invalidation metadata for one training delivery."""

        names = (
            "report.json",
            "audit.jsonl",
            "sft.jsonl",
            "general_agent.jsonl",
            "focus.sft.jsonl",
            "focus.general_agent.jsonl",
            "balanced.sft.jsonl",
            "balanced.general_agent.jsonl",
            "sampling.jsonl",
        )
        signature = []
        for name in names:
            path = root / name
            if not path.exists():
                continue
            stat = path.stat()
            signature.append([name, stat.st_mtime_ns, stat.st_size])
        return signature

    def _training_index_path(self, root: Path) -> Path:
        relative = self._relative(root)
        suffix = hashlib.sha256(relative.encode()).hexdigest()[:16]
        return self.calibration_dir / f"web-training-index-{suffix}.json"

    @staticmethod
    def _jsonl_lines(path: Path):
        """Yield byte offsets and complete rows from a JSONL file."""

        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    return
                line_number += 1
                if raw.strip():
                    yield line_number, offset, len(raw), raw

    @staticmethod
    def _task_id_from_sft_row(raw: bytes) -> str:
        match = re.search(rb'"task_id"\s*:\s*"([^"\\]+)"', raw)
        if not match:
            raise ValueError("Training SFT row has no task_id")
        return match.group(1).decode("utf-8")

    def _build_training_dataset_index(
        self, root: Path, signature: list[list[Any]]
    ) -> dict[str, Any]:
        """Build a compact random-access catalog without retaining row payloads."""

        variants: dict[str, list[dict[str, Any]]] = {}
        for variant in ("full", "focus", "balanced"):
            stem = "" if variant == "full" else f"{variant}."
            sft_path = root / f"{stem}sft.jsonl"
            general_path = root / f"{stem}general_agent.jsonl"
            if not sft_path.exists() or not general_path.exists():
                continue
            sampling_rows = None
            if variant == "balanced":
                sampling_rows = self._jsonl_lines(root / "sampling.jsonl")
            records = []
            sft_rows = self._jsonl_lines(sft_path)
            general_rows = self._jsonl_lines(general_path)
            while True:
                sft_item = next(sft_rows, None)
                general_item = next(general_rows, None)
                sampling_item = (
                    next(sampling_rows, None) if sampling_rows is not None else None
                )
                if sft_item is None and general_item is None:
                    if sampling_item is not None:
                        raise ValueError(
                            "sampling.jsonl has more rows than balanced data"
                        )
                    break
                if sft_item is None or general_item is None:
                    raise ValueError(f"{variant} SFT/general-agent row counts differ")
                if variant == "balanced" and sampling_item is None:
                    raise ValueError("sampling.jsonl has fewer rows than balanced data")
                line, sft_offset, sft_length, sft_raw = sft_item
                general_line, general_offset, general_length, _ = general_item
                if general_line != line:
                    raise ValueError(f"{variant} row alignment is invalid")
                source = {}
                if sampling_item is not None:
                    sampling_line, _, _, sampling_raw = sampling_item
                    if sampling_line != line:
                        raise ValueError("Balanced sampling row alignment is invalid")
                    source = json.loads(sampling_raw)
                records.append(
                    {
                        "line": line,
                        "task_id": self._task_id_from_sft_row(sft_raw),
                        "sft_offset": sft_offset,
                        "sft_length": sft_length,
                        "general_offset": general_offset,
                        "general_length": general_length,
                        "source_kind": source.get("source_kind") or variant,
                        "slot": source.get("slot"),
                        "shard_line": source.get("shard_line"),
                        "sample_hash": source.get("sample_hash"),
                    }
                )
            variants[variant] = records

        audit_offsets: dict[str, list[list[int]]] = {}
        audit_path = root / "audit.jsonl"
        if audit_path.exists():
            for _, offset, length, raw in self._jsonl_lines(audit_path):
                audit = json.loads(raw)
                key = f"{audit.get('task_id')}\0{audit.get('trial')}"
                audit_offsets.setdefault(key, []).append([offset, length])
        result = {
            "schema_version": 1,
            "signature": signature,
            "variants": variants,
            "audit_offsets": audit_offsets,
        }
        write_json_atomic(self._training_index_path(root), result)
        return result

    def _training_dataset_index(self, package: dict) -> dict[str, Any]:
        path = self._training_index_path(package["root"])
        if path.exists():
            try:
                cached = read_json_retry(path)
                if cached.get("signature") == package["training_signature"]:
                    return cached
            except Exception:
                pass
        return self._build_training_dataset_index(
            package["root"], package["training_signature"]
        )

    def _ensure_training_dataset(self, package: dict) -> None:
        """Load logical training samples from a cached line-offset catalog."""

        if package.get("training_loaded"):
            return
        index = self._training_dataset_index(package)
        records: dict[str, dict[str, Any]] = {}
        trajectory_count = 0
        root = package["root"]
        for variant, rows in index.get("variants", {}).items():
            stem = "" if variant == "full" else f"{variant}."
            for row in rows:
                task_id = row["task_id"]
                record = records.setdefault(
                    task_id,
                    {
                        "task": None,
                        "task_path": None,
                        "candidate": None,
                        "source": "training_dataset",
                        "splits": [],
                        "trajectory_ids": [],
                        "sample_counts": Counter(),
                        "slots": set(),
                    },
                )
                if row.get("slot"):
                    record["slots"].add(str(row["slot"]))
                relative = f"{package['relative_path']}/{variant}:{int(row['line'])}"
                trajectory_id = opaque_id("trajectory", relative)
                entry = {
                    "id": trajectory_id,
                    "bundle_id": None,
                    "package_id": package["id"],
                    "task_id": task_id,
                    "candidate_id": variant,
                    "trial": int(row["line"]),
                    "path": root / f"{stem}sft.jsonl",
                    "general_path": root / f"{stem}general_agent.jsonl",
                    "relative_path": relative,
                    "format": "training_sample",
                    "dataset_variant": variant,
                    "source_kind": row.get("source_kind"),
                    "slot": row.get("slot"),
                    "shard_line": row.get("shard_line"),
                    "sample_hash": row.get("sample_hash"),
                    "sft_offset": int(row["sft_offset"]),
                    "sft_length": int(row["sft_length"]),
                    "general_offset": int(row["general_offset"]),
                    "general_length": int(row["general_length"]),
                    "summary": {
                        "status": "training",
                        "dataset_variant": variant,
                        "source_kind": row.get("source_kind"),
                    },
                }
                self.trajectories[trajectory_id] = entry
                record["trajectory_ids"].append(trajectory_id)
                record["sample_counts"][variant] += 1
                trajectory_count += 1
        for record in records.values():
            slots = sorted(record.pop("slots"))
            record["slot_id"] = slots[0] if len(slots) == 1 else None
            record["slot_ids"] = slots
            record["sample_counts"] = dict(record["sample_counts"])
            if len(slots) == 1:
                world_path = root / "shards" / slots[0] / "world_view.json"
                record["task_path"] = world_path if world_path.exists() else None
        package["tasks"] = records
        package["task_count"] = len(records)
        package["trajectory_count"] = trajectory_count
        package["training_index"] = index
        package["training_loaded"] = True
        self._index_revision += 1

    @staticmethod
    def _read_jsonl_at(path: Path, offset: int, length: int) -> dict[str, Any]:
        """Read and decode exactly one trusted, indexed JSONL row."""

        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(length)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Indexed JSONL row is not an object")
        return value

    def _targeted_round_signature(self, roots: list[Path]) -> list[list[Any]]:
        """Return cheap invalidation metadata for targeted round catalogs."""

        tracked_names = (
            "plan.json",
            "run-state.json",
            "pilot/report.json",
            "formal/report.json",
        )
        signature = []
        for root in roots:
            for name in tracked_names:
                path = root / name
                if path.exists():
                    stat = path.stat()
                    signature.append(
                        [self._relative(path), stat.st_mtime_ns, stat.st_size]
                    )
        return signature

    def _targeted_round_artifact_paths(
        self, roots: list[Path]
    ) -> dict[Path, dict[str, list[Path]]]:
        """List canonical task and teacher results for all rounds in one pass."""

        signature = self._targeted_round_signature(roots)
        catalog_key = hashlib.sha256(
            "\n".join(self._relative(root) for root in roots).encode()
        ).hexdigest()[:16]
        catalog_path = (
            self.calibration_dir / f"web-targeted-round-index-{catalog_key}.json"
        )
        if catalog_path.exists():
            try:
                cached = read_json_retry(catalog_path)
                if cached.get("signature") == signature:
                    result = {root: {"tasks": [], "results": []} for root in roots}
                    root_by_relative = {self._relative(root): root for root in roots}
                    for relative_root, values in (cached.get("rounds") or {}).items():
                        root = root_by_relative.get(relative_root)
                        if root is None:
                            continue
                        for kind in ("tasks", "results"):
                            result[root][kind] = [
                                self.data_dir / value for value in values.get(kind, [])
                            ]
                    return result
            except Exception:
                pass
        ripgrep = shutil.which("rg")
        result = {root: {"tasks": [], "results": []} for root in roots}

        def scan_root(root: Path) -> tuple[Path, list[Path]]:
            slot_roots = [
                root / phase / "slots"
                for phase in ("pilot", "formal")
                if (root / phase / "slots").is_dir()
            ]
            paths: list[Path] = []
            if ripgrep and slot_roots:
                try:
                    completed = subprocess.run(
                        [
                            ripgrep,
                            "--files",
                            *[str(path) for path in slot_roots],
                            "--glob",
                            "**/candidate_*/world/tasks/*.json",
                            "--glob",
                            "**/teacher/*/result.json",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if completed.returncode in {0, 1}:
                        paths = [Path(value) for value in completed.stdout.splitlines()]
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"Timed out indexing targeted round {root.name}"
                    ) from exc
                except OSError:
                    pass
            if not paths:
                for slots in slot_roots:
                    paths.extend(slots.glob("*/candidate_*/world/tasks/*.json"))
                    paths.extend(slots.glob("*/teacher/*/result.json"))
            return root, paths

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(roots)))) as executor:
            for root, paths in executor.map(scan_root, roots):
                for path in paths:
                    kind = "tasks" if "world/tasks" in path.as_posix() else "results"
                    result[root][kind].append(path)
        for values in result.values():
            values["tasks"].sort()
            values["results"].sort()
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            catalog_path,
            {
                "updated_at": utc_now(),
                "signature": signature,
                "rounds": {
                    self._relative(root): {
                        kind: [
                            path.relative_to(self.data_dir).as_posix()
                            for path in values[kind]
                        ]
                        for kind in ("tasks", "results")
                    }
                    for root, values in result.items()
                },
            },
        )
        return result

    def _package_task(self, package: dict, task_id: str) -> dict:
        """Load one normalized task definition on demand."""

        record = self._require(package["tasks"], task_id, "task")
        if record.get("task") is not None:
            return record["task"]
        path = record.get("task_path")
        if path is None:
            raise KeyError("task")
        payload = read_json_retry(path)
        if record.get("source") == "admitted":
            task = payload.get("task")
        elif record.get("source") == "training_dataset":
            tasks = payload.get("tasks") or {}
            task = tasks.get(f"{task_id}.json") or tasks.get(task_id)
        else:
            task = payload
        if not isinstance(task, dict):
            raise ValueError("Synthetic task artifact is not an object")
        actual_id = str(task.get("id") or task_id)
        if actual_id != task_id:
            raise ValueError("Synthetic task ID does not match its logical locator")
        record["task"] = task
        return task

    def _scan_synthesis_rounds(self) -> None:
        """Index only cheap, fixed-name metadata below data/synthesis."""

        base = self.data_dir / "synthesis"
        rounds: dict[str, dict] = {}
        if not base.is_dir():
            self.synthesis_rounds = rounds
            return
        active_root = None
        active_path = base / "active-native-round.json"
        if active_path.exists():
            try:
                active = self._read_cached(active_path)
                candidate = Path(str(active.get("round_dir", ""))).resolve()
                if candidate.is_relative_to(base.resolve()):
                    active_root = candidate
            except Exception:
                pass
        for root in sorted(path for path in base.iterdir() if path.is_dir()):
            marker_paths = [
                root / "config.json",
                root / "supervisor.json",
                root / "summary.json",
                root / "repair-manifest.json",
            ]
            if (
                not any(path.exists() for path in marker_paths)
                and not (root / "tasks").is_dir()
            ):
                continue
            round_id = opaque_id("synthesis_round", self._relative(root))
            config = {}
            supervisor = {}
            summary = {}
            for path, target in (
                (root / "config.json", config),
                (root / "supervisor.json", supervisor),
                (root / "summary.json", summary),
            ):
                if path.exists():
                    try:
                        value = self._read_cached(path)
                        if isinstance(value, dict):
                            target.update(value)
                    except Exception:
                        pass
            reports = {}
            for stage in ("pilot", "small", "validation"):
                path = root / stage / "report.json"
                if path.exists():
                    try:
                        value = self._read_cached(path)
                        if isinstance(value, dict):
                            reports[stage] = value
                    except Exception:
                        reports[stage] = {"status": "unavailable"}
            stat = root.stat()
            stage_value = supervisor.get("stage")
            if isinstance(stage_value, list):
                stage_value = ", ".join(str(value) for value in stage_value)
            status = (
                supervisor.get("status")
                or summary.get("status")
                or next(
                    (
                        report.get("status")
                        for report in reversed(list(reports.values()))
                        if report.get("status")
                    ),
                    "unknown",
                )
            )
            rounds[round_id] = {
                "id": round_id,
                "name": root.name,
                "root": root,
                "relative_path": self._relative(root),
                "active": root.resolve() == active_root,
                "status": str(status),
                "stage": stage_value,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                "parent": Path(str(config.get("parent_round"))).name
                if config.get("parent_round")
                else None,
                "backend": config.get("backend"),
                "configured": {
                    "pilot": config.get("pilot_tasks"),
                    "small": config.get("small_tasks"),
                    "validation": config.get("validation_tasks"),
                },
                "reports": reports,
                "has_tasks": (root / "tasks").is_dir(),
                "error": None,
            }
        self.synthesis_rounds = rounds

    @staticmethod
    def _synthesis_stage_roots(root: Path) -> list[tuple[str, Path]]:
        return [
            (stage, path)
            for stage, path in (
                ("pilot", root / "tasks" / "pilot"),
                ("small", root / "tasks" / "train"),
                ("validation", root / "tasks" / "validation"),
            )
            if path.is_dir()
        ]

    def _synthesis_training_rows(self, round_record: dict) -> list[dict[str, Any]]:
        """Load manifest metadata without opening the associated large JSONL files."""

        rows = []
        root = round_record["root"]
        for stage in ("pilot", "small", "validation"):
            stage_root = root / stage
            manifest_path = stage_root / "training_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = self._read_cached(manifest_path)
            except Exception:
                continue
            indexed = manifest.get("rows_index") or []
            audits = {
                item.get("row_id"): item
                for item in (manifest.get("token_audits") or [])
                if isinstance(item, dict) and item.get("row_id")
            }
            for line, item in enumerate(indexed, start=1):
                if not isinstance(item, dict) or not item.get("row_id"):
                    continue
                rows.append(
                    copy.deepcopy(item)
                    | {
                        "stage": stage,
                        "line": line,
                        "token_audit": audits.get(item["row_id"]),
                        "sft_path": stage_root / "sft.jsonl",
                        "general_path": stage_root / "general_agent_reasoning.jsonl",
                    }
                )
        return rows

    def _synthesis_tasks(self, round_record: dict) -> dict[str, dict]:
        """Build and cache the selected round's small task catalog."""

        roots = self._synthesis_stage_roots(round_record["root"])
        signature = []
        slot_roots = []
        for stage, root in roots:
            stat = root.stat()
            signature.append((stage, stat.st_mtime_ns, stat.st_size))
            for slot_root in sorted(path for path in root.iterdir() if path.is_dir()):
                slot_roots.append((stage, slot_root))
                for name in ("task.json", "candidate.json", "provenance.json"):
                    path = slot_root / name
                    if path.exists():
                        file_stat = path.stat()
                        signature.append(
                            (
                                f"{stage}:{slot_root.name}:{name}",
                                file_stat.st_mtime_ns,
                                file_stat.st_size,
                            )
                        )
        for stage in ("pilot", "small", "validation"):
            manifest_path = round_record["root"] / stage / "training_manifest.json"
            if manifest_path.exists():
                stat = manifest_path.stat()
                signature.append((f"{stage}:manifest", stat.st_mtime_ns, stat.st_size))
        cached = self._synthesis_task_cache.get(round_record["id"])
        if cached and cached[0] == signature:
            return cached[1]
        sample_counts = Counter(
            row.get("task_id") for row in self._synthesis_training_rows(round_record)
        )
        records: dict[str, dict] = {}
        for stage, slot_root in slot_roots:
            task_path = slot_root / "task.json"
            if not task_path.exists():
                continue
            try:
                task = self._read_cached(task_path)
                task_id = str(task["id"])
                provenance_path = slot_root / "provenance.json"
                candidate_path = slot_root / "candidate.json"
                provenance = (
                    self._read_cached(provenance_path)
                    if provenance_path.exists()
                    else {}
                )
                candidate = (
                    self._read_cached(candidate_path) if candidate_path.exists() else {}
                )
                slot = provenance.get("slot") or candidate.get("slot") or {}
                records[task_id] = {
                    "id": task_id,
                    "stage": stage,
                    "slot": slot_root.name,
                    "family": slot.get("family"),
                    "difficulty": slot.get("difficulty"),
                    "origin": slot.get("origin"),
                    "purpose": (task.get("description") or {}).get("purpose"),
                    "target_operations": slot.get("target_operations") or [],
                    "protocol_challenges": slot.get("protocol_challenges") or [],
                    "checks": candidate.get("checks") or {},
                    "sample_count": sample_counts.get(task_id, 0),
                    "task_path": task_path,
                    "provenance_path": provenance_path,
                    "candidate_path": candidate_path,
                    "error": None,
                }
            except Exception as exc:
                logical_id = f"{stage}:{slot_root.name}"
                records[logical_id] = {
                    "id": logical_id,
                    "stage": stage,
                    "slot": slot_root.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "sample_count": 0,
                }
        self._synthesis_task_cache[round_record["id"]] = (signature, records)
        return records

    def list_synthesis_rounds(
        self, query: str, page: int, page_size: int
    ) -> dict[str, Any]:
        """List temporary native synthesis rounds without scanning their payload trees."""

        self.refresh()
        needle = query.casefold()
        rows = []
        for item in self.synthesis_rounds.values():
            if needle and needle not in f"{item['name']} {item['status']}".casefold():
                continue
            reports = item["reports"]
            stage_counts = {
                stage: {
                    "valid_tasks": report.get("valid_tasks"),
                    "expected_tasks": report.get("expected_tasks"),
                    "qualified_rows": report.get("qualified_rows"),
                    "status": report.get("status"),
                }
                for stage, report in reports.items()
            }
            rows.append(
                {key: value for key, value in item.items() if key != "root"}
                | {"reports": stage_counts}
            )
        rows.sort(key=lambda item: item["updated_at"], reverse=True)
        result = paginate(rows, page, page_size)
        result["summary"] = {
            "rounds": len(self.synthesis_rounds),
            "active": sum(item["active"] for item in self.synthesis_rounds.values()),
            "with_tasks": sum(
                item["has_tasks"] for item in self.synthesis_rounds.values()
            ),
        }
        return result

    def list_synthesis_tasks(
        self,
        round_id: str,
        query: str,
        stage: str,
        family: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        """List one selected round's task samples with filters and pagination."""

        self.refresh()
        round_record = self._require(self.synthesis_rounds, round_id, "synthesis round")
        records = self._synthesis_tasks(round_record)
        needle = query.casefold()
        rows = []
        for record in records.values():
            if stage and record.get("stage") != stage:
                continue
            if family and record.get("family") != family:
                continue
            searchable = (
                f"{record['id']} {record.get('purpose', '')} {record.get('family', '')}"
            )
            if needle and needle not in searchable.casefold():
                continue
            rows.append(
                {
                    key: value
                    for key, value in record.items()
                    if not key.endswith("_path")
                }
            )
        rows.sort(key=lambda item: (item.get("stage", ""), item.get("slot", "")))
        result = paginate(rows, page, page_size)
        result["round"] = {
            key: value for key, value in round_record.items() if key not in {"root"}
        }
        result["families"] = sorted(
            {record["family"] for record in records.values() if record.get("family")}
        )
        result["stages"] = sorted(
            {record["stage"] for record in records.values() if record.get("stage")}
        )
        result["summary"] = {
            "tasks": len(records),
            "training_samples": sum(
                record.get("sample_count", 0) for record in records.values()
            ),
            "errors": sum(bool(record.get("error")) for record in records.values()),
        }
        return result

    def synthesis_task_detail(self, round_id: str, task_id: str) -> dict[str, Any]:
        """Return task, provenance, checks, and related exported training rows."""

        self.refresh()
        round_record = self._require(self.synthesis_rounds, round_id, "synthesis round")
        record = self._require(self._synthesis_tasks(round_record), task_id, "task")
        if record.get("error"):
            raise ValueError(record["error"])
        task = self._read_cached(record["task_path"])
        provenance = (
            self._read_cached(record["provenance_path"])
            if record["provenance_path"].exists()
            else None
        )
        candidate = (
            self._read_cached(record["candidate_path"])
            if record["candidate_path"].exists()
            else None
        )
        samples = [
            {
                key: value
                for key, value in row.items()
                if key not in {"sft_path", "general_path", "token_audit"}
            }
            | {"token_audit": row.get("token_audit")}
            for row in self._synthesis_training_rows(round_record)
            if str(row.get("task_id")) == task_id
        ]
        return {
            "round": {
                key: value for key, value in round_record.items() if key != "root"
            },
            "task": task,
            "candidate": candidate,
            "provenance": provenance,
            "stage": record["stage"],
            "slot": record["slot"],
            "samples": samples,
            "sha256": digest(task),
        }

    def _synthesis_jsonl_offsets(self, path: Path) -> list[tuple[int, int]]:
        resolved = path.resolve()
        stat = resolved.stat()
        cached = self._synthesis_offset_cache.get(resolved)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        offsets = [
            (offset, length) for _, offset, length, _ in self._jsonl_lines(resolved)
        ]
        self._synthesis_offset_cache[resolved] = (
            stat.st_mtime_ns,
            stat.st_size,
            offsets,
        )
        return offsets

    def synthesis_sample_detail(self, round_id: str, row_id: str) -> dict[str, Any]:
        """Read one manifest-bound SFT/general-agent row by cached byte offset."""

        self.refresh()
        round_record = self._require(self.synthesis_rounds, round_id, "synthesis round")
        matches = [
            row
            for row in self._synthesis_training_rows(round_record)
            if row.get("row_id") == row_id
        ]
        if len(matches) != 1:
            raise KeyError("training sample")
        row = matches[0]
        sft_path = row["sft_path"]
        general_path = row["general_path"]
        if not sft_path.exists() or not general_path.exists():
            raise ValueError("Training manifest points to a missing JSONL export")
        sft_offsets = self._synthesis_jsonl_offsets(sft_path)
        general_offsets = self._synthesis_jsonl_offsets(general_path)
        index = int(row["line"]) - 1
        if index >= len(sft_offsets) or index >= len(general_offsets):
            raise ValueError("Training manifest and JSONL row counts are not aligned")
        sft = self._read_jsonl_at(sft_path, *sft_offsets[index])
        general = self._read_jsonl_at(general_path, *general_offsets[index])
        task_id = str(row.get("task_id"))
        if str(sft.get("task_id")) != task_id:
            raise ValueError("Training manifest task_id does not match the SFT row")
        task_record = self._require(
            self._synthesis_tasks(round_record), task_id, "task"
        )
        task = self._read_cached(task_record["task_path"])
        loss_mask = sft.get("loss_mask") or []
        messages = []
        system_messages = []
        for message_index, message in enumerate(sft.get("messages") or []):
            if message.get("role") == "system":
                system_messages.append(message)
                continue
            item = copy.deepcopy(message)
            item["turn_idx"] = message_index
            item["training_loss"] = bool(
                loss_mask[message_index] if message_index < len(loss_mask) else 0
            )
            normalized_calls = []
            for call in item.get("tool_calls") or []:
                function = call.get("function") or {}
                normalized_calls.append(
                    {
                        "id": call.get("id"),
                        "name": function.get("name") or call.get("name"),
                        "arguments": self._tool_arguments(
                            function.get("arguments", call.get("arguments"))
                        ),
                        "requestor": "assistant",
                    }
                )
            if normalized_calls:
                item["tool_calls"] = normalized_calls
            messages.append(item)
        artifact_id = opaque_id("synthesis_sample", f"{round_id}:{row_id}")
        artifact = ArtifactRef(
            kind="trajectory",
            id=artifact_id,
            package_id=round_id,
            task_id=task_id,
            trial=row.get("seed"),
        )
        result = self._decorate_trajectory(
            {
                "id": artifact_id,
                "task_id": task_id,
                "trial": row.get("seed"),
                "messages": messages,
                "reward_info": {
                    "reward": None,
                    "info": {"note": "临时合成训练样本不包含评测 reward。"},
                },
                "termination_reason": "training_sample",
                "policy": system_messages[0].get("content")
                if system_messages
                else None,
                "info": {"training_sample": True},
            },
            task,
            artifact=artifact,
            run_info={
                "agent_info": {"llm": sft.get("teacher_model")},
                "retrieval_config": sft.get("retrieval_config"),
            },
            capture={"visible_sample": {"messages": system_messages}},
        )
        result["verifier_diagnostics"] = None
        result["training_sample"] = {
            "dataset_variant": row["stage"],
            "source_kind": "temporary_synthesis",
            "line": row["line"],
            "slot": task_record["slot"],
            "shard_line": None,
            "sample_hash": row.get("row_hash"),
            "schema_version": sft.get("schema_version"),
            "retrieval_config": sft.get("retrieval_config"),
            "metadata": sft.get("metadata") or {},
            "loss_mask": loss_mask,
            "audits": [row["token_audit"]] if row.get("token_audit") else [],
            "representations": {
                "sft": {
                    "row_id": row_id,
                    "weight": row.get("weight"),
                    "seed": row.get("seed"),
                    "teacher_model": sft.get("teacher_model"),
                    "metadata": sft.get("metadata") or {},
                    "tools": sft.get("tools") or [],
                },
                "general_agent": general,
            },
            "note": (
                "这是 data/synthesis 的临时训练样本，不代表一次带 reward 的正式评测。"
            ),
        }
        result["synthesis_round"] = {
            "id": round_id,
            "name": round_record["name"],
            "stage": row["stage"],
            "row_id": row_id,
        }
        return result

    def _scan_documents(self) -> None:
        documents: dict[str, dict] = {}
        paths: dict[str, Path] = {}
        signatures: dict[Path, tuple[int, int]] = {}
        previous_ids = {path: doc_id for doc_id, path in self._document_paths.items()}
        base = self.data_dir / "tau3" / "domains" / "banking_knowledge" / "documents"
        if base.exists():
            for path in sorted(base.glob("*.json")):
                try:
                    stat = path.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    previous_id = previous_ids.get(path)
                    if (
                        previous_id
                        and self._document_signatures.get(path) == signature
                        and previous_id in self.documents
                    ):
                        documents[previous_id] = self.documents[previous_id]
                        paths[previous_id] = path
                        signatures[path] = signature
                        continue
                    document = read_json_retry(path)
                    document_id = str(document["id"])
                    documents[document_id] = {
                        "id": document_id,
                        "title": document.get("title", ""),
                        "content": document.get("content", ""),
                        "sha256": digest(document),
                    }
                    paths[document_id] = path
                    signatures[path] = signature
                except Exception:
                    continue
        self.documents, self._document_paths = documents, paths
        self._document_signatures = signatures

    def _load_change_sets(self) -> None:
        loaded: dict[str, ChangeSet] = {}
        directory = self.calibration_dir / "change_sets"
        if directory.exists():
            for path in directory.glob("*.json"):
                try:
                    item = ChangeSet.model_validate(read_json_retry(path))
                    loaded[item.id] = item
                except Exception:
                    continue
        self._change_sets = loaded

    def overview(self) -> dict[str, Any]:
        self.refresh()
        run_sims = [s for run in self.runs.values() for s in run["simulations"]]
        synth_sims = list(self.trajectories.values())
        success = sum(s.get("reward") == 1 for s in run_sims)
        success += sum((s.get("summary") or {}).get("reward") == 1 for s in synth_sims)
        total_trajectories = len(run_sims) + len(synth_sims)
        failures: Counter[str] = Counter()
        for simulation in run_sims:
            if simulation.get("reward") != 1:
                failures[
                    simulation.get("termination_reason") or "evaluation_failed"
                ] += 1
        for entry in synth_sims:
            summary = entry.get("summary") or {}
            if summary.get("reward") != 1:
                reason = (
                    "infrastructure_error"
                    if summary.get("infrastructure_error")
                    else summary.get("termination") or "evaluation_failed"
                )
                failures[str(reason)] += 1
        db = self._load_db()
        return {
            "index": self.index_status(),
            "runs": len(self.runs),
            "bundles": len(self.bundles),
            "synthetic_packages": len(self.synthetic_packages),
            "synthesis_rounds": len(self.synthesis_rounds),
            "published_bundles": sum(
                b["manifest"].get("status") == "published"
                for b in self.bundles.values()
            ),
            "stale_bundles": sum(b.get("stale") is True for b in self.bundles.values()),
            "tasks": sum(
                package.get("task_count", len(package["tasks"]))
                for package in self.synthetic_packages.values()
            ),
            "trajectories": total_trajectories,
            "successful_trajectories": success,
            "failure_types": [
                {"type": name, "count": count} for name, count in failures.most_common()
            ],
            "documents": len(self.documents),
            "database_tables": len(db),
            "database_records": sum(
                len((table or {}).get("data", {})) for table in db.values()
            ),
            "updated_at": utc_now(),
        }

    def list_runs(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        self.refresh()
        needle = query.casefold()
        rows = []
        for run in self.runs.values():
            if (
                needle
                and needle not in f"{run['name']} {run.get('domain', '')}".casefold()
            ):
                continue
            simulations = run["simulations"]
            rows.append(
                {
                    k: run.get(k)
                    for k in (
                        "id",
                        "name",
                        "file",
                        "relative_path",
                        "timestamp",
                        "domain",
                        "task_bundle",
                        "agent_model",
                        "user_model",
                        "mtime",
                        "error",
                    )
                }
                | {
                    "simulation_count": len(simulations),
                    "success_count": sum(s.get("reward") == 1 for s in simulations),
                }
            )
        rows.sort(key=lambda item: (item.get("mtime") or 0), reverse=True)
        result = paginate(rows, page, page_size, sort_by, order)
        result["summary"] = {
            "runs": len(rows),
            "trajectories": sum(item["simulation_count"] for item in rows),
            "successful": sum(item["success_count"] for item in rows),
        }
        return result

    def list_synthetic_trajectories(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
        status: str = "",
        bundle_id: str = "",
        bundle_task: str = "",
        candidate_trial: str = "",
        termination: str = "",
        model: str = "",
        review_status: str = "",
        package_prefix: str = "",
        task_prefix: str = "",
        dataset_variant: str = "",
    ) -> dict[str, Any]:
        """List indexed synthesis trials without reading full trajectory files."""

        self.refresh()
        selected_package = self.synthetic_packages.get(bundle_id)
        if selected_package and selected_package["format"] == "targeted_round":
            self._ensure_targeted_package(selected_package)
        elif selected_package and selected_package["format"] == "training_dataset":
            self._ensure_training_dataset(selected_package)
        needle = query.casefold()
        bundle_task_needle = bundle_task.casefold()
        candidate_trial_needle = candidate_trial.casefold()
        termination_needle = termination.casefold()
        model_needle = model.casefold()
        package_prefix_needle = package_prefix.casefold()
        task_prefix_needle = task_prefix.casefold()
        dataset_variant_needle = dataset_variant.casefold()
        rows = []
        filter_options: dict[str, set[str]] = {
            "bundle_task": set(),
            "candidate_trial": set(),
            "termination": set(),
            "model": set(),
            "dataset_variant": set(),
        }
        for trajectory in self.trajectories.values():
            bundle = self.bundles.get(trajectory.get("bundle_id"))
            package = self.synthetic_packages.get(trajectory.get("package_id"))
            owner = bundle or package
            if owner is None:
                continue
            trial = trajectory.get("summary") or {}
            termination_reason = trial.get("termination_reason") or trial.get(
                "termination"
            )
            config = owner["manifest"].get("config") or {}
            trajectory_model = trial.get("model") or config.get("teacher_model")
            filter_options["bundle_task"].update((owner["name"], trajectory["task_id"]))
            filter_options["candidate_trial"].update(
                str(value)
                for value in (
                    trajectory.get("candidate_id"),
                    trajectory.get("trial"),
                )
                if value not in (None, "")
            )
            if termination_reason:
                filter_options["termination"].add(str(termination_reason))
            if trajectory_model:
                filter_options["model"].add(str(trajectory_model))
            if trajectory.get("dataset_variant"):
                filter_options["dataset_variant"].add(
                    str(trajectory["dataset_variant"])
                )
            owner_id = trajectory.get("bundle_id") or trajectory.get("package_id")
            owner_prefix = package_name_prefix(owner["name"])
            trajectory_task_prefix = task_id_prefix(trajectory["task_id"])
            if bundle_id and owner_id != bundle_id:
                continue
            if package_prefix_needle and owner_prefix != package_prefix_needle:
                continue
            if task_prefix_needle and trajectory_task_prefix != task_prefix_needle:
                continue
            if (
                dataset_variant_needle
                and str(trajectory.get("dataset_variant") or "").casefold()
                != dataset_variant_needle
            ):
                continue
            reward = trial.get("reward")
            if trajectory.get("format") == "training_sample":
                outcome = "training"
            elif trial.get("infrastructure_error"):
                outcome = "error"
            elif reward == 1:
                outcome = "success"
            elif reward is not None:
                outcome = "failed"
            else:
                outcome = "unknown"
            if status and outcome != status:
                continue
            trial_review = trial.get("review")
            has_review = bool(trial_review)
            if review_status == "reviewed" and not has_review:
                continue
            if review_status == "unreviewed" and has_review:
                continue
            bundle_task_text = (
                f"{owner['name']} {owner_id} {trajectory['task_id']}"
            ).casefold()
            if bundle_task_needle and bundle_task_needle not in bundle_task_text:
                continue
            candidate_trial_text = (
                f"{trajectory.get('candidate_id') or ''} "
                f"{trajectory.get('trial') or ''}"
            ).casefold()
            if (
                candidate_trial_needle
                and candidate_trial_needle not in candidate_trial_text
            ):
                continue
            if (
                termination_needle
                and termination_needle not in str(termination_reason or "").casefold()
            ):
                continue
            if (
                model_needle
                and model_needle not in str(trajectory_model or "").casefold()
            ):
                continue
            searchable = " ".join(
                str(value or "")
                for value in (
                    owner["name"],
                    owner_id,
                    trajectory["task_id"],
                    trajectory.get("candidate_id"),
                    trajectory.get("trial"),
                    termination_reason,
                    trajectory_model,
                )
            ).casefold()
            if needle and needle not in searchable:
                continue
            review_summary = None
            if isinstance(trial_review, str):
                review_summary = trial_review
            elif isinstance(trial_review, dict):
                review_summary = next(
                    (
                        trial_review.get(key)
                        for key in ("summary", "explanation", "reasoning", "verdict")
                        if trial_review.get(key)
                    ),
                    None,
                )
            rows.append(
                {
                    "id": trajectory["id"],
                    "bundle_id": owner_id,
                    "bundle_name": owner["name"],
                    "bundle_status": owner["manifest"].get("status"),
                    "package_prefix": owner_prefix,
                    "task_id": trajectory["task_id"],
                    "task_prefix": trajectory_task_prefix,
                    "candidate_id": trajectory.get("candidate_id"),
                    "trial": trajectory.get("trial"),
                    "reward": reward,
                    "status": outcome,
                    "termination_reason": termination_reason,
                    "model": trajectory_model,
                    "timestamp": trial.get("timestamp")
                    or trial.get("completed_at")
                    or trial.get("created_at"),
                    "has_review": has_review,
                    "review_summary": review_summary,
                    "artifact_type": trajectory.get("format")
                    or "evaluation_trajectory",
                    "dataset_variant": trajectory.get("dataset_variant"),
                    "source_kind": trajectory.get("source_kind"),
                    "sample_line": trajectory.get("trial")
                    if trajectory.get("format") == "training_sample"
                    else None,
                }
            )
        rows.sort(
            key=lambda item: (
                item["bundle_name"],
                item["task_id"],
                str(item.get("trial") or ""),
            )
        )
        result = paginate(rows, page, page_size, sort_by, order)
        result["summary"] = {
            "trajectories": len(rows),
            "successful": sum(item["status"] == "success" for item in rows),
            "failed": sum(item["status"] == "failed" for item in rows),
            "errors": sum(item["status"] == "error" for item in rows),
            "training": sum(item["status"] == "training" for item in rows),
        }
        owners = sorted(
            [
                *self.bundles.values(),
                *(
                    package
                    for package in self.synthetic_packages.values()
                    if not package.get("source_bundle_id")
                ),
            ],
            key=lambda item: item["name"],
        )
        trajectory_counts = Counter(
            trajectory.get("bundle_id") or trajectory.get("package_id")
            for trajectory in self.trajectories.values()
        )
        result["bundle_options"] = [
            {
                "id": owner["id"],
                "name": owner["name"],
                "prefix": package_name_prefix(owner["name"]),
                "format": owner.get("format") or "tau3_aa",
                "loaded": (
                    owner.get("format") not in {"targeted_round", "training_dataset"}
                    or bool(owner.get("targeted_loaded"))
                    or bool(owner.get("training_loaded"))
                ),
                "task_count": owner.get("task_count", len(owner.get("tasks") or {})),
                "trajectory_count": trajectory_counts[owner["id"]],
            }
            for owner in owners
        ]
        group_counts: dict[str, dict[str, Any]] = {}
        for owner in result["bundle_options"]:
            group = group_counts.setdefault(
                owner["prefix"],
                {
                    "prefix": owner["prefix"],
                    "packages": 0,
                    "tasks": 0,
                    "trajectories": 0,
                },
            )
            group["packages"] += 1
            group["tasks"] += owner["task_count"]
            group["trajectories"] += owner["trajectory_count"]
        result["package_groups"] = sorted(
            group_counts.values(), key=lambda item: item["prefix"]
        )
        result["task_prefixes"] = sorted(
            {item["task_prefix"] for item in rows}, key=str.casefold
        )
        result["matching_packages"] = [
            owner
            for owner in result["bundle_options"]
            if not needle or needle in f"{owner['name']} {owner['prefix']}".casefold()
        ]
        option_limit = 250
        option_needles = {
            "bundle_task": bundle_task_needle,
            "candidate_trial": candidate_trial_needle,
            "termination": termination_needle,
            "model": model_needle,
            "dataset_variant": dataset_variant_needle,
        }
        limited_options = {}
        truncated_options = {}
        for name, values in filter_options.items():
            options = sorted(values, key=str.casefold)
            option_needle = option_needles[name]
            if option_needle:
                options = [
                    value for value in options if option_needle in value.casefold()
                ]
            truncated_options[name] = len(options) > option_limit
            limited_options[name] = options[:option_limit]
        result["filter_options"] = limited_options
        result["filter_options_truncated"] = truncated_options
        return result

    def list_classic_runs(self) -> list[dict[str, Any]]:
        """Return one terminal-view-compatible results artifact per run directory."""

        self.refresh()
        grouped: dict[Path, list[dict]] = {}
        for run in self.runs.values():
            grouped.setdefault(run["path"].parent, []).append(run)

        selected = []
        for runs in grouped.values():
            runs.sort(
                key=lambda item: (
                    item["file"] == "results_reviewed.json",
                    item["file"].startswith("results_reviewed-"),
                    item.get("mtime") or 0,
                ),
                reverse=True,
            )
            run = runs[0]
            selected.append(
                {
                    "id": run["id"],
                    "name": run["name"],
                    "file": run["file"],
                    "relative_path": run["relative_path"],
                    "timestamp": run.get("timestamp"),
                    "domain": run.get("domain"),
                    "simulation_count": len(run["simulations"]),
                    "error": run.get("error"),
                }
            )
        selected.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
        return selected

    def _load_run_simulation(self, run: dict, simulation_id: str) -> dict:
        raw = self._load_run_raw(run)
        simulation = next(
            (
                item
                for item in raw.get("simulations", [])
                if item.get("id") == simulation_id
            ),
            None,
        )
        if simulation is None:
            simulations_dir = run["path"].parent / "simulations"
            candidate = (simulations_dir / f"{simulation_id}.json").resolve()
            if (
                candidate.is_relative_to(simulations_dir.resolve())
                and candidate.exists()
            ):
                simulation = read_json_retry(candidate)
        if simulation is None:
            raise KeyError("simulation")
        return simulation

    @staticmethod
    def _action_fraction(partial: dict, kind: str) -> dict | None:
        value = partial.get(kind)
        if value:
            return {"correct": value.get("correct", 0), "count": value.get("count", 0)}
        if kind == "read" and partial.get("total"):
            value = partial["total"]
            return {"correct": value.get("correct", 0), "count": value.get("count", 0)}
        return None

    @staticmethod
    def _partial_action_reward(reward_info: dict) -> dict:
        """Recreate RewardInfo.partial_action_reward from serialized checks."""

        checks = reward_info.get("action_checks") or []
        if not checks:
            return {}
        if reward_info.get("partial_action_reward"):
            return reward_info["partial_action_reward"]

        def fraction(values: list[dict]) -> dict | None:
            if not values:
                return None
            return {
                "correct": sum(bool(value.get("action_match")) for value in values),
                "count": len(values),
            }

        result = {"total": fraction(checks)}
        for kind in ("read", "write"):
            values = [value for value in checks if value.get("tool_type") == kind]
            result[kind] = fraction(values)
        return result

    @staticmethod
    def _is_successful(reward: Any) -> bool:
        return isinstance(reward, (int, float)) and 1 - 1e-6 <= reward <= 1 + 1e-6

    @staticmethod
    def _review_summary(simulation: dict) -> dict[str, Any]:
        if simulation.get("review") is not None:
            errors = (simulation.get("review") or {}).get("errors") or []
        else:
            errors = (simulation.get("user_only_review") or {}).get("errors") or []
        agent_errors = [error for error in errors if error.get("source") == "agent"]
        user_errors = [
            error
            for error in errors
            if error.get("source") == "user" or not error.get("source")
        ]

        def tags(values: list[dict]) -> list[str]:
            return sorted(
                {
                    str(tag)
                    for value in values
                    for tag in (value.get("error_tags") or [])
                }
            )

        return {
            "agent_error_count": len(agent_errors),
            "user_error_count": len(user_errors),
            "agent_critical": any(
                error.get("severity") == "critical" for error in agent_errors
            ),
            "user_critical": any(
                error.get("severity")
                in {"critical", "critical_helped", "critical_hindered"}
                for error in user_errors
            ),
            "agent_tags": tags(agent_errors),
            "user_tags": tags(user_errors),
            "has_review": bool(
                simulation.get("review") or simulation.get("user_only_review")
            ),
        }

    def list_classic_simulations(
        self,
        run_id: str,
        mode: str,
        query: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        """Build the same result columns and failure filters as ``tau3 view``."""

        self.refresh()
        run = self._require(self.runs, run_id, "run")
        summaries = run["simulations"]
        success_by_task: Counter[str] = Counter()
        for summary in summaries:
            if self._is_successful(summary.get("reward")):
                success_by_task[str(summary.get("task_id"))] += 1

        filtered = []
        needle = query.casefold()
        for index, summary in enumerate(summaries, 1):
            reward = summary.get("reward")
            if mode == "failed" and (reward is None or self._is_successful(reward)):
                continue
            if (
                mode == "all_failed"
                and success_by_task[str(summary.get("task_id"))] > 0
            ):
                continue
            if (
                needle
                and needle
                not in f"{summary.get('task_id')} {summary.get('id')}".casefold()
            ):
                continue
            filtered.append((index, summary))

        total = len(filtered)
        page = max(1, page)
        page_size = min(100, max(1, page_size))
        selected = filtered[(page - 1) * page_size : page * page_size]
        rows = []
        for index, summary in selected:
            simulation = self._load_run_simulation(run, summary["id"])
            reward_info = simulation.get("reward_info") or {}
            db_check = reward_info.get("db_check") or {}
            partial = self._partial_action_reward(reward_info)
            auth = simulation.get("auth_classification") or {}
            rows.append(
                {
                    "index": index,
                    "id": simulation.get("id"),
                    "task_id": simulation.get("task_id"),
                    "trial": simulation.get("trial"),
                    "reward": reward_info.get("reward", summary.get("reward")),
                    "db_match": db_check.get("db_match") if db_check else None,
                    "read_actions": self._action_fraction(partial, "read"),
                    "write_actions": self._action_fraction(partial, "write"),
                    "reward_basis": reward_info.get("reward_basis") or [],
                    "reward_breakdown": reward_info.get("reward_breakdown") or {},
                    "verifier_note": (reward_info.get("info") or {}).get("note"),
                    "auth_status": auth.get("status"),
                    "termination_reason": simulation.get("termination_reason"),
                    "had_unresponsive_period": (simulation.get("info") or {}).get(
                        "had_unresponsive_period"
                    ),
                }
                | self._review_summary(simulation)
            )
        return {
            "items": rows,
            "page": page,
            "page_size": page_size,
            "total": total,
            "failed_task_count": sum(
                success_by_task[str(task_id)] == 0
                for task_id in {summary.get("task_id") for summary in summaries}
            ),
        }

    def list_classic_tasks(self, run_id: str) -> list[dict[str, Any]]:
        """Return task definitions embedded in a results artifact."""

        self.refresh()
        run = self._require(self.runs, run_id, "run")
        return self._load_run_raw(run).get("tasks") or []

    def list_run_simulations(
        self,
        run_id: str,
        query: str,
        failed: bool,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        self.refresh()
        run = self._require(self.runs, run_id, "run")
        needle = query.casefold()
        rows = []
        for sim in run["simulations"]:
            if (
                needle
                and needle not in f"{sim.get('task_id')} {sim.get('id')}".casefold()
            ):
                continue
            if failed and sim.get("reward") == 1:
                continue
            rows.append(sim | {"run_id": run_id})
        return paginate(rows, page, page_size, sort_by, order)

    def _load_run_raw(self, run: dict) -> dict:
        return self._read_cached(run["path"])

    def simulation_detail(self, run_id: str, simulation_id: str) -> dict[str, Any]:
        self.refresh()
        run = self._require(self.runs, run_id, "run")
        raw = self._load_run_raw(run)
        simulation = self._load_run_simulation(run, simulation_id)
        task = next(
            (
                t
                for t in raw.get("tasks", [])
                if str(t.get("id")) == str(simulation.get("task_id"))
            ),
            None,
        )
        return self._decorate_trajectory(
            simulation,
            task,
            run_info=raw.get("info") or {},
            artifact=ArtifactRef(
                kind="simulation",
                id=simulation_id,
                run_id=run_id,
                task_id=simulation.get("task_id"),
                trial=simulation.get("trial"),
            ),
        )

    @staticmethod
    def _diagnosis_task_filename(task_id: str) -> str:
        """Return a readable safe filename, hashing unusual task identifiers."""

        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", task_id):
            return f"{task_id}.json"
        suffix = hashlib.sha256(task_id.encode()).hexdigest()[:16]
        return f"task-{suffix}.json"

    def _fixed_diagnosis_context(self, artifact: ArtifactRef) -> dict[str, Any]:
        """Resolve a logical trajectory reference to its trusted diagnosis location."""

        self.refresh()
        if artifact.kind == "simulation":
            if not artifact.run_id:
                raise ValueError("A simulation diagnosis requires run_id")
            run = self._require(self.runs, artifact.run_id, "run")
            simulation = self._load_run_simulation(run, artifact.id)
            task_id = str(simulation.get("task_id"))
            trial = simulation.get("trial")
            namespace = artifact.run_id
            package = {
                "kind": "simulation_run",
                "id": namespace,
                "name": run["name"],
            }
            directory = self.fixed_attribution_dir / "simulations" / namespace
        elif artifact.kind == "trajectory":
            entry = self._require(self.trajectories, artifact.id, "trajectory")
            if entry.get("format") == "training_sample":
                raise ValueError("Training samples do not have evaluation diagnoses")
            task_id = str(entry["task_id"])
            trial = entry.get("trial")
            namespace = entry.get("package_id") or entry.get("bundle_id")
            if not namespace:
                raise ValueError("Trajectory has no synthetic package namespace")
            catalog = (
                self.synthetic_packages if entry.get("package_id") else self.bundles
            )
            package_record = self._require(catalog, namespace, "package")
            package = {
                "kind": "synthetic_package",
                "id": namespace,
                "name": package_record["name"],
            }
            directory = self.fixed_attribution_dir / "synthetic" / namespace
            raw = read_json_retry(entry["path"])
            simulation = raw.get("simulation") if entry.get("package_id") else raw
            if not isinstance(simulation, dict):
                raise ValueError("Trajectory artifact has no simulation object")
        else:
            raise ValueError(
                "Fixed diagnosis supports simulation or trajectory artifacts"
            )

        if artifact.task_id is not None and str(artifact.task_id) != task_id:
            raise ValueError("Artifact task_id does not match the indexed trajectory")
        if artifact.trial is not None and str(artifact.trial) != str(trial):
            raise ValueError("Artifact trial does not match the indexed trajectory")

        path = (directory / self._diagnosis_task_filename(task_id)).resolve()
        root = self.fixed_attribution_dir.resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                "Diagnosis path resolves outside the calibration directory"
            )
        valid_turns = {
            message.get("turn_idx", index)
            for index, message in enumerate(simulation.get("messages") or [])
            if isinstance(message, dict)
        }
        return {
            "artifact": artifact,
            "task_id": task_id,
            "trial": trial,
            "package": package,
            "path": path,
            "valid_turns": valid_turns,
        }

    @staticmethod
    def _validate_fixed_diagnosis_entry(
        value: Any, index: int, valid_turns: set[Any]
    ) -> dict[str, Any]:
        """Validate one fixed diagnosis without guessing absent model output."""

        label = f"diagnoses[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        artifact_id = value.get("artifact_id")
        if artifact_id is not None and not isinstance(artifact_id, str):
            raise ValueError(f"{label}.artifact_id must be a string")
        if "has_errors" not in value or not isinstance(value["has_errors"], bool):
            raise ValueError(f"{label}.has_errors must be an explicit boolean")
        summary = value.get("summary")
        if summary is not None and not isinstance(summary, str):
            raise ValueError(f"{label}.summary must be a string")
        errors = value.get("errors", [])
        if not isinstance(errors, list):
            raise ValueError(f"{label}.errors must be an array")
        if not value["has_errors"] and errors:
            raise ValueError(f"{label} declares has_errors=false but contains errors")
        if value["has_errors"] and not errors:
            raise ValueError(f"{label} declares has_errors=true but contains no errors")
        for error_index, error in enumerate(errors):
            error_label = f"{label}.errors[{error_index}]"
            if not isinstance(error, dict):
                raise ValueError(f"{error_label} must be an object")
            source = error.get("source", "unknown")
            if source not in {"agent", "user", "system", "unknown"}:
                raise ValueError(f"{error_label}.source is not supported")
            turn_idx = error.get("turn_idx")
            if turn_idx is not None:
                if isinstance(turn_idx, bool) or not isinstance(turn_idx, int):
                    raise ValueError(f"{error_label}.turn_idx must be an integer")
                if turn_idx < 0 or turn_idx not in valid_turns:
                    raise ValueError(
                        f"{error_label}.turn_idx does not exist in this trajectory"
                    )
            tags = error.get("error_tags", [])
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError(f"{error_label}.error_tags must be a string array")
            reasoning = error.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError(f"{error_label}.reasoning must be a non-empty string")
            correct = error.get("correct_behavior")
            if correct is not None and not isinstance(correct, str):
                raise ValueError(f"{error_label}.correct_behavior must be a string")
        return value

    def fixed_diagnosis(self, artifact: ArtifactRef) -> dict[str, Any]:
        """Load the fixed task file and select the current trajectory diagnosis."""

        context = self._fixed_diagnosis_context(artifact)
        path = context["path"]
        expected_file = self._relative(path)
        base = {
            "source": "fixed_file",
            "artifact": artifact.model_dump(mode="json"),
            "package": context["package"],
            "task_id": context["task_id"],
            "trial": context["trial"],
            "expected_file": expected_file,
            "diagnosis": None,
        }
        if not path.exists():
            return base | {"status": "file_missing"}

        payload = self._read_cached(path)
        if not isinstance(payload, dict):
            raise ValueError("Fixed diagnosis file must contain a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError("Fixed diagnosis file schema_version must be 1")
        if str(payload.get("task_id")) != context["task_id"]:
            raise ValueError("Fixed diagnosis task_id does not match its indexed task")
        package = payload.get("package")
        if not isinstance(package, dict):
            raise ValueError("Fixed diagnosis package must be an object")
        if package.get("kind") != context["package"]["kind"]:
            raise ValueError("Fixed diagnosis package.kind does not match")
        if package.get("id") != context["package"]["id"]:
            raise ValueError("Fixed diagnosis package.id does not match")
        diagnoses = payload.get("diagnoses")
        if not isinstance(diagnoses, list):
            raise ValueError("Fixed diagnosis diagnoses must be an array")

        validated = [
            self._validate_fixed_diagnosis_entry(value, index, context["valid_turns"])
            for index, value in enumerate(diagnoses)
        ]
        artifact_ids = [
            item["artifact_id"] for item in validated if item.get("artifact_id")
        ]
        duplicates = [
            value for value, count in Counter(artifact_ids).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"Fixed diagnosis contains duplicate artifact_id: {duplicates[0]}"
            )

        exact = [item for item in validated if item.get("artifact_id") == artifact.id]
        matches = exact
        if not matches:
            matches = [
                item
                for item in validated
                if item.get("trial") is not None
                and str(item["trial"]) == str(context["trial"])
            ]
        if len(matches) > 1:
            raise ValueError("Fixed diagnosis has multiple matches for this trajectory")
        if not matches:
            return base | {
                "status": "artifact_missing",
                "file_updated_at": datetime.fromtimestamp(
                    path.stat().st_mtime, UTC
                ).isoformat(),
            }
        return base | {
            "status": "matched",
            "diagnosis": matches[0],
            "file_updated_at": datetime.fromtimestamp(
                path.stat().st_mtime, UTC
            ).isoformat(),
        }

    def list_bundles(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
        status: str = "",
    ) -> dict[str, Any]:
        self.refresh()
        needle = query.casefold()
        rows = []
        for bundle in self.bundles.values():
            manifest = bundle["manifest"]
            if status and manifest.get("status") != status:
                continue
            if (
                needle
                and needle
                not in f"{bundle['name']} {manifest.get('status', '')}".casefold()
            ):
                continue
            rows.append(
                {
                    "id": bundle["id"],
                    "name": bundle["name"],
                    "relative_path": bundle["relative_path"],
                    "status": manifest.get("status"),
                    "domain": manifest.get("domain"),
                    "requested_tasks": manifest.get("requested_tasks"),
                    "published_tasks": manifest.get("published_tasks"),
                    "task_count": len(bundle["tasks"]),
                    "trajectory_count": sum(
                        t["bundle_id"] == bundle["id"]
                        for t in self.trajectories.values()
                    ),
                    "parent_bundle": manifest.get("parent_bundle"),
                    "stale": bundle["stale"],
                    "error": bundle["error"],
                }
            )
        rows.sort(key=lambda item: item["name"])
        return paginate(rows, page, page_size, sort_by, order)

    def list_synthetic_packages(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
        status: str = "",
        package_format: str = "",
    ) -> dict[str, Any]:
        """List normalized synthetic task packages across source formats."""

        self.refresh()
        needle = query.casefold()
        rows = []
        for package in self.synthetic_packages.values():
            if status and package["status"] != status:
                continue
            if package_format and package["format"] != package_format:
                continue
            searchable = (
                f"{package['name']} {package['relative_path']} "
                f"{package['format']} {package['status']} {package.get('domain') or ''}"
            ).casefold()
            if needle and needle not in searchable:
                continue
            direct_trajectories = package.get("trajectory_count")
            if direct_trajectories is None:
                source_bundle_id = package.get("source_bundle_id")
                direct_trajectories = (
                    sum(
                        trajectory.get("bundle_id") == source_bundle_id
                        for trajectory in self.trajectories.values()
                    )
                    if source_bundle_id
                    else 0
                )
            rows.append(
                {
                    "id": package["id"],
                    "name": package["name"],
                    "relative_path": package["relative_path"],
                    "format": package["format"],
                    "status": package["status"],
                    "domain": package["domain"],
                    "task_count": package.get("task_count", len(package["tasks"])),
                    "trajectory_count": direct_trajectories,
                    "validation": package["validation"],
                    "editable": package["editable"],
                    "training_summary": package.get("training_summary"),
                    "error": package["error"],
                }
            )
        rows.sort(key=lambda item: (item["name"], item["relative_path"]))
        result = paginate(rows, page, page_size, sort_by, order)
        result["formats"] = sorted({item["format"] for item in rows})
        result["summary"] = {
            "packages": len(rows),
            "tasks": sum(item["task_count"] for item in rows),
            "trajectories": sum(item["trajectory_count"] for item in rows),
        }
        return result

    def list_synthetic_package_tasks(
        self,
        package_id: str,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
        accepted: bool | None = None,
    ) -> dict[str, Any]:
        """List the namespaced child tasks of one synthetic package."""

        self.refresh()
        package = self._require(self.synthetic_packages, package_id, "package")
        if package["format"] == "targeted_round":
            self._ensure_targeted_package(package)
        elif package["format"] == "training_dataset":
            self._ensure_training_dataset(package)
        needle = query.casefold()
        rows = []
        for task_id, record in package["tasks"].items():
            candidate = record.get("candidate") or {}
            is_accepted = candidate.get("accepted")
            if accepted is not None and is_accepted is not accepted:
                continue
            if package["format"] == "targeted_round":
                task = {}
                purpose = (candidate.get("slot") or {}).get("recipe")
                required_documents = []
            elif package["format"] == "training_dataset":
                task = {}
                purpose = "训练数据 Task"
                required_documents = []
            else:
                try:
                    task = self._package_task(package, task_id)
                    purpose = (task.get("description") or {}).get("purpose")
                    required_documents = task.get("required_documents") or []
                except Exception:
                    task = {}
                    purpose = None
                    required_documents = []
            searchable = (
                f"{task_id} {purpose or ''} {record.get('stage') or ''} "
                f"{record.get('slot_id') or ''} {record.get('candidate_id') or ''}"
            ).casefold()
            if needle and needle not in searchable:
                continue
            trajectory_count = len(record.get("trajectory_ids") or [])
            if not record.get("trajectory_ids"):
                source_bundle_id = package.get("source_bundle_id")
                trajectory_count = (
                    sum(
                        trajectory.get("bundle_id") == source_bundle_id
                        and trajectory["task_id"] == task_id
                        for trajectory in self.trajectories.values()
                    )
                    if source_bundle_id
                    else 0
                )
            validation_dir = package["root"] / "validation" / task_id
            rows.append(
                {
                    "id": task_id,
                    "package_id": package_id,
                    "source": record.get("source"),
                    "family": (candidate.get("skeleton") or {}).get("family")
                    or purpose,
                    "purpose": purpose,
                    "accepted": is_accepted,
                    "status": candidate.get("status")
                    or ("validated" if validation_dir.exists() else package["status"]),
                    "splits": record.get("splits", []),
                    "required_document_count": len(required_documents),
                    "trajectory_count": trajectory_count,
                    "stage": record.get("stage"),
                    "slot_id": record.get("slot_id"),
                    "candidate_id": record.get("candidate_id"),
                    "task_prefix": task_id_prefix(task_id),
                    "sample_counts": record.get("sample_counts"),
                }
            )
        rows.sort(key=lambda item: item["id"])
        result = paginate(rows, page, page_size, sort_by, order)
        result["package"] = {
            "id": package_id,
            "name": package["name"],
            "format": package["format"],
            "status": package["status"],
            "relative_path": package["relative_path"],
            "validation": package["validation"],
            "training_summary": package.get("training_summary"),
        }
        return result

    def _associated_run_simulations(
        self, package: dict, task_id: str
    ) -> list[dict[str, Any]]:
        package_names = {
            package["name"],
            package["relative_path"],
            str(package["manifest"].get("bundle_id") or ""),
        }
        package_names.discard("")
        unique_task = (
            sum(task_id in item["tasks"] for item in self.synthetic_packages.values())
            == 1
        )
        linked = []
        for run in self.runs.values():
            explicit = str(run.get("task_bundle") or "") in package_names
            if not explicit and not unique_task:
                continue
            for simulation in run["simulations"]:
                if simulation.get("task_id") != task_id:
                    continue
                linked.append(
                    {
                        "kind": "simulation",
                        "run_id": run["id"],
                        "id": simulation["id"],
                        "trial": simulation.get("trial"),
                        "reward": simulation.get("reward"),
                        "association": "explicit" if explicit else "unique_task_id",
                    }
                )
        return linked

    def synthetic_package_task_detail(
        self, package_id: str, task_id: str
    ) -> dict[str, Any]:
        """Return a normalized task with validation evidence and executions."""

        self.refresh()
        package = self._require(self.synthetic_packages, package_id, "package")
        if package["format"] == "targeted_round":
            self._ensure_targeted_package(package)
        elif package["format"] == "training_dataset":
            self._ensure_training_dataset(package)
        record = self._require(package["tasks"], task_id, "task")
        task = self._package_task(package, task_id)
        if package["format"] == "targeted_round":
            index_path = record.get("index_path")
            if index_path and index_path.exists():
                index = read_json_retry(index_path)
                status = index.get("status")
                record["candidate"] = {
                    "status": status,
                    "accepted": status == "VALID" if status is not None else None,
                    "checks": index.get("checks") or {},
                    "slot": index.get("slot") or {},
                }
            trials_path = index_path.parent / "trials.json" if index_path else None
            if trials_path and trials_path.exists():
                trials = read_json_retry(trials_path)
                trial_by_id = {
                    str(item.get("trial")): item
                    for item in trials
                    if isinstance(item, dict)
                }
                for trajectory_id in record.get("trajectory_ids") or []:
                    entry = self.trajectories.get(trajectory_id)
                    if entry is None:
                        continue
                    summary = dict(trial_by_id.get(str(entry.get("trial"))) or {})
                    if (
                        summary.get("reward") is None
                        and summary.get("environment_success") is not None
                    ):
                        summary["reward"] = int(bool(summary["environment_success"]))
                    entry["summary"] = summary
        trajectories = [
            {
                "kind": "trajectory",
                "id": trajectory["id"],
                "trial": trajectory["trial"],
                "reward": (trajectory.get("summary") or {}).get("reward"),
                "association": "explicit",
                "artifact_type": trajectory.get("format") or "evaluation_trajectory",
                "dataset_variant": trajectory.get("dataset_variant"),
                "source_kind": trajectory.get("source_kind"),
                "slot": trajectory.get("slot"),
                "sample_hash": trajectory.get("sample_hash"),
            }
            for trajectory in self.trajectories.values()
            if (
                trajectory.get("package_id") == package_id
                or (
                    package.get("source_bundle_id")
                    and trajectory.get("bundle_id") == package.get("source_bundle_id")
                )
            )
            and trajectory["task_id"] == task_id
        ]
        validation_files = []
        validation_dir = package["root"] / "validation" / task_id
        if validation_dir.exists():
            validation_files = [
                path.name for path in sorted(validation_dir.glob("*.json"))
            ]
        return {
            "package": {
                "id": package_id,
                "name": package["name"],
                "format": package["format"],
                "status": package["status"],
                "relative_path": package["relative_path"],
                "editable": package["editable"],
                "source_bundle_id": package.get("source_bundle_id"),
            },
            "task": task,
            "source": record.get("source"),
            "splits": record.get("splits", []),
            "candidate": record.get("candidate"),
            "targeted_round": {
                "stage": record.get("stage"),
                "slot_id": record.get("slot_id"),
                "candidate_id": record.get("candidate_id"),
            }
            if package["format"] == "targeted_round"
            else None,
            "training_dataset": {
                "summary": package.get("training_summary") or {},
                "sample_counts": record.get("sample_counts") or {},
                "slot_ids": record.get("slot_ids") or [],
                "note": (
                    "这些是训练数据样本，不包含评测 reward、verifier 或模型 review。"
                ),
            }
            if package["format"] == "training_dataset"
            else None,
            "validation": {
                "package": package["validation"],
                "task_files": validation_files,
            },
            "trajectories": trajectories,
            "run_simulations": self._associated_run_simulations(package, task_id),
            "sha256": digest(task),
        }

    def list_tasks(
        self,
        bundle_id: str,
        query: str,
        accepted: bool | None,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        self.refresh()
        bundle = self._require(self.bundles, bundle_id, "bundle")
        needle = query.casefold()
        rows = []
        for task_id, record in bundle["tasks"].items():
            candidate = record.get("candidate") or {}
            is_accepted = candidate.get("accepted")
            if accepted is not None and is_accepted is not accepted:
                continue
            task = record["task"]
            text = json.dumps(task.get("description", {}), ensure_ascii=False)
            if needle and needle not in f"{task_id} {text}".casefold():
                continue
            trajectories = [
                t
                for t in self.trajectories.values()
                if t["bundle_id"] == bundle_id and t["task_id"] == task_id
            ]
            rows.append(
                {
                    "id": task_id,
                    "bundle_id": bundle_id,
                    "source": record["source"],
                    "family": (candidate.get("skeleton") or {}).get("family"),
                    "accepted": is_accepted,
                    "static_passed": candidate.get("static_passed"),
                    "text_checked": candidate.get("text_checked"),
                    "splits": record.get("splits", []),
                    "required_documents": task.get("required_documents") or [],
                    "trajectory_count": len(trajectories),
                    "purpose": (task.get("description") or {}).get("purpose"),
                }
            )
        rows.sort(key=lambda item: item["id"])
        return paginate(rows, page, page_size, sort_by, order)

    def task_detail(self, bundle_id: str, task_id: str) -> dict[str, Any]:
        self.refresh()
        bundle = self._require(self.bundles, bundle_id, "bundle")
        record = self._require(bundle["tasks"], task_id, "task")
        trajectories = [
            {
                "id": t["id"],
                "trial": t["trial"],
                "summary": t["summary"],
            }
            for t in self.trajectories.values()
            if t["bundle_id"] == bundle_id and t["task_id"] == task_id
        ]
        record_tables = {
            str(record_id): table
            for table, contents in self._load_db().items()
            for record_id in (contents.get("data") or {})
        }
        referenced_ids: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if str(key) in record_tables:
                        referenced_ids.add(str(key))
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
            elif isinstance(value, str) and value in record_tables:
                referenced_ids.add(value)

        collect(record["task"])
        return {
            "bundle": {
                "id": bundle_id,
                "name": bundle["name"],
                "status": bundle["manifest"].get("status"),
                "stale": bundle["stale"],
            },
            "task": record["task"],
            "candidate": record.get("candidate"),
            "splits": record.get("splits", []),
            "trajectories": trajectories,
            "db_references": [
                {"id": record_id, "table": record_tables[record_id]}
                for record_id in sorted(referenced_ids)
            ],
            "sha256": digest(record["task"]),
        }

    def trajectory_detail(self, trajectory_id: str) -> dict[str, Any]:
        self.refresh()
        entry = self._require(self.trajectories, trajectory_id, "trajectory")
        if entry.get("format") == "training_sample":
            return self._training_sample_detail(entry)
        raw = read_json_retry(entry["path"])
        simulation = raw.get("simulation") if entry.get("package_id") else raw
        if not isinstance(simulation, dict):
            raise ValueError("Trajectory artifact has no simulation object")
        package_id = entry.get("package_id")
        if package_id:
            package = self._require(self.synthetic_packages, package_id, "package")
            record = self._require(package["tasks"], entry["task_id"], "task")
            task = self._package_task(package, entry["task_id"])
            manifest = package["manifest"]
            capture_path = entry["path"].with_name("capture.json")
        else:
            bundle = self._require(self.bundles, entry["bundle_id"], "bundle")
            record = self._require(bundle["tasks"], entry["task_id"], "task")
            task = record["task"]
            manifest = bundle["manifest"]
            capture_path = entry["path"].with_suffix(".capture.json")
        capture = read_json_retry(capture_path) if capture_path.exists() else None
        config = manifest.get("config") or {}
        identity = raw.get("identity") if isinstance(raw, dict) else {}
        run_info = {
            "agent_info": {
                "llm": config.get("teacher_model")
                or (identity or {}).get("agent_model"),
                "llm_args": config.get("llm_args"),
            },
            "user_info": {
                "llm": config.get("user_model") or (identity or {}).get("user_model")
            },
            "retrieval_config": config.get("retrieval_config"),
        }
        return self._decorate_trajectory(
            simulation,
            task,
            run_info=run_info,
            artifact=ArtifactRef(
                kind="trajectory",
                id=trajectory_id,
                bundle_id=entry.get("bundle_id"),
                package_id=package_id,
                task_id=entry["task_id"],
                trial=entry["trial"],
            ),
            synthesis_review=entry["summary"].get("review"),
            capture=capture,
        )

    def _training_sample_detail(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Render one indexed SFT/general-agent pair as a read-only trajectory."""

        package = self._require(self.synthetic_packages, entry["package_id"], "package")
        self._ensure_training_dataset(package)
        sft = self._read_jsonl_at(
            entry["path"], entry["sft_offset"], entry["sft_length"]
        )
        general = self._read_jsonl_at(
            entry["general_path"],
            entry["general_offset"],
            entry["general_length"],
        )
        task = self._package_task(package, entry["task_id"])
        loss_mask = sft.get("loss_mask") or []
        messages = []
        system_messages = []
        for index, message in enumerate(sft.get("messages") or []):
            role = message.get("role")
            if role == "system":
                system_messages.append(message)
                continue
            item = copy.deepcopy(message)
            item["turn_idx"] = index
            item["training_loss"] = bool(
                loss_mask[index] if index < len(loss_mask) else 0
            )
            normalized_calls = []
            for call in item.get("tool_calls") or []:
                function = call.get("function") or {}
                normalized_calls.append(
                    {
                        "id": call.get("id"),
                        "name": function.get("name") or call.get("name"),
                        "arguments": self._tool_arguments(
                            function.get("arguments", call.get("arguments"))
                        ),
                        "requestor": "assistant",
                    }
                )
            if normalized_calls:
                item["tool_calls"] = normalized_calls
            messages.append(item)
        metadata = sft.get("metadata") or {}
        trial = metadata.get("trial")
        audits = []
        audit_key = f"{entry['task_id']}\0{trial}"
        audit_path = package["root"] / "audit.jsonl"
        for offset, length in (
            package.get("training_index", {})
            .get("audit_offsets", {})
            .get(audit_key, [])
        ):
            audits.append(self._read_jsonl_at(audit_path, offset, length))
        simulation = {
            "id": entry["id"],
            "task_id": entry["task_id"],
            "trial": trial,
            "messages": messages,
            "reward_info": {
                "reward": None,
                "info": {"note": "训练数据导出未包含评测 reward 或 verifier 结果。"},
            },
            "termination_reason": "training_sample",
            "policy": system_messages[0].get("content") if system_messages else None,
            "info": {"training_sample": True},
        }
        artifact = ArtifactRef(
            kind="trajectory",
            id=entry["id"],
            package_id=entry["package_id"],
            task_id=entry["task_id"],
            trial=entry["trial"],
        )
        result = self._decorate_trajectory(
            simulation,
            task,
            artifact=artifact,
            run_info={
                "agent_info": {"llm": metadata.get("teacher_model")},
                "retrieval_config": sft.get("retrieval_config"),
            },
            capture={"visible_sample": {"messages": system_messages}},
        )
        result["verifier_diagnostics"] = None
        result["training_sample"] = {
            "dataset_variant": entry.get("dataset_variant"),
            "source_kind": entry.get("source_kind"),
            "line": entry.get("trial"),
            "slot": entry.get("slot"),
            "shard_line": entry.get("shard_line"),
            "sample_hash": entry.get("sample_hash"),
            "schema_version": sft.get("schema_version"),
            "retrieval_config": sft.get("retrieval_config"),
            "metadata": metadata,
            "loss_mask": loss_mask,
            "audits": audits,
            "representations": {
                "sft": {
                    "schema_version": sft.get("schema_version"),
                    "task_id": sft.get("task_id"),
                    "metadata": metadata,
                    "retrieval_config": sft.get("retrieval_config"),
                    "tools": sft.get("tools") or [],
                },
                "general_agent": general,
            },
            "note": (
                "这是训练样本而非评测运行；同一逻辑样本的 SFT 与 "
                "general-agent 序列化不会重复计数。"
            ),
        }
        return result

    def _decorate_trajectory(
        self,
        simulation: dict,
        task: dict | None,
        *,
        artifact: ArtifactRef,
        run_info: dict | None = None,
        synthesis_review: dict | None = None,
        capture: dict | None = None,
    ) -> dict[str, Any]:
        messages = []
        for message in simulation.get("messages") or []:
            item = copy.deepcopy(message)
            if item.get("role") in {"assistant", "user"}:
                item["reasoning"] = reasoning_blocks(item)
            messages.append(item)
        prompts: list[dict[str, Any]] = []
        visible = (capture or {}).get("visible_sample", {})
        system_messages = [
            m for m in visible.get("messages", []) if m.get("role") == "system"
        ]
        if system_messages:
            prompts.append(
                {
                    "participant": "agent",
                    "content": system_messages[0].get("content"),
                    "origin": "saved",
                }
            )
        elif simulation.get("policy"):
            prompts.append(
                {
                    "participant": "agent",
                    "content": simulation["policy"],
                    "origin": "saved_policy",
                }
            )
        if task:
            prompts.append(
                {
                    "participant": "user",
                    "content": self._build_user_prompt(task, run_info or {}),
                    "origin": "reconstructed",
                }
            )
        return {
            "artifact": artifact.model_dump(),
            "simulation": simulation | {"messages": messages},
            "task": task,
            "prompts": prompts,
            "model_info": {
                "agent": (run_info or {}).get("agent_info"),
                "user": (run_info or {}).get("user_info"),
                "retrieval_config": (run_info or {}).get("retrieval_config"),
            },
            "synthesis_review": synthesis_review,
            "verifier_diagnostics": self._verifier_diagnostics(simulation, task),
            "human_attributions": self.list_attributions(artifact),
        }

    @staticmethod
    def _tool_arguments(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value if value is not None else {}

    @classmethod
    def _argument_diffs(
        cls, expected: Any, actual: Any, compare_args: list[str] | None
    ) -> list[dict[str, Any]]:
        """Return field-level differences for one expected/actual tool call."""

        expected = cls._tool_arguments(expected)
        actual = cls._tool_arguments(actual)
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return (
                []
                if expected == actual
                else [{"field": "$", "expected": expected, "actual": actual}]
            )
        fields = compare_args or sorted(set(expected) | set(actual))
        diffs = []
        for field in fields:
            expected_present = field in expected
            actual_present = field in actual
            if expected_present and actual_present and expected[field] == actual[field]:
                continue
            diffs.append(
                {
                    "field": field,
                    "expected": expected.get(field),
                    "actual": actual.get(field),
                    "expected_present": expected_present,
                    "actual_present": actual_present,
                }
            )
        return diffs

    @classmethod
    def _verifier_diagnostics(
        cls, simulation: dict, task: dict | None
    ) -> dict[str, Any]:
        """Explain verifier outcomes using expected and observed tool calls."""

        criteria = (task or {}).get("evaluation_criteria") or {}
        reward = simulation.get("reward_info") or {}
        expected_actions = criteria.get("actions") or []
        action_checks = reward.get("action_checks") or []
        actual_calls = []
        for message_index, message in enumerate(simulation.get("messages") or []):
            for call_index, call in enumerate(message.get("tool_calls") or []):
                actual_calls.append(
                    {
                        "key": f"{message_index}:{call_index}",
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "requestor": call.get("requestor") or message.get("role"),
                        "arguments": cls._tool_arguments(call.get("arguments")),
                        "turn_idx": message.get("turn_idx"),
                        "message_index": message_index,
                    }
                )
        consumed: set[str] = set()
        action_results = []
        for index, expected in enumerate(expected_actions):
            check = next(
                (
                    item
                    for item in action_checks
                    if (item.get("action") or {}).get("action_id")
                    == expected.get("action_id")
                ),
                action_checks[index] if index < len(action_checks) else None,
            )
            candidates = [
                call
                for call in actual_calls
                if call["name"] == expected.get("name")
                and (
                    not expected.get("requestor")
                    or call["requestor"] == expected.get("requestor")
                )
            ]
            comparisons = []
            matched = None
            for candidate in candidates:
                diffs = cls._argument_diffs(
                    expected.get("arguments") or {},
                    candidate["arguments"],
                    expected.get("compare_args"),
                )
                comparison = candidate | {"argument_diffs": diffs}
                comparisons.append(comparison)
                if not diffs and matched is None and candidate["key"] not in consumed:
                    matched = comparison
            if matched:
                consumed.add(matched["key"])
            if check is None:
                status = "not_evaluated"
            elif check.get("action_match"):
                status = "matched"
            elif candidates:
                status = "argument_mismatch"
            else:
                status = "missing"
            action_results.append(
                {
                    "index": index,
                    "action_id": expected.get("action_id"),
                    "expected": expected,
                    "check": check,
                    "status": status,
                    "matched_call": matched,
                    "observed_candidates": comparisons[:10],
                }
            )
        extras = [call for call in actual_calls if call["key"] not in consumed]
        failed_turns = sorted(
            {
                candidate["turn_idx"]
                for result in action_results
                if result["status"] not in {"matched", "not_evaluated"}
                for candidate in result["observed_candidates"]
                if candidate.get("turn_idx") is not None
            }
            | {call["turn_idx"] for call in extras if call.get("turn_idx") is not None}
        )
        termination = simulation.get("termination_reason")
        premature = termination not in {"agent_stop", "user_stop"}
        return {
            "evaluation_status": (
                "not_evaluated" if premature and not action_checks else "evaluated"
            ),
            "premature_termination": premature,
            "termination_reason": termination,
            "actions": action_results,
            "extra_calls": extras,
            "failed_turns": failed_turns,
            "summary": {
                "expected": len(expected_actions),
                "matched": sum(
                    result["status"] == "matched" for result in action_results
                ),
                "argument_mismatch": sum(
                    result["status"] == "argument_mismatch" for result in action_results
                ),
                "missing": sum(
                    result["status"] == "missing" for result in action_results
                ),
                "not_evaluated": sum(
                    result["status"] == "not_evaluated" for result in action_results
                ),
                "extra": len(extras),
            },
            "reward_info_note": (reward.get("info") or {}).get("note"),
        }

    def _build_user_prompt(self, task: dict, run_info: dict) -> str:
        from tau3.data_model.persona import PersonaConfig
        from tau3.data_model.tasks import Task, UserScenario
        from tau3.user_prompt import build_user_system_prompt

        parsed = Task.model_validate(task)
        persona_raw = (run_info.get("user_info") or {}).get("persona_config")
        persona = (
            PersonaConfig.model_validate(persona_raw)
            if persona_raw
            else PersonaConfig()
        )
        return build_user_system_prompt(
            instructions=str(UserScenario.model_validate(parsed.user_scenario)),
            persona_guidelines=persona.to_guidelines_text(),
            use_tools=bool(parsed.user_tools),
        )

    def list_documents(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        self.refresh()
        needle = query.casefold()
        rows = []
        for document in self.documents.values():
            if (
                needle
                and needle
                not in f"{document['id']} {document['title']} {document['content']}".casefold()
            ):
                continue
            references = []
            for bundle in self.bundles.values():
                for task_id, record in bundle["tasks"].items():
                    if document["id"] in (
                        record["task"].get("required_documents") or []
                    ):
                        references.append(
                            {"bundle_id": bundle["id"], "task_id": task_id}
                        )
            rows.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "content_preview": document["content"][:240],
                    "reference_count": len(references),
                    "sha256": document["sha256"],
                }
            )
        rows.sort(key=lambda item: item["id"])
        return paginate(rows, page, page_size, sort_by, order)

    def document_detail(self, document_id: str) -> dict[str, Any]:
        self.refresh()
        document = self._require(self.documents, document_id, "document")
        references = []
        for bundle in self.bundles.values():
            for task_id, record in bundle["tasks"].items():
                if document_id in (record["task"].get("required_documents") or []):
                    references.append({"bundle_id": bundle["id"], "task_id": task_id})
        return {
            "document": document,
            "references": references,
            "sha256": document["sha256"],
        }

    def _db_path(self) -> Path:
        return self.data_dir / "tau3" / "domains" / "banking_knowledge" / "db.json"

    def _load_db(self) -> dict:
        path = self._db_path()
        return copy.deepcopy(self._read_cached(path)) if path.exists() else {}

    def list_db_tables(
        self,
        query: str = "",
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        db = self._load_db()
        needle = query.casefold()
        rows = [
            {
                "name": name,
                "records": len((table or {}).get("data", {})),
                "notes": (table or {}).get("notes", ""),
            }
            for name, table in sorted(db.items())
            if not needle
            or needle in f"{name} {(table or {}).get('notes', '')}".casefold()
        ]
        return paginate(rows, page, page_size, sort_by, order)

    def list_db_records(
        self,
        table: str,
        query: str,
        page: int,
        page_size: int,
        sort_by: str = "",
        order: str = "asc",
    ) -> dict[str, Any]:
        db = self._load_db()
        if table not in db:
            raise KeyError("table")
        needle = query.casefold()
        rows = []
        for record_id, value in (db[table].get("data") or {}).items():
            if (
                needle
                and needle
                not in f"{record_id} {json.dumps(value, ensure_ascii=False)}".casefold()
            ):
                continue
            rows.append(
                {
                    "id": record_id,
                    "value": value,
                    "sha256": digest(value),
                    "table": table,
                }
            )
        rows.sort(key=lambda item: item["id"])
        return paginate(rows, page, page_size, sort_by, order)

    def db_table_detail(self, table: str) -> dict[str, Any]:
        db = self._load_db()
        value = self._require(db, table, "table")
        return {"name": table, "value": value, "sha256": digest(value)}

    def db_record_detail(self, table: str, record_id: str) -> dict[str, Any]:
        db = self._load_db()
        value = self._require(
            self._require(db, table, "table").get("data", {}), record_id, "record"
        )
        return {
            "table": table,
            "id": record_id,
            "value": value,
            "sha256": digest(value),
        }

    def create_change_set(self, payload: ChangeSetCreate) -> ChangeSet:
        kinds = {target.artifact.kind for target in payload.targets}
        environment = kinds & {"document", "seed_table", "seed_record"}
        bundle = kinds & {"task"}
        if environment and bundle:
            raise ValueError("Seed and bundle-task changes cannot be mixed")
        now = utc_now()
        item = ChangeSet(
            id=f"change_{uuid4().hex[:12]}",
            rationale=payload.rationale,
            author=payload.author,
            created_at=now,
            updated_at=now,
            targets=payload.targets,
        )
        self._save_change_set(item)
        self._change_sets[item.id] = item
        return item

    def list_change_sets(self) -> list[dict[str, Any]]:
        self.refresh()
        return [
            item.model_dump()
            for item in sorted(
                self._change_sets.values(), key=lambda x: x.created_at, reverse=True
            )
        ]

    def validate_change_set(self, change_id: str) -> ChangeSet:
        item = self._require(self._change_sets, change_id, "change set")
        previews = []
        bundle_ids: set[str] = set()
        for target in item.targets:
            original = self._target_value(target.artifact)
            if digest(original) != target.base_sha256:
                raise ValueError("Source changed since the edit was created")
            updated = apply_patch(original, target.operations)
            self._validate_target(target.artifact, original, updated)
            if target.artifact.bundle_id:
                bundle_ids.add(target.artifact.bundle_id)
            previews.append(
                {
                    "artifact": target.artifact.model_dump(),
                    "before": original,
                    "after": updated,
                }
            )
        if len(bundle_ids) > 1:
            raise ValueError("A task change set may target only one bundle")
        item.status = "validated"
        item.updated_at = utc_now()
        item.validation = {
            "valid": True,
            "previews": previews,
            "impact": self._impact(item),
        }
        self._save_change_set(item)
        return item

    def apply_change_set(self, change_id: str) -> ChangeSet:
        item = self.validate_change_set(change_id)
        kinds = {target.artifact.kind for target in item.targets}
        if kinds == {"task"}:
            result = self._apply_task_changes(item)
        elif kinds <= {"document", "seed_table", "seed_record"}:
            result = self._apply_environment_changes(item)
        else:
            raise ValueError("Unsupported calibration target")
        item.status = "applied"
        item.updated_at = utc_now()
        item.result = result
        self._save_change_set(item)
        cache_clear = getattr(environment_fingerprint, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
        self.refresh(force=True)
        return item

    def _target_value(self, artifact: ArtifactRef) -> Any:
        if artifact.kind == "document":
            return {
                key: value
                for key, value in self._require(
                    self.documents, artifact.id, "document"
                ).items()
                if key in {"id", "title", "content"}
            }
        if artifact.kind == "seed_record":
            if not artifact.task_id:
                raise ValueError("seed_record requires table in task_id")
            return self.db_record_detail(artifact.task_id, artifact.id)["value"]
        if artifact.kind == "seed_table":
            return self.db_table_detail(artifact.id)["value"]
        if artifact.kind == "task":
            if not artifact.bundle_id:
                raise ValueError("task requires bundle_id")
            return self.task_detail(artifact.bundle_id, artifact.id)["task"]
        raise ValueError(f"Artifact kind {artifact.kind} is not editable")

    def _validate_target(
        self, artifact: ArtifactRef, original: Any, updated: Any
    ) -> None:
        if artifact.kind == "document":
            from tau3.domains.banking_knowledge.data_model import Document

            parsed = Document.model_validate(updated)
            if parsed.id != original["id"]:
                raise ValueError("Document id is immutable")
        elif artifact.kind == "seed_record":
            from tau3.domains.banking_knowledge.data_model import TransactionalDB

            db = self._load_db()
            table = artifact.task_id
            assert table is not None
            db[table]["data"][artifact.id] = updated
            TransactionalDB.model_validate(db)
        elif artifact.kind == "seed_table":
            from tau3.domains.banking_knowledge.data_model import TransactionalDB

            if set(original.get("data", {})) != set(updated.get("data", {})):
                raise ValueError("Record ids are immutable in the first release")
            db = self._load_db()
            db[artifact.id] = updated
            TransactionalDB.model_validate(db)
        elif artifact.kind == "task":
            from tau3.data_model.tasks import Task

            parsed = Task.model_validate(updated)
            if str(parsed.id) != str(original["id"]):
                raise ValueError("Task id is immutable")
            missing = set(parsed.required_documents or []) - self.documents.keys()
            if missing:
                raise ValueError(f"Unknown required documents: {sorted(missing)}")

    def _impact(self, item: ChangeSet) -> dict[str, Any]:
        document_ids = {
            t.artifact.id for t in item.targets if t.artifact.kind == "document"
        }
        referenced_tasks = []
        for bundle in self.bundles.values():
            for task_id, record in bundle["tasks"].items():
                if document_ids & set(record["task"].get("required_documents") or []):
                    referenced_tasks.append(
                        {"bundle_id": bundle["id"], "task_id": task_id}
                    )
        return {
            "referenced_tasks": referenced_tasks,
            "bundles_become_stale": len(self.bundles)
            if any(
                target.artifact.kind in {"document", "seed_table", "seed_record"}
                for target in item.targets
            )
            else 0,
        }

    def _backup(self, change_id: str, path: Path) -> None:
        rel = self._relative(path)
        destination = self.calibration_dir / "backups" / change_id / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    def _apply_environment_changes(self, item: ChangeSet) -> dict[str, Any]:
        from tau3.domains.banking_knowledge.data_model import TransactionalDB

        db = self._load_db()
        db_changed = False
        written = []
        for target in item.targets:
            original = self._target_value(target.artifact)
            updated = apply_patch(original, target.operations)
            if target.artifact.kind == "document":
                path = self._document_paths[target.artifact.id]
                self._backup(item.id, path)
                write_json_atomic(path, updated)
                written.append(self._relative(path))
            elif target.artifact.kind == "seed_record":
                table = target.artifact.task_id
                assert table is not None
                db[table]["data"][target.artifact.id] = updated
                db_changed = True
            else:
                db[target.artifact.id] = updated
                db_changed = True
        if db_changed:
            path = self._db_path()
            self._backup(item.id, path)
            TransactionalDB.model_validate(db)
            write_json_atomic(path, db)
            written.append(self._relative(path))
        return {"written": written, "backup": f"calibrations/backups/{item.id}"}

    def _fork_bundle(self, bundle: dict) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        base = bundle["root"].with_name(f"{bundle['name']}-calibration-{stamp}")
        destination = base
        counter = 2
        while destination.exists():
            destination = base.with_name(f"{base.name}-{counter}")
            counter += 1

        def link_or_copy(source: str, target: str) -> str:
            try:
                os.link(source, target)
                return target
            except OSError:
                return shutil.copy2(source, target)

        shutil.copytree(bundle["root"], destination, copy_function=link_or_copy)
        manifest = read_json_retry(destination / "manifest.json")
        manifest["status"] = "draft"
        manifest["parent_bundle"] = bundle["relative_path"]
        for key in ("published_tasks", "tasks_hash", "splits_hash"):
            manifest.pop(key, None)
        write_json_atomic(destination / "manifest.json", manifest)
        for filename in (
            "tasks.json",
            "split_tasks.json",
            "metadata.jsonl",
            "rejected.jsonl",
            "report.json",
        ):
            path = destination / filename
            if path.exists():
                path.unlink()
        return destination

    def _apply_task_changes(self, item: ChangeSet) -> dict[str, Any]:
        bundle_id = item.targets[0].artifact.bundle_id
        assert bundle_id is not None
        source = self._require(self.bundles, bundle_id, "bundle")
        published = source["manifest"].get("status") == "published"
        root = self._fork_bundle(source) if published else source["root"]
        for target in item.targets:
            original = self._target_value(target.artifact)
            updated = apply_patch(original, target.operations)
            source_candidate = source["candidate_paths"].get(target.artifact.id)
            if source_candidate is None:
                raise ValueError(
                    f"No candidate checkpoint for task {target.artifact.id}"
                )
            candidate_path = root / source_candidate.relative_to(source["root"])
            candidate = read_json_retry(candidate_path)
            candidate["task"] = updated
            candidate.update(
                accepted=False,
                static_passed=False,
                text_checked=False,
                checks={},
                trials={},
                errors=[],
            )
            trajectory_dir = root / "trajectories" / target.artifact.id
            if trajectory_dir.exists():
                shutil.rmtree(trajectory_dir)
            self._backup(item.id, candidate_path)
            write_json_atomic(candidate_path, candidate)
        return {
            "bundle": self._relative(root),
            "forked": published,
            "parent": source["relative_path"] if published else None,
        }

    def _save_change_set(self, item: ChangeSet) -> None:
        path = self.calibration_dir / "change_sets" / f"{item.id}.json"
        write_json_atomic(path, item.model_dump(mode="json"))

    def add_attribution(self, attribution: HumanAttribution) -> HumanAttribution:
        attribution = attribution.model_copy(
            update={
                "id": attribution.id or f"attribution_{uuid4().hex[:12]}",
                "created_at": attribution.created_at or utc_now(),
            }
        )
        path = self.calibration_dir / "human_attributions.json"
        entries = read_json_retry(path) if path.exists() else []
        entries.append(attribution.model_dump(mode="json"))
        write_json_atomic(path, entries)
        return attribution

    def list_attributions(
        self, artifact: ArtifactRef | None = None
    ) -> list[dict[str, Any]]:
        path = self.calibration_dir / "human_attributions.json"
        entries = read_json_retry(path) if path.exists() else []
        if artifact is None:
            return entries
        return [
            item
            for item in entries
            if item.get("artifact", {}).get("kind") == artifact.kind
            and item.get("artifact", {}).get("id") == artifact.id
        ]

    @staticmethod
    def _require(mapping: dict, key: str, label: str) -> Any:
        if key not in mapping:
            raise KeyError(label)
        return mapping[key]
