"""Allowlisted background-job runner for the local web console."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psutil

from tau3.web_console.models import JobCreate, JobRecord
from tau3.web_console.store import ArtifactStore, read_json_retry, write_json_atomic


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobManager:
    """Runs only known tau3 commands and persists their status and logs."""

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.directory = store.calibration_dir / "jobs"
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._external_paths: list[Path] = []
        self._external_paths_checked_at = 0.0
        self._load()

    def _load(self) -> None:
        if not self.directory.exists():
            return
        for path in self.directory.glob("*.json"):
            try:
                job = JobRecord.model_validate(read_json_retry(path))
                if job.status in {"queued", "running"}:
                    job.status = "orphaned"
                    job.updated_at = _now()
                    self._save(job)
                self._jobs[job.id] = job
            except Exception:
                continue

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda item: item.created_at, reverse=True
            )

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("job")
            return self._jobs[job_id]

    def log(self, job_id: str, tail: int = 20000) -> str:
        job = self.get(job_id)
        path = self.directory / job.log_file
        if not path.exists():
            return ""
        data = path.read_text(errors="replace")
        return data[-min(max(tail, 1), 200000) :]

    def list_for_artifact(
        self,
        kind: str,
        artifact_id: str,
        run_id: str = "",
        bundle_id: str = "",
    ) -> list[JobRecord]:
        """Return trajectory-review jobs for one logical artifact."""

        matches = []
        for job in self.list():
            artifact = job.params.get("artifact") or {}
            if (
                job.type != "trajectory_review"
                or artifact.get("kind") != kind
                or artifact.get("id") != artifact_id
                or (run_id and artifact.get("run_id") != run_id)
                or (bundle_id and artifact.get("bundle_id") != bundle_id)
            ):
                continue
            matches.append(job)
        return matches

    def result(self, job_id: str) -> dict:
        """Load the compact review result produced by a trajectory job."""

        job = self.get(job_id)
        if job.type != "trajectory_review":
            raise ValueError("This job does not produce a trajectory review")
        if job.status != "succeeded":
            raise ValueError("Trajectory review is not complete")
        if not job.output_file:
            raise ValueError("Trajectory review output is missing")
        output = Path(job.output_file).resolve()
        review_root = (self.store.calibration_dir / "reviews").resolve()
        if not output.is_relative_to(review_root):
            raise ValueError(
                "Trajectory review output is outside the runtime directory"
            )
        payload = read_json_retry(output)
        simulations = payload.get("simulations") or []
        if len(simulations) != 1:
            raise ValueError("Trajectory review output is invalid")
        simulation = simulations[0]
        return {
            "job_id": job.id,
            "artifact": job.params.get("artifact"),
            "mode": job.params.get("review_mode", "full"),
            "review": simulation.get("review"),
            "user_only_review": simulation.get("user_only_review"),
            "auth_classification": simulation.get("auth_classification"),
        }

    def external_status(self) -> dict | None:
        """Report the repository's existing synthesis wrapper without owning it."""

        base = self.store.data_dir / "synthetic"
        if time.monotonic() - self._external_paths_checked_at >= 30:
            paths: set[Path] = set()
            if base.exists():
                for pattern in (
                    "job-status.json",
                    "*/job-status.json",
                ):
                    paths.update(base.glob(pattern))
                paths.update(
                    bundle["root"] / "job-status.json"
                    for bundle in self.store.bundles.values()
                    if (bundle["root"] / "job-status.json").exists()
                )
            self._external_paths = sorted(paths)
            self._external_paths_checked_at = time.monotonic()
        paths = [path for path in self._external_paths if path.exists()]

        def modified(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0

        paths.sort(key=modified, reverse=True)
        if not paths:
            return None
        latest: dict | None = None
        for path in paths:
            try:
                status = read_json_retry(path)
            except Exception as exc:
                status = {"error": f"{type(exc).__name__}: {exc}"}
            pid = status.get("pid")
            alive = False
            if isinstance(pid, int) and pid > 0:
                try:
                    command = " ".join(psutil.Process(pid).cmdline()).lower()
                    alive = "synthesize" in command or "run_synthesis_batch" in command
                except (psutil.Error, OSError):
                    pass
            result = status | {
                "process_alive": alive,
                "managed": False,
                "status_file": self.store._relative(path),
            }
            latest = latest or result
            if alive:
                return result
        return latest

    def create(self, payload: JobCreate) -> JobRecord:
        command, target_key, output_file = self._command(payload)
        with self._lock:
            if target_key and any(
                job.target_key == target_key and job.status in {"queued", "running"}
                for job in self._jobs.values()
            ):
                raise ValueError("A writer is already active for this target")
            now = _now()
            job = JobRecord(
                id=f"job_{uuid4().hex[:12]}",
                type=payload.type,
                status="queued",
                created_at=now,
                updated_at=now,
                params=payload.model_dump(mode="json"),
                target_key=target_key,
                log_file="pending.log",
                output_file=output_file,
            )
            job.log_file = f"{job.id}.log"
            self._jobs[job.id] = job
            self._save(job)
            thread = threading.Thread(
                target=self._run, args=(job.id, command), daemon=True
            )
            thread.start()
            return job

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel a process started by this manager instance only."""

        with self._lock:
            job = self.get(job_id)
            process = self._processes.get(job_id)
            if job.status == "queued" and process is None:
                job.status = "cancelled"
                job.updated_at = _now()
                self._save(job)
                return job
            if process is None or job.status != "running":
                raise ValueError(
                    "Only a running job owned by this server can be cancelled"
                )
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            job.status = "cancelled"
            job.updated_at = _now()
            self._save(job)
            return job

    def _command(self, payload: JobCreate) -> tuple[list[str], str | None, str | None]:
        base = [sys.executable, "-m", "tau3.cli"]
        if payload.type == "check_data":
            return base + ["check-data"], "banking_environment", None
        if payload.type in {"bundle_validate", "bundle_export"}:
            if not payload.bundle_id or payload.bundle_id not in self.store.bundles:
                raise ValueError("A known bundle_id is required")
            bundle = self.store.bundles[payload.bundle_id]
            bundle_path = str(bundle["root"])
            if payload.type == "bundle_validate":
                command = base + [
                    "synthesize",
                    "validate",
                    "--bundle",
                    bundle_path,
                    "--resume",
                ]
                if payload.offline:
                    command.append("--offline")
            else:
                command = base + ["synthesize", "export", "--bundle", bundle_path]
            return command, f"bundle:{payload.bundle_id}", None
        if payload.type == "results_review":
            if not payload.run_id or payload.run_id not in self.store.runs:
                raise ValueError("A known run_id is required")
            run = self.store.runs[payload.run_id]
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            output = run["path"].parent / f"results_reviewed-{stamp}.json"
            command = base + [
                "review",
                str(run["path"]),
                "--mode",
                payload.review_mode,
                "--output",
                str(output),
                "--max-concurrency",
                str(payload.max_concurrency),
            ]
            if payload.review_model:
                command += ["--review-model", payload.review_model]
            if payload.task_ids:
                command += ["--task-ids", *payload.task_ids]
            return command, f"run:{payload.run_id}", str(output)
        if payload.type == "trajectory_review":
            if payload.artifact is None:
                raise ValueError("An artifact is required for trajectory review")
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            review_dir = self.store.calibration_dir / "reviews"
            input_file = review_dir / f"trajectory-{stamp}.input.json"
            output = review_dir / f"trajectory-{stamp}.reviewed.json"
            write_json_atomic(input_file, self._trajectory_snapshot(payload))
            command = base + [
                "review",
                str(input_file),
                "--mode",
                payload.review_mode,
                "--output",
                str(output),
                "--max-concurrency",
                "1",
            ]
            if payload.review_model:
                command += ["--review-model", payload.review_model]
            artifact = payload.artifact
            namespace = (
                artifact.run_id or artifact.bundle_id or artifact.package_id or "global"
            )
            target = f"trajectory-review:{artifact.kind}:{namespace}:{artifact.id}"
            return command, target, str(output)
        raise ValueError("Unsupported job type")

    def _trajectory_snapshot(self, payload: JobCreate) -> dict:
        artifact = payload.artifact
        assert artifact is not None
        if artifact.kind == "simulation":
            if not artifact.run_id:
                raise ValueError("A run_id is required for a simulation review")
            run = self.store._require(self.store.runs, artifact.run_id, "run")
            simulation = self.store._load_run_simulation(run, artifact.id)
            if artifact.task_id and str(simulation.get("task_id")) != artifact.task_id:
                raise ValueError("The simulation does not belong to the requested task")
            source = self.store._load_run_raw(run)
            task = next(
                (
                    item
                    for item in source.get("tasks") or []
                    if str(item.get("id")) == str(simulation.get("task_id"))
                ),
                None,
            )
            if task is None:
                raise ValueError("The simulation task is missing from its results")
            info = source.get("info")
            timestamp = source.get("timestamp")
        elif artifact.kind == "trajectory":
            entry = self.store._require(
                self.store.trajectories, artifact.id, "trajectory"
            )
            if artifact.bundle_id and entry.get("bundle_id") != artifact.bundle_id:
                raise ValueError(
                    "The trajectory does not belong to the requested bundle"
                )
            if artifact.package_id and entry.get("package_id") != artifact.package_id:
                raise ValueError(
                    "The trajectory does not belong to the requested package"
                )
            if artifact.task_id and entry["task_id"] != artifact.task_id:
                raise ValueError("The trajectory does not belong to the requested task")
            if entry.get("package_id"):
                owner = self.store._require(
                    self.store.synthetic_packages, entry["package_id"], "package"
                )
                task = self.store._package_task(owner, entry["task_id"])
                raw = read_json_retry(entry["path"])
                simulation = raw.get("simulation")
                if not isinstance(simulation, dict):
                    raise ValueError("Trajectory artifact has no simulation object")
            else:
                owner = self.store._require(
                    self.store.bundles, entry["bundle_id"], "bundle"
                )
                record = self.store._require(owner["tasks"], entry["task_id"], "task")
                simulation = read_json_retry(entry["path"])
                task = record["task"]
            manifest = owner["manifest"]
            config = manifest.get("config") or {}
            policy = simulation.get("policy")
            if not policy and payload.review_mode == "full":
                raise ValueError("The trajectory has no saved agent policy")
            from tau3.user_prompt import get_global_user_sim_guidelines

            info = {
                "task_bundle": owner["relative_path"],
                "git_commit": manifest.get("git_commit") or "unknown",
                "num_trials": 1,
                "max_steps": config.get("max_steps") or 100,
                "max_errors": config.get("max_errors") or 10,
                "user_info": {
                    "implementation": "user_simulator",
                    "llm": config.get("user_model"),
                    "llm_args": config.get("llm_args"),
                    "global_simulation_guidelines": get_global_user_sim_guidelines(
                        use_tools=bool(task.get("user_tools"))
                    ),
                },
                "agent_info": {
                    "implementation": "llm_agent",
                    "llm": config.get("teacher_model"),
                    "llm_args": config.get("llm_args"),
                },
                "environment_info": {
                    "domain_name": manifest.get("domain") or "banking_knowledge",
                    "policy": policy or "",
                },
                "retrieval_config": config.get("retrieval_config"),
            }
            timestamp = simulation.get("timestamp")
        else:
            raise ValueError("Only simulation and trajectory artifacts can be reviewed")

        simulation = deepcopy(simulation)
        if simulation.get("trial") is None:
            simulation["trial"] = 0
        return {
            "timestamp": timestamp,
            "info": info,
            "tasks": [task],
            "simulations": [simulation],
        }

    def _run(self, job_id: str, command: list[str]) -> None:
        with self._lock:
            job = self.get(job_id)
            if job.status == "cancelled":
                return
        log_path = self.directory / job.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w") as stream:
                with self._lock:
                    if job.status == "cancelled":
                        return
                    process = subprocess.Popen(
                        command,
                        cwd=self.store.data_dir.parent,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        shell=False,
                        start_new_session=True,
                    )
                    self._processes[job_id] = process
                    job.pid = process.pid
                    job.status = "running"
                    job.updated_at = _now()
                    self._save(job)
                return_code = process.wait()
            with self._lock:
                if job.status != "cancelled":
                    job.status = "succeeded" if return_code == 0 else "failed"
                job.return_code = return_code
                job.updated_at = _now()
                self._processes.pop(job_id, None)
                self._save(job)
            self.store.refresh(force=True)
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.updated_at = _now()
                self._processes.pop(job_id, None)
                self._save(job)

    def _save(self, job: JobRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.directory / f"{job.id}.json", job.model_dump(mode="json")
        )
