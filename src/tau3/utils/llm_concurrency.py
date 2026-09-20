"""Opt-in host-wide admission for synchronous LLM requests across processes."""

import fcntl
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


@contextmanager
def file_lock(path: Path):
    """Serialize threads and processes using separate flock descriptors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class RequestPool:
    """Crash-recoverable leases on one host; the shared limit is immutable."""

    def __init__(self, root: Path, limit: int):
        if limit < 1:
            raise ValueError("LLM concurrency must be positive")
        self.root, self.limit = root, limit
        self.host = socket.gethostname()
        with self._state():
            pass

    @contextmanager
    def _state(self):
        with file_lock(self.root / "pool.lock"):
            path = self.root / "status.json"
            state = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "host": self.host,
                    "limit": self.limit,
                    "active": {},
                    "peak": 0,
                    "admitted": 0,
                    "released": 0,
                    "reclaimed": 0,
                }
            )
            if (state["host"], state["limit"]) != (self.host, self.limit):
                raise ValueError("Shared LLM pool host/limit mismatch")
            identities = {
                lease["pid"]: _process_identity(lease["pid"])
                for lease in {
                    item["pid"]: item for item in state["active"].values()
                }.values()
            }
            for key, lease in list(state["active"].items()):
                if identities[lease["pid"]] != lease["start"]:
                    del state["active"][key]
                    state["reclaimed"] += 1
            yield state
            state["updated_at"] = time.time()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(path)

    def snapshot(self):
        """Return current leases and the observed high-water mark."""
        with self._state() as state:
            return dict(state)

    @contextmanager
    def slot(self):
        """Wait without an API timeout running; release on all Python exits."""
        token = uuid.uuid4().hex
        try:
            while True:
                admitted = False
                with self._state() as state:
                    if len(state["active"]) < self.limit:
                        state["active"][token] = {
                            "pid": os.getpid(),
                            "start": _process_identity(os.getpid()),
                            "since": time.time(),
                        }
                        state["admitted"] += 1
                        state["peak"] = max(state["peak"], len(state["active"]))
                        admitted = True
                if admitted:
                    break
                time.sleep(0.05)
            yield
        finally:
            with self._state() as state:
                if state["active"].pop(token, None) is not None:
                    state["released"] += 1


def configured_concurrency(default=2):
    """Read the opt-in request ceiling without changing model/seed identities."""
    value = int(os.environ.get("TAU3_LLM_CONCURRENCY", default))
    if value < 1:
        raise ValueError("TAU3_LLM_CONCURRENCY must be positive")
    return value


def install_request_limit():
    """Wrap the common completion boundary, including agents, users and judges.

    SDK retries retain their slot, giving a conservative physical-request bound.
    Every participating process must use the same local pool directory.
    """
    if "TAU3_LLM_CONCURRENCY" not in os.environ:
        return
    root = os.environ.get("TAU3_LLM_POOL")
    if not root:
        raise ValueError("TAU3_LLM_POOL is required for cross-process admission")
    from tau3.utils import llm_utils

    identity = (str(Path(root).resolve()), configured_concurrency())
    current = llm_utils.completion
    if getattr(current, "_tau3_pool", None) == identity:
        return
    if hasattr(current, "_tau3_pool"):
        raise ValueError("Cannot change an installed LLM pool")
    pool = RequestPool(Path(identity[0]), identity[1])

    @wraps(current)
    def limited(*args, **kwargs):
        with pool.slot():
            return current(*args, **kwargs)

    limited._tau3_pool = identity
    llm_utils.completion = limited
