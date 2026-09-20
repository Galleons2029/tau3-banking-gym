"""Exercise real thread/process admission without external model calls."""

import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tau3.utils.llm_concurrency import RequestPool, install_request_limit


def _contend(root, active, peak):
    pool = RequestPool(Path(root), 3)

    def request(_):
        with pool.slot():
            with active.get_lock():
                active.value += 1
                peak.value = max(peak.value, active.value)
            time.sleep(0.02)
            with active.get_lock():
                active.value -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(request, range(12)))


def _abandon(root):
    with RequestPool(Path(root), 1).slot():
        os._exit(0)


def test_shared_limit_across_processes_and_threads(tmp_path):
    ctx = mp.get_context("spawn")
    active, peak = ctx.Value("i", 0), ctx.Value("i", 0)
    children = [
        ctx.Process(target=_contend, args=(str(tmp_path), active, peak))
        for _ in range(3)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(60)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(5)
    state = RequestPool(tmp_path, 3).snapshot()
    assert peak.value == state["peak"] == 3
    assert not state["active"]
    assert state["admitted"] == state["released"] == 36


@pytest.mark.parametrize("limit", [128, 256])
def test_slots_are_usable_and_exceptions_release(tmp_path, limit):
    pool = RequestPool(tmp_path, limit)
    barrier = Barrier(limit, timeout=30)

    def request(_):
        with pytest.raises(RuntimeError), pool.slot():
            barrier.wait()
            raise RuntimeError("model failed")

    with ThreadPoolExecutor(max_workers=limit) as executor:
        list(executor.map(request, range(limit)))
    state = pool.snapshot()
    assert state["peak"] == limit
    assert not state["active"]
    assert state["released"] == limit


def test_process_death_recovers_lease_and_limit_cannot_change(tmp_path):
    child = mp.get_context("spawn").Process(target=_abandon, args=(str(tmp_path),))
    child.start()
    try:
        child.join(60)  # Spawn imports the package on the shared filesystem.
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
        child.join(5)
    pool = RequestPool(tmp_path, 1)
    with pool.slot():
        assert len(pool.snapshot()["active"]) == 1
    assert pool.snapshot()["reclaimed"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        RequestPool(tmp_path, 2)


def test_common_completion_boundary_is_wrapped_once(tmp_path, monkeypatch):
    from tau3.utils import llm_utils

    pool = RequestPool(tmp_path, 2)

    def completion(**kwargs):
        assert len(pool.snapshot()["active"]) == 1
        assert kwargs == {"model": "unchanged", "num_retries": 2}
        return "reply"

    monkeypatch.setenv("TAU3_LLM_CONCURRENCY", "2")
    monkeypatch.setenv("TAU3_LLM_POOL", str(tmp_path))
    monkeypatch.setattr(llm_utils, "completion", completion)
    install_request_limit()
    installed = llm_utils.completion
    install_request_limit()
    assert llm_utils.completion is installed
    assert installed(model="unchanged", num_retries=2) == "reply"
    assert not pool.snapshot()["active"]
