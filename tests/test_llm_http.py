"""Exercise actual socket concurrency and distinguish HTTP pool waiting."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from tau3.utils.llm_http import make_http_clients


def server(expected):
    """Hold responses until the required number of real HTTP requests arrive."""
    lock, release = threading.Lock(), threading.Event()
    state = {"active": 0, "peak": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                if state["active"] == expected:
                    release.set()
            try:
                reached = release.wait(20)
                body = b"ok" if reached else b"concurrency bottleneck"
                self.send_response(200 if reached else 503)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 512
        daemon_threads = True

    httpd = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, release, state


@pytest.mark.parametrize("concurrency", [128, 256])
def test_actual_parallel_http_requests_and_private_data_not_logged(
    tmp_path, monkeypatch, concurrency
):
    monkeypatch.setenv("TAU3_LLM_CONCURRENCY", str(concurrency))
    monkeypatch.setenv("TAU3_HTTP_TIMING_DIR", str(tmp_path))
    client, async_client = make_http_clients(10, 5)
    httpd, release, state = server(concurrency)
    url = f"http://127.0.0.1:{httpd.server_port}/v1/probe?secret=QUERY_CANARY"
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            replies = list(
                executor.map(
                    lambda _: client.get(
                        url, headers={"Authorization": "HEADER_CANARY"}, timeout=30
                    ),
                    range(concurrency),
                )
            )
        assert all(r.status_code == 200 for r in replies)
        assert state["peak"] == concurrency
    finally:
        release.set()
        client.close()
        asyncio.run(async_client.aclose())
        httpd.shutdown()
        httpd.server_close()
    text = next(tmp_path.glob("requests-*.jsonl")).read_text()
    assert "QUERY_CANARY" not in text and "HEADER_CANARY" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == concurrency
    assert all(
        r["response_body_consumed"] and r["pre_wire_seconds"] is not None for r in rows
    )


def test_async_transport_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU3_LLM_CONCURRENCY", "32")
    monkeypatch.setenv("TAU3_HTTP_TIMING_DIR", str(tmp_path))
    client, async_client = make_http_clients(10, 5)
    httpd, release, state = server(32)

    async def run():
        async with async_client:
            responses = await asyncio.gather(
                *[
                    async_client.get(
                        f"http://127.0.0.1:{httpd.server_port}/", timeout=30
                    )
                    for _ in range(32)
                ]
            )
        assert all(r.status_code == 200 for r in responses)

    try:
        asyncio.run(run())
        assert state["peak"] == 32
    finally:
        release.set()
        client.close()
        httpd.shutdown()
        httpd.server_close()


def test_pool_wait_is_recorded_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU3_LLM_CONCURRENCY", "1")
    monkeypatch.setenv("TAU3_HTTP_TIMING_DIR", str(tmp_path))
    client, async_client = make_http_clients(10, 5)
    httpd, release, state = server(2)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.get, url, timeout=5)
            deadline = time.monotonic() + 5
            while state["active"] != 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert state["active"] == 1
            with pytest.raises(httpx.PoolTimeout):
                client.get(url, timeout=httpx.Timeout(5, pool=0.1))
            release.set()
            assert first.result().status_code == 200
    finally:
        release.set()
        client.close()
        asyncio.run(async_client.aclose())
        httpd.shutdown()
        httpd.server_close()
    rows = [
        json.loads(line)
        for line in next(tmp_path.glob("requests-*.jsonl")).read_text().splitlines()
    ]
    failure = next(r for r in rows if r["error_type"] == "PoolTimeout")
    assert failure["pre_wire_seconds"] >= 0.09
    assert failure["network_response_seconds"] is None


def test_defaults_and_invalid_limit(monkeypatch):
    monkeypatch.delenv("TAU3_LLM_CONCURRENCY", raising=False)
    monkeypatch.delenv("TAU3_HTTP_TIMING_DIR", raising=False)
    client, async_client = make_http_clients(10, 5)
    assert client._transport._pool._max_connections == 10
    assert async_client._transport._pool._max_connections == 10
    client.close()
    asyncio.run(async_client.aclose())
    monkeypatch.setenv("TAU3_LLM_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="limits"):
        make_http_clients(10, 5)
