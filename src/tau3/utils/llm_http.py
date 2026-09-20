"""HTTP connection capacity and body-free request timing for LLM clients."""

import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx


class RequestTiming:
    """Measure local pre-wire waiting separately from network/response time.

    Pre-wire time includes connection-pool waiting and client bookkeeping. It is
    not gateway queue time. Response time includes gateway queuing and generation;
    these cannot be separated without server-side instrumentation.
    """

    def __init__(self, request, directory):
        self.request, self.directory = request, directory
        self.started = time.perf_counter()
        self.timestamp = time.time()
        self.wire = None
        self.previous = request.extensions.get("trace")

    def event(self, name):
        if self.wire is None and name in {
            "connection.connect_tcp.started",
            "connection.connect_unix_socket.started",
            "http11.send_request_headers.started",
            "http2.send_request_headers.started",
        }:
            self.wire = time.perf_counter()

    def trace(self, name, info):
        self.event(name)
        if self.previous:
            self.previous(name, info)

    async def atrace(self, name, info):
        self.event(name)
        if self.previous:
            await self.previous(name, info)

    def finish(self, response, error, streamed):
        elapsed = time.perf_counter() - self.started
        if self.previous is None:
            self.request.extensions.pop("trace", None)
        else:
            self.request.extensions["trace"] = self.previous
        before_wire = self.wire - self.started if self.wire is not None else None
        if isinstance(error, httpx.PoolTimeout):
            before_wire = elapsed
        record = {
            "id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "started_at": self.timestamp,
            "method": self.request.method,
            "host": self.request.url.host,
            "port": self.request.url.port,
            "path": self.request.url.path,
            "status_code": response.status_code if response is not None else None,
            "error_type": type(error).__name__ if error else None,
            "elapsed_seconds": elapsed,
            "pre_wire_seconds": before_wire,
            "network_response_seconds": elapsed - before_wire
            if self.wire is not None
            else None,
            "response_body_consumed": response.is_stream_consumed
            if response is not None
            else False,
            "stream_requested": streamed,
        }
        # No bodies, query strings, headers, credentials or error text are logged.
        path = self.directory / f"requests-{os.getpid()}.jsonl"
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError as exc:
            # Diagnostics must not turn a completed response into a model retry.
            logging.getLogger(__name__).warning(
                "Unable to write HTTP timing: %s", type(exc).__name__
            )


class TimedClient(httpx.Client):
    """Retain HTTPX proxy, retry and streaming semantics while observing send."""

    def __init__(self, *, timing_directory, **kwargs):
        super().__init__(**kwargs)
        self.timing_directory = timing_directory

    def send(self, request, **kwargs):
        span = RequestTiming(request, self.timing_directory)
        request.extensions["trace"] = span.trace
        response, error = None, None
        try:
            response = super().send(request, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            span.finish(response, error, kwargs.get("stream", False))


class TimedAsyncClient(httpx.AsyncClient):
    """Observe asynchronous requests with the same credential-free schema."""

    def __init__(self, *, timing_directory, **kwargs):
        super().__init__(**kwargs)
        self.timing_directory = timing_directory

    async def send(self, request, **kwargs):
        span = RequestTiming(request, self.timing_directory)
        request.extensions["trace"] = span.atrace
        response, error = None, None
        try:
            response = await super().send(request, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            span.finish(response, error, kwargs.get("stream", False))


def make_http_clients(default_connections, default_keepalive):
    """Match transport capacity to the explicit shared LLM concurrency setting."""
    configured = os.environ.get("TAU3_LLM_CONCURRENCY")
    maximum = int(configured) if configured is not None else default_connections
    keepalive = maximum if configured is not None else default_keepalive
    if maximum < 1 or not 0 <= keepalive <= maximum:
        raise ValueError("Invalid LLM HTTP connection limits")
    limits = httpx.Limits(max_connections=maximum, max_keepalive_connections=keepalive)
    directory = os.environ.get("TAU3_HTTP_TIMING_DIR")
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return (
            TimedClient(limits=limits, timing_directory=directory),
            TimedAsyncClient(limits=limits, timing_directory=directory),
        )
    return httpx.Client(limits=limits), httpx.AsyncClient(limits=limits)
