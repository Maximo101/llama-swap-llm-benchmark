"""Async chat transports for the evaluation phase.

Both expose the same coroutine, so the runner does not care which one is in use:

    await transport.chat(payload, timeout)          -> parsed JSON dict   (raises BenchError / BenchTimeout)
    await transport.stream(payload, timeout, guard) -> StreamResult       (raises BenchError; a timeout is returned
                                                                           as abort="timeout" with the partial text)

`stream` reads the reply as server-sent events through a StreamGuard (streaming.py) and closes the connection the
moment the guard says stop (repetition loop / thinking budget) or the deadline passes, which also makes
llama-server stop generating.

  * AiohttpTransport - native asyncio, one pooled connection per in-flight request, no threads. Preferred.
  * ThreadTransport  - the same retry logic as the speed tests (client.post_chat, `requests`) run on a bounded
                       thread pool via concurrent.futures. Used when aiohttp is not installed or
                       EVAL_HTTP_BACKEND=threads.

Retry/classification rules are shared with the synchronous client (client.http_error / RETRY_STATUS / backoff_s),
so a 401, a 404 or a 503 is reported identically however the request was sent.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

from ..client import RETRY_STATUS, auth_headers, backoff_s, http_error, make_session, post_chat
from ..common import BenchError, BenchTimeout
from .streaming import DONE, parse_sse_line


def _stream_sync(cfg, session, payload, timeout, guard):
    """Blocking twin of AiohttpTransport.stream, run on a worker thread. `guard.cancelled` (set by the async side
    when the task is cancelled) stops the read loop so Ctrl-C does not leave a thread streaming for minutes."""
    t0 = time.monotonic()
    deadline, last = t0 + timeout, "server busy"
    for attempt in range(cfg.retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return guard.result("timeout", time.monotonic() - t0)
        resp = None
        try:
            try:
                resp = session.post(cfg.chat_url, json=payload, stream=True, timeout=(min(30.0, remaining), remaining))
            except requests.Timeout:
                return guard.result("timeout", time.monotonic() - t0)
            if resp.status_code >= 400:
                body = resp.text
                if resp.status_code in RETRY_STATUS and attempt < cfg.retries:
                    time.sleep(backoff_s(attempt))
                    continue
                raise http_error(cfg, resp.status_code, body)
            stop = None
            for raw in resp.iter_lines(chunk_size=None):
                if guard.cancelled:
                    stop = "timeout"
                    break
                if time.monotonic() > deadline:
                    stop = "timeout"
                    break
                ev = parse_sse_line(raw)
                if ev is None:
                    continue
                if ev is DONE:
                    break
                stop = guard.feed(ev)
                if stop:
                    break
            return guard.result(stop, time.monotonic() - t0)
        except requests.Timeout:                       # no bytes for the rest of the budget
            return guard.result("timeout", time.monotonic() - t0)
        except requests.RequestException as e:         # refused / reset / dropped mid-stream
            last = f"connection error: {e}"
            if guard.chunks == 0 and attempt < cfg.retries:
                time.sleep(backoff_s(attempt))
                continue
            raise BenchError(last)
        finally:
            if resp is not None:
                resp.close()
    raise BenchError(last)


class AiohttpTransport:
    name = "aiohttp"

    def __init__(self, cfg, concurrency):
        import aiohttp                      # imported lazily: aiohttp is optional
        self._aiohttp, self.cfg = aiohttp, cfg
        # Created inside the running loop (open_transport is a coroutine). limit = in-flight requests + headroom.
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=max(4, concurrency * 2), ttl_dns_cache=300),
            headers=auth_headers(cfg, cfg.backend))

    async def chat(self, payload, timeout):
        aiohttp, cfg, last = self._aiohttp, self.cfg, "server busy"
        for attempt in range(cfg.retries + 1):
            try:
                async with self.session.post(cfg.chat_url, json=payload,
                                             timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    status, body = resp.status, await resp.text()
            except asyncio.TimeoutError:
                raise BenchTimeout(f"timeout after {timeout:.0f}s")
            except aiohttp.ClientConnectionError as e:        # refused / reset / dropped mid-response
                last = f"connection error: {e}"
                if attempt < cfg.retries:
                    await asyncio.sleep(backoff_s(attempt))
                    continue
                raise BenchError(last)
            except aiohttp.ClientError as e:
                raise BenchError(f"connection error: {e}")
            if status in RETRY_STATUS and attempt < cfg.retries:
                await asyncio.sleep(backoff_s(attempt))
                continue
            err = http_error(cfg, status, body)
            if err:
                raise err
            try:
                return json.loads(body)
            except ValueError:
                raise BenchError("response was not valid JSON")
        raise BenchError(last)


    async def post_json(self, url, body, timeout):
        """POST any JSON body to `url` (llama-server native endpoints such as /apply-template and /completion)."""
        aiohttp = self._aiohttp
        try:
            async with self.session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                status, text = resp.status, await resp.text()
        except asyncio.TimeoutError:
            raise BenchTimeout(f"timeout after {timeout:.0f}s")
        except aiohttp.ClientError as e:
            raise BenchError(f"connection error: {e}")
        err = http_error(self.cfg, status, text)
        if err:
            raise err
        try:
            return json.loads(text)
        except ValueError:
            raise BenchError("response was not valid JSON")

    async def _pump(self, resp, guard, deadline):
        """Read events until the stream ends (None) or must be stopped ("loop" / "budget" / "timeout")."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            try:
                raw = await asyncio.wait_for(resp.content.readline(), remaining)
            except asyncio.TimeoutError:
                return "timeout"
            if not raw:
                return None
            ev = parse_sse_line(raw)
            if ev is None:
                continue
            if ev is DONE:
                return None
            stop = guard.feed(ev)
            if stop:
                return stop

    async def stream(self, payload, timeout, guard):
        aiohttp, cfg = self._aiohttp, self.cfg
        t0 = time.monotonic()
        deadline, last = t0 + timeout, "server busy"
        for attempt in range(cfg.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return guard.result("timeout", time.monotonic() - t0)
            resp = None
            try:
                try:
                    resp = await asyncio.wait_for(
                        self.session.post(cfg.chat_url, json=payload, timeout=aiohttp.ClientTimeout(total=None)),
                        remaining)
                except asyncio.TimeoutError:
                    return guard.result("timeout", time.monotonic() - t0)
                if resp.status >= 400:
                    body = await resp.text()
                    if resp.status in RETRY_STATUS and attempt < cfg.retries:
                        await asyncio.sleep(backoff_s(attempt))
                        continue
                    raise http_error(cfg, resp.status, body)
                stop = await self._pump(resp, guard, deadline)
                return guard.result(stop, time.monotonic() - t0)
            except (aiohttp.ClientError, ValueError) as e:        # refused / reset / dropped mid-stream / line too long
                last = f"connection error: {e}"
                if guard.chunks == 0 and attempt < cfg.retries:   # nothing received yet: safe to ask again
                    await asyncio.sleep(backoff_s(attempt))
                    continue
                raise BenchError(last)
            finally:
                if resp is not None:
                    resp.close()          # closes the socket: an aborted generation is cancelled server-side
        raise BenchError(last)

    async def close(self):
        await self.session.close()


class ThreadTransport:
    name = "threads"

    def __init__(self, cfg, concurrency):
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="eval-http")
        self._local = threading.local()          # one requests.Session per worker thread (Session is not thread-safe)

    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = make_session(self.cfg, self.cfg.backend)
            s.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
            s.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
            self._local.session = s
        return s

    async def chat(self, payload, timeout):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, lambda: post_chat(self.cfg, self._session(), payload, timeout))

    async def post_json(self, url, body, timeout):
        def call():
            try:
                resp = self._session().post(url, json=body, timeout=timeout)
            except requests.Timeout:
                raise BenchTimeout(f"timeout after {timeout:.0f}s")
            except requests.RequestException as e:
                raise BenchError(f"connection error: {e}")
            err = http_error(self.cfg, resp.status_code, resp.text)
            if err:
                raise err
            try:
                return resp.json()
            except ValueError:
                raise BenchError("response was not valid JSON")
        return await asyncio.get_running_loop().run_in_executor(self._pool, call)

    async def stream(self, payload, timeout, guard):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._pool, lambda: _stream_sync(self.cfg, self._session(), payload, timeout, guard))
        except asyncio.CancelledError:
            guard.cancel()
            raise

    async def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


async def open_transport(cfg, concurrency):
    """EVAL_HTTP_BACKEND: auto = aiohttp if installed, else threads; aiohttp = require it; threads = never use it."""
    want = cfg.eval_http_backend
    if want != "threads":
        try:
            import aiohttp  # noqa: F401
            return AiohttpTransport(cfg, concurrency)
        except ImportError:
            if want == "aiohttp":
                sys.exit("[eval] EVAL_HTTP_BACKEND=aiohttp but aiohttp is not installed. Run: pip install aiohttp")
    return ThreadTransport(cfg, concurrency)
