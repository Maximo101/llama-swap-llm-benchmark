"""A tiny OpenAI-compatible / llama-swap mock used by the self-tests (no GPU, no model needed).

Questions carry a marker so the mock knows what to answer:  [[ans:X]]  correct answer X
                                                           [[wrong]]   answer something else
                                                           [[long]]    first attempt is cut off (finish_reason=length);
                                                                       the No-Thinking retry succeeds
                                                           [[err]]     HTTP 500
Streaming-only markers (the first, thinking attempt misbehaves; the No-Thinking retry answers normally):
                                                           [[loop]]    reasoning repeats one sentence forever
                                                           [[ramble]]  endless NON-repeating reasoning (thinking cap)
                                                           [[partial]] reasoning, then the answer, then loops in the reply
                                                           [[hang]]    one token, then silence (client timeout)
"""
import asyncio
import json
import re
import threading

from aiohttp import web

LATENCY = 0.15          # seconds per chat request; makes concurrency speed-ups measurable
STATE = {"in_flight": 0, "max_in_flight": 0, "calls": 0, "flaky_503_left": 1, "requests": [], "disconnects": 0}


def _answer_for(prompt, thinking_off):
    if "[[err]]" in prompt:
        return None
    if "[[long]]" in prompt and not thinking_off:
        return {"content": "<think>hmm let me think about this for a very long time", "finish": "length"}
    m = re.search(r"\[\[ans:(.+?)\]\]", prompt)
    ans = m.group(1) if m else "A"
    if "[[wrong]]" in prompt:
        ans = "Z" if len(ans) == 1 and ans.isalpha() else "-12345"
    low = prompt.lower()
    if "def add(" in prompt:                                  # HumanEval
        return {"content": "```python\ndef add(a, b):\n    return a + b\n```", "finish": "stop"}
    if "def early(" in prompt:                                # HumanEval, program quits before the tests
        return {"content": "```python\nimport sys\ndef early(a):\n    return a\nsys.exit(0)\n```", "finish": "stop"}
    if "multiply" in low and "mbpp" in low or "pass these tests" in low:   # MBPP
        return {"content": "```python\ndef mul(a, b):\n    return a * b\n```", "finish": "stop"}
    if "### problem: double" in low:                          # LiveCodeBench (stdin)
        return {"content": "```python\nprint(int(input()) * 2)\n```", "finish": "stop"}
    if "math word problem" in low or "math problem" in low:   # GSM8K / MATH
        return {"content": f"Let me work it out.\n\nAnswer: {ans}", "finish": "stop"}
    return {"content": f"Reasoning...\nAnswer: {ans}", "finish": "stop"}        # multiple choice


def _sse(obj):
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def _delta(content=None, reasoning=None, finish=None):
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def _stream_plan(prompt, thinking_off):
    """Yield ('r'|'c', text) pieces, or ('sleep', seconds); the generator ends when the reply is complete."""
    if not thinking_off:
        if "[[loop]]" in prompt:
            for _ in range(100000):                         # word by word, like real tokens
                for w in "Wait, let me double check that calculation once more.".split():
                    yield "r", w + " "
            return
        if "[[ramble]]" in prompt:
            for i in range(100000):
                yield "r", f"consider{i} "
            return
        if "[[partial]]" in prompt:
            yield "r", "Hmm, thinking. "
            m = re.search(r"\[\[ans:(.+?)\]\]", prompt)
            yield "c", f"The result is clear.\nAnswer: {m.group(1) if m else 'A'}\n"
            for _ in range(100000):
                for w in "Let me restate the answer in other words, just to be sure.".split():
                    yield "c", w + " "
            return
        if "[[hang]]" in prompt:
            yield "r", "start "
            yield "sleep", 30
            return
    out = _answer_for(prompt, thinking_off)
    if out is None:
        return
    if not thinking_off:
        yield "r", "Let me think about this. "
    for w in re.findall(r"\S+\s*", out["content"]):
        yield "c", w
    yield "finish", out["finish"]


async def _chat_stream(request, body):
    prompt = body["messages"][0]["content"]
    off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    if _answer_for(prompt, off) is None and not any(k in prompt for k in ("[[loop]]", "[[ramble]]", "[[partial]]", "[[hang]]")):
        return web.Response(status=500, text="boom")
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    await asyncio.sleep(LATENCY)
    n, finish = 0, "stop"
    try:
        for kind, piece in _stream_plan(prompt, off):
            if kind == "sleep":
                for _ in range(int(piece * 10)):
                    if request.transport is None or request.transport.is_closing():
                        raise ConnectionResetError("client went away")
                    await asyncio.sleep(0.1)
                continue
            if kind == "finish":
                finish = piece
                continue
            n += 1
            await resp.write(_sse(_delta(content=piece) if kind == "c" else _delta(reasoning=piece)))
            await asyncio.sleep(0.0005)
        await resp.write(_sse(_delta(finish=finish)))
        await resp.write(_sse({"choices": [], "usage": {"completion_tokens": n, "prompt_tokens": 20},
                               "timings": {"predicted_per_second": 50.0, "predicted_n": n}}))
        await resp.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, ConnectionError):
        STATE["disconnects"] += 1                      # the client hung up on us: exactly what the loop abort does
    return resp


def _tool_reply(body):
    """Tool-calling eval: answer each built-in case correctly, except two deliberate mistakes (a wrong argument in
    simple-02, a call written as text in simple-03) so grading is exercised both ways."""
    from llmbench.evals.toolcall_cases import CASES
    prompt = body["messages"][-1]["content"]
    case = next((c for c in CASES if c["prompt"] == prompt), None)
    if case is None or not case["expect"]:
        return {"content": "Here is my answer without tools.", "tool_calls": None}
    calls = []
    for i, exp in enumerate(case["expect"]):
        args = {k: v[0] for k, v in exp["args"].items() if v[0] is not None}
        if case["id"] == "simple-02":
            args["amount"] = 25
        calls.append({"id": f"call_{i}", "type": "function",
                      "function": {"name": exp["name"], "arguments": json.dumps(args)}})
    if case["id"] == "simple-03":
        return {"content": "<tool_call>" + json.dumps({"name": calls[0]["function"]["name"],
                "arguments": json.loads(calls[0]["function"]["arguments"])}) + "</tool_call>", "tool_calls": None}
    return {"content": "", "tool_calls": calls}


def _vault_reply(prompt):
    """Long-context check: return the asked vaults' codes, read from the haystack."""
    codes = dict(re.findall(r"access code for vault (\w+) is (\d{6})", prompt))
    asked = re.search(r"each of these vaults: ([^.]+)\.", prompt).group(1).split(", ")
    return "\n".join(f"Vault {n}: {codes.get(n, '000000')}" for n in asked)


async def chat(request):
    body = await request.json()
    STATE["calls"] += 1
    STATE["requests"].append(body)
    if STATE["flaky_503_left"] > 0 and "hi" == body["messages"][0]["content"]:
        STATE["flaky_503_left"] -= 1
        return web.Response(status=503, text="swapping")
    STATE["in_flight"] += 1
    STATE["max_in_flight"] = max(STATE["max_in_flight"], STATE["in_flight"])
    try:
        if body.get("stream"):
            return await _chat_stream(request, body)
        await asyncio.sleep(LATENCY)
        prompt = body["messages"][-1]["content"]
        ptok = int(sum(len(str(m.get("content", "")).split()) for m in body["messages"]) * 1.2) + 10
        if body.get("tools"):
            t = _tool_reply(body)
            return web.json_response({"choices": [{"message": {"content": t["content"], "tool_calls": t["tool_calls"]},
                                                   "finish_reason": "tool_calls" if t["tool_calls"] else "stop"}],
                                      "usage": {"completion_tokens": 20, "prompt_tokens": ptok}})
        if "access code for vault" in prompt:
            return web.json_response({"choices": [{"message": {"content": _vault_reply(prompt)}, "finish_reason": "stop"}],
                                      "usage": {"completion_tokens": 20, "prompt_tokens": ptok}})
        off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        out = _answer_for(prompt, off)
        if out is None:
            return web.Response(status=500, text="boom")
        n = len(out["content"].split())
        return web.json_response({
            "choices": [{"message": {"content": out["content"]}, "finish_reason": out["finish"]}],
            "usage": {"completion_tokens": n, "prompt_tokens": ptok},
            "timings": {"predicted_per_second": 50.0, "prompt_per_second": 500.0 if ptok < 2000 else 300.0,
                        "prompt_ms": 1000.0 * ptok / 500.0, "predicted_ms": 1000.0 * n / 50.0, "predicted_n": n,
                        "prompt_n": ptok, "cache_n": 15}})
    finally:
        STATE["in_flight"] -= 1


async def models(request):
    return web.json_response({"data": [{"id": "Mock-9B-Q4_K_M.gguf"}, {"id": "Mock-30B-A3B-IQ3_XXS.gguf"}]})


CTX = {"n_ctx": 8192}


async def props(request):
    return web.json_response({"default_generation_settings": {"n_ctx": CTX["n_ctx"]},
                              "model_path": "/models/" + request.match_info["model"], "total_slots": 1})


async def unload(request):
    return web.json_response({"ok": True})


def make_app():
    app = web.Application()
    app.add_routes([web.post("/v1/chat/completions", chat), web.get("/v1/models", models),
                    web.get("/upstream/{model}/props", props), web.post("/api/models/unload", unload)])
    return app


class MockServer:
    """Runs the app on a background thread with its own event loop; `url` is ready after start()."""

    def start(self):
        self._ready, self.url = threading.Event(), None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(10), "mock server did not start"
        return self

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        runner = web.AppRunner(make_app())
        self._loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self._ready.set()
        self._loop.run_forever()

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
