"""Evaluation phase: ask every model every sampled question, grade the replies, aggregate per dataset.

CONCURRENCY MODEL (EVAL_CONCURRENCY = N)
  * One asyncio event loop for the whole phase. `N` worker coroutines per dataset pull questions from a shared
    iterator, so at most N requests are in flight and memory stays flat however many questions there are.
  * The HTTP request is awaited on the event loop (aiohttp), grading runs in a bounded thread pool: the regex work
    is instant, and sandboxed code execution blocks for seconds - it must never stall the loop that is driving
    the other N-1 requests. Code runs are further bounded by EVAL_SANDBOX_WORKERS.
  * N = 1 reproduces the old sequential behaviour exactly (same order, same counters, same output).
  * To get a speed-up the SERVER must serve requests in parallel: llama-server needs `--parallel N` (its context
    is split between the slots, which shrinks the per-question token budget the runner derives from n_ctx), and
    llama-swap must be allowed to route concurrent requests to the same model.

STREAMING, LOOP ABORT AND THE THINKING CAP (EVAL_STREAM, default on)
  Replies are read as a stream (streaming.py), so a runaway reply no longer has to run into the 10-minute timeout:
    * repetition loop        -> stopped within seconds, flagged "Fail: Degradation"
    * thinking budget used   -> stopped, flagged "Fail: Think cap"      (EVAL_THINK_BUDGET tokens of thinking)
    * timeout                -> stopped, with whatever had been written
  A stopped reply is handled in this order:  1) if its VISIBLE answer already contains a gradable answer, grade that
  (partial-answer salvage, EVAL_SALVAGE);  2) otherwise ask once more with thinking disabled (the existing No
  Thinking retry);  3) if that fails too the question counts as wrong and keeps its flag. Every stopped attempt is
  printed as it happens and kept, with the tail of what the model wrote, in eval_details_*.jsonl ("attempts").

WHAT THE s/q FIGURE MEANS NOW
  `sec` is the MEAN PER-QUESTION LATENCY (request + retries + grading), not dataset-wall-time / questions. The
  Agent score divides by it ("how long does one question take?"); if it were wall/n, running 8 requests at once
  would make every model look 8x faster and the score would no longer be comparable with earlier runs. The real
  elapsed time is reported separately as `wall`. Latency measured under load is somewhat higher than unloaded
  latency - keep EVAL_CONCURRENCY=1 for runs you want to compare with older results.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..client import get_ctx
from ..common import BenchError, BenchTimeout, fmt_hms, stamp_line
from ..config import EVAL_SPECS, FALLBACK_TOKENS
from .answers import extract_choice, strip_think
from .grading import grade_item
from .longctx import run_longctx
from .mathcheck import extract_math_answer
from .sandbox import sandbox_workers
from .streaming import FAIL_DEGRADATION, FAIL_THINK_CAP, StreamGuard, clamp_to_budget
from .toolcall import encode_reply
from .transport import open_transport


# --------------------------------------------------------------------------- #
# Per-request helpers
# --------------------------------------------------------------------------- #

def dynamic_max_tokens(cfg, kind, model_ctx, prompt):
    """max_tokens for ONE question: the model's context size minus the prompt size (dynamic_max).
    Falls back to FALLBACK_TOKENS when the context is unknown or the sum does not work out. If the user set
    EVAL_MAX_TOKENS_QA / _CODE explicitly, that number is used as an upper cap instead of the full dynamic_max.
    (math questions use the QA cap.)"""
    cap = cfg.eval_max_tokens_code if kind == "code" else cfg.eval_max_tokens_qa
    try:
        # No tokenizer is available here, so the prompt is estimated at ~1 token per 3 characters plus 64 for the
        # chat template. That slightly over-estimates, which keeps prompt + reply inside the context window.
        avail = int(model_ctx) - (len(prompt) // 3 + 64)
    except (TypeError, ValueError):
        avail = 0
    if avail <= 0:
        return cap or FALLBACK_TOKENS
    return min(cap, avail) if cap else avail


def tokens_label(v):
    return str(v) if v else "dynamic"


def no_thinking(payload):
    """A COPY of `payload` with thinking switched off (Qwen3-style chat template kwarg), merged with any
    chat_template_kwargs already in the payload. The original is never modified, so the switch cannot leak
    into the next question's first attempt."""
    p = dict(payload)
    p["chat_template_kwargs"] = {**(payload.get("chat_template_kwargs") or {}), "enable_thinking": False}
    return p


def _no_usable_answer(kind, item, text):
    """True when a reply that was CUT OFF contains nothing gradable (-> N/A instead of a plain wrong answer)."""
    if kind == "tool":
        return False                            # graded as it is: a cut-off reply without a call is simply wrong
    body = strip_think(text)
    if not body:
        return True
    if kind == "qa":
        return extract_choice(body, item["letters"]) is None
    if kind == "math":
        return extract_math_answer(body, numeric_fallback=False) is None
    return False


_FENCED = re.compile(r"```[^\n]*\n.*?```", re.S)


def has_complete_answer(kind, item, text):
    """True when the VISIBLE part of a stopped reply already holds an answer that can be graded: an explicit final
    answer for QA / math, a complete (closed) code block for code. Thinking text never counts."""
    if kind == "tool":
        return False
    body = strip_think(text)
    if not body:
        return False
    if kind == "qa":
        return extract_choice(body, item["letters"]) is not None
    if kind == "math":
        return extract_math_answer(body, numeric_fallback=False) is not None
    return bool(_FENCED.search(body))


@dataclass
class Reply:
    """One answer from the server, however it was read."""
    text: str = ""
    finish: str | None = None
    tokens: int = 0
    seconds: float = 0.0
    abort: str | None = None        # None | "loop" | "budget" | "timeout": we stopped reading
    loop: object = None             # streaming.LoopInfo when abort == "loop"
    tail: str = ""                  # last ~600 characters of what the model wrote (kept only for stopped replies)
    think_tokens: int = 0


_FB_LABEL = {"length": "cut off", "timeout": "timeout", "loop": "repetition loop", "budget": "thinking cap"}


@dataclass
class _Tally:
    n: int = 0              # questions finished (the score's denominator)
    correct: int = 0
    errors: int = 0         # API failures: counted as wrong
    trunc: int = 0          # replies cut off by the token limit
    na: int = 0             # cut off AND no usable answer: counted as wrong, tracked separately
    fallback: int = 0       # questions that needed the No Thinking retry
    fb_why: dict = None     # why: {"length": n, "timeout": n, "loop": n, "budget": n}
    degr: int = 0           # questions where ANY attempt fell into a repetition loop ("Fail: Degradation")
    degr_final: int = 0     # ... and the question still ended as a Degradation failure
    cap: int = 0            # questions where ANY attempt exceeded the thinking budget ("Fail: Think cap")
    cap_final: int = 0
    salv: int = 0           # questions graded from the partial answer of a stopped reply
    tok_sum: int = 0        # tokens generated (all attempts)
    consec: int = 0         # consecutive API failures, in completion order
    lat_sum: float = 0.0    # sum of per-question latencies
    last_reason: str = ""

    def __post_init__(self):
        self.fb_why = {}


# --------------------------------------------------------------------------- #
# One dataset for one model
# --------------------------------------------------------------------------- #

class DatasetRun:
    def __init__(self, cfg, transport, pool, model, key, items, sink, model_ctx):
        self.cfg, self.transport, self.pool = cfg, transport, pool
        self.model, self.key, self.items, self.sink, self.model_ctx = model, key, items, sink, model_ctx
        self.spec = EVAL_SPECS[key]
        self.timeout = max(cfg.eval_timeout, cfg.timeout_load)
        extra_kw = (cfg.eval_extra_body or {}).get("chat_template_kwargs")
        already_off = isinstance(extra_kw, dict) and extra_kw.get("enable_thinking") is False
        self.can_retry = cfg.eval_retry_on_length and not already_off     # nothing to switch off if the user did it
        # Tool calls are read non-streamed: streamed tool_call deltas differ between servers, and the replies are short.
        self.stream = cfg.eval_stream and self.spec["kind"] != "tool"
        self.tally = _Tally()
        self.live = sys.stdout.isatty()           # live counter only on a terminal; logs/cron just get the final line
        self._workers: list[asyncio.Task] = []
        self._aborted = False
        self._t0 = self._last_log = time.time()

    # -- one question ------------------------------------------------------
    def _payload(self, item):
        cfg = self.cfg
        max_tokens = clamp_to_budget(dynamic_max_tokens(cfg, self.spec["kind"], self.model_ctx, item["prompt"]),
                                     cfg.eval_think_budget)      # server-side backstop for the thinking budget
        p = {"model": self.model, "messages": item.get("messages") or [{"role": "user", "content": item["prompt"]}],
             "max_tokens": max_tokens, "temperature": cfg.eval_temperature}
        if item.get("tools"):
            p["tools"], p["tool_choice"] = item["tools"], "auto"
        if self.stream:
            p["stream_options"] = {"include_usage": True}        # exact token counts in the final event
        if cfg.eval_extra_body:
            p.update(cfg.eval_extra_body)
        p["stream"] = self.stream                                # EVAL_EXTRA_BODY must not switch the reader's mode
        return p

    async def _ask(self, payload):
        """One chat request -> Reply. A timeout is returned as abort="timeout" (with any partial text); every other
        failure raises BenchError."""
        t0 = time.time()
        if not self.stream:
            try:
                data = await self.transport.chat(payload, self.timeout)
            except BenchTimeout:
                return Reply(abort="timeout", seconds=time.time() - t0)
            ch = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            if "tools" in payload:                      # the grader needs the structured calls, not just the text
                text = encode_reply(text, msg.get("tool_calls"))
            return Reply(text=text, finish=ch.get("finish_reason"),
                         tokens=int(usage.get("completion_tokens") or 0), seconds=time.time() - t0)
        guard = StreamGuard(self.cfg.eval_think_budget, self.cfg.eval_loop_abort)
        try:
            res = await self.transport.stream(payload, self.timeout, guard)
        except asyncio.CancelledError:
            guard.cancel()                      # lets a worker thread stop reading
            raise
        return Reply(text=res.text, finish=res.finish, tokens=res.tokens, seconds=res.seconds, abort=res.abort,
                     loop=res.loop, tail=res.tail if res.abort else "", think_tokens=res.think_tokens)

    def _why(self, r):
        if r.abort == "loop":
            return f"repetition loop ({r.loop.detail})" if r.loop else "repetition loop"
        if r.abort == "budget":
            return f"{r.think_tokens:,} thinking tokens, budget {self.cfg.eval_think_budget:,}"
        return f"no result after {self.timeout:.0f}s"

    async def _attempt(self, payload, mode, attempts):
        """_ask + a summary line for eval_details_*.jsonl. Stopped replies keep the tail of what the model wrote."""
        try:
            r = await self._ask(payload)
        except BenchError as e:
            attempts.append({"mode": mode, "end": "error", "detail": str(e)})
            raise
        a = {"mode": mode, "end": r.abort or r.finish or "stop", "tokens": r.tokens, "seconds": round(r.seconds, 1)}
        if r.abort:
            a["detail"], a["tail"] = self._why(r), r.tail
        attempts.append(a)
        return r

    def _salvageable(self, kind, item, r):
        return self.cfg.eval_salvage and has_complete_answer(kind, item, r.text)

    async def _run_attempts(self, item, payload, attempts):
        """Attempt 1 (thinking as configured), then at most one more: partial-answer salvage OR the No Thinking retry.
        -> (reply, err, used_fallback, why, salvaged); `reply` is the one to grade."""
        kind = self.spec["kind"]
        err, reply, used_fallback, why, salvaged = None, None, False, "", False
        try:
            reply = await self._attempt(payload, "think", attempts)
        except BenchError as e:
            return None, e, False, "", False
        if reply.abort and self._salvageable(kind, item, reply):
            return reply, None, False, "", True                            # the visible answer is already there
        if self.can_retry and (reply.abort or reply.finish == "length"):
            used_fallback, why = True, reply.abort or "length"             # attempt 2 (the only retry): No Thinking
            try:
                second = await self._attempt(no_thinking(payload), "no-think", attempts)
            except BenchError as e:
                if reply.abort:         # attempt 1 was stopped and attempt 2 failed too: report the newer error.
                    err = e             # (If attempt 1 was just cut off, keep its reply and grade that.)
            else:
                reply = second
                salvaged = bool(reply.abort) and self._salvageable(kind, item, reply)
        return reply, err, used_fallback, why, salvaged

    async def _verdict(self, item, reply, salvaged):
        """Grade the chosen reply -> dict(ok, pred, reason, na, trunc, flag, finish)."""
        kind, cfg = self.spec["kind"], self.cfg
        v = {"ok": False, "pred": None, "reason": "", "na": False, "trunc": False, "flag": "", "finish": reply.finish}
        if reply.abort:
            v["finish"] = f"abort:{reply.abort}"
            if not salvaged:                                               # stopped, nothing gradable, no retry left
                v["flag"] = FAIL_DEGRADATION if reply.abort == "loop" else FAIL_THINK_CAP
                v["reason"] = f"{v['flag']}: {self._why(reply)}"
                return v
        elif reply.finish == "length":
            v["trunc"] = True
            if _no_usable_answer(kind, item, reply.text):
                v["na"], v["reason"] = True, "no answer: cut off by the token limit (N/A)"
                return v
        try:
            # grading off the event loop: sandboxed code can block for seconds. A stopped reply is graded as
            # "length": the last-number fallback of GSM8K must never see an unfinished text.
            v["ok"], v["pred"], v["reason"] = await asyncio.get_running_loop().run_in_executor(
                self.pool, grade_item, cfg, self.key, item, reply.text, "length" if reply.abort else reply.finish)
        except Exception as e:              # a grader/sandbox problem must never abort the whole benchmark
            v["reason"] = f"grader error: {e.__class__.__name__}: {e}"
        return v

    @staticmethod
    def _notes(item, attempts, salvaged):
        """One terminal line per attempt that was stopped, with what became of the question."""
        notes = []
        for i, a in enumerate(attempts):
            if a["end"] not in ("loop", "budget"):
                continue
            outcome = ("retried with thinking disabled" if i < len(attempts) - 1
                       else "graded from the partial answer" if salvaged else "counted as wrong")
            notes.append(f"[{FAIL_DEGRADATION if a['end'] == 'loop' else FAIL_THINK_CAP}] {item['id']}: {a['detail']}"
                         f" after ~{a['tokens']:,} tokens ({a['seconds']:.0f}s) -> {outcome}")
        return notes

    async def _evaluate(self, item):
        payload = self._payload(item)
        t1 = time.time()
        attempts = []
        reply, err, used_fallback, why, salvaged = await self._run_attempts(item, payload, attempts)
        if err is None and reply.abort == "timeout" and not salvaged:
            err = BenchTimeout(f"timeout after {self.timeout:.0f}s")       # same accounting as before: an API failure
        if err is not None:
            v = {"ok": False, "pred": None, "reason": f"api error: {err}", "na": False, "trunc": False, "flag": "",
                 "finish": None}
        else:
            v = await self._verdict(item, reply, salvaged)
        seconds = time.time() - t1
        text = reply.text if reply is not None else ""
        tokens = sum(a.get("tokens") or 0 for a in attempts)
        rec = {"model": self.model, "dataset": self.key, "id": item["id"], "correct": bool(v["ok"]), "na": v["na"],
               "thinking_fallback_used": used_fallback, "max_tokens": payload["max_tokens"], "answer": v["pred"],
               "gold": item.get("gold"), "reason": v["reason"], "finish": v["finish"], "flag": v["flag"],
               "salvaged": salvaged, "tokens": tokens, "seconds": round(seconds, 2), "attempts": attempts,
               "response": text[:4000]}
        return rec, {"err": err, "fallback": used_fallback, "fb_why": why, "trunc": v["trunc"], "na": v["na"],
                     "ok": v["ok"], "seconds": seconds, "degraded": any(a["end"] == "loop" for a in attempts),
                     "capped": any(a["end"] == "budget" for a in attempts), "salvaged": salvaged, "flag": v["flag"],
                     "tokens": tokens, "notes": self._notes(item, attempts, salvaged)}

    # -- bookkeeping (runs on the event loop between awaits, so it needs no lock) --------------
    def _say(self, msg):
        """A line between the live counter updates (a terminal) or just a line (a log file)."""
        print(("\r" if self.live else "") + "    " + msg + (" " * 12 if self.live else ""), flush=True)

    def _account(self, rec, f):
        t = self.tally
        t.fallback += f["fallback"]
        if f["fallback"]:
            t.fb_why[f["fb_why"]] = t.fb_why.get(f["fb_why"], 0) + 1
        t.degr += f["degraded"]
        t.cap += f["capped"]
        t.salv += f["salvaged"]
        t.degr_final += f["flag"] == FAIL_DEGRADATION
        t.cap_final += f["flag"] == FAIL_THINK_CAP
        t.tok_sum += f["tokens"]
        for note in f["notes"]:
            self._say(note)
        if f["err"] is not None:
            t.errors += 1
            t.consec += 1
            t.last_reason = rec["reason"]
        else:
            t.consec = 0
            t.trunc += f["trunc"]
        t.n += 1
        t.na += f["na"]
        t.correct += bool(f["ok"])
        t.lat_sum += f["seconds"]
        self.sink(rec)
        if self.live:
            print(f"\r    {self.spec['short']:<10} {t.n}/{len(self.items)}  {100 * t.correct / t.n:5.1f}%"
                  + (f"  ({t.na} N/A)" if t.na else "") + (f"  ({t.degr} loop)" if t.degr else ""), end="", flush=True)
        else:
            self._log_progress()
        if t.consec >= self.cfg.eval_max_consec_errors and not self._aborted:
            self._aborted = True                  # server is down / misconfigured: stop wasting time
            for w in self._workers:               # also drop requests still in flight
                if w is not asyncio.current_task():
                    w.cancel()

    def _log_progress(self, force=False):
        """Not a terminal (Unraid User Scripts, cron, nohup): there is no live counter, so print one plain line every
        EVAL_LOG_INTERVAL seconds (and at the start) so the log shows where a multi-hour run is up to."""
        every, now, t = self.cfg.eval_log_interval, time.time(), self.tally
        if every <= 0 or (not force and now - self._last_log < every):
            return
        self._last_log = now
        total, elapsed = len(self.items), now - self._t0
        eta = f", ~{fmt_hms(elapsed / t.n * (total - t.n))} left" if t.n else ""
        acc = f"{100 * t.correct / t.n:.1f}%" if t.n else "-"
        print(f"    {self.spec['short']:<10} {t.n}/{total}  {acc}  [{fmt_hms(elapsed)} elapsed{eta}]"
              + (f"  ({t.na} N/A)" if t.na else "") + (f"  ({t.degr} loop)" if t.degr else ""), flush=True)

    async def _worker(self, it):
        while not self._aborted:
            item = next(it, None)                 # shared iterator: safe, the loop is single-threaded
            if item is None:
                return
            rec, flags = await self._evaluate(item)
            self._account(rec, flags)

    async def run(self):
        cfg, t = self.cfg, self.tally
        t0 = self._t0 = self._last_log = time.time()
        if not self.live:
            self._log_progress(force=True)
        it = iter(self.items)
        n_workers = max(1, min(cfg.eval_concurrency, len(self.items)))
        self._workers = [asyncio.ensure_future(self._worker(it)) for _ in range(n_workers)]
        try:
            outcomes = await asyncio.gather(*self._workers, return_exceptions=True)
        except BaseException:                    # Ctrl-C / outer cancellation: stop every worker, then re-raise
            for w in self._workers:
                w.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            raise
        for o in outcomes:                        # a worker crashed (not a cancel from the abort logic)
            if isinstance(o, BaseException) and not isinstance(o, asyncio.CancelledError):
                raise o
        wall = time.time() - t0
        self._report(wall)
        n = t.n
        return {"pct": (100.0 * t.correct / n) if n - t.errors > 0 else None, "n": n, "err": t.errors, "na": t.na,
                "fallback": t.fallback, "sec": round(t.lat_sum / n, 1) if n else None,
                "trunc": (100.0 * t.trunc / n) if n else None,
                "degr": t.degr, "cap": t.cap, "salv": t.salv, "tok": round(t.tok_sum / n) if n and t.tok_sum else None,
                "wall": round(wall, 1), "conc": n_workers}

    def _report(self, wall):
        cfg, t, short = self.cfg, self.tally, self.spec["short"]
        n = t.n
        print(("\r" if self.live else "") + f"    {short:<10} {t.correct}/{n} = {100 * t.correct / n if n else 0:.1f}%"
              + (f"  [{t.na} N/A]" if t.na else "") + f"  ({fmt_hms(wall)})"
              + (f"  x{cfg.eval_concurrency} parallel" if cfg.eval_concurrency > 1 else "")
              + (" " * 12 if self.live else ""))
        if self._aborted:
            print(f"    [warn] stopped after {cfg.eval_max_consec_errors} consecutive API errors (last: {t.last_reason})")
        elif t.errors:
            print(f"    [warn] {t.errors} request(s) failed and were counted as wrong")
        self._report_stopped()
        if t.na:
            print(f"    [info] {t.na} reply(ies) had no usable answer (cut off by the token limit): N/A, counted as wrong")
        self._report_warnings()

    def _report_stopped(self):
        """Retries, repetition loops, the thinking cap and salvaged answers."""
        cfg, t = self.cfg, self.tally
        if t.fallback:
            why = ", ".join(f"{v} {_FB_LABEL.get(k, k)}" for k, v in sorted(t.fb_why.items(), key=lambda kv: -kv[1]))
            print(f"    [info] {t.fallback} question(s) were retried with No Thinking ({why})")
        if t.degr:
            print(f"    [fail] {FAIL_DEGRADATION}: {t.degr} question(s) fell into a repetition loop and were stopped early "
                  f"({t.degr - t.degr_final} recovered by the retry or the partial answer, {t.degr_final} counted as wrong)")
        if t.cap:
            print(f"    [info] {FAIL_THINK_CAP}: {t.cap} question(s) used more than the thinking budget of "
                  f"{cfg.eval_think_budget:,} tokens ({t.cap - t.cap_final} recovered, {t.cap_final} counted as wrong)")
        if t.salv:
            print(f"    [info] {t.salv} question(s) were graded from the partial answer of a stopped reply")

    def _report_warnings(self):
        cfg, t = self.cfg, self.tally
        n = t.n
        if not n:
            return
        if t.degr / n > 0.25:
            print(f"    [warn] {100 * t.degr / n:.0f}% of the questions fell into repetition loops. With a reasoning model this "
                  f"usually means greedy decoding (temperature {cfg.eval_temperature:g}): try EVAL_TEMPERATURE with the "
                  "sampling settings the model's authors recommend, or disable thinking.")
        if t.cap / n > 0.25:
            print(f"    [warn] {100 * t.cap / n:.0f}% of the questions exceeded the thinking budget. Raise EVAL_THINK_BUDGET "
                  "if you want long reasoning to count, or accept the cap as part of the score.")
        if t.trunc / n > 0.25:
            print(f"    [warn] {100 * t.trunc / n:.0f}% of replies were cut off at the token limit"
                  + (" even with thinking disabled" if self.can_retry else "")
                  + ". The model's context size (-c) is probably too small for this dataset"
                  + (" (with --parallel N the context is divided between the slots)." if cfg.eval_concurrency > 1 else "."))


# --------------------------------------------------------------------------- #
# The whole evaluation phase
# --------------------------------------------------------------------------- #

async def _run_all(cfg, session, results, data, stamp, path):
    todo = [r for r in results if r["status"] == "ok"]
    calls = sum(len(v) for v in data.values()) * len(todo)
    conc = cfg.eval_concurrency
    transport = await open_transport(cfg, conc)
    pool = ThreadPoolExecutor(max_workers=max(sandbox_workers(cfg), 2), thread_name_prefix="eval-grade")
    if cfg.eval_longctx and cfg.longctx_lengths:
        print(f"\nLong-context retrieval: {', '.join(f'{n // 1024}k' for n in cfg.longctx_lengths)} tokens, "
              f"{cfg.longctx_trials} prompt(s) x {cfg.longctx_keys} keys per length, KV at least {cfg.longctx_kv_min} "
              f"({'flagged' if cfg.longctx_kv_action == 'flag' else 'skipped'} below)")
    print(f"\nEvaluations: {len(todo)} model(s) x {len(data)} dataset(s) = {calls} requests "
          f"(max_tokens {tokens_label(cfg.eval_max_tokens_qa)} QA / {tokens_label(cfg.eval_max_tokens_code)} code, "
          f"temperature {cfg.eval_temperature:g}"
          + (", No Thinking retry" if cfg.eval_retry_on_length else ", no retry")
          + f"; HTTP {transport.name}, concurrency {conc})")
    if cfg.eval_stream:
        print("  streaming: thinking budget " + (f"{cfg.eval_think_budget:,} tokens" if cfg.eval_think_budget else "unlimited")
              + f", loop abort {'on' if cfg.eval_loop_abort else 'off'}, partial-answer salvage "
              + ("on" if cfg.eval_salvage else "off"))
    else:
        print("  [info] EVAL_STREAM=false: replies are not streamed, so loop abort, salvage and the live thinking "
              "budget are off (max_tokens still limits each reply)")
    if conc > 1:
        print("  [info] concurrency > 1: s/q is the mean per-question latency under load (see runner.py); "
              "keep EVAL_CONCURRENCY=1 for runs you want to compare with older results. "
              "llama-server needs --parallel >= EVAL_CONCURRENCY for this to speed anything up.")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            def sink(rec):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            mid = -(-len(todo) // 2)            # halfway point by model count; odd counts round up (5 models -> 3)
            for i, r in enumerate(todo, 1):
                print(f"[eval {i}/{len(todo)}] {r['model']}")
                ctx = r.get("ctx_size")                  # known from the speed test; evals-only runs look it up here
                if not ctx:
                    try:
                        ctx = await asyncio.to_thread(get_ctx, cfg, session, r["model"])
                    except Exception:
                        ctx = None
                if not ctx:
                    print(f"    [warn] context size unknown: using max_tokens {FALLBACK_TOKENS} for every question")
                if cfg.eval_longctx and cfg.longctx_lengths:
                    try:
                        await asyncio.to_thread(run_longctx, cfg, session, r, ctx, sink)
                    except Exception as e:             # optional check: never abort the evals for it
                        print(f"    [warn] long-context check failed: {e.__class__.__name__}: {e}")
                for key in cfg.eval_datasets:
                    if key not in data:
                        continue
                    res = await DatasetRun(cfg, transport, pool, r["model"], key, data[key], sink, ctx).run()
                    for field, v in res.items():
                        r[f"eval_{key}_{field}"] = v
                if i == mid:
                    stamp_line(f"Evals halfway ({i}/{len(todo)} models done)")
    finally:
        await transport.close()
        pool.shutdown(wait=False, cancel_futures=True)


def run_evals(cfg, session, results, data, stamp):
    """Evaluation phase: runs AFTER all speed tests, one model at a time (each model loads once).
    Fills r['eval_<dataset>_pct|n|err|na|fallback|sec|trunc|degr|cap|salv|tok|wall|conc'] and appends one JSON line per question to
    eval_details_<stamp>.jsonl. Returns that path."""
    path = cfg.output_dir / f"eval_details_{stamp}.jsonl"
    asyncio.run(_run_all(cfg, session, results, data, stamp, path))
    return path
