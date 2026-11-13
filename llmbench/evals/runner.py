"""Evaluation phase: ask every model every sampled question, grade the replies, aggregate per dataset.

RESULTS ARE RECORDS (v2.1)
  Every question produces one record (the line written to eval_details_*.jsonl). A dataset's score is ALWAYS
  aggregated from its records (aggregate()), whether they were asked just now, in an earlier funnel stage, or come from
  the result cache. That is what lets the funnel add questions stage by stage and a rerun skip questions already
  answered, with exactly the same arithmetic as a single uninterrupted run.

CONCURRENCY MODEL (EVAL_CONCURRENCY = N)
  * One asyncio event loop for the whole phase. Up to N worker coroutines per dataset pull questions from a shared
    iterator. N is capped PER MODEL (inflight_cap): never more than the server's slots (/props total_slots), and -
    when the slots share one KV pool (llama-server --kv-unified, the default when -np is left at auto) - never more
    requests than fit the pool at their worst case (prompt + max_tokens). A pool that overflows makes requests fail.
  * PIPELINED GRADING (v2.1): a worker hands the finished reply to a grading task and immediately asks the next
    question, so sandboxed code execution (seconds per program) overlaps with generation instead of adding to it.
    Grading tasks are bounded (backpressure), and the sandbox is unchanged: one isolated container per program.
  * N = 1 asks questions strictly one at a time; only grading overlaps.

STREAMING, LOOP ABORT AND THE THINKING CAP (EVAL_STREAM, default on)
  Replies are read as a stream (streaming.py), so a runaway reply no longer has to run into the 10-minute timeout:
    * repetition loop        -> stopped within seconds, flagged "Fail: Degradation"
    * thinking budget used   -> stopped, flagged "Fail: Think cap"   (EVAL_THINK_BUDGET, or EVAL_THINK_BUDGET_<DATASET>)
    * timeout                -> stopped, with whatever had been written
  A stopped reply is handled in this order:  1) if its VISIBLE answer already contains a gradable answer, grade that
  (partial-answer salvage, EVAL_SALVAGE);  2) at the thinking cap with EVAL_BUDGET_FORCING=true, close the thinking
  block and ask for the answer, keeping the reasoning already done ("budget forcing", Muennighoff et al. 2025);
  3) otherwise ask once more with thinking disabled (the No Thinking retry);  4) if that fails too the question counts
  as wrong and keeps its flag. Every attempt is kept in eval_details_*.jsonl ("attempts").

WHAT THE s/q FIGURE MEANS
  `sec` is the MEAN PER-QUESTION LATENCY (request + retries + grading), never dataset-wall-time / questions. Measured
  under concurrency it includes queueing behind the other requests, so v2.1 also derives `sec1`, the single-stream
  estimate (tokens / speed-test tok/s + prompt / prefill tok/s + grading). The Agent score uses `sec1` whenever any
  model ran with more than one request in flight (EVAL_SQ_BASIS=auto), so batching never changes a ranking.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import status
from ..client import get_ctx, get_props, is_outage, post_chat, props_ctx, upstream_url, wait_ready
from ..common import BenchError, BenchTimeout, add_time, avg, fmt_hms, stamp_line, now
from ..pause import CONTROL, PauseRestart
from ..config import (AUTO_BUDGET, EVAL_SPECS, FALLBACK_TOKENS, eval_temperature_for, funnel_stages, rows_for, sampling_label,
                      stage_rows, temperature_label, think_budget, think_budget_label, think_budget_basis)
from .answers import extract_choice, strip_think
from .cache import ResultCache, SanityCheck, model_fingerprint, question_key, same_answer
from .funnel import gate
from .grading import grade_item
from .longctx import run_longctx
from .mathcheck import extract_math_answer
from .sandbox import sandbox_workers
from .streaming import ANSWER_RESERVE, FAIL_DEGRADATION, FAIL_THINK_CAP, StreamGuard, clamp_to_budget
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


def _inline_reasoning(text):
    """Reasoning written inline: everything after the last opening <think> tag that was never closed."""
    m = list(re.finditer(r"<think(?:ing)?>", text or "", re.I))
    if not m:
        return ""
    tail = text[m[-1].end():]
    return "" if re.search(r"</think(?:ing)?>", tail, re.I) else tail


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
    reasoning: str = ""             # reasoning the server split out (reasoning_content)
    prompt_tokens: int | None = None
    ttft: float | None = None       # v2.3.1 (streamed replies): seconds to the first token / first answer token
    ttfa: float | None = None


_FB_LABEL = {"length": "cut off", "timeout": "timeout", "loop": "repetition loop", "budget": "thinking cap"}


# --------------------------------------------------------------------------- #
# Aggregation: records -> the per-dataset row fields
# --------------------------------------------------------------------------- #

def aggregate(recs, rows=None, conc=1, wall=0.0):
    """The dataset's result from its question records (same arithmetic as v2.0.0's running tally):
    pct = correct / n, but None when every request failed (no misleading 0 %); API errors count as wrong."""
    n = len(recs)
    s = lambda f: sum(1 for x in recs if x.get(f))
    err = s("api_error")
    tok = sum(x.get("tokens") or 0 for x in recs)
    think = sum(x.get("think_tokens") or 0 for x in recs)
    ptok = [x["prompt_tokens"] for x in recs if x.get("prompt_tokens")]
    # v2.3.1: what the client waited, as measured through llama-swap on the first attempt of each streamed question:
    # ttft = median seconds to the first token of any kind, ttfa = to the first VISIBLE answer token (after the
    # thinking), ttfa95 = its 95th percentile. Display only: no score uses them.
    first = sorted(x["ttft_s"] for x in recs if x.get("ttft_s") is not None)
    ans = sorted(x["ttfa_s"] for x in recs if x.get("ttfa_s") is not None)
    mid = lambda xs: round(xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 2)
    p95 = lambda xs: round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 2)
    return {"ttft": mid(first) if first else None, "ttfa": mid(ans) if ans else None,
            "ttfa95": p95(ans) if ans else None,
            "pct": (100.0 * s("correct") / n) if n - err > 0 else None, "n": n, "err": err, "na": s("na"),
            "fallback": s("thinking_fallback_used"),
            "sec": round(sum(x.get("seconds") or 0 for x in recs) / n, 1) if n else None,
            "trunc": (100.0 * s("trunc") / n) if n else None, "degr": s("degraded"), "cap": s("capped"),
            "salv": s("salvaged"), "forced": s("forced"), "tok": round(tok / n) if n and tok else None,
            "wall": round(wall, 1), "conc": conc, "rows": rows if rows is not None else n,
            "think": think, "answer": max(0, tok - think), "ptok": round(avg(ptok)) if ptok else None,
            "grade_s": round(sum(x.get("grade_s") or 0 for x in recs) / n, 3) if n else None,
            "cached": s("cached")}


# --------------------------------------------------------------------------- #
# What one model's evaluation needs
# --------------------------------------------------------------------------- #

class ModelEval:
    """Per model and stage: transport, grading pool, context, slots, cache fingerprint, chat-template facts."""

    def __init__(self, phase, r, ctx, props, fp):
        self.cfg, self.transport, self.pool = phase.cfg, phase.transport, phase.pool
        self.model, self.r, self.sink, self.cache = r["model"], r, phase.sink, phase.cache
        self.sanity, self.refresh = phase.sanity, phase.cfg.eval_cache_refresh
        self.ctx, self.props, self.fp = ctx, props or {}, fp
        self.slots = self.props.get("total_slots") or r.get("parallel")
        self.kv_unified = r.get("kv_unified")
        self.think_tags = "<think>" in (self.props.get("chat_template") or "")
        self.upstream = upstream_url(phase.cfg, r["model"])


def per_request_tokens(cfg, key, item, ctx):
    """Worst-case KV tokens one request can occupy: prompt + max_tokens (after the thinking-budget clamp)."""
    kind = EVAL_SPECS[key]["kind"]
    prompt = item.get("prompt") or ""
    mt = clamp_to_budget(dynamic_max_tokens(cfg, kind, ctx, prompt), think_budget(cfg, key, ctx))
    return len(prompt) // 3 + 64 + mt


def inflight_cap(cfg, me, key, items):
    """-> (requests in flight for this model and dataset, why it is lower than EVAL_CONCURRENCY or '')."""
    want = max(1, min(cfg.eval_concurrency, len(items) or 1))
    if want <= 1:
        return 1, ""
    why = []
    if me.slots and me.slots < want:
        want = int(me.slots)
        why.append(f"{me.slots} server slot(s)")
    if me.ctx and me.kv_unified is not False and want > 1:
        per = max(per_request_tokens(cfg, key, it, me.ctx) for it in items[:64])
        fit = max(1, me.ctx // per)
        if fit < want:
            want = fit
            why.append(f"KV pool {me.ctx:,} tokens / up to {per:,} per request"
                       + ("" if me.kv_unified else " (assumed shared: -kvu unknown)"))
    return want, "; ".join(why)


# --------------------------------------------------------------------------- #
# One dataset for one model (one stage)
# --------------------------------------------------------------------------- #

class _Tally:
    """Running counters for the live line and the per-stage report (the stored result comes from aggregate())."""

    def __init__(self):
        self.n = self.correct = self.errors = self.na = self.fallback = self.degr = self.degr_final = 0
        self.cap = self.cap_final = self.salv = self.forced = self.consec = self.cached = self.trunc = 0
        self.sanity = self.sanity_diff = 0
        self.fb_why = {}
        self.last_reason = ""


class DatasetRun:
    def __init__(self, me, key, items, stage="", inflight=1):
        self.me, self.cfg = me, me.cfg
        self.model, self.key, self.items, self.stage, self.inflight = me.model, key, items, stage, inflight
        self.spec = EVAL_SPECS[key]
        self.budget = think_budget(self.cfg, key, me.ctx)       # auto: a share of THIS model's context
        cfg = self.cfg
        self.timeout = max(cfg.eval_timeout, cfg.timeout_load)
        extra_kw = (cfg.eval_extra_body or {}).get("chat_template_kwargs")
        already_off = isinstance(extra_kw, dict) and extra_kw.get("enable_thinking") is False
        self.can_retry = cfg.eval_retry_on_length and not already_off     # nothing to switch off if the user did it
        # Tool calls are read non-streamed: streamed tool_call deltas differ between servers, and the replies are short.
        self.stream = cfg.eval_stream and self.spec["kind"] != "tool"
        self.can_force = cfg.eval_budget_forcing and self.stream and me.think_tags and cfg.backend != "strata"
        self.tally = _Tally()
        self.records = []
        self.live = sys.stdout.isatty()           # live counter only on a terminal; logs/cron just get the final line
        self._workers: list[asyncio.Task] = []
        self._pending: set = set()
        # backpressure: at most this many replies may wait for (or be in) grading; generation pauses beyond it
        self._grade_slots = asyncio.Semaphore(max(1, inflight) + max(2, 2 * sandbox_workers(cfg)))
        self._aborted = False
        self.interrupted = False                  # a pause stopped this run part-way: its records are to be thrown away
        self._t0 = self._last_log = now()

    # -- one question ------------------------------------------------------
    def _payload(self, item):
        cfg = self.cfg
        max_tokens = clamp_to_budget(dynamic_max_tokens(cfg, self.spec["kind"], self.me.ctx, item["prompt"]),
                                     self.budget)      # server-side backstop for the thinking budget
        p = {"model": self.model, "messages": item.get("messages") or [{"role": "user", "content": item["prompt"]}],
             "max_tokens": max_tokens}
        temp = eval_temperature_for(cfg, self.model)
        if temp is None:
            p["seed"] = cfg.eval_seed          # the model's own sampling: a fixed seed keeps one-at-a-time answers repeatable
        else:
            p["temperature"] = temp
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
        t0 = now()
        if not self.stream:
            try:
                data = await self.me.transport.chat(payload, self.timeout)
            except BenchTimeout:
                return Reply(abort="timeout", seconds=now() - t0)
            ch = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            if "tools" in payload:                      # the grader needs the structured calls, not just the text
                text = encode_reply(text, msg.get("tool_calls"))
            return Reply(text=text, finish=ch.get("finish_reason"), tokens=int(usage.get("completion_tokens") or 0),
                         seconds=now() - t0, reasoning=msg.get("reasoning_content") or "",
                         prompt_tokens=usage.get("prompt_tokens"))
        guard = StreamGuard(self.budget, self.cfg.eval_loop_abort)
        try:
            res = await self.me.transport.stream(payload, self.timeout, guard)
        except asyncio.CancelledError:
            guard.cancel()                      # lets a worker thread stop reading
            raise
        return Reply(text=res.text, finish=res.finish, tokens=res.tokens, seconds=res.seconds, abort=res.abort,
                     loop=res.loop, tail=res.tail if res.abort else "", think_tokens=res.think_tokens,
                     reasoning=res.reasoning, prompt_tokens=res.prompt_tokens, ttft=res.ttft, ttfa=res.ttfa)

    def _why(self, r):
        if r.abort == "loop":
            return f"repetition loop ({r.loop.detail})" if r.loop else "repetition loop"
        if r.abort == "budget":
            return f"{r.think_tokens:,} thinking tokens, budget {self.budget:,}"
        return f"no result after {self.timeout:.0f}s"

    async def _attempt(self, payload, mode, attempts):
        """_ask + a summary line for eval_details_*.jsonl. Stopped replies keep the tail of what the model wrote."""
        try:
            r = await self._ask(payload)
        except BenchError as e:
            attempts.append({"mode": mode, "end": "error", "detail": str(e)})
            raise
        a = {"mode": mode, "end": r.abort or r.finish or "stop", "tokens": r.tokens, "think": r.think_tokens,
             "seconds": round(r.seconds, 1)}
        if r.ttft is not None:
            a["ttft"] = round(r.ttft, 3)
        if r.ttfa is not None:
            a["ttfa"] = round(r.ttfa, 3)
        if r.abort:
            a["detail"], a["tail"] = self._why(r), r.tail
        attempts.append(a)
        return r

    async def _force(self, payload, reply, attempts):
        """Budget forcing: keep the reasoning written so far, close the thinking block and let the model answer.
        Uses llama-server's /apply-template + /completion. None when not possible (falls back to the No Thinking
        retry). Only for chat templates that use <think> tags."""
        reasoning = reply.reasoning or _inline_reasoning(reply.text)
        if not reasoning.strip():
            return None
        t0 = now()
        try:
            body = {"messages": payload["messages"]}
            if payload.get("chat_template_kwargs"):
                body["chat_template_kwargs"] = payload["chat_template_kwargs"]
            tpl = await self.me.transport.post_json(f"{self.me.upstream}/apply-template", body, 60)
            prompt = (tpl or {}).get("prompt")
            if not prompt:
                raise BenchError("apply-template returned no prompt")
            opened = prompt.rstrip().endswith("<think>")
            prompt += ("" if opened else "<think>\n") + reasoning.rstrip() + "\n</think>\n\n"
            body = {"prompt": prompt, "n_predict": ANSWER_RESERVE, "stream": False, "cache_prompt": True}
            temp = eval_temperature_for(self.cfg, self.model)
            body.update({"seed": self.cfg.eval_seed} if temp is None else {"temperature": temp})
            data = await self.me.transport.post_json(f"{self.me.upstream}/completion", body, self.timeout)
        except BenchError as e:
            attempts.append({"mode": "forced", "end": "error", "detail": str(e)})
            return None
        content = (data or {}).get("content") or ""
        limit = data.get("stopped_limit") or data.get("stop_type") == "limit"
        tokens = int(data.get("tokens_predicted") or len(content) // 4)
        attempts.append({"mode": "forced", "end": "length" if limit else "stop", "tokens": tokens, "think": 0,
                         "seconds": round(now() - t0, 1)})
        if not content.strip():
            return None
        return Reply(text=f"<think>{reasoning}</think>\n{content}", finish="length" if limit else "stop",
                     tokens=tokens, seconds=now() - t0, prompt_tokens=reply.prompt_tokens)

    def _salvageable(self, kind, item, r):
        return self.cfg.eval_salvage and has_complete_answer(kind, item, r.text)

    async def _run_attempts(self, item, payload, attempts):
        """Attempt 1 (thinking as configured), then at most one more: partial-answer salvage, budget forcing, OR the
        No Thinking retry. -> dict(reply, err, fallback, why, salvaged, forced); `reply` is the one to grade."""
        kind = self.spec["kind"]
        out = {"reply": None, "err": None, "fallback": False, "why": "", "salvaged": False, "forced": False}
        try:
            reply = await self._attempt(payload, "think", attempts)
        except BenchError as e:
            out["err"] = e
            return out
        out["reply"] = reply
        if reply.abort and self._salvageable(kind, item, reply):
            out["salvaged"] = True                                          # the visible answer is already there
            return out
        if reply.abort == "budget" and self.can_force:
            forced = await self._force(payload, reply, attempts)
            if forced is not None:
                out.update(reply=forced, forced=True)
                return out
        if self.can_retry and (reply.abort or reply.finish == "length"):
            out.update(fallback=True, why=reply.abort or "length")          # the only retry: No Thinking
            try:
                second = await self._attempt(no_thinking(payload), "no-think", attempts)
            except BenchError as e:
                if reply.abort:         # attempt 1 was stopped and attempt 2 failed too: report the newer error.
                    out["err"] = e      # (If attempt 1 was just cut off, keep its reply and grade that.)
            else:
                out["reply"] = second
                out["salvaged"] = bool(second.abort) and self._salvageable(kind, item, second)
        return out

    async def _verdict(self, item, reply, salvaged):
        """Grade the chosen reply -> dict(ok, pred, reason, na, trunc, flag, finish, grade_s)."""
        kind, cfg = self.spec["kind"], self.cfg
        v = {"ok": False, "pred": None, "reason": "", "na": False, "trunc": False, "flag": "", "finish": reply.finish,
             "grade_s": 0.0}
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
        t0 = now()
        try:
            # grading off the event loop: sandboxed code can block for seconds. A stopped reply is graded as
            # "length": the last-number fallback of GSM8K must never see an unfinished text.
            v["ok"], v["pred"], v["reason"] = await asyncio.get_running_loop().run_in_executor(
                self.me.pool, grade_item, cfg, self.key, item, reply.text, "length" if reply.abort else reply.finish)
        except Exception as e:              # a grader/sandbox problem must never abort the whole benchmark
            v["reason"] = f"grader error: {e.__class__.__name__}: {e}"
        v["grade_s"] = now() - t0
        return v

    @staticmethod
    def _notes(item, attempts, salvaged, forced):
        """One terminal line per attempt that was stopped, with what became of the question."""
        notes = []
        for i, a in enumerate(attempts):
            if a["end"] not in ("loop", "budget"):
                continue
            nxt = attempts[i + 1]["mode"] if i < len(attempts) - 1 else None
            outcome = ("answer forced after closing the thinking" if forced and nxt == "forced"
                       else "retried with thinking disabled" if nxt else
                       "graded from the partial answer" if salvaged else "counted as wrong")
            notes.append(f"[{FAIL_DEGRADATION if a['end'] == 'loop' else FAIL_THINK_CAP}] {item['id']}: {a['detail']}"
                         f" after ~{a['tokens']:,} tokens ({a['seconds']:.0f}s) -> {outcome}")
        return notes

    async def _finish(self, item, payload, gen, attempts, qkey, t_gen, check=None):
        """Grade (off the event loop), build the record, cache it, account it. Runs as its own task (pipelined).
        `check`: the cached record this question was re-asked against (EVAL_SANITY_PCT)."""
        try:
            reply, err = gen["reply"], gen["err"]
            if err is None and reply is not None and reply.abort == "timeout" and not gen["salvaged"]:
                err = BenchTimeout(f"timeout after {self.timeout:.0f}s")   # same accounting as before: an API failure
            if err is not None:
                v = {"ok": False, "pred": None, "reason": f"api error: {err}", "na": False, "trunc": False, "flag": "",
                     "finish": None, "grade_s": 0.0}
            else:
                v = await self._verdict(item, reply, gen["salvaged"])
            text = reply.text if reply is not None else ""
            rec = {"model": self.model, "dataset": self.key, "id": item["id"], "correct": bool(v["ok"]), "na": v["na"],
                   "thinking_fallback_used": gen["fallback"], "fallback_why": gen["why"],
                   "max_tokens": payload["max_tokens"], "answer": v["pred"], "gold": item.get("gold"),
                   "reason": v["reason"], "finish": v["finish"], "flag": v["flag"], "salvaged": gen["salvaged"],
                   "forced": gen["forced"], "api_error": err is not None, "trunc": v["trunc"],
                   "degraded": any(a["end"] == "loop" for a in attempts),
                   "capped": any(a["end"] == "budget" for a in attempts),
                   "tokens": sum(a.get("tokens") or 0 for a in attempts),
                   "think_tokens": sum(a.get("think") or 0 for a in attempts),
                   "prompt_tokens": reply.prompt_tokens if reply is not None else None,
                   "seconds": round(t_gen + v["grade_s"], 2), "grade_s": round(v["grade_s"], 3),
                   "ttft_s": (attempts[0].get("ttft") if attempts else None),
                   "ttfa_s": (attempts[0].get("ttfa") if attempts else None),
                   "stage": self.stage, "concurrency": self.inflight, "cached": False,
                   "attempts": attempts, "response": text[:4000]}
            notes = self._notes(item, attempts, gen["salvaged"], gen["forced"])
            if check is not None:
                if err is not None:            # the server failed, not the model: keep the cached answer, no verdict
                    check.update(cached=True, stage=self.stage)
                    self._account(check)
                    return
                same = same_answer(self.spec["kind"], rec, check)
                rec["sanity"] = "same" if same else "different"
                rec["sanity_cached"] = {"answer": check.get("answer"), "correct": bool(check.get("correct"))}
                self.me.sanity.result(self.model, same)
                self.tally.sanity += 1
                if not same:
                    self.tally.sanity_diff += 1
                    fmt = lambda a, ok: f"{str(a)[:40] if a is not None else '-'} ({'right' if ok else 'wrong'})"
                    notes = list(notes) + [f"[sanity] {item['id']}: cached {fmt(check.get('answer'), check.get('correct'))}"
                                           f", now {fmt(rec['answer'], rec['correct'])}; the new answer is scored"]
            if self.me.cache is not None and qkey:
                self.me.cache.store(self.model, self.me.fp, qkey, rec)
            self._account(rec, notes)
        finally:
            self._grade_slots.release()

    # -- bookkeeping (runs on the event loop between awaits, so it needs no lock) --------------
    def _say(self, msg):
        """A line between the live counter updates (a terminal) or just a line (a log file)."""
        print(("\r" if self.live else "") + "    " + msg + (" " * 12 if self.live else ""), flush=True)

    def _account(self, rec, notes=()):
        t = self.tally
        t.fallback += bool(rec.get("thinking_fallback_used"))
        if rec.get("thinking_fallback_used"):
            why = rec.get("fallback_why") or "length"
            t.fb_why[why] = t.fb_why.get(why, 0) + 1
        t.degr += bool(rec.get("degraded"))
        t.cap += bool(rec.get("capped"))
        t.salv += bool(rec.get("salvaged"))
        t.forced += bool(rec.get("forced"))
        t.degr_final += rec.get("flag") == FAIL_DEGRADATION
        t.cap_final += rec.get("flag") == FAIL_THINK_CAP
        t.cached += bool(rec.get("cached"))
        for note in notes:
            self._say(note)
        if rec.get("api_error"):
            t.errors += 1
            t.consec += 1
            t.last_reason = rec.get("reason", "")
        else:
            t.consec = 0
            t.trunc += bool(rec.get("trunc"))
        t.n += 1
        t.na += bool(rec.get("na"))
        t.correct += bool(rec.get("correct"))
        self.records.append(rec)
        self.me.sink(rec)
        status.update(dataset=self.spec["short"], done=t.n, total=len(self.items))
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
        every, t_now, t = self.cfg.eval_log_interval, now(), self.tally
        if every <= 0 or (not force and t_now - self._last_log < every):
            return
        self._last_log = t_now
        total, elapsed = len(self.items), t_now - self._t0
        eta = f", ~{fmt_hms(elapsed / t.n * (total - t.n))} left" if t.n else ""
        acc = f"{100 * t.correct / t.n:.1f}%" if t.n else "-"
        print(f"    {self.spec['short']:<10} {t.n}/{total}  {acc}  [{fmt_hms(elapsed)} elapsed{eta}]"
              + (f"  ({t.na} N/A)" if t.na else "") + (f"  ({t.degr} loop)" if t.degr else ""), flush=True)

    async def _worker(self, it):
        cfg, me = self.cfg, self.me
        while not self._aborted:
            if CONTROL.requested():               # a pause (by hand or MAX_GPU_TEMP): take no new question; the ones
                self.interrupted = True           # in flight finish and are graded (run() decides what that means)
                return
            item = next(it, None)                 # shared iterator: safe, the loop is single-threaded
            if item is None:
                return
            payload = self._payload(item)
            qkey, check = None, None
            if me.cache is not None and me.fp:
                qkey = question_key(cfg, self.key, item, payload, self.budget)
                hit = None if me.refresh else me.cache.lookup(self.model, me.fp, qkey)
                if hit is not None:
                    hit.pop("sanity", None)              # a verdict from an earlier run is not this run's check
                    hit.pop("sanity_cached", None)
                    if me.sanity is not None and me.sanity.pick(self.model):
                        check = hit                      # re-ask it and compare (EVAL_SANITY_PCT)
                    else:
                        hit.update(cached=True, stage=self.stage)
                        self._account(hit)
                        continue
            await self._grade_slots.acquire()     # wait here if grading has fallen behind
            t0 = now()
            attempts = []
            try:
                gen = await self._run_attempts(item, payload, attempts)
            except BaseException:
                self._grade_slots.release()
                raise
            task = asyncio.ensure_future(self._finish(item, payload, gen, attempts, qkey, now() - t0, check))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def run(self):
        """-> (records of this stage, wall seconds)."""
        t0 = self._t0 = self._last_log = now()
        if not self.live:
            self._log_progress(force=True)
        it = iter(self.items)
        self._workers = [asyncio.ensure_future(self._worker(it)) for _ in range(max(1, self.inflight))]
        try:
            outcomes = await asyncio.gather(*self._workers, return_exceptions=True)
            pending = await asyncio.gather(*list(self._pending), return_exceptions=True)
        except BaseException:                    # Ctrl-C / outer cancellation: stop every worker, then re-raise
            for w in self._workers + list(self._pending):
                w.cancel()
            await asyncio.gather(*self._workers, *list(self._pending), return_exceptions=True)
            raise
        for o in list(outcomes) + list(pending):  # a worker / grading task crashed (not a cancel from the abort logic)
            if isinstance(o, BaseException) and not isinstance(o, asyncio.CancelledError):
                raise o
        wall = now() - t0
        if self.interrupted and (self._aborted or len(self.records) >= len(self.items)):
            self.interrupted = False              # every question was taken before the pause: nothing is half done
        if self.interrupted:
            print(("\r" if self.live else "") + f"    {self.spec['short']:<10} paused at {len(self.records)}/"
                  f"{len(self.items)} questions: this dataset starts again from its first question when the run "
                  "carries on (answers already given come from the result cache)" + (" " * 12 if self.live else ""),
                  flush=True)
            return self.records, wall
        self._report(wall)
        return self.records, wall

    def _report(self, wall):
        cfg, t, short = self.cfg, self.tally, self.spec["short"]
        n = t.n
        print(("\r" if self.live else "") + f"    {short:<10} {t.correct}/{n} = {100 * t.correct / n if n else 0:.1f}%"
              + (f"  [{t.na} N/A]" if t.na else "") + f"  ({fmt_hms(wall)})"
              + (f"  x{self.inflight} parallel" if self.inflight > 1 else "")
              + (f"  [{t.cached} from cache]" if t.cached else "")
              + (f"  [sanity: {t.sanity} re-asked, {t.sanity_diff} different]" if t.sanity else "")
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
        """Retries, repetition loops, the thinking cap, forced answers and salvaged answers."""
        t = self.tally
        if t.fallback:
            why = ", ".join(f"{v} {_FB_LABEL.get(k, k)}" for k, v in sorted(t.fb_why.items(), key=lambda kv: -kv[1]))
            print(f"    [info] {t.fallback} question(s) were retried with No Thinking ({why})")
        if t.degr:
            print(f"    [fail] {FAIL_DEGRADATION}: {t.degr} question(s) fell into a repetition loop and were stopped early "
                  f"({t.degr - t.degr_final} recovered by the retry or the partial answer, {t.degr_final} counted as wrong)")
        if t.cap:
            print(f"    [info] {FAIL_THINK_CAP}: {t.cap} question(s) used more than the thinking budget of "
                  f"{self.budget:,} tokens ({t.cap - t.cap_final} recovered, {t.cap_final} counted as wrong)")
        if t.forced:
            print(f"    [info] {t.forced} question(s) were answered by budget forcing (thinking closed at the cap)")
        if t.salv:
            print(f"    [info] {t.salv} question(s) were graded from the partial answer of a stopped reply")

    def _report_warnings(self):
        cfg, t = self.cfg, self.tally
        n = t.n
        if not n:
            return
        if t.degr / n > 0.25:
            if eval_temperature_for(cfg, self.model) == 0:
                print(f"    [warn] {100 * t.degr / n:.0f}% of the questions fell into repetition loops. With a reasoning model "
                      "this usually means greedy decoding (temperature 0): put the sampling the model's authors recommend "
                      "in its llama-swap command and list it in EVAL_SERVER_SAMPLING_MODELS (or EVAL_TEMPERATURE=server "
                      "for every model), or disable thinking.")
            else:
                print(f"    [warn] {100 * t.degr / n:.0f}% of the questions fell into repetition loops "
                      f"({sampling_label(cfg, self.model, self.me.r.get('_launch_argv'))}): check the model's recommended "
                      "sampling and chat template, or disable thinking.")
        if t.cap / n > 0.25:
            print(f"    [warn] {100 * t.cap / n:.0f}% of the questions exceeded the thinking budget. Raise EVAL_THINK_BUDGET "
                  "if you want long reasoning to count, or accept the cap as part of the score.")
        if t.trunc / n > 0.25:
            print(f"    [warn] {100 * t.trunc / n:.0f}% of replies were cut off at the token limit"
                  + (" even with thinking disabled" if self.can_retry else "")
                  + ". The model's context size (-c) is probably too small for this dataset"
                  + (" (without --kv-unified, --parallel N divides it between the slots)." if self.inflight > 1 else "."))


# --------------------------------------------------------------------------- #
# The whole evaluation phase
# --------------------------------------------------------------------------- #

def longctx_done(cfg, r, ctx):
    """--resume: does this row already hold a long-context result for every configured length that fits?"""
    from ..config import LONGCTX_SLOTS
    for n in cfg.longctx_lengths:
        if ctx and n > ctx:
            continue
        if r.get(f"longctx_{LONGCTX_SLOTS[n]}_pct") is None:
            return False
    return bool(cfg.longctx_lengths)


def skip_reason(cfg, r):
    """EVAL_SKIP_INELIGIBLE: why this model's evals are pointless (it is already kept off the podium by a rule the
    speed phase measured), or None. auto = rules that keep a model off the podium; true = every failed budget rule."""
    if cfg.eval_skip_ineligible == "false" or r["status"] != "ok":
        return None
    why = []
    if r.get("ctx_ok") is False:
        why.append(f"ctx {r.get('ctx_size') or 0:,} below MIN_CTX {cfg.min_ctx or 0:,}")
    if r.get("budget_pass") is False and (cfg.eval_skip_ineligible == "true" or cfg.budget_podium_pass_only):
        why.append(f"budget {r.get('budget_profile')}: {r.get('budget_fails')}")
    vp = r.get("vram_pct")
    if vp is not None and vp > cfg.goldilocks_max_vram_pct:
        why.append(f"VRAM {vp:.0f}% above GOLDILOCKS_MAX_VRAM_PCT {cfg.goldilocks_max_vram_pct:g}%")
    return "; ".join(why) or None


def _stage_ids(cfg, data, stages):
    """{dataset: [set of question ids for each stage]} - nested prefixes of the seeded question order."""
    from .datasets import sample_indices
    out = {}
    for key, items in data.items():
        n, cand = (cfg.eval_pools or {}).get(key, (None, None))
        by_row = {it.get("_row"): it["id"] for it in items}
        sets = []
        for _, rows in stages:
            if n is None or None in by_row:
                sets.append({it["id"] for it in items})          # no row numbers (e.g. older callers): everything
            else:
                sets.append({by_row[i] for i in sample_indices(n, rows_for(key, rows), cfg.eval_seed, key, cand)
                             if i in by_row})
        out[key] = sets
    return out


class EvalPhase:
    def __init__(self, cfg, session, results, data, stamp, path, hardware="", on_model=None):
        self.on_model = on_model                 # called after every model of every stage (saves the run's rows)
        self.cfg, self.session, self.results, self.data = cfg, session, results, data
        self.stamp, self.path, self.hardware = stamp, path, hardware
        self.keys = [k for k in cfg.eval_datasets if k in data]
        self.stages = funnel_stages(cfg) if cfg.eval_profile == "funnel" else [(cfg.eval_profile, stage_rows(cfg))]
        self.store: dict = {}                    # model -> dataset -> {question id: record}
        self.funnel_report: list = []
        self.cache = ResultCache(cfg.eval_result_cache_dir) if cfg.eval_result_cache else None
        self.sanity = (SanityCheck(cfg.eval_sanity_pct, cfg.eval_sanity_adaptive)
                       if self.cache is not None and not cfg.eval_cache_refresh and cfg.eval_sanity_pct > 0 else None)
        self.transport = self.pool = self.sink = None
        self._fp_warned = False

    def _header(self, todo, skipped):
        cfg = self.cfg
        calls = sum(len(v) for v in self.data.values()) * len(todo)
        if cfg.eval_longctx and cfg.longctx_lengths:
            print(f"\nLong-context retrieval: {', '.join(f'{n // 1024}k' for n in cfg.longctx_lengths)} tokens, "
                  f"{cfg.longctx_trials} question(s) x {cfg.longctx_keys} keys per length on one shared document, "
                  f"KV at least {cfg.longctx_kv_min} ({'flagged' if cfg.longctx_kv_action == 'flag' else 'skipped'} below)")
        print(f"\nEvaluations: {len(todo)} model(s) x {len(self.data)} dataset(s)"
              + (f" = up to {calls} requests" if len(self.stages) == 1 else "")
              + f" (max_tokens {tokens_label(cfg.eval_max_tokens_qa)} QA / {tokens_label(cfg.eval_max_tokens_code)} "
              f"code, {temperature_label(cfg)}"
              + (", No Thinking retry" if cfg.eval_retry_on_length else ", no retry")
              + (", budget forcing" if cfg.eval_budget_forcing else "")
              + f"; HTTP {self.transport.name}, concurrency {cfg.eval_concurrency})")
        if len(self.stages) > 1:
            print("  funnel: " + " -> ".join(f"{n} ({r if r is not None else 'all'} rows)" for n, r in self.stages)
                  + f"; hard rules first (context / tool calling / long-context: no Agent score), then at each gate keep "
                  f"every model whose upper bound (+{cfg.eval_funnel_margin:g} margin) reaches the lower bound of model "
                  f"#{cfg.eval_leaders} ({100 * cfg.eval_funnel_conf:.0f}% one-sided, evals error only), never fewer "
                  f"than {cfg.eval_leaders}")
        if skipped:
            print(f"  skipped (EVAL_SKIP_INELIGIBLE={cfg.eval_skip_ineligible}): {len(skipped)} model(s) that cannot be "
                  "deployed under the active rules")
            for r in skipped:
                print(f"    - {r['model']}: {r['eval_skipped']}")
        if self.cache is not None and cfg.eval_cache_refresh:
            print(f"  result cache: refresh (EVAL_RESULT_CACHE=refresh): every question is asked again and the new "
                  f"answers replace the cached ones in {cfg.eval_result_cache_dir}")
        elif self.cache is not None:
            print(f"  result cache: {cfg.eval_result_cache_dir} (answers already obtained with identical settings are "
                  "reused; EVAL_RESULT_CACHE=refresh to ask everything again)")
            if self.sanity is not None:
                print(f"  sanity check: {cfg.eval_sanity_pct:g}% of cached answers are re-asked and compared"
                      + (" (per model, halved after each matching answer, back to the full rate after a different one)"
                         if cfg.eval_sanity_adaptive else "") + f"; seed {self.sanity.seed}")
        if cfg.eval_stream:
            budgets = {k: v for k, v in (cfg.eval_think_budgets or {}).items() if k in self.keys}
            print("  streaming: thinking budget " + think_budget_label(cfg.eval_think_budget, cfg)
                  + (" (" + ", ".join(f"{EVAL_SPECS[k]['short']} {v or 'unlimited'}" for k, v in budgets.items()) + ")"
                     if budgets else "")
                  + f", loop abort {'on' if cfg.eval_loop_abort else 'off'}, partial-answer salvage "
                  + ("on" if cfg.eval_salvage else "off"))
        else:
            print("  [info] EVAL_STREAM=false: replies are not streamed, so loop abort, salvage, budget forcing and the "
                  "live thinking budget are off (max_tokens still limits each reply)")
        if cfg.eval_concurrency > 1:
            print("  [info] concurrency > 1: needs llama-server slots (-np) and, to keep the full context per request, "
                  "--kv-unified. In-flight requests are capped per model to what the slots and KV pool can hold; the "
                  "Agent score then uses the single-stream s/q estimate, so batching cannot change a ranking.")

    async def _warm(self, r):
        """Load the model again after the server came back (it restarted, so nothing is loaded)."""
        t0 = now()
        try:
            await asyncio.to_thread(post_chat, self.cfg, self.session,
                                    {"model": r["model"], "max_tokens": 1, "stream": False,
                                     "messages": [{"role": "user", "content": "hi"}]}, self.cfg.timeout_load)
        except BenchError as e:
            print(f"    [warn] warm-up request failed ({e}); the questions will tell")
        add_time(r, "eval_load", now() - t0)

    async def _prepare(self, r):
        """Load the model with a tiny request (timed as eval_load, so it is never part of the first question's
        latency - v2.0.0 counted the model load in the first question's s/q), then read /props."""
        cfg, model = self.cfg, r["model"]
        t0 = now()
        try:
            await asyncio.to_thread(post_chat, cfg, self.session,
                                    {"model": model, "max_tokens": 1, "stream": False,
                                     "messages": [{"role": "user", "content": "hi"}]}, cfg.timeout_load)
        except BenchError as e:
            if is_outage(e) and await asyncio.to_thread(wait_ready, cfg, self.session):
                await self._warm(r)
            else:
                print(f"    [warn] warm-up request failed ({e}); the questions will tell")
        add_time(r, "eval_load", now() - t0)
        props = await asyncio.to_thread(get_props, cfg, self.session, model)
        ctx = r.get("ctx_size") or props_ctx(props)
        if not ctx:
            try:
                ctx = await asyncio.to_thread(get_ctx, cfg, self.session, model, props)
            except Exception:
                ctx = None
        if not ctx:
            print(f"    [warn] context size unknown: using max_tokens {FALLBACK_TOKENS} for every question")
        if props.get("total_slots") and not r.get("parallel"):
            r["parallel"] = props["total_slots"]
        fp = None
        if self.cache is not None:
            fp, strong = model_fingerprint(r, props, self.hardware or cfg.hw_gpu)
            if not strong and not self._fp_warned:
                self._fp_warned = True
                print("    [info] result cache: model file not readable from here, so cached answers are matched on "
                      "the server's report (path, ctx, build, template) rather than the file's own hash")
        return ModelEval(self, r, ctx, props, fp)

    async def _model_stage(self, r, si, stage_ids):
        cfg, model = self.cfg, r["model"]
        sname = self.stages[si][0]
        me = await self._prepare(r)
        if si == 0 and cfg.eval_stream and cfg.eval_think_budget == AUTO_BUDGET and self.keys:
            print(f"    thinking budget {think_budget(cfg, self.keys[0], me.ctx):,} tokens ({think_budget_basis(cfg, me.ctx)})")
        if si == 0 and cfg.eval_longctx and cfg.longctx_lengths:
            if r.get("_resumed") and longctx_done(cfg, r, me.ctx):
                print("    Long-ctx   kept from the resumed run")
            else:
                await self._longctx(r, me)
        done_model = self.store.setdefault(model, {})
        for key in self.keys:
            await CONTROL.await_resume(f"before {EVAL_SPECS[key]['short']} for {model}")     # a pause between datasets
            want = [it for it in self.data[key] if it["id"] in stage_ids[key][si]]
            done = done_model.setdefault(key, {})
            ask = [it for it in want if it["id"] not in done]
            inflight, why = inflight_cap(cfg, me, key, ask)
            if why and cfg.eval_concurrency > 1:
                print(f"    {EVAL_SPECS[key]['short']:<10} in flight: {inflight} (capped: {why})")
            r["eval_inflight"] = max(r.get("eval_inflight") or 1, inflight)
            if ask:
                recs, wall = await self._dataset(me, key, ask, sname, inflight)
                for rec in recs:
                    done[rec["id"]] = rec
                add_time(r, f"eval:{key}", wall)
                # v2.3.1: requests that failed because the SERVER was not there (refused, dropped, 5xx), and the
                # questions never asked because the dataset stopped on consecutive errors, are asked once more when
                # the server answers again. Timeouts are the model's own and stay as they are.
                lost = [it for it in ask if it["id"] in done and done[it["id"]].get("api_error")
                        and is_outage(done[it["id"]].get("reason"))]
                again = (lost + [it for it in ask if it["id"] not in done]) if lost else []
                if again and await asyncio.to_thread(wait_ready, cfg, self.session):
                    print(f"    {EVAL_SPECS[key]['short']:<10} asking {len(again)} question(s) again: the server did "
                          "not answer them")
                    await self._warm(r)
                    recs, wall = await self._dataset(me, key, again, sname, min(inflight, len(again)))
                    for rec in recs:
                        rec["reasked"] = True
                        done[rec["id"]] = rec
                    add_time(r, f"eval:{key}", wall)
                    r["eval_reasked_n"] = (r.get("eval_reasked_n") or 0) + len(again)
            # cached answers keep the concurrency they were measured at: the s/q basis must know about it
            inflight = max([inflight] + [int(x.get("concurrency") or 1) for x in done.values()])
            r["eval_inflight"] = max(r.get("eval_inflight") or 1, inflight)
            res = aggregate([done[i] for i in sorted(done)], rows=len(want), conc=inflight,
                            wall=(r.get("timing") or {}).get(f"eval:{key}", 0.0))
            for field, v in res.items():
                r[f"eval_{key}_{field}"] = v

    async def _dataset(self, me, key, items, sname, inflight):
        """One dataset run. A pause that stops it part-way throws it away; when the run carries on it is asked again from
        its first question. Answers already given are in the result cache, so only the unanswered ones cost time
        (with EVAL_RESULT_CACHE=false everything is asked again, which is what a fresh start means)."""
        short = EVAL_SPECS[key]["short"]
        while True:
            run = DatasetRun(me, key, items, sname, inflight)
            recs, wall = await run.run()
            if not run.interrupted:
                return recs, wall
            CONTROL.discard(f"{short} for {me.model}")
            await CONTROL.await_resume(f"{short} for {me.model}")

    def _longctx_thread(self, r, ctx, sink):
        with CONTROL.restartable():                    # too hot between two prompts: this test restarts (pause.py)
            run_longctx(self.cfg, self.session, r, ctx, sink)

    async def _longctx(self, r, me):
        """The long-context test of one model. If the GPU trips MAX_GPU_TEMP part-way the half-done test is dropped (the
        row goes back to how it was) and, after the pause, the test starts again from its first length."""
        cfg, snap, restarts = self.cfg, dict(r), 0
        while True:
            sink = self.sink if not restarts else (lambda rec, n=restarts: self.sink({**rec, "pause_restart": n}))
            try:
                for attempt in range(2):
                    try:
                        await asyncio.to_thread(self._longctx_thread, r, me.ctx, sink)
                    except Exception as e:             # optional check: never abort the evals for it
                        print(f"    [warn] long-context check failed: {e.__class__.__name__}: {e}")
                        break
                    # v2.3.1: the server was not there (not the model's fault): wait for it and test once more
                    if attempt or not is_outage(r.get("longctx_note")) \
                            or not await asyncio.to_thread(wait_ready, cfg, self.session):
                        break
                    await self._warm(r)
                return
            except PauseRestart:
                r.clear()
                r.update(snap)
                print("    [pause] the GPU got too hot during the long-context test: it is discarded")
                CONTROL.discard(f"the long-context test of {r['model']}")
                await CONTROL.await_resume(f"long-context test of {r['model']}")
                restarts += 1

    def _sanity_report(self, todo):
        checked = [r for r in todo if r.get("eval_sanity_n")]
        if not checked:
            print("Sanity check: nothing to check (no cached answers were used)")
            return
        n = sum(r["eval_sanity_n"] for r in checked)
        diff = sum(r["eval_sanity_diff"] for r in checked)
        print(f"Sanity check: {n} cached answer(s) re-asked, {diff} gave a different result")
        for r in checked:
            if r["eval_sanity_diff"]:
                print(f"    {r['model']}: {r['eval_sanity_diff']}/{r['eval_sanity_n']} different")
        if diff:
            print("    A different answer under identical settings usually means non-deterministic decoding (batched "
                  "requests, sampling at temperature > 0) rather than a stale cache; the new answers are scored.")

    def _stage1_avg(self, r, stage_ids):
        """Every model's score on the FIRST stage's questions only: the like-for-like comparison in a funnel run."""
        pcts = []
        for key in self.keys:
            recs = [x for i, x in (self.store.get(r["model"], {}).get(key) or {}).items() if i in stage_ids[key][0]]
            agg = aggregate(recs) if recs else None
            if not agg or agg["pct"] is None:
                return None
            pcts.append(agg["pct"])
        return avg(pcts)

    def _gate(self, alive, si):
        from ..scoring import finalize
        cfg = self.cfg
        finalize(self.results, cfg)                 # long-context averages etc. for the gate's score
        lc_n = cfg.longctx_trials * cfg.longctx_keys * max(1, len(cfg.longctx_lengths))
        keep, report, bar = gate(alive, self.keys, cfg.eval_leaders, cfg.eval_funnel_conf, cfg.agent_longctx_weight,
                                 lc_n, cfg=cfg)
        sname, nxt = self.stages[si][0], self.stages[si + 1][0]
        hard = [r for r in alive if report.get(r["model"], {}).get("state") == "no_agent"]
        shown = [r for r in alive if r["model"] in keep]
        print(f"\nFunnel gate after {sname}: {len(shown)} of {len(alive)} model(s) go on to {nxt}"
              + (f" (leaders' bar {bar:.2f}, margin {cfg.eval_funnel_margin:g})" if bar is not None else ""))
        if hard:
            print(f"    {len(hard)} model(s) fail a hard Agent rule (decided outside the funnel, not part of the bar): "
                  + ("not asked further stages" if cfg.eval_hard_fail == "stop" else "still asked (EVAL_HARD_FAIL=continue)"))
        order = {"keep": 0, "screened": 1, "no_agent": 2, "incomplete": 3}
        for r in sorted(alive, key=lambda r: (order.get(report.get(r["model"], {}).get("state"), 3),
                                              -(report.get(r["model"], {}).get("score") or -1))):
            rep = report.get(r["model"], {})
            st = rep.get("state") or ("keep" if r["model"] in keep else "screened")
            if rep.get("score") is not None:
                sc = f"{rep['score']:6.2f} +/-{rep['se_red']:.2f} (all parts +/-{rep['se']:.2f})"
                gap = f" gap {rep['gap']:+.2f}"
            else:
                sc, gap = "     -", ""
            label = {"keep": "KEEP        ", "screened": "SCREENED OUT", "no_agent": "NO AGENT SCORE",
                     "incomplete": "INCOMPLETE  "}.get(st, st)
            print(f"    {label}  {sc}{gap}  {r['model']}" + (f"  ({rep['why']})" if rep.get("why") else ""))
            self.funnel_report.append({"stage": sname, "model": r["model"], **{k: v for k, v in rep.items()}})
            if st == "no_agent":
                r["eval_hard_fail"] = rep["why"].replace("hard rule: ", "", 1)
                if r["model"] not in keep:
                    r["eval_stage"] = sname
            elif r["model"] not in keep:
                r["eval_stage"] = sname
                r["eval_screened"] = f"screened out at {sname}: {rep.get('why') or 'not among the leaders'}"
        return [r for r in alive if r["model"] in keep]

    async def run(self):
        cfg = self.cfg
        todo = [r for r in self.results if r["status"] == "ok"]
        if any(r.get("avg_tps") is not None for r in todo):
            from ..scoring import finalize
            finalize(self.results, cfg)             # budget checks / eligibility from the speed phase
        skipped = []
        for r in todo:
            why = skip_reason(cfg, r)
            if why:
                r["eval_skipped"] = why
                skipped.append(r)
        todo = [r for r in todo if not r.get("eval_skipped")]
        self.transport = await open_transport(cfg, cfg.eval_concurrency)
        self.pool = ThreadPoolExecutor(max_workers=max(sandbox_workers(cfg), 2), thread_name_prefix="eval-grade")
        self._header(todo, skipped)
        stage_ids = _stage_ids(cfg, self.data, self.stages)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                written = set()

                def sink(rec):
                    # a dataset restarted after a pause gets its earlier answers back from the result cache: they are in
                    # this file already, so they are not written twice
                    k = (rec.get("model"), rec.get("dataset"), rec.get("id"), rec.get("stage"))
                    if rec.get("cached") and k in written:
                        return
                    written.add(k)
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
                self.sink = sink
                alive = todo
                for si, (sname, srows) in enumerate(self.stages):
                    last = si == len(self.stages) - 1
                    if len(self.stages) > 1:
                        print(f"\n=== Funnel stage {si + 1}/{len(self.stages)}: {sname} "
                              f"({srows if srows is not None else 'all'} rows per dataset), {len(alive)} model(s) ===")
                        stamp_line(f"Funnel stage {sname} started")
                    mid = -(-len(alive) // 2)
                    for i, r in enumerate(alive, 1):
                        print(f"[eval {i}/{len(alive)}] {r['model']}" + (f"  [{sname}]" if len(self.stages) > 1 else ""))
                        status.update(force=True, phase="evals", model=r["model"], model_i=i, models=len(alive),
                                      stage=sname if len(self.stages) > 1 else None, dataset=None, done=None, total=None)
                        await CONTROL.await_resume(f"before the evals of {r['model']}")     # a pause between models
                        await self._model_stage(r, si, stage_ids)
                        if self.on_model:
                            self.on_model()
                        if i == mid and len(self.stages) == 1:
                            stamp_line(f"Evals halfway ({i}/{len(alive)} models done)")
                    if last:
                        if len(self.stages) > 1:
                            for r in alive:
                                r["eval_stage"] = sname
                    elif alive:
                        alive = self._gate(alive, si)
                if len(self.stages) > 1:
                    for r in todo:
                        r["eval_stage1_avg_pct"] = self._stage1_avg(r, stage_ids)
                for r in todo:
                    r["eval_cached_n"] = sum(r.get(f"eval_{k}_cached") or 0 for k in self.keys)
                    r["eval_sampling"] = sampling_label(self.cfg, r["model"], r.get("_launch_argv"))
                    if self.sanity is not None:
                        recs = [x for d in self.store.get(r["model"], {}).values() for x in d.values()]
                        r["eval_sanity_n"] = sum(1 for x in recs if x.get("sanity"))
                        r["eval_sanity_diff"] = sum(1 for x in recs if x.get("sanity") == "different")
                cached = sum(r["eval_cached_n"] for r in todo)
                if cached:
                    print(f"\nResult cache: {cached} answer(s) reused instead of asked again")
                if self.sanity is not None:
                    self._sanity_report(todo)
        finally:
            await self.transport.close()
            self.pool.shutdown(wait=False, cancel_futures=True)


def run_evals(cfg, session, results, data, stamp, hardware="", on_model=None):
    """Evaluation phase: runs AFTER all speed tests, one model at a time (each model loads once per stage).
    Fills r['eval_<dataset>_<field>'] (aggregate()) and appends one JSON line per question to
    eval_details_<stamp>.jsonl. Returns (that path, the funnel report list)."""
    path = cfg.output_dir / f"eval_details_{stamp}.jsonl"
    phase = EvalPhase(cfg, session, results, data, stamp, path, hardware, on_model)
    asyncio.run(phase.run())
    return path, phase.funnel_report
