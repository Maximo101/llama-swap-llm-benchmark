"""Streaming support for the evaluation phase: SSE parsing, the per-reply guard, repetition-loop detection.

Everything here is pure logic with no network access, so it is unit-tested directly (selftests/test_streaming.py).
The transports (transport.py) feed events into a StreamGuard and stop reading as soon as it says so.

WHY STREAM THE EVALS
  A non-streaming request returns nothing until the model has finished. A reasoning model that falls into a loop
  therefore burns the whole EVAL_TIMEOUT (10 minutes by default) and the benchmark learns nothing about why. With
  a stream the benchmark sees every token as it is produced, so it can
    * stop a repetition loop within seconds            -> "Fail: Degradation"
    * stop a reply that exceeds the thinking budget    -> "Fail: Think cap"
    * keep whatever the model had written when it was stopped (partial-answer salvage)
    * close the connection, which makes llama-server stop generating (a hung-up non-streaming request is not
      reliably cancelled, so the abandoned generation can keep the GPU busy and slow down the next question)
  The speed tests are NOT streamed: they are short, and their numbers come from llama-server's `timings` block.

TOKEN COUNTS ARE ESTIMATES WHILE STREAMING
  llama-server sends one SSE event per token (several with speculative decoding / MTP), and only the final event
  carries exact `usage`. The thinking budget is therefore checked against max(events, characters / 4), which is
  close enough for a safety cap. When the stream finishes normally the exact completion_tokens is used.

THINKING vs ANSWER
  Reasoning counts as "thinking" when the server splits it out (`delta.reasoning_content`, or `delta.reasoning`) or
  when it arrives inline between <think> and </think>. A template that has already consumed the opening <think>
  tag cannot be told apart from an answer; for those replies only the loop detector and max_tokens apply.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..common import BenchError

FAIL_DEGRADATION = "Fail: Degradation"     # a repetition loop was detected and the reply was stopped
FAIL_THINK_CAP = "Fail: Think cap"         # the reply used more than EVAL_THINK_BUDGET tokens of thinking

ANSWER_RESERVE = 4096       # tokens allowed for the visible answer on top of the thinking budget (hard max_tokens)


def clamp_to_budget(max_tokens, budget):
    """Server-side backstop for the thinking budget: max_tokens never exceeds budget + ANSWER_RESERVE."""
    return min(max_tokens, budget + ANSWER_RESERVE) if budget and budget > 0 else max_tokens


# --------------------------------------------------------------------------- #
# Server-sent events
# --------------------------------------------------------------------------- #

class _Done:
    def __repr__(self):
        return "DONE"


DONE = _Done()      # the "data: [DONE]" sentinel


def parse_sse_line(raw):
    """One line of an event stream -> a dict (an event), DONE, or None (blank line, comment, other field).
    Raises BenchError for an in-stream error event or a line that is not valid JSON."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    line = raw.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        body = line[5:].strip()
    elif line.startswith("{"):                  # some servers send a bare JSON error without the "data:" prefix
        body = line
    else:
        return None                             # "event: ...", "id: ...", "retry: ..."
    if body == "[DONE]":
        return DONE
    try:
        ev = json.loads(body)
    except ValueError:
        raise BenchError("malformed event in the response stream")
    if not isinstance(ev, dict):
        return None
    err = ev.get("error")
    if err:
        raise BenchError(f"server error: {err.get('message') if isinstance(err, dict) else err}")
    return ev


# --------------------------------------------------------------------------- #
# Repetition-loop detection
# --------------------------------------------------------------------------- #

@dataclass
class LoopInfo:
    kind: str       # "repeat" = the tail is one block repeated; "lines" = nearly identical lines over and over
    detail: str


_SEG_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+")


def _min_period(s):
    """Smallest p such that s is its first p characters repeated (the last copy may be partial): KMP failure function."""
    n = len(s)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        c = s[i]
        while k and c != s[k]:
            k = pi[k - 1]
        if c == s[k]:
            k += 1
        pi[i] = k
    return n - pi[-1]


def _norm_line(s):
    """Compare lines ignoring case and spacing - but NOT digits: an enumeration ("Case 1: x=1 gives 4, not
    divisible by 7. Case 2: x=2 gives 7, ...") is legitimate reasoning, not a loop."""
    return re.sub(r"\s+", " ", s.lower()).strip()


class LoopDetector:
    """Incremental detector for degenerate output. Two independent tests, both deliberately conservative
    (a false alarm throws away a good answer, a missed loop only costs time):

      repeat  the last `window` characters are one block of at most window/min_repeats characters repeated over and
              over (exact repetition, including single-character runs). Applies to everything the model emits.
      lines   among the last `seg_count` lines/sentences of THINKING text (at least `seg_min_len` characters each)
              at most `seg_max_distinct` are different: the same few sentences recycled in any order ("Wait, let me
              recompute. Hmm, same result. Wait, let me recompute. ..."). Lines must match exactly (case and spacing
              aside), so a numbered enumeration is never mistaken for a loop. Not applied to the visible answer, where
              legitimately repetitive code (a table of asserts, say) could trigger it.
    Loops that keep changing the wording, or whose period is longer than window / min_repeats characters, are not
    caught here; the thinking budget stops those.
    Checks run every `check_every` new characters, so the cost is negligible next to generating them."""

    def __init__(self, window=4000, min_repeats=4, check_every=500,
                 seg_count=40, seg_max_distinct=10, seg_min_len=20, span=12000):
        self.window, self.min_repeats, self.check_every = window, min_repeats, check_every
        self.seg_count, self.seg_max_distinct, self.seg_min_len, self.span = seg_count, seg_max_distinct, seg_min_len, span
        self.buf = ""
        self._since = 0

    def feed(self, piece, thinking=True):
        self.buf = (self.buf + piece)[-max(self.span, self.window):]
        self._since += len(piece)
        if self._since < self.check_every:
            return None
        self._since = 0
        return self._repeat() or (self._lines() if thinking else None)

    def _repeat(self):
        tail = self.buf[-self.window:]
        n = len(tail)
        if n < self.window:
            return None
        probe = tail[-48:]
        if tail.rfind(probe, 0, n - 48) < 0:        # cheap C-speed pre-check: ordinary text never gets past here
            return None
        p = _min_period(tail)
        if p * self.min_repeats <= n:
            return LoopInfo("repeat", f"a {p}-character block repeated {n // p}+ times")
        return None

    def _lines(self):
        segs = [_norm_line(s) for s in _SEG_SPLIT.split(self.buf)]
        segs = [s for s in segs if len(s) >= self.seg_min_len]
        if len(segs) < self.seg_count:
            return None
        last = segs[-self.seg_count:]
        distinct = len(set(last))
        if distinct <= self.seg_max_distinct:
            return LoopInfo("lines", f"only {distinct} different lines in the last {self.seg_count}")
        return None


# --------------------------------------------------------------------------- #
# One streamed reply
# --------------------------------------------------------------------------- #

@dataclass
class StreamResult:
    text: str                   # the visible reply (delta.content)
    reasoning: str              # delta.reasoning_content, when the server splits it out
    finish: str | None          # the server's finish_reason; None when the stream was stopped by us
    abort: str | None           # None | "loop" | "budget" | "timeout"
    loop: LoopInfo | None
    tokens: int                 # exact when the server reported usage, else estimated
    think_tokens: int           # estimated tokens spent thinking
    seconds: float
    timings: dict | None = None
    prompt_tokens: int | None = None    # from the final usage event (None when the server did not report it)

    @property
    def tail(self):
        """The last bit of what the model wrote: the evidence kept in eval_details_*.jsonl for stopped replies."""
        return (self.reasoning or self.text)[-600:]


class StreamGuard:
    """Accumulates one streamed reply and decides when to stop reading it. Single-use; not thread-safe (each
    request owns its guard, and only that request's reader touches it)."""

    def __init__(self, think_budget=0, loop_detect=True):
        self.budget = think_budget if think_budget and think_budget > 0 else 0
        self._detector = LoopDetector() if loop_detect else None
        self._content, self._reasoning = [], []
        self.finish = self.usage = self.timings = None
        self.abort = None
        self.loop = None
        self.cancelled = False              # set from the async side so a worker thread stops reading
        self.events = 0                     # events that carried text
        self.chars = 0
        self.think_events = 0
        self.think_chars = 0
        self.chunks = 0                     # every event received (0 = nothing arrived yet, so a retry is safe)
        self._in_reasoning = False
        self._inline_think = False
        self._tag_tail = ""

    # -- state -----------------------------------------------------------
    @property
    def thinking(self):
        return self._in_reasoning or self._inline_think

    @property
    def think_tokens(self):
        return max(self.think_events, self.think_chars // 4)

    @property
    def tokens(self):
        if isinstance(self.usage, dict) and self.usage.get("completion_tokens"):
            return int(self.usage["completion_tokens"])
        return max(self.events, self.chars // 4)

    def cancel(self):
        self.cancelled = True

    def _track_inline(self, piece):
        self._tag_tail = (self._tag_tail + piece)[-40:]
        t = self._tag_tail
        opened = max(t.rfind("<think>"), t.rfind("<thinking>"))
        closed = max(t.rfind("</think>"), t.rfind("</thinking>"))
        if opened > closed:
            self._inline_think = True
        elif closed >= 0:
            self._inline_think = False

    # -- feeding ---------------------------------------------------------
    def feed(self, ev):
        """Process one event. Returns "loop" / "budget" when the reply must be stopped, else None."""
        self.chunks += 1
        if ev.get("usage"):
            self.usage = ev["usage"]
        if ev.get("timings"):
            self.timings = ev["timings"]
        for ch in ev.get("choices") or []:
            self._take(ch)
        if self.abort is None and self.budget and self.think_tokens > self.budget:
            self.abort = "budget"
        return self.abort

    def _take(self, ch):
        """One choice of an event: record the text, track thinking vs answer, run the loop detector."""
        d = ch.get("delta") or {}
        r = d.get("reasoning_content") or d.get("reasoning") or ""
        c = d.get("content") or ""
        if r:
            self._reasoning.append(r)
            self._in_reasoning = True
        if c:
            self._content.append(c)
            if c.strip():
                self._in_reasoning = False
            self._track_inline(c)
        if ch.get("finish_reason"):
            self.finish = ch["finish_reason"]
        text = r + c
        if not text:
            return
        self.events += 1
        self.chars += len(text)
        thinking = self.thinking
        if thinking:
            self.think_events += 1
            self.think_chars += len(text)
        if self._detector is not None and self.abort is None:
            hit = self._detector.feed(text, thinking=thinking)
            if hit:
                self.loop, self.abort = hit, "loop"

    def result(self, abort=None, seconds=0.0):
        """Freeze into a StreamResult. `abort` overrides (the transport passes "timeout")."""
        abort = self.abort or abort
        usage = self.usage if isinstance(self.usage, dict) else {}
        return StreamResult(text="".join(self._content), reasoning="".join(self._reasoning),
                            finish=None if abort else self.finish, abort=abort, loop=self.loop,
                            tokens=self.tokens, think_tokens=self.think_tokens, seconds=seconds,
                            timings=self.timings if isinstance(self.timings, dict) else None,
                            prompt_tokens=usage.get("prompt_tokens"))
