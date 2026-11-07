"""Streaming helpers: SSE parsing, repetition-loop detection, the thinking budget, partial-answer detection."""
import random

import pytest

from llmbench.common import BenchError
from llmbench.evals.runner import has_complete_answer
from llmbench.evals.streaming import (ANSWER_RESERVE, DONE, LoopDetector, StreamGuard, clamp_to_budget,
                                      parse_sse_line)


# --------------------------------------------------------------------------- SSE
def test_parse_sse_line():
    assert parse_sse_line(b"") is None and parse_sse_line(b": keep-alive") is None and parse_sse_line("event: x") is None
    assert parse_sse_line(b"data: [DONE]") is DONE
    assert parse_sse_line('data: {"choices": []}\n') == {"choices": []}
    assert parse_sse_line('{"choices": [1]}') == {"choices": [1]}            # bare JSON line
    with pytest.raises(BenchError, match="malformed"):
        parse_sse_line("data: {oops")
    with pytest.raises(BenchError, match="out of memory"):
        parse_sse_line('data: {"error": {"message": "out of memory"}}')


# --------------------------------------------------------------------------- loop detector
def feed_all(det, text, step=50, thinking=True):
    for i in range(0, len(text), step):
        hit = det.feed(text[i:i + step], thinking=thinking)
        if hit:
            return hit, i
    return None, None


def test_exact_repetition_is_detected_whatever_the_block_size():
    for block in ("abc", "Wait, let me double check that. ", ("x" * 899) + "\n"):
        hit, at = feed_all(LoopDetector(), block * (12000 // len(block) + 1), thinking=False)   # "repeat" test only
        assert hit and hit.kind == "repeat", block[:10]
        assert at < 9000                                                # caught well before 12k characters
    hit, _ = feed_all(LoopDetector(), (("y" * 1099) + "\n") * 12, thinking=False)   # period 1100 > window / 4: blind spot
    assert hit is None


def test_ordinary_text_never_triggers():
    rng = random.Random(7)
    vocab = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randrange(3, 9))) for _ in range(3000)]
    text = " ".join(" ".join(rng.choice(vocab) for _ in range(8)) + "." for _ in range(900))
    assert feed_all(LoopDetector(), text)[0] is None
    # an enumeration is legitimate reasoning, not a loop (digits make every line different)
    enum = "".join(f"Case {i}: x={i} gives {i * 3 + 1}, which is not divisible by 7. " for i in range(1, 300))
    assert feed_all(LoopDetector(), enum)[0] is None
    # a long run of similar-but-not-identical code lines in the ANSWER (thinking=False) must not trip the line test
    code = "".join(f"    assert f({i}) == {i * 2}  # check number {i}\n" for i in range(300))
    assert feed_all(LoopDetector(), code, thinking=False)[0] is None


def test_recycled_sentences_in_thinking_only():
    rng = random.Random(3)
    pool = ["Wait, let me recompute this carefully.", "Hmm, that gives the same result again.",
            "Let me verify once more with the formula.", "Actually, maybe I should double check the setup."]
    text = " ".join(rng.choice(pool) for _ in range(400))          # random order: NOT an exact repetition of a block
    hit, _ = feed_all(LoopDetector(), text, thinking=True)
    assert hit and hit.kind == "lines" and "different lines" in hit.detail
    assert feed_all(LoopDetector(), text, thinking=False)[0] is None   # not applied to the visible answer


def test_detector_state_is_bounded():
    det = LoopDetector()
    feed_all(det, "z" * 500000 + "".join(random.Random(1).choice("abcdef ") for _ in range(100)))
    assert len(det.buf) <= det.span


# --------------------------------------------------------------------------- guard
def r(t):
    return {"choices": [{"delta": {"reasoning_content": t}, "finish_reason": None}]}


def c(t, finish=None):
    return {"choices": [{"delta": {"content": t}, "finish_reason": finish}]}


def test_budget_counts_thinking_only():
    g = StreamGuard(think_budget=10, loop_detect=False)
    assert all(g.feed(r("tok ")) is None for _ in range(10))
    assert g.feed(r("tok ")) == "budget"
    res = g.result(seconds=1.5)
    assert res.abort == "budget" and res.finish is None and res.think_tokens == 11 and res.seconds == 1.5
    # once the visible answer starts, the thinking budget no longer applies (max_tokens still does)
    g = StreamGuard(think_budget=10, loop_detect=False)
    for _ in range(5):
        g.feed(r("tok "))
    for _ in range(200):
        assert g.feed(c("word ")) is None
    assert not g.thinking and g.think_tokens == 5


def test_inline_think_tags_count_as_thinking():
    g = StreamGuard(think_budget=5, loop_detect=False)
    for piece in ("<think>", "a ", "b ", "c ", "d "):
        assert g.feed(c(piece)) is None
    assert g.feed(c("e ")) == "budget"
    g = StreamGuard(think_budget=5, loop_detect=False)
    for piece in ("<think>", "a ", "b", "</think>"):
        g.feed(c(piece))
    assert not g.thinking
    for _ in range(50):
        assert g.feed(c("answer ")) is None


def test_unlimited_budget_and_exact_usage():
    g = StreamGuard(think_budget=0, loop_detect=False)
    for _ in range(500):
        assert g.feed(r("tok ")) is None
    g.feed({"choices": [], "usage": {"completion_tokens": 123}, "timings": {"predicted_per_second": 9.0}})
    g.feed(c("", finish="stop"))
    res = g.result()
    assert res.tokens == 123 and res.finish == "stop" and res.abort is None and res.timings["predicted_per_second"] == 9.0


def test_guard_flags_a_loop_and_keeps_the_tail():
    g = StreamGuard(think_budget=0, loop_detect=True)
    stop = None
    for _ in range(2000):
        stop = g.feed(r("Wait, let me double check that. "))
        if stop:
            break
    res = g.result()
    assert stop == "loop" and res.abort == "loop" and res.loop.kind in ("lines", "repeat") and "double check" in res.tail


def test_clamp_to_budget():
    assert clamp_to_budget(65000, 16384) == 16384 + ANSWER_RESERVE
    assert clamp_to_budget(1000, 16384) == 1000 and clamp_to_budget(65000, 0) == 65000


# --------------------------------------------------------------------------- salvage
def test_has_complete_answer():
    qa, math_ = {"letters": "ABCD"}, {}
    assert has_complete_answer("qa", qa, "Reasoning...\nAnswer: C\nLet me restate")
    assert not has_complete_answer("qa", qa, "<think>maybe it is B")                  # unfinished thinking
    assert not has_complete_answer("qa", qa, "I am still working it out")
    assert has_complete_answer("math", math_, "so the total is 12.\nAnswer: 12")
    assert not has_complete_answer("math", math_, "so the total is 12 and then")        # no explicit final answer
    assert has_complete_answer("code", {}, "Here:\n```python\ndef f():\n    return 1\n```\nand more")
    assert not has_complete_answer("code", {}, "Here:\n```python\ndef f():\n    retu")  # fence never closed: cannot run
