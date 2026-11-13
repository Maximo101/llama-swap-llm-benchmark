"""v2.3 Agent quality: the composite, the context curve and the tool-calling minimum (pure logic, unit-tested).

The Agent role is an agentic reasoner (e.g. the Hermes agent): many tool calls and turns, long context. Its quality
figure is a weighted composite of three parts, each compared like for like across the ranked models:

    evals     mean of the dataset scores except tool calling              AGENT_*: the rest (default 0.45)
    tools     tool-calling accuracy                                       AGENT_TOOL_WEIGHT (0.30)
    long-ctx  long-context retrieval over the lengths ALL ranked models   AGENT_LONGCTX_WEIGHT (0.25)
              ran (a 90k model skips the 128k test; comparing everyone on the lengths they all ran keeps that
              from flattering it - the context curve carries the context preference instead)

A part the run does not have gives its weight back to the evals. If a ranked model has no long-context result at
all (it failed, or its KV was below LONGCTX_KV_MIN with LONGCTX_KV_ACTION=skip), long-context is left out of every
model's composite for this run and the reason is reported (the "basis" text).

The composite's standard error combines the binomial error of each part (weights squared), so the tie zone reflects
the real noise of what is ranked: tool calling has only 32 cases (+/-17 points) and long-ctx 8 keys per length.

Context (usable_ctx): the context a model can actually work with, not just the one it was loaded with. Walking the
long-context lengths of the run upwards, every length that fits the loaded context must reach AGENT_CTX_PASS_PCT
(75%); the usable context is the loaded context when all of them pass, else the longest length passed before the
first one that didn't (a 128k model that finds 12% at 128k works like a 64k model). A fitting length without a
result does not count as passed. Without long-context results in the run the loaded context is used.
Usable context below AGENT_CTX_MIN -> no Agent score; above it a mild factor (AGENT_CTX_CURVE, anchors ctx:factor,
linear between them, flat outside; default 128k 1.0, 77k 0.97, 64k 0.95) is applied after the tie zone, like the
retry penalty, so it also separates tied models. Tool calling below MIN_TOOLCALL_PCT gets no Agent score; within the tool set's margin of
error of that minimum (98/sqrt(n) points) the model is marked "borderline".
"""
from __future__ import annotations

import math
import re

from .config import EVAL_SPECS, LONGCTX_SLOTS

DEFAULT_CTX_CURVE = "131072:1.0,78848:0.97,65536:0.95"
DEFAULT_CTX_PASS = 75.0
TOOL_KEY = "toolcall"
LC_LABELS = list(LONGCTX_SLOTS.values())               # "32k", "64k", "90k", "128k"


# --------------------------------------------------------------------------- context
def parse_ctx_curve(raw):
    """'131072:1.0,92160:0.9,65536:0.75' (or '128k:1,90k:0.9') -> ((65536, 0.75), (92160, 0.9), (131072, 1.0)).
    Empty / 'off' -> () (no curve). Raises ValueError with a readable message."""
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw or raw.lower() in ("off", "none", "0", "false"):
        return ()
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k?)\s*:\s*(\d*\.?\d+)", part, re.I)
        if not m:
            raise ValueError(f"{part!r} is not ctx:factor (e.g. 131072:1.0 or 128k:1.0)")
        ctx = int(float(m.group(1)) * (1024 if m.group(2) else 1))
        f = float(m.group(3))
        if ctx <= 0 or not 0 <= f <= 1:
            raise ValueError(f"{part!r}: the context must be > 0 and the factor between 0 and 1")
        out[ctx] = f
    return tuple(sorted(out.items()))


def ctx_factor(ctx, curve):
    """0-1 factor for a loaded context; None when the context or the curve is unknown (then nothing is applied).
    Linear between neighbouring anchors, the top anchor's factor at or above it, the lowest anchor's below it."""
    if not ctx or not curve:
        return None
    if ctx >= curve[-1][0]:
        return curve[-1][1]
    if ctx <= curve[0][0]:
        return curve[0][1]
    for (c0, f0), (c1, f1) in zip(curve, curve[1:]):
        if c0 <= ctx <= c1:
            return f0 + (f1 - f0) * (ctx - c0) / (c1 - c0)
    return curve[-1][1]


def ctx_label(n):
    return f"{round(n / 1024)}k" if n and n >= 1024 else str(n)


LC_TOKENS = {label: tokens for tokens, label in LONGCTX_SLOTS.items()}


def run_lc_lengths(rows):
    """Long-context lengths (tokens) this run tested: those any model has a result for."""
    return sorted({LC_TOKENS[l] for r in rows for l in lc_results(r)})


def usable_ctx(r, cfg, run_lengths):
    """(usable context in tokens or None, why-not text or None).
    The loaded context, capped at the longest long-ctx length passed (>= AGENT_CTX_PASS_PCT) when a length that fits
    did not pass. None when the loaded context is unknown."""
    loaded = r.get("ctx_size")
    pass_pct = getattr(cfg, "agent_ctx_pass_pct", DEFAULT_CTX_PASS)
    if not loaded or not run_lengths or not pass_pct:
        return loaded, None
    lc = {LC_TOKENS[l]: p for l, p in lc_results(r).items()}
    passed = 0
    for length in run_lengths:
        if length > loaded:
            break                                          # does not fit: the loaded context decides above here
        p = lc.get(length)
        if p is None or p < pass_pct:
            why = (f"no long-ctx result at {ctx_label(length)}" if p is None else
                   f"long-ctx {p:.0f}% at {ctx_label(length)} is below AGENT_CTX_PASS_PCT {pass_pct:g}%")
            return passed, why
        passed = length
    return loaded, None


# --------------------------------------------------------------------------- the three parts
def eval_keys(rows):
    """Datasets of the run that count as 'evals' (everything except tool calling), and whether tools ran."""
    keys = [k for k in EVAL_SPECS if any(r.get(f"eval_{k}_pct") is not None or r.get(f"eval_{k}_n") for r in rows)]
    return [k for k in keys if k != TOOL_KEY], TOOL_KEY in keys


def lc_results(r):
    """{length label: pct} of the long-context results this model has."""
    return {l: r[f"longctx_{l}_pct"] for l in LC_LABELS if r.get(f"longctx_{l}_pct") is not None}


def common_lengths(rows):
    """(lengths every row has a long-context result for, models with no long-context result at all).
    Lengths come back in slot order (32k, 64k, 90k, 128k)."""
    if not rows:
        return [], []
    sets = [set(lc_results(r)) for r in rows]
    missing = [r["model"] for r, s in zip(rows, sets) if not s]
    common = set.intersection(*sets) if sets else set()
    return [l for l in LC_LABELS if l in common], missing


def weights(cfg, has_evals, has_tools, has_lc):
    """{part: weight} over the parts present, summing to 1. A missing part's weight goes back to the evals; if the
    evals are missing too (a tools-only run), the remaining parts are scaled up."""
    w_t = cfg.agent_tool_weight if has_tools else 0.0
    w_l = cfg.agent_longctx_weight if has_lc else 0.0
    w = {}
    if has_evals:
        w["evals"] = max(0.0, 1.0 - w_t - w_l)
    if has_tools:
        w["tools"] = w_t
    if has_lc:
        w["longctx"] = w_l
    total = sum(w.values())
    if not w:
        return {}
    if total <= 0:
        return {k: 1 / len(w) for k in w}
    return {k: v / total for k, v in w.items()}


def se_tie(p_pct, n):
    """Standard error (points) as the v2.2 tie zone computed it: p kept within 5-95% so a perfect small sample is
    not 'certain'."""
    if not n:
        return None
    p = min(0.95, max(0.05, p_pct / 100))
    return 100 * math.sqrt(p * (1 - p) / n)


def parts(r, ekeys, has_tools, lengths, lc_n_per_len):
    """{part: (value 0-100, [(pct, n) samples])} for one row, or None for a part the row lacks."""
    out = {}
    ev = [(r.get(f"eval_{k}_pct"), r.get(f"eval_{k}_n") or 0) for k in ekeys]
    out["evals"] = (sum(p for p, _ in ev) / len(ev), ev) if ev and all(p is not None for p, _ in ev) else None
    tp = r.get(f"eval_{TOOL_KEY}_pct")
    out["tools"] = (tp, [(tp, r.get(f"eval_{TOOL_KEY}_n") or 0)]) if has_tools and tp is not None else None
    lc = lc_results(r)
    if lengths and all(l in lc for l in lengths):
        v = sum(lc[l] for l in lengths) / len(lengths)
        out["longctx"] = (v, [(v, lc_n_per_len * len(lengths))])
    else:
        out["longctx"] = None
    return out


def composite(r, w, ekeys, has_tools, lengths, lc_n_per_len, se_fn=se_tie):
    """(composite 0-100, standard error in points) of one row under weights w; None when the row lacks a part.
    The evals part's error is that of a mean of K datasets: sqrt(sum SE_k^2) / K."""
    ps = parts(r, ekeys, has_tools, lengths, lc_n_per_len)
    val, var = 0.0, 0.0
    for part, wt in w.items():
        if ps.get(part) is None:
            return None
        v, samples = ps[part]
        ses = [se_fn(p, n) for p, n in samples]
        if any(s is None for s in ses):
            return None
        se = math.sqrt(sum(s * s for s in ses)) / len(ses)
        val += wt * v
        var += (wt * se) ** 2
    return val, math.sqrt(var)


# --------------------------------------------------------------------------- the ranked pool
def tool_margin(n):
    """95% margin of error of a tool-calling score from n cases, in points (as on the dashboard: 98/sqrt(n))."""
    return 98 / math.sqrt(n) if n else None


def exclusion(r, cfg, has_tools, run_lengths=()):
    """Why this (otherwise rankable) model gets no Agent score, or None."""
    ctx = r.get("ctx_size")
    if cfg.agent_ctx_min and ctx and ctx < cfg.agent_ctx_min:
        return (f"no Agent score: context {ctx_label(ctx)} is below AGENT_CTX_MIN {ctx_label(cfg.agent_ctx_min)}")
    use, why = usable_ctx(r, cfg, run_lengths)
    if cfg.agent_ctx_min and ctx and use is not None and use < cfg.agent_ctx_min:
        works = f"works up to {ctx_label(use)} only" if use else "no length passed"
        return f"no Agent score: {why} ({works}; AGENT_CTX_MIN {ctx_label(cfg.agent_ctx_min)})"
    tp = r.get(f"eval_{TOOL_KEY}_pct")
    if has_tools and cfg.min_toolcall_pct and tp is not None and tp < cfg.min_toolcall_pct:
        return f"no Agent score: tool calling {tp:.0f}% is below MIN_TOOLCALL_PCT {cfg.min_toolcall_pct:g}%"
    return None


def borderline(r, cfg):
    """True when the tool-calling score is within its margin of error of MIN_TOOLCALL_PCT (either side)."""
    tp, n = r.get(f"eval_{TOOL_KEY}_pct"), r.get(f"eval_{TOOL_KEY}_n")
    if tp is None or not n or not cfg.min_toolcall_pct:
        return None
    return abs(tp - cfg.min_toolcall_pct) <= tool_margin(n)


def basis(cfg, pool, all_rows):
    """Everything the composite needs for this pool, plus a short human-readable description:
    {"weights", "ekeys", "has_tools", "lengths", "lc_n", "text", "note"}. `all_rows` decides which parts the RUN has
    (a part no model has is simply not in the run; a part one ranked model lacks is a dropped part, with a note)."""
    ekeys, has_tools = eval_keys(all_rows)
    run_has_lc = any(lc_results(r) for r in all_rows)
    lengths, missing = common_lengths(pool) if run_has_lc else ([], [])
    note = None
    if run_has_lc and cfg.agent_longctx_weight > 0:
        if missing:
            shown = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
            note = f"long-ctx left out of the Agent score: no long-ctx result for {shown}"
        elif not lengths:
            note = "long-ctx left out of the Agent score: no length that every ranked model ran"
    has_lc = bool(lengths) and not missing and cfg.agent_longctx_weight > 0
    w = weights(cfg, bool(ekeys), has_tools, has_lc)
    lc_n = cfg.longctx_trials * cfg.longctx_keys
    names = {"evals": "evals", "tools": "tools", "longctx": "long-ctx"}
    text = " · ".join(f"{names[k]} {round(100 * v)}%" + (f" ({', '.join(lengths)})" if k == "longctx" else "")
                           for k, v in w.items())
    return {"weights": w, "ekeys": ekeys, "has_tools": has_tools, "lengths": lengths if has_lc else [],
            "lc_n": lc_n, "text": text + (f"; {note}" if note else ""), "note": note}
