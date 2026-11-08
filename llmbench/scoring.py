"""Result rows: the CSV schema, filename-derived model metadata, and the derived metrics / ranking scores."""
from __future__ import annotations

import math
import re
from datetime import datetime

from .common import avg, median
from .config import EVAL_SPECS, LONGCTX_SLOTS, PREFILL_SLOTS

# --------------------------------------------------------------------------- #
# Model metadata (parsed from the filename)
# --------------------------------------------------------------------------- #

_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[Bb](?![A-Za-z])")
_QUANT_RE = re.compile(
    r"(?<![A-Za-z])(IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32)(?![A-Za-z0-9])", re.I)
_ACTIVE_RE = re.compile(r"(?<![A-Za-z0-9])A(\d+(?:\.\d+)?)B(?![A-Za-z])")


def parse_params(name):
    """Total parameters in billions; 'A3B' (active params) is skipped by the lookbehind."""
    m = _PARAMS_RE.search(name)
    return float(m.group(1)) if m else None


def parse_active(name):
    """Active parameters (billions) of a MoE, from the 'A3B' tag; None for dense models."""
    m = _ACTIVE_RE.search(name)
    return float(m.group(1)) if m else None


def parse_quant(name):
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else "unknown"


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #

EVAL_FIELDS = [f"eval_{k}_{s}" for k in EVAL_SPECS
               for s in ("pct", "n", "err", "na", "fallback", "sec", "trunc", "degr", "rows", "sec1")] \
    + ["eval_avg_pct", "eval_avg_sec", "eval_retry_n", "eval_retry_pct", "eval_quality", "agent_quality",
       # v2.1: completeness (B1), skipped/screened models, funnel stage, concurrency actually used, s/q basis
       "eval_incomplete", "eval_skipped", "eval_screened", "eval_stage", "eval_stage1_avg_pct", "eval_inflight",
       "eval_sec_basis"]
# v2.1: where the time went (seconds) and what was generated (eval tokens, all attempts)
TIMING_FIELDS = ["time_total_s", "time_speed_s", "time_load_s", "time_eval_s", "time_longctx_s",
                 "eval_tok_think", "eval_tok_answer", "eval_cached_n"]
# v2.2: what the server actually loaded (launch flags + GGUF header) and the deployed-context tests
LAUNCH_FIELDS = ["kv_type", "flash_attn", "ngl", "n_cpu_moe", "ubatch", "batch", "parallel", "mmproj", "jinja",
                 "override_tensor", "kv_unified"]
GGUF_FIELDS = ["gguf_gb", "mmproj_gb", "bpw", "gguf_arch", "attn_type", "n_layers", "full_attn_layers", "swa_layers",
               "n_ctx_train", "expert_count", "expert_used", "kv_kib_token", "kv_swa_gb", "kv_approx", "kv_gb_ctx",
               "max_ctx_vram", "tok_per_word"]
CTX_FIELDS = ["ctx_ok"] + [f"prefill_{v}_tps" for v in PREFILL_SLOTS.values()] \
    + ["agent_prompt_tokens", "agent_ttft_s", "agent_ttft_warm_s"] \
    + [f"longctx_{v}_pct" for v in LONGCTX_SLOTS.values()] + ["longctx_avg_pct", "longctx_kv_ok", "longctx_note"]
BUDGET_FIELDS = ["budget_profile", "budget_pass", "budget_fails", "score_note"]
FIELDS = ["timestamp", "model", "status", "error", "params_b", "active_b", "arch", "offload", "quant", "ctx_size", "load_s",
          "tech_tps", "logic_tps", "avg_tps", "prefill_tps", "prefill_tokens", "cache_hit_pct", "cache_speedup",
          "ttft_ms", "gen_ms", "latency_s", "gpu_w", "gpu_peak_w", "idle_gpu_w", "gpu_util", "gpu_temp",
          "vram_gb", "vram_delta_gb", "vram_total_gb", "vram_pct", "cpu_w", "cpu_util",
          "ram_total_gb", "ram_used_gb", "ram_pct", "ram_idle_gb", "ram_delta_gb", "ram_proc_gb", "ram_spill_gb", "ram_overhead_gb",
          "energy_wh", "j_per_token", "cost_1m_peak", "cost_1m_off", "cost_1kq_peak", "cost_1kq_off",
          "tps_per_watt", "goldilocks", "score_chat", "score_agent", "vram_fit", "score_sweet", "eligible"] + EVAL_FIELDS \
    + LAUNCH_FIELDS + GGUF_FIELDS + CTX_FIELDS + BUDGET_FIELDS + TIMING_FIELDS


def blank(name):
    r = dict.fromkeys(FIELDS)
    r.update(model=name, status="ok", error="", eligible=False,
             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return r


# --------------------------------------------------------------------------- #
# finalize(): derived metrics and ranking scores (was one 65-line function)
# --------------------------------------------------------------------------- #

def _fill_identity(r):
    r["params_b"] = r["params_b"] or parse_params(r["model"])
    r["quant"] = r["quant"] or parse_quant(r["model"])
    r["active_b"] = r["active_b"] or parse_active(r["model"])
    r["arch"] = r["arch"] or ("MoE" if r["active_b"] else "Dense")


COMPLETE_SHARE = 0.9        # a dataset counts as complete when >= 90% of its planned questions got a real answer


def run_datasets(results):
    """{dataset: planned rows} for the datasets this run evaluated. Rows come from eval_<k>_rows (set by the runner);
    rows written by older versions / the demo have no such column, so the largest n seen is used instead."""
    ok = [r for r in results if r["status"] == "ok"]
    out = {}
    for k in EVAL_SPECS:
        rows = [r.get(f"eval_{k}_rows") for r in ok if r.get(f"eval_{k}_rows")]
        ns = [r.get(f"eval_{k}_n") for r in ok if r.get(f"eval_{k}_n")]
        if rows or ns:
            out[k] = max(rows) if rows else max(ns)
    return out


def eval_gaps(r, planned):
    """Datasets of the run this model has no trustworthy score for: [(dataset, reason)]. A model is only ranked on
    the FULL dataset set of the run - averaging over whatever happened to work would let a model whose hardest
    dataset failed (e.g. every GPQA request hit a template error) outrank a stronger model (v2.0.0 bug B1)."""
    gaps = []
    for k, run_rows in planned.items():
        rows = r.get(f"eval_{k}_rows") or (run_rows if r.get(f"eval_{k}_n") else None)
        n, err = r.get(f"eval_{k}_n") or 0, r.get(f"eval_{k}_err") or 0
        if not rows:
            gaps.append((k, "not run"))
        elif r.get(f"eval_{k}_pct") is None or n - err < COMPLETE_SHARE * rows:
            gaps.append((k, f"{max(0, n - err)}/{rows} answered"))
    return gaps


def _fill_eval_averages(r, cfg, planned=None):
    pcts = [r.get(f"eval_{k}_pct") for k in EVAL_SPECS if r.get(f"eval_{k}_pct") is not None]
    gaps = eval_gaps(r, planned) if planned else []
    any_eval = any(r.get(f"eval_{k}_n") for k in EVAL_SPECS)
    r["eval_incomplete"] = (", ".join(f"{EVAL_SPECS[k]['short']} {why}" for k, why in gaps)
                            if any_eval and gaps else None)
    r["eval_avg_pct"] = avg(pcts) if pcts and not r["eval_incomplete"] else None
    secs = [r.get(f"eval_{k}_sec") for k in EVAL_SPECS if r.get(f"eval_{k}_sec") is not None]
    r["eval_avg_sec"] = avg(secs) if secs else None          # mean seconds per question over the datasets run
    for k in EVAL_SPECS:                                      # single-stream s/q estimate (see _apply_sq_basis)
        r[f"eval_{k}_sec1"] = single_stream_sec(r, k) if r.get(f"eval_{k}_sec") is not None else None
    # Retries (No-Thinking fallbacks after a cut-off / loop / thinking cap / timeout). Many retries usually mean a chat
    # template or thinking problem, not a weak model; they cost time and are penalised: eval_quality = average minus
    # AGENT_FALLBACK_PENALTY x share of questions retried.
    n = sum(r.get(f"eval_{k}_n") or 0 for k in EVAL_SPECS if r.get(f"eval_{k}_pct") is not None)
    if n:
        retried = sum(r.get(f"eval_{k}_fallback") or 0 for k in EVAL_SPECS if r.get(f"eval_{k}_pct") is not None)
        r["eval_retry_n"], r["eval_retry_pct"] = retried, 100 * retried / n
    if r["eval_avg_pct"] is not None:
        share = (r.get("eval_retry_pct") or 0) / 100
        r["eval_quality"] = max(0.0, r["eval_avg_pct"] - cfg.agent_fallback_penalty * share)
    lc = [r.get(f"longctx_{v}_pct") for v in LONGCTX_SLOTS.values() if r.get(f"longctx_{v}_pct") is not None]
    r["longctx_avg_pct"] = avg(lc) if lc else None


def single_stream_sec(r, k):
    """Seconds one question of dataset k would take with ONE request in flight: mean tokens generated (all attempts)
    / the speed test's single-stream tok/s + mean prompt tokens / its prefill tok/s + mean grading time.
    None without a speed test (evals-only runs) or without token counts."""
    tok, tps = r.get(f"eval_{k}_tok"), r.get("avg_tps")
    if not tok or not tps:
        return None
    ptok, pre = r.get(f"eval_{k}_ptok"), r.get("prefill_tps")
    return round(tok / tps + (ptok / pre if ptok and pre else 0.0) + (r.get(f"eval_{k}_grade_s") or 0.0), 1)


def _apply_sq_basis(results, cfg):
    """Which s/q the Agent score uses. Measured latency includes queueing behind the other requests when several are
    in flight, and the in-flight cap differs per model (it depends on the KV pool), so with any concurrency the
    single-stream estimate is used for EVERY model (EVAL_SQ_BASIS=auto). With concurrency 1 everywhere the measured
    latency is used, exactly as in v2.0.0, so such runs rank identically to older ones."""
    ev = [r for r in results if r["status"] == "ok" and r.get("eval_avg_sec") is not None]
    if not ev:
        return None
    basis = cfg.eval_sq_basis
    if basis == "auto":
        basis = "single" if any((r.get("eval_inflight") or 1) > 1 for r in ev) else "measured"
    if basis == "single":
        complete = all(r.get(f"eval_{k}_sec1") is not None for r in ev for k in EVAL_SPECS
                       if r.get(f"eval_{k}_sec") is not None)
        if not complete:
            basis = "measured"            # no speed test (evals-only): nothing to estimate from
    for r in ev:
        r["eval_sec_basis"] = basis
        if basis == "single":
            r["eval_avg_sec"] = avg([r[f"eval_{k}_sec1"] for k in EVAL_SPECS if r.get(f"eval_{k}_sec1") is not None])
    return basis


_SPEED_PHASES = ("settle", "load", "speed", "prefill_ctx", "agent_ttft", "cooldown")


def _fill_timing(r):
    """CSV-friendly totals from r["timing"] (seconds per phase) and the eval token counts."""
    t = r.get("timing") or {}
    if t:
        r["time_speed_s"] = round(sum(t.get(p, 0.0) for p in _SPEED_PHASES), 1)
        r["time_load_s"] = round(t.get("load", 0.0) + t.get("eval_load", 0.0), 1)
        r["time_eval_s"] = round(sum(v for p, v in t.items() if p.startswith("eval:")), 1)
        r["time_longctx_s"] = round(t.get("longctx", 0.0), 1)
        r["time_total_s"] = round(sum(t.values()), 1)
    think = [r.get(f"eval_{k}_think") for k in EVAL_SPECS if r.get(f"eval_{k}_think") is not None]
    if think:
        r["eval_tok_think"] = sum(think)
        r["eval_tok_answer"] = sum(r.get(f"eval_{k}_answer") or 0 for k in EVAL_SPECS)


def _fill_memory(r, cfg, overhead):
    total = r["vram_total_gb"] or cfg.vram_limit_gb
    r["vram_total_gb"] = total
    if r["vram_gb"] is not None and total:
        r["vram_pct"] = r["vram_gb"] / total * 100
    if r["ram_used_gb"] is not None and r["ram_total_gb"]:
        r["ram_pct"] = r["ram_used_gb"] / r["ram_total_gb"] * 100
    r["ram_overhead_gb"] = overhead
    if r["offload"] == "none":                       # launched fully on the GPU: nothing is meant to spill
        r["ram_spill_gb"] = 0.0
        return
    src = r["ram_proc_gb"] if r["ram_proc_gb"] is not None else r["ram_delta_gb"]
    if src is None:
        return
    if r["offload"]:                                 # CPU-offload flags: server RAM above the typical overhead
        r["ram_spill_gb"] = max(0.0, src - (overhead or 0.0))
    else:                                            # launch flags unknown (e.g. Strata): size threshold
        r["ram_spill_gb"] = src if src >= cfg.spill_threshold_gb else 0.0


def _fill_efficiency_and_cost(r, cfg):
    watts = (r["gpu_w"] or 0) + (r["cpu_w"] or 0)
    if watts and r["avg_tps"]:
        r["tps_per_watt"] = r["avg_tps"] / watts
    if cfg.rate_peak is None:
        return
    off = cfg.rate_offpeak if cfg.rate_offpeak is not None else cfg.rate_peak
    if r["j_per_token"] is not None:        # J/token -> kWh per 1M tokens = J/3.6
        r["cost_1m_peak"], r["cost_1m_off"] = (r["j_per_token"] / 3.6 * x for x in (cfg.rate_peak, off))
    if r["energy_wh"] is not None:          # energy_wh covers 2 queries -> 1000 queries = 0.5 kWh per Wh
        r["cost_1kq_peak"], r["cost_1kq_off"] = (r["energy_wh"] * 0.5 * x for x in (cfg.rate_peak, off))


def _fill_scores(r, cfg):
    if r["params_b"] and r["avg_tps"]:
        p = math.log(r["params_b"]) if cfg.goldilocks_mode == "log" else r["params_b"]
        r["goldilocks"] = p * r["avg_tps"]
        fits = r["vram_pct"] is None or r["vram_pct"] <= cfg.goldilocks_max_vram_pct
        r["eligible"] = fits and not (cfg.goldilocks_vram_only and (r["ram_spill_gb"] or 0) > 0)
    # Category scores for the dashboard podium (need speed + VRAM data; the podium only ranks eligible models).
    if not (r["avg_tps"] and r["vram_pct"] is not None):
        return
    vram_room = max(0.0, 100 - r["vram_pct"])           # headroom left on the GPU, 0-100
    r["score_chat"] = r["avg_tps"] * vram_room


def agent_tolerance(best, cfg):
    """Eval-score difference (percentage points) that is just noise: AGENT_ACC_TOL, or 1 standard error of the best
    model's score over all the questions it answered (a 1-question gap is ~1.4 points with the 'basic' profile)."""
    if cfg.agent_acc_tol is not None:
        return cfg.agent_acc_tol
    n = sum(best.get(f"eval_{k}_n") or 0 for k in EVAL_SPECS if best.get(f"eval_{k}_pct") is not None)
    if not n:
        return 4.0
    p = min(0.95, max(0.05, best["_q"] / 100))
    return 100 * math.sqrt(p * (1 - p) / n)


# Typical bits per weight by quant name, for the no-eval proxy when the GGUF header could not be read.
_QUANT_BPW = {"IQ1_S": 1.6, "IQ1_M": 1.8, "IQ2_XXS": 2.1, "IQ2_XS": 2.3, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K": 2.9,
              "IQ3_XXS": 3.1, "IQ3_XS": 3.3, "IQ3_S": 3.4, "IQ3_M": 3.7, "Q3_K_S": 3.5, "Q3_K_M": 3.9, "Q3_K_L": 4.2,
              "IQ4_XS": 4.3, "IQ4_NL": 4.5, "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.6, "Q4_K_M": 4.9, "Q4_K_XL": 5.0,
              "Q5_0": 5.5, "Q5_K_S": 5.5, "Q5_K_M": 5.7, "Q6_K": 6.6, "Q8_0": 8.5, "BF16": 16, "F16": 16, "F32": 32}


def proxy_quality(r):
    """No evals in the run: a 0-1 stand-in for quality from model size and quantisation. Size = dense-equivalent
    parameters (a MoE counts as sqrt(total x active), a common rule of thumb), quantisation = a mild penalty below
    ~4.5 bits per weight (IQ2_M ~0.7, IQ3_XXS ~0.8, Q4 and up 1.0). A rough proxy: run evals to rank for real."""
    p = r.get("params_b")
    if not p:
        return None
    eff = math.sqrt(p * r["active_b"]) if r.get("active_b") else p
    bpw = r.get("bpw") or _QUANT_BPW.get((r.get("quant") or "").upper())
    qf = 1.0 if bpw is None else min(1.0, max(0.4, 1 - 0.12 * max(0.0, 4.5 - bpw) ** 1.5))
    return math.log1p(eff) * qf


def best_prefill(r):
    """The longest-prompt prefill speed measured (what an agent's 10-60k prompts see), else the short test."""
    for v in reversed(list(PREFILL_SLOTS.values())):
        if r.get(f"prefill_{v}_tps"):
            return r[f"prefill_{v}_tps"]
    return r.get("prefill_tps")


def speed_factor(r, cfg):
    """0-1 speed figure for the Agent score. Parts (weights): prompt processing 0.4 (agent-prompt TTFT if measured,
    else the longest prefill test), seconds per eval question 0.35, generation tokens/s 0.25. Missing parts are left
    out and the rest re-weighted. Prefill weighs most because an agent re-reads a 10-60k token context every turn."""
    parts = []
    if r.get("agent_ttft_s") is not None:
        parts.append((0.4, 1 / (1 + r["agent_ttft_s"] / cfg.agent_ttft_ref)))
    elif best_prefill(r):
        parts.append((0.4, min(1.0, math.log(max(1.0, best_prefill(r))) / math.log(cfg.agent_prefill_ref))))
    if r.get("eval_avg_sec") is not None:
        parts.append((0.35, 1 / (1 + r["eval_avg_sec"] / cfg.agent_sec_ref)))
    if r.get("avg_tps"):
        parts.append((0.25, min(1.0, math.log(max(1.0, r["avg_tps"])) / math.log(cfg.agent_tps_ref))))
    return sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else None


def _fill_agent_scores(results, cfg):
    """Agent score (0-100), quality first and speed second. VRAM is deliberately not part of it (see score_sweet).

    1. Quality. With evals: the eval average, blended with long-context retrieval (AGENT_LONGCTX_WEIGHT) when every
       model has it, compared with the best of the run: anything within the noise margin (agent_tolerance) of the best
       is a tie at 100; below that the credit falls steeply (quadratic, 0 at AGENT_ACC_SPAN points under the tie zone).
       Then the retry penalty: x (1 - AGENT_FALLBACK_PENALTY% x share of questions retried), applied after the tie zone
       so it separates tied models too. Without evals: proxy_quality() relative to the best of the run.
    2. Speed factor (speed_factor): can only remove up to AGENT_SPEED_WEIGHT of the credit, so speed separates models of
       equal quality but never lifts a clearly weaker one.
    score_agent = credit x (1 - W + W x speed); agent_quality = credit.

    When the run HAS evals, a model whose evals produced no score (every request failed, e.g. a template error) gets
    no Agent score at all. (v2.1 gave it the no-eval proxy, which is on a different scale and ranked it first.)"""
    ok = [r for r in results if r["status"] == "ok"]
    has_evals = any(r.get("eval_avg_pct") is not None for r in ok)
    w = cfg.agent_speed_weight
    if has_evals:
        # Funnel: models screened out at an earlier stage answered fewer questions and are not ranked against the
        # finalists (their stage score stays visible, with its larger margin of error).
        pool = [r for r in ok if r.get("eval_quality") is not None and not r.get("eval_screened")]
        lc_all = cfg.agent_longctx_weight > 0 and pool and all(r.get("longctx_avg_pct") is not None for r in pool)
        for r in pool:
            r["_q"] = r["eval_avg_pct"]
            if lc_all:
                r["_q"] = (1 - cfg.agent_longctx_weight) * r["eval_avg_pct"] + cfg.agent_longctx_weight * r["longctx_avg_pct"]
        for r in ok:
            if r.get("eval_quality") is None:
                errs = sum(r.get(f"eval_{k}_err") or 0 for k in EVAL_SPECS)
                if r.get("eval_skipped"):
                    r["score_note"] = f"evals skipped: {r['eval_skipped']}"
                elif r.get("eval_screened"):
                    r["score_note"] = f"not ranked: {r['eval_screened']}"
                elif r.get("eval_incomplete"):
                    r["score_note"] = (f"not ranked: incomplete evals ({r['eval_incomplete']}"
                                       + (f"; {errs} request errors)" if errs else ")"))
                else:
                    r["score_note"] = f"no eval score ({errs} request errors)" if errs else "no eval score"
        for r in ok:
            if r.get("eval_screened") and r.get("eval_quality") is not None:
                r["score_note"] = f"not ranked: {r['eval_screened']}"
        if not pool:
            return
        ranked = [r for r in pool if r["eligible"]] or pool       # the bar is the best model that can actually be used
        best = max(ranked, key=lambda r: r["_q"])
        tol = agent_tolerance(best, cfg)
        for r in pool:
            gap = max(0.0, best["_q"] - tol - r["_q"])
            # The retry penalty is applied AFTER the tie zone, so it still separates models whose scores tie.
            keep = 1 - cfg.agent_fallback_penalty / 100 * (r.get("eval_retry_pct") or 0) / 100
            r["agent_quality"] = 100 * max(0.0, 1 - gap / cfg.agent_acc_span) ** 2 * max(0.0, keep)
    else:
        for r in ok:
            if r.get("eval_skipped"):
                r["score_note"] = f"evals skipped: {r['eval_skipped']}"
        pool = [r for r in ok if proxy_quality(r) is not None and not r.get("eval_skipped")]
        if not pool:
            return
        top = max(proxy_quality(r) for r in pool) or 1
        for r in pool:
            r["agent_quality"] = 100 * proxy_quality(r) / top
            r["score_note"] = "proxy (no evals): size and quant, not measured quality"
    for r in pool:
        sp = speed_factor(r, cfg)
        r["score_agent"] = r["agent_quality"] * ((1 - w + w * sp) if sp is not None else 1.0)
        r.pop("_q", None)


def vram_fit(pct, cfg):
    """How well the VRAM use fits the sweet spot, 0-1. 1.0 inside [VRAM_SWEET_LOW, VRAM_SWEET_HIGH] (default 80-90%).
    Under-use is only a small loss (VRAM_UNDERUSE_FLOOR, default 0.85 at 0% -> 1.0 at the band): once a model fits,
    quality and speed should decide, not how full the GPU is. Above the band it falls to 0.5 at VRAM_SWEET_MAX (94%),
    then to 0 at 100%."""
    if pct is None:
        return None
    lo = cfg.vram_sweet_low
    hi = max(lo, cfg.vram_sweet_high)
    mx = min(99.9, max(hi + 0.1, cfg.vram_sweet_max))
    floor = cfg.vram_underuse_floor
    if pct < lo:
        return floor + (1 - floor) * max(0.0, pct) / lo
    if pct <= hi:
        return 1.0
    if pct <= mx:
        return 1 - 0.5 * (pct - hi) / (mx - hi)
    return max(0.0, 0.5 * (100 - pct) / (100 - mx))


def _fill_sweet(results, cfg):
    """Sweet spot = Agent score x VRAM fit: the best capability that also uses the GPU sensibly (some headroom, nothing
    wasted). Without evals the Agent score is the parameter-count proxy, so the same caveat applies."""
    for r in results:
        if r["status"] != "ok" or r.get("vram_pct") is None:
            continue
        r["vram_fit"] = vram_fit(r["vram_pct"], cfg)
        if r.get("score_agent") is not None:
            r["score_sweet"] = r["score_agent"] * r["vram_fit"]


# --------------------------------------------------------------------------- #
# Per-machine budget (BUDGET_PROFILE): pass / fail per rule
# --------------------------------------------------------------------------- #

def _rule(ok, value, limit, text):
    return {"ok": ok, "value": value, "limit": limit, "text": text}


def _num_rule(v, limit, higher_ok, fmt, unit=""):
    if v is None:
        return _rule(None, None, limit, "not measured")
    ok = v >= limit if higher_ok else v <= limit
    return _rule(ok, v, limit, f"{fmt.format(v)}{unit} {'>=' if higher_ok else '<='} {fmt.format(limit)}{unit}")


def budget_checks(r, cfg):
    """{rule: {"ok": True/False/None, "value", "limit", "text"}} for the active budget profile. None = not measured
    (never counts as a fail)."""
    from .modelinfo import kv_at_least
    rules, out = (cfg.budget or {}).get("rules", {}), {}
    if "MIN_CTX" in rules:
        out["MIN_CTX"] = _num_rule(r.get("ctx_size"), int(rules["MIN_CTX"]), True, "{:,.0f}")
    if "VRAM_RESERVE_GB" in rules or "MAX_VRAM_PCT" in rules:
        model = r.get("vram_delta_gb") if r.get("vram_delta_gb") is not None else r.get("vram_gb")
        total, reserve = r.get("vram_total_gb"), rules.get("VRAM_RESERVE_GB", 0.0)
        if model is None or not total:
            out["VRAM"] = _rule(None, None, None, "not measured")
        else:
            cap = total * rules.get("MAX_VRAM_PCT", 100.0) / 100
            out["VRAM"] = _rule(model + reserve <= cap, model, cap - reserve,
                                f"model {model:.1f} GB + reserve {reserve:g} GB = {model + reserve:.1f} of {cap:.1f} GB")
    if "KV_MIN" in rules:
        ok = kv_at_least(r.get("kv_type"), rules["KV_MIN"])
        out["KV_MIN"] = _rule(ok, r.get("kv_type"), rules["KV_MIN"],
                              f"KV {r.get('kv_type') or 'unknown'} vs minimum {rules['KV_MIN']}")
    for rule, key, higher, fmt, unit in (("MAX_SPILL_GB", "ram_spill_gb", False, "{:.1f}", " GB"),
                                         ("MIN_TPS", "avg_tps", True, "{:.0f}", " tok/s"),
                                         ("MIN_PREFILL_TPS", None, True, "{:.0f}", " tok/s"),
                                         ("MAX_AGENT_TTFT_S", "agent_ttft_s", False, "{:.1f}", " s"),
                                         ("MIN_EVAL_PCT", "eval_avg_pct", True, "{:.0f}", "%"),
                                         ("MAX_RETRY_PCT", "eval_retry_pct", False, "{:.0f}", "%"),
                                         ("MIN_LONGCTX_PCT", "longctx_avg_pct", True, "{:.0f}", "%")):
        if rule in rules:
            v = best_prefill(r) if key is None else r.get(key)
            out[rule] = _num_rule(v, rules[rule], higher, fmt, unit)
    return out


def _fill_budget(results, cfg):
    for r in results:
        if r["status"] != "ok":
            continue
        if r.get("ctx_ok") is False:          # MIN_CTX_ACTION=flag: measured, but not at the deployed context
            r["eligible"] = False
        if r.get("eval_screened") or r.get("eval_incomplete"):     # not ranked: never on the podium
            r["eligible"] = False
        if not cfg.budget:
            continue
        checks = budget_checks(r, cfg)
        fails = [f"{k}: {c['text']}" for k, c in checks.items() if c["ok"] is False]
        r["budget"], r["budget_profile"] = checks, cfg.budget["name"]
        r["budget_pass"], r["budget_fails"] = not fails, "; ".join(fails)
        if fails and cfg.budget_podium_pass_only:
            r["eligible"] = False


def finalize(results, cfg):
    """Derive params/quant/VRAM%/efficiency, the ranking scores and the budget checks."""
    from .modelinfo import max_ctx_at
    # Typical server RAM of models launched fully on the GPU (CUDA context, pinned buffers, leftover mapped pages)
    ov = [r["ram_proc_gb"] for r in results if r["status"] == "ok" and r["offload"] == "none"
          and r["ram_proc_gb"] is not None]
    overhead = cfg.ram_overhead_gb if cfg.ram_overhead_gb is not None else median(ov)
    planned = run_datasets(results)
    for r in results:
        _fill_identity(r)
        if r["status"] != "ok":
            continue
        for k in ("score_agent", "agent_quality", "score_sweet", "score_note"):     # finalize() can run twice
            r[k] = None
        r["avg_tps"] = avg([x for x in (r["tech_tps"], r["logic_tps"]) if x is not None])
        _fill_eval_averages(r, cfg, planned)
        _fill_memory(r, cfg, overhead)
        if r.get("kv_kib_token") and r.get("ctx_size"):
            r["kv_gb_ctx"] = r["kv_kib_token"] * r["ctx_size"] / 1024 ** 2 + (r.get("kv_swa_gb") or 0)
        r["max_ctx_vram"] = max_ctx_at(r, cfg.max_ctx_vram_pct)
        _fill_efficiency_and_cost(r, cfg)
        _fill_scores(r, cfg)
        _fill_timing(r)
    _apply_sq_basis(results, cfg)            # before the Agent score, which uses eval_avg_sec
    _fill_budget(results, cfg)               # before the Agent score: the bar is the best ELIGIBLE model
    _fill_agent_scores(results, cfg)
    _fill_sweet(results, cfg)
    return results
