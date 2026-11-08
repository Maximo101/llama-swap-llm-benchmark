"""--estimate: predict how long a run with the current .env settings would take, before starting it.

Sources, best first, per model:
  1. earlier runs of this tool: every benchmark_*.json in OUTPUT_DIR (newest wins per model and field). v2.1 runs
     carry measured per-phase seconds; v2.0 runs still give tok/s, prefill tok/s by length, load time, and per
     dataset the seconds and tokens per question.
  2. a batch-probe result (results/batch_probe_*.json) for the gain of EVAL_CONCURRENCY > 1.
  3. otherwise a rough model: generation tok/s from the model's size and quantisation (parsed from its name) on a
     card with ESTIMATE_GPU_BW_GBS of memory bandwidth (default 900, about an RTX 5070 Ti), and typical tokens per
     question for each dataset (thinking models write 4-10x more than non-thinking ones).
Every line says which source it used. Nothing is sent to the models; only GET /v1/models is used, if reachable.
"""
from __future__ import annotations

import glob
import json
from copy import copy
import math
import os
from pathlib import Path

from .common import fmt_hr_m_s, median
from .config import EVAL_SPECS, LONGCTX_SLOTS, PREFILL_SLOTS, PROFILE_ROWS, funnel_stages, stage_rows
from .scoring import parse_active, parse_params, parse_quant, _QUANT_BPW

# typical tokens generated per question (all attempts): (thinking model, non-thinking model).
# Calibrated on a real run (RTX 5070 Ti, 19 models, profile moderate, 2026-11-07): effective tokens per question
# (seconds x tok/s) had medians of MMLU-Pro ~4,100, HumanEval ~2,600, MBPP ~3,000, tool calling ~130, with nearly
# every model thinking. GPQA, LiveCodeBench, GSM8K and MATH are scaled from those by their usual ratios (unmeasured).
TYPICAL_TOK = {"gpqa": (10000, 700), "mmlu_pro": (4400, 450), "humaneval": (2800, 350), "mbpp": (3200, 300),
               "livecodebench": (12000, 900), "gsm8k": (2000, 250), "math": (5500, 600), "toolcall": (130, 130)}
THINKING_SHARE = 0.9          # share of models assumed to think, when nothing is known
AVAILABLE = {"gpqa": 198, "mmlu_pro": 12032, "humaneval": 164, "mbpp": 500, "livecodebench": 1055, "gsm8k": 1319,
             "math": 500, "toolcall": 32}
SANDBOX_S = 1.5               # seconds per program run in Docker (hidden by pipelined grading in v2.1)
FUNNEL_SURVIVE = 0.5          # share of models assumed to pass each funnel gate (a crowded field keeps more)


def _history(cfg):
    """{model: merged row-like dict} from earlier JSON exports, newest first."""
    files = sorted(glob.glob(str(cfg.output_dir / "benchmark_*.json")), key=os.path.getmtime, reverse=True)
    out = {}
    for f in files[:30]:
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in doc.get("models") or []:
            if m.get("status") != "ok":
                continue
            h = out.setdefault(m["model"], {"_sources": []})
            h["_sources"].append(Path(f).name)
            flat = {}
            for group, vals in m.items():                     # every flat group (speed, launch, budget, timing...)
                if isinstance(vals, dict) and group not in ("evaluation", "other"):
                    flat.update({k: v for k, v in vals.items() if v is not None and not isinstance(v, dict)})
            for k, d in ((m.get("evaluation") or {}).get("datasets") or {}).items():
                for f2, v in d.items():
                    if v is not None:
                        flat[f"eval_{k}_{f2}"] = v
            for k, v in flat.items():
                h.setdefault(k, v)                   # newest file wins
    return out


def _probe(cfg):
    """{model: {level: steady-state gain}} from batch_probe_*.json (recomputed, so older probe files count too)."""
    from .batchprobe import steady_gain
    files = [f for f in glob.glob(str(cfg.output_dir / "batch_probe_*.json")) if not f.endswith("_details.jsonl")]
    files.sort(key=os.path.getmtime, reverse=True)
    gains = {}
    for f in files[:10]:
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in doc.get("models") or []:
            base = next((x for x in m.get("levels") or [] if x.get("level") == 1), None)
            for lvl in m.get("levels") or []:
                g = steady_gain(lvl, base) if base else None
                gains.setdefault(m["model"], {}).setdefault(lvl["level"], g)
    return gains


def _class_gain(gains, history, model, level):
    """Gain for a model the probe did not cover: the median gain of probed models in the same class (fully on the
    GPU vs partly on the CPU, from the launch flags), else of all probed models, else 1 (no speed-up assumed)."""
    if level <= 1:
        return 1.0, "x1"
    own = (gains.get(model) or {}).get(level)
    if own:
        return own, "probe"
    off = lambda m: (history.get(m) or {}).get("offload") not in (None, "none")
    same = [g[level] for m, g in gains.items() if g.get(level) and off(m) == off(model)]
    pool = same or [g[level] for g in gains.values() if g.get(level)]
    if not pool:
        return 1.0, "no probe"
    from statistics import median
    return round(median(pool), 2), "probe class" if same else "probe median"


def _skipped(cfg, h):
    """Why EVAL_SKIP_INELIGIBLE would skip this model's evals, judged from its last results (or None)."""
    if cfg.eval_skip_ineligible == "false" or not h:
        return None
    if h.get("ctx_ok") is False:
        return "below MIN_CTX"
    if h.get("budget_pass") is False and (cfg.eval_skip_ineligible == "true" or cfg.budget_podium_pass_only) \
            and cfg.budget:
        return f"budget: {h.get('budget_fails') or 'fails'}"
    return None


def _guess_tps(model, bw):
    """Rough single-stream tok/s: bandwidth / bytes read per token (active params x bits per weight), at ~60%."""
    p = parse_params(model)
    if not p:
        return None
    active = parse_active(model) or p
    bpw = _QUANT_BPW.get(parse_quant(model).upper(), 4.8)
    return 0.6 * bw / max(0.3, active * bpw / 8)


def _prefill_s(tokens, h, guess_tps):
    """Seconds to prefill `tokens`: measured prefill tok/s at the nearest length, else ~40x generation speed,
    falling with length."""
    best = None
    for n, slot in PREFILL_SLOTS.items():
        v = h.get(f"prefill_{slot}_tps")
        if v and (best is None or abs(math.log(n / tokens)) < abs(math.log(best[0] / tokens))):
            best = (n, v)
    if best:
        return tokens / best[1]
    base = h.get("prefill_tps") or (guess_tps or 50) * 40
    return tokens / (base / (1 + math.log2(max(1, tokens / 4096)) * 0.35))


def _eval_secs(cfg, h, key, tps, conc_gain):
    """(seconds per question, source)."""
    if h.get(f"eval_{key}_sec1"):
        return h[f"eval_{key}_sec1"] / conc_gain, "measured (single-stream)"
    if h.get(f"eval_{key}_sec") and (h.get(f"eval_{key}_concurrency") or 1) == 1:
        return h[f"eval_{key}_sec"] / conc_gain, "measured"
    if h.get(f"eval_{key}_tok") and tps:
        return h[f"eval_{key}_tok"] / tps / conc_gain, "tokens measured"
    think, plain = TYPICAL_TOK[key]
    tok = THINKING_SHARE * think + (1 - THINKING_SHARE) * plain
    return (tok / tps if tps else 30.0) / conc_gain, "typical"


def estimate_model(cfg, model, h, bw, gains, history_all=None):
    """-> {phase: seconds}, notes."""
    history_all = history_all or {}
    tps = h.get("avg_tps") or (median([h.get("tech_tps"), h.get("logic_tps")]) if h.get("tech_tps") else None)
    src = "history" if tps else "guess"
    tps = tps or _guess_tps(model, bw) or 50.0
    out = {}
    if cfg.run_speed_tests:
        if h.get("time_speed_s"):
            out["speed"] = h["time_speed_s"]
        else:
            load = h.get("load_s") or 12.0
            sp = cfg.settle + cfg.cooldown + load
            sp += cfg.runs * (cfg.max_tokens_tech + cfg.max_tokens_logic) / tps
            if cfg.prefill_words:
                sp += cfg.runs * (3 if cfg.cache_test else 1) * _prefill_s(cfg.prefill_words * 1.25, h, tps)
            for n in cfg.prefill_ctx_tests:
                if n not in cfg.defer_prefill:
                    sp += _prefill_s(n, h, tps)
            if cfg.agent_prompt_tokens:
                sp += _prefill_s(cfg.agent_prompt_tokens, h, tps) + 1
            out["speed"] = sp
    skip = _skipped(cfg, h) if cfg.run_eval_tests else None
    if skip:
        out["_notes"] = f"evals skipped ({skip})"
        out["_skip"] = True
        return out
    if cfg.run_eval_tests:
        if cfg.eval_longctx:
            lc = 0.0
            ctx = h.get("ctx_size")
            for n in cfg.longctx_lengths:
                if ctx and n > ctx:
                    continue
                lc += _prefill_s(n, h, tps) + cfg.longctx_trials * (3 + 60 / tps)     # one cold prefill per length
            out["longctx"] = lc
        conc = cfg.eval_concurrency
        gain, gsrc = _class_gain(gains, history_all, model, conc)
        rows = stage_rows(cfg)
        ev = 0.0
        notes = set()
        for key in cfg.eval_datasets:
            k = AVAILABLE.get(key, 500) if rows is None else min(rows, AVAILABLE.get(key, rows))
            spq, how = _eval_secs(cfg, h, key, tps, gain)
            notes.add(how)
            ev += k * spq
            if EVAL_SPECS[key]["kind"] == "code" and conc <= 1:
                ev += 0.0               # pipelined grading hides the sandbox behind generation
        out["evals_full"] = ev + 15.0   # + the eval-phase model load
        out["_notes"] = f"tok/s {tps:.0f} ({src}); evals: {', '.join(sorted(notes)) or '-'}" + \
                        (f"; x{conc} gain {gain:.1f} ({gsrc})" if conc > 1 else "")
    return out


def _models(cfg, history):
    from .client import make_session, select_models
    try:
        s = make_session(cfg, "llama-swap")
        r = s.get(cfg.models_url, timeout=5)
        r.raise_for_status()
        ids = select_models(cfg, [m["id"] for m in r.json().get("data", [])])
        if ids:
            return ids, "server"
    except Exception:
        pass
    return sorted(history), "history (server not reachable)"


def estimate(cfg, brief=False):
    """Print the estimate. brief=True: one line, for the start of a normal run. Returns total seconds or None."""
    cfg = copy(cfg)                              # never change the settings of the run that asked
    history = _history(cfg)
    gains = _probe(cfg)
    bw = float(os.getenv("ESTIMATE_GPU_BW_GBS") or 900)
    if cfg.run_eval_tests and cfg.eval_longctx:
        cfg.defer_prefill = tuple(n for n in cfg.prefill_ctx_tests if n in cfg.longctx_lengths)
    models, where = _models(cfg, history)
    if not models:
        if not brief:
            print("--estimate: no models (server not reachable and no earlier JSON results in "
                  f"{cfg.output_dir}). Nothing to estimate.")
        return None
    per = {m: estimate_model(cfg, m, history.get(m, {}), bw, gains, history) for m in models}
    funnel = cfg.eval_profile == "funnel" and cfg.run_eval_tests
    total_full = sum(sum(v for k, v in e.items() if not k.startswith("_")) for e in per.values())
    total = total_full
    if funnel:
        stages = funnel_stages(cfg)
        final = stages[-1][1]
        fr = [(r if r is not None else 500) / (final if final is not None else 500) for _, r in stages]
        n = max(1, sum(1 for m in models if not per[m].get("_skip")))
        work = 0.0
        alive = float(n)
        prev = 0.0
        for i, f in enumerate(fr):
            work += alive * (f - prev)
            prev = f
            alive = max(min(cfg.eval_leaders, n), alive * FUNNEL_SURVIVE)
        share = work / n
        ev_full = sum(e.get("evals_full", 0) for e in per.values())
        total = total_full - ev_full + ev_full * share
    if brief:
        print(f"Estimated run time: ~{fmt_hr_m_s(total)} for {len(models)} model(s)"
              + (" (funnel: assumes half the models pass each gate)" if funnel else "")
              + (f"; {sum(1 for m in models if m in history)} with earlier results" if history else "; no earlier "
                 "results, rough guess"))
        return total
    print(f"Run-time estimate for {len(models)} model(s) from {where}; profile {cfg.eval_profile}"
          + (f" (final {cfg.eval_funnel_final})" if funnel else "")
          + f", datasets {', '.join(EVAL_SPECS[k]['short'] for k in cfg.eval_datasets) or 'none'}"
          + f", concurrency {cfg.eval_concurrency}\n")
    w = max(len(m) for m in models)
    print(f"  {'model':<{w}}  {'speed':>8}  {'long-ctx':>8}  {'evals':>8}  source")
    for m in models:
        e = per[m]
        f = lambda k: fmt_hr_m_s(e[k]).replace(" hr ", "h").replace(" m ", "m").replace(" s", "s") if e.get(k) else "-"
        print(f"  {m:<{w}}  {f('speed'):>8}  {f('longctx'):>8}  {f('evals_full'):>8}  {e.get('_notes', '')}")
    print(f"\n  Total: ~{fmt_hr_m_s(total)}" + (f"  (without the funnel: ~{fmt_hr_m_s(total_full)}; the funnel line "
                                               f"assumes {int(FUNNEL_SURVIVE * 100)}% of models pass each gate, never "
                                               f"fewer than EVAL_LEADERS={cfg.eval_leaders})" if funnel else ""))
    if cfg.eval_concurrency > 1 and not gains:
        print("  EVAL_CONCURRENCY > 1 but no batch-probe result: no speed-up assumed. Run --batch-probe to measure it.")
    skipped = [m for m in models if per[m].get("_skip")]
    if skipped:
        print(f"  {len(skipped)} model(s) skipped (EVAL_SKIP_INELIGIBLE={cfg.eval_skip_ineligible}, from their last "
              "results): " + ", ".join(skipped))
    missing = [m for m in models if m not in history]
    if missing:
        print(f"  {len(missing)} model(s) without earlier results use a rough guess (GPU bandwidth {bw:g} GB/s, "
              f"{int(THINKING_SHARE * 100)}% thinking models); a run records the real figures.")
    if cfg.eval_result_cache and cfg.eval_result_cache_dir and Path(cfg.eval_result_cache_dir).exists():
        print("  The result cache may hold answers from earlier runs: questions already answered with identical "
              "settings are not asked again, so the real run can be much shorter.")
    return total
