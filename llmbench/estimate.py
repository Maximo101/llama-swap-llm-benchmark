"""--estimate: predict how long a run with the current .env settings would take, before starting it.

Sources, best first, per model:
  1. earlier runs of this tool: every benchmark_*.json in OUTPUT_DIR (newest wins per model and field). v2.1 runs
     carry measured per-phase seconds; v2.0 runs still give tok/s, prefill tok/s by length, load time, and per
     dataset the seconds and tokens per question.
  2. a batch-probe result (results/batch_probe_*.json) for the gain of EVAL_CONCURRENCY > 1.
  3. otherwise a rough model: generation tok/s from the model's size and quantisation (parsed from its name) on a
     card with ESTIMATE_GPU_BW_GBS of memory bandwidth (default auto: the detected GPU in gpubw.GPU_TABLE, else 670
     GB/s), and typical tokens per
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

from . import gpubw
from .common import fmt_hr_m_s, median
from .config import EVAL_SPECS, LONGCTX_SLOTS, PREFILL_SLOTS, PROFILE_ROWS, depth_for, funnel_stages, rows_for, stage_rows
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
            ctx = h.get("ctx_size")
            depth = depth_for(cfg, ctx)
            if depth:                                                                       # prefill at depth (v2.3)
                if depth not in cfg.prefill_ctx_tests or depth in cfg.defer_prefill:
                    sp += _prefill_s(depth, h, tps)                                         # its own cold prompt
                sp += _prefill_s(4096, h, tps) * 1.3 + 1                                     # the new block, at depth
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
        out["_ev"] = []
        for key in cfg.eval_datasets:
            kr = rows_for(key, rows)                # tool calling: every case at every stage (v2.3)
            k = AVAILABLE.get(key, 500) if kr is None else min(kr, AVAILABLE.get(key, kr))
            spq, how = _eval_secs(cfg, h, key, tps, gain)
            notes.add(how)
            ev += k * spq
            out["_ev"].append((key, k, spq))
            if EVAL_SPECS[key]["kind"] == "code" and conc <= 1:
                ev += 0.0               # pipelined grading hides the sandbox behind generation
        out["evals_full"] = ev + 15.0   # + the eval-phase model load
        out["_notes"] = f"tok/s {tps:.0f} ({src}); evals: {', '.join(sorted(notes)) or '-'}" + \
                        (f"; x{conc} gain {gain:.1f} ({gsrc})" if conc > 1 else "")
    return out


def cache_mode(cfg):
    if not cfg.eval_result_cache:
        return "false"
    return "refresh" if cfg.eval_cache_refresh else "true"


def cached_answers(cfg):
    """{model: {dataset: number of questions with a cached answer}} from the result cache. Per model only its newest
    cache file counts: a new file means the model's fingerprint changed (file, launch flags, llama.cpp build, ...), and
    then the older answers are not reused. Whether the CURRENT fingerprint matches cannot be known without loading
    the model, so this is what a rerun with unchanged models and settings would reuse."""
    root = cfg.eval_result_cache_dir
    if cache_mode(cfg) != "true" or not root or not Path(root).is_dir():
        return {}
    newest = {}
    for f in Path(root).glob("*.jsonl"):
        try:
            with open(f, encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
        except (OSError, ValueError):
            continue
        model = (first.get("rec") or {}).get("model")
        if model and (model not in newest or f.stat().st_mtime > newest[model].stat().st_mtime):
            newest[model] = f
    out = {}
    for model, f in newest.items():
        ids = {}
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)["rec"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    if not rec.get("api_error"):
                        ids.setdefault(rec.get("dataset"), set()).add(rec.get("id"))
        except OSError:
            continue
        out[model] = {k: len(v) for k, v in ids.items()}
    return out


def _stage_shape(cfg, n_models):
    """[(cumulative share of the final rows, expected share of models still in)] per stage."""
    if cfg.eval_profile != "funnel":
        return [(1.0, 1.0)]
    stages = funnel_stages(cfg)
    final = stages[-1][1] if stages[-1][1] is not None else 500
    alive, out = float(n_models), []
    for _, r in stages:
        out.append(((r if r is not None else 500) / final, alive / max(1, n_models)))
        alive = max(min(cfg.eval_leaders, n_models), alive * FUNNEL_SURVIVE)
    return out


def _uncached_work(shape, cached_share):
    """Expected share of a dataset's final rows this model will be asked, with the first `cached_share` of the rows
    already answered (nested sampling: every profile is a prefix of one question order, so cached answers from
    earlier runs with the same EVAL_SEED are the first questions)."""
    work, prev = 0.0, 0.0
    for f, alive in shape:
        work += alive * max(0.0, f - max(prev, cached_share))
        prev = f
    return work


def _strata_models(cfg):
    """The one model a Strata run benchmarks, named as the run names it (STRATA_MODEL, else the first id from
    Strata's /v1/models, else "strata"), after INCLUDE/EXCLUDE/MODEL_FILTER. [] if Strata is not configured."""
    from .client import make_session, select_models
    if not getattr(cfg, "strata_url", None):
        return []
    name = cfg.strata_model or getattr(cfg, "strata_model_id", None)
    if not name:
        try:
            s = make_session(cfg, "strata")
            r = s.get(cfg.strata_url + "/v1/models", timeout=5)
            data = r.json().get("data") or [] if r.ok else []
            name = data[0].get("id") if data and isinstance(data[0], dict) else None
        except Exception:
            name = None
    return select_models(cfg, [name or "strata"])


def _models(cfg, history):
    from .client import make_session, select_models
    if getattr(cfg, "backend", None) == "strata":                # the run already fell back to Strata: one model
        return _strata_models(cfg), "Strata"
    try:
        s = make_session(cfg, "llama-swap")
        r = s.get(cfg.models_url, timeout=5)
        r.raise_for_status()
        ids = select_models(cfg, [m["id"] for m in r.json().get("data", [])])
        if ids:
            return ids, "server"
    except Exception:
        pass
    ids = _strata_models(cfg)                                    # --estimate on its own, llama-swap stopped
    if ids:
        return ids, "Strata (llama-swap not reachable)"
    return sorted(history), "history (server not reachable)"


def estimate(cfg, brief=False):
    """Print the estimate. brief=True: one line, for the start of a normal run. Returns total seconds or None."""
    cfg = copy(cfg)                              # never change the settings of the run that asked
    history = _history(cfg)
    gains = _probe(cfg)
    bw, bw_src = gpubw.resolve(os.getenv("ESTIMATE_GPU_BW_GBS"))
    if cfg.run_eval_tests and cfg.eval_longctx:
        cfg.defer_prefill = tuple(n for n in cfg.prefill_ctx_tests if n in cfg.longctx_lengths and n != cfg.prefill_depth)
    models, where = _models(cfg, history)
    if not models:
        if not brief:
            print("--estimate: no models (server not reachable and no earlier JSON results in "
                  f"{cfg.output_dir}). Nothing to estimate.")
        return None
    per = {m: estimate_model(cfg, m, history.get(m, {}), bw, gains, history) for m in models}
    funnel = cfg.eval_profile == "funnel" and cfg.run_eval_tests
    total_full = sum(sum(v for k, v in e.items() if not k.startswith("_")) for e in per.values())
    mode = cache_mode(cfg) if cfg.run_eval_tests else "false"
    cache = cached_answers(cfg) if cfg.run_eval_tests else {}
    shape = _stage_shape(cfg, max(1, sum(1 for m in models if not per[m].get("_skip"))))
    sanity = cfg.eval_sanity_pct / 100 if mode == "true" else 0.0
    total = 0.0
    for m, e in per.items():
        ev, cached_n, planned_n, recheck = 0.0, 0, 0, 0.0
        for key, k, spq in e.get("_ev") or []:
            c = min(k, (cache.get(m) or {}).get(key, 0))
            sh = [(1.0, 1.0)] if EVAL_SPECS[key].get("full") else shape     # asked in full at the first stage
            share = _uncached_work(sh, c / k if k else 0.0)
            ev += k * spq * share
            exp_rows = k * _uncached_work(sh, 0.0)               # expected rows asked with an empty cache
            cached_n += round(exp_rows - k * share)
            planned_n += round(exp_rows)
            if sanity and c:
                reused = exp_rows - k * share
                # adaptive: each matching answer halves the rate, so a stable model is checked only a few times
                n_chk = sanity * reused if not cfg.eval_sanity_adaptive else min(sanity * reused, 2 + math.log2(1 + sanity * reused))
                recheck += n_chk
                ev += n_chk * spq
        if e.get("_ev"):
            e["evals"] = ev + 15.0 * len(shape)
            e["_cached"] = (cached_n, planned_n, recheck)
        total += sum(v for k, v in e.items() if not k.startswith("_") and k != "evals_full")
    if brief:
        reused = sum((e.get("_cached") or (0, 0, 0))[0] for e in per.values())
        print(f"Estimated run time: ~{fmt_hr_m_s(total)} for {len(models)} model(s)"
              + (" (funnel: assumes half the models pass each gate)" if funnel else "")
              + (f"; ~{reused} answers from the result cache" if reused else
                 "; full re-run (EVAL_RESULT_CACHE=refresh)" if mode == "refresh" else
                 "; full re-run (result cache off)" if mode == "false" and cfg.run_eval_tests else "")
              + (f"; {sum(1 for m in models if m in history)} with earlier results" if history else "; no earlier "
                 "results, rough guess"))
        return total
    print(f"Run-time estimate for {len(models)} model(s) from {where}; profile {cfg.eval_profile}"
          + (f" (final {cfg.eval_funnel_final})" if funnel else "")
          + f", datasets {', '.join(EVAL_SPECS[k]['short'] for k in cfg.eval_datasets) or 'none'}"
          + f", concurrency {cfg.eval_concurrency}\n")
    if cfg.run_eval_tests:
        print("  Result cache: " + {
            "true": "on: answers already in the cache are reused (counted below from each model's newest cache file)"
                    + (f"; sanity check re-asks {cfg.eval_sanity_pct:g}% of them"
                       + (", halving per model while answers match" if cfg.eval_sanity_adaptive else "")
                       if sanity else ""),
            "refresh": "refresh: FULL re-run, every question is asked again and replaces the cached answer",
            "false": "off: FULL re-run, nothing is reused or stored"}[mode] + "\n")
    w = max(len(m) for m in models)
    show_c = mode == "true" and cfg.run_eval_tests
    print(f"  {'model':<{w}}  {'speed':>8}  {'long-ctx':>8}  {'evals':>8}" + (f"  {'cached':>11}" if show_c else "")
          + "  source")
    for m in models:
        e = per[m]
        f = lambda k: fmt_hr_m_s(e[k]).replace(" hr ", "h").replace(" m ", "m").replace(" s", "s") if e.get(k) else "-"
        c = e.get("_cached")
        cs = (f"{c[0]}/{c[1]}" + (f" +{c[2]:.0f}?" if c[2] >= 0.5 else "") if c else "-") if show_c else None
        print(f"  {m:<{w}}  {f('speed'):>8}  {f('longctx'):>8}  {f('evals'):>8}" + (f"  {cs:>11}" if show_c else "")
              + f"  {e.get('_notes', '')}")
    reused = sum((e.get("_cached") or (0, 0, 0))[0] for e in per.values())
    print(f"\n  Total: ~{fmt_hr_m_s(total)}" + (f"  (the funnel assumes {int(FUNNEL_SURVIVE * 100)}% of models pass each "
                                               f"gate, never fewer than EVAL_LEADERS={cfg.eval_leaders})" if funnel else ""))
    if show_c:
        print(f"  cached = answers expected to be reused / questions expected to be asked"
              + (" (+ ~sanity re-asks)" if sanity else "") + (f"; {reused} reused in all" if reused else "; nothing "
              "cached yet") + ". Reuse needs the same model file, launch flags, llama.cpp build, context, hardware, "
              "EVAL_SEED and eval settings as the cached run; anything changed is asked again. Speed tests and "
              "long context are always measured.")
    if cfg.eval_concurrency > 1 and not gains:
        print("  EVAL_CONCURRENCY > 1 but no batch-probe result: no speed-up assumed. Run --batch-probe to measure it.")
    skipped = [m for m in models if per[m].get("_skip")]
    if skipped:
        print(f"  {len(skipped)} model(s) skipped (EVAL_SKIP_INELIGIBLE={cfg.eval_skip_ineligible}, from their last "
              "results): " + ", ".join(skipped))
    missing = [m for m in models if m not in history]
    if missing:
        print(f"  {len(missing)} model(s) without earlier results use a rough guess (GPU bandwidth {bw:g} GB/s from {bw_src}; "
              "ESTIMATE_GPU_BW_GBS sets it, "
              f"{int(THINKING_SHARE * 100)}% thinking models); a run records the real figures.")
    return total
