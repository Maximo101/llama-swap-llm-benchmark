"""--batch-probe: measure what answering several eval questions at once gains on THIS GPU, before relying on it.

For each selected model (set INCLUDE_MODELS to 2-3 representative ones: a small dense, a mid dense, an offloaded MoE)
the same BATCH_PROBE_QUESTIONS questions (spread over the enabled datasets, nested sample) are asked with
BATCH_PROBE_LEVELS requests in flight (default 1, 2, 4, 8). Per level it records:

  wall time, aggregate generation tok/s and its gain over level 1, mean per-question latency, accuracy, how many
  answers AGREE with level 1 (same verdict and same extracted answer), request errors, and peak VRAM.

Why agreement matters: llama.cpp is not batch-invariant, so greedy decoding can produce slightly different tokens when
requests share a batch. A few flips are noise; many flips, or errors (a shared KV pool that overflows), mean that
model should not be batched. The recommended level is the smallest one that reaches 90% of the best throughput with
>= 95% agreement, no errors, and accuracy within the margin of error of level 1.

The server needs slots for this: llama-server -np N (N >= the highest level) and, to keep the full context per
request, --kv-unified (the default when -np is left at auto in current llama.cpp). Results go to
results/batch_probe_<stamp>.json, which --estimate reads.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from copy import copy
from datetime import datetime

import sys

from . import modelinfo
from .client import detect_backend, get_props, post_chat
from .common import BenchError, avg, now
from .config import EVAL_SPECS
from .evals.datasets import prepare_eval_data
from .evals.runner import DatasetRun, ModelEval, per_request_tokens
from .evals.transport import open_transport
from .telemetry import HostMeter, open_gpu_monitor

MAX_MODELS = 4                  # a probe is for 2-3 representative models, not the whole server


class _Phase:
    """The minimum ModelEval needs (no cache). Records go to batch_probe_<stamp>_details.jsonl, so answers that
    differ between levels can be inspected."""

    def __init__(self, cfg, transport, pool, sink=None):
        self.cfg, self.transport, self.pool, self.cache, self.sanity = cfg, transport, pool, None, None
        self.sink = sink or (lambda rec: None)


def _questions(cfg):
    keys = list(cfg.eval_datasets)
    if not keys:
        return {}
    per = max(1, math.ceil(cfg.batch_probe_questions / len(keys)))
    data = prepare_eval_data(cfg, rows=per)
    return {k: v[:per] for k, v in data.items()}


def steady_gain(lv, base):
    """Throughput gain of level L over one request at a time, as it would be in a long run where the queue is always
    full: L requests in flight, each generating at (its tokens / its latency). Little's law: throughput = in flight /
    latency, per token. The wall time of a short probe is dominated by the slowest question of the last batch (with 4
    questions per dataset and 4 in flight, every dataset is ONE batch), which understates the gain badly; the first
    probe version reported tokens / wall time and showed 1.2-1.45x where the steady state is 1.9-3.3x."""
    try:
        tok_l, tok_1 = lv["tok_s"] * lv["wall_s"], base["tok_s"] * base["wall_s"]
        per_l = tok_l / (lv["latency_s"] * lv["n"])          # tokens/s of one request at level L
        per_1 = tok_1 / (base["latency_s"] * base["n"])
        return round(lv["level"] * per_l / per_1, 2)
    except (TypeError, ZeroDivisionError, KeyError):
        return None


def recommend(levels):
    """The smallest level that reaches >= 90% of the best steady-state gain, with no request errors and accuracy
    within the margin of error of level 1. Agreement is reported but not a hard rule for small samples: greedy decoding
    in long reasoning is chaotic, so a model near 90% accuracy flips ~15-20% of its answers under ANY numeric change
    (batching, another build, another GPU) - noise, not a bias. It becomes a rule (>= 90%) from 40 questions up."""
    base = next((x for x in levels if x["level"] == 1), None)
    if not base:
        return None
    moe = 98 / math.sqrt(base["n"]) if base["n"] else 100
    ok = [x for x in levels if x["errors"] == 0 and abs((x["acc"] or 0) - (base["acc"] or 0)) <= moe
          and (x["n"] < 40 or x["agree"] is None or x["agree"] >= 0.90)]
    if not ok:
        return 1
    best = max(x.get("steady_gain") or 1.0 for x in ok)
    return min((x for x in ok if (x.get("steady_gain") or 1.0) >= 0.9 * best), key=lambda x: x["level"])["level"]


def _other_models(cfg, session, model):
    """Models other than `model` that llama-swap reports as running: another client (e.g. a benchmark run still
    going) is using the server, which makes every timing of the probe meaningless."""
    try:
        r = session.get(cfg.base_url + "/running", timeout=5)
        j = r.json() if r.ok else None
    except Exception:
        return []
    entries = j.get("running", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
    return sorted({e.get("model") for e in entries if isinstance(e, dict) and e.get("model")
                   and e.get("model") != model and str(e.get("state", "ready")).lower() not in ("stopped", "stopping")})


async def _probe_model(cfg, session, transport, pool, model, data, gpu, host, sink=None):
    try:
        post_chat(cfg, session, {"model": model, "max_tokens": 1, "stream": False,
                                 "messages": [{"role": "user", "content": "hi"}]}, cfg.timeout_load)
    except BenchError as e:
        print(f"  [warn] could not load {model}: {e}")
        return None
    # The same reader the speed phase uses: /props, then the server's real command line from /proc (as root on the
    # server host), then llama-swap's /running. (The first probe version only tried /props and /running.)
    r = {"model": model, "status": "ok"}
    host.ram.rescan()
    modelinfo.collect(cfg, session, host, model, r, False, {})
    props = get_props(cfg, session, model)
    me = ModelEval(_Phase(cfg, transport, pool, sink), r, r.get("ctx_size"), props, None)
    if not me.ctx:
        # Without the context size every reply would be capped at the 4,096-token fallback: reasoning models get cut
        # off, retried without thinking, and the probe would measure that instead of the real workload.
        print(f"\n{model}: [skip] context size not readable (/props, /proc and /running all failed), so replies would "
              "be capped at 4,096 tokens. Run the probe as root on the server host (python benchmark.py --check shows "
              "what can be read).")
        return None
    print(f"\n{model}: slots {me.slots or '?'}, ctx {me.ctx or '?'}, unified KV "
          f"{'yes' if me.kv_unified else 'no' if me.kv_unified is False else 'unknown'}")
    levels, base_ans = [], None
    for level in cfg.batch_probe_levels:
        if me.slots and level > me.slots:
            print(f"  x{level}: skipped (the server has {me.slots} slot(s); raise -np to test it)")
            continue
        if me.ctx and me.kv_unified is not False:
            per = max(per_request_tokens(cfg, k, it, me.ctx) for k, items in data.items() for it in items)
            if level * per > me.ctx:
                print(f"  x{level}: [note] {level} requests x up to {per:,} tokens exceed the {me.ctx:,}-token pool; "
                      "errors here mean the pool overflowed")
        others = _other_models(cfg, session, model)
        if others:
            print(f"  [warn] llama-swap is also serving {', '.join(others)}: another client is using the server, so "
                  "these timings are not valid. Stop it and probe again.")
        recs, wall = [], 0.0
        t0 = now()
        for key, items in data.items():
            got, w = await DatasetRun(me, key, items, f"probe-x{level}", level).run()
            recs += got
            wall += w
        vram = [s.get("vram_gb") for s in gpu.window(t0, now())] if gpu and gpu.ok else []
        vram = [v for v in vram if v is not None]
        tok = sum(x.get("tokens") or 0 for x in recs)
        ans = {(x["dataset"], x["id"]): (bool(x["correct"]), x.get("answer")) for x in recs}
        if level == 1:
            base_ans = ans
        agree = (sum(1 for k, v in ans.items() if base_ans.get(k) == v) / len(ans)) if base_ans and ans else None
        lv = {"level": level, "n": len(recs), "wall_s": round(wall, 1), "tok_s": round(tok / wall, 1) if wall else None,
              "latency_s": round(avg([x["seconds"] for x in recs]) or 0, 2),
              "acc": round(100 * sum(bool(x["correct"]) for x in recs) / len(recs), 1) if recs else None,
              "agree": round(agree, 3) if agree is not None else None,
              "errors": sum(1 for x in recs if x.get("api_error")), "vram_peak_gb": max(vram) if vram else None}
        levels.append(lv)
    base = next((x for x in levels if x["level"] == 1), None)
    for lv in levels:
        lv["gain"] = round(lv["tok_s"] / base["tok_s"], 2) if base and base["tok_s"] and lv["tok_s"] else None
        lv["steady_gain"] = steady_gain(lv, base) if base else None
    print(f"  {'level':>5}  {'wall':>7}  {'tok/s':>7}  {'gain':>5}  {'s/q':>6}  {'acc':>6}  {'agree':>6}  {'err':>3}  VRAM"
          "      (gain = steady state, full queue)")
    for lv in levels:
        print(f"  x{lv['level']:<4}  {lv['wall_s']:>6.0f}s  {lv['tok_s'] or 0:>7.1f}  {lv['steady_gain'] or 0:>5.2f}  "
              f"{lv['latency_s']:>6.1f}  {lv['acc'] or 0:>5.1f}%  "
              f"{(str(round(100 * lv['agree'])) + '%') if lv['agree'] is not None else '-':>6}  {lv['errors']:>3}  "
              f"{(str(round(lv['vram_peak_gb'], 1)) + ' GB') if lv['vram_peak_gb'] else '-'}")
    rec = recommend(levels)
    print(f"  recommended EVAL_CONCURRENCY for this model: {rec}")
    return {"model": model, "slots": me.slots, "ctx": me.ctx, "kv_unified": me.kv_unified, "levels": levels,
            "recommended": rec}


def batch_probe(cfg):
    from concurrent.futures import ThreadPoolExecutor
    cfg = copy(cfg)
    cfg.eval_result_cache = False
    cfg.eval_log_interval = 0
    if not cfg.eval_datasets:
        cfg.eval_datasets = ("mmlu_pro", "humaneval")
        print("No dataset enabled in .env: probing with MMLU-Pro and HumanEval")
    session, models = detect_backend(cfg)
    if len(models) > MAX_MODELS:
        sys.exit(f"[batch-probe] {len(models)} models selected. Probe 2-3 representative ones, e.g.\n"
                 "  INCLUDE_MODELS=small-moe.gguf,offloaded-moe.gguf,dense-27b.gguf BATCH_PROBE_LEVELS=1,4 "
                 "BATCH_PROBE_QUESTIONS=16 python benchmark.py --batch-probe\n"
                 f"(at most {MAX_MODELS}; every model x level answers the same questions, so a full set takes hours)")
    busy = sorted({o for m in models for o in _other_models(cfg, session, m)} - set(models))
    if busy:
        print(f"[warn] llama-swap is serving other model(s) right now ({', '.join(busy)}). If another benchmark run is "
              "still going, stop it first: the probe's timings are only valid on an otherwise idle server.")
    data = _questions(cfg)
    gpu = open_gpu_monitor(cfg)
    host = HostMeter(cfg)
    host.ram.start()
    out = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    det = open(cfg.output_dir / f"batch_probe_{stamp}_details.jsonl", "a", encoding="utf-8")

    def sink(rec):
        det.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        det.flush()

    async def go():
        transport = await open_transport(cfg, max(cfg.batch_probe_levels))
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="probe-grade")
        try:
            for m in models:
                res = await _probe_model(cfg, session, transport, pool, m, data, gpu, host, sink)
                if res:
                    out.append(res)
        finally:
            await transport.close()
            pool.shutdown(wait=False, cancel_futures=True)

    try:
        asyncio.run(go())
    finally:
        gpu.stop()
        host.ram.stop()
        det.close()
    path = cfg.output_dir / f"batch_probe_{stamp}.json"
    path.write_text(json.dumps({"levels": list(cfg.batch_probe_levels), "questions": {k: len(v) for k, v in data.items()},
                                "datasets": [EVAL_SPECS[k]["label"] for k in data], "models": out}, indent=2),
                    encoding="utf-8")
    print(f"\nBatch probe written to {path} (--estimate uses it)")
