"""Speed / telemetry measurements for one model (run BEFORE the evaluations, one request at a time)."""
from __future__ import annotations

import random
import time

from . import modelinfo
from .client import post_chat, unload_all
from .common import BenchError, avg, col, median, vmax
from .config import PREFILL_SLOTS
from .scoring import blank, parse_quant
from .telemetry.host import HostMeter

_VOCAB = ("river stone copper window engine garden signal marble orbit harvest paper lantern thunder "
          "velvet compass anchor meadow circuit whisper granite ember canvas harbor quartz timber "
          "mirror falcon cobalt prairie summit voyage lattice monsoon plume fabric glacier beacon").split()


def prefill_prompt(words):
    """Long, unique-per-call prompt so no prefix cache can inflate prompt-processing speed."""
    rng = random.Random()
    body = " ".join(rng.choice(_VOCAB) for _ in range(words))
    return f"[{rng.getrandbits(32):08x}] Summarise the theme of this text in one short sentence.\n\n{body}"


def request_energy_j(gpu_w, cpu_w, wall, e0=None, e1=None):
    """Joules for one request. GPU part: the hardware energy-counter delta when NVML provides it (exact, independent
    of the polling rate), else mean sampled power x wall time. CPU part: mean power x wall time. None if neither."""
    gpu_j = (e1 - e0) / 1000.0 if e0 is not None and e1 is not None and e1 >= e0 else None
    if gpu_j is None and gpu_w is not None:
        gpu_j = gpu_w * wall
    if gpu_j is None and cpu_w is None:
        return None
    return (gpu_j or 0) + (cpu_w or 0) * wall


def measure(cfg, session, gpu, host, ha, model, prompt, max_tokens, timeout, cache=False, messages=None) -> dict:
    """One timed generation plus the telemetry captured during exactly that window."""
    payload = {"model": model, "messages": messages or [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": cfg.temperature, "stream": False,
               "cache_prompt": cache}
    a = host.snap()
    e0 = gpu.energy_mj()                      # exact GPU energy counter (NVML, opt-in) before the request ...
    t0 = time.time()
    try:
        data = post_chat(cfg, session, payload, timeout)
    finally:
        t1 = time.time()
        e1 = gpu.energy_mj()                  # ... and after it
        b = host.snap()

    usage, timings = data.get("usage") or {}, data.get("timings") or {}
    if not usage and not timings:
        raise BenchError("response contained neither 'usage' nor 'timings'")
    wall = t1 - t0
    tokens = usage.get("completion_tokens") or timings.get("predicted_n") or 0
    tps = timings.get("predicted_per_second")
    if tps is None and tokens and wall > 0:     # fallback includes prompt processing time
        tps = tokens / wall
    if tps is None:
        raise BenchError("could not determine tokens/sec")

    g = gpu.window(t0, t1)
    rw = host.ram.window(t0, t1)
    h = host.diff(a, b)
    cpu_w = h.get("cpu_w")
    if cpu_w is None and ha:
        cpu_w = ha.window_avg(t0, t1)
    gpu_w = avg(col(g, "gpu_w"))
    energy = request_energy_j(gpu_w, cpu_w, wall, e0, e1)
    return {"tps": float(tps), "prompt_tps": timings.get("prompt_per_second"),
            "ttft_ms": timings.get("prompt_ms"), "gen_ms": timings.get("predicted_ms"),
            "wall_s": wall, "tokens": tokens, "energy_j": energy,
            "cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
            "gpu_w": gpu_w, "gpu_peak_w": vmax(col(g, "gpu_w")), "gpu_util": avg(col(g, "gpu_util")),
            "gpu_temp": vmax(col(g, "gpu_temp")), "vram_gb": vmax(col(g, "vram_gb")),
            "vram_total_gb": vmax(col(g, "vram_total_gb")),
            "cpu_w": cpu_w, "cpu_util": h.get("cpu_util"),
            "ram_gb": vmax(col(rw, "ram_gb")), "proc_gb": vmax(col(rw, "proc_gb"))}


def cache_test(cfg, session, gpu, host, ha, model) -> dict:
    """Multi-turn proxy: send a long unique prompt (cold), then the same prompt plus a follow-up (warm).
    llama.cpp reports how many prompt tokens it reused (timings.cache_n)."""
    text = prefill_prompt(cfg.prefill_words)
    cold = measure(cfg, session, gpu, host, ha, model, text, 16, cfg.timeout_logic, cache=True)
    warm = measure(cfg, session, gpu, host, ha, model, text + "\n\nNow list three keywords.", 16,
                   cfg.timeout_logic, cache=True)
    reused, fresh = warm["cache_n"] or 0, warm["prompt_n"] or 0
    hit = 100 * reused / (reused + fresh) if reused + fresh else None
    speed = cold["ttft_ms"] / warm["ttft_ms"] if cold["ttft_ms"] and warm["ttft_ms"] else None
    return {"hit": hit, "speedup": speed}


# --------------------------------------------------------------------------- #
# bench_model, split into four steps (it was one 85-line function with five nested concerns)
# --------------------------------------------------------------------------- #

def _idle_baseline(cfg, session, gpu, host, model, strata, unload_state):
    """Step 1: clean slate. Unload everything, let the machine settle, sample the idle GPU/RAM so VRAM and power
    can be attributed to this model. Strata serves one always-loaded model: nothing to unload."""
    unloaded = False if strata else unload_all(cfg, session)
    if not strata and not unloaded and not unload_state["warned"]:
        unload_state["warned"] = True
        print("  [warn] could not unload models via the API; load time and VRAM delta will be skipped")
    time.sleep(min(cfg.settle, 1.0) if strata else cfg.settle)
    now = time.time()
    span = max(0.5, min(3.0, cfg.settle))
    base = gpu.window(now - span, now) if gpu.ok else []
    return {"unloaded": unloaded, "idle_gpu_w": avg(col(base, "gpu_w")), "base_vram": avg(col(base, "vram_gb")),
            "ram_idle": avg(col(host.ram.window(now - span, now), "ram_gb"))}


def _cold_load(cfg, session, host, model, r, strata, unloaded, unload_state):
    """Step 2: the first request loads the model (it doubles as the warm-up). Then reads what was loaded: context
    size, launch flags (KV type, -fa, -ub/-b, offload, mmproj) and the GGUF header (modelinfo.collect)."""
    t = time.time()
    post_chat(cfg, session, {"model": model, "max_tokens": 8, "stream": False,
                             "messages": [{"role": "user", "content": "hi"}]}, cfg.timeout_load)
    r["load_s"] = (time.time() - t) if unloaded else None
    host.ram.rescan()                           # the server process (re)started during the load
    modelinfo.collect(cfg, session, host, model, r, strata, unload_state)
    if r["ctx_size"] is None and not unload_state.get("ctx_warned"):
        unload_state["ctx_warned"] = True
        print("  [warn] could not read the context size ("
              + ("set STRATA_CTX in .env)" if strata else "tried /upstream/<model>/props and /running)"))


def check_min_ctx(cfg, r):
    """MIN_CTX: True / False / None (unknown or no minimum). With MIN_CTX_ACTION=refuse a model below the minimum
    is not benchmarked at all (status 'skipped')."""
    if not cfg.min_ctx:
        return None
    if not r.get("ctx_size"):
        print(f"  [warn] context size unknown: cannot check MIN_CTX {cfg.min_ctx:,}")
        return None
    ok = r["ctx_size"] >= cfg.min_ctx
    r["ctx_ok"] = ok
    if not ok:
        msg = f"loaded with ctx {r['ctx_size']:,}, below MIN_CTX {cfg.min_ctx:,}"
        if cfg.min_ctx_action == "refuse":
            r.update(status="skipped", error=msg + " (MIN_CTX_ACTION=refuse)")
        else:
            print(f"  [flag] {msg}: results are not representative of the deployed context; kept off the podium")
    return ok


def _timed_runs(cfg, session, gpu, host, ha, model):
    """Step 3: tech / logic / prefill prompts (RUNS_PER_PROMPT each) and the optional prompt-cache test."""
    spec = {"tech": (cfg.tech_prompt, cfg.max_tokens_tech, cfg.timeout_tech, False),
            "logic": (cfg.logic_prompt, cfg.max_tokens_logic, cfg.timeout_logic, False)}
    if cfg.prefill_words > 0:
        spec["prefill"] = (None, 16, cfg.timeout_logic, True)
    runs = {}
    for key, (prompt, mt, to, fresh) in spec.items():
        runs[key] = [measure(cfg, session, gpu, host, ha, model,
                             prefill_prompt(cfg.prefill_words) if fresh else prompt, mt, to)
                     for _ in range(cfg.runs)]
        if fresh:
            for x in runs[key]:
                x["_words"] = cfg.prefill_words
    cache_runs = []
    if cfg.prefill_words > 0 and cfg.cache_test:
        try:
            cache_runs = [cache_test(cfg, session, gpu, host, ha, model) for _ in range(cfg.runs)]
        except BenchError:
            cache_runs = []                     # the cache test is optional; never fail the model for it
    return runs, cache_runs


# --------------------------------------------------------------------------- #
# Deployed-context tests: prefill at long prompt lengths, agent-prompt time to first token
# --------------------------------------------------------------------------- #

def tokens_per_word(runs):
    """Measured prompt tokens per filler word (from the PREFILL_WORDS test), else a typical 1.2."""
    pre = [x for x in runs.get("prefill", []) if x.get("prompt_n")]
    return median([x["prompt_n"] / (x["_words"] + 14) for x in pre if x.get("_words")]) or 1.2


def _fit(target, ctx, reserve=96):
    """Prompt size that is `target` tokens but never more than the context minus room for the reply."""
    return min(target, ctx - reserve) if ctx else target


def prefill_at_lengths(cfg, session, gpu, host, ha, model, r, tpw):
    """PREFILL_CTX_TESTS: one cold, unique prompt per length (no prefix cache), prompt tokens/s from llama-server's
    timings. A length longer than the loaded context is skipped (blank)."""
    for n in cfg.prefill_ctx_tests:
        if r.get("ctx_size") and n > r["ctx_size"]:
            continue
        words = max(50, int((_fit(n, r.get("ctx_size")) * 0.97 - 20) / tpw))
        try:
            x = measure(cfg, session, gpu, host, ha, model, prefill_prompt(words), 4, max(cfg.timeout_logic, n / 100))
        except BenchError as e:
            print(f"    [warn] prefill at {PREFILL_SLOTS[n]}: {e}")
            continue
        r[f"prefill_{PREFILL_SLOTS[n]}_tps"] = x["prompt_tps"]
    got = [f"{PREFILL_SLOTS[n]} {r[f'prefill_{PREFILL_SLOTS[n]}_tps']:.0f}" for n in cfg.prefill_ctx_tests
           if r.get(f"prefill_{PREFILL_SLOTS[n]}_tps")]
    if got:
        print("  prefill t/s by prompt length: " + ", ".join(got))


_TOOL_STUB = ('{"type":"function","function":{"name":"%s","description":"%s","parameters":{"type":"object",'
              '"properties":{"query":{"type":"string"},"limit":{"type":"integer"},"path":{"type":"string"}},'
              '"required":["query"]}}}')
_TOOL_NAMES = ["web_search", "read_file", "write_file", "list_dir", "run_shell", "ha_get_state", "ha_call_service",
               "calendar_events", "send_message", "memory_search", "memory_write", "fetch_url", "python_exec",
               "image_describe", "todo_add", "todo_list"]


def agent_messages(tokens, tpw, seed=None):
    """An agent-shaped request of about `tokens` tokens: a system prompt with a tool catalogue, then a long history of
    turns and tool results, then a new user message. Unique per call (random filler), so nothing is cached."""
    rng = random.Random(seed)
    tools = ",".join(_TOOL_STUB % (t, " ".join(rng.choice(_VOCAB) for _ in range(12))) for t in _TOOL_NAMES)
    system = ("You are a home-server assistant agent. Use the tools when needed and answer briefly.\n"
              f"[{rng.getrandbits(32):08x}] Available tools: [{tools}]")
    budget = max(200, int(tokens / tpw) - len(system.split()) * 2)
    msgs, turn = [{"role": "system", "content": system}], 0
    while budget > 0:
        n = min(budget, rng.randint(150, 600))
        body = " ".join(rng.choice(_VOCAB) for _ in range(n))
        role = "user" if turn % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": (f"Tool result {turn}: " if role == "assistant" else "") + body})
        budget -= n
        turn += 1
    if msgs[-1]["role"] == "user":
        msgs.append({"role": "assistant", "content": "Noted."})
    msgs.append({"role": "user", "content": "Summarise what the last tool result was about in one sentence."})
    return msgs


def agent_ttft(cfg, session, gpu, host, ha, model, r, tpw):
    """AGENT_PROMPT_TOKENS: time to first token for a cold agent-sized prompt, then for the next turn of the same
    conversation (prefix cached, as an agent sees it). Skipped if the prompt does not fit the context."""
    n = cfg.agent_prompt_tokens
    if not n or (r.get("ctx_size") and n + 512 > r["ctx_size"]):
        return
    msgs = agent_messages(_fit(n, r.get("ctx_size"), 512) * 0.97, tpw)
    to = max(cfg.timeout_logic, n / 100)
    try:
        cold = measure(cfg, session, gpu, host, ha, model, None, 8, to, cache=True, messages=msgs)
        warm = measure(cfg, session, gpu, host, ha, model, None, 8, to, cache=True,
                       messages=msgs + [{"role": "assistant", "content": "It was a status report."},
                                        {"role": "user", "content": "Thanks. Anything urgent in it?"}])
    except BenchError as e:
        print(f"    [warn] agent-prompt TTFT: {e}")
        return
    r["agent_prompt_tokens"] = cold.get("prompt_n")
    r["agent_ttft_s"] = cold["ttft_ms"] / 1000 if cold.get("ttft_ms") else cold["wall_s"]
    r["agent_ttft_warm_s"] = warm["ttft_ms"] / 1000 if warm.get("ttft_ms") else None
    print(f"  agent prompt ({r['agent_prompt_tokens'] or '?'} tok): TTFT {r['agent_ttft_s']:.1f}s cold"
          + (f", {r['agent_ttft_warm_s']:.2f}s next turn" if r["agent_ttft_warm_s"] is not None else ""))


def _summarise(r, runs, cache_runs, base):
    """Step 4: collapse the raw runs into the per-model row (medians for speed, means/maxima for telemetry)."""
    med = lambda k, f: median([x[f] for x in runs[k]])
    gen = runs["tech"] + runs["logic"]
    every = gen + runs.get("prefill", [])
    r["tech_tps"], r["logic_tps"] = med("tech", "tps"), med("logic", "tps")
    if "prefill" in runs:
        r["prefill_tps"] = med("prefill", "prompt_tps")
        r["prefill_tokens"] = med("prefill", "prompt_n")
    if cache_runs:
        r["cache_hit_pct"] = median([c["hit"] for c in cache_runs])
        r["cache_speedup"] = median([c["speedup"] for c in cache_runs])
    r["ttft_ms"] = avg([v for v in (med("tech", "ttft_ms"), med("logic", "ttft_ms")) if v is not None])
    r["gen_ms"] = avg([v for v in (med("tech", "gen_ms"), med("logic", "gen_ms")) if v is not None])
    r["latency_s"] = avg([med("tech", "wall_s"), med("logic", "wall_s")])
    r["gpu_w"], r["gpu_util"] = avg(col(gen, "gpu_w")), avg(col(gen, "gpu_util"))
    r["cpu_w"], r["cpu_util"] = avg(col(gen, "cpu_w")), avg(col(gen, "cpu_util"))
    r["gpu_peak_w"], r["gpu_temp"] = vmax(col(every, "gpu_peak_w")), vmax(col(every, "gpu_temp"))
    r["vram_gb"], r["vram_total_gb"] = vmax(col(every, "vram_gb")), vmax(col(every, "vram_total_gb"))
    r["ram_used_gb"], r["ram_proc_gb"] = vmax(col(every, "ram_gb")), vmax(col(every, "proc_gb"))
    r["ram_total_gb"] = HostMeter._mem_total_gb()
    if r["ram_used_gb"] is not None and base["ram_idle"] is not None and base["unloaded"]:
        r["ram_idle_gb"], r["ram_delta_gb"] = base["ram_idle"], max(0.0, r["ram_used_gb"] - base["ram_idle"])
    if r["vram_gb"] is not None and base["base_vram"] is not None and base["unloaded"]:
        r["vram_delta_gb"] = max(0.0, r["vram_gb"] - base["base_vram"])
    e_tech, e_logic = avg(col(runs["tech"], "energy_j")), avg(col(runs["logic"], "energy_j"))
    if e_tech is not None and e_logic is not None:
        r["energy_wh"] = (e_tech + e_logic) / 3600
        tokens = sum(x["tokens"] for x in gen)
        r["j_per_token"] = sum(col(gen, "energy_j")) / tokens if tokens else None


def bench_model(cfg, session, gpu, host, ha, model, unload_state) -> dict:
    r = blank(model)
    strata = cfg.backend == "strata"
    if strata:      # no filename to parse, so take the model facts from .env
        r.update(params_b=cfg.strata_params_b, active_b=cfg.strata_active_b,
                 quant=cfg.strata_quant or parse_quant(cfg.strata_model_id or model))
    try:
        base = _idle_baseline(cfg, session, gpu, host, model, strata, unload_state)
        r["idle_gpu_w"] = base["idle_gpu_w"]
        _cold_load(cfg, session, host, model, r, strata, base["unloaded"], unload_state)
        check_min_ctx(cfg, r)
        if r["status"] == "skipped":
            return r
        runs, cache_runs = _timed_runs(cfg, session, gpu, host, ha, model)
    except BenchError as e:
        r.update(status="error", error=str(e))
        return r
    _summarise(r, runs, cache_runs, base)
    tpw = tokens_per_word(runs)
    try:                                        # optional, slow tests: a failure never fails the model
        prefill_at_lengths(cfg, session, gpu, host, ha, model, r, tpw)
        agent_ttft(cfg, session, gpu, host, ha, model, r, tpw)
    except BenchError as e:
        print(f"    [warn] long-prompt tests stopped: {e}")
    r["tok_per_word"] = tpw
    return r
