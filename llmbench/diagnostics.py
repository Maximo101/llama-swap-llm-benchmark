"""--demo synthetic data and the --check telemetry self-test."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .client import make_session
from .config import BUDGET_DEFAULTS, EVAL_SPECS
from .evals.sandbox import resolve_sandbox
from .scoring import blank, parse_params
from .telemetry import HostMeter, RamSampler, detect_hardware, find_listener_pids
from .telemetry.gpu import NvmlGpuMonitor, SmiGpuMonitor

def demo_results(cfg):
    specs = [("Demo-9B-Q4_K_M.gguf", 110, 6.5, 120, 25), ("Demo-27B-IQ3_XXS.gguf", 42, 12.8, 260, 40),
             ("Demo-35B-A3B-IQ3_XXS.gguf", 88, 15.1, 190, 85), ("Demo-8B-Q8_0.gguf", 95, 9.9, 170, 20),
             ("Demo-30B-A3B-Q4_0.gguf", 27, 15.6, 210, 70), ("Demo-26B-A4B-Q4_K_XL.gguf", 64, 14.2, 200, 55)]
    out = []
    for name, tps, vram, gw, cw in specs:
        r = blank(name)
        lat = 300 / tps + 1
        r.update(ctx_size=65536 if vram > 12 else 131072, prefill_tokens=1500, cache_hit_pct=97 - vram / 4,
                 cache_speedup=6 + tps / 20, load_s=vram / 2.5, tech_tps=tps * 1.04, logic_tps=tps * 0.96, prefill_tps=tps * 14,
                 ttft_ms=60 + 15000 / tps, gen_ms=300 / tps * 1000, latency_s=lat, gpu_w=gw,
                 gpu_peak_w=gw * 1.15, idle_gpu_w=12, gpu_util=88, gpu_temp=62 + gw / 20, vram_gb=vram,
                 vram_delta_gb=vram - 0.6, vram_total_gb=cfg.vram_limit_gb, cpu_w=cw, cpu_util=25,
                 ram_total_gb=64, ram_used_gb=24 + cw / 4, ram_idle_gb=24, ram_delta_gb=cw / 4, ram_proc_gb=(cw / 3 if cw >= 55 else 1.4),
                 offload=("n-cpu-moe 22" if cw >= 55 else "none"), energy_wh=(gw + cw) * lat * 2 / 3600, j_per_token=(gw + cw) / tps)
        rng = random.Random(name)
        moe = "A3B" in name or "A4B" in name
        kib = rng.choice([64, 96, 128, 160]) if not moe else rng.choice([20, 40, 60])
        kvt = rng.choice(["q8_0", "q8_0", "f16", "q4_0"])
        r.update(kv_type=kvt, flash_attn="on", ubatch=rng.choice([512, 1024, 2048]), batch=2048, jinja=True,
                 ngl=99, n_cpu_moe=22 if cw >= 55 else None, mmproj="mmproj-F16.gguf" if "26B" in name else None,
                 gguf_gb=vram * 0.75, bpw=rng.choice([3.1, 4.5, 4.9, 8.5]), gguf_arch="qwen3moe" if moe else "qwen3",
                 attn_type=("hybrid GQA + SSM" if "35B" in name else "MLA" if "30B" in name else "GQA"),
                 n_layers=48, full_attn_layers=12 if "35B" in name else 48, n_ctx_train=262144,
                 expert_count=128 if moe else None, expert_used=8 if moe else None, kv_kib_token=kib,
                 tok_per_word=1.2, ctx_ok=r["ctx_size"] >= 65536,
                 prefill_8k_tps=tps * 12, prefill_32k_tps=tps * 9, prefill_64k_tps=tps * 6,
                 agent_prompt_tokens=20000, agent_ttft_s=20000 / (tps * 10), agent_ttft_warm_s=0.3,
                 longctx_64k_pct=rng.choice([100, 87.5, 75, 62.5]), longctx_kv_ok=kvt != "q4_0",
                 longctx_128k_pct=rng.choice([87.5, 62.5, 50]) if r["ctx_size"] >= 131072 else None,
                 longctx_note=None if r["ctx_size"] >= 131072 else "128k: ctx 64k too small")
        q = min(1.0, 0.35 + (parse_params(name) or 10) / 45)       # bigger demo models score higher
        for key, (lo, hi) in {"gpqa": (22, 58), "mmlu_pro": (30, 82), "humaneval": (45, 94),
                              "mbpp": (40, 84), "livecodebench": (5, 48),
                              "gsm8k": (55, 96), "math": (12, 78)}.items():
            r[f"eval_{key}_pct"] = lo + (hi - lo) * q * rng.uniform(0.85, 1.0)
            r[f"eval_{key}_n"], r[f"eval_{key}_err"] = 50, 0
            r[f"eval_{key}_trunc"] = rng.choice([0.0, 0.0, 2.0, 6.0])
            r[f"eval_{key}_degr"] = rng.choice([0, 0, 0, 1, 3])
            r[f"eval_{key}_sec"] = round(rng.uniform(8, 60) * (150 / tps) * (3 if key == "gpqa" else 1), 1)
            r[f"eval_{key}_fallback"] = rng.choice([0, 0, 1, 2, 25])
        r["eval_toolcall_pct"], r["eval_toolcall_n"] = 60 + 40 * q * rng.uniform(0.8, 1.0), 32
        out.append(r)
    cfg.eval_info = {"gpqa": {"rows": 50, "available": 198}, "mmlu_pro": {"rows": 50, "available": 12032},
                     "humaneval": {"rows": 50, "available": 164}, "mbpp": {"rows": 50, "available": 500},
                     "livecodebench": {"rows": 50, "available": 1055},
                     "gsm8k": {"rows": 50, "available": 1319}, "math": {"rows": 50, "available": 500},
                     "toolcall": {"rows": 32, "available": 32}}
    cfg.eval_longctx = True
    cfg.longctx_lengths = (65536, 131072)
    cfg.prefill_ctx_tests = (8192, 32768, 65536)
    cfg.agent_prompt_tokens = 20000
    if cfg.budget is None:                      # show the per-rule columns in the demo
        cfg.budget = {"name": "server", "rules": dict(BUDGET_DEFAULTS["server"])}
    cfg.min_ctx = cfg.min_ctx or 65536
    cfg.eval_sandbox_resolved = "subprocess"
    return out


def _check_servers(cfg):
    print(f"Hardware: {detect_hardware(cfg) or 'not detected'}")
    print(f"llama-swap: {cfg.base_url}")
    ls_ok = False
    try:
        n = len(make_session(cfg, "llama-swap").get(cfg.models_url, timeout=5).json().get("data", []))
        print(f"  OK, {n} model(s) listed")
        ls_ok = True
    except (requests.RequestException, ValueError) as e:
        print(f"  FAILED: {e.__class__.__name__}")
    if not cfg.strata_url:
        return
    print(f"Strata fallback: {cfg.strata_url} (API key {'set' if cfg.strata_api_key else 'NOT set'})"
          + ("  [not used while llama-swap is up]" if ls_ok else ""))
    s = make_session(cfg, "strata")
    for method, path, body in (("GET", "/health", None), ("GET", "/v1/models", None),
                               ("POST", "/v1/chat/completions",
                                {"model": cfg.strata_model or "strata", "max_tokens": 1,
                                 "messages": [{"role": "user", "content": "hi"}]})):
        try:
            r = s.request(method, cfg.strata_url + path, json=body, timeout=15)
            hint = "  <- authentication failed: check STRATA_API_KEY" if r.status_code in (401, 403) else ""
            print(f"  {method} {path} -> HTTP {r.status_code}{hint}")
        except requests.RequestException as e:
            print(f"  {method} {path} -> FAILED: {e.__class__.__name__}")


def _check_gpu(cfg):
    """Every GPU backend is tried (not just the one GPU_BACKEND would pick) so --check shows what each can do."""
    for cls, label in ((NvmlGpuMonitor, "NVML (pynvml)"), (SmiGpuMonitor, "nvidia-smi stream")):
        mon = cls(cfg)
        ok = mon.start()
        time.sleep(1.5)
        if ok:
            s = mon.window(time.time() - 1.5, time.time())
            print(f"GPU ({label}): OK, {len(s)} samples in 1.5s; latest = "
                  + ", ".join(f"{k}={v:.1f}" for k, v in s[-1].items() if v is not None))
            if cls is NvmlGpuMonitor and cfg.gpu_energy_counter:
                print("  exact energy counter: " + ("supported" if mon.energy_mj() is not None else "NOT supported by this GPU"))
        else:
            hint = ("pip install nvidia-ml-py  (and check the NVIDIA driver is loaded)" if cls is NvmlGpuMonitor
                    else "is nvidia-smi on PATH and the driver loaded?")
            print(f"GPU ({label}): NOT AVAILABLE - {hint}")
        mon.stop()


def _check_eval_client(cfg):
    try:
        import aiohttp
        print(f"Async eval client: aiohttp {aiohttp.__version__} (EVAL_CONCURRENCY={cfg.eval_concurrency})")
    except ImportError:
        print("Async eval client: aiohttp not installed -> evals use a thread pool (pip install aiohttp for the native client)")
    if cfg.eval_stream:
        print(f"Eval streaming: ON  (thinking budget {cfg.eval_think_budget or 'unlimited'}"
              f"{' tokens' if cfg.eval_think_budget else ''}, loop abort {'on' if cfg.eval_loop_abort else 'off'}, "
              f"salvage {'on' if cfg.eval_salvage else 'off'}, timeout {cfg.eval_timeout:.0f}s)")
    else:
        print("Eval streaming: OFF (EVAL_STREAM=false) - no loop abort, no salvage; max_tokens is the only limit")


def _check_host(cfg):
    host = HostMeter(cfg)
    a = host.snap(); time.sleep(2); h = host.diff(a, host.snap())
    print(f"CPU power via RAPL: {'OK, %.1f W' % h['cpu_w'] if 'cpu_w' in h else 'NOT AVAILABLE'}"
          + ("" if host.rapl_ok else f" (no readable domains under {cfg.rapl_path}; run as root, or set HA_* as a fallback)"))
    used, total = h.get("sys_mem_gb"), HostMeter._mem_total_gb()
    print(f"CPU util: {h.get('cpu_util', float('nan')):.0f}%   RAM: "
          + (f"{used:.1f} / {total:.1f} GB used ({used / total * 100:.0f}%)" if used and total else "unavailable"))


def _comm(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


def _largest_processes(ram, n=5):
    big = [(gb, _comm(d), d) for d in os.listdir("/proc") if d.isdigit() for gb in [ram.rss_gb([int(d)])] if gb]
    return sorted(big, reverse=True)[:n]


def _check_server_process(cfg):
    ram = RamSampler(cfg)
    if cfg.strata_url:
        port = urlparse(cfg.strata_url).port or 80
        ram.listen_port = port
        pids = find_listener_pids(port)
        print(f"Strata server process (listening on port {port}): "
              + (", ".join(f"{_comm(p)} (pid {p}) using {ram.rss_gb([p]) or 0:.1f} GB RAM" for p in sorted(pids))
                 if pids else "not found on this host (needs root; Docker bridge ports show docker-proxy)"))
        print("  Run `python benchmark.py --probe-strata` to see what Strata's API exposes.")
    procs = ram.find_processes()
    for pid in sorted(procs):
        print(f"Inference process matching RAM_PROCESS_PATTERN={cfg.ram_process_pattern!r}: "
              f"{_comm(pid)} (pid {pid}) using {ram.rss_gb([pid]) or 0:.1f} GB RAM")
        _check_launch_and_gguf(cfg, pid)
    if not procs:
        print(f"No process matches RAM_PROCESS_PATTERN={cfg.ram_process_pattern!r} (server RAM will be blank; "
              "RAM spill falls back to the system-RAM delta). Largest processes by RAM - set the pattern to your "
              "inference server's name: " + ", ".join(f"{n} {g:.1f} GB" for g, n, _ in _largest_processes(ram)))


def _check_launch_and_gguf(cfg, pid):
    """What v2.2 reads from a running llama-server: its flags and the GGUF header (only shown when a model is loaded)."""
    from .gguf import GGUFError, find_readable, inspect, kv_bytes
    from .modelinfo import kv_label
    from .telemetry.host import parse_launch_args
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore").split("\0")
    except OSError as e:
        print(f"  launch flags: not readable ({e})")
        return
    info = parse_launch_args([a for a in argv[1:] if a])
    kv = kv_label(info["ctk"], info["ctv"]) or "f16 (default)"
    print(f"  flags: ctx {info['ctx']}, KV {kv}, -fa {info['fa'] or 'auto'}, -ub {info['ubatch'] or 512} / "
          f"-b {info['batch'] or 2048}, -np {info['parallel'] or 1}, mmproj {'yes' if info['mmproj'] else 'no'}, "
          f"--jinja {'yes' if info['jinja'] else 'no'}")
    if not info["model_path"]:
        return
    real = find_readable(info["model_path"], pid, cfg.gguf_path_map, cfg.gguf_search_dirs)
    if not real:
        print(f"  GGUF {info['model_path']}: NOT readable from here (run as root, or set GGUF_PATH_MAP / GGUF_SEARCH_DIRS)")
        return
    try:
        g = inspect(real)
    except (OSError, GGUFError) as e:
        print(f"  GGUF {real}: header unreadable ({e})")
        return
    k, v = ((info["ctk"] or "f16"), (info["ctv"] or "f16"))
    per_tok = kv_bytes(g, k, v)[0]
    print(f"  GGUF {real}: {g['file_bytes'] / 1024 ** 3:.2f} GB, {g['bpw']:.2f} bpw, arch {g['gguf_arch']}, "
          f"{'MoE ' + str(g['expert_used']) + '/' + str(g['expert_count']) if g['expert_count'] else 'dense'}, "
          f"{g.get('attn_type')}, {g.get('full_attn_layers')} full-attention layers"
          + (f", KV {per_tok / 1024:.0f} KiB/token" if per_tok else ""))


def check(cfg):
    """Verify every telemetry source for a few seconds. Run this before a long benchmark."""
    _check_servers(cfg)
    _check_gpu(cfg)
    _check_eval_client(cfg)
    _check_host(cfg)
    _check_server_process(cfg)
    if cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity:
        print("Home Assistant CPU-power fallback: configured")
    if any(EVAL_SPECS[k]["kind"] == "code" for k in cfg.eval_datasets):
        print("Code sandbox:")
        resolve_sandbox(cfg)
