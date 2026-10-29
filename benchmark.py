#!/usr/bin/env python3
"""
llama-swap benchmark (run it ON the machine that has the GPU)
=============================================================
For every model on a llama-swap server it measures, under real load:

  * generation speed (Technical + Logic prompts), prefill speed, load time, TTFT*
  * GPU power / utilisation / temperature / VRAM  -> one persistent `nvidia-smi -lms` stream
  * CPU power (Intel/AMD RAPL, or a Home Assistant sensor), CPU utilisation, RAM
  * idle baseline per model (after unloading everything) so VRAM/power are attributable
  * energy (Wh) and energy per token (J/token)

and writes a CSV, a self-contained interactive HTML dashboard (Chart.js) and an optional
Markdown report. Everything is configured through a .env file (see .env.example).

    python benchmark.py --check    # verify telemetry sources before a long run
    python benchmark.py            # run the benchmark
    python benchmark.py --demo     # build a dashboard from synthetic data

* TTFT is approximated by llama.cpp's `timings.prompt_ms` (prompt processing time).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _load_dotenv():
    """python-dotenv if installed, otherwise a minimal built-in .env reader."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(name, default=None, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        return cast(raw.strip())
    except ValueError:
        sys.exit(f"[config] Invalid value for {name}: {raw!r}")


@dataclass
class Config:
    base_url: str
    api_key: str | None
    vram_limit_gb: float
    gpu_index: str | None
    gpu_poll_ms: int
    rapl_path: str
    ha_url: str | None
    ha_token: str | None
    ha_cpu_entity: str | None
    ha_interval: float
    timeout_tech: float
    timeout_logic: float
    timeout_load: float
    max_tokens_tech: int
    max_tokens_logic: int
    temperature: float
    runs: int
    retries: int
    cooldown: float
    settle: float
    prefill_words: int
    model_filter: str | None
    goldilocks_mode: str
    goldilocks_max_vram_pct: float
    output_dir: Path
    write_markdown: bool
    tech_prompt: str
    logic_prompt: str

    @property
    def chat_url(self): return f"{self.base_url}/v1/chat/completions"
    @property
    def models_url(self): return f"{self.base_url}/v1/models"


def load_config() -> Config:
    _load_dotenv()
    return Config(
        base_url=(_env("LLAMA_SWAP_URL", "http://localhost:8080")).rstrip("/"),
        api_key=_env("LLAMA_SWAP_API_KEY"),
        vram_limit_gb=_env("VRAM_LIMIT_GB", 16.0, float),
        gpu_index=_env("GPU_INDEX"),
        gpu_poll_ms=max(100, _env("GPU_POLL_MS", 200, int)),
        rapl_path=_env("RAPL_PATH", "/sys/class/powercap"),
        ha_url=(_env("HA_URL") or "").rstrip("/") or None,
        ha_token=_env("HA_TOKEN"),
        ha_cpu_entity=_env("HA_CPU_POWER_ENTITY"),
        ha_interval=_env("HA_POLL_INTERVAL", 1.0, float),
        timeout_tech=_env("TIMEOUT_TECH", 60.0, float),
        timeout_logic=_env("TIMEOUT_LOGIC", 300.0, float),
        timeout_load=_env("TIMEOUT_LOAD", 300.0, float),
        max_tokens_tech=_env("MAX_TOKENS_TECH", 300, int),
        max_tokens_logic=_env("MAX_TOKENS_LOGIC", 500, int),
        temperature=_env("TEMPERATURE", 0.2, float),
        runs=max(1, _env("RUNS_PER_PROMPT", 3, int)),
        retries=_env("RETRIES", 2, int),
        cooldown=_env("COOLDOWN_SECONDS", 5.0, float),
        settle=_env("SETTLE_SECONDS", 5.0, float),
        prefill_words=_env("PREFILL_WORDS", 1200, int),
        model_filter=_env("MODEL_FILTER"),
        goldilocks_mode=_env("GOLDILOCKS_MODE", "linear").lower(),
        goldilocks_max_vram_pct=_env("GOLDILOCKS_MAX_VRAM_PCT", 100.0, float),
        output_dir=Path(_env("OUTPUT_DIR", Path(__file__).parent / "results")),
        write_markdown=_env("WRITE_MARKDOWN", True, bool),
        tech_prompt=_env("TECH_PROMPT", "Explain quantum computing in 3 sentences."),
        logic_prompt=_env("LOGIC_PROMPT", "Outline a 101 Biology textbook. Give 2 experiments and a 3-question quiz."),
    )


class BenchError(Exception):
    """A recoverable per-model failure (timeout, HTTP error, malformed response...)."""


# --------------------------------------------------------------------------- #
# Model metadata (parsed from the filename)
# --------------------------------------------------------------------------- #

_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[Bb](?![A-Za-z])")
_QUANT_RE = re.compile(
    r"(?<![A-Za-z])(IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32)(?![A-Za-z0-9])", re.I)


def parse_params(name):
    """Total parameters in billions; 'A3B' (active params) is skipped by the lookbehind."""
    m = _PARAMS_RE.search(name)
    return float(m.group(1)) if m else None


def parse_quant(name):
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else "unknown"


# --------------------------------------------------------------------------- #
# Telemetry (all local to this machine)
# --------------------------------------------------------------------------- #

def _avg(xs): return sum(xs) / len(xs) if xs else None
def _max(xs): return max(xs) if xs else None
def _col(rows, key): return [r[key] for r in rows if r.get(key) is not None]


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


class GpuStream:
    """One persistent `nvidia-smi -lms N` process; a reader thread timestamps every line.
    Much cheaper and more accurate than spawning nvidia-smi for every sample. Multiple GPUs
    are aggregated (power/VRAM summed, temp/util = max) unless GPU_INDEX selects one."""

    QUERY = "index,power.draw,memory.used,memory.total,temperature.gpu,utilization.gpu"

    def __init__(self, cfg):
        self.cfg, self.samples, self.latest = cfg, [], {}
        self.lock, self.proc = threading.Lock(), None
        self.ok = False

    def start(self) -> bool:
        cmd = ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
               "-lms", str(self.cfg.gpu_poll_ms)]
        if self.cfg.gpu_index:
            cmd += ["-i", self.cfg.gpu_index]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1)
        except OSError:
            return False
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.time() + 6
        while time.time() < deadline:
            if self.samples:
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(float(parts[0]))
                w, used, total, temp, util = (float(x) for x in parts[1:6])
            except ValueError:            # "[N/A]" fields
                continue
            self.latest[idx] = (w, used, total, temp, util)
            v = list(self.latest.values())
            agg = {"gpu_w": sum(x[0] for x in v), "vram_gb": sum(x[1] for x in v) / 1024,
                   "vram_total_gb": sum(x[2] for x in v) / 1024,
                   "gpu_temp": max(x[3] for x in v), "gpu_util": max(x[4] for x in v)}
            with self.lock:
                self.samples.append((time.time(), agg))

    def window(self, t0, t1):
        with self.lock:
            got = [s for t, s in self.samples if t0 <= t <= t1]
            if not got and self.samples:
                got = [self.samples[-1][1]]
        return got

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class HostMeter:
    """CPU package power (RAPL energy counters), CPU utilisation (/proc/stat), RAM (/proc/meminfo).
    RAPL energy_uj is often root-only; run as root (e.g. the Unraid terminal) or relax permissions."""

    def __init__(self, cfg):
        self.domains = []
        for d in sorted(Path(cfg.rapl_path).glob("intel-rapl:*")):
            if re.fullmatch(r"intel-rapl:\d+", d.name):     # package domains only, no sub-zones
                try:
                    int((d / "energy_uj").read_text())
                    self.domains.append((d / "energy_uj", int((d / "max_energy_range_uj").read_text())))
                except (OSError, ValueError):
                    pass

    @property
    def rapl_ok(self): return bool(self.domains)

    @staticmethod
    def _stat():
        try:
            f = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
            return sum(f), f[3] + f[4]
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _mem_gb():
        try:
            m = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
            kb = lambda k: float(m[k].split()[0])
            return (kb("MemTotal") - kb("MemAvailable")) / 1024 ** 2
        except (OSError, ValueError, KeyError):
            return None

    def snap(self):
        energy = []
        for path, mx in self.domains:
            try:
                energy.append((int(path.read_text()), mx))
            except (OSError, ValueError):
                energy.append(None)
        return {"t": time.time(), "energy": energy, "stat": self._stat()}

    def diff(self, a, b) -> dict:
        out, dt = {}, b["t"] - a["t"]
        if self.domains and dt > 0 and None not in a["energy"] + b["energy"]:
            uj = 0
            for (e0, mx), (e1, _) in zip(a["energy"], b["energy"]):
                uj += e1 - e0 + (mx if e1 < e0 else 0)      # counter wrap
            out["cpu_w"] = uj / 1e6 / dt
        if a["stat"] and b["stat"] and b["stat"][0] > a["stat"][0]:
            out["cpu_util"] = 100 * (1 - (b["stat"][1] - a["stat"][1]) / (b["stat"][0] - a["stat"][0]))
        out["sys_mem_gb"] = self._mem_gb()
        return out


class HaCpuPoller(threading.Thread):
    """Optional CPU power from a Home Assistant sensor (used only when RAPL is unavailable).
    HA sensors update slowly, so values lag behind short requests."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg, self.samples, self._halt = cfg, [], threading.Event()
        self.headers = {"Authorization": f"Bearer {cfg.ha_token}"}

    def run(self):
        url = f"{self.cfg.ha_url}/api/states/{self.cfg.ha_cpu_entity}"
        while not self._halt.is_set():
            try:
                r = requests.get(url, headers=self.headers, timeout=3)
                if r.ok:
                    self.samples.append((time.time(), float(r.json().get("state"))))
            except (requests.RequestException, ValueError, TypeError):
                pass
            self._halt.wait(self.cfg.ha_interval)

    def window_avg(self, t0, t1):
        xs = [w for t, w in self.samples if t0 - self.cfg.ha_interval * 2 <= t <= t1]
        return _avg(xs) if xs else (self.samples[-1][1] if self.samples else None)

    def stop(self): self._halt.set()


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #

def post_chat(cfg, session, payload, timeout):
    for attempt in range(cfg.retries + 1):
        try:
            resp = session.post(cfg.chat_url, json=payload, timeout=timeout)
        except requests.Timeout:
            raise BenchError(f"timeout after {timeout:.0f}s")
        except requests.RequestException as e:
            raise BenchError(f"connection error: {e}")
        if resp.status_code == 503 and attempt < cfg.retries:
            time.sleep(2)
            continue
        if resp.status_code == 404:
            raise BenchError("model not found (404)")
        if resp.status_code >= 400:
            raise BenchError(f"HTTP {resp.status_code}: {resp.text[:160]}")
        try:
            return resp.json()
        except ValueError:
            raise BenchError("response was not valid JSON")
    raise BenchError("server busy (503)")


def unload_all(cfg, session) -> bool:
    for path in ("/api/models/unload", "/unload"):
        try:
            if session.post(cfg.base_url + path, timeout=30).status_code < 400:
                return True
        except requests.RequestException:
            pass
    return False


_VOCAB = ("river stone copper window engine garden signal marble orbit harvest paper lantern thunder "
          "velvet compass anchor meadow circuit whisper granite ember canvas harbor quartz timber "
          "mirror falcon cobalt prairie summit voyage lattice monsoon plume fabric glacier beacon").split()


def prefill_prompt(words):
    """Long, unique-per-call prompt so no prefix cache can inflate prompt-processing speed."""
    rng = random.Random()
    body = " ".join(rng.choice(_VOCAB) for _ in range(words))
    return f"[{rng.getrandbits(32):08x}] Summarise the theme of this text in one short sentence.\n\n{body}"


def measure(cfg, session, gpu, host, ha, model, prompt, max_tokens, timeout) -> dict:
    """One timed generation plus the telemetry captured during exactly that window."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": cfg.temperature, "stream": False,
               "cache_prompt": False}
    a = host.snap()
    t0 = time.time()
    try:
        data = post_chat(cfg, session, payload, timeout)
    finally:
        t1 = time.time()
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
    h = host.diff(a, b)
    cpu_w = h.get("cpu_w")
    if cpu_w is None and ha:
        cpu_w = ha.window_avg(t0, t1)
    gpu_w = _avg(_col(g, "gpu_w"))
    energy = ((gpu_w or 0) + (cpu_w or 0)) * wall if (gpu_w is not None or cpu_w is not None) else None
    return {"tps": float(tps), "prompt_tps": timings.get("prompt_per_second"),
            "ttft_ms": timings.get("prompt_ms"), "gen_ms": timings.get("predicted_ms"),
            "wall_s": wall, "tokens": tokens, "energy_j": energy,
            "gpu_w": gpu_w, "gpu_peak_w": _max(_col(g, "gpu_w")), "gpu_util": _avg(_col(g, "gpu_util")),
            "gpu_temp": _max(_col(g, "gpu_temp")), "vram_gb": _max(_col(g, "vram_gb")),
            "vram_total_gb": _max(_col(g, "vram_total_gb")),
            "cpu_w": cpu_w, "cpu_util": h.get("cpu_util"), "sys_mem_gb": h.get("sys_mem_gb")}


# --------------------------------------------------------------------------- #
# Per-model benchmark and post-processing
# --------------------------------------------------------------------------- #

FIELDS = ["timestamp", "model", "status", "error", "params_b", "quant", "load_s", "tech_tps", "logic_tps",
          "avg_tps", "prefill_tps", "ttft_ms", "gen_ms", "latency_s", "gpu_w", "gpu_peak_w", "idle_gpu_w",
          "gpu_util", "gpu_temp", "vram_gb", "vram_delta_gb", "vram_total_gb", "vram_pct", "cpu_w",
          "cpu_util", "sys_mem_gb", "energy_wh", "j_per_token", "tps_per_watt", "goldilocks", "eligible"]


def blank(name):
    r = dict.fromkeys(FIELDS)
    r.update(model=name, status="ok", error="", eligible=False,
             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return r


def bench_model(cfg, session, gpu, host, ha, model, unload_state) -> dict:
    r = blank(model)
    try:
        # 1. clean slate + idle baseline, so VRAM / power can be attributed to this model
        unloaded = unload_all(cfg, session)
        if not unloaded and not unload_state["warned"]:
            unload_state["warned"] = True
            print("  [warn] could not unload models via the API; load time and VRAM delta will be skipped")
        time.sleep(cfg.settle)
        now = time.time()
        base = gpu.window(now - max(0.5, min(3.0, cfg.settle)), now) if gpu.ok else []
        r["idle_gpu_w"] = _avg(_col(base, "gpu_w"))
        base_vram = _avg(_col(base, "vram_gb"))

        # 2. cold load (also the warm-up)
        t = time.time()
        post_chat(cfg, session, {"model": model, "max_tokens": 8, "stream": False,
                                 "messages": [{"role": "user", "content": "hi"}]}, cfg.timeout_load)
        r["load_s"] = (time.time() - t) if unloaded else None

        # 3. timed tests
        spec = {"tech": (cfg.tech_prompt, cfg.max_tokens_tech, cfg.timeout_tech, False),
                "logic": (cfg.logic_prompt, cfg.max_tokens_logic, cfg.timeout_logic, False)}
        if cfg.prefill_words > 0:
            spec["prefill"] = (None, 16, cfg.timeout_logic, True)
        runs = {}
        for key, (prompt, mt, to, fresh) in spec.items():
            runs[key] = [measure(cfg, session, gpu, host, ha, model,
                                 prefill_prompt(cfg.prefill_words) if fresh else prompt, mt, to)
                         for _ in range(cfg.runs)]
    except BenchError as e:
        r.update(status="error", error=str(e))
        return r

    med = lambda k, f: _median([x[f] for x in runs[k]])
    gen = runs["tech"] + runs["logic"]
    every = gen + runs.get("prefill", [])
    r["tech_tps"], r["logic_tps"] = med("tech", "tps"), med("logic", "tps")
    if "prefill" in runs:
        r["prefill_tps"] = med("prefill", "prompt_tps")
    r["ttft_ms"] = _avg([v for v in (med("tech", "ttft_ms"), med("logic", "ttft_ms")) if v is not None])
    r["gen_ms"] = _avg([v for v in (med("tech", "gen_ms"), med("logic", "gen_ms")) if v is not None])
    r["latency_s"] = _avg([med("tech", "wall_s"), med("logic", "wall_s")])
    r["gpu_w"], r["gpu_util"] = _avg(_col(gen, "gpu_w")), _avg(_col(gen, "gpu_util"))
    r["cpu_w"], r["cpu_util"] = _avg(_col(gen, "cpu_w")), _avg(_col(gen, "cpu_util"))
    r["gpu_peak_w"], r["gpu_temp"] = _max(_col(every, "gpu_peak_w")), _max(_col(every, "gpu_temp"))
    r["vram_gb"], r["vram_total_gb"] = _max(_col(every, "vram_gb")), _max(_col(every, "vram_total_gb"))
    r["sys_mem_gb"] = _max(_col(every, "sys_mem_gb"))
    if r["vram_gb"] is not None and base_vram is not None and unloaded:
        r["vram_delta_gb"] = max(0.0, r["vram_gb"] - base_vram)
    e_tech, e_logic = _avg(_col(runs["tech"], "energy_j")), _avg(_col(runs["logic"], "energy_j"))
    if e_tech is not None and e_logic is not None:
        r["energy_wh"] = (e_tech + e_logic) / 3600
        tokens = sum(x["tokens"] for x in gen)
        r["j_per_token"] = sum(_col(gen, "energy_j")) / tokens if tokens else None
    return r


def finalize(results, cfg):
    """Derive params/quant/VRAM%/efficiency and the Goldilocks score."""
    for r in results:
        r["params_b"] = r["params_b"] or parse_params(r["model"])
        r["quant"] = r["quant"] or parse_quant(r["model"])
        if r["status"] != "ok":
            continue
        r["avg_tps"] = _avg([x for x in (r["tech_tps"], r["logic_tps"]) if x is not None])
        total = r["vram_total_gb"] or cfg.vram_limit_gb
        r["vram_total_gb"] = total
        if r["vram_gb"] is not None and total:
            r["vram_pct"] = r["vram_gb"] / total * 100
        watts = (r["gpu_w"] or 0) + (r["cpu_w"] or 0)
        if watts and r["avg_tps"]:
            r["tps_per_watt"] = r["avg_tps"] / watts
        if r["params_b"] and r["avg_tps"]:
            p = math.log(r["params_b"]) if cfg.goldilocks_mode == "log" else r["params_b"]
            r["goldilocks"] = p * r["avg_tps"]
            r["eligible"] = r["vram_pct"] is None or r["vram_pct"] <= cfg.goldilocks_max_vram_pct
    return results


def _rounded(r): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def write_csv(results, path, append=False):
    new = not (append and Path(path).exists())
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(_rounded(r) for r in results)


def discover_models(cfg, session):
    try:
        r = session.get(cfg.models_url, timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError) as e:
        sys.exit(f"[error] Could not list models from {cfg.models_url}: {e}")
    if cfg.model_filter:
        ids = [m for m in ids if re.search(cfg.model_filter, m)]
    if not ids:
        sys.exit("[error] No models found (check LLAMA_SWAP_URL / MODEL_FILTER).")
    return ids


# --------------------------------------------------------------------------- #
# Demo data and --check
# --------------------------------------------------------------------------- #

def demo_results(cfg):
    specs = [("Demo-9B-Q4_K_M.gguf", 110, 6.5, 120, 25), ("Demo-27B-IQ3_XXS.gguf", 42, 12.8, 260, 40),
             ("Demo-35B-A3B-IQ3_XXS.gguf", 88, 15.1, 190, 85), ("Demo-8B-Q8_0.gguf", 95, 9.9, 170, 20),
             ("Demo-30B-A3B-Q4_0.gguf", 27, 15.6, 210, 70), ("Demo-26B-A4B-Q4_K_XL.gguf", 64, 14.2, 200, 55)]
    out = []
    for name, tps, vram, gw, cw in specs:
        r = blank(name)
        lat = 300 / tps + 1
        r.update(load_s=vram / 2.5, tech_tps=tps * 1.04, logic_tps=tps * 0.96, prefill_tps=tps * 14,
                 ttft_ms=60 + 15000 / tps, gen_ms=300 / tps * 1000, latency_s=lat, gpu_w=gw,
                 gpu_peak_w=gw * 1.15, idle_gpu_w=12, gpu_util=88, gpu_temp=62 + gw / 20, vram_gb=vram,
                 vram_delta_gb=vram - 0.6, vram_total_gb=cfg.vram_limit_gb, cpu_w=cw, cpu_util=25,
                 sys_mem_gb=18, energy_wh=(gw + cw) * lat * 2 / 3600, j_per_token=(gw + cw) / tps)
        out.append(r)
    return out


def check(cfg):
    """Verify every telemetry source for ~3 seconds. Run this before a long benchmark."""
    print(f"llama-swap: {cfg.base_url}")
    try:
        n = len(requests.get(cfg.models_url, timeout=5).json().get("data", []))
        print(f"  OK, {n} model(s) listed")
    except (requests.RequestException, ValueError) as e:
        print(f"  FAILED: {e}")
    gpu = GpuStream(cfg)
    ok = gpu.start()
    time.sleep(2)
    if ok:
        s = gpu.window(time.time() - 2, time.time())
        print(f"GPU (nvidia-smi stream): OK, {len(s)} samples in 2s; latest = "
              + ", ".join(f"{k}={v:.1f}" for k, v in s[-1].items()))
    else:
        print("GPU (nvidia-smi stream): NOT AVAILABLE - is nvidia-smi on PATH and the driver loaded?")
    gpu.stop()
    host = HostMeter(cfg)
    a = host.snap(); time.sleep(2); h = host.diff(a, host.snap())
    print(f"CPU power via RAPL: {'OK, %.1f W' % h['cpu_w'] if 'cpu_w' in h else 'NOT AVAILABLE'}"
          + ("" if host.rapl_ok else f" (no readable domains under {cfg.rapl_path}; run as root, or set HA_* as a fallback)"))
    print(f"CPU util: {h.get('cpu_util', float('nan')):.0f}%   RAM used: {h.get('sys_mem_gb') or float('nan'):.1f} GB")
    if cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity:
        print("Home Assistant CPU-power fallback: configured")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def _emoji(t): return "🟣" if t > 100 else "🔵" if t > 80 else "🟢" if t > 50 else "🟡" if t > 30 else "🩷"
def _f(v, d=1): return "–" if v is None else f"{v:.{d}f}"


def render_markdown(results, cfg, path):
    ok = [r for r in results if r["status"] == "ok" and r["avg_tps"] is not None]
    top = sorted((r for r in ok if r["eligible"]), key=lambda r: -r["goldilocks"])[:3]
    L = ["---", "tags: [llm/benchmark, llama-swap]", f"date: {datetime.now():%Y-%m-%d}", "---", "",
         "# llama-swap Benchmark Report", "", f"> {len(results)} models · {datetime.now():%Y-%m-%d %H:%M} · "
         f"VRAM limit {cfg.vram_limit_gb:g} GB", "", "## Top 3 — Goldilocks", "",
         "| Rank | Model | TPS | VRAM | GPU W | Score |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(top, 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} | {_f(r['vram_gb'])} GB "
                 f"| {_f(r['gpu_w'], 0)} | {_f(r['goldilocks'], 0)} |")
    L += ["", "## Speed Overview", "", "| Rank | Model | Avg TPS |", "|---|---|---|"]
    for i, r in enumerate(sorted(ok, key=lambda r: -r["avg_tps"]), 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")
    L += ["", "## Full Model Comparison", "",
          "| Model | Params | Quant | Load s | VRAM | GPU W | CPU W | Wh | J/tok | TTFT ms | Prefill t/s | Tech | Logic | Avg |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] != "ok":
            L.append(f"| {r['model']} | | | | | | | | | | | | | ERR: {r['error']} |")
            continue
        p = r["vram_pct"]
        vram = "–" if p is None else f"{r['vram_gb']:.1f} GB [{'=' * round(min(p, 100) / 10):<10}] {p:.0f}%"
        L.append(f"| {r['model']} | {_f(r['params_b'], 0)}B | {r['quant']} | {_f(r['load_s'])} | {vram} | "
                 f"{_f(r['gpu_w'], 0)} | {_f(r['cpu_w'], 0)} | {_f(r['energy_wh'], 3)} | {_f(r['j_per_token'])} | "
                 f"{_f(r['ttft_ms'], 0)} | {_f(r['prefill_tps'], 0)} | {_f(r['tech_tps'])} | {_f(r['logic_tps'])} | "
                 f"{_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")
    groups = {}
    for r in ok:
        groups.setdefault(r["quant"], []).append(r)
    L += ["", "## Quantization Analysis", "", "| Quant | Models | Avg TPS |", "|---|---|---|"]
    for q, rs in sorted(groups.items(), key=lambda kv: -_avg([r["avg_tps"] for r in kv[1]])):
        L.append(f"| {q} | {len(rs)} | {_f(_avg([r['avg_tps'] for r in rs]))} |")
    L += ["", "## Methodology", "",
          f"- Median of {cfg.runs} run(s) per prompt; prompt cache disabled; each model is unloaded first, "
          "idle baseline sampled, then cold-loaded.",
          f"- GPU: `nvidia-smi` stream every {cfg.gpu_poll_ms} ms. CPU power: RAPL (or Home Assistant). "
          "Energy = (mean GPU W + mean CPU W) × request time.",
          f"- Goldilocks = {'ln(params B)' if cfg.goldilocks_mode == 'log' else 'params B'} × avg TPS (total params)."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def render_dashboard(results, cfg, path, cpu_source="none"):
    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "count": len(results),
            "vram_limit": cfg.vram_limit_gb, "runs": cfg.runs, "goldilocks_mode": cfg.goldilocks_mode,
            "timeouts": [cfg.timeout_tech, cfg.timeout_logic], "max_tokens": [cfg.max_tokens_tech, cfg.max_tokens_logic],
            "prompts": [cfg.tech_prompt, cfg.logic_prompt], "max_vram_pct": cfg.goldilocks_max_vram_pct,
            "poll_ms": cfg.gpu_poll_ms, "cpu_source": cpu_source, "prefill_words": cfg.prefill_words}
    blob = json.dumps({"meta": meta, "models": results}).replace("</", "<\\/")
    Path(path).write_text(HTML_TEMPLATE.replace("__DATA__", blob), encoding="utf-8")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-swap Benchmark Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#e6edf3;--mute:#8b949e}
*{box-sizing:border-box}
body{margin:0 auto;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px;max-width:1500px}
h1{margin:0 0 4px}h2{margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mute)}
.grid{display:grid;gap:16px}.podium{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.charts{grid-template-columns:repeat(auto-fit,minmax(520px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow:hidden}
.card h3{margin:0 0 8px;font-size:15px}
.rank{font-size:12px;font-weight:700;letter-spacing:.08em}
.gold{border-color:#d4a72c}.gold .rank{color:#d4a72c}.silver{border-color:#9aa4b0}.silver .rank{color:#9aa4b0}
.bronze{border-color:#bf7a3a}.bronze .rank{color:#bf7a3a}
.pname{font-weight:600;word-break:break-all;margin:4px 0 8px}.stat{color:var(--mute)}.stat b{color:var(--text)}
.chartbox{position:relative;height:380px}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th{cursor:pointer;color:var(--mute);user-select:none;position:sticky;top:0;background:var(--card)}
th:first-child,td:first-child{text-align:left}th:hover{color:var(--text)}
.bar{display:inline-block;width:90px;height:8px;background:#21262d;border-radius:4px;vertical-align:middle;margin:0 6px;overflow:hidden}
.bar i{display:block;height:100%}.err{color:#f85149}
@media(max-width:600px){body{padding:12px}.charts{grid-template-columns:1fr}}
</style></head><body>
<h1>llama-swap Benchmark</h1>
<div class="sub" id="hdr"></div>

<h2>Top 3 &mdash; Goldilocks</h2><div class="grid podium" id="podium"></div>

<h2>Charts</h2>
<div class="grid charts">
 <div class="card"><h3>Avg generation speed (tok/s)</h3><div class="chartbox"><canvas id="cTps"></canvas></div></div>
 <div class="card"><h3>VRAM vs speed (bubble = params)</h3><div class="chartbox"><canvas id="cVram"></canvas></div></div>
 <div class="card"><h3>Power efficiency (tok/s per W, GPU + CPU)</h3><div class="chartbox"><canvas id="cEff"></canvas></div></div>
 <div class="card"><h3>Energy per token (J, lower is better)</h3><div class="chartbox"><canvas id="cJ"></canvas></div></div>
 <div class="card"><h3>Avg TPS by quantization</h3><div class="chartbox"><canvas id="cQuant"></canvas></div></div>
</div>

<h2>Full model comparison</h2>
<div class="card tablewrap"><table id="tbl"></table></div>

<h2>Quantization analysis</h2>
<div class="card tablewrap"><table id="qtbl"></table></div>

<h2>Methodology</h2><div class="card" id="meth"></div>

<script>
const D = __DATA__;
const M = D.models, OK = M.filter(m => m.status === 'ok' && m.avg_tps != null);

// Chart.js cannot resolve CSS var(): every chart colour below is a literal hex value.
function tpsColor(t){return t>100?'#a371f7':t>80?'#58a6ff':t>50?'#3fb950':t>30?'#d29922':'#f778ba';}
function vramColor(p){return p>90?'#f85149':p>70?'#db6d28':'#3fb950';}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v,d=1) => v==null ? '&ndash;' : Number(v).toFixed(d);

Chart.defaults.color = '#e6edf3';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.maintainAspectRatio = false;
const valueLabels = {id:'valueLabels', afterDatasetsDraw(chart){
  const c = chart.ctx, horiz = chart.options.indexAxis === 'y';
  c.save(); c.fillStyle = '#e6edf3'; c.font = '12px sans-serif'; c.textBaseline = 'middle';
  c.textAlign = horiz ? 'left' : 'center';
  chart.getDatasetMeta(0).data.forEach((bar,i)=>{
    const v = chart.data.datasets[0].data[i]; if (v==null) return;
    c.fillText(v.toFixed(1), horiz ? bar.x+6 : bar.x, horiz ? bar.y : bar.y-10);
  }); c.restore();
}};
function barChart(id, labels, data, colors, horiz, xTitle){
  new Chart(document.getElementById(id), {type:'bar', plugins:[valueLabels],
    data:{labels, datasets:[{data, backgroundColor:colors, borderColor:colors, borderWidth:1}]},
    options:{indexAxis: horiz?'y':'x', plugins:{legend:{display:false}},
      scales:{x:{grace: horiz?'12%':0, title:{display:!!xTitle, text:xTitle||'', color:'#e6edf3'}}, y:{grace: horiz?0:'12%', beginAtZero:true}}}});
}
const none = id => document.getElementById(id).parentElement.innerHTML = '<p class="sub">No data for this chart (telemetry unavailable).</p>';

document.getElementById('hdr').textContent =
  `${D.meta.count} models \u00b7 ${D.meta.generated} \u00b7 llama.cpp via llama-swap \u00b7 VRAM limit ${D.meta.vram_limit} GB`;

// podium
const podium = OK.filter(m=>m.eligible).sort((a,b)=>b.goldilocks-a.goldilocks).slice(0,3);
const cls = ['gold','silver','bronze'];
document.getElementById('podium').innerHTML = podium.length ? podium.map((m,i)=>`
  <div class="card ${cls[i]}"><div class="rank">#${i+1} GOLDILOCKS</div>
  <div class="pname">${esc(m.model)}</div>
  <div class="stat"><b style="color:${tpsColor(m.avg_tps)}">${fmt(m.avg_tps)}</b> tok/s &middot; ${m.params_b}B params</div>
  <div class="stat">VRAM <b>${fmt(m.vram_gb)} GB</b> (${fmt(m.vram_pct,0)}%) &middot; GPU <b>${fmt(m.gpu_w,0)} W</b> &middot; CPU <b>${fmt(m.cpu_w,0)} W</b></div>
  <div class="stat">Score <b>${fmt(m.goldilocks,0)}</b> &middot; <b>${fmt(m.j_per_token)}</b> J/token</div></div>`).join('')
  : '<div class="card sub">No eligible models (need parsable params and a speed result).</div>';

// charts
const bySpeed = [...OK].sort((a,b)=>b.avg_tps-a.avg_tps);
barChart('cTps', bySpeed.map(m=>m.model), bySpeed.map(m=>m.avg_tps), bySpeed.map(m=>tpsColor(m.avg_tps)), true, 'tokens/sec');

const vp = OK.filter(m=>m.vram_gb!=null);
if (vp.length){
  const ps = vp.map(m=>m.params_b).filter(p=>p!=null), pmin=Math.min(...ps), pmax=Math.max(...ps);
  const rad = p => p==null ? 8 : (pmax===pmin ? 15 : 5 + (p-pmin)/(pmax-pmin)*20);
  new Chart(document.getElementById('cVram'), {type:'bubble',
    data:{datasets:[{data:vp.map(m=>({x:m.vram_gb,y:m.avg_tps,r:rad(m.params_b),name:m.model,p:m.params_b})),
      backgroundColor:vp.map(m=>tpsColor(m.avg_tps)+'b3'), borderColor:vp.map(m=>tpsColor(m.avg_tps))}]},
    options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.raw.name}: ${c.raw.y.toFixed(1)} tok/s, ${c.raw.x.toFixed(1)} GB, ${c.raw.p??'?'}B`}}},
      scales:{x:{min:Math.floor(Math.min(...vp.map(m=>m.vram_gb))-1), max:Math.ceil(Math.max(...vp.map(m=>m.vram_gb))+1), title:{display:true,text:'VRAM used (GB)',color:'#e6edf3'}},
              y:{min:0, max:Math.ceil(Math.max(...vp.map(m=>m.avg_tps))+20), title:{display:true,text:'Avg tok/s',color:'#e6edf3'}}}}});
} else none('cVram');

const ef = OK.filter(m=>m.tps_per_watt!=null).sort((a,b)=>b.tps_per_watt-a.tps_per_watt);
if (ef.length) barChart('cEff', ef.map(m=>m.model), ef.map(m=>m.tps_per_watt), ef.map(m=>tpsColor(m.avg_tps)), false); else none('cEff');

const jt = OK.filter(m=>m.j_per_token!=null).sort((a,b)=>a.j_per_token-b.j_per_token);
if (jt.length) barChart('cJ', jt.map(m=>m.model), jt.map(m=>m.j_per_token), jt.map(m=>tpsColor(m.avg_tps)), true, 'joules per token'); else none('cJ');

const groups = {}; OK.forEach(m=>(groups[m.quant] = groups[m.quant]||[]).push(m));
const qs = Object.entries(groups).map(([q,ms])=>({q, n:ms.length, avg:ms.reduce((s,m)=>s+m.avg_tps,0)/ms.length, names:ms.map(m=>m.model)}))
  .sort((a,b)=>b.avg-a.avg);
barChart('cQuant', qs.map(x=>x.q), qs.map(x=>x.avg), qs.map(x=>tpsColor(x.avg)), false);
document.getElementById('qtbl').innerHTML = '<tr><th>Quant</th><th>Models</th><th>Avg TPS</th><th style="text-align:left">Members</th></tr>' +
  qs.map(x=>`<tr><td>${esc(x.q)}</td><td>${x.n}</td><td style="color:${tpsColor(x.avg)}">${fmt(x.avg)}</td><td style="text-align:left;white-space:normal">${x.names.map(esc).join(', ')}</td></tr>`).join('');

// sortable comparison table
const COLS = [['model','Model'],['params_b','Params'],['quant','Quant'],['load_s','Load s'],['vram_gb','VRAM'],['gpu_w','GPU W'],
  ['gpu_peak_w','Peak W'],['cpu_w','CPU W'],['gpu_util','GPU %'],['gpu_temp','Temp \u00b0C'],['ttft_ms','TTFT ms'],['latency_s','Latency s'],
  ['energy_wh','Energy Wh'],['j_per_token','J/token'],['prefill_tps','Prefill t/s'],['tech_tps','Tech TPS'],['logic_tps','Logic TPS'],
  ['avg_tps','Avg TPS'],['goldilocks','Goldilocks']];
const DEC = {energy_wh:3, goldilocks:0, ttft_ms:0, prefill_tps:0, gpu_w:0, gpu_peak_w:0, cpu_w:0, gpu_util:0};
let sk='avg_tps', sd=-1;
function cell(k,m){
  const v=m[k];
  if(k==='model') return `<td>${esc(v)}${m.status!=='ok'?` <span class="err">ERR: ${esc(m.error)}</span>`:''}</td>`;
  if(k==='quant') return `<td>${esc(v)}</td>`;
  if(k==='params_b') return `<td>${v==null?'&ndash;':v+'B'}</td>`;
  if(k==='vram_gb'){ if(v==null) return '<td>&ndash;</td>'; const p=Math.min(m.vram_pct??0,100);
    return `<td>${v.toFixed(1)} GB<span class="bar"><i style="width:${p}%;background:${vramColor(p)}"></i></span>${p.toFixed(0)}%</td>`; }
  if(['tech_tps','logic_tps','avg_tps'].includes(k)) return `<td style="color:${v==null?'inherit':tpsColor(v)}">${fmt(v)}</td>`;
  return `<td>${fmt(v, DEC[k]??1)}</td>`;
}
function renderTable(){
  const rows=[...M].sort((a,b)=>{const x=a[sk],y=b[sk]; if(x==null) return 1; if(y==null) return -1;
    return (typeof x==='string'? x.localeCompare(y) : x-y)*sd;});
  document.getElementById('tbl').innerHTML = '<tr>'+COLS.map(([k,l])=>`<th data-k="${k}">${l}${sk===k?(sd<0?' \u25BC':' \u25B2'):''}</th>`).join('')+'</tr>'
    + rows.map(m=>'<tr>'+COLS.map(([k])=>cell(k,m)).join('')+'</tr>').join('');
  document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{const k=th.dataset.k; sd = (sk===k)? -sd : (k==='model'||k==='quant'?1:-1); sk=k; renderTable();});
}
renderTable();

document.getElementById('meth').innerHTML = `<ul>
 <li><b>Backend:</b> llama-swap (OpenAI-compatible <code>/v1/chat/completions</code>), non-streaming, prompt cache disabled.</li>
 <li><b>Per model:</b> all models unloaded &rarr; idle baseline sampled &rarr; cold load (Load s) &rarr; Technical, Logic${D.meta.prefill_words?` and prefill (~${D.meta.prefill_words}-word prompt)`:''} tests, median of ${D.meta.runs} run(s).</li>
 <li><b>Prompts:</b> Technical &mdash; &ldquo;${esc(D.meta.prompts[0])}&rdquo;; Logic &mdash; &ldquo;${esc(D.meta.prompts[1])}&rdquo;.</li>
 <li><b>Speed:</b> <code>timings.predicted_per_second</code>; falls back to completion_tokens / wall time. <b>TTFT</b> is approximated by <code>timings.prompt_ms</code>; <b>Prefill t/s</b> is <code>timings.prompt_per_second</code>.</li>
 <li><b>GPU telemetry:</b> one persistent <code>nvidia-smi</code> stream every ${D.meta.poll_ms} ms, windowed to each request. Power = mean, peak/temp/VRAM = maximum. <b>CPU:</b> ${esc(D.meta.cpu_source)}.</li>
 <li><b>Energy:</b> (mean GPU W + mean CPU W) &times; request time. <b>J/token:</b> energy &divide; generated tokens. <b>Efficiency:</b> avg TPS &divide; (GPU W + CPU W).</li>
 <li><b>Goldilocks:</b> ${D.meta.goldilocks_mode==='log'?'ln(params&nbsp;B)':'params&nbsp;B'} &times; avg TPS, VRAM &le; ${D.meta.max_vram_pct}% of ${D.meta.vram_limit} GB. Params are parsed from the filename and use <i>total</i> (not active) parameters.</li>
 <li><b>Limits:</b> max_tokens ${D.meta.max_tokens.join(' / ')}; timeouts ${D.meta.timeouts.join('s / ')}s. Context size and KV-cache type differ per model config and affect speed and VRAM.</li>
</ul>`;
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Benchmark every model on a llama-swap server (run on the GPU host).")
    ap.add_argument("--check", action="store_true", help="test telemetry sources and exit")
    ap.add_argument("--demo", action="store_true", help="build a dashboard from synthetic data (no server)")
    args = ap.parse_args()

    cfg = load_config()
    if args.check:
        return check(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results, cpu_source = [], "n/a"

    if args.demo:
        results, cpu_source = demo_results(cfg), "demo data"
    else:
        session = requests.Session()
        if cfg.api_key:
            session.headers["Authorization"] = f"Bearer {cfg.api_key}"
        models = discover_models(cfg, session)

        gpu = GpuStream(cfg)
        gpu.ok = gpu.start()
        host = HostMeter(cfg)
        ha = None
        if host.rapl_ok:
            cpu_source = "RAPL package power"
        elif cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity:
            ha = HaCpuPoller(cfg)
            ha.start()
            cpu_source = "Home Assistant sensor (polled, may lag)"
        else:
            cpu_source = "unavailable (no RAPL / Home Assistant)"
        print(f"Found {len(models)} model(s). GPU telemetry: {'nvidia-smi stream' if gpu.ok else 'UNAVAILABLE'}; "
              f"CPU power: {cpu_source}")
        if not gpu.ok:
            print("  [warn] no GPU data: run this on the GPU host with nvidia-smi available (try --check).")

        state = {"warned": False}
        try:
            for i, m in enumerate(models, 1):
                print(f"[{i}/{len(models)}] {m}")
                r = bench_model(cfg, session, gpu, host, ha, m, state)
                if r["status"] != "ok":
                    print("  ERROR: " + r["error"])
                else:
                    print(f"  tech {r['tech_tps']:.1f} / logic {r['logic_tps']:.1f} tok/s"
                          + (f" | GPU {r['gpu_w']:.0f} W, VRAM {r['vram_gb']:.1f} GB" if r["gpu_w"] and r["vram_gb"] else "")
                          + (f" | CPU {r['cpu_w']:.0f} W" if r["cpu_w"] else ""))
                results.append(r)
                if i < len(models):
                    time.sleep(cfg.cooldown)
        except KeyboardInterrupt:
            print("\nInterrupted: writing partial results.")
        finally:
            gpu.stop()
            if ha:
                ha.stop()

    if not results:
        sys.exit("No results to write.")
    finalize(results, cfg)
    ok = [r for r in results if r["status"] == "ok"]
    if not args.demo and ok and all((r["vram_gb"] or 0) < 0.5 for r in ok):
        print("\n[warn] VRAM looks ~0 for every model: GPU telemetry is missing or from the wrong GPU (see GPU_INDEX).")
    csv_path = cfg.output_dir / f"benchmark_{stamp}.csv"
    html_path = cfg.output_dir / "benchmark_dashboard.html"
    write_csv(results, csv_path)
    if not args.demo:
        write_csv(results, cfg.output_dir / "benchmark_history_v2.csv", append=True)
    render_dashboard(results, cfg, html_path, cpu_source)
    out = [f"CSV:       {csv_path}", f"Dashboard: {html_path}"]
    if cfg.write_markdown:
        md_path = cfg.output_dir / "benchmark_report.md"
        render_markdown(results, cfg, md_path)
        out.append(f"Markdown:  {md_path}")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
