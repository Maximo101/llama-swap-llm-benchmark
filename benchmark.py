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

# Everything (.env, results/) is resolved relative to this file, never the current directory,
# so it behaves the same from a cron job / Unraid User Script as from inside the folder.
SCRIPT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _load_dotenv():
    """python-dotenv if installed, otherwise a minimal built-in .env reader."""
    try:
        from dotenv import load_dotenv
        load_dotenv(SCRIPT_DIR / ".env")
        return
    except ImportError:
        pass
    p = SCRIPT_DIR / ".env"
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
    cache_test: bool
    rate_peak: float | None
    rate_offpeak: float | None
    currency: str
    hw_cpu: str | None
    hw_ram: str | None
    hw_gpu: str | None
    tech_prompt: str
    logic_prompt: str

    @property
    def chat_url(self): return f"{self.base_url}/v1/chat/completions"
    @property
    def models_url(self): return f"{self.base_url}/v1/models"


def _output_dir() -> Path:
    """OUTPUT_DIR (default: results/ next to the script). Relative paths are relative to the script."""
    out = Path(_env("OUTPUT_DIR", "results")).expanduser()
    return out if out.is_absolute() else SCRIPT_DIR / out


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
        output_dir=_output_dir(),
        write_markdown=_env("WRITE_MARKDOWN", True, bool),
        cache_test=_env("CACHE_TEST", True, bool),
        rate_peak=_env("ELECTRICITY_RATE_PEAK", None, float),
        rate_offpeak=_env("ELECTRICITY_RATE_OFFPEAK", None, float),
        currency=_env("CURRENCY_SYMBOL", "$"),
        hw_cpu=_env("HW_CPU"), hw_ram=_env("HW_RAM"), hw_gpu=_env("HW_GPU"),
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


_ACTIVE_RE = re.compile(r"(?<![A-Za-z0-9])A(\d+(?:\.\d+)?)B(?![A-Za-z])")


def parse_active(name):
    """Active parameters (billions) of a MoE, from the 'A3B' tag; None for dense models."""
    m = _ACTIVE_RE.search(name)
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


def detect_hardware(cfg) -> str:
    """'CPU · RAM · GPU' for the dashboard header. HW_CPU / HW_RAM / HW_GPU override auto-detection
    (auto-detect cannot know e.g. the RAM generation, so set HW_RAM='64 GB DDR5' if you want it shown)."""
    cpu = cfg.hw_cpu
    if not cpu:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        if cpu:
            cpu = re.sub(r"\((?:R|TM)\)", "", cpu)
            cpu = re.sub(r"\s+@.*$", "", cpu)
            cpu = re.sub(r"\bCPU\b", "", cpu)
            cpu = re.sub(r"\s+", " ", cpu).strip()
    ram = cfg.hw_ram
    if not ram:
        try:
            kb = next(float(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                      if l.startswith("MemTotal"))
            gb, step = kb / 1024 ** 2, None
            step = 8 if gb >= 24 else 4
            ram = f"{max(step, round(gb / step) * step)} GB RAM"
        except (OSError, StopIteration, ValueError):
            pass
    gpu = cfg.hw_gpu
    if not gpu:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            names = []
            for line in out.strip().splitlines():
                name, mib = [x.strip() for x in line.rsplit(",", 1)]
                names.append(f"{name.replace('NVIDIA ', '').replace('GeForce ', '')} {round(float(mib) / 1024)} GB")
            gpu = " + ".join(names) or None
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return " · ".join(x for x in (cpu, ram, gpu) if x)


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


def get_ctx(cfg, session, model):
    """Active context size (tokens) of the loaded model. Primary: llama-server's /props, reached
    through llama-swap's /upstream/<model>/ route. Fallback: a -c/--ctx-size flag in /running (if shown)."""
    from urllib.parse import quote
    try:
        r = session.get(f"{cfg.base_url}/upstream/{quote(model, safe='')}/props", timeout=10)
        if r.ok:
            j = r.json()
            n = (j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx")
            if n:
                return int(n)
    except (requests.RequestException, ValueError):
        pass
    try:
        r = session.get(cfg.base_url + "/running", timeout=5)
        if r.ok:
            j = r.json()
            for entry in (j.get("running", []) if isinstance(j, dict) else j):
                if isinstance(entry, dict) and entry.get("model") == model:
                    m = re.search(r"\s(?:-c|--ctx-size)[ =](\d+)", json.dumps(entry))
                    if m:
                        return int(m.group(1))
    except (requests.RequestException, ValueError):
        pass
    return None


def prefill_prompt(words):
    """Long, unique-per-call prompt so no prefix cache can inflate prompt-processing speed."""
    rng = random.Random()
    body = " ".join(rng.choice(_VOCAB) for _ in range(words))
    return f"[{rng.getrandbits(32):08x}] Summarise the theme of this text in one short sentence.\n\n{body}"


def measure(cfg, session, gpu, host, ha, model, prompt, max_tokens, timeout, cache=False) -> dict:
    """One timed generation plus the telemetry captured during exactly that window."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": cfg.temperature, "stream": False,
               "cache_prompt": cache}
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
            "cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
            "gpu_w": gpu_w, "gpu_peak_w": _max(_col(g, "gpu_w")), "gpu_util": _avg(_col(g, "gpu_util")),
            "gpu_temp": _max(_col(g, "gpu_temp")), "vram_gb": _max(_col(g, "vram_gb")),
            "vram_total_gb": _max(_col(g, "vram_total_gb")),
            "cpu_w": cpu_w, "cpu_util": h.get("cpu_util"), "sys_mem_gb": h.get("sys_mem_gb")}


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
# Per-model benchmark and post-processing
# --------------------------------------------------------------------------- #

FIELDS = ["timestamp", "model", "status", "error", "params_b", "active_b", "arch", "quant", "ctx_size", "load_s",
          "tech_tps", "logic_tps", "avg_tps", "prefill_tps", "prefill_tokens", "cache_hit_pct", "cache_speedup",
          "ttft_ms", "gen_ms", "latency_s", "gpu_w", "gpu_peak_w", "idle_gpu_w", "gpu_util", "gpu_temp",
          "vram_gb", "vram_delta_gb", "vram_total_gb", "vram_pct", "cpu_w", "cpu_util", "sys_mem_gb",
          "energy_wh", "j_per_token", "cost_1m_peak", "cost_1m_off", "cost_1kq_peak", "cost_1kq_off",
          "tps_per_watt", "goldilocks", "eligible"]


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
        r["ctx_size"] = get_ctx(cfg, session, model)
        if r["ctx_size"] is None and not unload_state.get("ctx_warned"):
            unload_state["ctx_warned"] = True
            print("  [warn] could not read the context size (tried /upstream/<model>/props and /running)")

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
        cache_runs = []
        if cfg.prefill_words > 0 and cfg.cache_test:
            try:
                cache_runs = [cache_test(cfg, session, gpu, host, ha, model) for _ in range(cfg.runs)]
            except BenchError:
                cache_runs = []          # the cache test is optional; never fail the model for it
    except BenchError as e:
        r.update(status="error", error=str(e))
        return r

    med = lambda k, f: _median([x[f] for x in runs[k]])
    gen = runs["tech"] + runs["logic"]
    every = gen + runs.get("prefill", [])
    r["tech_tps"], r["logic_tps"] = med("tech", "tps"), med("logic", "tps")
    if "prefill" in runs:
        r["prefill_tps"] = med("prefill", "prompt_tps")
        r["prefill_tokens"] = med("prefill", "prompt_n")
    if cache_runs:
        r["cache_hit_pct"] = _median([c["hit"] for c in cache_runs])
        r["cache_speedup"] = _median([c["speedup"] for c in cache_runs])
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
        r["active_b"] = r["active_b"] or parse_active(r["model"])
        r["arch"] = r["arch"] or ("MoE" if r["active_b"] else "Dense")
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
        if cfg.rate_peak is not None:
            off = cfg.rate_offpeak if cfg.rate_offpeak is not None else cfg.rate_peak
            if r["j_per_token"] is not None:        # J/token -> kWh per 1M tokens = J/3.6
                r["cost_1m_peak"], r["cost_1m_off"] = (r["j_per_token"] / 3.6 * x for x in (cfg.rate_peak, off))
            if r["energy_wh"] is not None:          # energy_wh covers 2 queries -> 1000 queries = 0.5 kWh per Wh
                r["cost_1kq_peak"], r["cost_1kq_off"] = (r["energy_wh"] * 0.5 * x for x in (cfg.rate_peak, off))
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
        r.update(ctx_size=65536 if vram > 12 else 131072, prefill_tokens=1500, cache_hit_pct=97 - vram / 4,
                 cache_speedup=6 + tps / 20, load_s=vram / 2.5, tech_tps=tps * 1.04, logic_tps=tps * 0.96, prefill_tps=tps * 14,
                 ttft_ms=60 + 15000 / tps, gen_ms=300 / tps * 1000, latency_s=lat, gpu_w=gw,
                 gpu_peak_w=gw * 1.15, idle_gpu_w=12, gpu_util=88, gpu_temp=62 + gw / 20, vram_gb=vram,
                 vram_delta_gb=vram - 0.6, vram_total_gb=cfg.vram_limit_gb, cpu_w=cw, cpu_util=25,
                 sys_mem_gb=18, energy_wh=(gw + cw) * lat * 2 / 3600, j_per_token=(gw + cw) / tps)
        out.append(r)
    return out


def check(cfg):
    """Verify every telemetry source for ~3 seconds. Run this before a long benchmark."""
    print(f"Hardware: {detect_hardware(cfg) or 'not detected'}")
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
          "| Model | Params | Quant | Ctx | Load s | VRAM | GPU W | CPU W | Wh | J/tok | TTFT ms | Prefill t/s | Tech | Logic | Avg |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] != "ok":
            L.append(f"| {r['model']} | | | | | | | | | | | | | | ERR: {r['error']} |")
            continue
        p = r["vram_pct"]
        vram = "–" if p is None else f"{r['vram_gb']:.1f} GB [{'=' * round(min(p, 100) / 10):<10}] {p:.0f}%"
        L.append(f"| {r['model']} | {_f(r['params_b'], 0)}B | {r['quant']} | {_f((r['ctx_size'] or 0) / 1024 or None, 0)}k | {_f(r['load_s'])} | {vram} | "
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


def render_dashboard(results, cfg, path, cpu_source="none", hardware=""):
    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "count": len(results),
            "vram_limit": cfg.vram_limit_gb, "runs": cfg.runs, "goldilocks_mode": cfg.goldilocks_mode,
            "timeouts": [cfg.timeout_tech, cfg.timeout_logic], "max_tokens": [cfg.max_tokens_tech, cfg.max_tokens_logic],
            "prompts": [cfg.tech_prompt, cfg.logic_prompt], "max_vram_pct": cfg.goldilocks_max_vram_pct,
            "poll_ms": cfg.gpu_poll_ms, "cpu_source": cpu_source, "prefill_words": cfg.prefill_words,
            "hardware": hardware, "currency": cfg.currency, "has_cost": cfg.rate_peak is not None,
            "rate_peak": cfg.rate_peak, "rate_off": cfg.rate_offpeak if cfg.rate_offpeak is not None else cfg.rate_peak,
            "cache_test": cfg.cache_test}
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
body{margin:0 auto;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px;max-width:1600px}
a{color:#58a6ff}
h1{margin:0 0 4px}h2{margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mute)}
.grid{display:grid;gap:16px;align-items:start}.podium{grid-template-columns:repeat(auto-fit,minmax(270px,1fr))}
.charts{grid-template-columns:repeat(auto-fit,minmax(560px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow:hidden}
.card h3{margin:0 0 8px;font-size:15px}
.rank{font-size:12px;font-weight:700;letter-spacing:.08em}
.gold{border-color:#d4a72c}.gold .rank{color:#d4a72c}.silver{border-color:#9aa4b0}.silver .rank{color:#9aa4b0}
.bronze{border-color:#bf7a3a}.bronze .rank{color:#bf7a3a}
.pname{font-weight:600;word-break:break-all;margin:4px 0 8px}.stat{color:var(--mute)}.stat b{color:var(--text)}
.radarbox{position:relative;height:210px;margin-top:8px}
.chartbox{position:relative;height:380px}
/* filter bar */
.filters{position:sticky;top:0;z-index:30;margin:16px 0;padding:10px 14px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center}
.fg{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.fg>span{color:var(--mute);font-size:12px;margin-right:2px}
.tog{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:14px;padding:3px 11px;font:inherit;font-size:12px;cursor:pointer}
.tog:hover{border-color:#58a6ff}.tog.on{background:#1f3a5f;border-color:#58a6ff}
/* table: vertical scroll, frozen header and first five columns */
#tblwrap{max-height:72vh;overflow:auto;position:relative;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;background:var(--card)}
th{position:sticky;top:0;z-index:3;cursor:pointer;color:var(--mute);user-select:none}
th:hover{color:var(--text)}
th:first-child,td:first-child{text-align:left}
.fz{position:sticky;z-index:2}th.fz{z-index:5}.fz.last{border-right:2px solid #484f58}
td.warn{background:#3a1d1f}
tr[data-m]{cursor:pointer}tr[data-m]:hover td{background:#1c2128}tr.sel td{background:#1f2a3d}
.clip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{display:inline-block;width:70px;height:8px;background:#21262d;border-radius:4px;vertical-align:middle;margin:0 6px;overflow:hidden}
.bar i{display:block;height:100%}
.err{color:#f85149;font-weight:600}
small.d{font-size:11px;margin-left:4px}.good{color:#3fb950}.bad{color:#f85149}.dn{color:var(--mute)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin-bottom:10px;min-height:28px}
.tablewrap2{overflow-x:auto}
.qt th,.qt td{background:transparent}.qt th{position:static}
footer{margin-top:40px}
@media(max-width:900px){body{padding:12px}.charts{grid-template-columns:1fr}.fz{position:static}}
</style></head><body>
<h1>llama-swap Benchmark</h1>
<div class="sub" id="hdr"></div>

<div class="card filters" id="filters"></div>

<h2>Top 3 &mdash; Goldilocks</h2><div class="grid podium" id="podium"></div>

<h2>Charts</h2>
<div class="grid charts" id="charts"></div>

<h2>Full model comparison</h2>
<div class="toolbar" id="toolbar"></div>
<div id="tblwrap"><table id="tbl"></table></div>

<h2>Quantization analysis</h2>
<div class="card tablewrap2"><table id="qtbl" class="qt"></table></div>

<h2>Methodology</h2><div class="card" id="meth"></div>

<footer>
<h2>Project &amp; source code</h2>
<div class="card">Source, issues and updates:
 <a href="https://github.com/Maximo101/llama-swap-llm-benchmark" target="_blank" rel="noopener noreferrer">https://github.com/Maximo101/llama-swap-llm-benchmark</a></div>
</footer>

<script>
const D = __DATA__, META = D.meta, ALL = D.models;
const $ = id => document.getElementById(id);
const isOK = m => m.status === 'ok' && m.avg_tps != null;
const ALLOK = ALL.filter(isOK);
const cur = META.currency;

// Chart.js cannot resolve CSS var(): every chart colour below is a literal hex value.
function tpsColor(t){return t>100?'#a371f7':t>80?'#58a6ff':t>50?'#3fb950':t>30?'#d29922':'#f778ba';}
function vramColor(p){return p>90?'#f85149':p>70?'#db6d28':'#3fb950';}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v,d=1) => v==null ? '&ndash;' : Number(v).toFixed(d);
const ctxFmt = v => v==null ? '&ndash;' : (v>=1024 ? Math.round(v/1024)+'k' : String(v));

Chart.defaults.color = '#e6edf3';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.maintainAspectRatio = false;

const state = {filters:{quant:new Set(), arch:new Set(), size:new Set(), vram95:false, hideErr:false},
               sk:'avg_tps', sd:-1, base:null, rate:'peak'};

/* ---------- filters ---------- */
const BUCKETS = [['lt10','<10B',p=>p<10],['10-20','10\u201320B',p=>p>=10&&p<20],['20-40','20\u201340B',p=>p>=20&&p<40],['gt40','\u226540B',p=>p>=40]];
const bucket = p => p==null ? 'unk' : BUCKETS.find(b=>b[2](p))[0];
function pass(m){
  const f = state.filters;
  if (f.quant.size && !f.quant.has(m.quant)) return false;
  if (f.arch.size && !f.arch.has(m.arch)) return false;
  if (f.size.size && !f.size.has(bucket(m.params_b))) return false;
  if (f.vram95 && m.vram_pct!=null && m.vram_pct>=95) return false;
  if (f.hideErr && m.status!=='ok') return false;
  return true;
}
function buildFilters(){
  const quants = [...new Set(ALL.map(m=>m.quant))].sort();
  const archs = [...new Set(ALL.map(m=>m.arch).filter(Boolean))].sort();
  const sizes = BUCKETS.filter(b=>ALL.some(m=>m.params_b!=null && b[2](m.params_b))).map(b=>[b[0],b[1]]);
  if (ALL.some(m=>m.params_b==null)) sizes.push(['unk','unknown']);
  const grp = (label,g,items) => `<div class="fg"><span>${label}</span>${items.map(([v,t])=>`<button class="tog" data-g="${g}" data-v="${esc(v)}">${esc(t)}</button>`).join('')}</div>`;
  $('filters').innerHTML =
    grp('Quant','quant',quants.map(q=>[q,q])) +
    (archs.length>1 ? grp('Type','arch',archs.map(a=>[a,a])) : '') +
    grp('Size','size',sizes) +
    grp('Show','x',[['vram95','Hide VRAM \u226595%'],['hideErr','Hide failed']]) +
    `<div class="fg"><button class="tog" data-reset="1">Reset</button><span id="fcount"></span></div>`;
  $('filters').onclick = e => {
    const b = e.target.closest('button'); if(!b) return;
    if (b.dataset.reset){ Object.values(state.filters).forEach(v=>v instanceof Set?v.clear():0); state.filters.vram95=state.filters.hideErr=false;
      document.querySelectorAll('#filters .tog').forEach(x=>x.classList.remove('on')); return render(); }
    const g=b.dataset.g, v=b.dataset.v; if(!g) return;
    if (g==='x') state.filters[v] = !state.filters[v];
    else { const s=state.filters[g]; s.has(v)?s.delete(v):s.add(v); }
    b.classList.toggle('on'); render();
  };
}

/* ---------- charts ---------- */
const charts = {};
function mk(id, cfg){ if(charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); }
function drop(id){ if(charts[id]){ charts[id].destroy(); delete charts[id]; } }
// Shows/hides a chart card and sizes it to fit every label.
function box(key, items, h){
  const b=$('b_'+key), e=$('e_'+key);
  if(!items.length){ drop('c_'+key); b.style.display='none'; e.hidden=false; return false; }
  b.style.display=''; e.hidden=true; b.style.height=h+'px'; return true;
}
const rowsH = n => Math.max(340, n*26+90);
const valueLabels = {id:'valueLabels', afterDatasetsDraw(chart){
  if (chart.data.datasets.length!==1) return;
  const c = chart.ctx, horiz = chart.options.indexAxis === 'y';
  c.save(); c.fillStyle = '#e6edf3'; c.font = '12px sans-serif'; c.textBaseline = 'middle';
  c.textAlign = horiz ? 'left' : 'center';
  chart.getDatasetMeta(0).data.forEach((bar,i)=>{
    const v = chart.data.datasets[0].data[i]; if (v==null) return;
    c.fillText(v.toFixed(v<10?2:1), horiz ? bar.x+6 : bar.x, horiz ? bar.y : bar.y-10);
  }); c.restore();
}};
const yTicks = {autoSkip:false, font:{size:11}};
const axTitle = t => ({display:!!t, text:t||'', color:'#e6edf3'});
function hbar(key, items, valFn, xTitle){
  if(!box(key, items, rowsH(items.length))) return;
  const colors = items.map(m=>tpsColor(m.avg_tps));
  mk('c_'+key, {type:'bar', plugins:[valueLabels],
    data:{labels:items.map(m=>m.model), datasets:[{data:items.map(valFn), backgroundColor:colors, borderColor:colors, borderWidth:1}]},
    options:{indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{grace:'12%', beginAtZero:true, title:axTitle(xTitle)}, y:{ticks:yTicks}}}});
}
const pt = (m,x,y) => ({x, y, name:m.model});

function renderCharts(OKR){
  hbar('tps', [...OKR].sort((a,b)=>b.avg_tps-a.avg_tps), m=>m.avg_tps, 'tokens/sec');

  const vp = OKR.filter(m=>m.vram_gb!=null);
  if (box('vram', vp, 420)){
    const ps = vp.map(m=>m.params_b).filter(p=>p!=null), pmin=Math.min(...ps), pmax=Math.max(...ps);
    const rad = p => p==null ? 8 : (pmax===pmin ? 15 : 5 + (p-pmin)/(pmax-pmin)*20);
    mk('c_vram', {type:'bubble',
      data:{datasets:[{data:vp.map(m=>({x:m.vram_gb,y:m.avg_tps,r:rad(m.params_b),name:m.model,p:m.params_b})),
        backgroundColor:vp.map(m=>tpsColor(m.avg_tps)+'b3'), borderColor:vp.map(m=>tpsColor(m.avg_tps))}]},
      options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.raw.name}: ${c.raw.y.toFixed(1)} tok/s, ${c.raw.x.toFixed(1)} GB, ${c.raw.p??'?'}B`}}},
        scales:{x:{min:Math.floor(Math.min(...vp.map(m=>m.vram_gb))-1), max:Math.ceil(Math.max(...vp.map(m=>m.vram_gb))+1), title:axTitle('VRAM used (GB)')},
                y:{min:0, max:Math.ceil(Math.max(...vp.map(m=>m.avg_tps))+20), title:axTitle('Avg tok/s')}}}});
  }

  hbar('eff', OKR.filter(m=>m.tps_per_watt!=null).sort((a,b)=>b.tps_per_watt-a.tps_per_watt), m=>m.tps_per_watt, 'tok/s per W');
  hbar('jt', OKR.filter(m=>m.j_per_token!=null).sort((a,b)=>a.j_per_token-b.j_per_token), m=>m.j_per_token, 'joules per token');
  hbar('ttft', OKR.filter(m=>m.ttft_ms!=null).sort((a,b)=>a.ttft_ms-b.ttft_ms), m=>m.ttft_ms, 'milliseconds');

  const lp = OKR.filter(m=>m.ttft_ms!=null);
  if (box('lat', lp, 420)){
    mk('c_lat', {type:'scatter',
      data:{datasets:[{data:lp.map(m=>pt(m,m.ttft_ms,m.avg_tps)), pointRadius:7, pointHoverRadius:10,
        backgroundColor:lp.map(m=>tpsColor(m.avg_tps)+'cc'), borderColor:lp.map(m=>tpsColor(m.avg_tps))}]},
      options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.raw.name}: TTFT ${c.raw.x.toFixed(0)} ms, ${c.raw.y.toFixed(1)} tok/s`}}},
        scales:{x:{min:0, title:axTitle('TTFT (ms) \u2014 lower is more responsive')}, y:{min:0, title:axTitle('Avg tok/s \u2014 higher is more throughput')}}}});
  }

  const pg = OKR.filter(m=>m.prefill_tps!=null).sort((a,b)=>b.prefill_tps-a.prefill_tps);
  if (box('pg', pg, Math.max(380, pg.length*44+100))){
    mk('c_pg', {type:'bar',
      data:{labels:pg.map(m=>m.model), datasets:[
        {label:'Prefill (prompt processing)', data:pg.map(m=>m.prefill_tps), backgroundColor:'#58a6ff', borderColor:'#58a6ff', borderWidth:1},
        {label:'Generation (avg)', data:pg.map(m=>m.avg_tps), backgroundColor:'#3fb950', borderColor:'#3fb950', borderWidth:1}]},
      options:{indexAxis:'y', plugins:{legend:{display:true}}, scales:{x:{type:'logarithmic', min:1, title:axTitle('tokens/sec (log scale)')}, y:{ticks:yTicks}}}});
  }

  const pw = OKR.filter(m=>m.gpu_w!=null).sort((a,b)=>b.gpu_w-a.gpu_w);
  if (box('pwr', pw, Math.max(380, pw.length*30+100))){
    mk('c_pwr', {type:'bar',
      data:{labels:pw.map(m=>m.model), datasets:[
        {label:'Idle (baseline)', data:pw.map(m=>m.idle_gpu_w??0), backgroundColor:'#6e7681', borderColor:'#6e7681', borderWidth:1},
        {label:'Active increase', data:pw.map(m=>Math.max(0,m.gpu_w-(m.idle_gpu_w??0))), backgroundColor:'#f0883e', borderColor:'#f0883e', borderWidth:1}]},
      options:{indexAxis:'y', plugins:{legend:{display:true}, tooltip:{callbacks:{footer:c=>'Active total: '+c.reduce((s,x)=>s+x.parsed.x,0).toFixed(0)+' W'}}},
        scales:{x:{stacked:true, title:axTitle('GPU watts')}, y:{stacked:true, ticks:yTicks}}}});
  }

  const qs = quantGroups(OKR);
  if (box('quant', qs, 340)){
    const colors = qs.map(x=>tpsColor(x.avg));
    mk('c_quant', {type:'bar', plugins:[valueLabels],
      data:{labels:qs.map(x=>x.q), datasets:[{data:qs.map(x=>x.avg), backgroundColor:colors, borderColor:colors, borderWidth:1}]},
      options:{plugins:{legend:{display:false}}, scales:{x:{ticks:{autoSkip:false}}, y:{grace:'12%', beginAtZero:true}}}});
  }
}

/* ---------- podium + radar ---------- */
const maxOf = f => Math.max(1e-9, ...ALLOK.map(m=>f(m)||0));
const MAXTPS = maxOf(m=>m.avg_tps), MAXEFF = maxOf(m=>m.tps_per_watt), MAXP = maxOf(m=>m.params_b);
// All four axes are 0-1: speed, efficiency and params relative to the best model; VRAM economy = 1 - VRAM%.
const radarVals = m => [m.avg_tps/MAXTPS, (m.tps_per_watt||0)/MAXEFF, (m.params_b||0)/MAXP, m.vram_pct==null?0:Math.max(0,1-m.vram_pct/100)];
function renderPodium(OKR){
  [0,1,2].forEach(i=>drop('rad'+i));
  const top3 = OKR.filter(m=>m.eligible).sort((a,b)=>b.goldilocks-a.goldilocks).slice(0,3);
  const cls = ['gold','silver','bronze'];
  $('podium').innerHTML = top3.length ? top3.map((m,i)=>`
    <div class="card ${cls[i]}"><div class="rank">#${i+1} GOLDILOCKS</div>
    <div class="pname">${esc(m.model)}</div>
    <div class="stat"><b style="color:${tpsColor(m.avg_tps)}">${fmt(m.avg_tps)}</b> tok/s &middot; ${m.params_b}B params &middot; ctx ${ctxFmt(m.ctx_size)}</div>
    <div class="stat">VRAM <b>${fmt(m.vram_gb)} GB</b> (${fmt(m.vram_pct,0)}%) &middot; GPU <b>${fmt(m.gpu_w,0)} W</b> &middot; CPU <b>${fmt(m.cpu_w,0)} W</b></div>
    <div class="stat">Score <b>${fmt(m.goldilocks,0)}</b> &middot; <b>${fmt(m.j_per_token,2)}</b> J/token</div>
    <div class="radarbox"><canvas id="rad${i}"></canvas></div></div>`).join('')
    : '<div class="card sub">No eligible models for the current filters (need parsable params and a speed result).</div>';
  top3.forEach((m,i)=>mk('rad'+i, {type:'radar',
    data:{labels:['Speed','Efficiency','Reasoning (params)','VRAM economy'],
      datasets:[{data:radarVals(m), borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.25)', pointBackgroundColor:'#58a6ff', borderWidth:2}]},
    options:{plugins:{legend:{display:false}},
      scales:{r:{min:0, max:1, ticks:{display:false, stepSize:0.25}, grid:{color:'#30363d'}, angleLines:{color:'#30363d'}, pointLabels:{color:'#e6edf3', font:{size:10}}}}}}));
}

/* ---------- quantization table ---------- */
function quantGroups(OKR){
  const g = {}; OKR.forEach(m=>(g[m.quant] = g[m.quant]||[]).push(m));
  return Object.entries(g).map(([q,ms])=>({q, n:ms.length, avg:ms.reduce((s,m)=>s+m.avg_tps,0)/ms.length, names:ms.map(m=>m.model)})).sort((a,b)=>b.avg-a.avg);
}
function renderQuant(OKR){
  $('qtbl').innerHTML = '<tr><th>Quant</th><th>Models</th><th>Avg TPS</th><th style="text-align:left">Members</th></tr>' +
    quantGroups(OKR).map(x=>`<tr><td>${esc(x.q)}</td><td>${x.n}</td><td style="color:${tpsColor(x.avg)}">${fmt(x.avg)}</td><td style="text-align:left;white-space:normal">${x.names.map(esc).join(', ')}</td></tr>`).join('');
}

/* ---------- comparison table ---------- */
const FZ = [260,64,84,78,236], FL = [0,260,324,408,486];   // widths / left offsets of the 5 frozen columns
const rateKey = () => state.rate==='off' ? 'off' : 'peak';
const COLS = [
 {k:'model',l:'Model'},
 {k:'params_b',l:'Params',f:v=>v+'B',nodelta:1},
 {k:'quant',l:'Quant',nodelta:1},
 {k:'load_s',l:'Load s',dec:1,better:'low'},
 {k:'vram_gb',l:'VRAM',better:'low'},
 {k:'ctx_size',l:'Ctx',f:ctxFmt,nodelta:1},
 {k:'arch',l:'Type',nodelta:1},
 {k:'gpu_w',l:'GPU W',dec:0,better:'low'},
 {k:'idle_gpu_w',l:'Idle W',dec:0,better:'low'},
 {k:'gpu_peak_w',l:'Peak W',dec:0,better:'low'},
 {k:'cpu_w',l:'CPU W',dec:0,better:'low'},
 {k:'gpu_util',l:'GPU %',dec:0},
 {k:'gpu_temp',l:'Temp \u00b0C',dec:0,better:'low'},
 {k:'ttft_ms',l:'TTFT ms',dec:0,better:'low'},
 {k:'latency_s',l:'Latency s',dec:1,better:'low'},
 {k:'energy_wh',l:'Energy Wh',dec:3,better:'low'},
 {k:'j_per_token',l:'J/token',dec:2,better:'low'},
 {k:'cost1m',l:cur+'/1M tok',dec:3,better:'low',cost:1,get:m=>m['cost_1m_'+rateKey()]},
 {k:'cost1kq',l:cur+'/1k queries',dec:4,better:'low',cost:1,get:m=>m['cost_1kq_'+rateKey()]},
 {k:'prefill_tps',l:'Prefill t/s',dec:0,better:'high'},
 {k:'prefill_tokens',l:'Prefill tok',dec:0,nodelta:1},
 {k:'cache_hit_pct',l:'Cache hit %',dec:0,better:'high'},
 {k:'cache_speedup',l:'Cache \u00d7',dec:1,better:'high'},
 {k:'tech_tps',l:'Tech TPS',tps:1,better:'high'},
 {k:'logic_tps',l:'Logic TPS',tps:1,better:'high'},
 {k:'avg_tps',l:'Avg TPS',tps:1,better:'high'},
 {k:'goldilocks',l:'Goldilocks',dec:0,better:'high'}
].filter(c=>!c.cost || META.has_cost);
const val = (m,c) => c.get ? c.get(m) : m[c.k];

function deltaHtml(m, c, v, base){
  if (!base || m.model===base.model || c.nodelta || v==null) return '';
  const b = val(base,c); if (b==null || b===0) return '';
  const pct = (v/b-1)*100; if (!isFinite(pct)) return '';
  const small = Math.abs(pct) < 0.5;
  const good = c.better ? ((c.better==='high') === (pct>0)) : null;
  const cl = small || good==null ? 'dn' : (good ? 'good' : 'bad');
  return `<small class="d ${cl}">${pct>0?'+':''}${pct.toFixed(0)}%</small>`;
}
function cell(m, c, ci, base){
  const v = val(m,c), fz = ci<5;
  const st = fz ? ` style="left:${FL[ci]}px;width:${FZ[ci]}px;min-width:${FZ[ci]}px;max-width:${FZ[ci]}px"` : '';
  let cls = fz ? 'fz'+(ci===4?' last':'') : '', inner;
  const dl = deltaHtml(m,c,v,base);
  if (c.k==='model'){
    const star = base && base.model===m.model ? '\u2605 ' : '';
    inner = `<div class="clip" title="${esc(v)}${m.error?' \u2014 '+esc(m.error):''}">${star}${esc(v)}${m.status!=='ok'?` <span class="err">ERR: ${esc(m.error)}</span>`:''}</div>`;
  } else if (c.k==='vram_gb'){
    if (v==null) inner = '&ndash;';
    else {
      const raw = m.vram_pct ?? 0, p = Math.min(raw,100), warn = raw >= 95;
      if (warn) cls += ' warn';
      inner = `<div class="clip">${v.toFixed(1)} GB<span class="bar"><i style="width:${p}%;background:${warn?'#f85149':vramColor(p)}"></i></span><span class="${warn?'err':''}">${warn?'\u26A0 ':''}${raw.toFixed(0)}%</span>${dl}</div>`;
    }
  } else if (c.k==='quant' || c.k==='arch') inner = esc(v ?? '\u2013');
  else if (v==null) inner = '&ndash;';
  else if (c.f) inner = c.f(v);
  else {
    const t = (c.cost ? cur : '') + Number(v).toFixed(c.dec ?? 1);
    inner = (c.tps ? `<span style="color:${tpsColor(v)}">${t}</span>` : t) + dl;
  }
  return `<td class="${cls}"${st}>${inner}</td>`;
}
function renderTable(){
  const wrap = $('tblwrap'), sx = wrap.scrollLeft, sy = wrap.scrollTop;
  const base = state.base ? ALL.find(m=>m.model===state.base && m.status==='ok') : null;
  const col = COLS.find(c=>c.k===state.sk) || COLS[0];
  const rows = [...cur_rows].sort((a,b)=>{
    const x=val(a,col), y=val(b,col);
    if (x==null && y==null) return 0; if (x==null) return 1; if (y==null) return -1;
    return (typeof x==='string' ? x.localeCompare(y) : x-y) * state.sd;
  });
  $('tbl').innerHTML = '<thead><tr>' + COLS.map((c,i)=>{
      const fz = i<5, st = fz ? ` style="left:${FL[i]}px;width:${FZ[i]}px;min-width:${FZ[i]}px;max-width:${FZ[i]}px"` : '';
      return `<th class="${fz?'fz'+(i===4?' last':''):''}" data-k="${c.k}"${st}>${c.l}${state.sk===c.k?(state.sd<0?' \u25BC':' \u25B2'):''}</th>`;
    }).join('') + '</tr></thead><tbody>' +
    rows.map(m=>`<tr data-m="${encodeURIComponent(m.model)}"${base&&base.model===m.model?' class="sel"':''}>${COLS.map((c,i)=>cell(m,c,i,base)).join('')}</tr>`).join('') + '</tbody>';
  wrap.scrollLeft = sx; wrap.scrollTop = sy;
  renderToolbar(base);
}
function renderToolbar(base){
  const rate = (META.has_cost && META.rate_peak!==META.rate_off)
    ? `<span class="fg"><span>Electricity rate</span><button class="tog ${state.rate==='peak'?'on':''}" data-rate="peak">Peak ${cur}${META.rate_peak}/kWh</button><button class="tog ${state.rate==='off'?'on':''}" data-rate="off">Off-peak ${cur}${META.rate_off}/kWh</button></span>`
    : (META.has_cost ? `<span class="sub">Electricity ${cur}${META.rate_peak}/kWh</span>` : '');
  const bl = base
    ? `<span><span class="good">\u2605 Baseline:</span> <b>${esc(base.model)}</b> &mdash; other rows show change vs this model (green = better, red = worse). <button class="tog" data-clear="1">Clear baseline</button></span>`
    : '<span class="sub">Click a column header to sort (click again to reverse). Click a row to use it as a 100% baseline; click it again to clear.</span>';
  $('toolbar').innerHTML = bl + rate;
}
function bindTable(){
  $('tblwrap').onclick = e => {
    const th = e.target.closest('th');
    if (th){ const k = th.dataset.k; if (state.sk===k) state.sd = -state.sd; else { state.sk=k; state.sd=1; } return renderTable(); }
    const tr = e.target.closest('tr[data-m]');
    if (tr){ const name = decodeURIComponent(tr.dataset.m);
      if (!ALL.find(m=>m.model===name && m.status==='ok')) return;
      state.base = state.base===name ? null : name; renderTable(); }
  };
  $('toolbar').onclick = e => {
    const b = e.target.closest('button'); if(!b) return;
    if (b.dataset.clear) state.base = null;
    if (b.dataset.rate) state.rate = b.dataset.rate;
    renderTable();
  };
}

/* ---------- page ---------- */
let cur_rows = [];
function render(){
  cur_rows = ALL.filter(pass);
  const OKR = cur_rows.filter(isOK);
  $('fcount').textContent = `Showing ${cur_rows.length} of ${ALL.length} models`;
  renderPodium(OKR); renderCharts(OKR); renderQuant(OKR); renderTable();
}

$('hdr').innerHTML = `${META.count} models &middot; ${esc(META.generated)} &middot; llama.cpp via llama-swap<br>` +
  `<b style="color:#e6edf3">Host:</b> ${META.hardware ? esc(META.hardware) : 'VRAM limit '+META.vram_limit+' GB'} &middot; ` +
  `<b style="color:#e6edf3">Power:</b> GPU via nvidia-smi (${META.poll_ms} ms), CPU via ${esc(META.cpu_source)}`;
const CH = [['tps','Avg generation speed (tok/s)'],['vram','VRAM vs speed (bubble = params)'],['eff','Power efficiency (tok/s per W, GPU + CPU)'],
  ['jt','Energy per token (J, lower is better)'],['ttft','TTFT leaderboard (ms, lower is better)'],['lat','Latency vs throughput (top-left is best)'],
  ['pg','Prefill vs generation speed (tok/s, log scale)'],['pwr','Active vs idle GPU power (W)'],['quant','Avg TPS by quantization']];
$('charts').innerHTML = CH.map(([id,t])=>`<div class="card"><h3>${t}</h3><div class="chartbox" id="b_${id}"><canvas id="c_${id}"></canvas></div><p class="sub" id="e_${id}" hidden>No data for this chart (telemetry unavailable, or filtered out).</p></div>`).join('');
buildFilters(); bindTable(); render();

$('meth').innerHTML = `<ul>
 <li><b>Backend:</b> llama-swap (OpenAI-compatible <code>/v1/chat/completions</code>), non-streaming. The prompt cache is disabled for all speed tests.</li>
 <li><b>Per model:</b> all models unloaded &rarr; idle baseline sampled &rarr; cold load (Load s) &rarr; Technical, Logic${META.prefill_words?` and prefill (~${META.prefill_words}-word, unique prompt)`:''} tests, median of ${META.runs} run(s).</li>
 <li><b>Prompts:</b> Technical &mdash; &ldquo;${esc(META.prompts[0])}&rdquo;; Logic &mdash; &ldquo;${esc(META.prompts[1])}&rdquo;.</li>
 <li><b>Speed:</b> <code>timings.predicted_per_second</code>, falling back to completion_tokens / wall time. <b>TTFT</b> is approximated by <code>timings.prompt_ms</code>; <b>Prefill t/s</b> is <code>timings.prompt_per_second</code> on the long prompt (<b>Prefill tok</b> = its size).</li>
 <li><b>Ctx</b> is the context size the model was loaded with (read from llama-server <code>/props</code>). VRAM and prefill speed depend on it, so compare models with similar Ctx.</li>
 ${META.cache_test?'<li><b>Cache hit %</b> / <b>Cache &times;:</b> a long prompt is sent cold, then again with a short follow-up appended (a multi-turn proxy). Hit % = share of the second prompt reused from the KV cache (<code>cache_n</code>); &times; = how much faster its prompt processing was than the cold one.</li>':''}
 <li><b>GPU:</b> one persistent <code>nvidia-smi</code> stream, windowed to each request. Power = mean; peak W, temp and VRAM = maximum. <b>Idle W</b> is sampled with no model loaded. <b>CPU:</b> ${esc(META.cpu_source)}.</li>
 <li><b>Energy:</b> (mean GPU W + mean CPU W) &times; request time; Energy Wh covers one Technical plus one Logic request. <b>J/token</b> = energy &divide; generated tokens. <b>Efficiency</b> = avg TPS &divide; (GPU W + CPU W).</li>
 ${META.has_cost?`<li><b>Cost:</b> measured GPU + CPU energy only (not the rest of the system) &times; your electricity rate. ${cur}/1k queries = average of the Technical and Logic requests above, so it scales with the max_tokens setting.</li>`:''}
 <li><b>Goldilocks:</b> ${META.goldilocks_mode==='log'?'ln(params&nbsp;B)':'params&nbsp;B'} &times; avg TPS, VRAM &le; ${META.max_vram_pct}% of the GPU. Params are parsed from the filename and are <i>total</i> (not active) parameters. <b>Radar:</b> Speed, Efficiency and Reasoning (params) are relative to the best model in this run; VRAM economy = 100% &minus; VRAM%.</li>
 <li><b>Limits:</b> max_tokens ${META.max_tokens.join(' / ')}; timeouts ${META.timeouts.join('s / ')}s. KV-cache type differs per model config and affects speed and VRAM.</li>
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
    results, cpu_source, hardware = [], "n/a", ""

    if args.demo:
        results, cpu_source = demo_results(cfg), "demo data"
        hardware = detect_hardware(cfg) or "Demo system (synthetic data)"
    else:
        session = requests.Session()
        if cfg.api_key:
            session.headers["Authorization"] = f"Bearer {cfg.api_key}"
        models = discover_models(cfg, session)
        hardware = detect_hardware(cfg)

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
    render_dashboard(results, cfg, html_path, cpu_source, hardware)
    out = [f"CSV:       {csv_path}", f"Dashboard: {html_path}"]
    if cfg.write_markdown:
        md_path = cfg.output_dir / "benchmark_report.md"
        render_markdown(results, cfg, md_path)
        out.append(f"Markdown:  {md_path}")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
