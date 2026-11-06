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

    python benchmark.py --check              # verify telemetry sources before a long run
    python benchmark.py                      # run the benchmark (+ evals if RUN_EVAL_TESTS=true)
    python benchmark.py --demo               # build a dashboard from synthetic data
    python benchmark.py --list-models        # print the models the server offers, then exit
    python benchmark.py --download-datasets  # cache the eval datasets enabled in .env into ./tests, then exit
    python benchmark.py --speed-only         # runs the speed only version
    python benchmark.py --evals-only         # runs the eval only version
    python benchmark.py --help               # --h  or --help For all CLI Flags
    
Optionally (RUN_EVAL_TESTS=true) it then runs dataset-based evaluations AFTER the speed tests:
GPQA Diamond and MMLU-Pro (multiple-choice accuracy) and HumanEval, MBPP and LiveCodeBench (code pass@1,
graded by executing the generated code in a sandbox). Each dataset is toggled in .env.

* TTFT is approximated by llama.cpp's `timings.prompt_ms` (prompt processing time).
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import fnmatch
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import signal
import statistics
import string
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
from urllib.parse import urlparse
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


def _export_hf_token():
    """Make HF_TOKEN (from .env or the shell) visible to huggingface_hub / datasets via os.environ so downloads
    are authenticated and avoid the anonymous rate limit. Empty / unset = no authentication (the default)."""
    tok = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
    if tok:
        os.environ["HF_TOKEN"] = tok
    else:
        os.environ.pop("HF_TOKEN", None)       # an empty value must not be sent as a bogus token


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


def _env_tokens(name):
    """Optional token cap from .env. Unset / empty / 'auto' / 'dynamic' -> None, meaning 'use the dynamic limit'."""
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() in ("", "auto", "dynamic"):
        return None
    try:
        v = int(raw.strip())
    except ValueError:
        sys.exit(f"[config] Invalid value for {name}: {raw!r} (a number, or leave it empty for the dynamic limit)")
    return v if v > 0 else None


FALLBACK_TOKENS = 4096      # max_tokens used when the model's context size or the prompt size is unknown


def _env_list(name, allow_all=False):
    """Comma-separated .env value -> tuple of stripped, non-empty items. With allow_all, 'all' / '*' means no filter."""
    raw = os.getenv(name)
    if raw is None:
        return ()
    items = tuple(x.strip().strip('"').strip("'") for x in raw.split(",") if x.strip().strip('"').strip("'"))
    if allow_all and any(x.lower() in ("all", "*") for x in items):
        return ()
    return items


# --- Evaluation datasets (toggle each one in .env) ---------------------------
# kind "qa"   = multiple choice, scored as accuracy;  kind "code" = executed against tests, scored as pass@1.
EVAL_SPECS = {
    "gpqa":          {"label": "GPQA Diamond",  "short": "GPQA",      "kind": "qa",   "env": "EVAL_GPQA",
                      "repo": "fingertap/GPQA-Diamond",               "split": "test"},
    "mmlu_pro":      {"label": "MMLU-Pro",      "short": "MMLU-Pro",  "kind": "qa",   "env": "EVAL_MMLU_PRO",
                      "repo": "TIGER-Lab/MMLU-Pro",                   "split": "test"},
    "humaneval":     {"label": "HumanEval",     "short": "HumanEval", "kind": "code", "env": "EVAL_HUMANEVAL",
                      "repo": "openai/openai_humaneval",              "split": "test"},
    "mbpp":          {"label": "MBPP",          "short": "MBPP",      "kind": "code", "env": "EVAL_MBPP",
                      "repo": "google-research-datasets/mbpp",        "split": "test"},
    "livecodebench": {"label": "LiveCodeBench", "short": "LCB",       "kind": "code", "env": "EVAL_LIVECODEBENCH",
                      "repo": "bzantium/livecodebench",               "split": "test"},
}
# Rows per active dataset. Sizes come from n = 0.9604 / E^2 (95% confidence, worst-case variance), where E is the
# margin of error shown in brackets. A dataset smaller than the profile is used in full (e.g. GPQA has 198 rows).
PROFILE_ROWS = {"test": 6,          # about +/-40%
                "quick": 11,        # +/-30%
                "basic": 24,        # +/-20%
                "fast": 43,         # +/-15%
                "standard": 96,     # +/-10%
                "thorough": 196,    # +/-7%
                "deep": 384,        # +/-5%
                "ultra": 9604,      # +/-1% (larger than every dataset except MMLU-Pro)
                "full": None}       # every row


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
    strata_url: str | None
    strata_api_key: str | None
    strata_model: str | None
    strata_params_b: float | None
    strata_active_b: float | None
    strata_quant: str | None
    strata_ctx: int | None
    goldilocks_vram_only: bool
    ram_process_pattern: str
    ram_poll_ms: int
    spill_threshold_gb: float
    ram_overhead_gb: float | None
    tech_prompt: str
    logic_prompt: str
    backend: str = "llama-swap"       # set at run time: "llama-swap" or "strata"
    strata_model_id: str | None = None      # id reported by Strata's /v1/models
    strata_ctx_detected: int | None = None  # context size read from Strata's API
    strata_details: list | None = None      # [[endpoint, [[key, value], ...]], ...] for the dashboard
    # --- model selection ---
    include_models: tuple = ()              # empty = all models
    exclude_models: tuple = ()
    # --- evaluations ---
    run_eval_tests: bool = False
    run_speed_tests: bool = True            # false = API-client mode: no GPU/CPU/RAM telemetry at all
    eval_datasets: tuple = ()               # enabled dataset keys (EVAL_SPECS order)
    eval_profile: str = "quick"
    eval_seed: int = 42
    eval_temperature: float = 0.0
    eval_max_tokens_qa: int | None = None       # None = dynamic: model context minus prompt, per question
    eval_max_tokens_code: int | None = None
    eval_timeout: float = 600.0
    eval_extra_body: dict | None = None     # merged into every eval request (e.g. disable "thinking")
    eval_cache_dir: Path = SCRIPT_DIR / "tests"
    eval_code_sandbox: str = "auto"         # auto | docker | subprocess
    eval_code_timeout: float = 10.0         # seconds per execution of generated code
    eval_code_mem_mb: int = 2048
    eval_code_docker_image: str = "python:3.11-slim"
    eval_lcb_max_tests: int = 20            # 0 = every test
    eval_max_consec_errors: int = 5
    agent_sec_ref: float = 60.0             # AGENT_SEC_REF: s/q at which the agent quality is halved
    log_enabled: bool = True                # LOG: also write everything printed to results/benchmark_<stamp>.log
    csv_enabled: bool = True                # CSV: write benchmark_<stamp>.csv and append to the history CSV
    eval_retry_on_length: bool = True       # cut off / timed out -> retry the item once with thinking disabled
    eval_sandbox_resolved: str = ""         # set at run time: "docker" or "subprocess"
    eval_info: dict | None = None           # {dataset: {"rows": sampled, "available": total}} for the dashboard

    @property
    def chat_url(self): return f"{self.base_url}/v1/chat/completions"
    @property
    def models_url(self): return f"{self.base_url}/v1/models"


def _output_dir() -> Path:
    """OUTPUT_DIR (default: results/ next to the script). Relative paths are relative to the script."""
    out = Path(_env("OUTPUT_DIR", "results")).expanduser()
    return out if out.is_absolute() else SCRIPT_DIR / out


def _eval_cache_dir() -> Path:
    """EVAL_CACHE_DIR (default ./tests next to the script). Relative paths are relative to the script."""
    p = Path(_env("EVAL_CACHE_DIR", "tests")).expanduser()
    return p if p.is_absolute() else SCRIPT_DIR / p


def _eval_extra_body():
    raw = _env("EVAL_EXTRA_BODY")
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except ValueError as e:
        sys.exit(f"[config] EVAL_EXTRA_BODY must be a JSON object, e.g. "
                 f'{{"chat_template_kwargs": {{"enable_thinking": false}}}} ({e})')
    if not isinstance(obj, dict):
        sys.exit("[config] EVAL_EXTRA_BODY must be a JSON object")
    return obj


def load_config() -> Config:
    _load_dotenv()
    _export_hf_token()
    profile = (_env("EVAL_PROFILE", "quick") or "quick").lower()
    if profile not in PROFILE_ROWS:
        sys.exit(f"[config] EVAL_PROFILE must be one of {', '.join(PROFILE_ROWS)} (got {profile!r})")
    sandbox = (_env("EVAL_CODE_SANDBOX", "auto") or "auto").lower()
    if sandbox not in ("auto", "docker", "subprocess"):
        sys.exit(f"[config] EVAL_CODE_SANDBOX must be auto, docker or subprocess (got {sandbox!r})")
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
        strata_url=(_env("STRATA_URL") or _env("STRATA_SWAP_URL") or "").rstrip("/") or None,
        strata_api_key=_env("STRATA_API_KEY"),
        strata_model=_env("STRATA_MODEL"),
        strata_params_b=_env("STRATA_PARAMS_B", None, float),
        strata_active_b=_env("STRATA_ACTIVE_B", None, float),
        strata_quant=_env("STRATA_QUANT"),
        strata_ctx=_env("STRATA_CTX", None, int),
        goldilocks_vram_only=_env("GOLDILOCKS_VRAM_ONLY", False, bool),
        ram_process_pattern=_env("RAM_PROCESS_PATTERN", "llama"),
        ram_poll_ms=max(200, _env("RAM_POLL_MS", 500, int)),
        spill_threshold_gb=_env("RAM_SPILL_THRESHOLD_GB", 2.0, float),
        ram_overhead_gb=_env("RAM_OVERHEAD_GB", None, float),
        tech_prompt=_env("TECH_PROMPT", "Explain quantum computing in 3 sentences."),
        logic_prompt=_env("LOGIC_PROMPT", "Outline a 101 Biology textbook. Give 2 experiments and a 3-question quiz."),
        include_models=_env_list("INCLUDE_MODELS", allow_all=True),
        exclude_models=_env_list("EXCLUDE_MODELS"),
        run_eval_tests=_env("RUN_EVAL_TESTS", False, bool),
        run_speed_tests=_env("RUN_SPEED_TESTS", True, bool),
        eval_datasets=tuple(k for k, s in EVAL_SPECS.items() if _env(s["env"], False, bool)),
        eval_profile=profile,
        eval_seed=_env("EVAL_SEED", 42, int),
        eval_temperature=_env("EVAL_TEMPERATURE", 0.0, float),
        eval_max_tokens_qa=_env_tokens("EVAL_MAX_TOKENS_QA"),
        eval_max_tokens_code=_env_tokens("EVAL_MAX_TOKENS_CODE"),
        eval_timeout=_env("EVAL_TIMEOUT", 600.0, float),
        eval_extra_body=_eval_extra_body(),
        eval_cache_dir=_eval_cache_dir(),
        eval_code_sandbox=sandbox,
        eval_code_timeout=_env("EVAL_CODE_TIMEOUT", 10.0, float),
        eval_code_mem_mb=_env("EVAL_CODE_MEM_MB", 2048, int),
        eval_code_docker_image=_env("EVAL_CODE_DOCKER_IMAGE", "python:3.11-slim"),
        eval_lcb_max_tests=max(0, _env("EVAL_LCB_MAX_TESTS", 20, int)),
        agent_sec_ref=max(1.0, _env("AGENT_SEC_REF", 60.0, float)),
        log_enabled=_env("LOG", True, bool),
        csv_enabled=_env("CSV", True, bool),
        eval_retry_on_length=_env("EVAL_RETRY_ON_LENGTH", True, bool),
    )


class BenchError(Exception):
    """A recoverable per-model failure (timeout, HTTP error, malformed response...)."""


class BenchTimeout(BenchError):
    """The request timed out (an eval question that thinks for too long ends up here)."""


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
        self.ram = RamSampler(cfg)
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

    @staticmethod
    def _mem_total_gb():
        try:
            m = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
            return float(m["MemTotal"].split()[0]) / 1024 ** 2
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


def parse_launch_args(argv) -> dict:
    """Read the llama-server launch flags that decide what lives in system RAM instead of VRAM."""
    toks = []
    for a in argv:
        toks.extend(a.split("=", 1) if a.startswith("--") and "=" in a else [a])
    info = {"ngl": None, "n_cpu_moe": None, "cpu_moe": False, "no_kv_offload": False, "ot_cpu": False, "ctx": None}
    for i, a in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        try:
            if a in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
                info["ngl"] = 999 if nxt.lower() == "all" else int(nxt)
            elif a in ("--n-cpu-moe", "-ncmoe"):
                info["n_cpu_moe"] = int(nxt)
            elif a in ("--cpu-moe", "-cmoe"):
                info["cpu_moe"] = True
            elif a in ("--no-kv-offload", "-nkvo"):
                info["no_kv_offload"] = True
            elif a in ("-ot", "--override-tensor"):
                info["ot_cpu"] = info["ot_cpu"] or "cpu" in nxt.lower()
            elif a in ("-c", "--ctx-size"):
                info["ctx"] = int(nxt)
        except ValueError:
            pass
    return info


def offload_summary(info) -> str:
    """Human-readable list of CPU-offload flags; '' means the model was launched fully on the GPU."""
    parts = []
    if info["n_cpu_moe"]:
        parts.append(f"n-cpu-moe {info['n_cpu_moe']}")
    if info["cpu_moe"]:
        parts.append("cpu-moe")
    if info["ngl"] is not None and info["ngl"] < 99:
        parts.append(f"ngl {info['ngl']}")
    if info["no_kv_offload"]:
        parts.append("KV on CPU")
    if info["ot_cpu"]:
        parts.append("override-tensor CPU")
    return ", ".join(parts)


def find_listener_pids(port) -> set:
    """PIDs of processes LISTENing on a TCP port, whatever they are called (needs root to see other users' fds)."""
    inodes = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(parts[9])
    pids = set()
    if not inodes:
        return pids
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{d}/fd"):
                try:
                    link = os.readlink(f"/proc/{d}/fd/{fd}")
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    pids.add(int(d))
                    break
        except OSError:
            continue
    return pids


class RamSampler(threading.Thread):
    """Every RAM_POLL_MS: system RAM in use (MemTotal - MemAvailable, the figure htop / Home Assistant show)
    and the resident memory (RSS) of the inference-server process(es), matched by executable name
    (RAM_PROCESS_PATTERN). The process RSS is the best available estimate of how much of the model and
    cache lives in system RAM instead of VRAM. Processes in Docker are visible from the host's /proc."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.interval = cfg.ram_poll_ms / 1000
        self.pattern = re.compile(cfg.ram_process_pattern)
        self.page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        self.samples, self.pids, self._scanned = [], set(), 0.0
        self.listen_port = None      # set in Strata mode: find the server by the port it listens on
        self._halt, self._lock = threading.Event(), threading.Lock()

    def find_processes(self) -> set:
        if self.listen_port:      # Strata: no known executable name, so use the listening socket
            def comm(p):
                try:
                    return Path(f"/proc/{p}/comm").read_text().strip()
                except OSError:
                    return ""
            pids = {p for p in find_listener_pids(self.listen_port) if comm(p) != "docker-proxy"}
            if pids:
                return pids
        found = set()
        try:
            names = os.listdir("/proc")
        except OSError:
            return found
        for d in names:
            if not d.isdigit() or int(d) == os.getpid():
                continue
            try:
                argv0 = Path(f"/proc/{d}/cmdline").read_bytes().split(b"\0")[0].decode("utf-8", "ignore")
            except OSError:
                continue
            if argv0 and self.pattern.search(os.path.basename(argv0)):
                found.add(int(d))
        return found

    def launch_info(self):
        """Launch flags of the (largest) inference-server process, or None if unreadable."""
        with self._lock:
            pids = list(self.pids)
        best = max(pids, key=lambda p: self.rss_gb([p]) or 0, default=None)
        if best is None:
            return None
        try:
            argv = Path(f"/proc/{best}/cmdline").read_bytes().decode("utf-8", "ignore").split("\0")
        except OSError:
            return None
        return parse_launch_args([a for a in argv[1:] if a])

    def rescan(self):
        found = self.find_processes()
        with self._lock:
            self.pids, self._scanned = found, time.time()

    def rss_gb(self, pids=None):
        with self._lock:
            pids = list(self.pids) if pids is None else pids
        total, seen = 0, False
        for pid in pids:
            try:
                total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * self.page
                seen = True
            except (OSError, ValueError, IndexError):
                with self._lock:
                    self.pids.discard(pid)
        return total / 1024 ** 3 if seen else None

    def run(self):
        while not self._halt.is_set():
            if time.time() - self._scanned > 3:
                self.rescan()
            self.samples.append((time.time(), {"ram_gb": HostMeter._mem_gb(), "proc_gb": self.rss_gb()}))
            self._halt.wait(self.interval)

    def window(self, t0, t1):
        got = [s for t, s in self.samples if t0 <= t <= t1]
        return got or ([self.samples[-1][1]] if self.samples else [])

    def stop(self): self._halt.set()


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
            raise BenchTimeout(f"timeout after {timeout:.0f}s")
        except requests.RequestException as e:
            raise BenchError(f"connection error: {e}")
        if resp.status_code == 503 and attempt < cfg.retries:
            time.sleep(2)
            continue
        if resp.status_code in (401, 403):
            raise BenchError(f"HTTP {resp.status_code}: authentication failed - check the API key for "
                             f"{cfg.backend} ({resp.text[:120]})")
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


_STRATA_PATHS = ["/api/about", "/api/info", "/api/status", "/api/stats", "/api/system", "/api/monitor",
                 "/api/model", "/api/models", "/api/config", "/api/props", "/api/hardware", "/api/version",
                 "/props", "/health", "/v1/models"]
# Never call anything that could change state or expose chats, even with GET.
_STRATA_SKIP = re.compile(r"unload|load|stop|shutdown|delete|reset|clear|restart|kill|chat|complet|messages|"
                          r"embed|history|conversation|session|log|stream|event|request", re.I)
_STRATA_INFO = re.compile(r"about|info|model|system|config|status|health|version|props|hardware|engine", re.I)


def strata_discover(cfg, session) -> dict:
    """Find what Strata's own web UI reads: endpoints referenced in its JavaScript plus common guesses.
    GET only, state-changing / chat endpoints are skipped. Returns {path: json | {'_status': code}}."""
    from urllib.parse import urljoin
    paths = list(_STRATA_PATHS)
    try:
        html = session.get(cfg.strata_url + "/", timeout=5).text
        for src in re.findall(r'(?:src|href)=["\']([^"\']+\.js)["\']', html)[:8]:
            js = session.get(urljoin(cfg.strata_url + "/", src), timeout=10).text
            for m in re.findall(r'["\'`](/(?:api|v1)/[A-Za-z0-9_\-/]+)["\'`]', js) + \
                     re.findall(r'fetch\(\s*["\'`](/[A-Za-z0-9_\-/]+)["\'`]', js):
                if m not in paths:
                    paths.append(m)
    except requests.RequestException:
        pass
    found = {}
    for path in paths[:40]:
        if path != "/v1/models" and _STRATA_SKIP.search(path):
            continue
        try:
            r = session.get(cfg.strata_url + path, timeout=5)
        except requests.RequestException:
            continue
        if r.status_code >= 400:
            found[path] = {"_status": r.status_code}
            continue
        try:
            found[path] = r.json()
        except ValueError:
            found[path] = {"_text": r.text[:500]}
    return found


def _scalars(obj, prefix="", out=None, depth=0):
    out = out if out is not None else []
    if len(out) >= 60 or depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scalars(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:6]):
            _scalars(v, f"{prefix}[{i}]", out, depth + 1)
    elif isinstance(obj, (str, int, float, bool)) and not (isinstance(obj, str) and len(obj) > 120):
        out.append([prefix, obj])
    return out


def strata_details(found: dict) -> list:
    """Scalar values from the info-like endpoints, for the dashboard's 'Strata details' panel."""
    out = []
    for path, data in found.items():
        if path.startswith("/v1/") or not _STRATA_INFO.search(path) or not isinstance(data, dict) or "_status" in data:
            continue
        rows = _scalars(data)
        if rows:
            out.append([path, rows])
    return out


def ctx_from_details(details):
    """Context size (tokens) from Strata's info endpoints, if a plausible field is present."""
    pat = re.compile(r"(?:^|\.)(?:n_ctx|ctx_size|ctx|context_size|context_length|context_tokens|context|max_context)$", re.I)
    for _, rows in details or []:
        for k, v in rows:
            if pat.search(k) and isinstance(v, (int, float)) and 512 <= v <= 4_194_304:
                return int(v)
    return None


def probe_strata(cfg):
    """--probe-strata: list what Strata's API exposes and save the raw JSON for inspection."""
    if not cfg.strata_url:
        sys.exit("[error] Set STRATA_URL in .env first.")
    cfg.backend, cfg.base_url = "strata", cfg.strata_url
    found = strata_discover(cfg, make_session(cfg, "strata"))
    for path, data in found.items():
        if isinstance(data, dict) and "_status" in data:
            print(f"  GET {path} -> HTTP {data['_status']}")
        else:
            keys = list(data)[:14] if isinstance(data, dict) else f"list of {len(data)}"
            print(f"  GET {path} -> OK, {keys}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "strata_probe.json"
    text = json.dumps(found, indent=1)
    path.write_text(text if len(text) < 300_000 else text[:300_000], encoding="utf-8")
    print(f"\nRaw responses saved to {path}. Review it before sharing; it is data from your Strata server.")


def get_ctx(cfg, session, model):
    """Active context size (tokens) of the loaded model. Primary: llama-server's /props, reached
    through llama-swap's /upstream/<model>/ route. Fallback: a -c/--ctx-size flag in /running (if shown)."""
    from urllib.parse import quote
    if cfg.backend == "strata":
        if cfg.strata_ctx or cfg.strata_ctx_detected:
            return cfg.strata_ctx or cfg.strata_ctx_detected
        try:   # works if Strata is a llama-server; otherwise set STRATA_CTX
            j = session.get(f"{cfg.base_url}/props", timeout=10).json()
            n = (j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx")
            return int(n) if n else None
        except (requests.RequestException, ValueError, AttributeError):
            return None
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
    rw = host.ram.window(t0, t1)
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
            "cpu_w": cpu_w, "cpu_util": h.get("cpu_util"),
            "ram_gb": _max(_col(rw, "ram_gb")), "proc_gb": _max(_col(rw, "proc_gb"))}


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

EVAL_FIELDS = [f"eval_{k}_{s}" for k in EVAL_SPECS for s in ("pct", "n", "err", "na", "fallback", "sec", "trunc")] + ["eval_avg_pct", "eval_avg_sec", "agent_quality"]
FIELDS = ["timestamp", "model", "status", "error", "params_b", "active_b", "arch", "offload", "quant", "ctx_size", "load_s",
          "tech_tps", "logic_tps", "avg_tps", "prefill_tps", "prefill_tokens", "cache_hit_pct", "cache_speedup",
          "ttft_ms", "gen_ms", "latency_s", "gpu_w", "gpu_peak_w", "idle_gpu_w", "gpu_util", "gpu_temp",
          "vram_gb", "vram_delta_gb", "vram_total_gb", "vram_pct", "cpu_w", "cpu_util",
          "ram_total_gb", "ram_used_gb", "ram_pct", "ram_idle_gb", "ram_delta_gb", "ram_proc_gb", "ram_spill_gb", "ram_overhead_gb",
          "energy_wh", "j_per_token", "cost_1m_peak", "cost_1m_off", "cost_1kq_peak", "cost_1kq_off",
          "tps_per_watt", "goldilocks", "score_chat", "score_agent", "eligible"] + EVAL_FIELDS


def blank(name):
    r = dict.fromkeys(FIELDS)
    r.update(model=name, status="ok", error="", eligible=False,
             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return r


def bench_model(cfg, session, gpu, host, ha, model, unload_state) -> dict:
    r = blank(model)
    strata = cfg.backend == "strata"
    if strata:      # no filename to parse, so take the model facts from .env
        r.update(params_b=cfg.strata_params_b, active_b=cfg.strata_active_b,
                 quant=cfg.strata_quant or parse_quant(cfg.strata_model_id or model))
    try:
        # 1. clean slate + idle baseline, so VRAM / power can be attributed to this model.
        #    Strata serves one always-loaded model: nothing to unload, so no load time / VRAM delta.
        unloaded = False if strata else unload_all(cfg, session)
        if not strata and not unloaded and not unload_state["warned"]:
            unload_state["warned"] = True
            print("  [warn] could not unload models via the API; load time and VRAM delta will be skipped")
        time.sleep(min(cfg.settle, 1.0) if strata else cfg.settle)
        now = time.time()
        base = gpu.window(now - max(0.5, min(3.0, cfg.settle)), now) if gpu.ok else []
        r["idle_gpu_w"] = _avg(_col(base, "gpu_w"))
        base_vram = _avg(_col(base, "vram_gb"))
        ram_idle = _avg(_col(host.ram.window(now - max(0.5, min(3.0, cfg.settle)), now), "ram_gb"))

        # 2. cold load (also the warm-up)
        t = time.time()
        post_chat(cfg, session, {"model": model, "max_tokens": 8, "stream": False,
                                 "messages": [{"role": "user", "content": "hi"}]}, cfg.timeout_load)
        r["load_s"] = (time.time() - t) if unloaded else None
        host.ram.rescan()                       # the server process (re)started during the load
        r["ctx_size"] = get_ctx(cfg, session, model)
        if not strata:          # flags are only meaningful for llama.cpp children launched by llama-swap
            launch = host.ram.launch_info()
            if launch:
                r["offload"] = offload_summary(launch) or "none"
                r["ctx_size"] = r["ctx_size"] or launch["ctx"]
        if r["ctx_size"] is None and not unload_state.get("ctx_warned"):
            unload_state["ctx_warned"] = True
            print("  [warn] could not read the context size ("
                  + ("set STRATA_CTX in .env)" if strata else "tried /upstream/<model>/props and /running)"))

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
    r["ram_used_gb"], r["ram_proc_gb"] = _max(_col(every, "ram_gb")), _max(_col(every, "proc_gb"))
    r["ram_total_gb"] = HostMeter._mem_total_gb()
    if r["ram_used_gb"] is not None and ram_idle is not None and unloaded:
        r["ram_idle_gb"], r["ram_delta_gb"] = ram_idle, max(0.0, r["ram_used_gb"] - ram_idle)
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
    # Typical server RAM of models launched fully on the GPU (CUDA context, pinned buffers, leftover mapped pages)
    ov = [r["ram_proc_gb"] for r in results if r["status"] == "ok" and r["offload"] == "none"
          and r["ram_proc_gb"] is not None]
    overhead = cfg.ram_overhead_gb if cfg.ram_overhead_gb is not None else _median(ov)
    for r in results:
        r["params_b"] = r["params_b"] or parse_params(r["model"])
        r["quant"] = r["quant"] or parse_quant(r["model"])
        r["active_b"] = r["active_b"] or parse_active(r["model"])
        r["arch"] = r["arch"] or ("MoE" if r["active_b"] else "Dense")
        if r["status"] != "ok":
            continue
        r["avg_tps"] = _avg([x for x in (r["tech_tps"], r["logic_tps"]) if x is not None])
        pcts = [r.get(f"eval_{k}_pct") for k in EVAL_SPECS if r.get(f"eval_{k}_pct") is not None]
        r["eval_avg_pct"] = _avg(pcts) if pcts else None
        secs = [r.get(f"eval_{k}_sec") for k in EVAL_SPECS if r.get(f"eval_{k}_sec") is not None]
        r["eval_avg_sec"] = _avg(secs) if secs else None          # mean seconds per question over the datasets run
        total = r["vram_total_gb"] or cfg.vram_limit_gb
        r["vram_total_gb"] = total
        if r["vram_gb"] is not None and total:
            r["vram_pct"] = r["vram_gb"] / total * 100
        if r["ram_used_gb"] is not None and r["ram_total_gb"]:
            r["ram_pct"] = r["ram_used_gb"] / r["ram_total_gb"] * 100
        # RAM spill estimate
        r["ram_overhead_gb"] = overhead
        if r["offload"] == "none":                       # launched fully on the GPU: nothing is meant to spill
            r["ram_spill_gb"] = 0.0
        elif r["offload"]:                               # CPU-offload flags: server RAM above the typical overhead
            src = r["ram_proc_gb"] if r["ram_proc_gb"] is not None else r["ram_delta_gb"]
            if src is not None:
                r["ram_spill_gb"] = max(0.0, src - (overhead or 0.0))
        else:                                            # launch flags unknown (e.g. Strata): size threshold
            src = r["ram_proc_gb"] if r["ram_proc_gb"] is not None else r["ram_delta_gb"]
            if src is not None:
                r["ram_spill_gb"] = src if src >= cfg.spill_threshold_gb else 0.0
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
            fits = r["vram_pct"] is None or r["vram_pct"] <= cfg.goldilocks_max_vram_pct
            r["eligible"] = fits and not (cfg.goldilocks_vram_only and (r["ram_spill_gb"] or 0) > 0)
        # Category scores for the dashboard podium (need speed + VRAM data; the podium only ranks eligible models).
        if r["avg_tps"] and r["vram_pct"] is not None:
            vram_room = max(0.0, 100 - r["vram_pct"])           # headroom left on the GPU, 0-100
            r["score_chat"] = r["avg_tps"] * vram_room
            # Quality = average eval score, lowered by how long a question takes (s/q, retries and timeouts included):
            # quality = eval_avg / (1 + s_per_q / AGENT_SEC_REF). 0 s keeps 100 %, AGENT_SEC_REF seconds keeps 50 %.
            # Without evals the parameter count stands in as a proxy (no time adjustment is possible).
            if r.get("eval_avg_pct") is not None:
                quality = r["eval_avg_pct"]
                if r.get("eval_avg_sec") is not None:
                    quality /= 1 + r["eval_avg_sec"] / cfg.agent_sec_ref
            else:
                quality = r["params_b"]
            if quality is not None:
                r["agent_quality"] = quality
                r["score_agent"] = quality * vram_room * math.log(max(1.1, r["avg_tps"]))
    return results


def _rounded(r): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def write_csv(results, path, append=False):
    new = not (append and Path(path).exists())
    if not new:
        # The column set grew (eval columns): re-write an older history file under the new header first,
        # otherwise the appended rows would sit under the wrong columns. A .bak copy is kept.
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            old_fields = rd.fieldnames or []
            old_rows = list(rd) if old_fields != FIELDS else None
        if old_rows is not None:
            shutil.copy2(path, str(path) + ".bak")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(old_rows)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(_rounded(r) for r in results)


def make_session(cfg, backend):
    """HTTP session with the right credentials. The Strata auth scheme is not known here, so the key is
    sent both as 'Authorization: Bearer' and 'x-api-key' (servers ignore the header they don't use)."""
    s = requests.Session()
    key = cfg.strata_api_key if backend == "strata" else cfg.api_key
    if key:
        s.headers["Authorization"] = f"Bearer {key}"
        if backend == "strata":
            s.headers["x-api-key"] = key
    return s


def detect_backend(cfg):
    """Use llama-swap if it answers; otherwise fall back to a single Strata endpoint (if STRATA_URL is set).
    Returns (session, model_ids) and sets cfg.backend / cfg.base_url."""
    session, why = make_session(cfg, "llama-swap"), ""
    try:
        r = session.get(cfg.models_url, timeout=5)
        r.raise_for_status()
        ids = select_models(cfg, [m["id"] for m in r.json().get("data", [])])
        if ids:
            return session, ids
        why = "no model is left after MODEL_FILTER / INCLUDE_MODELS / EXCLUDE_MODELS, or the server lists none"
    except (requests.RequestException, ValueError, KeyError) as e:
        why = e.__class__.__name__
    if not cfg.strata_url:
        sys.exit(f"[error] llama-swap at {cfg.base_url} is unavailable ({why}). "
                 "Set STRATA_URL in .env to fall back to a single Strata endpoint.")

    print(f"[info] llama-swap unavailable ({why}); falling back to Strata at {cfg.strata_url}")
    cfg.backend, cfg.base_url = "strata", cfg.strata_url
    session = make_session(cfg, "strata")
    reachable, name = False, cfg.strata_model
    for path in ("/health", "/v1/models"):
        try:
            r = session.get(cfg.strata_url + path, timeout=5)
            reachable = True
            if path == "/v1/models" and r.ok:
                data = r.json().get("data") or []
                cfg.strata_model_id = data[0].get("id") if data and isinstance(data[0], dict) else None
                name = name or cfg.strata_model_id
        except (requests.RequestException, ValueError):
            pass
    if not reachable:
        sys.exit(f"[error] Strata at {cfg.strata_url} is not reachable either.")
    chosen = select_models(cfg, [name or "strata"])
    if not chosen:
        sys.exit("[error] The Strata model is excluded by INCLUDE_MODELS / EXCLUDE_MODELS / MODEL_FILTER.")
    return session, chosen


# --------------------------------------------------------------------------- #
# Model discovery and selection  (--list-models, INCLUDE_MODELS / EXCLUDE_MODELS)
# --------------------------------------------------------------------------- #

def _name_match(pattern, name):
    """Case-insensitive exact match; '*' and '?' act as wildcards (e.g. 'qwen*')."""
    p, n = pattern.lower(), name.lower()
    return fnmatch.fnmatchcase(n, p) if any(c in p for c in "*?") else p == n


def model_skip_reason(cfg, name):
    """None if the model is selected, otherwise a short reason why it is skipped."""
    if cfg.model_filter and not re.search(cfg.model_filter, name):
        return "no match for MODEL_FILTER"
    if cfg.include_models and not any(_name_match(p, name) for p in cfg.include_models):
        return "not in INCLUDE_MODELS"
    if any(_name_match(p, name) for p in cfg.exclude_models):
        return "excluded by EXCLUDE_MODELS"
    return None


def select_models(cfg, ids):
    """Apply MODEL_FILTER (regex), INCLUDE_MODELS and EXCLUDE_MODELS to the ids a server offers."""
    chosen = [m for m in ids if model_skip_reason(cfg, m) is None]
    unknown = [p for p in cfg.include_models if not any(_name_match(p, m) for m in ids)]
    if unknown and ids:
        print(f"  [warn] INCLUDE_MODELS entries not offered by the server: {', '.join(unknown)} "
              "(run --list-models to see the exact ids)")
    return chosen


def list_models(cfg):
    """--list-models: print what GET /v1/models returns (llama-swap, else Strata) and which are selected."""
    attempts = [("llama-swap", cfg.base_url)]
    if cfg.strata_url and cfg.strata_url != cfg.base_url:
        attempts.append(("Strata", cfg.strata_url))
    ids, errors = None, []
    for backend, base in attempts:
        session = make_session(cfg, "strata" if backend == "Strata" else "llama-swap")
        try:
            r = session.get(f"{base}/v1/models", timeout=10)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]
            print(f"Models at {base}/v1/models ({backend}):\n")
            break
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            errors.append(f"{base}/v1/models: {e.__class__.__name__}: {e}")
    if ids is None:
        sys.exit("[error] Could not list models.\n  " + "\n  ".join(errors))
    if not ids:
        print("  (the server lists no models)")
        return
    width = max(len(m) for m in ids)
    picked = 0
    for m in ids:
        why = model_skip_reason(cfg, m)
        picked += why is None
        print(f"  [{'x' if why is None else ' '}] {m:<{width}}" + (f"   <- skipped: {why}" if why else ""))
    print(f"\n{picked} of {len(ids)} selected for benchmarking "
          "([x] = will run). Narrow with INCLUDE_MODELS / EXCLUDE_MODELS in .env.")


# --------------------------------------------------------------------------- #
# Evaluations: dataset-based accuracy (QA) and pass@1 (code)
# --------------------------------------------------------------------------- #
# Flow: the datasets are loaded and sampled once (same rows for every model), then each model is asked
# every row over the API. QA answers are parsed to a letter and string-matched; generated code is
# extracted from the reply and executed against the dataset's tests in a sandbox (see _sandbox()).

def load_eval_split(cfg, key, offline_first=True):
    """Load one dataset split with cache_dir=cfg.eval_cache_dir (./tests). Normal runs try the local cache
    first (no network, milliseconds) and only download if the dataset is not cached yet."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[eval] The 'datasets' package is missing. Run: pip install -r requirements.txt")
    spec, cache = EVAL_SPECS[key], str(cfg.eval_cache_dir)
    ds = None
    if offline_first:
        try:
            from datasets import DownloadConfig
            ds = load_dataset(spec["repo"], cache_dir=cache, download_config=DownloadConfig(local_files_only=True))
        except Exception:
            ds = None
    if ds is None:
        if offline_first:
            print(f"  [eval] {spec['label']} is not in {cache} yet - downloading it now "
                  "(run --download-datasets next time to do this up front)")
        ds = load_dataset(spec["repo"], cache_dir=cache)
    if isinstance(ds, dict):                       # DatasetDict
        ds = ds[spec["split"]] if spec["split"] in ds else ds[next(iter(ds))]
    return ds


def download_datasets(cfg):
    """--download-datasets: cache every dataset marked true in .env into ./tests and exit."""
    if not cfg.eval_datasets:
        print("No datasets are enabled. Set at least one of EVAL_GPQA, EVAL_MMLU_PRO, EVAL_HUMANEVAL, "
              "EVAL_MBPP, EVAL_LIVECODEBENCH to true in .env.")
        return
    cfg.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching {len(cfg.eval_datasets)} dataset(s) into {cfg.eval_cache_dir}")
    failed = []
    for key in cfg.eval_datasets:
        spec = EVAL_SPECS[key]
        print(f"  {spec['label']} ({spec['repo']}) ...", flush=True)
        if key == "livecodebench":
            print("    note: LiveCodeBench is large (about 4 GB); this can take a while")
        try:
            ds = load_eval_split(cfg, key, offline_first=False)
            print(f"    ok: {len(ds)} rows in split '{spec['split']}'")
        except SystemExit:
            raise
        except Exception as e:
            failed.append(key)
            print(f"    FAILED: {e.__class__.__name__}: {e}")
    if failed:
        sys.exit(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
    print("\nDone. Normal runs now read these from local disk.")


def sample_indices(n, k, seed, key):
    """Deterministic sample, so every model sees exactly the same rows."""
    if k is None or k >= n:
        return list(range(n))
    return sorted(random.Random(f"{seed}:{key}").sample(range(n), k))


# ----- prompts --------------------------------------------------------------

_QA_FORMAT = ("Think step by step, then finish your reply with a final line in exactly this form:\n"
              "Answer: X\nwhere X is a single letter ({letters}).")


def prompt_gpqa(row):
    # The dataset's question text already contains its four options.
    return (f"Answer the following multiple choice question. The options are labelled A, B, C and D.\n"
            f"{_QA_FORMAT.format(letters='A, B, C or D')}\n\n{row['question'].strip()}")


def prompt_mmlu_pro(row):
    letters = string.ascii_uppercase[:len(row["options"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    cat = (row.get("category") or "general knowledge").strip()
    return (f"The following is a multiple choice question about {cat}.\n"
            f"{_QA_FORMAT.format(letters=', '.join(letters[:-1]) + ' or ' + letters[-1])}\n\n"
            f"Question: {row['question'].strip()}\nOptions:\n{opts}")


def prompt_humaneval(row):
    return ("Complete the following Python function. Reply with the complete function (signature included, plus any "
            "imports it needs) in a single ```python code block. Do not include tests or example usage.\n\n"
            f"```python\n{row['prompt'].rstrip()}\n```")


def prompt_mbpp(row):
    tests = "\n".join(row["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {row['text'].strip()} "
            f"Your code should pass these tests:\n\n{tests}\n\n"
            "Reply with the complete Python code in a single ```python code block. Do not include the tests.")


def prompt_lcb(row, mode, func):
    if mode == "call":
        fmt = ("Use the following starter code and complete it:\n```python\n"
               f"{(row.get('starter_code') or '').rstrip()}\n```")
    else:
        fmt = "Read the input from standard input and write the answer to standard output."
    return ("You are an expert competitive programmer. Solve the following problem in Python 3.\n\n"
            f"### Problem: {row['question_title']}\n{row['question_content'].strip()}\n\n### Format\n{fmt}\n\n"
            "Reply with the complete solution in a single ```python code block.")


# ----- item builders --------------------------------------------------------

def _lcb_mode(row):
    """('call', func_name) for LeetCode-style problems, ('stdin', None) for AtCoder/Codeforces ones."""
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except ValueError:
        meta = {}
    func = meta.get("func_name") if isinstance(meta, dict) else None
    if not func and (row.get("starter_code") or "").strip():
        m = re.search(r"def\s+(\w+)\s*\(\s*self", row["starter_code"])
        func = m.group(1) if m else None
    return ("call", func) if func and (row.get("starter_code") or "").strip() else ("stdin", None)


def build_items(cfg, key, ds, idxs):
    items = []
    if key == "livecodebench":
        # keep the (huge, compressed) private tests out of memory: read only the light columns here
        light = ds.select_columns(["question_title", "question_content", "starter_code", "metadata",
                                   "public_test_cases", "question_id"])
        for i in idxs:
            row = light[i]
            mode, func = _lcb_mode(row)
            items.append({"id": str(row["question_id"]), "prompt": prompt_lcb(row, mode, func), "row": row,
                          "ds": ds, "idx": i, "mode": mode, "func": func, "tests": None})
        return items
    for i in idxs:
        row = ds[i]
        if key == "gpqa":
            gold = str(row["answer"]).strip().upper()[:1]
            items.append({"id": f"gpqa-{i}", "prompt": prompt_gpqa(row), "gold": gold, "letters": "ABCD"})
        elif key == "mmlu_pro":
            letters = string.ascii_uppercase[:len(row["options"])]
            gold = str(row.get("answer") or "").strip().upper()[:1]
            if gold not in letters and row.get("answer_index") is not None:
                gold = letters[int(row["answer_index"])]
            items.append({"id": str(row.get("question_id", i)), "prompt": prompt_mmlu_pro(row), "gold": gold,
                          "letters": letters})
        elif key == "humaneval":
            items.append({"id": row["task_id"], "prompt": prompt_humaneval(row), "row": row})
        elif key == "mbpp":
            items.append({"id": f"mbpp-{row['task_id']}", "prompt": prompt_mbpp(row), "row": row})
    return items


def prepare_eval_data(cfg):
    """Load + sample every enabled dataset up front, so a missing package, dataset or cache fails
    before the (long) speed test rather than after it."""
    rows, data, info = PROFILE_ROWS[cfg.eval_profile], {}, {}
    cfg.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading evaluation datasets (profile {cfg.eval_profile}: "
          f"{'all rows' if rows is None else f'{rows} rows each'}; cache {cfg.eval_cache_dir})")
    for key in cfg.eval_datasets:
        spec = EVAL_SPECS[key]
        t = time.time()
        try:
            ds = load_eval_split(cfg, key)
            idxs = sample_indices(len(ds), rows, cfg.eval_seed, key)
            data[key] = build_items(cfg, key, ds, idxs)
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"[eval] Could not load {spec['label']} ({spec['repo']}): {e.__class__.__name__}: {e}\n"
                     "        Check the network, or run: python benchmark.py --download-datasets")
        info[key] = {"rows": len(data[key]), "available": len(ds)}
        print(f"  {spec['label']}: {len(data[key])} of {len(ds)} rows ({time.time() - t:.1f}s)")
    cfg.eval_info = info
    if any(EVAL_SPECS[k]["kind"] == "code" for k in cfg.eval_datasets):
        cfg.eval_sandbox_resolved = resolve_sandbox(cfg)
    return data


# ----- response parsing -----------------------------------------------------

def strip_think(text):
    """Remove reasoning blocks so answers are only looked for in the final reply."""
    text = text or ""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    if re.search(r"</think(?:ing)?>", text, re.I):          # the opening tag was consumed by the chat template
        text = re.split(r"</think(?:ing)?>", text, flags=re.I)[-1]
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)   # truncated thinking: no answer
    return text.strip()


_KEYWORD = re.compile(r"(?i:final\s+answer|correct\s+(?:answer|option|choice)|answer|choice|option)")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?\s*([A-J])\b")
_LEAD = re.compile(r"^(?:[\s:=\-\u2192*_`\"'(\[{$\\]+|is\b|are\b|option\b|choice\b|\\?boxed\{|\\?text(?:bf)?\{)", re.I)


def _letter_after(tail, letters):
    """Letter that directly follows an 'Answer:'-style keyword, or None."""
    t = tail[:60]
    while True:
        m = _LEAD.match(t)
        if not m:
            break
        t = t[m.end():]
    m = re.match(r"([A-J])(?![A-Za-z0-9])", t)
    if not m or m.group(1) not in letters:
        return None
    if m.group(1) in "AI" and re.match(r"\s+[a-z]{2,}", t[1:]):    # "answer is A great ...", "Answer: I think"
        return None
    return m.group(1)


def extract_choice(text, letters="ABCDEFGHIJ"):
    """Final multiple-choice letter from a free-form reply, or None. Looks for (in order of trust, taking the
    LAST occurrence): 'Answer: X' / 'the answer is X' / 'correct option is X', \\boxed{X}, then a last line
    that is just the letter."""
    text = strip_think(text)
    if not text:
        return None
    hits = []
    for m in _KEYWORD.finditer(text):
        got = _letter_after(text[m.end():], letters)
        if got:
            hits.append((m.start(), got))
    for m in _BOXED.finditer(text):
        if m.group(1) in letters:
            hits.append((m.start(), m.group(1)))
    if hits:
        return max(hits)[1]
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        m = re.match(r"^\W*\(?([A-J])\)?[.):]?\W*$", lines[-1].strip())
        if m and m.group(1) in letters:
            return m.group(1)
    return None


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_CODEISH = re.compile(r"\b(def|class|import|from|print|input|return|for|while)\b|sys\.stdin")


def extract_code(text, entry=None):
    """Python source from a reply: the last fenced python block (preferring one that defines `entry`).
    Tolerates a missing closing fence (truncated replies) and replies with no fence at all."""
    text = strip_think(text)
    blocks = [(lang.lower(), body.strip("\n")) for lang, body in _FENCE.findall(text) if body.strip()]
    py = [b for lang, b in blocks if lang in ("", "python", "py", "python3", "py3")] or [b for _, b in blocks]
    if entry:
        hit = [b for b in py if re.search(rf"\bdef\s+{re.escape(entry)}\b", b)]
        if hit:
            return hit[-1]
    real = [b for b in py if _CODEISH.search(b)]
    if real:
        return real[-1]
    if py:
        return py[-1]
    return text


def strip_main_guard(code):
    """Drop top-level `if __name__ == '__main__':` blocks (example usage that may read stdin or crash).
    Used for HumanEval / MBPP only; competitive-programming solutions legitimately run from main()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def guard(n):
        return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__")
    if not any(guard(n) for n in tree.body) or not hasattr(ast, "unparse"):
        return code
    tree.body = [n for n in tree.body if not guard(n)]
    return ast.unparse(tree)


# ----- sandboxed execution of generated code ---------------------------------
# SECURITY: model-written code is untrusted. Every execution happens in a child process with a stripped
# environment (your API keys are NOT passed on), a temp working dir, a wall-clock timeout, and (POSIX)
# memory / file-size rlimits. Mode "subprocess" is NOT a security boundary against deliberately hostile
# code (it still runs as your user). Mode "docker" is: no network, read-only filesystem, no capabilities,
# memory / CPU / pids limits. EVAL_CODE_SANDBOX=auto picks Docker when it is available.

_RUNNER_PY = r'''
import json, os, subprocess, sys, tempfile
try:
    import resource
    _mem = int(os.environ.get("LLMB_MEM_MB", "2048")) * 1024 * 1024
    for _lim, _val in ((resource.RLIMIT_AS, _mem), (resource.RLIMIT_FSIZE, 64 * 1024 * 1024), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(_lim, (_val, _val))
        except (ValueError, OSError):
            pass
except ImportError:
    pass

spec = json.load(open(sys.argv[1], encoding="utf-8"))
PROG = os.path.abspath("prog.py")
SCRATCH = tempfile.mkdtemp()
TIMEOUT = float(spec["timeout"])


def run(stdin_text):
    op, ep = os.path.join(SCRATCH, "out"), os.path.join(SCRATCH, "err")
    with open(op, "wb") as fo, open(ep, "wb") as fe:
        try:
            p = subprocess.run([sys.executable, "-I", PROG], input=stdin_text.encode("utf-8", "replace"),
                               stdout=fo, stderr=fe, cwd=SCRATCH, timeout=TIMEOUT)
            rc, to = p.returncode, False
        except subprocess.TimeoutExpired:
            rc, to = -9, True
    out = open(op, "rb").read(4000000).decode("utf-8", "replace")
    err = open(ep, "rb").read(20000).decode("utf-8", "replace")
    return rc, out, err, to


def num_eq(u, v):
    if u == v:
        return True
    try:
        if u.lstrip("+-").isdigit() and v.lstrip("+-").isdigit():
            return int(u) == int(v)
        fu, fv = float(u), float(v)
    except ValueError:
        return False
    return abs(fu - fv) <= 1e-6 * max(1.0, abs(fv))


def same_text(out, exp):
    a = [l.rstrip() for l in out.strip().splitlines()]
    b = [l.rstrip() for l in exp.strip().splitlines()]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty) or not all(num_eq(u, v) for u, v in zip(tx, ty)):
            return False
    return True


def json_eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_eq(a[k], b[k]) for k in a)
    return a == b


def same_json(out, exp):
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if not lines:
        return False
    try:
        return json_eq(json.loads(lines[-1]), json.loads(exp))
    except ValueError:
        return lines[-1].strip() == exp.strip()


mode = spec["mode"]
res = {"ok": False, "reason": ""}
if mode == "script":
    rc, out, err, to = run("")
    if to:
        res["reason"] = "timeout"
    elif rc != 0:
        res["reason"] = (err.strip().splitlines() or ["exit %d" % rc])[-1][:200]
    else:
        res["ok"] = True
else:
    tests, passed = spec["tests"], 0
    res["total"] = len(tests)
    for i, t in enumerate(tests):
        rc, out, err, to = run(t["input"])
        if to:
            res["reason"] = "timeout on test %d" % i
            break
        if rc != 0:
            res["reason"] = "runtime error on test %d: %s" % (i, (err.strip().splitlines() or ["exit %d" % rc])[-1][:160])
            break
        if not (same_json(out, t["output"]) if mode == "call" else same_text(out, t["output"])):
            res["reason"] = "wrong answer on test %d" % i
            break
        passed += 1
    else:
        res["ok"] = True
    res["passed"] = passed
print("@@RESULT@@" + json.dumps(res), flush=True)
'''

_NETLESS = None          # cached probe: can we drop network access with `unshare -rn`?


def _can_unshare():
    global _NETLESS
    if _NETLESS is None:
        _NETLESS = False
        if sys.platform.startswith("linux") and shutil.which("unshare"):
            try:
                _NETLESS = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
            except (OSError, subprocess.SubprocessError):
                _NETLESS = False
    return _NETLESS


def resolve_sandbox(cfg):
    """Decide between Docker and plain subprocess (EVAL_CODE_SANDBOX) and print what will be used."""
    want = cfg.eval_code_sandbox
    if want in ("auto", "docker"):
        ok, why = False, "docker not found on PATH"
        if shutil.which("docker"):
            try:
                if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
                    why = "the docker daemon is not reachable"
                elif subprocess.run(["docker", "image", "inspect", cfg.eval_code_docker_image],
                                    capture_output=True, timeout=30).returncode == 0:
                    ok = True
                else:
                    print(f"  [eval] pulling {cfg.eval_code_docker_image} for the code sandbox ...")
                    ok = subprocess.run(["docker", "pull", cfg.eval_code_docker_image], timeout=600).returncode == 0
                    why = "could not pull the sandbox image"
            except (OSError, subprocess.SubprocessError) as e:
                why = str(e)
        if ok:
            print("  [eval] code sandbox: Docker (no network, read-only, memory/CPU/pid limits)")
            return "docker"
        if want == "docker":
            sys.exit(f"[eval] EVAL_CODE_SANDBOX=docker but {why}.")
        print(f"  [warn] Docker unavailable ({why}); generated code will run in a plain subprocess "
              "(timeout + rlimits" + (", no network" if _can_unshare() else "") + "), which is NOT a security boundary. "
              "Set EVAL_CODE_SANDBOX=docker once Docker works, or only benchmark models you trust.")
    else:
        print("  [eval] code sandbox: subprocess (timeout + rlimits" + (", no network" if _can_unshare() else "")
              + "); not a security boundary against hostile code - use EVAL_CODE_SANDBOX=docker for that")
    return "subprocess"


def _sandbox(cfg, program, mode, tests, per_test_timeout):
    """Run `program` in the sandbox. mode: 'script' (pass = exit code 0), 'stdin' or 'call' (graded per test).
    Returns (ok, reason)."""
    runs = max(1, len(tests)) if mode != "script" else 1
    total = per_test_timeout * runs + 20
    with tempfile.TemporaryDirectory(prefix="llmbench_") as td:
        os.chmod(td, 0o755)
        files = {"prog.py": program, "runner.py": _RUNNER_PY,
                 "spec.json": json.dumps({"mode": mode, "tests": tests, "timeout": per_test_timeout})}
        for name, text in files.items():
            p = os.path.join(td, name)
            with open(p, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
            os.chmod(p, 0o644)
        tmp = os.path.join(td, "tmp")
        os.makedirs(tmp, exist_ok=True)
        docker = cfg.eval_sandbox_resolved == "docker"
        name = f"llmbench-{uuid.uuid4().hex[:12]}"
        if docker:
            mem = max(256, cfg.eval_code_mem_mb)
            cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", f"{mem}m",
                   "--memory-swap", f"{mem}m", "--cpus", "1", "--pids-limit", "256", "--read-only",
                   "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                   "-e", f"LLMB_MEM_MB={mem}", "-v", f"{td}:/work:ro", "-w", "/work",
                   cfg.eval_code_docker_image, "python", "-I", "runner.py", "spec.json"]
            env = None
        else:
            env = {"PATH": os.environ.get("PATH", ""), "HOME": td, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp,
                   "LLMB_MEM_MB": str(cfg.eval_code_mem_mb), "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
            if os.name == "nt":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
            cmd = [sys.executable, "-I", "runner.py", "spec.json"]
            if _can_unshare():
                cmd = ["unshare", "-rn"] + cmd
        proc = subprocess.Popen(cmd, cwd=td, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=(os.name == "posix"))
        try:
            out, err = proc.communicate(timeout=total)
        except subprocess.TimeoutExpired:
            if docker:
                subprocess.run(["docker", "kill", name], capture_output=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
            except (OSError, ProcessLookupError):
                pass
            proc.communicate()
            return False, "sandbox timeout"
    for line in reversed(out.decode("utf-8", "replace").splitlines()):
        if line.startswith("@@RESULT@@"):
            res = json.loads(line[len("@@RESULT@@"):])
            return bool(res.get("ok")), res.get("reason", "")
    tail = (err.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1][:200]
    return False, f"sandbox error: {tail}"


# ----- grading --------------------------------------------------------------

def humaneval_program(row, code):
    code = strip_main_guard(code)
    entry = row["entry_point"]
    if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", code):
        # Keep the prompt's imports/helper functions; stub its own (unfinished) function, then let the
        # model's full definition override the stub.
        body = row["prompt"].rstrip() + "\n    pass\n\n" + code
    else:                                   # the model returned just a function body: continue the prompt
        body = row["prompt"] + code
    return body + "\n\n" + row["test"] + f"\n\ncheck({entry})\n"


def mbpp_program(row, code):
    return ((row.get("test_setup_code") or "") + "\n" + strip_main_guard(code) + "\n\n"
            + "\n".join(row["test_list"]) + "\n")


class _SafeUnpickler(pickle.Unpickler):
    """LiveCodeBench stores private tests as base64(zlib(pickle(json_string))). A pickle of a plain string
    needs no classes, so every global lookup is refused: a malicious pickle cannot run code."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _decode_lcb_private(raw):
    if not raw:
        return []
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        obj = _SafeUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).load()
        return json.loads(obj) if isinstance(obj, (str, bytes)) else obj
    except Exception:
        return []


def lcb_tests(cfg, item):
    """Public tests + (a seeded sample of) the private tests, decoded once and cached on the item."""
    if item["tests"] is None:
        pub = json.loads(item["row"].get("public_test_cases") or "[]")
        try:
            raw = item["ds"].select_columns(["private_test_cases"])[item["idx"]]["private_test_cases"]
        except Exception:
            raw = ""
        priv = _decode_lcb_private(raw)
        cap = cfg.eval_lcb_max_tests
        if cap and len(pub) + len(priv) > cap:
            pub = pub[:cap]
            priv = random.Random(f"{cfg.eval_seed}:{item['id']}").sample(priv, min(len(priv), cap - len(pub)))
        item["tests"] = [{"input": t["input"], "output": t["output"]} for t in pub + priv
                         if isinstance(t, dict) and "input" in t and "output" in t]
    return item["tests"]


_LCB_PRELUDE = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
                "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
                "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
                "from operator import *\nfrom typing import *\nfrom builtins import *\n"
                "import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, "
                "itertools, functools, operator, sys, json\n")


def lcb_program(item, code):
    if item["mode"] != "call":
        return code
    future = [l for l in code.splitlines() if l.strip().startswith("from __future__")]
    body = "\n".join(l for l in code.splitlines() if l not in future)
    return ("\n".join(future) + "\n" + _LCB_PRELUDE + body + "\n\n"
            "_args = [json.loads(_l) for _l in sys.stdin.read().split('\\n') if _l.strip() != '']\n"
            f"print(json.dumps(Solution().{item['func']}(*_args)))\n")


def grade_item(cfg, key, item, text):
    """Returns (correct, parsed_answer_or_None, reason)."""
    if EVAL_SPECS[key]["kind"] == "qa":
        pred = extract_choice(text, item["letters"])
        if pred is None:
            return False, None, "no answer found"
        return pred == item["gold"], pred, ("" if pred == item["gold"] else f"expected {item['gold']}")
    t = cfg.eval_code_timeout
    if key == "humaneval":
        code = extract_code(text, item["row"]["entry_point"])
        ok, why = _sandbox(cfg, humaneval_program(item["row"], code), "script", [], t)
    elif key == "mbpp":
        code = extract_code(text)
        ok, why = _sandbox(cfg, mbpp_program(item["row"], code), "script", [], t)
    else:
        code = extract_code(text)
        tests = lcb_tests(cfg, item)
        if not tests:
            return False, None, "no test cases available"
        ok, why = _sandbox(cfg, lcb_program(item, code), item["mode"], tests, t)
    return ok, None, why


# ----- running a model through the datasets ------------------------------------

def dynamic_max_tokens(cfg, kind, model_ctx, prompt):
    """max_tokens for ONE question: the model's context size minus the prompt size (dynamic_max).
    Falls back to FALLBACK_TOKENS when the context is unknown or the sum does not work out. If the user set
    EVAL_MAX_TOKENS_QA / _CODE explicitly, that number is used as an upper cap instead of the full dynamic_max."""
    cap = cfg.eval_max_tokens_qa if kind == "qa" else cfg.eval_max_tokens_code
    try:
        # No tokenizer is available here, so the prompt is estimated at ~1 token per 3 characters plus 64 for the
        # chat template. That slightly over-estimates, which keeps prompt + reply inside the context window.
        avail = int(model_ctx) - (len(prompt) // 3 + 64)
    except (TypeError, ValueError):
        avail = 0
    if avail <= 0:
        return cap or FALLBACK_TOKENS
    return min(cap, avail) if cap else avail


def tokens_label(v):
    return str(v) if v else "dynamic"


def no_thinking(payload):
    """A COPY of `payload` with thinking switched off (Qwen3-style chat template kwarg), merged with any
    chat_template_kwargs already in the payload. The original is never modified, so the switch cannot leak
    into the next question's first attempt."""
    p = dict(payload)
    p["chat_template_kwargs"] = {**(payload.get("chat_template_kwargs") or {}), "enable_thinking": False}
    return p


def eval_dataset(cfg, session, model, key, items, sink, model_ctx=None):
    """Ask `model` every item of one dataset. API failures count as wrong answers; if nothing at all came back
    the score is left empty (None) instead of a misleading 0 %.
    Per question: max_tokens = model_ctx - prompt (see dynamic_max_tokens). If the reply is cut off by that limit,
    or the request times out, the SAME question is asked once more with thinking disabled ("No Thinking")."""
    spec = EVAL_SPECS[key]
    timeout = max(cfg.eval_timeout, cfg.timeout_load)
    extra_kw = (cfg.eval_extra_body or {}).get("chat_template_kwargs")
    already_off = isinstance(extra_kw, dict) and extra_kw.get("enable_thinking") is False
    can_retry = cfg.eval_retry_on_length and not already_off      # nothing to switch off if the user did it already
    # n = every question asked (the score's denominator). na = the subset with no usable answer because the reply
    # was cut off by the token limit (even after the retry): they count as wrong, but are tracked separately so
    # "wrong" and "no answer" can be told apart. fallback = questions that needed the No Thinking retry.
    n = correct = errors = trunc = consec = na = fallback = fb_timeouts = 0
    aborted, t0 = False, time.time()
    live = sys.stdout.isatty()          # live counter only on a terminal; logs/cron just get the final line

    def ask(payload):
        """One chat request -> (text, finish_reason). Raises BenchError (BenchTimeout on a timeout)."""
        data = post_chat(cfg, session, payload, timeout)
        ch = (data.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason")

    for i, item in enumerate(items, 1):
        payload = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
                   "max_tokens": dynamic_max_tokens(cfg, spec["kind"], model_ctx, item["prompt"]),
                   "temperature": cfg.eval_temperature, "stream": False}
        if cfg.eval_extra_body:
            payload.update(cfg.eval_extra_body)
        ok, pred, reason, finish, text, is_na, used_fallback = False, None, "", None, "", False, False
        t1 = time.time()
        err = None
        try:
            text, finish = ask(payload)                                  # attempt 1: thinking as configured
        except BenchError as e:
            err = e
        if can_retry and (isinstance(err, BenchTimeout) or (err is None and finish == "length")):
            used_fallback = True                                         # attempt 2 (the only retry): No Thinking
            fallback += 1
            fb_timeouts += isinstance(err, BenchTimeout)
            try:
                text, finish = ask(no_thinking(payload))
                err = None
            except BenchError as e:
                if err is not None:         # attempt 1 timed out and attempt 2 failed too: report the newer error.
                    err = e                 # (If attempt 1 was just cut off, keep its reply and grade that.)
        if err is not None:
            errors += 1
            consec += 1
            reason = f"api error: {err}"
        else:
            consec = 0
            if finish == "length":
                trunc += 1
                body = strip_think(text)
                if not body or (spec["kind"] == "qa" and extract_choice(body, item["letters"]) is None):
                    is_na, reason = True, "no answer: cut off by the token limit (N/A)"
            if not is_na:
                try:
                    ok, pred, reason = grade_item(cfg, key, item, text)
                except Exception as e:      # a grader/sandbox problem must never abort the whole benchmark
                    reason = f"grader error: {e.__class__.__name__}: {e}"
        n += 1
        na += is_na
        correct += bool(ok)
        sink({"model": model, "dataset": key, "id": item["id"], "correct": bool(ok), "na": is_na,
              "thinking_fallback_used": used_fallback, "max_tokens": payload["max_tokens"], "answer": pred,
              "gold": item.get("gold"), "reason": reason, "finish": finish,
              "seconds": round(time.time() - t1, 2), "response": text[:4000]})
        if live:
            print(f"\r    {spec['short']:<10} {i}/{len(items)}  {100 * correct / n if n else 0:5.1f}%"
                  + (f"  ({na} N/A)" if na else ""), end="", flush=True)
        if consec >= cfg.eval_max_consec_errors:
            aborted = True
            break
    print(("\r" if live else "") + f"    {spec['short']:<10} {correct}/{n} = {100 * correct / n if n else 0:.1f}%"
          + (f"  [{na} N/A]" if na else "") + f"  ({fmt_hms(time.time() - t0)})" + (" " * 12 if live else ""))
    if aborted:
        print(f"    [warn] stopped after {cfg.eval_max_consec_errors} consecutive API errors (last: {reason})")
    elif errors:
        print(f"    [warn] {errors} request(s) failed and were counted as wrong")
    if fallback:
        print(f"    [info] {fallback} reply(ies) hit the token limit and were retried with No Thinking"
              + (f" ({fb_timeouts} after a timeout)" if fb_timeouts else ""))
    if na:
        print(f"    [info] {na} reply(ies) had no usable answer (cut off by the token limit): N/A, counted as wrong")
    if n and trunc / n > 0.25:
        print(f"    [warn] {100 * trunc / n:.0f}% of replies were cut off at the token limit"
              + (" even with thinking disabled" if can_retry else "")
              + ". The model's context size (-c) is probably too small for this dataset.")
    return {"pct": (100.0 * correct / n) if n - errors > 0 else None, "n": n, "err": errors, "na": na,
            "fallback": fallback, "sec": round((time.time() - t0) / n, 1) if n else None,
            "trunc": (100.0 * trunc / n) if n else None}


def write_dashboards(cfg, results, stamp, cpu_source, hardware, partial=False):
    """Write results/benchmark_dashboard_<stamp>.html and copy it to <script folder>/benchmark_dashboard.html."""
    stamped = cfg.output_dir / f"benchmark_dashboard_{stamp}.html"      # one per run, in ./results
    latest = SCRIPT_DIR / "benchmark_dashboard.html"                    # always the newest, next to benchmark.py
    render_dashboard(results, cfg, stamped, cpu_source, hardware, partial)
    shutil.copy2(stamped, latest)
    return stamped, latest


def run_evals(cfg, session, results, data, stamp):
    """Evaluation phase: runs AFTER all speed tests, one model at a time (each model loads once).
    Fills r['eval_<dataset>_pct|n|err|trunc'] and appends one JSON line per question to eval_details_<stamp>.jsonl."""
    todo = [r for r in results if r["status"] == "ok"]
    path = cfg.output_dir / f"eval_details_{stamp}.jsonl"
    calls = sum(len(v) for v in data.values()) * len(todo)
    print(f"\nEvaluations: {len(todo)} model(s) x {len(data)} dataset(s) = {calls} requests "
          f"(max_tokens {tokens_label(cfg.eval_max_tokens_qa)} QA / {tokens_label(cfg.eval_max_tokens_code)} code, "
          f"temperature {cfg.eval_temperature:g}"
          + (", No Thinking retry on cut-off" if cfg.eval_retry_on_length else ", no retry") + ")")
    with open(path, "a", encoding="utf-8") as fh:
        def sink(rec):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        mid = -(-len(todo) // 2)            # halfway point by model count; odd counts round up (5 models -> 3)
        for i, r in enumerate(todo, 1):
            print(f"[eval {i}/{len(todo)}] {r['model']}")
            ctx = r.get("ctx_size")                  # known from the speed test; evals-only runs look it up here
            if not ctx:
                try:
                    ctx = get_ctx(cfg, session, r["model"])
                except Exception:
                    ctx = None
            if not ctx:
                print(f"    [warn] context size unknown: using max_tokens {FALLBACK_TOKENS} for every question")
            for key in cfg.eval_datasets:
                res = eval_dataset(cfg, session, r["model"], key, data[key], sink, ctx)
                for field, v in res.items():
                    r[f"eval_{key}_{field}"] = v
            if i == mid:
                stamp_line(f"Evals halfway ({i}/{len(todo)} models done)")
    return path


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
                 ram_total_gb=64, ram_used_gb=24 + cw / 4, ram_idle_gb=24, ram_delta_gb=cw / 4, ram_proc_gb=(cw / 3 if cw >= 55 else 1.4),
                 offload=("n-cpu-moe 22" if cw >= 55 else "none"), energy_wh=(gw + cw) * lat * 2 / 3600, j_per_token=(gw + cw) / tps)
        rng = random.Random(name)
        q = min(1.0, 0.35 + (parse_params(name) or 10) / 45)       # bigger demo models score higher
        for key, (lo, hi) in {"gpqa": (22, 58), "mmlu_pro": (30, 82), "humaneval": (45, 94),
                              "mbpp": (40, 84), "livecodebench": (5, 48)}.items():
            r[f"eval_{key}_pct"] = lo + (hi - lo) * q * rng.uniform(0.85, 1.0)
            r[f"eval_{key}_n"], r[f"eval_{key}_err"] = 50, 0
            r[f"eval_{key}_trunc"] = rng.choice([0.0, 0.0, 2.0, 6.0])
            r[f"eval_{key}_sec"] = round(rng.uniform(8, 60) * (150 / tps) * (3 if key == "gpqa" else 1), 1)
        out.append(r)
    cfg.eval_info = {"gpqa": {"rows": 50, "available": 198}, "mmlu_pro": {"rows": 50, "available": 12032},
                     "humaneval": {"rows": 50, "available": 164}, "mbpp": {"rows": 50, "available": 500},
                     "livecodebench": {"rows": 50, "available": 1055}}
    cfg.eval_sandbox_resolved = "subprocess"
    return out


def check(cfg):
    """Verify every telemetry source for ~3 seconds. Run this before a long benchmark."""
    print(f"Hardware: {detect_hardware(cfg) or 'not detected'}")
    print(f"llama-swap: {cfg.base_url}")
    ls_ok = False
    try:
        n = len(make_session(cfg, "llama-swap").get(cfg.models_url, timeout=5).json().get("data", []))
        print(f"  OK, {n} model(s) listed")
        ls_ok = True
    except (requests.RequestException, ValueError) as e:
        print(f"  FAILED: {e.__class__.__name__}")
    if cfg.strata_url:
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
    used, total = h.get("sys_mem_gb"), HostMeter._mem_total_gb()
    print(f"CPU util: {h.get('cpu_util', float('nan')):.0f}%   RAM: "
          + (f"{used:.1f} / {total:.1f} GB used ({used / total * 100:.0f}%)" if used and total else "unavailable"))
    ram = RamSampler(cfg)
    if cfg.strata_url:
        port = urlparse(cfg.strata_url).port or 80
        ram.listen_port = port
        pids = find_listener_pids(port)
        print(f"Strata server process (listening on port {port}): "
              + (", ".join(f"{Path(f'/proc/{p}/comm').read_text().strip()} (pid {p}) using {ram.rss_gb([p]) or 0:.1f} GB RAM"
                           for p in sorted(pids)) if pids else "not found on this host (needs root; Docker bridge ports show docker-proxy)"))
        print("  Run `python benchmark.py --probe-strata` to see what Strata's API exposes.")
    procs = ram.find_processes()
    if procs:
        for pid in sorted(procs):
            try:
                name = Path(f"/proc/{pid}/comm").read_text().strip()
            except OSError:
                name = "?"
            print(f"Inference process matching RAM_PROCESS_PATTERN={cfg.ram_process_pattern!r}: "
                  f"{name} (pid {pid}) using {ram.rss_gb([pid]) or 0:.1f} GB RAM")
    else:
        big = []
        for d in os.listdir("/proc"):
            if d.isdigit():
                gb = ram.rss_gb([int(d)])
                if gb:
                    try:
                        big.append((gb, Path(f"/proc/{d}/comm").read_text().strip(), d))
                    except OSError:
                        pass
        print(f"No process matches RAM_PROCESS_PATTERN={cfg.ram_process_pattern!r} (server RAM will be blank; "
              "RAM spill falls back to the system-RAM delta). Largest processes by RAM - set the pattern to your "
              "inference server's name: " + ", ".join(f"{n} {g:.1f} GB" for g, n, _ in sorted(big, reverse=True)[:5]))
    if cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity:
        print("Home Assistant CPU-power fallback: configured")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def _emoji(t):
    if t is None:
        return ""
    return "🟣" if t > 100 else "🔵" if t > 80 else "🟢" if t > 50 else "🟡" if t > 30 else "🩷"
def _f(v, d=1): return "–" if v is None else f"{v:.{d}f}"


def render_markdown(results, cfg, path):
    ok = [r for r in results if r["status"] == "ok" and r["avg_tps"] is not None]
    top = sorted((r for r in ok if r["eligible"]), key=lambda r: -r["goldilocks"])[:3]
    speed = bool(ok)
    L = ["---", "tags: [llm/benchmark, llama-swap]", f"date: {datetime.now():%Y-%m-%d}", "---", "",
         "# llama-swap Benchmark Report", "", f"> {len(results)} models · {datetime.now():%Y-%m-%d %H:%M}"
         + (f" · VRAM limit {cfg.vram_limit_gb:g} GB" if speed else " · evaluations only (no speed/hardware telemetry)")]
    if speed:
        L += ["", "## Top 3 — Goldilocks", "", "| Rank | Model | TPS | VRAM | GPU W | Score |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(top if speed else [], 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} | {_f(r['vram_gb'])} GB "
                 f"| {_f(r['gpu_w'], 0)} | {_f(r['goldilocks'], 0)} |")
    if speed:
      L += ["", "## Speed Overview", "", "| Rank | Model | Avg TPS |", "|---|---|---|"]
    for i, r in enumerate(sorted(ok, key=lambda r: -r["avg_tps"]), 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")
    if speed:
      L += ["", "## Full Model Comparison", "",
          "| Model | Params | Quant | Ctx | Load s | VRAM | RAM spill GB | GPU W | CPU W | Wh | J/tok | TTFT ms | Prefill t/s | Tech | Logic | Avg |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in (results if speed else []):
        if r["status"] != "ok":
            L.append(f"| {r['model']} | | | | | | | | | | | | | | | ERR: {r['error']} |")
            continue
        p = r["vram_pct"]
        vram = "–" if p is None else f"{r['vram_gb']:.1f} GB [{'=' * round(min(p, 100) / 10):<10}] {p:.0f}%"
        L.append(f"| {r['model']} | {_f(r['params_b'], 0)}B | {r['quant']} | {_f((r['ctx_size'] or 0) / 1024 or None, 0)}k | {_f(r['load_s'])} | {vram} | {_f(r['ram_spill_gb'])} | "
                 f"{_f(r['gpu_w'], 0)} | {_f(r['cpu_w'], 0)} | {_f(r['energy_wh'], 3)} | {_f(r['j_per_token'])} | "
                 f"{_f(r['ttft_ms'], 0)} | {_f(r['prefill_tps'], 0)} | {_f(r['tech_tps'])} | {_f(r['logic_tps'])} | "
                 f"{_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")
    groups = {}
    for r in ok:
        groups.setdefault(r["quant"], []).append(r)
    if speed:
      L += ["", "## Quantization Analysis", "", "| Quant | Models | Avg TPS |", "|---|---|---|"]
    for q, rs in sorted(groups.items(), key=lambda kv: -_avg([r["avg_tps"] for r in kv[1]])):
        L.append(f"| {q} | {len(rs)} | {_f(_avg([r['avg_tps'] for r in rs]))} |")
    evk = [k for k in EVAL_SPECS if any(r.get(f"eval_{k}_n") or r.get(f"eval_{k}_na") for r in results)]
    if evk:
        L += ["", "## Accuracy & Pass Rates", "",
              "| Model | " + " | ".join(f"{EVAL_SPECS[k]['short']} | {EVAL_SPECS[k]['short']} s/q" for k in evk) + " | Avg |",
              "|---|" + "---|" * (2 * len(evk) + 1)]
        for r in sorted((r for r in results if r["status"] == "ok" and r.get("eval_avg_pct") is not None),
                        key=lambda r: -r["eval_avg_pct"]):
            cells = []
            for k in evk:
                p, n = r.get(f"eval_{k}_pct"), r.get(f"eval_{k}_n")
                na = r.get(f"eval_{k}_na") or 0
                cells.append("–" if p is None else f"{p:.1f}% ({n}" + (f", {na} N/A" if na else "") + ")")
                sq = r.get(f"eval_{k}_sec")
                cells.append("–" if sq is None else f"{sq:.1f}")
            L.append(f"| {r['model']} | " + " | ".join(cells) + f" | {r['eval_avg_pct']:.1f}% |")
        L += ["", f"_Profile `{cfg.eval_profile}`: QA = accuracy, code = pass@1; the number in brackets is how many "
              "questions were asked; N/A = no answer because the reply was cut off, counted as wrong; s/q = average seconds per "
              "question (total dataset time / questions asked, so it includes retries and timeouts)._"]
    L += ["", "## Methodology", ""]
    if speed:
        L += [f"- Median of {cfg.runs} run(s) per prompt; prompt cache disabled; each model is unloaded first, "
              "idle baseline sampled, then cold-loaded.",
              f"- GPU: `nvidia-smi` stream every {cfg.gpu_poll_ms} ms. CPU power: RAPL (or Home Assistant). "
              "Energy = (mean GPU W + mean CPU W) × request time.",
              f"- Goldilocks = {'ln(params B)' if cfg.goldilocks_mode == 'log' else 'params B'} × avg TPS (total params)."]
    if evk:
        L.append(f"- Evals (run after the speed tests): {', '.join(EVAL_SPECS[k]['label'] for k in evk)}; profile "
                 f"`{cfg.eval_profile}`, seed {cfg.eval_seed}, temperature {cfg.eval_temperature:g}. QA is string-matched "
                 "on the parsed final letter; code is executed against the dataset's tests in a sandbox.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def render_dashboard(results, cfg, path, cpu_source="none", hardware="", partial=False):
    meta = {"partial": partial, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "count": len(results),
            "vram_limit": cfg.vram_limit_gb, "runs": cfg.runs, "goldilocks_mode": cfg.goldilocks_mode,
            "timeouts": [cfg.timeout_tech, cfg.timeout_logic], "max_tokens": [cfg.max_tokens_tech, cfg.max_tokens_logic],
            "prompts": [cfg.tech_prompt, cfg.logic_prompt], "max_vram_pct": cfg.goldilocks_max_vram_pct,
            "poll_ms": cfg.gpu_poll_ms, "cpu_source": cpu_source, "prefill_words": cfg.prefill_words,
            "hardware": hardware, "currency": cfg.currency, "has_cost": cfg.rate_peak is not None,
            "rate_peak": cfg.rate_peak, "rate_off": cfg.rate_offpeak if cfg.rate_offpeak is not None else cfg.rate_peak,
            "cache_test": cfg.cache_test, "backend": "Strata" if cfg.backend == "strata" else "llama.cpp via llama-swap",
            "spill_threshold_gb": cfg.spill_threshold_gb,
            "has_speed": any(r.get("avg_tps") is not None for r in results), "agent_sec_ref": cfg.agent_sec_ref,
            "strata_details": cfg.strata_details if cfg.backend == "strata" else None}
    evs = []
    for k, s in EVAL_SPECS.items():
        if any(r.get(f"eval_{k}_n") or r.get(f"eval_{k}_na") for r in results):
            info = (cfg.eval_info or {}).get(k, {})
            evs.append({"key": k, "label": s["label"], "short": s["short"], "kind": s["kind"], "repo": s["repo"],
                        "rows": info.get("rows"), "available": info.get("available")})
    meta["tests"] = (["Speed test"] if meta["has_speed"] else []) + [e["short"] for e in evs]
    meta["evals"] = {"profile": cfg.eval_profile, "seed": cfg.eval_seed, "datasets": evs,
                     "temperature": cfg.eval_temperature, "max_tokens_qa": tokens_label(cfg.eval_max_tokens_qa),
                     "max_tokens_code": tokens_label(cfg.eval_max_tokens_code), "code_sandbox": cfg.eval_sandbox_resolved or "subprocess",
                     "code_timeout": cfg.eval_code_timeout, "lcb_max_tests": cfg.eval_lcb_max_tests,
                     "retry_nothink": bool(cfg.eval_retry_on_length)} if evs else None
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
summary{cursor:pointer}
footer{margin-top:40px}
@media(max-width:900px){body{padding:12px}.charts{grid-template-columns:1fr}.fz{position:static}}
</style></head><body>
<h1>llama-swap Benchmark</h1>
<div class="sub" id="hdr"></div>

<div id="sdetails" style="margin-top:12px"></div>

<div class="card filters" id="filters"></div>

<div id="speedtop">
<h2>Category winners &mdash; top 3</h2><div id="podium"></div>

<h2>Charts</h2>
<div class="grid charts" id="charts"></div>
</div>

<div id="evsec" hidden>
<h2>Accuracy &amp; pass rates</h2>
<div class="sub" id="evsub"></div>
<div class="grid charts" id="evcharts" style="margin-top:12px"></div>
</div>

<h2>Full model comparison</h2>
<div class="toolbar" id="toolbar"></div>
<div id="tblwrap"><table id="tbl"></table></div>

<div id="quantsec"><h2>Quantization analysis</h2>
<div class="card tablewrap2"><table id="qtbl" class="qt"></table></div></div>

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
const OVH = ALL.find(m=>m.ram_overhead_gb!=null)?.ram_overhead_gb;
const IDLE = (() => { const v = ALL.map(m=>m.ram_idle_gb).filter(x=>x!=null).sort((a,b)=>a-b); return v.length ? v[Math.floor(v.length/2)] : null; })();
const cur = META.currency;
// ---- evaluations (only datasets that were actually run are in META.evals) ----
const EVM = META.evals, EV = EVM ? EVM.datasets : [];
const evPct = (m,e) => m['eval_'+e.key+'_pct'];
const evN = (m,e) => m['eval_'+e.key+'_n'];
const EVPAL = ['#58a6ff','#3fb950','#f0883e','#a371f7','#f778ba'];
const accColor = p => p>=75?'#3fb950':p>=50?'#58a6ff':p>=25?'#d29922':'#f778ba';
const EVKINDS = [['qa','Accuracy \u2014 knowledge &amp; reasoning (%)'],['code','Pass rate \u2014 coding, pass@1 (%)']].filter(([k])=>EV.some(e=>e.kind===k));

// Chart.js cannot resolve CSS var(): every chart colour below is a literal hex value.
function tpsColor(t){return t>100?'#a371f7':t>80?'#58a6ff':t>50?'#3fb950':t>30?'#d29922':'#f778ba';}
function vramColor(p){return p>90?'#f85149':p>70?'#db6d28':'#3fb950';}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v,d=1) => v==null ? '&ndash;' : Number(v).toFixed(d);
const ctxFmt = v => v==null ? '&ndash;' : (v>=1024 ? Math.round(v/1024)+'k' : String(v));

Chart.defaults.color = '#e6edf3';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.maintainAspectRatio = false;

const state = {filters:{quant:new Set(), arch:new Set(), size:new Set(), vram95:false, hideErr:false, noSpill:false},
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
  if (f.noSpill && (m.ram_spill_gb||0) > 0) return false;
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
    grp('Show','x',[['vram95','Hide VRAM \u226595%'],['noSpill','VRAM-only (hide RAM spill)'],['hideErr','Hide failed']]) +
    `<div class="fg"><button class="tog" data-reset="1">Reset</button><span id="fcount"></span></div>`;
  $('filters').onclick = e => {
    const b = e.target.closest('button'); if(!b) return;
    if (b.dataset.reset){ Object.values(state.filters).forEach(v=>v instanceof Set?v.clear():0); state.filters.vram95=state.filters.hideErr=state.filters.noSpill=false;
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

  const mm = OKR.filter(m=>m.vram_gb!=null).sort((a,b)=>((b.vram_gb||0)+(b.ram_proc_gb||0))-((a.vram_gb||0)+(a.ram_proc_gb||0)));
  if (box('mem', mm, Math.max(380, mm.length*30+100))){
    mk('c_mem', {type:'bar',
      data:{labels:mm.map(m=>m.model), datasets:[
        {label:'VRAM used', data:mm.map(m=>m.vram_gb), backgroundColor:'#58a6ff', borderColor:'#58a6ff', borderWidth:1},
        {label:'Inference-server RAM', data:mm.map(m=>m.ram_proc_gb??0), backgroundColor:'#f0883e', borderColor:'#f0883e', borderWidth:1}]},
      options:{indexAxis:'y', plugins:{legend:{display:true}, tooltip:{callbacks:{footer:c=>'Total: '+c.reduce((s,x)=>s+x.parsed.x,0).toFixed(1)+' GB'}}},
        scales:{x:{stacked:true, title:axTitle('GB')}, y:{stacked:true, ticks:yTicks}}}});
  }

  const qs = quantGroups(OKR);
  if (box('quant', qs, 340)){
    const colors = qs.map(x=>tpsColor(x.avg));
    mk('c_quant', {type:'bar', plugins:[valueLabels],
      data:{labels:qs.map(x=>x.q), datasets:[{data:qs.map(x=>x.avg), backgroundColor:colors, borderColor:colors, borderWidth:1}]},
      options:{plugins:{legend:{display:false}}, scales:{x:{ticks:{autoSkip:false}}, y:{grace:'12%', beginAtZero:true}}}});
  }
}

/* ---------- evaluations: accuracy (QA) and pass rate (code) ---------- */
const evLabels = {id:'evLabels', afterDatasetsDraw(chart){
  const c = chart.ctx; c.save(); c.fillStyle = '#e6edf3'; c.font = '11px sans-serif'; c.textBaseline = 'middle'; c.textAlign = 'left';
  chart.data.datasets.forEach((d,di)=>chart.getDatasetMeta(di).data.forEach((bar,i)=>{
    const v = d.data[i]; if (v==null) return; c.fillText(v.toFixed(1)+'%', bar.x+5, bar.y); }));
  c.restore();
}};
const evMean = (m, ds) => { const v = ds.map(e=>evPct(m,e)).filter(x=>x!=null); return v.length ? v.reduce((s,x)=>s+x,0)/v.length : -1; };
function renderEvalCharts(OKR){
  if (!EV.length) return;
  EVKINDS.forEach(([kind])=>{
    const key = 'ev'+kind, ds = EV.filter(e=>e.kind===kind);
    const items = OKR.filter(m=>ds.some(e=>evPct(m,e)!=null)).sort((a,b)=>evMean(b,ds)-evMean(a,ds));
    if (!box(key, items, Math.max(380, items.length*(ds.length*15+14)+100))) return;
    mk('c_'+key, {type:'bar', plugins:[evLabels],
      data:{labels:items.map(m=>m.model), datasets:ds.map(e=>{ const col = EVPAL[EV.indexOf(e)%EVPAL.length];
        return {label:e.label, data:items.map(m=>evPct(m,e)), backgroundColor:col, borderColor:col, borderWidth:1}; })},
      options:{indexAxis:'y', plugins:{legend:{display:true}, tooltip:{callbacks:{label:c=>{
          const m = items[c.dataIndex], e = ds[c.datasetIndex], p = evPct(m,e), n = evN(m,e);
          const na = m['eval_'+e.key+'_na']||0, sq = m['eval_'+e.key+'_sec'];
          return p==null ? `${e.label}: not run` : `${e.label}: ${p.toFixed(1)}% (${Math.round(p*n/100)}/${n} correct${na?`, ${na} N/A`:''}${sq!=null?`, ${sq.toFixed(1)} s/question`:''})`; }}}},
        scales:{x:{min:0, max:112, ticks:{callback:v=>v<=100?v+'%':''}, title:axTitle(kind==='qa'?'accuracy (%)':'pass@1 (%)')}, y:{ticks:yTicks}}}});
  });
  const sp = OKR.filter(m=>m.eval_avg_pct!=null && m.avg_tps!=null);
  if (box('evspeed', sp, 420)){
    mk('c_evspeed', {type:'scatter',
      data:{datasets:[{data:sp.map(m=>pt(m,m.eval_avg_pct,m.avg_tps)), pointRadius:7, pointHoverRadius:10,
        backgroundColor:sp.map(m=>tpsColor(m.avg_tps)+'cc'), borderColor:sp.map(m=>tpsColor(m.avg_tps))}]},
      options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.raw.name}: ${c.raw.x.toFixed(1)}% avg score, ${c.raw.y.toFixed(1)} tok/s`}}},
        scales:{x:{min:0, max:100, title:axTitle('Average score over '+EV.map(e=>e.short).join(', ')+' (%)')}, y:{min:0, title:axTitle('Avg tok/s')}}}});
  }
}

/* ---------- podium + radar ---------- */
const maxOf = f => Math.max(1e-9, ...ALLOK.map(m=>f(m)||0));
const MAXTPS = maxOf(m=>m.avg_tps), MAXEFF = maxOf(m=>m.tps_per_watt), MAXP = maxOf(m=>m.params_b);
// All four axes are 0-1: speed, efficiency and params relative to the best model; VRAM economy = 1 - VRAM%.
const radarVals = m => [m.avg_tps/MAXTPS, (m.tps_per_watt||0)/MAXEFF, (m.params_b||0)/MAXP, m.vram_pct==null?0:Math.max(0,1-m.vram_pct/100)];
// Top 3 per category (a model can appear in several rows): ranked by `f` among eligible models that have it.
const topBy = (arr, f, k=3) => arr.filter(m=>f(m)!=null).sort((a,b)=>f(b)-f(a)).slice(0,k);
const PODIUM = [
  {title:'OVERALL BALANCE',  name:'Balance score', f:m=>m.goldilocks},
  {title:'CHAT DRIVER',      name:'Chat score',    f:m=>m.score_chat},
  {title:'AGENTIC REASONER', name:'Agent score',   f:m=>m.score_agent, agent:true}
];
const RANKCLS = ['gold','silver','bronze'];
function renderPodium(OKR){
  for (let i=0;i<9;i++) drop('rad'+i);
  const pool = OKR.filter(m=>m.eligible);
  const radars = [];                                   // [{m, id}] drawn after the HTML exists
  const rows = PODIUM.map(cat=>{
    const cards = topBy(pool, cat.f).map((m,rank)=>{
      const id = radars.length; radars.push(m);
      return `
    <div class="card ${RANKCLS[rank]}"><div class="rank">#${rank+1} ${cat.title}</div>
    <div class="pname">${esc(m.model)}</div>
    <div class="stat"><b style="color:${tpsColor(m.avg_tps)}">${fmt(m.avg_tps)}</b> tok/s &middot; ${m.params_b}B params &middot; ctx ${ctxFmt(m.ctx_size)}</div>
    <div class="stat">VRAM <b>${fmt(m.vram_gb)} GB</b> (${fmt(m.vram_pct,0)}%) &middot; GPU <b>${fmt(m.gpu_w,0)} W</b> &middot; CPU <b>${fmt(m.cpu_w,0)} W</b>${m.ram_spill_gb>0?` &middot; <span class="err">RAM spill ${fmt(m.ram_spill_gb)} GB</span>`:''}</div>
    <div class="stat">${cat.name} <b>${fmt(cat.f(m),0)}</b>${cat.agent&&m.eval_avg_pct!=null?` (eval ${fmt(m.eval_avg_pct,0)}%${m.eval_avg_sec!=null?`, ${fmt(m.eval_avg_sec,0)} s/q`:''})`:''} &middot; <b>${fmt(m.j_per_token,2)}</b> J/token</div>
    <div class="radarbox"><canvas id="rad${id}"></canvas></div></div>`;
    });
    return cards.length ? `<div class="grid podium" style="margin-bottom:16px">${cards.join('')}</div>` : '';
  });
  $('podium').innerHTML = radars.length ? rows.join('')
    : '<div class="card sub">No eligible models for the current filters (need parsable params and a speed result).</div>';
  radars.forEach((m,i)=>mk('rad'+i, {type:'radar',
    data:{labels:['Speed','Efficiency','Reasoning (params)','VRAM economy'],
      datasets:[{data:radarVals(m), borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.25)', pointBackgroundColor:'#58a6ff', borderWidth:2}]},
    options:{plugins:{legend:{display:false}},
      scales:{r:{min:0, max:1, ticks:{display:false, stepSize:0.25}, grid:{color:'#30363d'}, angleLines:{color:'#30363d'}, pointLabels:{color:'#e6edf3', font:{size:10}}}}}}));
}

/* ---------- quantization table ---------- */
function quantGroups(OKR){
  const g = {}; OKR.forEach(m=>(g[m.quant] = g[m.quant]||[]).push(m));
  const mean = xs => xs.length ? xs.reduce((s,x)=>s+x,0)/xs.length : null;
  return Object.entries(g).map(([q,ms])=>({q, n:ms.length, avg:ms.reduce((s,m)=>s+m.avg_tps,0)/ms.length,
    ev:mean(ms.map(m=>m.eval_avg_pct).filter(x=>x!=null)), names:ms.map(m=>m.model)})).sort((a,b)=>b.avg-a.avg);
}
function renderQuant(OKR){
  const qg = quantGroups(OKR), showEv = EV.length>0 && qg.some(x=>x.ev!=null);
  $('qtbl').innerHTML = '<tr><th>Quant</th><th>Models</th><th>Avg TPS</th>' + (showEv?'<th>Eval avg %</th>':'') + '<th style="text-align:left">Members</th></tr>' +
    qg.map(x=>`<tr><td>${esc(x.q)}</td><td>${x.n}</td><td style="color:${tpsColor(x.avg)}">${fmt(x.avg)}</td>${showEv?`<td style="color:${x.ev==null?'inherit':accColor(x.ev)}">${fmt(x.ev)}</td>`:''}<td style="text-align:left;white-space:normal">${x.names.map(esc).join(', ')}</td></tr>`).join('');
}

/* ---------- comparison table ---------- */
const FROZEN = {model:260, params_b:64, quant:84, load_s:78, vram_gb:236};   // widths of the frozen columns
const rateKey = () => state.rate==='off' ? 'off' : 'peak';
const ALLCOLS = [
 {k:'model',l:'Model'},
 {k:'params_b',l:'Params',f:v=>v+'B',nodelta:1},
 {k:'quant',l:'Quant',nodelta:1},
 {k:'load_s',l:'Load s',dec:1,better:'low'},
 {k:'vram_gb',l:'VRAM',better:'low'},
 {k:'ctx_size',l:'Ctx',f:ctxFmt,nodelta:1},
 {k:'arch',l:'Type',nodelta:1},
 {k:'offload',l:'CPU offload',nodelta:1},
 {k:'ram_delta_gb',l:'RAM vs idle GB',dec:1,better:'low'},
 {k:'ram_used_gb',l:'System RAM',better:'low'},
 {k:'ram_spill_gb',l:'RAM spill GB',dec:1,better:'low'},
 {k:'ram_proc_gb',l:'Server RAM GB',dec:1,better:'low'},
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
 ...EV.flatMap(e=>[{k:'eval_'+e.key+'_pct',l:e.short+' %',dec:1,better:'high',acc:1},
                   {k:'eval_'+e.key+'_sec',l:e.short+' s/q',dec:1,better:'low'}]),
 ...(EV.length>1?[{k:'eval_avg_pct',l:'Eval avg %',dec:1,better:'high',acc:1},{k:'eval_avg_sec',l:'Avg s/q',dec:1,better:'low'}]:[]),
 {k:'goldilocks',l:'Goldilocks',dec:0,better:'high'},
 {k:'score_chat',l:'Chat score',dec:0,better:'high'},
 {k:'score_agent',l:'Agent score',dec:0,better:'high'}
];
const val = (m,c) => c.get ? c.get(m) : m[c.k];
// Columns with no data in this run (e.g. Load s / RAM delta on Strata) are hidden instead of shown empty.
const COLS = ALLCOLS.filter(c => c.k==='model' || ALL.some(m => m.status==='ok' && val(m,c)!=null));
const HIDDEN = ALLCOLS.filter(c => !COLS.includes(c)).map(c => c.l);
const FZ = {}; let fzLeft = 0;
COLS.forEach(c => { if (FROZEN[c.k]) { FZ[c.k] = {left:fzLeft, w:FROZEN[c.k]}; fzLeft += FROZEN[c.k]; } });
const LASTFZ = [...COLS].reverse().find(c => FZ[c.k])?.k;
const fzStyle = c => FZ[c.k] ? ` style="left:${FZ[c.k].left}px;width:${FZ[c.k].w}px;min-width:${FZ[c.k].w}px;max-width:${FZ[c.k].w}px"` : '';
const fzClass = c => FZ[c.k] ? 'fz'+(c.k===LASTFZ?' last':'') : '';

function deltaHtml(m, c, v, base){
  if (!base || m.model===base.model || c.nodelta || v==null) return '';
  const b = val(base,c); if (b==null || b===0) return '';
  const pct = (v/b-1)*100; if (!isFinite(pct)) return '';
  const small = Math.abs(pct) < 0.5;
  const good = c.better ? ((c.better==='high') === (pct>0)) : null;
  const cl = small || good==null ? 'dn' : (good ? 'good' : 'bad');
  return `<small class="d ${cl}">${pct>0?'+':''}${pct.toFixed(0)}%</small>`;
}
function cell(m, c, base){
  const v = val(m,c), st = fzStyle(c);
  let cls = fzClass(c), inner;
  const dl = deltaHtml(m,c,v,base);
  if (c.k==='model'){
    const star = base && base.model===m.model ? '\u2605 ' : '';
    inner = `<div class="clip" title="${esc(v)}${m.error?' \u2014 '+esc(m.error):''}">${star}${esc(v)}${m.status!=='ok'?` <span class="err">ERR: ${esc(m.error)}</span>`:''}</div>`;
  } else if (c.k==='vram_gb' || c.k==='ram_used_gb'){
    if (v==null) inner = '&ndash;';
    else {
      const raw = (c.k==='vram_gb' ? m.vram_pct : m.ram_pct) ?? 0, p = Math.min(raw,100), warn = raw >= 95;
      if (warn) cls += ' warn';
      inner = `<div class="clip">${v.toFixed(1)} GB<span class="bar"><i style="width:${p}%;background:${warn?'#f85149':vramColor(p)}"></i></span><span class="${warn?'err':''}">${warn?'\u26A0 ':''}${raw.toFixed(0)}%</span>${dl}</div>`;
    }
  } else if (c.k==='ram_spill_gb'){
    inner = (v==null ? '&ndash;' : (v>0 ? `<span class="err">\u26A0 ${v.toFixed(1)}</span>` : '0')) + dl;
  } else if (c.k==='quant' || c.k==='arch' || c.k==='offload') inner = esc(v ?? '\u2013');
  else if (v==null) inner = '&ndash;';
  else if (c.f) inner = c.f(v);
  else {
    const t = (c.cost ? cur : '') + Number(v).toFixed(c.dec ?? 1);
    const colr = c.tps ? tpsColor(v) : (c.acc ? accColor(v) : null);
    inner = (colr ? `<span style="color:${colr}">${t}</span>` : t) + dl;
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
  $('tbl').innerHTML = '<thead><tr>' + COLS.map(c=>{
      return `<th class="${fzClass(c)}" data-k="${c.k}"${fzStyle(c)}>${c.l}${state.sk===c.k?(state.sd<0?' \u25BC':' \u25B2'):''}</th>`;
    }).join('') + '</tr></thead><tbody>' +
    rows.map(m=>`<tr data-m="${encodeURIComponent(m.model)}"${base&&base.model===m.model?' class="sel"':''}>${COLS.map(c=>cell(m,c,base)).join('')}</tr>`).join('') + '</tbody>';
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
  const hid = HIDDEN.length ? `<span class="sub">Hidden (no data this run): ${HIDDEN.join(', ')}</span>` : '';
  $('toolbar').innerHTML = bl + rate + hid;
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
  if ($('fcount')) $('fcount').textContent = `Showing ${cur_rows.length} of ${ALL.length} models`;
  renderPodium(OKR); renderCharts(OKR); renderEvalCharts(cur_rows.filter(m=>m.status==='ok')); renderQuant(OKR); renderTable();
}

$('hdr').innerHTML = `${META.count} models` + (META.tests.length ? ` &middot; <b style="color:#e6edf3">Tests:</b> ${META.tests.map(esc).join(', ')}${META.evals?` (profile ${esc(META.evals.profile)})`:''}` : '') +
  ` &middot; ${esc(META.generated)} &middot; ${esc(META.backend)}<br>` +
  (META.partial ? '<b style="color:#d29922">Speed test only &mdash; the evaluations had not finished when this dashboard was saved. It is replaced automatically when they complete.</b><br>' : '') +
  (!META.has_speed ? 'Evaluations only &mdash; speed and hardware telemetry were not measured in this run.' : `<b style="color:#e6edf3">Host:</b> ${META.hardware ? esc(META.hardware) : 'VRAM limit '+META.vram_limit+' GB'} &middot; ` +
  `<b style="color:#e6edf3">Power:</b> GPU via nvidia-smi (${META.poll_ms} ms), CPU via ${esc(META.cpu_source)}` +
  (IDLE!=null ? ` &middot; <b style="color:#e6edf3">Idle RAM baseline:</b> ${IDLE.toFixed(1)} GB` : ''));
const CH = [['tps','Avg generation speed (tok/s)'],['vram','VRAM vs speed (bubble = params)'],['eff','Power efficiency (tok/s per W, GPU + CPU)'],
  ['jt','Energy per token (J, lower is better)'],['ttft','TTFT leaderboard (ms, lower is better)'],['lat','Latency vs throughput (top-left is best)'],
  ['pg','Prefill vs generation speed (tok/s, log scale)'],['pwr','Active vs idle GPU power (W)'],['mem','Memory footprint: VRAM vs inference-server RAM (GB)'],['quant','Avg TPS by quantization']];
const chartCard = ([id,t])=>`<div class="card"><h3>${t}</h3><div class="chartbox" id="b_${id}"><canvas id="c_${id}"></canvas></div><p class="sub" id="e_${id}" hidden>No data for this chart (telemetry unavailable, or filtered out).</p></div>`;
$('charts').innerHTML = CH.map(chartCard).join('');
if (EV.length){
  $('evsec').hidden = false;
  $('evsub').innerHTML = `${EV.map(e=>`<b>${esc(e.label)}</b> (${e.rows??'?'}${e.available?' of '+e.available:''} rows)`).join(' &middot; ')} &mdash; profile <b>${esc(EVM.profile)}</b>. Only the tests and models that were actually run are shown.`;
  $('evcharts').innerHTML = [...EVKINDS.map(([k,t])=>['ev'+k,t]), ['evspeed','Quality vs speed (top-right is best)']].map(chartCard).join('');
}
if (META.strata_details && META.strata_details.length)
  $('sdetails').innerHTML = '<details class="card"><summary><b>Strata details</b> <span class="sub">(read from its API; click to expand)</span></summary>' +
    META.strata_details.map(([p,rows])=>`<h3>${esc(p)}</h3><table class="qt">${rows.map(([k,v])=>`<tr><td style="text-align:left">${esc(k)}</td><td style="text-align:left">${esc(v)}</td></tr>`).join('')}</table>`).join('') + '</details>';
if (!META.has_speed){      // evals-only run: no speed / hardware data to show
  $('speedtop').hidden = true; $('quantsec').hidden = true;
  document.querySelector('#filters').hidden = true;
}
buildFilters(); bindTable(); render();

$('meth').innerHTML = `<ul>
 ${!META.has_speed?'':`<li><b>Backend:</b> ${esc(META.backend)} (OpenAI-compatible <code>/v1/chat/completions</code>), non-streaming. The prompt cache is disabled for all speed tests.</li>
 <li><b>Per model:</b> ${META.backend==='Strata'?'single always-loaded model (no unload, load time or VRAM delta); idle power sampled with the model resident':'all models unloaded &rarr; idle baseline sampled &rarr; cold load (Load s)'} &rarr; Technical, Logic${META.prefill_words?` and prefill (~${META.prefill_words}-word, unique prompt)`:''} tests, median of ${META.runs} run(s).</li>
 <li><b>Prompts:</b> Technical &mdash; &ldquo;${esc(META.prompts[0])}&rdquo;; Logic &mdash; &ldquo;${esc(META.prompts[1])}&rdquo;.</li>
 <li><b>Speed:</b> <code>timings.predicted_per_second</code>, falling back to completion_tokens / wall time. <b>TTFT</b> is approximated by <code>timings.prompt_ms</code>; <b>Prefill t/s</b> is <code>timings.prompt_per_second</code> on the long prompt (<b>Prefill tok</b> = its size).</li>
 <li><b>Ctx</b> is the context size the model was loaded with (read from llama-server <code>/props</code>). VRAM and prefill speed depend on it, so compare models with similar Ctx.</li>
 ${META.cache_test?'<li><b>Cache hit %</b> / <b>Cache &times;:</b> a long prompt is sent cold, then again with a short follow-up appended (a multi-turn proxy). Hit % = share of the second prompt reused from the KV cache (<code>cache_n</code>); &times; = how much faster its prompt processing was than the cold one.</li>':''}
 <li><b>GPU:</b> one persistent <code>nvidia-smi</code> stream, windowed to each request. Power = mean; peak W, temp and VRAM = maximum. <b>Idle W</b> is sampled with no model loaded. <b>CPU:</b> ${esc(META.cpu_source)}.</li>
 <li><b>RAM:</b> <b>RAM vs idle</b> is the increase in system RAM over a baseline sampled with all models unloaded (llama-swap only). <b>System RAM</b> is the absolute figure (MemTotal &minus; MemAvailable, peak during the tests); it can differ a little from other dashboards that use another formula or sample at another moment. <b>Server RAM</b> is the peak resident memory of the inference-server process, which always includes some overhead (CUDA context, pinned buffers, mapped pages). <b>CPU offload</b> is read from the server's launch command (<code>-ngl</code> below 99, <code>--n-cpu-moe</code>, <code>--cpu-moe</code>, <code>--no-kv-offload</code>, <code>-ot ...CPU</code>). <b>RAM spill</b> is 0 for models launched fully on the GPU; for models with CPU offload it is Server RAM minus the typical overhead${OVH!=null?` (${OVH.toFixed(1)} GB, the median Server RAM of this run's fully-GPU models)`:''}. When launch flags can't be read (e.g. Strata) it is Server RAM if at least ${META.spill_threshold_gb} GB, otherwise 0. It is an estimate.</li>
 <li><b>Energy:</b> (mean GPU W + mean CPU W) &times; request time; Energy Wh covers one Technical plus one Logic request. <b>J/token</b> = energy &divide; generated tokens. <b>Efficiency</b> = avg TPS &divide; (GPU W + CPU W).</li>
 ${META.has_cost?`<li><b>Cost:</b> measured GPU + CPU energy only (not the rest of the system) &times; your electricity rate. ${cur}/1k queries = average of the Technical and Logic requests above, so it scales with the max_tokens setting.</li>`:''}
 <li><b>Goldilocks:</b> ${META.goldilocks_mode==='log'?'ln(params&nbsp;B)':'params&nbsp;B'} &times; avg TPS, VRAM &le; ${META.max_vram_pct}% of the GPU. Params are parsed from the filename and are <i>total</i> (not active) parameters. <b>Radar:</b> Speed, Efficiency and Reasoning (params) are relative to the best model in this run; VRAM economy = 100% &minus; VRAM%.</li>
 <li><b>Podium:</b> the top 3 in each of three categories, among eligible models: <b>Overall balance</b> (ranked by Goldilocks), <b>Chat driver</b> (ranked by Chat score) and <b>Agentic reasoner</b> (ranked by Agent score). The same model can appear in more than one row.</li>
 <li><b>Chat score:</b> <code>avg TPS &times; (100 &minus; VRAM%)</code>. Rewards a fast model that also leaves GPU memory free (for context or other apps).</li>
 ${META.evals?`<li><b>Agent score:</b> <code>quality &times; (100 &minus; VRAM%) &times; ln(avg TPS)</code>, where <b>quality</b> = average eval score &divide; (1 + avg s/q &divide; ${META.agent_sec_ref}). s/q is the average seconds per question (total dataset time &divide; questions, retries and timeouts included), so a model that answers instantly keeps its full eval score, one that takes ${META.agent_sec_ref}&nbsp;s per question keeps 50%, and one that takes ${META.agent_sec_ref*4}&nbsp;s keeps 20%. The ${META.agent_sec_ref}&nbsp;s reference is <code>AGENT_SEC_REF</code> in .env (higher = time matters less).</li>`
   :`<li><b>Agent score:</b> <code>quality &times; (100 &minus; VRAM%) &times; ln(avg TPS)</code>. No evaluations ran, so quality is the parameter count (a rough proxy, with no time adjustment). Those values are on a different scale from eval-based scores; only compare them with runs that used the same source.</li>`}
 <li><b>Limits:</b> max_tokens ${META.max_tokens.join(' / ')}; timeouts ${META.timeouts.join('s / ')}s. KV-cache type differs per model config and affects speed and VRAM.</li>`}
 ${EVM?`<li><b>Evaluations</b> (run after all speed tests, one model at a time): ${EV.map(e=>esc(e.label)+' ('+(e.rows??'?')+' rows)').join(', ')}. Profile <code>${esc(EVM.profile)}</code>; rows are sampled with seed ${EVM.seed}, so every model gets the same questions. Temperature ${EVM.temperature}, max_tokens ${EVM.max_tokens_qa} (QA) / ${EVM.max_tokens_code} (code)${EVM.max_tokens_qa==='dynamic'||EVM.max_tokens_code==='dynamic'?' (dynamic = model context minus prompt)':''}. Reasoning blocks are stripped before grading. ${EVM.retry_nothink?`A reply that is cut off or times out is retried once with thinking disabled (No Thinking). `:''}A reply that is still cut off with no usable answer counts as wrong, but is tracked separately as N/A (shown in the chart tooltips and the CSV). <b>s/q</b> = average seconds per question: total time for the dataset divided by the questions asked, so it includes retries and timeouts. Compare it between models that scored about the same.</li>
 ${EV.some(e=>e.kind==='qa')?'<li><b>QA grading:</b> the model is told to end with <code>Answer: X</code>; the last <i>Answer:</i> / <i>the answer is</i> / <code>\\boxed{}</code> letter (else a final line that is just a letter) is compared with the dataset\'s answer. API failures count as wrong; a model that returned nothing at all gets no score.</li>':''}
 ${EV.some(e=>e.kind==='code')?`<li><b>Code grading (pass@1):</b> the last python block of the reply is executed in a <code>${esc(EVM.code_sandbox)}</code> sandbox, ${EVM.code_timeout}s per run, no network where available. <b>HumanEval:</b> the task's own <code>check()</code>. <b>MBPP:</b> the <code>test_list</code> asserts (also shown in the prompt, the standard MBPP protocol). <b>LCB:</b> public tests plus up to ${EVM.lcb_max_tests||'all'} tests per problem (stdin/stdout or starter-code function).</li>`:''}
 <li><b>Read the scores with care:</b> with ${Math.min(...EV.map(e=>e.rows||Infinity))} questions the 95% margin of error is about &plusmn;${Math.round(98/Math.sqrt(Math.min(...EV.map(e=>e.rows||Infinity))))} points, so small gaps between models are noise. Use the <code>standard</code> or <code>full</code> profile for finer comparisons.</li>`:''}
</ul>`;
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def stamp_line(label):
    """Print a timestamp marker line (ends up in the console and in the log)."""
    print(f"[timestamp] {label}: {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}", flush=True)


def _hms(seconds):
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    return h, m, s


def fmt_hms(seconds):
    """758 -> '00h:12m:38s' (used for the per-dataset time in the eval lines)."""
    return "{:02d}h:{:02d}m:{:02d}s".format(*_hms(seconds))


def fmt_hr_m_s(seconds):
    """36250 -> '10 hr 4 m 10 s' (used for the total time at the end of the run)."""
    return "{} hr {} m {} s".format(*_hms(seconds))


def run_footer(cfg, info, t_start, finished_ok):
    """Closing banner: when the run ended, how long it took, how many models, which profile and which tests ran."""
    now = datetime.now().astimezone()
    bar = "=" * 50
    lines = [bar,
             f"Benchmark {'complete' if finished_ok else 'stopped'} at: {now:%a %b} {now.day:>2} {now:%H:%M:%S} "
             f"{now.tzname() or ''} {now:%Y}",
             f"Time taken to run benchmark: {fmt_hr_m_s(time.time() - t_start)}"]
    results = info.get("results") or []
    if info.get("demo"):
        lines.append("Demo run: synthetic data, no real tests were run")
    elif results:
        ran = [EVAL_SPECS[k]["short"] for k in EVAL_SPECS if any(r.get(f"eval_{k}_n") for r in results)]
        lines.append(f"{len(results)} Models" + (f" using EVAL_PROFILE '{cfg.eval_profile}'" if ran else ""))
        tests = (["Speed test"] if info.get("speed") else []) + ran
        if tests:
            lines.append("Running benchmarking tests: " + ", ".join(tests))
    lines += [bar, f"Script Finished {now:%b %d, %Y}  {now:%H:%M.%S}"]
    return "\n".join(lines)


class Tee:
    """Copy everything written to a stream into the log file as well. Live '\\r' progress updates are collapsed the
    way a terminal would show them, so the log keeps one clean line per result instead of every counter tick."""

    def __init__(self, stream, fh):
        self.stream, self.fh, self._pending = stream, fh, ""

    def write(self, s):
        self.stream.write(s)
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self.fh.write(line.split("\r")[-1].rstrip() + "\n")
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()

    def close_log(self):
        if self._pending.strip():
            self.fh.write(self._pending.split("\r")[-1].rstrip() + "\n")
            self._pending = ""
        self.fh.flush()

    def __getattr__(self, name):        # isatty(), encoding, fileno()... come from the real stream
        return getattr(self.stream, name)


def main():
    ap = argparse.ArgumentParser(description="Benchmark every model on a llama-swap server (run on the GPU host).")
    ap.add_argument("--check", action="store_true", help="test telemetry sources and exit")
    ap.add_argument("--probe-strata", action="store_true", help="list what Strata's API exposes and save it")
    ap.add_argument("--demo", action="store_true", help="build a dashboard from synthetic data (no server)")
    ap.add_argument("--evals-only", action="store_true",
                    help="skip the speed/telemetry tests and run only the evaluations (works from any machine "
                         "that can reach the API; same as RUN_SPEED_TESTS=false RUN_EVAL_TESTS=true)")
    ap.add_argument("--speed-only", action="store_true",
                    help="run only the speed/hardware tests and skip the evaluations completely "
                         "(same as RUN_SPEED_TESTS=true RUN_EVAL_TESTS=false)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models the server offers (GET /v1/models) and exit")
    ap.add_argument("--download-datasets", action="store_true",
                    help="download and cache the eval datasets marked true in .env into ./tests, then exit")
    args = ap.parse_args()
    if args.speed_only and args.evals_only:
        ap.error("--speed-only and --evals-only cannot be used together")

    cfg = load_config()
    if args.speed_only:
        cfg.run_eval_tests, cfg.run_speed_tests = False, True
    if args.check:
        return check(cfg)
    if args.probe_strata:
        return probe_strata(cfg)
    if args.list_models:
        return list_models(cfg)
    if args.download_datasets:
        return download_datasets(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path, log_fh, real_out, real_err = cfg.output_dir / f"benchmark_{stamp}.log", None, sys.stdout, sys.stderr
    if cfg.log_enabled:                      # LOG=false in .env turns the log file off
        try:
            log_fh = open(log_path, "w", encoding="utf-8")
            sys.stdout, sys.stderr = Tee(real_out, log_fh), Tee(real_err, log_fh)
        except OSError as e:
            print(f"[warn] could not create the log file {log_path}: {e}")
    bar = "=" * 50
    code, info = 0, {}
    try:
        print(f"{bar}\nBenchmark started: {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}"
              + (f"\nLog file: {log_path}" if log_fh else "") + f"\n{bar}")
        _run(args, cfg, stamp, info)
    except SystemExit as e:                  # keep the exit message in the log too
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            code = 1
        elif e.code:
            code = e.code
    except KeyboardInterrupt:
        print("\nInterrupted.")
        code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        print(run_footer(cfg, info, t_start, code == 0))
        if log_fh:
            for s in (sys.stdout, sys.stderr):
                if isinstance(s, Tee):
                    s.close_log()
            sys.stdout, sys.stderr = real_out, real_err
            log_fh.close()
    if code:
        sys.exit(code)


def _run(args, cfg, stamp, info):
    results, cpu_source, hardware = [], "n/a", ""
    info["results"] = results     # shared with main(): the closing banner reads models/tests from it
    eval_data, interrupted, details_path = {}, False, None
    evals_only = (args.evals_only or not cfg.run_speed_tests) and not args.demo
    if evals_only:
        cfg.run_eval_tests = True

    if args.demo:
        results, cpu_source = demo_results(cfg), "demo data"
        info.update(results=results, demo=True)
        hardware = detect_hardware(cfg) or "Demo system (synthetic data)"
    else:
        session, models = detect_backend(cfg)
        hardware = "" if evals_only else detect_hardware(cfg)
        if cfg.run_eval_tests and not cfg.eval_datasets:
            print("[warn] RUN_EVAL_TESTS=true but no dataset is enabled (EVAL_GPQA, EVAL_MMLU_PRO, EVAL_HUMANEVAL, "
                  "EVAL_MBPP, EVAL_LIVECODEBENCH); skipping the evals.")
        elif cfg.run_eval_tests:
            eval_data = prepare_eval_data(cfg)       # fail fast, before the long speed test

        if evals_only:
            # API-client mode: no nvidia-smi / RAPL / RAM sampling, nothing local is read.
            cpu_source = "not measured (evals-only run)"
            if not eval_data:
                sys.exit("[error] Nothing to do: enable at least one dataset (EVAL_GPQA, EVAL_MMLU_PRO, EVAL_HUMANEVAL, "
                         "EVAL_MBPP, EVAL_LIVECODEBENCH) in .env.")
            results = [blank(m) for m in models]
            info["results"] = results
            print(f"Evals-only run: {len(models)} model(s); speed and hardware telemetry are skipped.")
            try:
                details_path = run_evals(cfg, session, results, eval_data, stamp)
            except KeyboardInterrupt:
                print("\nInterrupted: writing partial results.")
        else:
            gpu = GpuStream(cfg)
            gpu.ok = gpu.start()
            host = HostMeter(cfg)
            host.ram.start()
            if cfg.backend == "strata":
                host.ram.listen_port = urlparse(cfg.base_url).port or 80
                found = strata_discover(cfg, session)
                cfg.strata_details = strata_details(found)
                cfg.strata_ctx_detected = ctx_from_details(cfg.strata_details)
                print(f"Strata: read {len(cfg.strata_details)} info endpoint(s); context "
                      f"{cfg.strata_ctx or cfg.strata_ctx_detected or 'unknown'}")
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
            print("Starting Speed Test")
            info["speed"] = True
            stamp_line("Speed test started")
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
                interrupted = True
                print("\nInterrupted: writing partial results.")
            finally:
                gpu.stop()
                host.ram.stop()
                if ha:
                    ha.stop()
            print("\nSpeed Test Complete")
            stamp_line("Speed test finished")

            # Evaluations run AFTER the speed tests, once telemetry is stopped so they cannot disturb it.
            if eval_data and not interrupted and any(r["status"] == "ok" for r in results):
                # Safety net for the long eval phase: save a speed-test-only dashboard now. If the evals crash or the
                # run is killed, it is still there; when the evals finish, the final dashboard replaces it.
                try:
                    finalize(results, cfg)
                    early, _ = write_dashboards(cfg, results, stamp, cpu_source, hardware, partial=True)
                    print(f"Speed-test dashboard saved (replaced when the evals finish): {early}")
                except Exception as e:                      # never let this stop the evaluations
                    print(f"[warn] could not save the speed-test dashboard: {e.__class__.__name__}: {e}")
                try:
                    details_path = run_evals(cfg, session, results, eval_data, stamp)
                except KeyboardInterrupt:
                    print("\nInterrupted: writing partial results.")

    if not results:
        sys.exit("No results to write.")
    finalize(results, cfg)
    ok = [r for r in results if r["status"] == "ok"]
    csv_path = cfg.output_dir / f"benchmark_{stamp}.csv"
    if not args.demo and not ok:
        if cfg.csv_enabled:
            write_csv(results, csv_path)
            sys.exit(f"\nNo model completed successfully, so no dashboard was written (errors are in {csv_path}).")
        sys.exit("\nNo model completed successfully, so no dashboard was written (CSV output is off; see the log).")
    if not args.demo and not evals_only and ok and all((r["vram_gb"] or 0) < 0.5 for r in ok):
        print("\n[warn] VRAM looks ~0 for every model: GPU telemetry is missing or from the wrong GPU (see GPU_INDEX).")
    out = []
    if cfg.csv_enabled:                      # CSV=false in .env skips both the per-run CSV and the history CSV
        write_csv(results, csv_path)
        if not args.demo:
            write_csv(results, cfg.output_dir / "benchmark_history_v2.csv", append=True)
        out.append(f"CSV:       {csv_path}")
    html_stamped, html_latest = write_dashboards(cfg, results, stamp, cpu_source, hardware)
    out += [f"Dashboard: {html_stamped}", f"Latest:    {html_latest}"]
    if details_path:
        out.append(f"Eval log:  {details_path}")
    if cfg.write_markdown:
        md_path = cfg.output_dir / "benchmark_report.md"
        render_markdown(results, cfg, md_path)
        out.append(f"Markdown:  {md_path}")
    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
