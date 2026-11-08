"""Configuration: .env loading, the Config dataclass, evaluation dataset specs."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Everything (.env, results/, tests/) is resolved relative to benchmark.py (the folder that CONTAINS this package),
# never the current directory, so it behaves the same from a cron job / Unraid User Script as from inside the folder.
SCRIPT_DIR = Path(__file__).resolve().parent.parent

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
                v = v.strip()
                if v and v[0] in "\"'":                       # quoted value: take what is inside the quotes
                    end = v.find(v[0], 1)
                    v = v[1:end] if end > 0 else v[1:]
                else:                                    # unquoted: "3   # comment" -> "3" (same as python-dotenv)
                    v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
                os.environ.setdefault(k.strip(), v)


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


def _env_budget(name, default):
    """Thinking budget in tokens: a number, or 0 / off / none / unlimited for no limit."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in ("off", "none", "unlimited", "false", "no"):
        return 0
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        sys.exit(f"[config] Invalid value for {name}: {raw!r} (a number of tokens, or 0 for no limit)")


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


def _env_ints(name, default=()):
    """Comma-separated integers ('8192,32768' or '8k,32k'); empty / 0 / off -> ()."""
    raw = os.getenv(name)
    if raw is None:
        return tuple(default)
    out = []
    for x in raw.split(","):
        x = x.strip().lower().strip('"').strip("'")
        if not x or x in ("0", "off", "none", "false"):
            continue
        mult = 1024 if x.endswith("k") else 1
        try:
            out.append(int(float(x.rstrip("k")) * mult))
        except ValueError:
            sys.exit(f"[config] Invalid value in {name}: {x!r} (token counts such as 8192 or 8k, comma-separated)")
    return tuple(out)


# Fixed result columns for the prompt-length tests (CSV needs a stable header). Lengths in .env are snapped to these.
PREFILL_SLOTS = {8192: "8k", 16384: "16k", 32768: "32k", 65536: "64k", 131072: "128k"}
LONGCTX_SLOTS = {32768: "32k", 65536: "64k", 131072: "128k"}


def _slots(name, raw, slots):
    """Keep the lengths that match a result column; warn about the rest."""
    keep, bad = [], []
    for n in raw:
        (keep if n in slots else bad).append(n)
    if bad:
        print(f"[config] {name}: {', '.join(map(str, bad))} ignored; supported lengths are "
              f"{', '.join(f'{k} ({v})' for k, v in slots.items())}")
    return tuple(dict.fromkeys(keep))


# --- Per-machine budget profiles --------------------------------------------
# Rules a model must pass to be deployable on a given machine. Each rule is a .env value BUDGET_<PROFILE>_<RULE>;
# the built-in profiles below are only defaults. A rule set to empty / "off" is not checked.
#   MIN_CTX           loaded context (tokens) must be at least this
#   VRAM_RESERVE_GB   VRAM that must stay free for other things on that machine (server: Immich face/CLIP models;
#                     main PC: the desktop baseline). Rule: model VRAM + reserve <= GPU total x MAX_VRAM_PCT
#   MAX_VRAM_PCT      (default 100) share of the GPU the model + reserve may use
#   KV_MIN            lowest acceptable KV-cache type (f16 > q8_0 > q5_1 > q5_0 > q4_1 > q4_0)
#   MAX_SPILL_GB      RAM spill limit (GB)                      MIN_TPS          generation tokens/s
#   MIN_PREFILL_TPS   prefill tokens/s (long-prompt test)       MAX_AGENT_TTFT_S time to first token, agent prompt
#   MIN_EVAL_PCT      eval average                              MAX_RETRY_PCT    share of eval questions retried
#   MIN_LONGCTX_PCT   long-context retrieval average
BUDGET_RULES = ("MIN_CTX", "VRAM_RESERVE_GB", "MAX_VRAM_PCT", "KV_MIN", "MAX_SPILL_GB", "MIN_TPS", "MIN_PREFILL_TPS",
                "MAX_AGENT_TTFT_S", "MIN_EVAL_PCT", "MAX_RETRY_PCT", "MIN_LONGCTX_PCT")
BUDGET_DEFAULTS = {
    "server": {"MIN_CTX": 65536, "VRAM_RESERVE_GB": 2.0, "MAX_VRAM_PCT": 100.0, "KV_MIN": "q8_0", "MAX_RETRY_PCT": 25.0},
    "main":   {"MIN_CTX": 65536, "VRAM_RESERVE_GB": 1.5, "MAX_VRAM_PCT": 100.0, "MAX_RETRY_PCT": 25.0},
}


def _budget():
    """BUDGET_PROFILE (server | main | any name | none) -> {"name", "rules": {RULE: value}} or None."""
    name = (_env("BUDGET_PROFILE", "none") or "none").strip().lower()
    if name in ("none", "off", "false", ""):
        return None
    rules = dict(BUDGET_DEFAULTS.get(name, {}))
    for rule in BUDGET_RULES:
        raw = os.getenv(f"BUDGET_{name.upper()}_{rule}")
        if raw is None:
            continue
        raw = re.split(r"\s+#", raw.strip(), maxsplit=1)[0].strip()
        if raw.lower() in ("", "off", "none", "false"):
            rules.pop(rule, None)
        elif rule == "KV_MIN":
            rules[rule] = raw.lower()
        else:
            try:
                rules[rule] = float(raw.rstrip("kK")) * (1024 if raw.lower().endswith("k") else 1)
            except ValueError:
                sys.exit(f"[config] BUDGET_{name.upper()}_{rule} must be a number (got {raw!r})")
    if not rules:
        print(f"[config] BUDGET_PROFILE={name}: no rules set (define BUDGET_{name.upper()}_<RULE> in .env)")
    return {"name": name, "rules": rules}


# --- Evaluation datasets (toggle each one in .env) ---------------------------
# kind "qa"   = multiple choice, scored as accuracy;  kind "code" = executed against tests, scored as pass@1;
# kind "math" = free-form final answer (number / LaTeX expression), scored as accuracy against the reference answer.
# Optional "config" = the Hugging Face dataset configuration name (GSM8K has no default config).
EVAL_SPECS = {
    "gpqa":          {"desc": "198 graduate-level multiple-choice questions in biology, physics and chemistry, written by PhD experts so that they cannot be answered by searching. Tests deep scientific reasoning. 4 options (25% by guessing).", "label": "GPQA Diamond",  "short": "GPQA",      "kind": "qa",   "env": "EVAL_GPQA",
                      "repo": "fingertap/GPQA-Diamond",               "split": "test"},
    "mmlu_pro":      {"desc": "Harder version of MMLU: about 12,000 multiple-choice questions over 14 subjects (law, engineering, psychology, economics, ...) with 10 options each (10% by guessing). Tests broad knowledge plus reasoning.", "label": "MMLU-Pro",      "short": "MMLU-Pro",  "kind": "qa",   "env": "EVAL_MMLU_PRO",
                      "repo": "TIGER-Lab/MMLU-Pro",                   "split": "test"},
    "humaneval":     {"desc": "164 Python tasks: write a function from its signature and docstring. The code is run in the sandbox against the task's unit tests (pass@1). Tests basic coding.", "label": "HumanEval",     "short": "HumanEval", "kind": "code", "env": "EVAL_HUMANEVAL",
                      "repo": "openai/openai_humaneval",              "split": "test"},
    "mbpp":          {"desc": "Mostly Basic Python Problems: about 500 short everyday programming tasks; the test asserts are shown in the prompt and the code is run against them (pass@1). Tests practical coding.", "label": "MBPP",          "short": "MBPP",      "kind": "code", "env": "EVAL_MBPP",
                      "repo": "google-research-datasets/mbpp",        "split": "test"},
    "livecodebench": {"desc": "Competitive-programming problems from LeetCode, AtCoder and Codeforces, collected over time so newer problems are less likely to be in a model's training data. Code is run against the problem's tests (pass@1). Tests hard coding.", "label": "LiveCodeBench", "short": "LCB",       "kind": "code", "env": "EVAL_LIVECODEBENCH",
                      "repo": "bzantium/livecodebench",               "split": "test"},
    "gsm8k":         {"desc": "Grade-school math word problems that need several steps of arithmetic; the final number must match. Tests step-by-step reasoning.", "label": "GSM8K",         "short": "GSM8K",     "kind": "math", "env": "EVAL_GSM8K",
                      "repo": "openai/gsm8k",  "config": "main",       "split": "test"},
    "math":          {"desc": "500 competition math problems (algebra, geometry, number theory, probability, precalculus, ...) from the MATH dataset; the final answer is checked for mathematical equivalence. Tests harder math.", "label": "MATH-500",      "short": "MATH",      "kind": "math", "env": "EVAL_MATH",
                      "repo": "HuggingFaceH4/MATH-500",               "split": "test"},
    # Built into the tool (llmbench/evals/toolcall_cases.py), nothing to download. BFCL-style: native OpenAI `tools`
    # requests, graded by matching the function name and arguments (AST-style), including "should not call" cases.
    "toolcall":      {"desc": "32 built-in function-calling cases sent as OpenAI 'tools' requests: pick the right tool, fill its arguments, call several tools at once, or correctly call none. Graded on the function name and arguments. Tests agent readiness.", "label": "Tool calling",  "short": "Tools",     "kind": "tool", "env": "EVAL_TOOLCALL",
                      "repo": "builtin",                              "split": "test", "builtin": True},
}
# Rows per active dataset. Sizes come from n = 0.9604 / E^2 (95% confidence, worst-case variance), where E is the
# margin of error shown in brackets. A dataset smaller than the profile is used in full (e.g. GPQA has 198 rows).
PROFILE_ROWS = {"test": 6,          # about +/-40%
                "quick": 11,        # +/-30%
                "basic": 24,        # +/-20%
                "moderate": 43,     # +/-15%
                "standard": 96,     # +/-10%
                "thorough": 196,    # +/-7%
                "deep": 384,        # +/-5%
                "ultra": 9604,      # +/-1% (larger than every dataset except MMLU-Pro)
                "full": None}       # every row
# EVAL_PROFILE=funnel (v2.1): every model answers the first stage, then confidence-interval gates drop the models that
# are clearly below the leaders, and the survivors answer the next (nested) stage, up to EVAL_FUNNEL_FINAL.
FUNNEL_CHAIN = ("quick", "basic", "moderate", "standard", "thorough", "deep")


def funnel_stages(cfg):
    """[(profile, rows)] for EVAL_PROFILE=funnel: EVAL_FUNNEL_STAGES if set, else every FUNNEL_CHAIN profile below
    EVAL_FUNNEL_FINAL, then the final profile. Rows grow strictly (each stage asks only the new questions)."""
    if cfg.eval_funnel_stages:
        names = list(cfg.eval_funnel_stages)
    else:
        final = cfg.eval_funnel_final
        cap = PROFILE_ROWS[final] if PROFILE_ROWS[final] is not None else float("inf")
        names = [p for p in FUNNEL_CHAIN if PROFILE_ROWS[p] < cap] + [final]
    out = []
    for n in names:
        rows = PROFILE_ROWS[n]
        if out and (out[-1][1] is None or (rows is not None and rows <= out[-1][1])):
            sys.exit(f"[config] EVAL_FUNNEL_STAGES must grow: {n} ({rows}) does not have more rows than {out[-1][0]}")
        out.append((n, rows))
    return out


def stage_rows(cfg):
    """Rows of the LARGEST stage (what has to be loaded): the profile's rows, or the funnel's final stage."""
    return funnel_stages(cfg)[-1][1] if cfg.eval_profile == "funnel" else PROFILE_ROWS[cfg.eval_profile]


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
    agent_sec_ref: float = 60.0             # AGENT_SEC_REF: s/q at which the time factor of the Agent score is halved
    agent_acc_tol: float | None = None      # AGENT_ACC_TOL: points below the best eval score still counted as a tie (None = 1 std. error)
    agent_acc_span: float = 40.0            # AGENT_ACC_SPAN: points below the tie zone at which the accuracy credit reaches 0
    agent_speed_weight: float = 0.3         # AGENT_SPEED_WEIGHT: most of the Agent score that speed can remove (0 = accuracy only)
    agent_tps_ref: float = 200.0            # AGENT_TPS_REF: tokens/s that earns the full token-speed credit
    vram_sweet_low: float = 80.0            # VRAM_SWEET_LOW: bottom of the ideal VRAM band (% of the GPU)
    vram_sweet_high: float = 90.0           # VRAM_SWEET_HIGH: top of the ideal band; above it the fit starts to fall
    vram_sweet_max: float = 94.0            # VRAM_SWEET_MAX: above this the fit is below 50% ("too tight")
    log_enabled: bool = True                # LOG: also write everything printed to results/benchmark_<stamp>.log
    csv_enabled: bool = True                # CSV: write benchmark_<stamp>.csv and append to the history CSV
    eval_retry_on_length: bool = True       # cut off / timed out -> retry the item once with thinking disabled
    eval_sandbox_resolved: str = ""         # set at run time: "docker" or "subprocess"
    eval_info: dict | None = None           # {dataset: {"rows": sampled, "available": total}} for the dashboard
    # --- v2 additions ---
    gpu_backend: str = "auto"               # GPU_BACKEND: auto | nvml | smi
    gpu_energy_counter: bool = False        # GPU_ENERGY_COUNTER: exact per-request GPU energy from NVML (opt-in)
    gpu_backend_used: str = "none"          # set at run time: "nvml" | "smi" | "none"
    eval_concurrency: int = 1               # EVAL_CONCURRENCY: simultaneous requests per dataset (1 = sequential)
    eval_http_backend: str = "auto"         # EVAL_HTTP_BACKEND: auto | aiohttp | threads
    eval_sandbox_workers: int = 0           # EVAL_SANDBOX_WORKERS: parallel code executions (0 = auto)
    json_enabled: bool = True               # JSON: write benchmark_<stamp>.json and results.json
    # --- v2.1: streamed evaluations ---
    eval_log_interval: float = 300.0        # EVAL_LOG_INTERVAL: seconds between progress lines when output is a log/pipe (0 = off)
    eval_stream: bool = True                # EVAL_STREAM: read eval replies as a stream (enables everything below)
    eval_think_budget: int = 16384          # EVAL_THINK_BUDGET: max thinking tokens per question (0 = unlimited)
    eval_loop_abort: bool = True            # EVAL_LOOP_ABORT: stop a reply that falls into a repetition loop
    eval_salvage: bool = True               # EVAL_SALVAGE: grade the visible answer of a stopped reply if it has one
    # --- v2.2: what decides VRAM, deployed context, decisive evals, budgets ---
    gguf_inspect: bool = True               # GGUF_INSPECT: read the model file's header (arch, params, BPW, KV size)
    gguf_path_map: tuple = ()               # GGUF_PATH_MAP: container path -> host path pairs
    gguf_search_dirs: tuple = ()            # GGUF_SEARCH_DIRS: folders to look for the model file by name
    max_ctx_vram_pct: float = 85.0          # MAX_CTX_VRAM_PCT: "max ctx" column = context that fits at this % of VRAM
    min_ctx: int | None = None              # MIN_CTX (else the budget profile's MIN_CTX)
    min_ctx_action: str = "flag"            # MIN_CTX_ACTION: flag (run, mark, keep off the podium) | refuse (skip model)
    prefill_ctx_tests: tuple = ()           # PREFILL_CTX_TESTS: prompt lengths (tokens) for the prefill-at-length test
    agent_prompt_tokens: int = 0            # AGENT_PROMPT_TOKENS: size of the agent-style prompt for TTFT (0 = off)
    eval_toolcall_strict: bool = True       # EVAL_TOOLCALL_STRICT: only native tool_calls count (not JSON in the text)
    eval_longctx: bool = False              # EVAL_LONGCTX: multi-key retrieval at LONGCTX_LENGTHS
    longctx_lengths: tuple = ()             # LONGCTX_LENGTHS
    longctx_kv_min: str = "q8_0"            # LONGCTX_KV_MIN: lowest KV type the retrieval result is trusted at
    longctx_kv_action: str = "flag"         # LONGCTX_KV_ACTION: flag | skip, when the model's KV is below LONGCTX_KV_MIN
    longctx_trials: int = 2                 # LONGCTX_TRIALS: prompts per length (different needles and depths)
    longctx_keys: int = 4                   # LONGCTX_KEYS: needles hidden per prompt (all of them are asked for)
    longctx_no_think: bool = True           # LONGCTX_NO_THINK: ask with thinking disabled (a pure retrieval test)
    agent_fallback_penalty: float = 10.0    # AGENT_FALLBACK_PENALTY: % of the Agent quality lost if EVERY question was retried
    agent_prefill_ref: float = 4000.0       # AGENT_PREFILL_REF: prefill tok/s that earns the full prefill credit
    agent_ttft_ref: float = 15.0            # AGENT_TTFT_REF: agent-prompt TTFT (s) at which its credit halves
    agent_longctx_weight: float = 0.25      # AGENT_LONGCTX_WEIGHT: share of the quality figure from long-context retrieval
    vram_underuse_floor: float = 0.85       # VRAM_UNDERUSE_FLOOR: VRAM fit of a model that uses almost no VRAM
    budget: dict | None = None              # BUDGET_PROFILE + BUDGET_<PROFILE>_<RULE>
    budget_podium_pass_only: bool = True    # BUDGET_PODIUM_PASS_ONLY: models that fail a budget rule stay off the podium
    # --- v2.1: faster runs ---
    eval_skip_ineligible: str = "auto"      # EVAL_SKIP_INELIGIBLE: auto | true | false (skip evals of undeployable models)
    eval_funnel_final: str = "standard"     # EVAL_FUNNEL_FINAL: profile the funnel's survivors finish at
    eval_funnel_stages: tuple = ()          # EVAL_FUNNEL_STAGES: explicit stage profiles (default: chain up to the final)
    eval_leaders: int = 5                   # EVAL_LEADERS: minimum number of models kept at every funnel gate
    eval_funnel_conf: float = 0.95          # EVAL_FUNNEL_CONF: one-sided confidence of each gate's bounds
    eval_result_cache: bool = True          # EVAL_RESULT_CACHE: reuse answers already obtained (resume / reruns)
    eval_result_cache_dir: Path | None = None   # EVAL_RESULT_CACHE_DIR (default <OUTPUT_DIR>/cache)
    eval_think_budgets: dict | None = None  # EVAL_THINK_BUDGET_<DATASET>: per-dataset thinking budget (opt-in)
    eval_budget_forcing: bool = False       # EVAL_BUDGET_FORCING: at the thinking cap, close the thinking and ask for the answer
    eval_lcb_since: str = ""                # EVAL_LCB_SINCE: only LiveCodeBench problems released on/after this date
    eval_revisions: dict | None = None      # EVAL_<DATASET>_REVISION: pin a Hugging Face dataset revision
    eval_sq_basis: str = "auto"             # EVAL_SQ_BASIS: auto | measured | single (s/q used by the Agent score)
    batch_probe_levels: tuple = (1, 2, 4, 8)    # BATCH_PROBE_LEVELS
    batch_probe_questions: int = 32         # BATCH_PROBE_QUESTIONS
    eval_pools: dict | None = None          # set at run time: {dataset: (rows available, candidate row indices | None)}
    defer_prefill: tuple = ()               # set at run time: prefill lengths measured by the long-context test instead

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


def _cache_dir() -> Path:
    """EVAL_RESULT_CACHE_DIR (default <OUTPUT_DIR>/cache). Relative paths are relative to the script."""
    raw = _env("EVAL_RESULT_CACHE_DIR")
    if not raw:
        return _output_dir() / "cache"
    p = Path(raw).expanduser()
    return p if p.is_absolute() else SCRIPT_DIR / p


def _think_budgets():
    """EVAL_THINK_BUDGET_<DATASET> (e.g. EVAL_THINK_BUDGET_GSM8K=2048): per-dataset thinking budgets, opt-in.
    The dataset part is the dataset's EVAL_<NAME> switch without the EVAL_ prefix (GPQA, MMLU_PRO, LIVECODEBENCH...)."""
    out = {}
    for k, s in EVAL_SPECS.items():
        name = "EVAL_THINK_BUDGET_" + s["env"][len("EVAL_"):]
        if os.getenv(name, "").strip():
            out[k] = _env_budget(name, None)
    return out


def think_budget(cfg, key):
    """Thinking budget for one dataset: its EVAL_THINK_BUDGET_<DATASET> override, else EVAL_THINK_BUDGET."""
    b = (cfg.eval_think_budgets or {}).get(key)
    return cfg.eval_think_budget if b is None else b


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
    if profile not in PROFILE_ROWS and profile != "funnel":
        sys.exit(f"[config] EVAL_PROFILE must be one of {', '.join(PROFILE_ROWS)} or funnel (got {profile!r})")
    funnel_final = (_env("EVAL_FUNNEL_FINAL", "standard") or "standard").lower()
    if funnel_final not in PROFILE_ROWS or funnel_final in ("test", "quick"):
        sys.exit(f"[config] EVAL_FUNNEL_FINAL must be a profile larger than quick (got {funnel_final!r})")
    funnel_stages_env = tuple(x.lower() for x in _env_list("EVAL_FUNNEL_STAGES"))
    bad = [x for x in funnel_stages_env if x not in PROFILE_ROWS]
    if bad:
        sys.exit(f"[config] EVAL_FUNNEL_STAGES: unknown profile(s) {', '.join(bad)}")
    skip_inel = (_env("EVAL_SKIP_INELIGIBLE", "auto") or "auto").lower()
    if skip_inel in ("1", "yes", "on"):
        skip_inel = "true"
    if skip_inel in ("0", "no", "off"):
        skip_inel = "false"
    if skip_inel not in ("auto", "true", "false"):
        sys.exit(f"[config] EVAL_SKIP_INELIGIBLE must be auto, true or false (got {skip_inel!r})")
    sq_basis = (_env("EVAL_SQ_BASIS", "auto") or "auto").lower()
    if sq_basis not in ("auto", "measured", "single"):
        sys.exit(f"[config] EVAL_SQ_BASIS must be auto, measured or single (got {sq_basis!r})")
    lcb_since = (_env("EVAL_LCB_SINCE", "") or "").strip()
    if lcb_since and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", lcb_since):
        sys.exit(f"[config] EVAL_LCB_SINCE must be a date like 2025-01-01 (got {lcb_since!r})")
    gpu_backend = (_env("GPU_BACKEND", "auto") or "auto").lower()
    if gpu_backend not in ("auto", "nvml", "smi"):
        sys.exit(f"[config] GPU_BACKEND must be auto, nvml or smi (got {gpu_backend!r})")
    http_backend = (_env("EVAL_HTTP_BACKEND", "auto") or "auto").lower()
    if http_backend not in ("auto", "aiohttp", "threads"):
        sys.exit(f"[config] EVAL_HTTP_BACKEND must be auto, aiohttp or threads (got {http_backend!r})")
    sandbox = (_env("EVAL_CODE_SANDBOX", "auto") or "auto").lower()
    if sandbox not in ("auto", "docker", "subprocess"):
        sys.exit(f"[config] EVAL_CODE_SANDBOX must be auto, docker or subprocess (got {sandbox!r})")
    min_ctx_action = (_env("MIN_CTX_ACTION", "flag") or "flag").lower()
    if min_ctx_action not in ("flag", "refuse"):
        sys.exit(f"[config] MIN_CTX_ACTION must be flag or refuse (got {min_ctx_action!r})")
    kv_action = (_env("LONGCTX_KV_ACTION", "flag") or "flag").lower()
    if kv_action not in ("flag", "skip"):
        sys.exit(f"[config] LONGCTX_KV_ACTION must be flag or skip (got {kv_action!r})")
    from .gguf import KV_PRECISION, parse_path_map
    kv_min = (_env("LONGCTX_KV_MIN", "q8_0") or "q8_0").lower()
    if kv_min not in KV_PRECISION:
        sys.exit(f"[config] LONGCTX_KV_MIN must be one of {', '.join(KV_PRECISION)} (got {kv_min!r})")
    budget = _budget()
    if budget and budget["rules"].get("KV_MIN") and budget["rules"]["KV_MIN"] not in KV_PRECISION:
        sys.exit(f"[config] BUDGET_{budget['name'].upper()}_KV_MIN must be one of {', '.join(KV_PRECISION)}")
    min_ctx = _env_ints("MIN_CTX")
    min_ctx = min_ctx[0] if min_ctx else (int(budget["rules"]["MIN_CTX"]) if budget and budget["rules"].get("MIN_CTX") else None)
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
        agent_acc_tol=(lambda v: None if v is None else max(0.0, v))(_env("AGENT_ACC_TOL", None, float)),
        agent_acc_span=max(1.0, _env("AGENT_ACC_SPAN", 40.0, float)),
        agent_speed_weight=min(1.0, max(0.0, _env("AGENT_SPEED_WEIGHT", 0.3, float))),
        agent_tps_ref=max(2.0, _env("AGENT_TPS_REF", 200.0, float)),
        vram_sweet_low=min(99.0, max(1.0, _env("VRAM_SWEET_LOW", 80.0, float))),
        vram_sweet_high=min(99.0, max(1.0, _env("VRAM_SWEET_HIGH", 90.0, float))),
        vram_sweet_max=min(99.5, max(1.0, _env("VRAM_SWEET_MAX", 94.0, float))),
        log_enabled=_env("LOG", True, bool),
        csv_enabled=_env("CSV", True, bool),
        eval_retry_on_length=_env("EVAL_RETRY_ON_LENGTH", True, bool),
        gpu_backend=gpu_backend,
        gpu_energy_counter=_env("GPU_ENERGY_COUNTER", False, bool),
        eval_concurrency=max(1, _env("EVAL_CONCURRENCY", 1, int)),
        eval_http_backend=http_backend,
        eval_sandbox_workers=max(0, _env("EVAL_SANDBOX_WORKERS", 0, int)),
        json_enabled=_env("JSON", True, bool),
        eval_log_interval=_env("EVAL_LOG_INTERVAL", 300.0, float),
        eval_stream=_env("EVAL_STREAM", True, bool),
        eval_think_budget=_env_budget("EVAL_THINK_BUDGET", 16384),
        eval_loop_abort=_env("EVAL_LOOP_ABORT", True, bool),
        eval_salvage=_env("EVAL_SALVAGE", True, bool),
        gguf_inspect=_env("GGUF_INSPECT", True, bool),
        gguf_path_map=tuple(parse_path_map(_env("GGUF_PATH_MAP", ""))),
        gguf_search_dirs=_env_list("GGUF_SEARCH_DIRS"),
        max_ctx_vram_pct=min(100.0, max(10.0, _env("MAX_CTX_VRAM_PCT", 85.0, float))),
        min_ctx=min_ctx,
        min_ctx_action=min_ctx_action,
        prefill_ctx_tests=_slots("PREFILL_CTX_TESTS", _env_ints("PREFILL_CTX_TESTS"), PREFILL_SLOTS),
        agent_prompt_tokens=max(0, _env("AGENT_PROMPT_TOKENS", 0, int)),
        eval_toolcall_strict=_env("EVAL_TOOLCALL_STRICT", True, bool),
        eval_longctx=_env("EVAL_LONGCTX", False, bool),
        longctx_lengths=_slots("LONGCTX_LENGTHS", _env_ints("LONGCTX_LENGTHS", (65536, 131072)), LONGCTX_SLOTS),
        longctx_kv_min=kv_min,
        longctx_kv_action=kv_action,
        longctx_trials=max(1, _env("LONGCTX_TRIALS", 2, int)),
        longctx_keys=min(12, max(1, _env("LONGCTX_KEYS", 4, int))),
        longctx_no_think=_env("LONGCTX_NO_THINK", True, bool),
        agent_fallback_penalty=max(0.0, _env("AGENT_FALLBACK_PENALTY", 10.0, float)),
        agent_prefill_ref=max(10.0, _env("AGENT_PREFILL_REF", 4000.0, float)),
        agent_ttft_ref=max(0.5, _env("AGENT_TTFT_REF", 15.0, float)),
        agent_longctx_weight=min(1.0, max(0.0, _env("AGENT_LONGCTX_WEIGHT", 0.25, float))),
        vram_underuse_floor=min(1.0, max(0.0, _env("VRAM_UNDERUSE_FLOOR", 0.85, float))),
        budget=budget,
        budget_podium_pass_only=_env("BUDGET_PODIUM_PASS_ONLY", True, bool),
        eval_skip_ineligible=skip_inel,
        eval_funnel_final=funnel_final,
        eval_funnel_stages=funnel_stages_env,
        eval_leaders=max(1, _env("EVAL_LEADERS", 5, int)),
        eval_funnel_conf=min(0.999, max(0.5, _env("EVAL_FUNNEL_CONF", 0.95, float))),
        eval_result_cache=_env("EVAL_RESULT_CACHE", True, bool),
        eval_result_cache_dir=_cache_dir(),
        eval_think_budgets=_think_budgets(),
        eval_budget_forcing=_env("EVAL_BUDGET_FORCING", False, bool),
        eval_lcb_since=lcb_since,
        eval_revisions={k: _env(s["env"] + "_REVISION") for k, s in EVAL_SPECS.items()
                        if not s.get("builtin") and _env(s["env"] + "_REVISION")},
        eval_sq_basis=sq_basis,
        batch_probe_levels=tuple(sorted(set(max(1, n) for n in _env_ints("BATCH_PROBE_LEVELS", (1, 2, 4, 8))))) or (1, 2, 4, 8),
        batch_probe_questions=max(4, _env("BATCH_PROBE_QUESTIONS", 32, int)),
    )
