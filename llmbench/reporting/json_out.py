"""Structured JSON export for programmatic use (dashboards, databases, notebooks).

Writes, into OUTPUT_DIR:
    benchmark_<stamp>.json   one file per run (never overwritten)
    results.json             always the newest run (stable path for scripts to poll)

Layout (schema_version 1):
{
  "schema_version": 1,
  "tool": {"name", "version"},
  "run":   {"id", "started_at", "finished_at", "duration_s", "status", "mode", "backend", "models_total", ...},
  "host":  {"hardware", "platform", "python", "gpu_backend", "cpu_power_source"},
  "config": { ...non-secret settings... },
  "evaluation": {"enabled", "profile", "seed", "temperature", "concurrency", "http_backend", "code_sandbox",
                 "stream", "think_budget", "loop_abort", "salvage",
                 "datasets": {key: {"label", "kind", "repo", "rows", "available"}}},
  "artifacts": {"csv", "dashboard", "markdown", "eval_details", "log"},
  "models": [ {"model", "status", "error", "timestamp",
               "metadata": {...}, "speed": {...}, "gpu": {...}, "cpu": {...}, "memory": {...},
               "energy": {...}, "scores": {...},
               "evaluation": {"avg_pct", "avg_sec", "datasets": {key: {"pct","n","err","na","fallback","sec",
                                                                       "trunc","degr","cap","salv","tok","wall",
                                                                       "concurrency"}}}} ]
  degr = questions that fell into a repetition loop ("Fail: Degradation"); cap = questions that exceeded the thinking
  budget ("Fail: Think cap"); salv = questions graded from a stopped reply's partial answer; tok = mean tokens
  generated per question.
}
Missing measurements are JSON null (never NaN/Infinity, which are not valid JSON). Secrets (API keys, tokens)
are never written. Anything in a result row that is not mapped to a group lands in model["other"], so no column
is ever silently dropped when new fields are added.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from ..common import SCHEMA_VERSION, __version__
from ..config import EVAL_SPECS
from .margin import margin_of_error

_SECRETS = {"api_key", "strata_api_key", "ha_token"}

_GROUPS = {
    "metadata": ["params_b", "active_b", "arch", "quant", "offload", "ctx_size"],
    "speed": ["load_s", "tech_tps", "logic_tps", "avg_tps", "prefill_tps", "prefill_tokens", "cache_hit_pct",
              "cache_speedup", "ttft_ms", "gen_ms", "latency_s"],
    "gpu": ["gpu_w", "gpu_peak_w", "idle_gpu_w", "gpu_util", "gpu_temp", "vram_gb", "vram_delta_gb",
            "vram_total_gb", "vram_pct"],
    "cpu": ["cpu_w", "cpu_util"],
    "memory": ["ram_total_gb", "ram_used_gb", "ram_pct", "ram_idle_gb", "ram_delta_gb", "ram_proc_gb",
               "ram_spill_gb", "ram_overhead_gb"],
    "energy": ["energy_wh", "j_per_token", "tps_per_watt", "cost_1m_peak", "cost_1m_off", "cost_1kq_peak",
               "cost_1kq_off"],
    "scores": ["goldilocks", "score_chat", "score_agent", "agent_quality", "vram_fit", "score_sweet", "eligible",
               "score_note"],
    "launch": ["kv_type", "flash_attn", "ngl", "n_cpu_moe", "ubatch", "batch", "parallel", "mmproj", "jinja",
               "override_tensor"],
    "gguf": ["gguf_gb", "mmproj_gb", "bpw", "gguf_arch", "attn_type", "n_layers", "full_attn_layers", "swa_layers",
             "n_ctx_train", "expert_count", "expert_used", "kv_kib_token", "kv_swa_gb", "kv_approx", "kv_gb_ctx",
             "max_ctx_vram", "tok_per_word"],
    "context": ["ctx_ok", "prefill_8k_tps", "prefill_16k_tps", "prefill_32k_tps", "prefill_64k_tps", "prefill_128k_tps",
                "agent_prompt_tokens", "agent_ttft_s", "agent_ttft_warm_s"],
    "long_context": ["longctx_32k_pct", "longctx_64k_pct", "longctx_128k_pct", "longctx_avg_pct", "longctx_kv_ok",
                     "longctx_note"],
    "budget": ["budget_profile", "budget_pass", "budget_fails", "budget"],
}
_TOP = {"model", "status", "error", "timestamp"}
_EVAL_SUFFIXES = ("pct", "n", "err", "na", "fallback", "sec", "trunc", "degr", "cap", "salv", "tok", "wall", "conc")


def _clean(v):
    """Make a value strictly JSON-valid: non-finite floats -> None, Paths -> str, containers recursed."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


def _model_entry(r):
    used = set(_TOP)
    out = {k: r.get(k) for k in ("model", "status", "error", "timestamp")}
    for group, keys in _GROUPS.items():
        out[group] = {k: r.get(k) for k in keys}
        used.update(keys)
    datasets = {}
    for key in EVAL_SPECS:
        entry = {s: r.get(f"eval_{key}_{s}") for s in _EVAL_SUFFIXES}
        used.update(f"eval_{key}_{s}" for s in _EVAL_SUFFIXES)
        if any(v is not None for v in entry.values()):
            entry["concurrency"] = entry.pop("conc")
            datasets[key] = entry
    used.update(("eval_avg_pct", "eval_avg_sec", "eval_retry_n", "eval_retry_pct", "eval_quality"))
    out["evaluation"] = {"avg_pct": r.get("eval_avg_pct"), "avg_sec": r.get("eval_avg_sec"),
                         "retries": r.get("eval_retry_n"), "retry_pct": r.get("eval_retry_pct"),
                         "quality_after_retry_penalty": r.get("eval_quality"), "datasets": datasets}
    out["other"] = {k: v for k, v in r.items() if k not in used}       # nothing is dropped silently
    return out


def _config_snapshot(cfg):
    snap = {k: v for k, v in dataclasses.asdict(cfg).items()
            if k not in _SECRETS and k not in ("strata_details", "eval_info")}
    snap["ha_configured"] = bool(cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity)
    snap["api_key_set"] = bool(cfg.api_key)
    return snap


def build_report(results, cfg, info):
    """`info` carries run-level facts gathered by the CLI: stamp, started/finished (epoch s), hardware, cpu_source,
    mode, status, artifact paths."""
    ds_info = cfg.eval_info or {}
    datasets = {}
    for key, spec in EVAL_SPECS.items():
        if any(r.get(f"eval_{key}_n") or r.get(f"eval_{key}_na") for r in results):
            datasets[key] = {"label": spec["label"], "kind": spec["kind"], "repo": spec["repo"],
                             "rows": ds_info.get(key, {}).get("rows"), "available": ds_info.get(key, {}).get("available")}
    started, finished = info.get("started"), info.get("finished")
    iso = lambda t: datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds") if t else None
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "llama-swap-llm-benchmark", "version": __version__},
        "run": {"id": info.get("stamp"), "started_at": iso(started), "finished_at": iso(finished),
                "duration_s": round(finished - started, 1) if started and finished else None,
                "status": info.get("status", "complete"), "mode": info.get("mode"),
                "backend": cfg.backend, "base_url": cfg.base_url,
                "models_total": len(results), "models_ok": sum(r["status"] == "ok" for r in results)},
        "host": {"hardware": info.get("hardware"), "platform": platform.platform(),
                 "python": sys.version.split()[0], "gpu_backend": cfg.gpu_backend_used,
                 "cpu_power_source": info.get("cpu_source")},
        "config": _config_snapshot(cfg),
        "evaluation": ({"enabled": True, "profile": cfg.eval_profile, "seed": cfg.eval_seed,
                        "temperature": cfg.eval_temperature, "concurrency": cfg.eval_concurrency,
                        "http_backend": cfg.eval_http_backend, "code_sandbox": cfg.eval_sandbox_resolved or None,
                        "stream": cfg.eval_stream, "think_budget": cfg.eval_think_budget or None,
                        "loop_abort": cfg.eval_loop_abort, "salvage": cfg.eval_salvage,
                        "margin_of_error": margin_of_error(cfg, results),
                        "long_context": ({"lengths": list(cfg.longctx_lengths), "trials": cfg.longctx_trials,
                                          "keys": cfg.longctx_keys, "kv_min": cfg.longctx_kv_min,
                                          "kv_action": cfg.longctx_kv_action} if cfg.eval_longctx else None),
                        "datasets": datasets} if datasets else {"enabled": False, "datasets": {}}),
        "budget": cfg.budget,
        "artifacts": info.get("artifacts", {}),
        "models": [_model_entry(r) for r in results],
    }


def _atomic_write(path, text):
    """Write via a temp file + rename, so a reader polling results.json never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def json_paths(cfg, stamp):
    """(per-run path, stable 'latest' path). Known before writing so the file can list itself under `artifacts`."""
    return cfg.output_dir / f"benchmark_{stamp}.json", cfg.output_dir / "results.json"


def write_json(results, cfg, info, stamp):
    """Returns (per-run path, results.json path)."""
    doc = _clean(build_report(results, cfg, info))
    text = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    per_run, latest = json_paths(cfg, stamp)
    _atomic_write(per_run, text)
    _atomic_write(latest, text)
    return per_run, latest
