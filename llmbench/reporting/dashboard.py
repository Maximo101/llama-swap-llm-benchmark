"""Self-contained interactive HTML dashboard (Chart.js); the template lives in dashboard.html."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from ..config import EVAL_SPECS, LONGCTX_SLOTS, PREFILL_SLOTS, SCRIPT_DIR, funnel_stages
from ..evals.runner import tokens_label
from .margin import margin_of_error

TEMPLATE_PATH = Path(__file__).with_name("dashboard.html")
GPU_LABELS = {"nvml": "NVML (pynvml)", "smi": "nvidia-smi", "none": "no GPU telemetry"}
REPO_URL = "https://github.com/Maximo101/llama-swap-llm-benchmark"
BUILTIN_URLS = {"toolcall": REPO_URL + "/blob/main/llmbench/evals/toolcall_cases.py"}


def dataset_url(spec, revision=None):
    """Link for a dataset in the dashboard header: its Hugging Face page (the pinned revision when set),
    or the source file for built-in sets."""
    if spec.get("builtin"):
        return BUILTIN_URLS.get(next((k for k, s in EVAL_SPECS.items() if s is spec), ""), REPO_URL)
    url = f"https://huggingface.co/datasets/{spec['repo']}"
    return f"{url}/tree/{revision}" if revision else url

def render_dashboard(results, cfg, path, cpu_source="none", hardware="", partial=False):
    meta = {"partial": partial, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "count": len(results),
            "vram_limit": cfg.vram_limit_gb, "runs": cfg.runs, "goldilocks_mode": cfg.goldilocks_mode,
            "timeouts": [cfg.timeout_tech, cfg.timeout_logic], "max_tokens": [cfg.max_tokens_tech, cfg.max_tokens_logic],
            "prompts": [cfg.tech_prompt, cfg.logic_prompt], "max_vram_pct": cfg.goldilocks_max_vram_pct,
            "poll_ms": cfg.gpu_poll_ms, "cpu_source": cpu_source, "gpu_source": GPU_LABELS.get(cfg.gpu_backend_used, "nvidia-smi"), "prefill_words": cfg.prefill_words,
            "hardware": hardware, "currency": cfg.currency, "has_cost": cfg.rate_peak is not None,
            "rate_peak": cfg.rate_peak, "rate_off": cfg.rate_offpeak if cfg.rate_offpeak is not None else cfg.rate_peak,
            "cache_test": cfg.cache_test, "backend": "Strata" if cfg.backend == "strata" else "llama.cpp via llama-swap",
            "spill_threshold_gb": cfg.spill_threshold_gb,
            "has_speed": any(r.get("avg_tps") is not None for r in results), "agent_sec_ref": cfg.agent_sec_ref,
            "agent_acc_span": cfg.agent_acc_span, "agent_speed_weight": cfg.agent_speed_weight, "agent_tps_ref": cfg.agent_tps_ref,
            "vram_sweet_low": cfg.vram_sweet_low, "vram_sweet_high": cfg.vram_sweet_high, "vram_sweet_max": cfg.vram_sweet_max,
            "strata_details": cfg.strata_details if cfg.backend == "strata" else None}
    evs = []
    for k, s in EVAL_SPECS.items():
        if any(r.get(f"eval_{k}_n") or r.get(f"eval_{k}_na") for r in results):
            info = (cfg.eval_info or {}).get(k, {})
            rev = (cfg.eval_revisions or {}).get(k)
            evs.append({"key": k, "label": s["label"], "short": s["short"], "kind": s["kind"], "repo": s["repo"],
                        "desc": s.get("desc", ""), "revision": rev, "url": dataset_url(s, rev),
                        "rows": info.get("rows"), "available": info.get("available")})
    meta["tests"] = (["Speed test"] if meta["has_speed"] else []) + [e["short"] for e in evs]
    meta["evals"] = {"profile": cfg.eval_profile, "seed": cfg.eval_seed, "datasets": evs,
                     "temperature": cfg.eval_temperature, "max_tokens_qa": tokens_label(cfg.eval_max_tokens_qa),
                     "max_tokens_code": tokens_label(cfg.eval_max_tokens_code), "code_sandbox": cfg.eval_sandbox_resolved or "subprocess",
                     "code_timeout": cfg.eval_code_timeout, "lcb_max_tests": cfg.eval_lcb_max_tests,
                     "retry_nothink": bool(cfg.eval_retry_on_length), "concurrency": cfg.eval_concurrency,
                     "stream": cfg.eval_stream, "think_budget": cfg.eval_think_budget, "loop_abort": cfg.eval_loop_abort,
                     "salvage": cfg.eval_salvage} if evs else None
    if meta["evals"] is not None:
        meta["evals"]["margin"] = margin_of_error(cfg, results)
        meta["evals"]["fallback_penalty"] = cfg.agent_fallback_penalty
        meta["evals"]["funnel"] = ({"stages": [list(x) for x in funnel_stages(cfg)], "leaders": cfg.eval_leaders,
                                    "confidence": cfg.eval_funnel_conf} if cfg.eval_profile == "funnel" else None)
        meta["evals"]["sq_basis"] = next((r.get("eval_sec_basis") for r in results if r.get("eval_sec_basis")), None)
        meta["evals"]["cache"] = bool(cfg.eval_result_cache)
        meta["evals"]["budget_forcing"] = cfg.eval_budget_forcing
        meta["evals"]["think_budgets"] = {EVAL_SPECS[k]["short"]: v for k, v in (cfg.eval_think_budgets or {}).items()}
    if cfg.eval_longctx and any(r.get("longctx_avg_pct") is not None or r.get("longctx_note") for r in results):
        meta["longctx"] = {"lengths": [LONGCTX_SLOTS[n] for n in cfg.longctx_lengths], "trials": cfg.longctx_trials,
                           "keys": cfg.longctx_keys, "kv_min": cfg.longctx_kv_min, "kv_action": cfg.longctx_kv_action}
        meta["tests"].append("Long-ctx")
    meta.update(max_ctx_pct=cfg.max_ctx_vram_pct, min_ctx=cfg.min_ctx, min_ctx_action=cfg.min_ctx_action,
                prefill_lengths=[PREFILL_SLOTS[n] for n in cfg.prefill_ctx_tests], agent_prompt_tokens=cfg.agent_prompt_tokens,
                budget=cfg.budget, budget_pass_only=cfg.budget_podium_pass_only, underuse_floor=cfg.vram_underuse_floor,
                agent_prefill_ref=cfg.agent_prefill_ref, agent_ttft_ref=cfg.agent_ttft_ref,
                agent_longctx_weight=cfg.agent_longctx_weight)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    blob = json.dumps({"meta": meta, "models": rows}, default=str).replace("</", "<\\/")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    Path(path).write_text(template.replace("__DATA__", blob), encoding="utf-8")


def write_dashboards(cfg, results, stamp, cpu_source, hardware, partial=False):
    """Write results/benchmark_dashboard_<stamp>.html and copy it to <script folder>/benchmark_dashboard.html."""
    stamped = cfg.output_dir / f"benchmark_dashboard_{stamp}.html"      # one per run, in ./results
    latest = SCRIPT_DIR / "benchmark_dashboard.html"                    # always the newest, next to benchmark.py
    render_dashboard(results, cfg, stamped, cpu_source, hardware, partial)
    shutil.copy2(stamped, latest)
    return stamped, latest
