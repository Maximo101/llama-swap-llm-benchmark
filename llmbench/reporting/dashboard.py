"""Self-contained interactive HTML dashboard (Chart.js); the template lives in dashboard.html."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from ..config import (AUTO_BUDGET, EVAL_SPECS, LONGCTX_SLOTS, PREFILL_SLOTS, SCRIPT_DIR, funnel_stages, temperature_label,
                      think_budget_value)
from ..evals.runner import tokens_label
from .colours import scales_meta
from .margin import margin_of_error

TEMPLATE_PATH = Path(__file__).with_name("dashboard.html")
# v2.3.1: Chart.js 4.4.1 (MIT, (c) Chart.js Contributors; vendor/LICENSE-chartjs.md) is written INTO every dashboard:
# the file opens without internet access and no longer runs a script fetched from a CDN at view time.
CHARTJS_PATH = Path(__file__).with_name("vendor") / "chart.umd.js"
CDN_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'


def chartjs_tag():
    """The <script> holding Chart.js itself; the CDN tag only if the bundled file is missing (a partial copy)."""
    try:
        js = CHARTJS_PATH.read_text(encoding="utf-8")
    except OSError:
        return CDN_TAG
    return "<script>" + js.replace("</script", "<\\/script") + "</script>"
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

def render_dashboard(results, cfg, path, cpu_source="none", hardware="", partial=False, extra_meta=None):
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
                     "temperature": temperature_label(cfg), "max_tokens_qa": tokens_label(cfg.eval_max_tokens_qa),
                     "max_tokens_code": tokens_label(cfg.eval_max_tokens_code), "code_sandbox": cfg.eval_sandbox_resolved or "subprocess",
                     "code_timeout": cfg.eval_code_timeout, "lcb_max_tests": cfg.eval_lcb_max_tests,
                     "retry_nothink": bool(cfg.eval_retry_on_length), "concurrency": cfg.eval_concurrency,
                     "stream": cfg.eval_stream, "think_budget": think_budget_value(cfg.eval_think_budget),
                     "think_budget_pct": cfg.eval_think_budget_pct if cfg.eval_think_budget == AUTO_BUDGET else None,
                     "think_budget_min": cfg.eval_think_budget_min if cfg.eval_think_budget == AUTO_BUDGET else None,
                     "think_budget_max_pct": cfg.eval_think_budget_max_pct if cfg.eval_think_budget == AUTO_BUDGET else None,
                     "loop_abort": cfg.eval_loop_abort,
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
                prefill_depth=cfg.prefill_depth, prefill_depth_auto=cfg.prefill_depth_auto,
                budget=cfg.budget, budget_pass_only=cfg.budget_podium_pass_only, underuse_floor=cfg.vram_underuse_floor,
                agent_prefill_ref=cfg.agent_prefill_ref, agent_ttft_ref=cfg.agent_ttft_ref,
                agent_longctx_weight=cfg.agent_longctx_weight,
                # v2.3: Agent score rules and the view settings (colour tiers, families)
                agent_tool_weight=cfg.agent_tool_weight, agent_ctx_min=cfg.agent_ctx_min,
                agent_ctx_curve=[list(x) for x in (cfg.agent_ctx_curve or ())], min_toolcall_pct=cfg.min_toolcall_pct,
                agent_ctx_pass_pct=cfg.agent_ctx_pass_pct,
                colours={"style": cfg.colour_style, "mode": cfg.colour_mode,
                         "scales": scales_meta(cfg.colour_scales or {})},
                families=[{"name": n, "patterns": list(p)} for n, p in (cfg.dash_families or ())])
    meta.update(extra_meta or {})
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    blob = json.dumps({"meta": meta, "models": rows}, default=str).replace("</", "<\\/")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    head, sep, rest = template.partition(CDN_TAG)       # str.replace would also expand "__DATA__"-like text in the JS
    page = head + (chartjs_tag() if sep else "") + rest.replace("__DATA__", blob)
    Path(path).write_text(page if sep else template.replace("__DATA__", blob), encoding="utf-8")


def write_dashboards(cfg, results, stamp, cpu_source, hardware, partial=False, update_latest=True):
    """Write results/benchmark_dashboard_<stamp>.html and copy it to <script folder>/benchmark_dashboard.html
    (update_latest=False: a demo run, which must never replace the real latest dashboard)."""
    stamped = cfg.output_dir / f"benchmark_dashboard_{stamp}.html"      # one per run, in ./results
    latest = SCRIPT_DIR / "benchmark_dashboard.html"                    # always the newest, next to benchmark.py
    render_dashboard(results, cfg, stamped, cpu_source, hardware, partial)
    if not update_latest:
        return stamped, f"{latest} not replaced (demo run)"
    return stamped, copy_latest(stamped, latest)


def copy_latest(src, latest):
    """Copy a dashboard to its 'latest' name next to benchmark.py. Not fatal when that file can't be written (e.g. it
    belongs to root on a server share and this run is another user): warn and return a note instead of the path."""
    try:
        shutil.copy2(src, latest)
        return latest
    except OSError as e:
        print(f"[warn] could not update {latest} ({e.strerror or e}); the dashboard is in {src}. On a share, the file "
              "may belong to another user (e.g. root from Unraid User Scripts).")
        return f"{latest} not updated ({e.strerror or e}); use {src}"
