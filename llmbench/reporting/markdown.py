"""Markdown report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..common import avg
from ..config import EVAL_SPECS, LONGCTX_SLOTS, temperature_label
from .margin import margin_of_error

def _emoji(t):
    if t is None:
        return ""
    return "🟣" if t > 100 else "🔵" if t > 80 else "🟢" if t > 50 else "🟡" if t > 30 else "🩷"
def _f(v, d=1): return "–" if v is None else f"{v:.{d}f}"


def _md_header(results, cfg, speed):
    return ["---", "tags: [llm/benchmark, llama-swap]", f"date: {datetime.now():%Y-%m-%d}", "---", "",
            "# Local LLM Benchmark Suite Report", "", f"> {len(results)} models · {datetime.now():%Y-%m-%d %H:%M}"
            + (f" · VRAM limit {cfg.vram_limit_gb:g} GB" if speed else " · evaluations only (no speed/hardware telemetry)")]


def _md_top3(top):
    L = ["", "## Top 3 — Goldilocks", "", "| Rank | Model | TPS | VRAM | GPU W | Score |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(top, 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} | {_f(r['vram_gb'])} GB "
                 f"| {_f(r['gpu_w'], 0)} | {_f(r['goldilocks'], 0)} |")
    return L


def _md_speed_overview(ok):
    L = ["", "## Speed Overview", "", "| Rank | Model | Avg TPS |", "|---|---|---|"]
    for i, r in enumerate(sorted(ok, key=lambda r: -r["avg_tps"]), 1):
        L.append(f"| {i} | {r['model']} | {_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")
    return L


def _md_full_row(r):
    if r["status"] != "ok":
        return f"| {r['model']} | | | | | | | | | | | | | | | ERR: {r['error']} |"
    p = r["vram_pct"]
    vram = "–" if p is None else f"{r['vram_gb']:.1f} GB [{'=' * round(min(p, 100) / 10):<10}] {p:.0f}%"
    return (f"| {r['model']} | {_f(r['params_b'], 0)}B | {r['quant']} | {_f((r['ctx_size'] or 0) / 1024 or None, 0)}k | {_f(r['load_s'])} | {vram} | {_f(r['ram_spill_gb'])} | "
            f"{_f(r['gpu_w'], 0)} | {_f(r['cpu_w'], 0)} | {_f(r['energy_wh'], 3)} | {_f(r['j_per_token'])} | "
            f"{_f(r['ttft_ms'], 0)} | {_f(r['prefill_tps'], 0)} | {_f(r['tech_tps'])} | {_f(r['logic_tps'])} | "
            f"{_emoji(r['avg_tps'])} {_f(r['avg_tps'])} |")


def _md_full_comparison(results):
    return ["", "## Full Model Comparison", "",
            "| Model | Params | Quant | Ctx | Load s | VRAM | RAM spill GB | GPU W | CPU W | Wh | J/tok | TTFT ms | Prefill t/s | Tech | Logic | Avg |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + [_md_full_row(r) for r in results]


def _md_quant_analysis(ok):
    groups = {}
    for r in ok:
        groups.setdefault(r["quant"], []).append(r)
    L = ["", "## Quantization Analysis", "", "| Quant | Models | Avg TPS |", "|---|---|---|"]
    for q, rs in sorted(groups.items(), key=lambda kv: -avg([r["avg_tps"] for r in kv[1]])):
        L.append(f"| {q} | {len(rs)} | {_f(avg([r['avg_tps'] for r in rs]))} |")
    return L


def _md_eval_cells(r, evk):
    cells = []
    for k in evk:
        p, n = r.get(f"eval_{k}_pct"), r.get(f"eval_{k}_n")
        na, dg = r.get(f"eval_{k}_na") or 0, r.get(f"eval_{k}_degr") or 0
        cells.append("–" if p is None else f"{p:.1f}% ({n}" + (f", {na} N/A" if na else "")
                     + (f", {dg} Degr." if dg else "") + ")")
        sq = r.get(f"eval_{k}_sec")
        cells.append("–" if sq is None else f"{sq:.1f}")
    return cells


def _md_margin(cfg, results):
    m = margin_of_error(cfg, results)
    if not m:
        return []
    per = ", ".join(f"{EVAL_SPECS[k]['short']} ±{v['moe']:.0f} pts ({v['n']} q)" for k, v in m["per_dataset"].items())
    L = ["", f"**Test profile `{m['profile']}` — margin of error (95%):** {per}; average ±{m['average']['moe']:.0f} pts "
         f"({m['average']['n']} questions)."]
    if m["rough"]:
        L.append(f"> ⚠ `{m['profile']}` is only a rough guide: differences smaller than the margin are noise. Re-run the "
                 "finalists with `EVAL_PROFILE=standard` or `thorough` before choosing.")
    return L


def _md_evals(results, cfg, evk):
    L = ["", "## Accuracy & Pass Rates"] + _md_margin(cfg, results) + ["",
         "| Model | " + " | ".join(f"{EVAL_SPECS[k]['short']} | {EVAL_SPECS[k]['short']} s/q" for k in evk)
         + " | Avg | Retries |",
         "|---|" + "---|" * (2 * len(evk) + 2)]
    # funnel finalists first: a model screened out after fewer questions never sorts above one that finished
    ranked = sorted((r for r in results if r["status"] == "ok" and r.get("eval_avg_pct") is not None),
                    key=lambda r: (bool(r.get("eval_screened")), -r["eval_avg_pct"]))
    for r in ranked:
        rt = r.get("eval_retry_n")
        retry = "–" if rt is None else f"{rt} ({r['eval_retry_pct']:.0f}%)" + (" ⚠" if r["eval_retry_pct"] > 20 else "")
        name = r["model"] + (f" _(screened out at {r['eval_stage']})_" if r.get("eval_screened") else "")
        L.append(f"| {name} | " + " | ".join(_md_eval_cells(r, evk)) + f" | {r['eval_avg_pct']:.1f}% | {retry} |")
    unranked = [r for r in results if r["status"] == "ok" and (r.get("eval_incomplete") or r.get("eval_skipped"))]
    if unranked:
        L += ["", "Not ranked:"] + [f"- {r['model']}: " + (f"evals skipped ({r['eval_skipped']})" if r.get("eval_skipped")
                                                          else f"incomplete evals ({r['eval_incomplete']})")
                                     for r in unranked]
    L += ["", f"_Retries = questions that needed the No-Thinking retry (cut off, loop, thinking cap or timeout); many "
          "retries usually point to a chat-template or thinking problem, not the model. The Agent score loses "
          f"{cfg.agent_fallback_penalty:g}% × the retried share._"]
    L += ["", f"_Profile `{cfg.eval_profile}`: QA and math = accuracy, code = pass@1; the number in brackets is how many "
          "questions were asked; N/A = no answer because the reply was cut off, counted as wrong; Degr. = Fail: Degradation, i.e. the model fell into a repetition loop and was stopped early "
          "(the question is then answered by the retry or counted as wrong); s/q = mean latency of one "
          "question (request, retries and grading; measured with "
          + (f"up to {cfg.eval_concurrency} requests in flight; the Agent score uses the single-stream estimate"
             if cfg.eval_concurrency > 1 else "one request at a time") + ")._"]
    return L


def _md_methodology(cfg, speed, evk):
    L = ["", "## Methodology", ""]
    if speed:
        gpu = {"nvml": "NVML", "smi": "nvidia-smi"}.get(cfg.gpu_backend_used, "n/a")
        L += [f"- Median of {cfg.runs} run(s) per prompt; prompt cache disabled; each model is unloaded first, "
              "idle baseline sampled, then cold-loaded.",
              f"- GPU: `{gpu}` sampled every {cfg.gpu_poll_ms} ms. CPU power: RAPL (or Home Assistant). "
              + ("Energy = GPU hardware energy counter + mean CPU W × request time." if cfg.gpu_energy_counter else
                 "Energy = (mean GPU W + mean CPU W) × request time."),
              f"- Goldilocks = {'ln(params B)' if cfg.goldilocks_mode == 'log' else 'params B'} × avg TPS (total params)."]
    if evk:
        L.append(f"- Evals (run after the speed tests): {', '.join(EVAL_SPECS[k]['label'] for k in evk)}; profile "
                 f"`{cfg.eval_profile}`, seed {cfg.eval_seed}, {temperature_label(cfg)}. QA is string-matched "
                 "on the parsed final letter; math answers are normalised and compared numerically/textually with the reference; "
                 "code is executed against the dataset's tests in a sandbox.")
    return L


def _k(v):
    return "–" if not v else f"{v / 1024:.0f}k"


def _md_deploy(results, cfg):
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return []
    budget = cfg.budget
    L = ["", "## Deployment fit", "",
         f"_Max ctx = context that fits at {cfg.max_ctx_vram_pct:g}% VRAM (from the measured VRAM and the KV cost per "
         "token)._" + (f" _Budget profile `{budget['name']}`._" if budget else ""), "",
         "| Model | Arch | Attention | KV | Ctx | Max ctx | KV KiB/tok | BPW | Prefill 32k/64k | Prefill @depth | Agent TTFT | "
         + " | ".join(f"Long-ctx {v}" for v in LONGCTX_SLOTS.values()) + (" | Budget |" if budget else " |"),
         "|---" * (11 + len(LONGCTX_SLOTS) + (1 if budget else 0)) + "|"]
    for r in ok:
        pf = "/".join(_f(r.get(f"prefill_{v}_tps"), 0) for v in ("32k", "64k"))
        flag = " ⚠" if r.get("ctx_ok") is False else ""
        lc = [_f(r.get(f"longctx_{v}_pct"), 0) + ("%" if r.get(f"longctx_{v}_pct") is not None else "")
              for v in LONGCTX_SLOTS.values()]
        b = ("" if not budget else " | " + ("✅" if r.get("budget_pass") else "❌ " + (r.get("budget_fails") or "")))
        L.append(f"| {r['model']} | {r.get('arch') or '–'} | {r.get('attn_type') or '–'} | {r.get('kv_type') or '–'} | "
                 f"{_k(r.get('ctx_size'))}{flag} | {_k(r.get('max_ctx_vram'))} | {_f(r.get('kv_kib_token'), 0)} | "
                 f"{_f(r.get('bpw'), 2)} | {pf} | {_f(r.get('prefill_depth_tps'), 0)} | {_f(r.get('agent_ttft_s'))}{' s' if r.get('agent_ttft_s') is not None else ''} | " + " | ".join(lc) + b + " |")
    return L


def render_markdown(results, cfg, path):
    ok = [r for r in results if r["status"] == "ok" and r["avg_tps"] is not None]
    top = sorted((r for r in ok if r["eligible"]), key=lambda r: -r["goldilocks"])[:3]
    speed = bool(ok)
    evk = [k for k in EVAL_SPECS if any(r.get(f"eval_{k}_n") or r.get(f"eval_{k}_na") for r in results)]
    L = _md_header(results, cfg, speed)
    if speed:
        L += _md_top3(top) + _md_speed_overview(ok) + _md_full_comparison(results) + _md_quant_analysis(ok)
    if speed:
        L += _md_deploy(results, cfg)
    if evk:
        L += _md_evals(results, cfg, evk)
    L += _md_methodology(cfg, speed, evk)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
