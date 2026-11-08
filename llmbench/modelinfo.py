"""What the server actually loaded: launch flags, /props and the GGUF header, flattened into result-row fields.

Sources, in order of trust:
  1. the llama-server process's command line (/proc/<pid>/cmdline; the RAM sampler already finds the process)
  2. llama-swap's /running entry, if it includes the command (used when /proc is not readable)
  3. llama-server's /props (n_ctx, model_path, total_slots)
  4. the GGUF file header (architecture, experts, attention layout, total/active params, file size)
Everything is optional: a missing source leaves its columns blank and prints one warning per run.
"""
from __future__ import annotations

import os

from .client import get_props, props_ctx, running_cmd
from .gguf import KV_PRECISION, GGUFError, find_readable, inspect, kv_bytes
from .telemetry.host import offload_summary, parse_launch_args

_CACHE: dict = {}           # GGUF path -> inspect() result (a model benchmarked twice is read once)
GB = 1024 ** 3


def kv_label(k, v):
    """'q8_0' when K and V match, 'q8_0/q4_0' when they differ, None when unknown."""
    if not k and not v:
        return None
    k, v = k or "f16", v or "f16"
    return k if k == v else f"{k}/{v}"


def kv_rank(label):
    """Precision rank of a KV label (the LOWER of K and V). None if unknown."""
    if not label:
        return None
    ranks = [KV_PRECISION.get(x) for x in label.split("/")]
    return None if None in ranks else min(ranks)


def kv_at_least(label, minimum):
    """True / False when both are known, None when the model's KV type is unknown."""
    a, b = kv_rank(label), KV_PRECISION.get(minimum)
    return None if a is None or b is None else a >= b


def _launch(cfg, session, host, model, strata):
    if strata:
        return None, "none"
    info = host.ram.launch_info() if host is not None else None
    if info:
        return info, "process"
    argv = running_cmd(cfg, session, model)
    if argv:
        info = parse_launch_args(argv[1:] if argv and not argv[0].startswith("-") else argv)
        info["pid"] = None
        return info, "/running"
    return None, "none"


def _apply_launch(r, info):
    r["offload"] = offload_summary(info) or "none"
    r["kv_type"] = kv_label(info["ctk"], info["ctv"]) or "f16"        # llama.cpp's default when no flag is given
    r["flash_attn"] = info["fa"] or "auto"
    r["ngl"] = info["ngl"]
    r["n_cpu_moe"] = info["n_cpu_moe"]
    r["ubatch"] = info["ubatch"] or 512
    r["batch"] = info["batch"] or 2048
    r["parallel"] = info["parallel"]
    # Shared KV pool across slots? Explicit flag wins; with -np left at auto, current llama.cpp enables it by default
    # (older builds did not), so an unset flag stays None = unknown, and the eval runner then assumes a shared pool.
    r["kv_unified"] = info.get("kv_unified")
    if r["kv_unified"] is None and info.get("parallel") not in (None, -1):
        r["kv_unified"] = False         # explicit -np N without -kvu: every slot owns ctx / N
    r["_launch_argv"] = info.get("argv")
    r["mmproj"] = os.path.basename(info["mmproj"]) if info["mmproj"] else None
    r["jinja"] = info["jinja"]
    r["override_tensor"] = info["ot"]


def _apply_gguf(cfg, r, g, kv):
    r["gguf_gb"] = g["file_bytes"] / GB
    r["bpw"] = g["bpw"]
    r["gguf_arch"] = g["gguf_arch"]
    r["attn_type"] = g.get("attn_type")
    r["n_layers"] = g.get("n_layers")
    r["full_attn_layers"] = g.get("full_attn_layers")
    r["swa_layers"] = g.get("swa_layers")
    r["n_ctx_train"] = g.get("n_ctx_train")
    r["expert_count"], r["expert_used"] = g.get("expert_count"), g.get("expert_used")
    if g.get("params_total"):
        r["params_b"] = round(g["params_total"] / 1e9, 1)
    if g.get("params_active"):
        r["active_b"] = round(g["params_active"] / 1e9, 1)
    r["arch"] = "MoE" if g.get("expert_count") else "Dense"
    k, v = (kv.split("/") + [kv])[:2] if kv else ("f16", "f16")
    per_tok, swa_fixed = kv_bytes(g, k, v)
    if per_tok is not None:
        r["kv_kib_token"] = per_tok / 1024
        r["kv_swa_gb"] = (swa_fixed or 0) / GB or None
        r["kv_approx"] = bool(g.get("kv_approx")) or not kv       # KV type unknown: f16 assumed


def collect(cfg, session, host, model, r, strata, warned):
    """Fill r with launch flags, /props and GGUF facts. `warned` is a per-run dict so each warning prints once."""
    props = get_props(cfg, session, model)
    r["ctx_size"] = props_ctx(props) or (cfg.strata_ctx or cfg.strata_ctx_detected if strata else None)
    if props.get("total_slots"):
        r["parallel"] = props["total_slots"]
    info, source = _launch(cfg, session, host, model, strata)
    if info:
        _apply_launch(r, info)
        r["ctx_size"] = r["ctx_size"] or info["ctx"]
        if props.get("total_slots"):
            r["parallel"] = props["total_slots"]
    elif not strata and not warned.get("launch"):
        warned["launch"] = True
        print("  [warn] launch flags not readable (no matching process in /proc, and /running shows no command): "
              "KV type, -fa, -ub/-b and CPU offload are blank. Run on the server host, as root, and check "
              "RAM_PROCESS_PATTERN with --check")
    if not cfg.gguf_inspect:
        return
    path = (info or {}).get("model_path") or props.get("model_path")
    if not path:
        return
    real = find_readable(path, (info or {}).get("pid"), cfg.gguf_path_map, cfg.gguf_search_dirs)
    if not real:
        if not warned.get("gguf"):
            warned["gguf"] = True
            print(f"  [warn] model file not readable from here ({path}): GGUF size, BPW and architecture are blank. "
                  "Run as root (reads it through /proc/<pid>/root) or set GGUF_PATH_MAP=/container/path=/host/path")
        return
    r["_gguf_path"] = real                      # for the result-cache fingerprint (never exported)
    try:
        if real not in _CACHE:
            _CACHE[real] = inspect(real)
        g = _CACHE[real]
    except (OSError, GGUFError, ValueError) as e:
        if not warned.get("gguf_err"):
            warned["gguf_err"] = True
            print(f"  [warn] could not read the GGUF header of {real}: {e}")
        return
    _apply_gguf(cfg, r, g, r.get("kv_type"))
    mm = (info or {}).get("mmproj")
    if mm:
        mreal = find_readable(mm, (info or {}).get("pid"), cfg.gguf_path_map, cfg.gguf_search_dirs)
        if mreal:
            try:
                r["mmproj_gb"] = os.path.getsize(mreal) / GB
            except OSError:
                pass


def max_ctx_at(r, pct):
    """Context (tokens) that would fit with VRAM at `pct` % of the GPU, from the measured VRAM at the loaded context
    and the KV cost per token: ctx + (pct% x total - measured) / bytes-per-token. None when KV is not on the GPU
    (-nkvo), or a figure is missing. Capped at the model's training context."""
    kib, ctx, used, total = r.get("kv_kib_token"), r.get("ctx_size"), r.get("vram_gb"), r.get("vram_total_gb")
    if not kib or not ctx or used is None or not total or "KV on CPU" in (r.get("offload") or ""):
        return None
    extra = (pct / 100 * total - used) * GB / (kib * 1024)
    n = int(max(0, ctx + extra) // 1024 * 1024)
    return min(n, int(r["n_ctx_train"])) if r.get("n_ctx_train") else n
