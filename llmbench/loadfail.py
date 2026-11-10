"""Why did a model fail to load? (v2.3)

The benchmark cannot change how llama-swap starts a model (the command, its -c and -ngl live in llama-swap's config),
so it cannot retry a failed load with other settings itself. What it does: label the failure ("Failed to load (out of
memory)"), name the usual fix, and show the evidence: llama-swap's own log lines about the failure and, when the GGUF
file is readable here (GGUF_SEARCH_DIRS / GGUF_PATH_MAP), the model's weight size, trained context and KV-cache cost.
The label goes in front of the row's error, so the log line and the dashboard's badge both start with it.
"""
from __future__ import annotations

import re

PREFIX = "Failed to load"
_LOAD_FAIL = re.compile(r"exited prematurely|failed to load|upstream command|error loading model|"
                        r"model .* not (found|loaded)|health ?check|HTTP 50[023]", re.I)
_UNREACHABLE = re.compile(r"connection refused|max retries exceeded|name or service not known|failed to establish", re.I)
_LOG_HINT = re.compile(r"out of memory|cudamalloc|failed to allocate|unable to allocate|not enough memory|"
                       r"outofdevicememory|error|failed|exceeds|timed? ?out|unknown|invalid argument", re.I)

FIT = ("add --fit on to the model's llama-swap command and remove its -c and -ngl, so llama-server lowers the context "
       "and offloads layers until it fits (--fit-ctx sets the smallest context it may pick, e.g. 16384); or lower -c "
       "yourself, offload experts with --n-cpu-moe, or use a q8_0 / q4_0 KV cache (-ctk / -ctv)")
# (kind, regex over llama-swap's log lines + the error, advice). First match wins; the order matters.
CAUSES = [
    ("out of memory", re.compile(r"out of memory|cudamalloc failed|failed to allocate|unable to allocate|"
                                 r"not enough memory|outofdevicememory|cuda error 2\b|insufficient memory", re.I), FIT),
    ("bad launch flag", re.compile(r"invalid argument|unknown argument|error: unrecognized|unknown option", re.I),
     "llama-server rejected a flag in the model's llama-swap command (a typo, or a flag this llama.cpp build does not "
     "know yet, e.g. --fit on an older image): fix the command or update the llama-swap image"),
    ("model file not found", re.compile(r"failed to open|no such file|cannot open|gguf_init_from_file", re.I),
     "llama-server could not open the GGUF: check the -m path in llama-swap's command (inside the container, e.g. "
     "/models/...) and the volume mapping"),
    ("unsupported model", re.compile(r"unknown model architecture|unsupported model|unknown pre-tokenizer|"
                                     r"wrong number of tensors|invalid magic", re.I),
     "this llama.cpp build does not support the model (new architecture, or a damaged / partial download): update "
     "the llama-swap image or re-download the file"),
    ("load timed out", re.compile(r"health ?check.*(timed? ?out|timeout)|timed out waiting", re.I),
     "the model did not become ready in time: raise healthCheckTimeout in llama-swap's config (large or offloaded "
     "models can take minutes to load)"),
]
UNKNOWN = ("cause not reported, most often out of memory",
           "llama-server stopped while loading; most often the weights plus the KV cache at -c don't fit in VRAM: "
           + FIT + ". llama-swap's log shows the real reason")


def is_load_failure(err):
    return bool(err and _LOAD_FAIL.search(str(err)) and not _UNREACHABLE.search(str(err)))


def server_log_lines(cfg, session, model, n=4):
    """The last llama-swap log lines that look like errors (GET /logs, plain text). [] when not available."""
    try:
        resp = session.get(f"{cfg.base_url}/logs", headers={"Accept": "text/plain"}, timeout=5)
        if resp.status_code != 200 or "html" in resp.headers.get("Content-Type", "").lower():
            return []
        lines = resp.text[-200_000:].splitlines()
    except Exception:
        return []
    hits = [ln.strip() for ln in lines if _LOG_HINT.search(ln)]
    return [h[:220] for h in hits[-n:]]


def classify(err, logs=()):
    """(kind, advice) for a load failure from the error text and llama-swap's log lines."""
    text = "\n".join([str(err or "")] + list(logs))
    for kind, rx, advice in CAUSES:
        if rx.search(text):
            return kind, advice
    return UNKNOWN


def gguf_facts(cfg, model):
    """Weights GB, trained context and q8_0 KV GB per 1k tokens from the GGUF, when it can be read here. {} if not."""
    try:
        from .gguf import find_readable, inspect, kv_bytes
        path = find_readable(model if model.endswith(".gguf") else model + ".gguf", None,
                             cfg.gguf_path_map, cfg.gguf_search_dirs)
        if not path:
            return {}
        info = inspect(path)
        per_tok, _ = kv_bytes(info, "q8_0", "q8_0")
        return {"weights_gb": info.get("file_bytes", 0) / 1e9, "n_ctx_train": info.get("n_ctx_train"),
                "kv_gb_per_1k": per_tok * 1024 / 1e9 if per_tok else None}
    except Exception:
        return {}


def describe(cfg, session, model, err, vram_total_gb=None):
    """(label, hint) for a failed load, or (None, None) when the error is not a load failure. The label is what the
    log and the dashboard show first, e.g. 'Failed to load (out of memory)'."""
    if not is_load_failure(err):
        return None, None
    logs = server_log_lines(cfg, session, model)
    kind, advice = classify(err, logs)
    parts = [advice[0].upper() + advice[1:]]
    f = gguf_facts(cfg, model)
    if f.get("weights_gb"):
        s = f"Model: weights {f['weights_gb']:.1f} GB"
        if f.get("kv_gb_per_1k"):
            s += f", q8_0 KV about {f['kv_gb_per_1k'] * 16:.1f} GB per 16k of context"
        if f.get("n_ctx_train"):
            s += f", trained for {int(f['n_ctx_train']) // 1024}k (no point loading it with more)"
        if vram_total_gb:
            s += f", GPU {vram_total_gb:.0f} GB"
        parts.append(s)
    if logs:
        parts.append("llama-swap log: " + " | ".join(logs))
    return f"{PREFIX} ({kind})", ". ".join(p.rstrip(".") for p in parts) + "."


def hint(cfg, session, model, err, vram_total_gb=None):
    """The hint text only (see describe)."""
    return describe(cfg, session, model, err, vram_total_gb)[1]
