"""--reload: rebuild the dashboard from a finished run's JSON with the current dashboard code.

Meant for working on the dashboard: no server, GPU or run is needed, and nothing is measured or rescored.
Every number shown is the one stored in the JSON (scores included), so the reloaded dashboard ranks models
exactly as the original run did. Only the presentation changes.

Source: `--reload` alone uses OUTPUT_DIR/results.json (the newest finished run); `--reload PATH` takes a
benchmark_<stamp>.json file or a folder that holds results.json.
Output: OUTPUT_DIR/benchmark_dashboard_<run id>_reload.html (the run's original dashboard is left alone),
copied to benchmark_dashboard.html next to benchmark.py like a normal run.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import typing
from pathlib import Path

from .config import EVAL_SPECS, SCRIPT_DIR
from .reporting.dashboard import render_dashboard

# Run-level eval keys: row key -> key in model["evaluation"] (the inverse of json_out._model_entry)
_EVAL_RUN = {"eval_avg_pct": "avg_pct", "eval_avg_sec": "avg_sec", "eval_retry_n": "retries",
             "eval_retry_pct": "retry_pct", "eval_quality": "quality_after_retry_penalty",
             "eval_incomplete": "incomplete", "eval_skipped": "skipped", "eval_screened": "screened",
             "eval_stage": "stage", "eval_stage1_avg_pct": "stage1_avg_pct", "eval_inflight": "in_flight",
             "eval_sec_basis": "sec_basis", "eval_cached_n": "cached", "eval_tok_think": "tokens_think",
             "eval_tok_answer": "tokens_answer"}
_SKIP_GROUPS = {"model", "status", "error", "timestamp", "evaluation", "timing", "other"}


def find_json(cfg, arg=None):
    """The JSON to reload: an explicit file, results.json in an explicit folder, else OUTPUT_DIR/results.json,
    else the newest OUTPUT_DIR/benchmark_*.json."""
    if arg:
        p = Path(arg).expanduser()
        if p.is_dir():
            p = p / "results.json"
        if not p.is_file():
            sys.exit(f"[error] --reload: {p} not found.")
        return p
    p = cfg.output_dir / "results.json"
    if p.is_file():
        return p
    runs = sorted(cfg.output_dir.glob("benchmark_*.json"), key=lambda f: f.stat().st_mtime)
    if not runs:
        sys.exit(f"[error] --reload: no results.json or benchmark_*.json in {cfg.output_dir} "
                 "(a run writes them when it finishes; JSON=false in .env turns them off).")
    return runs[-1]


def rows_from_json(doc):
    """Flatten the JSON's model entries back into the result rows the dashboard reads."""
    rows = []
    for m in doc.get("models") or []:
        r = {k: m.get(k) for k in ("model", "status", "error", "timestamp")}
        for group, vals in m.items():
            if group not in _SKIP_GROUPS and isinstance(vals, dict):
                r.update(vals)
        ev = m.get("evaluation") or {}
        for row_key, json_key in _EVAL_RUN.items():
            r[row_key] = ev.get(json_key)
        for key, d in (ev.get("datasets") or {}).items():
            for s, v in d.items():
                r[f"eval_{key}_{'conc' if s == 'concurrency' else s}"] = v
        t = m.get("timing") or {}
        r["timing"] = t.get("phases_s") or {}
        r.update({k: v for k, v in t.items() if k != "phases_s"})
        r.update(m.get("other") or {})            # fields the export didn't group are kept here, never dropped
        rows.append(r)
    return rows


def _coerce(value, current, hint):
    """JSON gives lists and strings; the dataclass may want a tuple or a Path."""
    if value is None:
        return None
    if isinstance(current, Path) or hint is Path or (isinstance(hint, str) and hint.startswith("Path")):
        return Path(value)
    if isinstance(current, tuple) or hint is tuple or (isinstance(hint, str) and hint.startswith("tuple")):
        return tuple(tuple(x) if isinstance(x, list) else x for x in value) if isinstance(value, list) else value
    return value


def config_from_json(cfg, doc):
    """The run's own settings on top of today's Config: the dashboard must describe the run as it was, not the
    current .env. Secrets were never exported, and OUTPUT_DIR stays the current one (where the output goes)."""
    snap = doc.get("config") or {}
    try:
        hints = typing.get_type_hints(type(cfg))
    except Exception:                         # unresolvable annotation: fall back to the current values' types
        hints = {f.name: f.type for f in dataclasses.fields(cfg)}
    for f in dataclasses.fields(cfg):
        if f.name in snap and f.name not in ("output_dir", "eval_cache_dir"):
            setattr(cfg, f.name, _coerce(snap[f.name], getattr(cfg, f.name), hints.get(f.name)))
    host = doc.get("host") or {}
    cfg.gpu_backend_used = host.get("gpu_backend") or snap.get("gpu_backend_used") or "none"
    ev = doc.get("evaluation") or {}
    if ev.get("code_sandbox"):
        cfg.eval_sandbox_resolved = ev["code_sandbox"]
    ds = ev.get("datasets") or {}
    cfg.eval_info = {k: {"rows": d.get("rows"), "available": d.get("available"), "revision": d.get("revision"),
                         "since": d.get("since")} for k, d in ds.items()}
    revs = dict(cfg.eval_revisions or {})
    revs.update({k: d["revision"] for k, d in ds.items() if d.get("revision")})
    cfg.eval_revisions = revs
    return cfg


def reload_dashboard(cfg, arg=None):
    src = find_json(cfg, arg)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"[error] --reload: could not read {src}: {e}")
    if not isinstance(doc, dict) or "models" not in doc:
        sys.exit(f"[error] --reload: {src} is not a benchmark results file.")
    run, host = doc.get("run") or {}, doc.get("host") or {}
    rows = rows_from_json(doc)
    if not any(r.get("status") == "ok" for r in rows):
        sys.exit(f"[error] --reload: {src} has no successful model, so there is nothing to show.")
    unknown = [k for k in (doc.get("evaluation") or {}).get("datasets", {}) if k not in EVAL_SPECS]
    if unknown:
        print(f"[warn] datasets this version doesn't know are left out: {', '.join(unknown)}")
    config_from_json(cfg, doc)
    run_id = run.get("id") or src.stem.replace("benchmark_", "")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"benchmark_dashboard_{run_id}_reload.html"
    render_dashboard(rows, cfg, out, host.get("cpu_power_source") or "none", host.get("hardware") or "",
                     partial=run.get("status") not in (None, "complete"))
    latest = SCRIPT_DIR / "benchmark_dashboard.html"
    shutil.copy2(out, latest)
    made_by = (doc.get("tool") or {}).get("version")
    print(f"Reloaded run {run_id} ({run.get('status') or 'unknown status'}, {len(rows)} models"
          + (f", written by v{made_by}" if made_by else "") + f") from {src}")
    print("Scores and measurements are the run's own; only the dashboard code is current.")
    print(f"Dashboard: {out}\nLatest:    {latest}")
    return out
