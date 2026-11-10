"""--compare: combine finished runs into one dashboard, e.g. a model tested on Strata with the models of a llama-swap run.

    python benchmark.py --compare                      newest llama-swap run + newest Strata run (from OUTPUT_DIR)
    python benchmark.py --compare A.json B.json ...    these runs (benchmark_<stamp>.json files); the first is the base

Nothing is measured. Every model keeps its own measurements and eval results; what is computed again is everything
that compares models with each other (the Agent score's best-model bar, tie zone and long-context blend, the sweet
spot, budget eligibility), over all models together and with the current scoring code and the base run's settings.
So the combined ranking is the one a single run with all these models would give, provided the runs asked the same
questions. That is checked first:

  must match (else refused, --force to combine anyway): eval seed, temperature, thinking budget(s), streaming /
      loop-abort / salvage / budget-forcing switches. These change which questions are asked or what is measured.
  noted: different datasets, question counts or long-context lengths (a model missing a dataset or questions is then
      shown "not ranked: incomplete evals", exactly as within one run), a different tool version or hardware.

Run each side with the same .env eval settings (a funnel run with one model goes straight through to the final
profile, so a lone Strata model answers the same questions as a llama-swap run's finalists).
Output: OUTPUT_DIR/benchmark_compare_<stamp>.html, copied to benchmark_compare.html next to benchmark.py. Each model
carries a "Run" label; the dashboard's Run filter shows or hides a run's rows (scores stay as computed together).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import EVAL_SPECS, SCRIPT_DIR
from .reload import config_from_json, rows_from_json
from .reporting.dashboard import render_dashboard
from .scoring import finalize

# evaluation settings that change which questions are asked or what an answer means
_MUST_MATCH = {"seed": "eval seed (EVAL_SEED)", "temperature": "temperature (EVAL_TEMPERATURE)",
               "server_sampling_models": "models using their own sampling (EVAL_SERVER_SAMPLING_MODELS)",
               "think_budget": "thinking budget (EVAL_THINK_BUDGET)",
               "think_budgets": "per-dataset thinking budgets (EVAL_THINK_BUDGET_<DATASET>)",
               "stream": "streaming (EVAL_STREAM)", "loop_abort": "loop abort (EVAL_LOOP_ABORT)",
               "salvage": "partial-answer salvage (EVAL_SALVAGE)", "budget_forcing": "budget forcing (EVAL_BUDGET_FORCING)"}


def _load(path):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"[error] --compare: could not read {path}: {e}")
    if not isinstance(doc, dict) or "models" not in doc:
        sys.exit(f"[error] --compare: {path} is not a benchmark results file.")
    return doc


def _backend(doc):
    return ((doc.get("run") or {}).get("backend") or "llama-swap").lower()


def find_runs(cfg, args):
    """Explicit files / folders, else the newest run per backend in OUTPUT_DIR (llama-swap first)."""
    if args:
        paths = []
        for a in args:
            p = Path(a).expanduser()
            p = p / "results.json" if p.is_dir() else p
            if not p.is_file():
                sys.exit(f"[error] --compare: {p} not found.")
            paths.append(p)
        if len({p.resolve() for p in paths}) < len(paths):
            sys.exit("[error] --compare: the same file was given twice.")
        if len(paths) < 2:
            sys.exit("[error] --compare needs at least two runs (or none, for the newest llama-swap + Strata runs).")
        return paths
    newest = {}
    for f in sorted(glob.glob(str(cfg.output_dir / "benchmark_*.json")), key=os.path.getmtime, reverse=True):
        try:
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("models") and (doc.get("run") or {}).get("mode") != "demo":
            newest.setdefault(_backend(doc), Path(f))
    if len(newest) < 2:
        found = ", ".join(f"{b}: {p.name}" for b, p in newest.items()) or "none"
        sys.exit("[error] --compare without files combines the newest llama-swap run with the newest Strata run, but "
                 f"{cfg.output_dir} has runs from only one backend ({found}). Name the runs instead: "
                 "--compare results/benchmark_A.json results/benchmark_B.json")
    return sorted(newest.values(), key=lambda p: (_backend(json.loads(p.read_text(encoding='utf-8'))) != "llama-swap",
                                                  p.name))


def label_of(doc, path):
    run = doc.get("run") or {}
    b = "Strata" if _backend(doc) == "strata" else "llama-swap"
    rid = run.get("id") or Path(path).stem.replace("benchmark_", "")
    try:
        when = datetime.strptime(rid, "%Y%m%d_%H%M%S").strftime("%d %b %H:%M")
    except ValueError:
        when = rid
    return f"{b} {when}"


def scoring_generation(version):
    """'pre-2.3' or '2.3+': v2.3 changed what the Agent score measures (prefill at depth, the composite)."""
    try:
        major, minor = (int(x) for x in str(version).split(".")[:2])
    except (TypeError, ValueError):
        return "pre-2.3"
    return "2.3+" if (major, minor) >= (2, 3) else "pre-2.3"


def check(docs, labels):
    """-> (problems that refuse the merge, notes)."""
    problems, notes = [], []
    base_ev, base_lbl = docs[0].get("evaluation") or {}, labels[0]
    for doc, lbl in zip(docs[1:], labels[1:]):
        ev = doc.get("evaluation") or {}
        if not ev.get("datasets") or not base_ev.get("datasets"):
            notes.append(f"{lbl if not ev.get('datasets') else base_lbl} has no evaluations: its models are compared "
                         "on speed only and get no Agent score")
            continue
        for k, what in _MUST_MATCH.items():
            if (ev.get(k) or None) != (base_ev.get(k) or None):
                problems.append(f"{what}: {base_lbl} {base_ev.get(k)!r} vs {lbl} {ev.get(k)!r}")
        a, b = base_ev.get("datasets") or {}, ev.get("datasets") or {}
        if set(a) != set(b):
            only = lambda x, y: ", ".join(EVAL_SPECS.get(k, {}).get("short", k) for k in sorted(set(x) - set(y)))
            notes.append("datasets differ (" + "; ".join(x for x in (f"only {base_lbl}: {only(a, b)}" if set(a) - set(b)
                         else "", f"only {lbl}: {only(b, a)}" if set(b) - set(a) else "") if x)
                         + "): models without one are not ranked")
        for k in set(a) & set(b):
            if (a[k].get("rows"), a[k].get("revision")) != (b[k].get("rows"), b[k].get("revision")):
                notes.append(f"{EVAL_SPECS.get(k, {}).get('short', k)}: {a[k].get('rows')} questions"
                             f"{' @ ' + a[k]['revision'] if a[k].get('revision') else ''} in {base_lbl} vs "
                             f"{b[k].get('rows')}{' @ ' + b[k]['revision'] if b[k].get('revision') else ''} in {lbl}"
                             " (models with fewer questions than the largest set are not ranked)")
        la, lb = (base_ev.get("long_context") or {}).get("lengths"), (ev.get("long_context") or {}).get("lengths")
        if la != lb:
            notes.append(f"long-context lengths {la} vs {lb}: the Agent score compares long context over the "
                         "lengths every ranked model ran")
        if ev.get("profile") != base_ev.get("profile"):
            notes.append(f"profile {base_ev.get('profile')} vs {ev.get('profile')} (fine if the final question sets "
                         "match; see the question counts)")
    vers = {(d.get("tool") or {}).get("version") for d in docs}
    gens = {scoring_generation(v) for v in vers}
    if len(gens) > 1:
        problems.append("runs from before and after v2.3 (" + ", ".join(sorted(v or "?" for v in vers)) + "): the v2.3 "
                        "Agent score uses prefill at depth, which older runs did not measure")
    if len(vers) > 1:
        notes.append(f"made by different versions ({', '.join(sorted(v or '?' for v in vers))})")
    hw = {(d.get("host") or {}).get("hardware") for d in docs}
    if len(hw) > 1:
        notes.append("different host hardware: speed, power and VRAM are not like-for-like")
    return problems, notes


def compare_runs(cfg, args=None, force=False):
    paths = find_runs(cfg, args or [])
    docs = [_load(p) for p in paths]
    labels = [label_of(d, p) for d, p in zip(docs, paths)]
    for i, lbl in enumerate(labels):                      # two runs started in the same minute on one backend
        if labels.count(lbl) > 1:
            labels[i] = f"{lbl} ({Path(paths[i]).stem.replace('benchmark_', '')})"
    problems, notes = check(docs, labels)
    print("Combining " + " + ".join(f"{lbl} ({len(d.get('models') or [])} models, {Path(p).name})"
                                    for lbl, d, p in zip(labels, docs, paths)))
    if problems:
        print("These runs did not measure evals the same way, so their scores are not comparable:")
        for x in problems:
            print(f"    - {x}")
        if not force:
            sys.exit("[error] --compare refused. Rerun one side with the same settings, or add --force to combine "
                     "anyway (the dashboard then carries these notes).")
        notes = [f"settings differ: {x}" for x in problems] + notes
    for x in notes:
        print(f"  [note] {x}")

    rows, seen = [], {}
    for doc, lbl in zip(docs, labels):
        for r in rows_from_json(doc):
            r["source"] = lbl
            seen.setdefault(r["model"], []).append(r)
            rows.append(r)
    for name, rs in seen.items():                         # the same model in several runs: keep each, labelled
        if len(rs) > 1:
            for r in rs:
                r["model"] = f"{name} [{r['source']}]"

    base_cfg = config_from_json(cfg, docs[0])
    datasets = {}
    for doc in docs:                                      # every dataset of every run, the largest question set wins
        for k, d in ((doc.get("evaluation") or {}).get("datasets") or {}).items():
            if (d.get("rows") or 0) >= ((datasets.get(k) or {}).get("rows") or 0):
                datasets[k] = {"rows": d.get("rows"), "available": d.get("available"), "revision": d.get("revision"),
                               "since": d.get("since")}
    base_cfg.eval_info = datasets
    base_cfg.eval_longctx = any((d.get("evaluation") or {}).get("long_context") for d in docs)
    for r in rows:                                        # derived fields are computed again over the whole set
        r.pop("eligible", None)
    finalize(rows, base_cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = base_cfg.output_dir / f"benchmark_compare_{stamp}.html"
    host = docs[0].get("host") or {}
    backends = list(dict.fromkeys("Strata" if _backend(d) == "strata" else "llama.cpp via llama-swap" for d in docs))
    extra = {"compare": {"runs": [{"label": lbl, "file": Path(p).name, "models": len(d.get("models") or [])}
                                  for lbl, d, p in zip(labels, docs, paths)], "warnings": notes},
             "backend": " + ".join(backends)}
    render_dashboard(rows, base_cfg, out, host.get("cpu_power_source") or "none", host.get("hardware") or "",
                     extra_meta=extra)
    from .reporting.dashboard import copy_latest
    latest = copy_latest(out, SCRIPT_DIR / "benchmark_compare.html")
    ranked = sorted((r for r in rows if r.get("score_agent") is not None), key=lambda r: -r["score_agent"])
    if ranked:
        print("\nAgent score over all combined models (top 10):")
        for r in ranked[:10]:
            print(f"    {r['score_agent']:5.1f}  {r['model']}" + ("" if r["model"].endswith("]") else f"  [{r['source']}]"))
    print(f"\nDashboard: {out}\nLatest:    {latest}")
    return out, rows
