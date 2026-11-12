"""EVAL_PROFILE=funnel: confidence-interval gates between nested eval stages (pure logic, unit-tested).

WHY NOT "KEEP THE TOP K"
  With 28 models whose best six are within a few points of each other, a `quick` stage (11 questions per dataset)
  is too noisy to pick a top 5: in a simulation the true best model was screened out in 18-33% of runs. A model is
  only dropped here when the data says it is CLEARLY below the leaders:

      keep  <=>  upper bound of the model  >=  lower bound of the EVAL_LEADERS-th best model
      (one-sided bounds at EVAL_FUNNEL_CONF, default 95%)          ... and never fewer than EVAL_LEADERS models.

  In the same simulation this lost the true best in <=0.1% of runs, picked as well as running everything at the
  final profile, and still saved ~40-50% of the eval work. A crowded field keeps more models (and saves less);
  a spread-out field drops more.

WHAT IS COMPARED
  The quantity the Agent score ranks on (v2.3, llmbench/agent.py): the composite of the evals (mean of the dataset
  scores except tool calling), tool calling and long-context retrieval over the lengths every Agent-rankable model
  of the stage ran, weighted AGENT_* (0.45 / 0.30 / 0.25 by default). Its standard error combines the binomial error
  of each part (weights squared; the evals part is a mean of D datasets: sqrt(sum SE_d^2) / D). The context factor
  and the retry penalty are NOT applied here: a gate only drops a model whose quality is clearly below the leaders,
  so leaving them out keeps more models (the safe side). A model that cannot get an Agent score (context below
  AGENT_CTX_MIN, tools below MIN_TOOLCALL_PCT) is still gated, on the parts it has. Models with incomplete evals (a
  dataset that failed) cannot be ranked and are dropped.
  Without a Config (unit tests, older callers) it is the v2.2 figure: the mean of the dataset scores, blended with
  long-ctx at lc_weight when every model has it.

v2.3.3: TWO THINGS CHANGED (with a Config)
  1. HARD RULES ARE DECIDED OUTSIDE THE FUNNEL. A model that cannot get an Agent score for a measured reason (loaded or
     usable context below AGENT_CTX_MIN, tool calling below MIN_TOOLCALL_PCT: agent.exclusion) is not "a weaker
     model that lost the ranking": it fails a rule. It is labelled "No Agent score", never "Screened out", it is not
     counted when the leaders' bar is set, and with EVAL_HARD_FAIL=stop (default) it is not asked further stages.
  2. THE BOUNDS USE THE REDUCIBLE ERROR ONLY. The tool-calling (32 cases) and long-context (8 per length) parts of the
     composite are asked in full at the first stage; more stages do not shrink their error, so counting it in a gate
     only kept every model alive (v2.3.1 real runs: 1 of 26 pruned). The gate now uses the error of the evals part
     (which does shrink with every stage) plus a margin of EVAL_FUNNEL_MARGIN points:

         keep  <=>  score + z x SE_evals + margin  >=  (score - z x SE_evals) of the EVAL_LEADERS-th best model
"""
from __future__ import annotations

import math
from statistics import NormalDist


def z_one_sided(conf):
    return NormalDist().inv_cdf(conf)


def _se_pct(p_pct, n):
    """Binomial standard error in percentage points; p is kept off 0 / 1 so a perfect small sample is not 'certain'."""
    if not n:
        return None
    p = min(max(p_pct / 100, 0.5 / n), 1 - 0.5 / n)
    return 100 * math.sqrt(p * (1 - p) / n)


def model_estimate(r, keys, lc_weight=0.0, lc_n=None):
    """v2.2 figure (no Config): (score, standard error) over the datasets `keys`, or None when a dataset has no
    score. Kept for callers without a Config; the gate uses composite_estimate() when it has one."""
    pcts, ses = [], []
    for k in keys:
        p, n, err = r.get(f"eval_{k}_pct"), r.get(f"eval_{k}_n") or 0, r.get(f"eval_{k}_err") or 0
        if p is None or n - err <= 0:
            return None
        pcts.append(p)
        ses.append(_se_pct(p, n))
    if not pcts:
        return None
    mean = sum(pcts) / len(pcts)
    se = math.sqrt(sum(s * s for s in ses)) / len(pcts)
    lc = r.get("longctx_avg_pct")
    if lc_weight and lc is not None:
        lc_se = _se_pct(lc, lc_n) if lc_n else 0.0
        mean = (1 - lc_weight) * mean + lc_weight * lc
        se = math.sqrt(((1 - lc_weight) * se) ** 2 + (lc_weight * lc_se) ** 2)
    return mean, se


def _complete(r, keys):
    return all(r.get(f"eval_{k}_pct") is not None
               and (r.get(f"eval_{k}_n") or 0) - (r.get(f"eval_{k}_err") or 0) > 0 for k in keys)


def hard_failures(rows, keys, cfg):
    """{model: reason} for models that fail a hard Agent rule on what they have measured so far (v2.3.3): loaded or
    usable context below AGENT_CTX_MIN, tool calling below MIN_TOOLCALL_PCT. Pure rules: no estimate, no margin of
    error, no comparison with other models. Models with a missing dataset are not judged (they are 'incomplete')."""
    from .. import agent
    done = [r for r in rows if _complete(r, keys)]
    _, has_tools = agent.eval_keys(rows)
    lengths = agent.run_lc_lengths(rows)
    out = {}
    for r in done:
        why = agent.exclusion(r, cfg, has_tools, lengths)
        if why:
            out[r["model"]] = why.replace("no Agent score: ", "", 1)
    return out


def composite_estimates(rows, keys, cfg):
    """{model: (score, se) or None} on the Agent composite (v2.3), plus the basis text."""
    est, text, _ = composite_estimates_red(rows, keys, cfg)
    return est, text


def composite_estimates_red(rows, keys, cfg):
    """As composite_estimates, plus {model: reducible standard error}: the error of the evals part alone, weighted
    as in the composite (the part more questions can still shrink)."""
    from .. import agent
    complete = [r for r in rows if _complete(r, keys)]
    _, has_tools = agent.eval_keys(rows)
    rankable = [r for r in complete if not agent.exclusion(r, cfg, has_tools)]
    b = agent.basis(cfg, rankable or complete, rows)
    est, red = {}, {}
    for r in rows:
        if r not in complete:
            est[r["model"]] = None
            continue
        e = agent.composite(r, b["weights"], b["ekeys"], b["has_tools"], b["lengths"], b["lc_n"], _se_pct)
        if e is None:                      # e.g. no long-ctx at the common lengths: gate it on the parts it has
            w = {k: v for k, v in b["weights"].items() if k != "longctx"}
            tot = sum(w.values()) or 1
            e = agent.composite(r, {k: v / tot for k, v in w.items()}, b["ekeys"], b["has_tools"], [], b["lc_n"],
                                _se_pct)
        est[r["model"]] = e
        w_ev = b["weights"].get("evals")
        ev = agent.composite(r, {"evals": w_ev}, b["ekeys"], False, [], b["lc_n"], _se_pct) if w_ev else None
        red[r["model"]] = ev[1] if ev else (e[1] if e else None)
    return est, b["text"], red


def gate(rows, keys, leaders, conf=0.95, lc_weight=0.0, lc_n=None, cfg=None, margin=None, hard_stop=None):
    """Decide which models go on to the next stage.

    rows      result rows evaluated in this stage (dicts with eval_<k>_pct / _n / _err, longctx_<len>_pct)
    keys      datasets that count
    cfg       the run's Config: rank on the Agent composite with the v2.3.3 rule (hard rules first, reducible error,
              margin). Without it: the v2.2 figure (model_estimate) and the v2.2 rule, kept for unit tests.
    margin    points added to a model's upper bound (default cfg.eval_funnel_margin)
    hard_stop True: a model that fails a hard rule is not asked further stages (default cfg.eval_hard_fail == "stop")
    Returns (keep: set of model names, report, bar). report[model] = {"state": "keep" | "screened" | "no_agent" |
    "incomplete", "score", "se" (total), "se_red" (what the gate used), "ub", "lb", "gap" (ub - bar, signed, in
    points), "kept", "why"}. A hard-rule model is never in the leaders' bar; with hard_stop it is not in `keep`."""
    z = z_one_sided(conf)
    if cfg is None:
        return _gate_v22(rows, keys, leaders, z, lc_weight, lc_n)
    margin = cfg.eval_funnel_margin if margin is None else margin
    hard_stop = (cfg.eval_hard_fail == "stop") if hard_stop is None else hard_stop
    hard = hard_failures(rows, keys, cfg)
    cand = [r for r in rows if r["model"] not in hard]
    est, _, red = composite_estimates_red(cand, keys, cfg) if cand else ({}, "", {})
    ranked = sorted((m for m, e in est.items() if e), key=lambda m: -est[m][0])
    report = {}
    bar = None
    keep = set()
    if ranked:
        kth = ranked[min(leaders, len(ranked)) - 1]
        bar = est[kth][0] - z * red[kth]
        keep = set(ranked[:leaders])
    for r in rows:
        m = r["model"]
        if m in hard:
            if not hard_stop:
                keep.add(m)
            report[m] = {"state": "no_agent", "kept": m in keep, "why": f"hard rule: {hard[m]}"}
            continue
        e = est.get(m)
        if e is None:
            report[m] = {"state": "incomplete", "kept": False, "why": "incomplete evals: cannot be ranked"}
            continue
        sr = red[m]
        ub, lb = e[0] + z * sr + margin, e[0] - z * sr
        gap = ub - bar
        if gap >= 0:
            keep.add(m)
        kept = m in keep
        report[m] = {"state": "keep" if kept else "screened", "score": e[0], "se": e[1], "se_red": sr, "ub": ub,
                     "lb": lb, "gap": gap, "kept": kept,
                     "why": "" if kept else (f"upper bound {ub:.2f} (score {e[0]:.2f} + {z:.2f} x {sr:.2f} + margin "
                                             f"{margin:g}) is {-gap:.2f} below the leaders' bar {bar:.2f}")}
    return keep, report, bar


def _gate_v22(rows, keys, leaders, z, lc_weight, lc_n):
    """The v2.2 rule (no Config): total standard error, no hard rules, no margin."""
    use_lc = lc_weight > 0 and rows and all(r.get("longctx_avg_pct") is not None for r in rows)
    est = {r["model"]: model_estimate(r, keys, lc_weight if use_lc else 0.0, lc_n) for r in rows}
    ranked = sorted((m for m, e in est.items() if e), key=lambda m: -est[m][0])
    report = {}
    if not ranked:
        return set(), {m: {"kept": False, "why": "no complete eval score"} for m in est}, None
    kth = ranked[min(leaders, len(ranked)) - 1]
    bar = est[kth][0] - z * est[kth][1]
    keep = set(ranked[:leaders])
    for m, e in est.items():
        if e is None:
            report[m] = {"kept": False, "why": "incomplete evals: cannot be ranked"}
            continue
        ub, lb = e[0] + z * e[1], e[0] - z * e[1]
        if ub >= bar:
            keep.add(m)
        kept = m in keep
        report[m] = {"score": e[0], "se": e[1], "ub": ub, "lb": lb, "kept": kept,
                     "why": ("" if kept else f"upper bound {ub:.1f} < leaders' bar {bar:.1f}")}
    return keep, report, bar
