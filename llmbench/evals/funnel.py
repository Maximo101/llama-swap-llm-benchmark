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


def composite_estimates(rows, keys, cfg):
    """{model: (score, se) or None} on the Agent composite (v2.3), plus the basis text."""
    from .. import agent
    complete = [r for r in rows if _complete(r, keys)]
    _, has_tools = agent.eval_keys(rows)
    rankable = [r for r in complete if not agent.exclusion(r, cfg, has_tools)]
    b = agent.basis(cfg, rankable or complete, rows)
    est = {}
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
    return est, b["text"]


def gate(rows, keys, leaders, conf=0.95, lc_weight=0.0, lc_n=None, cfg=None):
    """Decide which models go on to the next stage.

    rows      result rows evaluated in this stage (dicts with eval_<k>_pct / _n / _err, longctx_<len>_pct)
    keys      datasets that count
    cfg       the run's Config: rank on the Agent composite (v2.3). Without it: the v2.2 figure (model_estimate).
    Returns (keep: set of model names, report: {model: {"score", "se", "ub", "lb", "kept", "why"}}, bar)."""
    z = z_one_sided(conf)
    if cfg is not None:
        est, _ = composite_estimates(rows, keys, cfg)
    else:
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
