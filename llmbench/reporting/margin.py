"""How much an eval score can move by chance: the 95% margin of error for the number of questions asked."""
from __future__ import annotations

import math

from ..config import EVAL_SPECS

ROUGH_PROFILES = ("test", "quick")


def moe_pct(n):
    """Worst-case (p = 0.5) 95% margin of error in percentage points for n questions: 98 / sqrt(n)."""
    return 98 / math.sqrt(n) if n else None


def margin_of_error(cfg, results):
    """{"profile", "per_dataset": {key: {"n", "moe"}}, "average": {"n", "moe"}, "rough": bool} for the datasets run.
    The average's margin uses all its questions together (it is a mean over datasets of similar size)."""
    per, total = {}, 0
    for key in EVAL_SPECS:
        ns = [r.get(f"eval_{key}_n") for r in results if r.get(f"eval_{key}_n")]
        if ns:
            n = max(ns)
            per[key] = {"n": n, "moe": moe_pct(n)}
            total += n
    if not per:
        return None
    return {"profile": cfg.eval_profile, "per_dataset": per, "average": {"n": total, "moe": moe_pct(total)},
            "rough": cfg.eval_profile in ROUGH_PROFILES}
