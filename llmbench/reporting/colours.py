"""v2.3 dashboard colour tiers and model families (display only: nothing here changes a score).

Six tiers, worst to best: red (with a hazard sign, so colour is never the only signal), pink, amber, green, blue,
purple. A scale has five thresholds; "higher is better" scales read them as "at or above", "lower is better" scales
(times, retries, power) as "at or below", so their thresholds go down. Defaults are about usefulness, not about any
one GPU: e.g. generation below 30 tok/s is unusable, 50-79 the sweet spot, 100+ best. Override any scale with
COLOUR_<METRIC>=a,b,c,d,e in .env. COLOUR_MODE=relative spreads the tiers over the run's own range instead
(log scale for speeds and times), which suits small GPUs. VRAM % keeps its own sweet-spot logic (VRAM_SWEET_*).
"""
from __future__ import annotations

import os
import re

# metric: (default thresholds, "high" = higher is better | "low", short label, unit, log scale in relative mode)
SCALES = {
    "GEN":        ((30, 40, 50, 80, 100), "high", "Generation", "tok/s", True),
    "PREFILL":    ((500, 1000, 1500, 2500, 4000), "high", "Prefill (all lengths, at depth too)", "tok/s", True),
    "ACC":        ((50, 60, 70, 80, 90), "high", "Accuracy, tool calling, long-ctx", "%", False),
    "SCORE":      ((50, 60, 70, 80, 90), "high", "Agent / sweet-spot scores", "0-100", False),
    "TTFT_MS":    ((2000, 1000, 500, 250, 100), "low", "TTFT, short prompts", "ms", True),
    "AGENT_TTFT": ((40, 20, 13, 8, 5), "low", "Agent TTFT (20k prompt, cold)", "s", True),
    "DEPTH_TTFT": ((8, 4, 2.7, 1.6, 1), "low", "Depth TTFT (4k new at 64k)", "s", True),
    "SQ":         ((120, 90, 60, 30, 15), "low", "Seconds per question", "s", True),
    "RETRY":      ((25, 20, 15, 10, 5), "low", "Retries", "%", False),
    "POWER":      ((300, 250, 200, 150, 100), "low", "GPU power", "W", True),
}
STYLES = ("bands", "smooth")
MODES = ("absolute", "relative")


def parse_thresholds(metric, raw):
    """'30,40,50,80,100' -> (30.0, 40.0, 50.0, 80.0, 100.0), checked against the metric's direction.
    Raises ValueError with a readable message."""
    if metric not in SCALES:
        raise ValueError(f"unknown metric {metric!r} (known: {', '.join(SCALES)})")
    try:
        vals = tuple(float(x) for x in str(raw).replace(";", ",").split(",") if x.strip())
    except ValueError:
        raise ValueError(f"COLOUR_{metric}: numbers expected, got {raw!r}") from None
    if len(vals) != 5:
        raise ValueError(f"COLOUR_{metric}: five thresholds expected (6 tiers), got {len(vals)}")
    up = SCALES[metric][1] == "high"
    if any((b <= a) if up else (b >= a) for a, b in zip(vals, vals[1:])):
        raise ValueError(f"COLOUR_{metric}: thresholds must {'rise' if up else 'fall'} "
                         f"({'higher' if up else 'lower'} is better), got {raw!r}")
    return vals


def scales_from_env(environ=None):
    """{metric: thresholds} with COLOUR_<METRIC> overrides applied. Raises ValueError on a bad value."""
    env = os.environ if environ is None else environ
    out = {}
    for m, (default, *_rest) in SCALES.items():
        raw = (env.get(f"COLOUR_{m}") or "").strip().strip('"').strip("'")
        out[m] = parse_thresholds(m, raw) if raw else tuple(float(x) for x in default)
    return out


def scales_meta(scales):
    """The dashboard's view of the scales: {metric: {"t", "dir", "label", "unit", "log"}}."""
    return {m: {"t": [float(x) for x in (scales or {}).get(m, dflt)], "dir": d, "label": lbl, "unit": unit, "log": log}
            for m, (dflt, d, lbl, unit, log) in SCALES.items()}


# --------------------------------------------------------------------------- model families
def parse_families(raw):
    """DASH_FAMILIES="Gemma 4=*gemma-4*|Air*|Wicked*; BigBang=*BigBang*" -> (("Gemma 4", ("*gemma-4*", "Air*",
    "Wicked*")), ("BigBang", ("*BigBang*",))). Families are separated by ';', a name and its patterns by '=',
    patterns by '|'. Raises ValueError on a malformed entry."""
    raw = (raw or "").strip().strip('"').strip("'")
    out = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{part!r}: expected Name=pattern|pattern")
        name, pats = part.split("=", 1)
        pats = tuple(p.strip() for p in pats.split("|") if p.strip())
        if not name.strip() or not pats:
            raise ValueError(f"{part!r}: a family needs a name and at least one pattern")
        out.append((name.strip(), pats))
    return tuple(out)


def wildcard_re(pattern):
    """Case-insensitive pattern -> regex. '*' and '?' are wildcards matched against the whole name; a pattern without
    wildcards matches anywhere in the name (so 'gemma' finds 'agent-gemma-4-26B...')."""
    p = pattern.strip()
    if not any(c in p for c in "*?"):
        return re.compile(re.escape(p), re.I)
    return re.compile("^" + re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$", re.I)


def matches(name, pattern):
    return bool(wildcard_re(pattern).search(name))


def auto_family(name):
    """The leading word of a model name: the letters of its first '-' / '_' / '.' separated token
    (Qwen3.6-35B -> Qwen, Ornith-1.5 -> Ornith, LFM2.5-24B -> LFM). '' when it has none."""
    tok = re.split(r"[-_.]", name.strip(), maxsplit=1)[0]
    return "".join(c for c in tok if c.isalpha())
