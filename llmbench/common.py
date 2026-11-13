"""Shared building blocks: exceptions, small stat helpers, time formatting and a thread-safe time series.

Nothing in here imports another llmbench module, so every other module can depend on it freely.
"""
from __future__ import annotations

import bisect
import statistics
import threading
import time
from datetime import datetime

__version__ = "2.3.5"
SCHEMA_VERSION = 1          # version of the results.json layout (bump when keys are renamed/removed)


_CLOCK0 = time.time() - time.monotonic()


def now():
    """Seconds since the epoch as of process start, advanced by the MONOTONIC clock (v2.3.1). Reads like time.time()
    but never jumps when NTP steps the system clock, so durations, telemetry windows and TimeSeries (which bisects
    on its timestamps) stay correct over a run of many hours. Use it for every interval; datetime.now() is still
    the source of the timestamps shown to people."""
    return _CLOCK0 + time.monotonic()


class BenchError(Exception):
    """A recoverable per-model failure (timeout, HTTP error, malformed response...)."""


class BenchTimeout(BenchError):
    """The request timed out (an eval question that thinks for too long ends up here)."""


# --------------------------------------------------------------------------- #
# Small statistics helpers (None-tolerant, so missing telemetry never raises)
# --------------------------------------------------------------------------- #

def avg(xs):
    return sum(xs) / len(xs) if xs else None


def vmax(xs):
    return max(xs) if xs else None


def col(rows, key):
    """Non-None values of `key` across a list of dict rows."""
    return [r[key] for r in rows if r.get(key) is not None]


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# --------------------------------------------------------------------------- #
# Per-model timing (v2.1): r["timing"] = {phase: seconds}
# --------------------------------------------------------------------------- #

def add_time(r, phase, seconds):
    """Add `seconds` to phase `phase` of result row `r` (phases: settle, load, speed, prefill_ctx, agent_ttft,
    cooldown, eval_load, longctx, eval:<dataset>, grading). Timing never raises."""
    try:
        t = r.get("timing")
        if not isinstance(t, dict):
            t = r["timing"] = {}
        t[phase] = round(t.get(phase, 0.0) + max(0.0, float(seconds)), 3)
    except Exception:
        pass


class Clock:
    """`with Clock(r, "load"):` adds the block's wall time to r["timing"]["load"]."""

    def __init__(self, r, phase):
        self.r, self.phase = r, phase

    def __enter__(self):
        self.t0 = now()
        return self

    def __exit__(self, *exc):
        add_time(self.r, self.phase, now() - self.t0)
        return False


# --------------------------------------------------------------------------- #
# Time formatting
# --------------------------------------------------------------------------- #

def stamp_line(label):
    """Print a timestamp marker line (ends up in the console and in the log)."""
    print(f"[timestamp] {label}: {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}", flush=True)


def _hms(seconds):
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    return h, m, s


def fmt_hms(seconds):
    """758 -> '00h:12m:38s' (used for the per-dataset time in the eval lines)."""
    return "{:02d}h:{:02d}m:{:02d}s".format(*_hms(seconds))


def fmt_hr_m_s(seconds):
    """36250 -> '10 hr 4 m 10 s' (used for the total time at the end of the run)."""
    return "{} hr {} m {} s".format(*_hms(seconds))


# --------------------------------------------------------------------------- #
# TimeSeries
# --------------------------------------------------------------------------- #

class TimeSeries:
    """Append-only (timestamp, value) series with O(log n) window queries.

    PERFORMANCE: the old samplers kept a plain list and scanned the WHOLE history on every request
    (`[s for t, s in samples if t0 <= t <= t1]`). A 10-hour run at 200 ms is ~180k samples, scanned
    several times per measurement. Timestamps are monotonic, so bisect gives the same answer in O(log n).
    All methods are safe to call from the sampler thread and the benchmark thread at the same time.
    """

    def __init__(self):
        self._t: list[float] = []
        self._v: list = []
        self._lock = threading.Lock()

    def append(self, value, t: float | None = None):
        with self._lock:
            self._t.append(now() if t is None else t)
            self._v.append(value)

    def window(self, t0: float, t1: float) -> list:
        """Values with t0 <= t <= t1. If nothing was sampled in the window (a very short request), the most
        recent sample is returned instead so callers still get a reading; [] only when the series is empty."""
        with self._lock:
            lo, hi = bisect.bisect_left(self._t, t0), bisect.bisect_right(self._t, t1)
            got = self._v[lo:hi]
            if not got and self._v:
                got = [self._v[-1]]
        return got

    def last(self):
        with self._lock:
            return self._v[-1] if self._v else None

    def __len__(self):
        with self._lock:
            return len(self._v)
