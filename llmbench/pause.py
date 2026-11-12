"""Pause and resume, and the GPU temperature guard.

A run can be PAUSED while it is going, by hand or by the GPU getting too hot, and then carries on by itself or when
you say so. It is the same process that waits: the lock stays, the settings page shows "paused", the model stays
loaded. Nothing is held back from the disk while it waits:

run_control.json        OUTPUT_DIR/run_control.json  {"pause": true|false}. The only way in from outside: the settings page
                        (Pause run / Resume run) and `python benchmark.py --pause` / `--resume` write it, the run reads it
                        about once a second. A file, not a signal, so it works on Windows and for a run on another machine
                        that shares OUTPUT_DIR. A leftover file from an earlier run is cleared when a run starts.
run_<stamp>.pause.json  written when a run pauses, deleted when it carries on or ends: why it paused, where it was
                        (phase, model i of N, dataset, question n of m), which models are finished, what restarts, and a
                        one-line summary of every row. The rows themselves are saved to run_<stamp>.rows.json at the same
                        moment. If the process is lost while it is paused (power cut, kill -9) this file says where it
                        stopped and `python benchmark.py --resume` carries on from the rows file as for any broken run.

WHAT STOPS WHEN
  manual pause   the step that is going FINISHES, so no request is cut off mid-way: the speed test of the current model,
                 or the requests in flight of the current dataset. Then the run waits.
  GPU too hot    (MAX_GPU_TEMP) the guard notes it at once; the run stops at the end of the request that is going (a request
                 is never dropped) and waits until the GPU has been at or below MAX_GPU_TEMP - GPU_TEMP_RESUME_DELTA for a
                 few seconds, then carries on by itself.
  FRESH START    whatever test line was half done when it stopped is thrown away and done again from its first step: the
                 speed test of a model (numbers measured on a hot or interrupted GPU are not kept), the current dataset of an
                 eval stage, a model's long-context test. Finished models and answers already given stay: the speed rows
                 are in the rows file and every graded answer is in the result cache, so the restarted dataset asks only the
                 questions that were not answered yet. Scores are therefore the same as an uninterrupted run's.

A GPU that keeps tripping the limit on the same test line ends the run after GPU_TEMP_MAX_RESTARTS restarts (a run
that can never finish is worse than a stopped one); it is resumable with --resume.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import threading
import time
from pathlib import Path

from . import status
from .common import __version__, col, now, vmax

CONTROL_NAME = "run_control.json"
PAUSE_VERSION = 1
POLL_S = 2.0              # a waiting run looks again this often
REPORT_S = 60.0           # ... and says it is still waiting this often (the log has no live counter)
COOL_SAMPLES = 3          # samples in a row at or below the resume level before the GPU counts as cool


class PauseRestart(BaseException):
    """The GPU got too hot while a test line was going: drop that line and do it again after the pause. A BaseException
    so that no `except Exception` on the way up can swallow it (a swallowed one would keep a hot measurement)."""


class PauseGaveUp(KeyboardInterrupt):
    """The same test line tripped the limit too often. A KeyboardInterrupt so the phases end the run the way Ctrl+C does:
    partial results written, resumable."""


# --------------------------------------------------------------------------- the files
def control_path(output_dir):
    return Path(output_dir) / CONTROL_NAME


def write_control(output_dir, pause, by="settings page"):
    """Ask the run in `output_dir` to pause (True) or carry on (False)."""
    status._atomic(control_path(output_dir), json.dumps({"pause": bool(pause), "by": by, "at": status._stamp()}))


def read_control(output_dir):
    try:
        d = json.loads(control_path(output_dir).read_text(encoding="utf-8"))
        return bool(d.get("pause")) if isinstance(d, dict) else False
    except (OSError, ValueError):
        return False


def clear_control(output_dir):
    try:
        control_path(output_dir).unlink()
    except OSError:
        pass


def pause_path(output_dir, stamp):
    return Path(output_dir) / f"run_{stamp}.pause.json"


def find_pause(output_dir):
    """The newest run_<stamp>.pause.json in OUTPUT_DIR (a run lost while paused leaves one), or None."""
    found = sorted(Path(output_dir).glob("run_*.pause.json"), key=lambda f: f.stat().st_mtime)
    return found[-1] if found else None


def read_pause(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- the guard
class ThermalGuard:
    """Watches the GPU temperature (the hottest GPU) on its own sampler, so it covers the evals too (the speed test's
    sampler is stopped before they start) and needs no knowledge of what the run is doing. `feed()` is the whole
    decision, so it can be tested without a GPU."""

    def __init__(self, ctl, monitor, max_c, resume_c, poll=0.5):
        self.ctl, self.mon, self.max_c, self.resume_c, self.poll = ctl, monitor, max_c, resume_c, poll
        self.temp = None
        self.cool_n = 0
        self._halt = threading.Event()
        self._thread = None

    def feed(self, temp):
        if temp is None:                       # no reading this time: neither hotter nor cooler than before
            return
        self.temp = self.ctl.temp = temp
        if temp >= self.max_c:
            self.cool_n = 0
            self.ctl.trip(temp)
        elif temp <= self.resume_c:
            self.cool_n += 1
        else:
            self.cool_n = 0

    def cooled(self):
        return self.cool_n >= COOL_SAMPLES

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-temp-guard")
        self._thread.start()

    def _loop(self):
        t_prev = now()
        while not self._halt.wait(self.poll):
            t = now()
            try:
                self.feed(vmax(col(self.mon.window(t_prev, t), "gpu_temp")))
            except Exception:                  # a sampler hiccup must not end the guard
                pass
            t_prev = t

    def stop(self):
        self._halt.set()
        try:
            self.mon.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- the controller
class Controller:
    """One per process (`CONTROL`). Inactive until `configure()`: a demo, a utility command or a test never pauses."""

    def __init__(self):
        self.reset()

    def reset(self):
        guard = getattr(self, "guard", None)
        if guard:
            guard.stop()
        self.output_dir = self.stamp = self.results = self.save = None
        self.manual = False                # pause asked for through run_control.json
        self.tripped = False               # the GPU reached MAX_GPU_TEMP and the test line it touched is not yet restarted
        self.trip_temp = self.temp = None
        self.guard = None
        self.max_c = self.resume_c = 0.0
        self.max_restarts = 3
        self.paused_s, self.pauses = 0.0, 0
        self.restarts = {}
        self.discarding = None
        self._restartable = 0
        self._last_poll = 0.0
        self._last_report = 0.0
        self._how = ""

    # -- set-up -------------------------------------------------------------
    def configure(self, cfg, stamp, results=None, save=None):
        """Start following run_control.json in cfg.output_dir, and the GPU when MAX_GPU_TEMP is set."""
        self.reset()
        self.output_dir, self.stamp, self.results, self.save = Path(cfg.output_dir), stamp, results, save
        self.max_restarts = int(getattr(cfg, "gpu_temp_max_restarts", 3))
        clear_control(self.output_dir)         # a pause asked for in an earlier run is not this run's
        max_c = float(getattr(cfg, "max_gpu_temp", 0) or 0)
        if max_c > 0:
            self._start_guard(cfg, max_c)

    def _start_guard(self, cfg, max_c):
        from .locality import server_is_local
        local, host = server_is_local(cfg.base_url)
        if local is False:
            print(f"[warn] MAX_GPU_TEMP is not enforced: the server ({host}) is on another machine and the GPU "
                  "temperature is read on this one. Run the benchmark on the GPU machine.")
            return
        from .telemetry.gpu import open_gpu_monitor
        sampler_cfg = copy.copy(cfg)
        sampler_cfg.gpu_poll_ms = 500
        mon = open_gpu_monitor(sampler_cfg)
        first = mon.series.last() if getattr(mon, "series", None) is not None else None
        if not mon.ok or not first or first.get("gpu_temp") is None:
            mon.stop()
            print(f"{'!' * 50}\n[warn] MAX_GPU_TEMP={max_c:g} is NOT enforced: no GPU temperature can be read here "
                  "(NVML or nvidia-smi is needed; try --check).\n" + "!" * 50)
            return
        self.max_c = max_c
        self.resume_c = max_c - float(getattr(cfg, "gpu_temp_resume_delta", 10.0))
        self.guard = ThermalGuard(self, mon, self.max_c, self.resume_c)
        self.guard.feed(first["gpu_temp"])
        self.guard.start()
        print(f"Thermal safety: the run pauses at {self.max_c:g} C and carries on at {self.resume_c:g} C or below "
              f"(GPU reads {first['gpu_temp']:.0f} C now)")

    def shutdown(self):
        """The run is over: stop the guard, remove the pause file and any pending request."""
        if self.output_dir is None:
            return
        if self.stamp:
            with contextlib.suppress(OSError):
                pause_path(self.output_dir, self.stamp).unlink()
        clear_control(self.output_dir)
        status.update(force=True, paused=False, pause_reason=None, pause_restart=None, pause_temp=None)
        paused, pauses = self.paused_s, self.pauses
        self.reset()
        self.paused_s, self.pauses = paused, pauses       # kept for the run's timing summary

    # -- what the run asks ----------------------------------------------------
    @property
    def active(self):
        return self.output_dir is not None

    def trip(self, temp):
        if not self.tripped:
            self.trip_temp = temp
        else:
            self.trip_temp = max(self.trip_temp or temp, temp)
        self.tripped = True

    def _poll(self, force=False):
        t = time.monotonic()
        if force or t - self._last_poll >= 1.0:
            self._last_poll = t
            self.manual = read_control(self.output_dir)

    def requested(self):
        """Is a pause asked for (by hand, or by the GPU)? Cheap: the control file is read about once a second."""
        if self.output_dir is None:
            return False
        self._poll()
        return self.manual or self.tripped

    @contextlib.contextmanager
    def restartable(self):
        """While inside, `checkpoint()` may raise PauseRestart. Only code that catches it opens one."""
        self._restartable += 1
        try:
            yield
        finally:
            self._restartable -= 1

    def checkpoint(self):
        """Call after a request of a test line has completed (never in the middle of one). When the GPU has tripped the
        limit and the caller said it can restart, abandon the line."""
        if self._restartable and self.tripped:
            raise PauseRestart()

    def discard(self, what):
        """The test line `what` was thrown away. Counts thermal restarts of it; too many ends the run."""
        self.discarding = what
        if not self.tripped:
            return
        n = self.restarts[what] = self.restarts.get(what, 0) + 1
        if self.max_restarts > 0 and n > self.max_restarts:
            msg = (f"{what} was stopped by the GPU temperature limit {n} times (MAX_GPU_TEMP={self.max_c:g} C, "
                   f"GPU_TEMP_MAX_RESTARTS={self.max_restarts}): the run ends here. Improve the cooling or raise "
                   "MAX_GPU_TEMP, then continue with --resume.")
            print(f"\n[pause] {msg}")
            raise PauseGaveUp(msg)

    # -- waiting ----------------------------------------------------------------
    def wait(self, where=""):
        """Block while a pause is asked for. Returns the seconds waited (0 = no pause)."""
        if not self.requested():
            return 0.0
        t0 = self._begin(where)
        while not self._released():
            time.sleep(POLL_S)
            self._report(t0)
        return self._end(t0)

    async def await_resume(self, where=""):
        """The same for the evals (event loop): other tasks keep running, this one waits."""
        if not self.requested():
            return 0.0
        t0 = self._begin(where)
        while not self._released():
            await asyncio.sleep(POLL_S)
            self._report(t0)
        return self._end(t0)

    def _released(self):
        self._poll(force=True)
        if self.tripped and (self.guard is None or self.guard.cooled()):
            self.tripped = False
        return not (self.manual or self.tripped)

    def _why(self):
        parts = []
        if self.tripped:
            parts.append(f"GPU temperature reached {self.trip_temp:.0f} C (MAX_GPU_TEMP {self.max_c:g} C); "
                         f"carries on at {self.resume_c:g} C or below")
        if self.manual:
            parts.append("paused by request; carries on when you press Resume (or run: python benchmark.py --resume)")
        return "; ".join(parts)

    def _reason(self):
        return "manual+thermal" if self.manual and self.tripped else "thermal" if self.tripped else "manual"

    def _begin(self, where):
        self.pauses += 1
        self._last_report = time.monotonic()
        self._how = self._reason()
        snap = status.snapshot()
        if self.save:
            with contextlib.suppress(Exception):
                self.save()                                  # the rows file is current at the moment of the pause
        self._write_state(snap)
        status.update(force=True, paused=True, pause_reason=self._how, pause_since=status._stamp(),
                      pause_temp=self.temp, pause_max=self.max_c or None, pause_resume_below=self.resume_c or None,
                      pause_restart=self.discarding)
        print(f"\n[pause] {self._why()}" + (f" ({where})" if where else "")
              + (f"\n[pause] restarted from its first step when the run carries on: {self.discarding}"
                 if self.discarding else "") + "\n[pause] waiting...", flush=True)
        return now()

    def _report(self, t0):
        t = time.monotonic()
        status.update(pause_temp=self.temp)
        if t - self._last_report >= REPORT_S:
            self._last_report = t
            hot = f", GPU {self.temp:.0f} C" if self.temp is not None else ""
            print(f"[pause] still paused ({_mins(now() - t0)}{hot})", flush=True)

    def _end(self, t0):
        dt = now() - t0
        self.paused_s += dt
        with contextlib.suppress(OSError):
            pause_path(self.output_dir, self.stamp).unlink()
        status.update(force=True, paused=False, pause_reason=None, pause_restart=None, pause_temp=None)
        print(f"[pause] carrying on after {_mins(dt)}"
              + (f": starting again from the beginning: {self.discarding}" if self.discarding else ""), flush=True)
        self.discarding = None
        return dt

    def _write_state(self, snap):
        rows = self.results or []
        doc = {"pause_version": PAUSE_VERSION, "tool_version": __version__, "stamp": self.stamp,
               "paused_at": status._stamp(), "reason": self._how, "gpu_temp": self.temp,
               "max_gpu_temp": self.max_c or None, "resume_below": self.resume_c or None,
               "phase": snap.get("phase"), "model": snap.get("model"), "model_i": snap.get("model_i"),
               "models": snap.get("models"), "stage": snap.get("stage"), "dataset": snap.get("dataset"),
               "questions_done": snap.get("done"), "questions_total": snap.get("total"),
               "restarts": self.discarding or "nothing: the run stopped between two steps",
               "completed_models": [r.get("model") for r in rows
                                    if r.get("status") == "ok" and r.get("avg_tps") is not None],
               "rows_file": status.rows_path(self.output_dir, self.stamp).name,
               "results": [{"model": r.get("model"), "status": r.get("status"), "avg_tps": r.get("avg_tps"),
                            "datasets_scored": sum(1 for k in r if k.startswith("eval_") and k.endswith("_pct")
                                                   and r[k] is not None)} for r in rows]}
        try:
            status._atomic(pause_path(self.output_dir, self.stamp), json.dumps(doc, default=str))
        except (OSError, TypeError, ValueError) as e:
            print(f"[warn] could not write the pause file ({e.__class__.__name__}: {e})")


def _mins(seconds):
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"


CONTROL = Controller()


# --------------------------------------------------------------------------- from the command line and the page
def request(output_dir, pause):
    """Pause (True) or resume (False) the live run in `output_dir`. Raises RuntimeError with what to do when none is
    going. Returns what happened, in words."""
    from .cli import lock_info
    live = lock_info(output_dir)
    if not live:
        raise RuntimeError("no run is going in " + str(output_dir))
    if pause:
        write_control(output_dir, True, "request")
        return "Pause requested: the run finishes its current step, then waits."
    st = status.read(output_dir) or {}
    if not (st.get("paused") or read_control(output_dir)):
        raise RuntimeError("the run that is going is not paused (--resume continues a run that has stopped; this one is still "
                           "running). To pause it: python benchmark.py --pause")
    write_control(output_dir, False, "request")
    return "Resume requested: the run carries on within a few seconds."
