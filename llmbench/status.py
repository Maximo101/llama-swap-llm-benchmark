"""Run status and the per-run rows file (v2.3.1).

run_status.json   OUTPUT_DIR/run_status.json, rewritten (atomically) while a run is going: phase, which model of how
                  many, the dataset and question count, when it started and when this file was last written, host and
                  pid. The settings page shows it; anything else (a Home Assistant sensor, a script) can read it. It is
                  also the heartbeat that tells a run on ANOTHER machine apart from a dead one (cli.lock_info).

run_<stamp>.rows.json   the raw result rows of a run, saved after every model (speed phase) and every model of every
                  eval stage, replaced atomically. If the run dies - a crash, a power cut, kill -9 - everything measured
                  up to the last finished model is still there, and `python benchmark.py --resume` carries on from it:
                  models whose speed test finished are not benchmarked again, and the result cache answers every
                  eval question that was already asked. A finished run keeps its rows file too, so --resume can also
                  repair it (models that failed to load, lost their long-context test or have incomplete evals).

Neither file holds model answers or reference answers (those are in eval_details_*.jsonl and results/cache).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

from .common import __version__

STATUS_NAME = "run_status.json"
ROWS_VERSION = 1
_S = {"path": None, "data": {}, "last": 0.0, "touch": None}
_LOCK = threading.Lock()
_DROP_VALUE = {"--api-key", "--api-key-file", "--ssl-key-file"}      # never written to disk with their value


def hostname():
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _atomic(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- run_status.json
def start(output_dir, touch=None, **base):
    """Begin reporting. `touch`: the run lock, whose modification time is refreshed with every write (heartbeat)."""
    with _LOCK:
        _S.update(path=Path(output_dir) / STATUS_NAME, touch=touch, last=0.0,
                  data={"version": __version__, "host": hostname(), "pid": os.getpid(), "started": _stamp(),
                        "state": "running", **base})
    update(force=True)


def update(force=False, every=10.0, **kw):
    """Merge `kw` into the status and write it, at most once per `every` seconds unless `force`. Never raises: a
    status file that cannot be written must not stop a benchmark."""
    with _LOCK:
        if _S["path"] is None:
            return
        _S["data"].update(kw)
        t = time.monotonic()
        if not force and t - _S["last"] < every:
            return
        _S["last"] = t
        _S["data"]["updated"], _S["data"]["updated_epoch"] = _stamp(), round(time.time(), 1)
        try:
            _atomic(_S["path"], json.dumps(_S["data"], default=str))
            if _S["touch"]:
                os.utime(_S["touch"], None)
        except OSError:
            pass


def snapshot():
    """A copy of what is being reported now (phase, model, dataset, question count): the pause file records it."""
    with _LOCK:
        return dict(_S["data"])


def finish(state):
    """The run ended: 'complete', 'partial', 'interrupted' or 'failed'. Reporting stops."""
    update(force=True, state=state, finished=_stamp())
    with _LOCK:
        _S.update(path=None, touch=None)


def read(output_dir):
    try:
        d = json.loads((Path(output_dir) / STATUS_NAME).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- run_<stamp>.rows.json
def rows_path(output_dir, stamp):
    return Path(output_dir) / f"run_{stamp}.rows.json"


def _safe_argv(argv):
    out, skip = [], False
    for a in argv or []:
        if skip:
            skip = False
            continue
        if a in _DROP_VALUE:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in _DROP_VALUE):
            continue
        out.append(a)
    return out


def _row(r):
    d = dict(r)
    if d.get("_launch_argv"):
        d["_launch_argv"] = _safe_argv(d["_launch_argv"])     # the cache fingerprint ignores these flags anyway
    return d


def save_rows(cfg, stamp, results, state="partial", info=None):
    """Save every row as it is now. Never raises."""
    info = info or {}
    doc = {"rows_version": ROWS_VERSION, "tool_version": __version__, "stamp": stamp, "state": state,
           "saved": _stamp(), "host": hostname(), "base_url": cfg.base_url, "backend": cfg.backend,
           "hardware": info.get("hardware"), "cpu_source": info.get("cpu_source"), "mode": info.get("mode"),
           "resumed_from": info.get("resumed_from"), "rows": [_row(r) for r in results]}
    try:
        _atomic(rows_path(cfg.output_dir, stamp), json.dumps(doc, default=str))
    except (OSError, TypeError, ValueError) as e:
        print(f"[warn] could not save the run's rows ({e.__class__.__name__}: {e}); --resume will not see this run")


def load_rows(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        raise ValueError("not a rows file")
    return doc


def find_rows(output_dir, which=None):
    """The rows file to resume: a path, a stamp (20261110_103848), or None = the newest in OUTPUT_DIR."""
    d = Path(output_dir)
    if which:
        p = Path(which).expanduser()
        if p.is_file():
            return p
        p = rows_path(d, str(which).replace("benchmark_", "").replace(".json", ""))
        return p if p.is_file() else None
    found = sorted(d.glob("run_*.rows.json"), key=lambda f: f.stat().st_mtime)
    return found[-1] if found else None
