"""python benchmark.py --ui: the settings page (v2.3). Standard library only (http.server).

What it does: shows every .env setting (part, section, description, default, current value, a widget by kind),
validates and saves .env in the .env.example layout (keeping .env.bak), manages favourites (configs/<name>.env),
and runs a FIXED set of the tool's own commands: --check, --check-env, --list-models, --estimate (with the unsaved
settings via --env), --reload, --compare, a benchmark run, and SIGINT to stop it. It is not a shell: no request
can choose a program or an argument other than a file picked from the lists it shows.

Security (also in the README):
  * Binds 127.0.0.1 unless UI_HOST is set (e.g. 0.0.0.0 for the LAN). On the LAN a token is required: UI_TOKEN from
    .env, or one generated on first start, saved to .env and printed in the URL. The first visit with ?token= sets
    an HttpOnly, SameSite=Strict cookie. UI_TOKEN set on localhost is required there too.
  * Host header: an IP address, localhost, or this machine's own name; anything else (DNS rebinding) is refused.
  * Every state-changing request is a POST with JSON, a custom X-LLMB-UI header and, when the browser sends one, an
    Origin equal to the page's own: other websites cannot drive it (CSRF).
  * Secrets (API keys, tokens) are never sent in the page state; "show" fetches one on request. They are saved to
    .env only: never to favourites, logs or dashboards.
  * Plain HTTP: on the LAN the token and any secret you reveal travel unencrypted. Fine on a home LAN; put a
    reverse proxy with HTTPS in front if you need more.
"""
from __future__ import annotations

import errno
import hmac
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from .. import config, envfile, envspec, pause as pause_mod, status
from ..locality import remote_warning
from ..common import __version__
from ..config import SCRIPT_DIR, env_path

PAGE = Path(__file__).with_name("settings.html")
DEFAULT_PORT = 41010
MIN_LAN_TOKEN = 16          # v2.3.1: on the LAN the token is the only thing between the network and this machine
MAX_BAD_TOKENS = 10         # wrong tokens from one address ...
BAD_TOKEN_WINDOW = 300      # ... within this many seconds lock that address out for as long
_LOG_RE = re.compile(r"^(benchmark|batch_probe)_[A-Za-z0-9_.-]+\.log$")
# Dashboards are files made from server data. Served in a sandbox they get an origin of their own: their scripts run,
# but can reach neither this page's cookie nor its API (and so no stored secret), whatever a dashboard contains.
DASH_CSP = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox"
COOKIE = "llmb_ui"
_DASH_RE = re.compile(r"^benchmark_(dashboard|compare)[A-Za-z0-9_.-]*\.html$")
_RUN_RE = re.compile(r"^(benchmark_\d{8}_\d{6}|results)\.json$")
TIMEOUTS = {"check": 180, "check-env": 60, "list-models": 60, "estimate": 300, "reload": 600, "compare": 600,
            "validate": 60}


# --------------------------------------------------------------------------- settings of the page itself
def _setting(vals, key, default=None):
    """Like python-dotenv: the shell environment wins over .env."""
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        v = vals.get(key)
    return default if v is None or str(v).strip() == "" else str(v).strip().strip('"').strip("'")


def clean_host(v):
    """UI_HOST as an address to listen on: a URL or host:port written there is cut down to the host
    ("http://192.168.2.200:41010/" -> "192.168.2.200"); "[::1]" -> "::1"."""
    v = (v or "").strip()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0].split("?", 1)[0]
    if "@" in v:
        v = v.rsplit("@", 1)[1]
    if v.startswith("["):
        return v[1:].split("]", 1)[0]
    if v.count(":") == 1:                                # host:port (a bare IPv6 address has more colons)
        v = v.split(":", 1)[0]
    return v


def page_settings():
    vals = envfile.read_values(env_path())
    raw = _setting(vals, "UI_HOST", "127.0.0.1")
    host = clean_host(raw) or "127.0.0.1"
    if host != raw:
        print(f"[config] UI_HOST={raw} is read as {host} (UI_HOST is an address to listen on, not a URL)")
    try:
        port = int(_setting(vals, "UI_PORT", str(DEFAULT_PORT)))
    except ValueError:
        sys.exit("[config] UI_PORT must be a port number")
    return SimpleNamespace(host=host, port=port, token=_setting(vals, "UI_TOKEN", ""),
                           lan=not _is_loopback(host))


def _is_loopback(host):
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def output_dir(vals=None):
    vals = vals if vals is not None else envfile.read_values(env_path())
    p = Path(_setting(vals, "OUTPUT_DIR", "results")).expanduser()
    return p if p.is_absolute() else SCRIPT_DIR / p


def configs_dir():
    return SCRIPT_DIR / "configs"


def allowed_hostnames():
    names = {"localhost"}
    try:
        h = socket.gethostname().lower()
        names |= {h, h.split(".")[0], socket.getfqdn().lower(), h.split(".")[0] + ".local"}
    except OSError:
        pass
    return names


def host_ok(host_header, port):
    """DNS-rebinding guard: the Host must be an IP literal, localhost or this machine's name (and our port)."""
    if not host_header:
        return False
    m = re.match(r"^\[?([^\]]+?)\]?(?::(\d+))?$", host_header.strip().lower())
    if not m:
        return False
    name, p = m.group(1), m.group(2)
    if p is not None and int(p) != port:
        return False
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return name in allowed_hostnames()


# --------------------------------------------------------------------------- state for the page
def _entries():
    code = envspec.code_settings()
    out = []
    for e in envspec.parse_example():
        kind = code.get(e.key, (None, "str"))[1] if not e.family else "family"
        out.append({"key": e.key, "part": e.part, "part_title": e.part_title, "section": e.section, "desc": e.desc,
                    "default": e.default, "shipped": e.value if e.active else e.default, "active": e.active,
                    "kind": kind, "choices": list(envspec.CHOICES.get(e.key, ())),
                    "secret": e.key in envspec.SECRETS, "family": e.family,
                    "pattern": e.pattern().pattern if e.family else None})
    return out


def masked(values):
    return {k: ("" if k in envspec.SECRETS else v) for k, v in values.items()}


def report_dict(rep):
    return {"problems": {k: why for k, _, _, why in rep.invalid},
            "unknown": [{"key": k, "line": n, "hint": hint} for k, _, n, hint in rep.unknown],
            "legacy": [{"key": k, "now": new} for k, new, _ in rep.legacy],
            "duplicates": [{"key": k, "lines": ln} for k, ln, _ in rep.duplicates],
            "shell": [{"key": k} for k, _, _ in rep.shell], "info": list(rep.info),
            "missing": [e.key for e in rep.missing]}


def check_values(values):
    """--check-env on a candidate set of values (a private temp file; nothing written to .env)."""
    with _tempenv(values) as p:
        return report_dict(envspec.check_env(p, environ=dict(os.environ)))


class _tempenv:
    """A private temporary .env holding `values` (it may hold secrets): mode 0600, removed afterwards."""
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        fd, name = tempfile.mkstemp(prefix="llmb-ui-", suffix=".env")
        os.close(fd)
        self.path = Path(name)
        os.chmod(self.path, 0o600)
        self.path.write_text(envfile.render_env(self.values), encoding="utf-8")
        return self.path

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


def merge_secrets(values, current, secrets_in=None):
    """The page never holds the stored secrets. A secret named in `secrets_in` is set to that value ("" clears it);
    one not named keeps what .env has now."""
    out = {k: v for k, v in values.items() if k not in envspec.SECRETS}
    for k in envspec.SECRETS:
        if secrets_in and k in secrets_in:
            if secrets_in[k]:
                out[k] = str(secrets_in[k])
        elif current.get(k):
            out[k] = current[k]
    return out


# ------------------------------------------------------------------ guides shown under some settings on the page
RECOMMENDED = {"EVAL_PROFILE": "funnel"}
# rows per dataset (about; a profile larger than the dataset uses all of it)
DATASET_ROWS = {"gpqa": 198, "mmlu_pro": 12032, "humaneval": 164, "mbpp": 500, "livecodebench": None,
                "gsm8k": 1319, "math": 500}


def margin(n):
    """95% margin of error in points for n questions (worst case): 98 / sqrt(n), as everywhere in this tool."""
    return round(98 / n ** 0.5, 1) if n else None


def guides(vals):
    """Data for the guides on the settings page: profiles with their margins per test type, the funnel stages for
    each EVAL_FUNNEL_FINAL, the GPU bandwidth table with the detected GPU, the recommended choices."""
    from types import SimpleNamespace
    from .. import gpubw
    from ..config import EVAL_SPECS, PROFILE_ROWS, funnel_stages

    def num(k, d):
        try:
            return int(_setting(vals, k, d) or d)
        except ValueError:
            return d
    lc_n = num("LONGCTX_TRIALS", 2) * num("LONGCTX_KEYS", 4)
    tests = []
    for key, spec in EVAL_SPECS.items():
        size = 32 if key == "toolcall" else DATASET_ROWS.get(key)
        tests.append({"key": key, "label": spec.get("label", key), "size": size, "full": bool(spec.get("full"))})
    profiles = []
    for name, rows in PROFILE_ROWS.items():
        per = {}
        for t in tests:
            n = t["size"] if t["full"] else (rows if t["size"] is None else
                                             (t["size"] if rows is None else min(rows, t["size"])))
            per[t["key"]] = {"n": n, "margin": margin(n) if n else None}
        profiles.append({"name": name, "rows": rows, "margin": margin(rows) if rows else None, "tests": per})
    finals = []
    for final in PROFILE_ROWS:
        if final in ("test", "quick"):
            continue
        st = funnel_stages(SimpleNamespace(eval_funnel_stages=(), eval_funnel_final=final))
        finals.append({"final": final, "stages": [{"name": n, "rows": r, "margin": margin(r) if r else None}
                                                   for n, r in st]})
    gpus = gpubw.detected()
    bw, src = gpubw.resolve(_setting(vals, "ESTIMATE_GPU_BW_GBS", "auto"), gpus)
    return {"recommended": RECOMMENDED, "tests": tests, "profiles": profiles, "finals": finals,
            "longctx_n": lc_n, "longctx_margin": margin(lc_n),
            "gpu": {"table": [list(r[1:]) for r in gpubw.GPU_TABLE], "detected": [[n, round(g, 1)] for n, g in gpus],
                    "bw": bw, "source": src, "fallback": gpubw.FALLBACK_GBS}}


_CHILD = {"proc": None, "started": None}           # the run this page started (until its lock file appears)
_LAST = {"log": None}                               # the last run's log: its tail is read after the lock is gone


def run_state(vals=None):
    """The live run: its lock file in OUTPUT_DIR (wherever it was started), else a run this page started that has not
    written its lock yet (Python start-up, loading the settings)."""
    from ..cli import lock_info
    info = lock_info(output_dir(vals))
    if info:
        _LAST["log"] = info.get("log") or _LAST["log"]
        return {"running": True, **info, "status": status.read(output_dir(vals)),
                "pause_requested": pause_mod.read_control(output_dir(vals))}
    proc = _CHILD["proc"]
    if proc is not None and proc.poll() is None:
        return {"running": True, "pid": proc.pid, "started": _CHILD["started"], "log": None, "starting": True}
    out = {"running": False}
    end = _ended_status(vals)
    if end:                                         # how the run ended (v2.3.1 fix): the page used to say "ended" for a failure
        out.update(status=end, log=end.get("log"))
    elif proc is not None:                          # this page started a run that never wrote a status (refused, crashed on start)
        out["exited"] = proc.returncode
    return out


def _ended_status(vals=None):
    """run_status.json of a run that has ended. When this page started a run, only that run's status counts: an older
    run's 'complete' must not pass for the one that just died."""
    st = status.read(output_dir(vals))
    if not st or st.get("state") in (None, "running"):
        return None
    mine = _CHILD["started"] if _CHILD["proc"] is not None else None
    if mine and str(st.get("started") or "") < mine:
        return None
    return st


def newest_dashboard(kind="dashboard", vals=None):
    """The newest benchmark_<kind>*.html: the copy next to benchmark.py or the per-run files in OUTPUT_DIR. The copy
    can be stale when it couldn't be overwritten (a share where it belongs to root), so modification time decides."""
    found = [SCRIPT_DIR / f"benchmark_{kind}.html"]
    d = output_dir(vals)
    if d.is_dir():
        found += [p for p in d.iterdir() if p.name.startswith(f"benchmark_{kind}") and _DASH_RE.match(p.name)]
    found = [p for p in found if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def outputs(vals=None):
    d = output_dir(vals)
    if not d.is_dir():
        return {"dashboards": [], "runs": [], "latest": newest_dashboard("dashboard", vals) is not None,
                "resumable": False}
    dash = sorted((p.name for p in d.iterdir() if _DASH_RE.match(p.name)), reverse=True)
    runs = sorted((p.name for p in d.iterdir() if _RUN_RE.match(p.name) and p.name != "results.json"), reverse=True)
    return {"dashboards": dash[:60], "runs": runs[:60], "latest": newest_dashboard("dashboard", vals) is not None,
            "resumable": resumable(vals)}


def last_run_models(vals=None):
    """{model: {vram_gb, vram_pct, ctx, status, note}} from OUTPUT_DIR/results.json (cheap: one file)."""
    p = output_dir(vals) / "results.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for m in doc.get("models") or []:
        g, md, sc = m.get("gpu") or {}, m.get("metadata") or {}, m.get("scores") or {}
        gg = m.get("gguf") or {}
        out[m.get("model")] = {"vram_gb": g.get("vram_gb"), "vram_pct": g.get("vram_pct"), "ctx": md.get("ctx_size"),
                               "status": m.get("status"), "note": sc.get("score_note") or m.get("error") or None,
                               "agent": sc.get("score_agent"), "gguf_gb": gg.get("gguf_gb"),
                               "params_b": md.get("params_b"), "active_b": md.get("active_b"),
                               "n_ctx_train": gg.get("n_ctx_train"), "arch": gg.get("gguf_arch"), "bpw": gg.get("bpw")}
    return out


_GGUF_INFO = {}                                   # (path, size, mtime) -> header facts: read each file once


def gguf_meta(model, values, last=None):
    """File size, parameters, trained context and architecture for the models list. From the GGUF header when the
    file is readable here (GGUF_SEARCH_DIRS / GGUF_PATH_MAP, the model id as the file name), else from the last
    run's results, else parameters parsed from the name."""
    from ..gguf import find_readable, inspect, parse_path_map
    from ..scoring import parse_active, parse_params
    dirs = tuple(x.strip().strip('"').strip("'") for x in (_setting(values, "GGUF_SEARCH_DIRS", "") or "").split(",")
                 if x.strip())
    pmap = parse_path_map(_setting(values, "GGUF_PATH_MAP", "") or "")
    path = None
    if dirs or pmap:
        try:
            path = find_readable(model if model.lower().endswith(".gguf") else model + ".gguf", None, pmap, dirs)
        except OSError:
            path = None
    if path:
        try:
            st = os.stat(path)
            key = (path, st.st_size, st.st_mtime)
            if key not in _GGUF_INFO:
                _GGUF_INFO[key] = inspect(path)
            i = _GGUF_INFO[key]
            return {"size_gb": i.get("file_bytes", 0) / 1e9 or None,
                    "params_b": i["params_total"] / 1e9 if i.get("params_total") else None,
                    "active_b": (i["params_active"] / 1e9 if i.get("params_active")
                                 and i.get("params_total") and i["params_active"] < i["params_total"] else None),
                    "n_ctx_train": i.get("n_ctx_train"), "arch": i.get("gguf_arch"), "bpw": i.get("bpw"),
                    "src": "GGUF header"}
        except Exception:
            pass
    if last and (last.get("gguf_gb") or last.get("params_b")):
        return {"size_gb": last.get("gguf_gb"), "params_b": last.get("params_b"), "active_b": last.get("active_b"),
                "n_ctx_train": last.get("n_ctx_train"), "arch": last.get("arch"), "bpw": last.get("bpw"),
                "src": "last run"}
    return {"size_gb": None, "params_b": parse_params(model), "active_b": parse_active(model), "n_ctx_train": None,
            "arch": None, "bpw": None, "src": "model name"}


def _fetch_model_ids(url, key, strata=False):
    """GET <url>/v1/models. Returns the ids; raises on any failure. Strata gets its key both ways, like the run."""
    import requests
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
        if strata:
            headers["x-api-key"] = key
    r = requests.get(f"{url}/v1/models", timeout=8, headers=headers)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]


def list_server_models(values):
    """GET <LLAMA_SWAP_URL>/v1/models with the form's URL / key, and which ones INCLUDE/EXCLUDE/MODEL_FILTER pick.
    Like the run itself, falls back to STRATA_URL when llama-swap does not answer (or lists nothing), so the
    page lists the Strata model while llama-swap is stopped."""
    from ..client import model_skip_reason
    url = (values.get("LLAMA_SWAP_URL") or "http://localhost:8080").rstrip("/")
    saved = envfile.read_values(env_path())
    key = values.get("LLAMA_SWAP_API_KEY") or saved.get("LLAMA_SWAP_API_KEY")
    surl = (values.get("STRATA_URL") or values.get("STRATA_SWAP_URL") or "").strip().rstrip("/")
    skey = values.get("STRATA_API_KEY") or saved.get("STRATA_API_KEY")
    backend, note = "llama-swap", None
    try:
        ids = _fetch_model_ids(url, key)
        err = None if ids else f"{url}/v1/models lists no models"
    except Exception as e:                                      # shown on the page, never fatal
        ids, err = [], f"{url}/v1/models: {e.__class__.__name__}: {e}"
    if not ids and surl and surl != url:
        try:
            ids = _fetch_model_ids(surl, skey, strata=True)
            name = (values.get("STRATA_MODEL") or "").strip()
            if name and name not in ids:                        # the run sends STRATA_MODEL, whatever /v1/models says
                ids = [name] + ids
            elif not ids:
                ids = [name or "strata"]
            backend, url = "Strata", surl
            note = f"llama-swap did not answer ({err}); listing Strata at {surl}, as a run would."
        except Exception as e:
            return {"error": f"{err}; Strata {surl}/v1/models: {e.__class__.__name__}: {e}", "models": []}
    elif not ids:
        return {"error": err, "models": []}
    lst = lambda k: tuple(x.strip().strip('"').strip("'") for x in (values.get(k) or "").split(",") if x.strip())
    inc = () if any(x.lower() in ("all", "*") for x in lst("INCLUDE_MODELS")) else lst("INCLUDE_MODELS")
    ns = SimpleNamespace(model_filter=values.get("MODEL_FILTER") or "", include_models=inc,
                         exclude_models=lst("EXCLUDE_MODELS"))
    last = last_run_models()
    out = []
    for m in ids:
        try:
            why = model_skip_reason(ns, m)
        except re.error as e:
            why = f"MODEL_FILTER is not a valid regex: {e}"
        out.append({"id": m, "skip": why, "last": last.get(m), "gguf": gguf_meta(m, values, last.get(m))})
    return {"url": url, "backend": backend, "note": note, "models": out,
            "gguf_hint": None if any(x["gguf"]["src"] == "GGUF header" for x in out) else
            "File sizes come from the GGUF headers when the files are readable here: set GGUF_SEARCH_DIRS (e.g. "
            "/mnt/AI-Models) or GGUF_PATH_MAP (e.g. /models=/mnt/AI-Models) below."}


# --------------------------------------------------------------------------- fixed commands
def command_argv(cmd, body):
    """The ONLY commands the page can run, as fixed argv lists. A file argument must be one the page listed."""
    if cmd not in TIMEOUTS:
        raise ValueError(f"unknown command {cmd!r}")
    argv = {"check": ["--check"], "check-env": ["--check-env"], "list-models": ["--list-models"],
            "estimate": ["--estimate"], "reload": ["--reload"], "validate": ["--validate"]}.get(cmd)
    if cmd == "compare":
        files = body.get("files") or []
        known = set(outputs()["runs"])
        if not files or not all(isinstance(f, str) and f in known for f in files):
            raise ValueError("pick runs from the list")
        d = output_dir()
        argv = ["--compare", *[str(d / f) for f in files]] + (["--force"] if body.get("force") else [])
    return argv


def _env_arg():
    """--ui --env FILE: the page's commands and runs use that file too."""
    return ["--env", str(config.ENV_FILE)] if config.ENV_FILE else []


def run_command(argv, values=None, timeout=300):
    """Run `python benchmark.py <argv>` (with --env <temp file> when values are given) and return its output."""
    base = [sys.executable, str(SCRIPT_DIR / "benchmark.py")]
    try:
        if values is not None:
            with _tempenv(values) as p:
                r = subprocess.run(base + ["--env", str(p)] + argv, cwd=SCRIPT_DIR, capture_output=True, text=True,
                                   timeout=timeout, env=dict(os.environ))
        else:
            r = subprocess.run(base + _env_arg() + argv, cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=timeout,
                               env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return {"code": -1, "output": f"(stopped after {timeout}s)"}
    return {"code": r.returncode, "output": (r.stdout or "") + (r.stderr or "")}


def start_run(mode):
    if run_state()["running"]:
        raise RuntimeError("a run is already going (one at a time)")
    flags = {"all": [], "speed": ["--speed-only"], "evals": ["--evals-only"], "probe": ["--batch-probe"],
             "resume": ["--resume"]}.get(mode)
    if flags is None:
        raise ValueError("mode is all, speed, evals, probe or resume")
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"   # Stop = Ctrl+Break
              else {"start_new_session": True})
    p = subprocess.Popen([sys.executable, str(SCRIPT_DIR / "benchmark.py"), *_env_arg(), *flags], cwd=SCRIPT_DIR,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=dict(os.environ), **detach)                # the run writes its own log file
    _CHILD.update(proc=p, started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    _LAST["log"] = None                             # the new run's log appears with its lock file
    return p.pid


def pause_run(pause):
    """Pause (True) or resume (False) the run that is going, through run_control.json in OUTPUT_DIR. The run reads it
    within a second or two and reports `paused` in run_status.json. Works for a run on another machine that shares
    OUTPUT_DIR as well (the file is all it needs)."""
    st = run_state()
    if not st["running"]:
        raise RuntimeError("no run is going")
    if "--batch-probe" in (st.get("argv") or []):
        raise RuntimeError("a batch probe cannot be paused: Stop run ends it")
    if st.get("starting") and not st.get("status"):
        raise RuntimeError("the run is still starting: try again in a moment")
    cur = st.get("status") or {}
    if not pause and not (cur.get("paused") or st.get("pause_requested")):
        raise RuntimeError("the run is not paused")
    pause_mod.write_control(output_dir(), pause, "settings page")
    return True


def resumable(vals=None):
    """Is there a run that stopped and can be continued (python benchmark.py --resume)? Its rows file says so; a run lost
    while paused also left a pause file."""
    d = output_dir(vals)
    if not d.is_dir():
        return False
    if pause_mod.find_pause(d):
        return True
    p = status.find_rows(d)
    if not p:
        return False
    try:
        return status.load_rows(p).get("state") != "complete"
    except (OSError, ValueError):
        return False


def stop_run():
    st = run_state()
    if not st["running"]:
        return False
    if st.get("remote"):
        raise RuntimeError(f"that run is on another machine ({st.get('host')}): stop it there (its settings page, or "
                           "Ctrl+C where it runs)")
    try:
        if os.name == "nt":                             # reaches runs this page started (their own process group)
            os.kill(int(st["pid"]), signal.CTRL_BREAK_EVENT)
        else:
            os.kill(int(st["pid"]), signal.SIGINT)      # the run writes partial outputs, as with Ctrl+C
    except OSError as e:
        raise RuntimeError(f"could not stop the run (pid {st['pid']}): {e}. Stop it with Ctrl+C where it runs")
    return True


def safe_log(log, vals=None):
    """v2.3.1: the log path comes from the lock file, which sits in OUTPUT_DIR (often a share other machines can
    write to). Only a benchmark log inside OUTPUT_DIR is ever read: the page is not a way to read other files."""
    if not log:
        return None
    try:
        p, d = Path(log).resolve(), output_dir(vals).resolve()
    except OSError:
        return None
    return str(p) if p.parent == d and _LOG_RE.match(p.name) and p.is_file() else None


def read_log(offset):
    st = run_state()
    log = safe_log(st.get("log") or _LAST["log"])     # a finished run: still hand out the end of its log
    if not log or not Path(log).is_file():
        return {"running": st["running"], "text": "", "offset": 0, "log": log, "status": st.get("status"),
                "pid": st.get("pid"), "started": st.get("started"),     # v2.3.2: the status line read "pid undefined"
                "exited": st.get("exited"), "pause_requested": st.get("pause_requested")}   # until the run's log file existed
    with open(log, "rb") as fh:
        size = fh.seek(0, 2)
        offset = max(0, min(int(offset), size))
        if size - offset > 256_000:
            offset = size - 256_000
        fh.seek(offset)
        data = fh.read()
    return {"running": st["running"], "text": data.decode("utf-8", "replace"), "offset": offset + len(data),
            "log": log, "pid": st.get("pid"), "started": st.get("started"), "status": st.get("status"),
            "exited": st.get("exited"), "pause_requested": st.get("pause_requested")}


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "llmbench-ui"
    sys_version = ""

    def log_message(self, fmt, *args):                  # quiet; never log query strings (tokens)
        pass

    # ---- helpers
    def _send(self, code, body, ctype="application/json", extra=None, page=False):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if page:
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code, msg):
        self._send(code, {"error": msg})

    def _authed(self, query):
        """-> (ok, token given in the URL or None, tried: a wrong token or cookie was presented)."""
        cfg = self.server.cfg
        if not cfg.token:
            return True, None, False
        same = lambda a: hmac.compare_digest(str(a).encode("utf-8", "replace"), cfg.token.encode("utf-8"))
        tried = False
        try:
            c = cookies.SimpleCookie(self.headers.get("Cookie") or "")
        except cookies.CookieError:
            c = {}
        if COOKIE in c:
            if same(c[COOKIE].value):
                return True, None, False
            tried = True
        t = (query.get("token") or [""])[0]
        if t:
            if same(t):
                return True, t, False
            tried = True
        return False, None, tried

    def _guard(self, post):
        cfg = self.server.cfg
        if not host_ok(self.headers.get("Host"), cfg.port):
            self._err(403, "Host not allowed (open the page by IP address, localhost or this machine's name)")
            return None
        url = urlparse(self.path)
        query = parse_qs(url.query)
        who = self.client_address[0]
        if self.server.locked_out(who):
            self._err(429, "too many wrong tokens from this address: wait a few minutes")
            return None
        ok, via_query, tried = self._authed(query)
        if tried:
            self.server.bad_token(who)
        if not ok:
            if post or url.path.startswith("/api/"):
                self._err(401, "token required: open the URL printed when the settings page started")
            else:
                self._send(401, "<!DOCTYPE html><meta charset=utf-8><title>Token required</title><body style='font:15px "
                           "sans-serif;padding:30px'><h2>Token required</h2><p>Open the URL that <code>python "
                           "benchmark.py --ui</code> printed (it ends in <code>?token=...</code>). The token is "
                           "UI_TOKEN in .env.</p>", ctype="text/html")
            return None
        if post:
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") != f"http://{self.headers.get('Host')}":
                self._err(403, "cross-origin request refused")
                return None
            if self.headers.get("X-LLMB-UI") != "1" or "application/json" not in (self.headers.get("Content-Type") or ""):
                self._err(403, "requests must come from the settings page")
                return None
        return url, query, via_query

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _file(self, path, ctype="text/html"):
        try:
            self._send(200, Path(path).read_bytes(), ctype, extra={"Content-Security-Policy": DASH_CSP})
        except OSError:
            self._err(404, "not found")

    # ---- GET
    def do_GET(self):
        g = self._guard(post=False)
        if not g:
            return
        url, query, via_query = g
        if via_query:                                   # first visit with ?token=: set the cookie, drop the token
            ck = f"{COOKIE}={self.server.cfg.token}; HttpOnly; SameSite=Strict; Path=/"
            return self._send(303, b"", "text/plain", {"Location": url.path or "/", "Set-Cookie": ck})
        p = url.path
        if p == "/":
            return self._send(200, PAGE.read_bytes(), "text/html", page=True)
        if p == "/api/state":
            return self._send(200, self.server.state())
        if p == "/api/run":
            return self._send(200, run_state())
        if p == "/api/log":
            return self._send(200, read_log((query.get("offset") or ["0"])[0]))
        if p in ("/dashboard", "/compare"):                   # the newest one, wherever it could be written
            f = newest_dashboard("compare" if p == "/compare" else "dashboard")
            return self._file(f) if f else self._err(404, "no dashboard yet")
        if p.startswith("/results/"):
            name = p[len("/results/"):]
            if not _DASH_RE.match(name):
                return self._err(404, "not found")
            return self._file(output_dir() / name)
        self._err(404, "not found")

    # ---- POST
    def do_POST(self):
        g = self._guard(post=True)
        if not g:
            return
        try:
            body = self._body()
            out = self.server.post(urlparse(self.path).path, body)
        except (ValueError, envfile.EnvValueError) as e:
            return self._err(400, str(e))
        except RuntimeError as e:
            return self._err(409, str(e))
        if out is None:
            return self._err(404, "not found")
        self._send(200, out)

    def do_PUT(self):
        self._err(405, "not allowed")

    do_DELETE = do_PATCH = do_OPTIONS = do_PUT


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.created_main = False
        self._bad = {}                           # client address -> times of its wrong tokens
        super().__init__((cfg.host, cfg.port), Handler)

    def bad_token(self, who):
        with self.lock:
            t = time.monotonic()
            self._bad[who] = [x for x in self._bad.get(who, []) if t - x < BAD_TOKEN_WINDOW][-MAX_BAD_TOKENS:] + [t]
            if len(self._bad) > 4096:            # never grows without bound
                self._bad.clear()

    def locked_out(self, who):
        with self.lock:
            t = time.monotonic()
            recent = [x for x in self._bad.get(who, []) if t - x < BAD_TOKEN_WINDOW]
            return len(recent) >= MAX_BAD_TOKENS

    def current(self):
        return envfile.read_values(env_path())

    def state(self):
        cur = self.current()
        favs = envfile.list_favourites(configs_dir())
        just_created = False                                           # announced once, not on every reload
        if not favs and cur and not self.created_main:                 # first visit: keep today's .env as "main"
            envfile.save_favourite(configs_dir(), "main", cur, __version__)
            favs, self.created_main, just_created = ["main"], True, True
        rep = report_dict(envspec.check_env(env_path(), environ=dict(os.environ))) if env_path().is_file() else None
        return {"version": __version__, "env": str(env_path()), "env_exists": env_path().is_file(),
                "entries": _entries(), "values": masked(cur),
                "secrets_set": {k: bool(cur.get(k)) for k in envspec.SECRETS},
                "report": rep, "favourites": favs, "created_main": just_created, "run": run_state(cur),
                "outputs": outputs(cur), "lan": self.cfg.lan, "port": self.cfg.port,
                "legacy": envspec.LEGACY, "guides": guides(cur),
                "remote": remote_warning(_setting(cur, "LLAMA_SWAP_URL", "http://localhost:8080"))}

    def post(self, path, body):
        cur = self.current()
        values = body.get("values")
        if values is not None and not isinstance(values, dict):
            raise ValueError("values must be an object")
        if path == "/api/check":
            return check_values(merge_secrets(values or {}, cur, body.get("secrets")))
        if path in ("/api/save", "/api/defaults") and run_state()["running"]:
            # v2.3.2: .env is not rewritten under a live run (a --resume of it, or a second page, would read other
            # settings than the run started with). The page locks these buttons; this covers every other client.
            raise RuntimeError("Run active: the settings can't be changed until the run ends or is stopped")
        if path == "/api/save":
            merged = merge_secrets(values or {}, cur, body.get("secrets"))
            rep = check_values(merged)
            if rep["problems"]:
                return {"saved": False, "report": rep}
            v = run_command(["--validate"], merged, TIMEOUTS["validate"])
            if v["code"] != 0:
                return {"saved": False, "report": rep, "error": v["output"].strip()[-2000:]}
            with self.lock:
                bak = envfile.write_env(merged, env_path())
            return {"saved": True, "backup": str(bak) if bak else None, "report": rep}
        if path == "/api/defaults":
            with self.lock:
                bak = envfile.write_env(envfile.defaults_with_secrets(cur), env_path())
            return {"saved": True, "backup": str(bak) if bak else None}
        if path == "/api/secret":
            k = body.get("key")
            if k not in envspec.SECRETS:
                raise ValueError("not a secret setting")
            return {"key": k, "value": _setting(cur, k, "")}
        if path == "/api/models":
            return list_server_models(merge_secrets(values or cur, cur))
        if path == "/api/cmd":
            cmd = body.get("cmd")
            argv = command_argv(cmd, body)
            use = merge_secrets(values, cur, body.get("secrets")) if (values is not None and cmd in ("estimate", "check-env", "list-models")) else None
            if cmd == "check-env" and use is not None:
                return {"code": 0, "output": "", "report": check_values(use)}
            return run_command(argv, use, TIMEOUTS[cmd])
        if path == "/api/run":
            return {"started": True, "pid": start_run(body.get("mode", "all"))}
        if path == "/api/stop":
            return {"stopping": stop_run()}
        if path == "/api/pause":
            return {"pausing": pause_run(True)}
        if path == "/api/resume":
            return {"resuming": pause_run(False)}
        if path == "/api/fav/save":
            name = body.get("name", "")
            p = envfile.save_favourite(configs_dir(), name, {k: v for k, v in (values or {}).items()
                                                             if k not in envspec.SECRETS}, __version__)
            return {"saved": p.name, "favourites": envfile.list_favourites(configs_dir())}
        if path == "/api/fav/load":
            p = envfile.favourite_path(configs_dir(), body.get("name", ""))
            if not p.is_file():
                raise ValueError("no such favourite")
            res = envfile.apply_favourite(p, cur)
            res["values"] = masked(res["values"])
            return res
        if path == "/api/fav/delete":
            p = envfile.favourite_path(configs_dir(), body.get("name", ""))
            if p.is_file():
                p.unlink()
            return {"favourites": envfile.list_favourites(configs_dir())}
        return None


def _lan_urls(port, token):
    ips = set()
    try:
        ips |= {a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127.")}
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))                     # no packet is sent: just picks the LAN interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return [f"http://{ip}:{port}/?token={token}" for ip in sorted(ips)]


def _has_desktop():
    """Open a browser only on a desktop: on a headless Linux server webbrowser may start a text browser (lynx, w3m)
    in this terminal."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def bind_hint(e, host):
    """What to change when the page cannot listen on UI_HOST:UI_PORT."""
    if isinstance(e, socket.gaierror) or e.errno in (errno.EADDRNOTAVAIL, getattr(errno, "WSAEADDRNOTAVAIL", -1)):
        return (f"UI_HOST={host} is not an address of this machine. Use UI_HOST=0.0.0.0 (all of this machine's "
                "addresses), one of its own LAN IPs, or leave it empty for this PC only.")
    if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
        return "Another program (or a settings page already running) uses that port: stop it or set UI_PORT."
    if e.errno in (errno.EACCES, errno.EPERM):
        return "Not allowed to use that port: pick one above 1024 with UI_PORT."
    return "Check UI_HOST and UI_PORT in PART 3 of .env."


def serve():
    cfg = page_settings()
    if cfg.lan and cfg.token and len(cfg.token) < MIN_LAN_TOKEN:
        print(f"[error] UI_TOKEN is {len(cfg.token)} characters. With UI_HOST={cfg.host} the page is open to the "
              f"network, where the token is all that protects this machine: whoever has it can change the settings and "
              f"start runs. Use at least {MIN_LAN_TOKEN} characters, or delete the UI_TOKEN line from "
              f"{env_path()} and a random one is made on the next start (and printed in the URL).")
        return 1
    new_token = cfg.lan and not cfg.token                # LAN without a token: make one, kept in .env once it opens
    if new_token:
        cfg.token = secrets.token_urlsafe(24)
    try:
        srv = UIServer(cfg)
    except OSError as e:
        print(f"[error] cannot open {cfg.host}:{cfg.port}: {e}")
        print("  " + bind_hint(e, cfg.host))
        return 1
    if new_token:
        cur = envfile.read_values(env_path())
        cur["UI_TOKEN"] = cfg.token
        envfile.write_env(cur, env_path())
        print(f"Generated UI_TOKEN and saved it to {env_path()} (previous file: .env.bak)")
    local = f"http://{'localhost' if not cfg.lan else cfg.host}:{cfg.port}/"
    print(f"Settings page (v{__version__}) for {env_path()}")
    if cfg.lan:
        urls = _lan_urls(cfg.port, cfg.token) if cfg.host in ("0.0.0.0", "::") else \
            [f"http://{cfg.host}:{cfg.port}/?token={cfg.token}"]
        print("Open from another PC (the token is set as a cookie on the first visit):")
        for u in urls:
            print(f"  {u}")
        print("Plain HTTP: the token and any secret you reveal travel unencrypted on the LAN.")
    else:
        url = local + (f"?token={cfg.token}" if cfg.token else "")
        print(f"Open: {url}")
        if _has_desktop():
            try:
                webbrowser.open(url)
            except Exception:
                pass
    print("Stop with Ctrl+C.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nSettings page stopped.")
    finally:
        srv.server_close()
    return 0
