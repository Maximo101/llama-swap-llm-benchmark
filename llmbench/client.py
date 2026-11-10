"""Sync HTTP client helpers for llama-swap / Strata, backend detection and model selection."""
from __future__ import annotations

import fnmatch
import json
import re
import sys
import time
from urllib.parse import quote

import requests

from .common import BenchError, BenchTimeout

RETRY_STATUS = (502, 503, 504)      # a proxy/llama-swap that is mid-swap or briefly overloaded; worth another try


def http_error(cfg, status, body):
    """BenchError for a failed HTTP status, or None if the status is fine. Shared by the sync (requests) and the
    async (aiohttp) clients so both report errors identically."""
    if status in (401, 403):
        return BenchError(f"HTTP {status}: authentication failed - check the API key for {cfg.backend} ({body[:120]})")
    if status == 404:
        return BenchError("model not found (404)")
    if status >= 400:
        return BenchError(f"HTTP {status}: {body[:160]}")
    return None


def backoff_s(attempt):
    """2 s, 4 s, 8 s ... capped at 10 s."""
    return min(10.0, 2.0 * (2 ** attempt))


def auth_headers(cfg, backend):
    """Credentials for a backend. The Strata auth scheme is not known here, so its key is sent both as
    'Authorization: Bearer' and 'x-api-key' (servers ignore the header they don't use)."""
    key = cfg.strata_api_key if backend == "strata" else cfg.api_key
    if not key:
        return {}
    h = {"Authorization": f"Bearer {key}"}
    if backend == "strata":
        h["x-api-key"] = key
    return h


def post_chat(cfg, session, payload, timeout, on_attempt=None):
    """Synchronous chat completion with retries (502/503/504 and connection errors, exponential backoff).
    Timeouts are NOT retried: a request that already ran for `timeout` seconds would only burn the same time again."""
    last = "server busy"
    for attempt in range(cfg.retries + 1):
        if on_attempt is not None:
            on_attempt(attempt)                  # the caller re-reads its clocks: a retry starts a new measurement
        try:
            resp = session.post(cfg.chat_url, json=payload, timeout=timeout)
        except requests.Timeout:
            raise BenchTimeout(f"timeout after {timeout:.0f}s")
        except requests.ConnectionError as e:
            last = f"connection error: {e}"
            if attempt < cfg.retries:
                time.sleep(backoff_s(attempt))
                continue
            raise BenchError(last)
        except requests.RequestException as e:
            raise BenchError(f"connection error: {e}")
        if resp.status_code in RETRY_STATUS and attempt < cfg.retries:
            time.sleep(backoff_s(attempt))
            continue
        err = http_error(cfg, resp.status_code, resp.text)
        if err:
            raise err
        try:
            return resp.json()
        except ValueError:
            raise BenchError("response was not valid JSON")
    raise BenchError(last)


def is_outage(msg):
    """Did this request fail because the SERVER was not there (refused, reset, dropped, 5xx), rather than because
    of the model (a timeout, a bad request)? Those are worth asking again once the server is back (v2.3.1)."""
    m = str(msg or "").lower()
    if "timeout" in m:
        return False
    return "connection error" in m or bool(re.search(r"http 5\d\d", m))


def wait_ready(cfg, session=None, deadline_s=None, quiet=False):
    """v2.3.1: True as soon as the server lists its models again, False after SERVER_WAIT_SECONDS (0 = do not wait).
    Called when requests fail with connection errors mid-run: a restart of llama-swap (a config reload, a Docker
    restart, a crash) used to cost every model tested in that window its result."""
    limit = cfg.server_wait if deadline_s is None else deadline_s
    if limit <= 0:
        return False
    session = session or make_session(cfg, cfg.backend)
    url = cfg.models_url if cfg.backend != "strata" else cfg.base_url + "/v1/models"
    end, said = time.monotonic() + limit, False
    while True:
        try:
            if session.get(url, timeout=5).status_code < 500:
                if said and not quiet:
                    print("    [info] the server is answering again", flush=True)
                return True
        except requests.RequestException:
            pass
        if time.monotonic() >= end:
            return False
        if not said and not quiet:
            said = True
            print(f"    [wait] the server at {cfg.base_url} is not answering: waiting up to {limit:.0f}s for it "
                  "(SERVER_WAIT_SECONDS)", flush=True)
        time.sleep(min(5.0, max(0.2, end - time.monotonic())))


def unload_all(cfg, session) -> bool:
    for path in ("/api/models/unload", "/unload"):
        try:
            if session.post(cfg.base_url + path, timeout=30).status_code < 400:
                return True
        except requests.RequestException:
            pass
    return False



_STRATA_PATHS = ["/api/about", "/api/info", "/api/status", "/api/stats", "/api/system", "/api/monitor",
                 "/api/model", "/api/models", "/api/config", "/api/props", "/api/hardware", "/api/version",
                 "/props", "/health", "/v1/models"]
# Never call anything that could change state or expose chats, even with GET.
_STRATA_SKIP = re.compile(r"unload|load|stop|shutdown|delete|reset|clear|restart|kill|chat|complet|messages|"
                          r"embed|history|conversation|session|log|stream|event|request", re.I)
_STRATA_INFO = re.compile(r"about|info|model|system|config|status|health|version|props|hardware|engine", re.I)


def strata_discover(cfg, session) -> dict:
    """Find what Strata's own web UI reads: endpoints referenced in its JavaScript plus common guesses.
    GET only, state-changing / chat endpoints are skipped. Returns {path: json | {'_status': code}}."""
    from urllib.parse import urljoin
    paths = list(_STRATA_PATHS)
    try:
        html = session.get(cfg.strata_url + "/", timeout=5).text
        for src in re.findall(r'(?:src|href)=["\']([^"\']+\.js)["\']', html)[:8]:
            js = session.get(urljoin(cfg.strata_url + "/", src), timeout=10).text
            for m in re.findall(r'["\'`](/(?:api|v1)/[A-Za-z0-9_\-/]+)["\'`]', js) + \
                     re.findall(r'fetch\(\s*["\'`](/[A-Za-z0-9_\-/]+)["\'`]', js):
                if m not in paths:
                    paths.append(m)
    except requests.RequestException:
        pass
    found = {}
    for path in paths[:40]:
        if path != "/v1/models" and _STRATA_SKIP.search(path):
            continue
        try:
            r = session.get(cfg.strata_url + path, timeout=5)
        except requests.RequestException:
            continue
        if r.status_code >= 400:
            found[path] = {"_status": r.status_code}
            continue
        try:
            found[path] = r.json()
        except ValueError:
            found[path] = {"_text": r.text[:500]}
    return found


def _scalars(obj, prefix="", out=None, depth=0):
    out = out if out is not None else []
    if len(out) >= 60 or depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scalars(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:6]):
            _scalars(v, f"{prefix}[{i}]", out, depth + 1)
    elif isinstance(obj, (str, int, float, bool)) and not (isinstance(obj, str) and len(obj) > 120):
        out.append([prefix, obj])
    return out


def strata_details(found: dict) -> list:
    """Scalar values from the info-like endpoints, for the dashboard's 'Strata details' panel."""
    out = []
    for path, data in found.items():
        if path.startswith("/v1/") or not _STRATA_INFO.search(path) or not isinstance(data, dict) or "_status" in data:
            continue
        rows = _scalars(data)
        if rows:
            out.append([path, rows])
    return out


def ctx_from_details(details):
    """Context size (tokens) from Strata's info endpoints, if a plausible field is present."""
    pat = re.compile(r"(?:^|\.)(?:n_ctx|ctx_size|ctx|context_size|context_length|context_tokens|context|max_context)$", re.I)
    for _, rows in details or []:
        for k, v in rows:
            if pat.search(k) and isinstance(v, (int, float)) and 512 <= v <= 4_194_304:
                return int(v)
    return None


def probe_strata(cfg):
    """--probe-strata: list what Strata's API exposes and save the raw JSON for inspection."""
    if not cfg.strata_url:
        sys.exit("[error] Set STRATA_URL in .env first.")
    cfg.backend, cfg.base_url = "strata", cfg.strata_url
    found = strata_discover(cfg, make_session(cfg, "strata"))
    for path, data in found.items():
        if isinstance(data, dict) and "_status" in data:
            print(f"  GET {path} -> HTTP {data['_status']}")
        else:
            keys = list(data)[:14] if isinstance(data, dict) else f"list of {len(data)}"
            print(f"  GET {path} -> OK, {keys}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "strata_probe.json"
    text = json.dumps(found, indent=1)
    path.write_text(text if len(text) < 300_000 else text[:300_000], encoding="utf-8")
    print(f"\nRaw responses saved to {path}. Review it before sharing; it is data from your Strata server.")


def upstream_url(cfg, model):
    """Base URL of the model's own llama-server: llama-swap's /upstream/<model> route, or the Strata base URL.
    llama-server's native endpoints (/props, /apply-template, /completion, /slots) hang off it."""
    return cfg.base_url if cfg.backend == "strata" else f"{cfg.base_url}/upstream/{quote(model, safe='')}"


def get_props(cfg, session, model):
    """llama-server's /props for the loaded model (through llama-swap's /upstream/<model>/ route, or the Strata base
    URL). {} when unavailable. Holds n_ctx, model_path, total_slots (= --parallel), chat_template and build_info."""
    url = f"{upstream_url(cfg, model)}/props"
    try:
        r = session.get(url, timeout=10)
        if r.ok:
            j = r.json()
            return j if isinstance(j, dict) else {}
    except (requests.RequestException, ValueError):
        pass
    return {}


def props_ctx(props):
    n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    try:
        return int(n) if n else None
    except (TypeError, ValueError):
        return None


def running_cmd(cfg, session, model):
    """The launch command llama-swap reports in /running for `model`, as an argv list, or None. A fallback for the
    launch flags when the server process cannot be read from /proc (e.g. the benchmark runs on another machine)."""
    import shlex
    try:
        r = session.get(cfg.base_url + "/running", timeout=5)
        j = r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None
    for entry in (j.get("running", []) if isinstance(j, dict) else j or []):
        if not isinstance(entry, dict) or entry.get("model") != model:
            continue
        for v in entry.values():
            if isinstance(v, str) and re.search(r"(?:^|\s)(?:-m|--model|-c|--ctx-size)\s", v):
                try:
                    return shlex.split(v.replace("\\\n", " "))
                except ValueError:
                    return v.split()
    return None


def get_ctx(cfg, session, model, props=None):
    """Active context size (tokens) of the loaded model. Primary: llama-server's /props, reached
    through llama-swap's /upstream/<model>/ route. Fallback: a -c/--ctx-size flag in /running (if shown)."""
    if cfg.backend == "strata" and (cfg.strata_ctx or cfg.strata_ctx_detected):
        return cfg.strata_ctx or cfg.strata_ctx_detected
    n = props_ctx(props if props is not None else get_props(cfg, session, model))
    if n or cfg.backend == "strata":       # Strata: works if it is a llama-server; otherwise set STRATA_CTX
        return n
    from .telemetry.host import parse_launch_args
    argv = running_cmd(cfg, session, model)
    return parse_launch_args(argv)["ctx"] if argv else None


def make_session(cfg, backend):
    """requests session with the right credentials (see auth_headers)."""
    s = requests.Session()
    s.headers.update(auth_headers(cfg, backend))
    return s


def detect_backend(cfg):
    """Use llama-swap if it answers; otherwise fall back to a single Strata endpoint (if STRATA_URL is set).
    Returns (session, model_ids) and sets cfg.backend / cfg.base_url."""
    session, why = make_session(cfg, "llama-swap"), ""
    try:
        r = session.get(cfg.models_url, timeout=5)
        r.raise_for_status()
        ids = select_models(cfg, [m["id"] for m in r.json().get("data", [])])
        if ids:
            return session, ids
        why = "no model is left after MODEL_FILTER / INCLUDE_MODELS / EXCLUDE_MODELS, or the server lists none"
    except (requests.RequestException, ValueError, KeyError) as e:
        why = e.__class__.__name__
    if not cfg.strata_url:
        sys.exit(f"[error] llama-swap at {cfg.base_url} is unavailable ({why}). "
                 "Set STRATA_URL in .env to fall back to a single Strata endpoint.")

    print(f"[info] llama-swap unavailable ({why}); falling back to Strata at {cfg.strata_url}")
    cfg.backend, cfg.base_url = "strata", cfg.strata_url
    session = make_session(cfg, "strata")
    reachable, name = False, cfg.strata_model
    for path in ("/health", "/v1/models"):
        try:
            r = session.get(cfg.strata_url + path, timeout=5)
            reachable = True
            if path == "/v1/models" and r.ok:
                data = r.json().get("data") or []
                cfg.strata_model_id = data[0].get("id") if data and isinstance(data[0], dict) else None
                name = name or cfg.strata_model_id
        except (requests.RequestException, ValueError):
            pass
    if not reachable:
        sys.exit(f"[error] Strata at {cfg.strata_url} is not reachable either.")
    chosen = select_models(cfg, [name or "strata"])
    if not chosen:
        sys.exit("[error] The Strata model is excluded by INCLUDE_MODELS / EXCLUDE_MODELS / MODEL_FILTER.")
    return session, chosen


# --------------------------------------------------------------------------- #
# Model discovery and selection  (--list-models, INCLUDE_MODELS / EXCLUDE_MODELS)
# --------------------------------------------------------------------------- #

def _name_match(pattern, name):
    """Case-insensitive exact match; '*' and '?' act as wildcards (e.g. 'qwen*')."""
    p, n = pattern.lower(), name.lower()
    return fnmatch.fnmatchcase(n, p) if any(c in p for c in "*?") else p == n


def model_skip_reason(cfg, name):
    """None if the model is selected, otherwise a short reason why it is skipped."""
    if cfg.model_filter and not re.search(cfg.model_filter, name):
        return "no match for MODEL_FILTER"
    if cfg.include_models and not any(_name_match(p, name) for p in cfg.include_models):
        return "not in INCLUDE_MODELS"
    if any(_name_match(p, name) for p in cfg.exclude_models):
        return "excluded by EXCLUDE_MODELS"
    return None


def select_models(cfg, ids):
    """Apply MODEL_FILTER (regex), INCLUDE_MODELS and EXCLUDE_MODELS to the ids a server offers."""
    chosen = [m for m in ids if model_skip_reason(cfg, m) is None]
    unknown = [p for p in cfg.include_models if not any(_name_match(p, m) for m in ids)]
    if unknown and ids:
        print(f"  [warn] INCLUDE_MODELS entries not offered by the server: {', '.join(unknown)} "
              "(run --list-models to see the exact ids)")
    return chosen


def list_models(cfg):
    """--list-models: print what GET /v1/models returns (llama-swap, else Strata) and which are selected."""
    attempts = [("llama-swap", cfg.base_url)]
    if cfg.strata_url and cfg.strata_url != cfg.base_url:
        attempts.append(("Strata", cfg.strata_url))
    ids, errors = None, []
    for backend, base in attempts:
        session = make_session(cfg, "strata" if backend == "Strata" else "llama-swap")
        try:
            r = session.get(f"{base}/v1/models", timeout=10)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]
            print(f"Models at {base}/v1/models ({backend}):\n")
            break
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            errors.append(f"{base}/v1/models: {e.__class__.__name__}: {e}")
    if ids is None:
        sys.exit("[error] Could not list models.\n  " + "\n  ".join(errors))
    if not ids:
        print("  (the server lists no models)")
        return
    width = max(len(m) for m in ids)
    picked = 0
    for m in ids:
        why = model_skip_reason(cfg, m)
        picked += why is None
        print(f"  [{'x' if why is None else ' '}] {m:<{width}}" + (f"   <- skipped: {why}" if why else ""))
    print(f"\n{picked} of {len(ids)} selected for benchmarking "
          "([x] = will run). Narrow with INCLUDE_MODELS / EXCLUDE_MODELS in .env.")
