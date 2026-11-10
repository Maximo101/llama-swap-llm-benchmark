"""v2.3.1: a run that goes wrong keeps what it measured and can be resumed; outages are waited out; measurements are
taken over the attempt that answered; the settings page and the sandbox are harder to misuse."""
import json
import os
import stat
import sys
import time
import types

import pytest

import mock_server
from llmbench import cli, envfile, scoring, speed, status
from llmbench.client import is_outage
from llmbench.evals import sandbox
from llmbench.evals.runner import aggregate
from llmbench.evals.streaming import StreamGuard
from llmbench.reporting import dashboard
from llmbench.scoring import parse_active, parse_params
from llmbench.ui import server as ui_server
from test_pipeline import ev, run_cli
from test_v23_ui import _req, ui  # noqa: F401  (fixture)

M1, M2 = "Mock-9B-Q4_K_M.gguf", "Mock-30B-A3B-IQ3_XXS.gguf"


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


@pytest.fixture
def one_model(monkeypatch):
    """--models writes INCLUDE_MODELS into the environment for the run (as a shell variable would be); registering it
    with monkeypatch first puts the environment back afterwards."""
    monkeypatch.setenv("INCLUDE_MODELS", "")
    return ["--models", M1]


def _speed_requests(model):
    """Speed-test generations sent for `model` (the tech / logic prompts carry a max_tokens of 300 / 500)."""
    return [b for b in mock_server.STATE["requests"] if b.get("model") == model and b.get("max_tokens") in (300, 500)]


# --------------------------------------------------------------------------- scoring: the s/q basis
def _row(name, **kw):
    r = {"model": name, "status": "ok", "eval_avg_sec": 50.0, "eval_inflight": 4, "avg_tps": 100.0}
    r.update(kw)
    return r


def test_one_failed_dataset_does_not_switch_the_run_to_measured_latency():
    """v2.3.0: a dataset where every request failed has no single-stream estimate, and that put EVERY model on the
    measured (queue-inflated) latency. The 9 Nov 2026 run was scored that way after a llama-swap outage."""
    cfg = types.SimpleNamespace(eval_sq_basis="auto")
    ok = _row("healthy", eval_mmlu_pro_sec=80.0, eval_mmlu_pro_sec1=35.0, eval_mmlu_pro_n=11,
              eval_mbpp_sec=47.0, eval_mbpp_sec1=23.0, eval_mbpp_n=11)
    hit = _row("outage", eval_mmlu_pro_sec=68.0, eval_mmlu_pro_sec1=25.0, eval_mmlu_pro_n=11,
               eval_mbpp_sec=111.7, eval_mbpp_sec1=None, eval_mbpp_n=8, eval_mbpp_err=8)
    assert scoring._apply_sq_basis([ok, hit], cfg) == "single" and ok["eval_avg_sec"] == 29.0
    # an evals-only run (no speed test, so no estimate for anyone) still falls back to the measured latency
    a = _row("a", avg_tps=None, eval_mbpp_sec=40.0, eval_mbpp_sec1=None, eval_mbpp_n=11)
    assert scoring._apply_sq_basis([a], cfg) == "measured"


# --------------------------------------------------------------------------- nothing measured is lost
def test_an_internal_error_in_the_evals_still_writes_the_results(monkeypatch, tmp_path, server):
    import llmbench.evals.runner as runner
    real, seen = runner.aggregate, {"n": 0}

    def boom(*a, **k):
        seen["n"] += 1
        if seen["n"] >= 3:
            raise KeyError("a bug in a late code path")
        return real(*a, **k)
    monkeypatch.setattr(runner, "aggregate", boom)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["mmlu_pro", "gsm8k"])
    assert code == 2 and doc["run"]["status"] == "interrupted"            # partial results, and a non-zero exit code
    assert [m["status"] for m in doc["models"]] == ["ok", "ok"] and doc["models"][0]["speed"]["tech_tps"]
    out = tmp_path / "results"
    assert list(out.glob("benchmark_*.csv")) and list(out.glob("benchmark_dashboard_*.html"))
    assert status.load_rows(next(out.glob("run_*.rows.json")))["state"] == "interrupted"


def test_an_internal_error_on_one_model_keeps_the_others(monkeypatch, tmp_path, server):
    real, seen = speed._summarise, {"n": 0}

    def boom(*a, **k):
        seen["n"] += 1
        if seen["n"] == 2:
            raise ZeroDivisionError("a bug on model 2")
        return real(*a, **k)
    monkeypatch.setattr(speed, "_summarise", boom)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[])
    assert code == 0 and [m["status"] for m in doc["models"]] == ["ok", "error"]
    assert "internal error: ZeroDivisionError" in doc["models"][1]["error"]


def test_rows_are_saved_after_every_model_and_status_is_reported(monkeypatch, tmp_path, server):
    saved = []
    real = status.save_rows

    def spy(cfg, stamp, results, state="partial", info=None):
        saved.append((state, len(results)))
        return real(cfg, stamp, results, state, info)
    monkeypatch.setattr(status, "save_rows", spy)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gsm8k"])
    assert code == 0
    assert saved[:2] == [("partial", 1), ("partial", 2)] and saved[-1] == ("complete", 2) and len(saved) >= 5
    st = status.read(tmp_path / "results")
    assert st["state"] == "complete" and st["stamp"] == doc["run"]["id"] and st["host"] == status.hostname()
    assert doc["artifacts"]["rows"].endswith(f"run_{doc['run']['id']}.rows.json")
    rows = status.load_rows(doc["artifacts"]["rows"])
    assert rows["state"] == "complete" and all("--api-key" not in (r.get("_launch_argv") or []) for r in rows["rows"])


def test_rows_file_never_holds_an_api_key():
    assert status._safe_argv(["-m", "x.gguf", "--api-key", "sk-secret", "--api-key-file=/k", "-c", "8192"]) == \
        ["-m", "x.gguf", "-c", "8192"]


# --------------------------------------------------------------------------- --resume
def test_resume_keeps_finished_speed_tests_and_cached_answers(monkeypatch, tmp_path, server):
    code, _, first = run_cli(monkeypatch, tmp_path, server, [], datasets=["gsm8k"])
    assert code == 0 and len(_speed_requests(M1)) > 0 and len(_speed_requests(M2)) > 0
    rows_file = next((tmp_path / "results").glob("run_*.rows.json"))
    doc = json.loads(rows_file.read_text())
    doc["state"] = "partial"                                   # as if the run died while model 2 was being benchmarked
    doc["rows"] = [r for r in doc["rows"] if r["model"] == M1]
    rows_file.write_text(json.dumps(doc))
    time.sleep(1.1)                                            # a new stamp (they are per second)
    code, _, second = run_cli(monkeypatch, tmp_path, server, ["--resume"], datasets=["gsm8k"])
    assert code == 0
    assert _speed_requests(M1) == [] and len(_speed_requests(M2)) > 0            # only model 2 was benchmarked
    assert [m["status"] for m in second["models"]] == ["ok", "ok"]
    a, b = first["models"][0], second["models"][0]
    assert b["speed"]["tech_tps"] == a["speed"]["tech_tps"] and b["gpu"] == a["gpu"]      # model 1: the kept figures
    assert ev(b, "gsm8k")["pct"] == ev(a, "gsm8k")["pct"] and ev(b, "gsm8k")["cached"] == ev(b, "gsm8k")["n"]
    assert second["artifacts"]["resumed_from"] == first["run"]["id"] and second["run"]["id"] != first["run"]["id"]
    assert b["scores"]["score_chat"] == pytest.approx(a["scores"]["score_chat"])


def test_resume_benchmarks_a_model_that_failed_before(monkeypatch, tmp_path, server):
    mock_server.STATE["load_fail"] = {M2}
    try:
        code, _, first = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[])
    finally:
        mock_server.STATE["load_fail"] = set()
    assert [m["status"] for m in first["models"]] == ["ok", "error"]
    time.sleep(1.1)
    code, _, second = run_cli(monkeypatch, tmp_path, server, ["--speed-only", "--resume", first["run"]["id"]], datasets=[])
    assert code == 0 and [m["status"] for m in second["models"]] == ["ok", "ok"] and _speed_requests(M1) == []


def test_resume_without_a_run_says_what_to_do(monkeypatch, tmp_path, server, capsys):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only", "--resume"], datasets=[])
    assert code == 1 and doc is None and "no run to resume" in capsys.readouterr().err


# --------------------------------------------------------------------------- outages
def test_is_outage_tells_the_server_from_the_model():
    assert is_outage("api error: connection error: Cannot connect to host 192.168.2.200:8000")
    assert is_outage("api error: HTTP 503: llama-swap is restarting") and is_outage("HTTP 502: bad gateway")
    assert not is_outage("api error: timeout after 600s") and not is_outage("HTTP 400: request exceeds the context")
    assert not is_outage("") and not is_outage(None)


def test_questions_lost_to_an_outage_are_asked_again(monkeypatch, tmp_path, server, one_model):
    """The server answers 503 to the next 4 questions (RETRIES=0, so each is an API error), then works again. v2.3.0
    counted those as wrong answers; now they are asked once more."""
    mock_server.STATE["outage_left"] = 4
    try:
        code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only", *one_model], datasets=["gsm8k"],
                               RETRIES="0", EVAL_RESULT_CACHE="false", EVAL_MAX_CONSEC_ERRORS="50",
                               SERVER_WAIT_SECONDS="20")
    finally:
        mock_server.STATE["outage_left"] = 0
    g = ev(doc["models"][0], "gsm8k")
    assert code == 0 and len(doc["models"]) == 1                              # --models picked one
    assert g["err"] == 0 and g["n"] == 6 and g["pct"] == pytest.approx(83.33, abs=0.01)
    assert doc["models"][0]["other"]["eval_reasked_n"] == 4


def test_no_waiting_when_switched_off(monkeypatch, tmp_path, server, one_model):
    mock_server.STATE["outage_left"] = 4
    try:
        code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only", *one_model], datasets=["gsm8k"],
                               RETRIES="0", EVAL_RESULT_CACHE="false", EVAL_MAX_CONSEC_ERRORS="50",
                               SERVER_WAIT_SECONDS="0")
    finally:
        mock_server.STATE["outage_left"] = 0
    assert ev(doc["models"][0], "gsm8k")["err"] == 4                           # as v2.3.0


# --------------------------------------------------------------------------- what is timed
class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.text = code, body, "busy"

    def json(self):
        return self._b


def test_a_retry_is_not_part_of_the_measurement():
    class Sess:
        n = 0

        def post(self, url, json=None, timeout=None):
            self.n += 1
            if self.n == 1:
                return _Resp(503)                                  # then post_chat sleeps 2 s and sends again
            time.sleep(0.05)
            return _Resp(200, {"usage": {"completion_tokens": 100}, "timings": {"predicted_per_second": 100.0}})

    class Gpu:
        ok = False

        def energy_mj(self):
            return int(time.monotonic() * 300_000)                 # a GPU drawing 300 W

        def window(self, a, b):
            return []
    host = types.SimpleNamespace(ram=types.SimpleNamespace(window=lambda a, b: []), snap=lambda: {},
                                 diff=lambda a, b: {})
    cfg = types.SimpleNamespace(temperature=None, retries=2, chat_url="x", backend="llama-swap", gpu_poll_ms=200)
    x = speed.measure(cfg, Sess(), Gpu(), host, None, "m", "p", 100, 60)
    assert x["attempts"] == 2 and x["wall_s"] < 0.5 and x["energy_j"] < 150 and x["tps_basis"] == "server"


def test_tokens_per_second_without_server_timings_is_labelled():
    sess = types.SimpleNamespace(post=lambda url, json=None, timeout=None: _Resp(200, {"usage": {"completion_tokens": 50}}))
    gpu = types.SimpleNamespace(ok=False, energy_mj=lambda: None, window=lambda a, b: [])
    host = types.SimpleNamespace(ram=types.SimpleNamespace(window=lambda a, b: []), snap=lambda: {}, diff=lambda a, b: {})
    cfg = types.SimpleNamespace(temperature=None, retries=0, chat_url="x", backend="strata", gpu_poll_ms=200)
    assert speed.measure(cfg, sess, gpu, host, None, "m", "p", 50, 60)["tps_basis"] == "wall"


def test_speed_samples_and_spread_are_kept(monkeypatch, tmp_path, server):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[], RUNS_PER_PROMPT="3")
    sp = doc["models"][0]["speed"]
    assert code == 0 and len(sp["tps_samples"]["tech"]) == 3 and sp["tps_spread_pct"] == 0.0 and sp["tps_basis"] == "server"


def test_clock_does_not_follow_the_wall_clock(monkeypatch):
    from llmbench import common
    a = common.now()
    monkeypatch.setattr(time, "time", lambda: 0.0)                 # NTP steps the clock back to 1970
    b = common.now()
    assert 0 <= b - a < 1 and abs(a - common._CLOCK0 - time.monotonic()) < 1


def test_first_token_times_from_the_stream():
    g = StreamGuard()
    g._t0 -= 2.0                                                   # as if the request went out 2 s ago
    g.feed({"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
    assert 2.0 <= g.ttft < 2.5 and g.ttfa is None                  # a token, but nothing visible yet
    g._t0 -= 3.0
    g.feed({"choices": [{"delta": {"content": "Answer: B"}}]})
    res = g.result()
    assert 2.0 <= res.ttft < 2.5 and 5.0 <= res.ttfa < 5.5
    agg = aggregate([{"correct": True, "ttft_s": 1.0, "ttfa_s": 4.0}, {"correct": True, "ttft_s": 3.0, "ttfa_s": 8.0},
                     {"correct": False}])
    assert (agg["ttft"], agg["ttfa"], agg["ttfa95"]) == (2.0, 6.0, 8.0) and agg["pct"] == pytest.approx(66.67, abs=0.01)


def test_streamed_evals_report_first_token_times(monkeypatch, tmp_path, server, one_model):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only", *one_model], datasets=["gsm8k"],
                           EVAL_RESULT_CACHE="false")
    g = ev(doc["models"][0], "gsm8k")
    assert code == 0 and g["ttft"] is not None and g["ttfa"] is not None and g["ttfa95"] >= g["ttfa"] >= g["ttft"]


def test_deferred_prefill_needs_a_cold_prompt(monkeypatch):
    """The long-context document is the same in every run. If the server answers its first question from the prompt
    cache, the reported prompt speed covers a handful of tokens and is not a prefill-at-length measurement."""
    from llmbench.evals import longctx
    cfg = types.SimpleNamespace(longctx_kv_min="q8_0", longctx_kv_action="flag", longctx_lengths=(65536,),
                                longctx_trials=1, longctx_keys=2, eval_timeout=60, eval_seed=1, defer_prefill=(65536,))
    for cache_n, kept in ((61000, None), (0, 2500.0), (None, 2500.0)):
        monkeypatch.setattr(longctx, "_ask", lambda *a, _c=cache_n, **k: ("", 62000, {"prompt_per_second": 2500.0,
                                                                                     "cache_n": _c}))
        r = {"model": "m", "kv_type": "q8_0", "tok_per_word": 1.2}
        longctx.run_longctx(cfg, None, r, 131072)
        assert r.get("prefill_64k_tps") == kept
        assert ("not cold" in (r["longctx_note"] or "")) == (kept is None)


# --------------------------------------------------------------------------- change against the previous run
def test_speed_change_against_the_previous_run():
    prev = {"m": {"tps": 100.0, "build": "b100", "ctx": 8192, "gb": 5.0, "offload": "none", "run": "old"},
            "other-ctx": {"tps": 100.0, "build": "b100", "ctx": 4096, "gb": 5.0, "offload": "none", "run": "old"}}
    rows = [{"model": "m", "status": "ok", "avg_tps": 80.0, "ctx_size": 8192, "gguf_gb": 5.0, "offload": "none",
             "build_info": "b200"},
            {"model": "other-ctx", "status": "ok", "avg_tps": 80.0, "ctx_size": 8192, "gguf_gb": 5.0, "offload": "none"},
            {"model": "new", "status": "ok", "avg_tps": 80.0}]
    lines = cli.speed_changes(rows, prev, "new")
    assert rows[0]["tps_change_pct"] == -20.0 and rows[0]["build_prev"] == "b100" and "b100 -> b200" in lines[0]
    assert "tps_change_pct" not in rows[1] and "tps_change_pct" not in rows[2] and len(lines) == 1


# --------------------------------------------------------------------------- run lock
def test_a_speed_run_refuses_to_start_beside_another(tmp_path):
    cfg = types.SimpleNamespace(output_dir=tmp_path)
    child = __import__("subprocess").Popen([sys.executable, "-c", "import time\ntime.sleep(60)", "benchmark"])
    try:
        (tmp_path / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None}))
        with pytest.raises(SystemExit) as e:
            cli.run_lock(cfg, None, exclusive=True)
        assert "was not started" in str(e.value) and json.loads((tmp_path / cli.LOCK_NAME).read_text())["pid"] == child.pid
        assert cli.run_lock(cfg, None, exclusive=False) is not None           # an evals-only run is only warned
        (tmp_path / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None}))
        assert cli.run_lock(cfg, None, exclusive=True, force=True) is not None
    finally:
        child.kill()
        child.wait()


def test_lock_from_another_machine_is_live_while_fresh(tmp_path):
    p = tmp_path / cli.LOCK_NAME
    p.write_text(json.dumps({"pid": 1, "started": "now", "log": None, "host": "some-other-host"}))
    info = cli.lock_info(tmp_path)
    assert info["remote"] is True and info["host"] == "some-other-host"
    old = time.time() - cli.REMOTE_LOCK_MAX_AGE - 60
    os.utime(p, (old, old))
    assert cli.lock_info(tmp_path) is None                                    # no heartbeat for too long: stale
    cfg = types.SimpleNamespace(output_dir=tmp_path)
    lock = cli.run_lock(cfg, None, exclusive=True)
    assert json.loads(lock.read_text())["host"] == status.hostname()


# --------------------------------------------------------------------------- model names
def test_effective_parameters_in_a_name():
    assert parse_params("NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf") == 4 and parse_params("gemma-3n-E2B-it") == 2
    assert parse_params("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") == 35 and parse_active("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") == 3
    assert parse_params("Model-E4B-12B-Q4.gguf") == 12                # a plain size wins
    assert parse_params("SomeE4B.gguf") is None and parse_active("X-E4B.gguf") is None


# --------------------------------------------------------------------------- settings page and secrets
def test_wrong_tokens_lock_an_address_out(ui):  # noqa: F811
    u = ui(token="a-long-enough-token-1234")
    for _ in range(ui_server.MAX_BAD_TOKENS):
        assert _req(u.port, "GET", "/?token=guess")[0] == 401
    assert _req(u.port, "GET", "/?token=a-long-enough-token-1234")[0] == 429        # even the right one, for a while
    u.srv._bad.clear()
    assert _req(u.port, "GET", "/?token=a-long-enough-token-1234")[0] == 303
    assert _req(u.port, "GET", "/api/state")[0] == 401 and not u.srv._bad           # no token at all is not a guess


def test_short_lan_token_is_refused(monkeypatch, tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text("UI_HOST=0.0.0.0\nUI_TOKEN=abc123\n")
    monkeypatch.setattr(ui_server.config, "ENV_FILE", env)
    for k in ("UI_HOST", "UI_TOKEN", "UI_PORT"):
        monkeypatch.delenv(k, raising=False)
    assert ui_server.serve() == 1 and "at least 16 characters" in capsys.readouterr().out


def test_dashboards_are_served_in_a_sandbox(ui):  # noqa: F811
    u = ui()
    (u.dir / "results" / "benchmark_dashboard_20261110_000000.html").write_text("<html>dash</html>")
    st, hd, _ = _req(u.port, "GET", "/results/benchmark_dashboard_20261110_000000.html")
    assert st == 200 and hd["Content-Security-Policy"].startswith("sandbox allow-scripts")
    assert "allow-same-origin" not in hd["Content-Security-Policy"]
    st, hd, _ = _req(u.port, "GET", "/")
    assert "sandbox" not in hd["Content-Security-Policy"]                           # the page itself keeps its origin


def test_only_a_run_log_in_the_output_folder_is_read(ui, tmp_path):  # noqa: F811
    u = ui()
    good = u.dir / "results" / "benchmark_20261110_000000.log"
    good.write_text("hello")
    secret = u.dir / "secret.txt"
    secret.write_text("TOKEN")
    assert ui_server.safe_log(str(good)) == str(good.resolve())
    assert ui_server.safe_log(str(secret)) is None and ui_server.safe_log("/etc/passwd") is None
    assert ui_server.safe_log(str(u.dir / "results" / ".." / "secret.txt")) is None
    (u.dir / "results" / "notes.log").write_text("x")
    assert ui_server.safe_log(str(u.dir / "results" / "notes.log")) is None and ui_server.safe_log(None) is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_env_files_are_private(tmp_path):
    env = tmp_path / ".env"
    old = os.umask(0)                                               # the worst case: a umask that hides nothing
    try:
        envfile.write_env({"HF_TOKEN": "hf_x"}, env)
        envfile.write_env({"HF_TOKEN": "hf_y"}, env)
    finally:
        os.umask(old)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / ".env.bak").stat().st_mode) == 0o600


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="only matters when the benchmark is root")
def test_generated_code_does_not_run_as_root(monkeypatch):
    """As root, the subprocess sandbox starts the program as 'nobody': a file only root may read stays unread. (Inside
    `unshare -rn` the program still SEES uid 0; that is the namespace's root, mapped to nobody outside.)"""
    import tempfile
    monkeypatch.setattr(sandbox, "_DROP", None)
    monkeypatch.setattr(sandbox, "_NETLESS", None)
    cfg = types.SimpleNamespace(eval_sandbox_resolved="subprocess", eval_code_mem_mb=2048)
    fd, secret = tempfile.mkstemp(prefix="llmb-root-only-", dir="/tmp")
    os.write(fd, b"TOKEN")
    os.close(fd)
    os.chmod(secret, 0o600)
    try:
        assert sandbox.drop_privileges()["user"] == sandbox.NOBODY
        ok, why = sandbox.run_in_sandbox(cfg, f"try:\n    open({secret!r}).read()\n    raise SystemExit(1)\n"
                                              "except PermissionError:\n    pass\n", "script", [], 10)
        assert ok, why
        ok, why = sandbox.run_in_sandbox(cfg, "import tempfile\nopen(tempfile.gettempdir() + '/x', 'w').write('1')\n",
                                         "script", [], 10)
        assert ok, why                                              # its own scratch folder is still writable
    finally:
        os.unlink(secret)


# --------------------------------------------------------------------------- dashboard
def test_dashboard_carries_chart_js_and_needs_no_cdn(monkeypatch, tmp_path, server):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[])
    html = next((tmp_path / "results").glob("benchmark_dashboard_*.html")).read_text(encoding="utf-8")
    assert code == 0 and "cdnjs.cloudflare.com" not in html and "Chart.js v4.4.1" in html
    assert "__DATA__" not in html.split("Chart.js v4.4.1")[0] and '"models"' in html
    assert dashboard.CHARTJS_PATH.is_file() and (dashboard.CHARTJS_PATH.parent / "LICENSE-chartjs.md").is_file()
