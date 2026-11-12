"""Pause and resume, and the GPU temperature guard (pause.py): by hand and by MAX_GPU_TEMP; the run waits without
dropping a request, saves its state, and starts the half-done test line again from its first step; finished work and
scores are unchanged."""
import asyncio
import json
import sys
import threading
import time
import types

import pytest

import fake_datasets
import mock_server
from llmbench import cli, pause, speed, status
from llmbench.evals import runner
from llmbench.pause import CONTROL, PauseGaveUp, PauseRestart, ThermalGuard
from llmbench.ui import server as ui_server
from test_pipeline import ev, run_cli
from test_v23_ui import _req, ui  # noqa: F401  (fixture)

M1, M2 = "Mock-9B-Q4_K_M.gguf", "Mock-30B-A3B-IQ3_XXS.gguf"


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def fresh_control(monkeypatch):
    CONTROL.reset()
    monkeypatch.setattr(pause, "POLL_S", 0.1)
    monkeypatch.setattr(pause, "REPORT_S", 0.5)
    yield
    CONTROL.reset()
    status._S.update(path=None, touch=None, data={})


def wait_for(cond, timeout=40.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting")


def cfg_for(tmp_path, **kw):
    base = dict(output_dir=tmp_path, base_url="http://127.0.0.1:1", max_gpu_temp=0, gpu_temp_resume_delta=10,
                gpu_temp_max_restarts=3, gpu_poll_ms=200)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- the guard's decision
def test_guard_trips_at_the_limit_and_cools_after_a_few_samples_at_the_resume_level():
    g = ThermalGuard(CONTROL, None, 85, 75)
    g.feed(80)
    assert not CONTROL.tripped
    g.feed(85)                                         # the limit itself counts
    assert CONTROL.tripped and CONTROL.trip_temp == 85
    g.feed(None)                                       # no reading: nothing changes
    g.feed(80)
    g.feed(75)
    g.feed(74)
    assert not g.cooled()                              # two samples at the resume level are not enough
    g.feed(79)                                         # between the two levels: the count starts again
    g.feed(70)
    g.feed(70)
    assert not g.cooled()
    g.feed(70)
    assert g.cooled()
    g.feed(86)                                         # hot again: cool count back to zero, hottest reading kept
    assert not g.cooled() and CONTROL.trip_temp == 86


def test_a_run_that_is_not_configured_never_pauses(tmp_path):
    assert CONTROL.requested() is False and CONTROL.wait() == 0.0
    CONTROL.tripped = True                             # even a stray flag: the checkpoint only acts inside restartable()
    CONTROL.checkpoint()
    with CONTROL.restartable(), pytest.raises(PauseRestart):
        CONTROL.checkpoint()


def test_configure_clears_a_pause_left_by_an_earlier_run(tmp_path):
    pause.write_control(tmp_path, True)
    CONTROL.configure(cfg_for(tmp_path), "20261112_120000")
    assert not pause.control_path(tmp_path).exists() and CONTROL.requested() is False


def test_too_many_thermal_restarts_end_the_run_like_ctrl_c(tmp_path):
    CONTROL.configure(cfg_for(tmp_path, gpu_temp_max_restarts=2), "s1")
    CONTROL.max_c = 85.0
    for _ in range(2):
        CONTROL.tripped = True
        CONTROL.discard("the speed test of X")         # restarts 1 and 2 are allowed
    CONTROL.tripped = True
    with pytest.raises(PauseGaveUp) as e:
        CONTROL.discard("the speed test of X")
    assert isinstance(e.value, KeyboardInterrupt) and "MAX_GPU_TEMP" in str(e.value) and "--resume" in str(e.value)
    CONTROL.tripped = False
    CONTROL.discard("the speed test of Y")             # a manual pause is never counted
    assert CONTROL.restarts == {"the speed test of X": 3}


# --------------------------------------------------------------------------- waiting, and the state files
def test_manual_pause_waits_saves_state_and_carries_on(tmp_path, capsys):
    saved = []
    rows = [{"model": "a", "status": "ok", "avg_tps": 50.0, "eval_gsm8k_pct": 80.0},
            {"model": "b", "status": "error", "avg_tps": None}]
    status.start(tmp_path, stamp="s2", phase="evals")
    status.update(force=True, phase="evals", model="b", model_i=2, models=3, dataset="GSM8K", done=7, total=20)
    CONTROL.configure(cfg_for(tmp_path), "s2", rows, lambda: saved.append(1))
    pause.write_control(tmp_path, True)
    seen = {}

    def release():
        wait_for(lambda: pause.pause_path(tmp_path, "s2").exists())
        seen["file"] = json.loads(pause.pause_path(tmp_path, "s2").read_text())
        wait_for(lambda: (status.read(tmp_path) or {}).get("paused"))
        seen["status"] = status.read(tmp_path)
        pause.write_control(tmp_path, False)
    t = threading.Thread(target=release)
    t.start()
    waited = CONTROL.wait("test")
    t.join()
    assert waited > 0 and saved == [1]                                    # the rows were saved at the moment of the pause
    d = seen["file"]
    assert d["reason"] == "manual" and d["phase"] == "evals" and d["model"] == "b" and d["model_i"] == 2
    assert d["dataset"] == "GSM8K" and d["questions_done"] == 7 and d["questions_total"] == 20
    assert d["completed_models"] == ["a"] and d["rows_file"] == "run_s2.rows.json"
    assert d["results"][0]["datasets_scored"] == 1 and "restarts" in d
    assert seen["status"]["paused"] is True and seen["status"]["pause_reason"] == "manual"
    assert not pause.pause_path(tmp_path, "s2").exists()                  # temporary: gone once the run carries on
    assert status.read(tmp_path)["paused"] is False
    out = capsys.readouterr().out
    assert "[pause]" in out and "carrying on" in out
    CONTROL.shutdown()
    assert CONTROL.paused_s > 0 and CONTROL.pauses == 1


def test_thermal_pause_resumes_by_itself_when_the_gpu_is_cool(tmp_path):
    CONTROL.configure(cfg_for(tmp_path), "s3")
    CONTROL.max_c, CONTROL.resume_c = 85.0, 75.0
    CONTROL.guard = ThermalGuard(CONTROL, None, 85, 75)
    CONTROL.guard.feed(88)
    assert CONTROL.requested()
    temps = iter([84, 80, 74, 73, 72, 72, 72])
    t0 = time.time()

    def cool():
        for v in temps:
            time.sleep(0.15)
            CONTROL.guard.feed(v)
    threading.Thread(target=cool, daemon=True).start()
    waited = CONTROL.wait("test")
    assert waited >= 0.5 and not CONTROL.tripped and time.time() - t0 < 10
    assert CONTROL.requested() is False


def test_a_spike_that_cooled_before_the_checkpoint_still_restarts_the_line(tmp_path):
    CONTROL.configure(cfg_for(tmp_path), "s4")
    CONTROL.max_c, CONTROL.resume_c = 85.0, 75.0
    g = CONTROL.guard = ThermalGuard(CONTROL, None, 85, 75)
    g.feed(90)
    for _ in range(3):
        g.feed(60)                                      # cool again before the request ended
    with CONTROL.restartable(), pytest.raises(PauseRestart):
        CONTROL.checkpoint()                            # the measurement is hot all the same: dropped
    assert CONTROL.wait("test") >= 0 and not CONTROL.tripped      # already cool: released at once


def test_manual_pause_while_a_gpu_trip_waits_for_both(tmp_path):
    CONTROL.configure(cfg_for(tmp_path), "s5")
    CONTROL.max_c, CONTROL.resume_c = 85.0, 75.0
    g = CONTROL.guard = ThermalGuard(CONTROL, None, 85, 75)
    g.feed(90)
    for _ in range(3):
        g.feed(60)
    pause.write_control(tmp_path, True)
    done = []
    t = threading.Thread(target=lambda: done.append(CONTROL.wait("x")))
    t.start()
    time.sleep(0.5)
    assert not done                                     # cool, but the manual pause still holds
    pause.write_control(tmp_path, False)
    t.join(10)
    assert done and not t.is_alive()


# --------------------------------------------------------------------------- the speed test restarts as a whole
def test_a_speed_test_stopped_by_the_gpu_is_discarded_and_done_again(tmp_path, monkeypatch, capsys):
    CONTROL.configure(cfg_for(tmp_path), "s6")
    CONTROL.max_c, CONTROL.resume_c = 85.0, 75.0
    g = CONTROL.guard = ThermalGuard(CONTROL, None, 85, 75)
    calls = []

    def fake(cfg, session, gpu, host, ha, model, unload_state):
        calls.append(model)
        if len(calls) == 1:                              # the first try: a request ends and the GPU is too hot
            g.feed(90)
            CONTROL.checkpoint()
        return {"model": model, "status": "ok", "avg_tps": 99.0}
    monkeypatch.setattr(speed, "_bench_model", fake)
    threading.Timer(0.3, lambda: [g.feed(60) for _ in range(3)]).start()
    r = speed.bench_model(None, None, None, None, None, "M", {})
    assert calls == ["M", "M"] and r["avg_tps"] == 99.0
    assert "discarded" in capsys.readouterr().out and CONTROL.restarts == {"the speed test of M": 1}


# --------------------------------------------------------------------------- the long-context test restarts as a whole
def test_a_long_context_test_stopped_by_the_gpu_leaves_the_row_as_it_was(tmp_path, monkeypatch):
    CONTROL.configure(cfg_for(tmp_path), "s7")
    CONTROL.max_c, CONTROL.resume_c = 85.0, 75.0
    g = CONTROL.guard = ThermalGuard(CONTROL, None, 85, 75)
    r = {"model": "M", "status": "ok", "avg_tps": 90.0, "prefill_8k_tps": 1000.0}
    runs, sunk = [], []

    def fake_longctx(cfg, session, row, ctx, sink=None):
        runs.append(1)
        row["longctx_8k_pct"] = 100.0
        row["prefill_32k_tps"] = 500.0
        sink({"id": "longctx-8k-0", "model": "M", "dataset": "longctx"})
        if len(runs) == 1:
            g.feed(91)
            CONTROL.checkpoint()                         # too hot after the first length
        row["longctx_note"] = None
    monkeypatch.setattr(runner, "run_longctx", fake_longctx)
    phase = runner.EvalPhase.__new__(runner.EvalPhase)
    phase.cfg, phase.session, phase.sink = cfg_for(tmp_path), None, sunk.append
    threading.Timer(0.3, lambda: [g.feed(60) for _ in range(3)]).start()
    asyncio.run(phase._longctx(r, types.SimpleNamespace(ctx=65536)))
    assert len(runs) == 2 and r["longctx_8k_pct"] == 100.0 and r["prefill_8k_tps"] == 1000.0
    assert "longctx_note" in r
    assert [s.get("pause_restart") for s in sunk] == [None, 1]            # the repeated record says it is a repeat


# --------------------------------------------------------------------------- whole runs
def _paused(out):
    return lambda: (status.read(out) or {}).get("paused")


def test_pause_during_the_evals_restarts_the_dataset_and_changes_no_score(monkeypatch, tmp_path, server):
    """Pause part-way through GSM8K: the requests in flight finish, the run waits, then the dataset starts again from
    its first question and the answers already given come from the result cache. Scores equal an uninterrupted run's."""
    big = {"gsm8k": lambda: fake_datasets.gsm8k(24)}
    (tmp_path / "plain").mkdir()
    (tmp_path / "paused").mkdir()
    code, _, plain = run_cli(monkeypatch, tmp_path / "plain", server, [], datasets=["gsm8k"], fake=big)
    assert code == 0
    out = tmp_path / "paused" / "results"
    seen = {}

    def gsm_lines():                                   # answers written so far (the status file is throttled)
        return sum(f.read_text().count('"dataset": "gsm8k"') for f in out.glob("eval_details_*.jsonl"))

    def driver():
        wait_for(lambda: gsm_lines() >= 6)
        pause.write_control(out, True)
        wait_for(_paused(out))
        seen["file"] = json.loads(next(out.glob("run_*.pause.json")).read_text())
        seen["rows"] = next(out.glob("run_*.rows.json")).exists()
        time.sleep(0.3)
        pause.write_control(out, False)
    t = threading.Thread(target=driver)
    t.start()
    code, _, paused_doc = run_cli(monkeypatch, tmp_path / "paused", server, [], datasets=["gsm8k"], fake=big)
    t.join(30)
    assert code == 0 and seen["rows"]
    d = seen["file"]
    assert d["phase"] == "evals" and d["reason"] == "manual" and d["questions_total"] == 24
    assert " for Mock-" in d["restarts"]                                       # names the dataset and the model
    for a, b in zip(plain["models"], paused_doc["models"]):
        assert ev(b, "gsm8k")["n"] == 24 and ev(b, "gsm8k")["pct"] == ev(a, "gsm8k")["pct"]
        assert b["scores"]["score_chat"] == pytest.approx(a["scores"]["score_chat"])
    assert any(ev(m, "gsm8k")["cached"] > 0 for m in paused_doc["models"])      # the restart reused the answers given
    lines = [json.loads(x) for f in out.glob("eval_details_*.jsonl") for x in f.read_text().splitlines()]
    for m in (M1, M2):
        ids = [x["id"] for x in lines if x["model"] == m and x["dataset"] == "gsm8k"]
        assert len(ids) == len(set(ids)) == 24                                  # nothing written twice
    assert not list(out.glob("run_*.pause.json")) and not (out / pause.CONTROL_NAME).exists()
    assert "paused_s" in json.dumps(paused_doc)


def test_the_gpu_gets_too_hot_mid_speed_test_and_the_run_carries_on_by_itself(monkeypatch, tmp_path, server, capsys):
    """MAX_GPU_TEMP end to end with a fake GPU sensor: the model being measured when the limit is reached is tested
    again from the start once the GPU is cool; both models end up measured."""
    temp = {"v": 60.0}
    mock_server.STATE["calls"] = 0

    class FakeMonitor:
        ok, backend = True, "nvml"

        def __init__(self, cfg):
            self.series = types.SimpleNamespace(last=lambda: {"gpu_temp": temp["v"]})

        def window(self, t0, t1):
            return [{"gpu_temp": temp["v"]}]

        def stop(self):
            pass
    import llmbench.telemetry.gpu as gpu_mod
    real = gpu_mod.open_gpu_monitor
    monkeypatch.setattr(gpu_mod, "open_gpu_monitor", lambda cfg, verbose=False: FakeMonitor(cfg))
    out = tmp_path / "results"

    def driver():
        wait_for(lambda: mock_server.STATE["calls"] >= 3)
        temp["v"] = 91.0                                                   # too hot during model 1's speed test
        wait_for(_paused(out))
        assert status.read(out)["pause_reason"] == "thermal" and status.read(out)["pause_max"] == 85.0
        temp["v"] = 60.0                                                   # cooled down
    t = threading.Thread(target=driver)
    t.start()
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[],
                           MAX_GPU_TEMP="85", GPU_TEMP_RESUME_DELTA="10")
    t.join(30)
    assert gpu_mod.open_gpu_monitor is not real
    assert code == 0 and [m["status"] for m in doc["models"]] == ["ok", "ok"]
    text = capsys.readouterr().out
    assert "Thermal safety: the run pauses at 85 C" in text and "it is discarded" in text and "carrying on" in text
    assert not list(out.glob("run_*.pause.json"))


def test_max_gpu_temp_is_not_enforced_for_a_remote_server_and_says_so(tmp_path, capsys, monkeypatch):
    import llmbench.locality as loc
    monkeypatch.setattr(loc, "server_is_local", lambda url: (False, "far.example"))
    CONTROL.configure(cfg_for(tmp_path, max_gpu_temp=85.0, base_url="http://far.example:8080"), "s8")
    assert CONTROL.guard is None and "not enforced" in capsys.readouterr().out


def test_max_gpu_temp_without_a_sensor_warns_loudly(tmp_path, capsys, monkeypatch):
    import llmbench.telemetry.gpu as gpu_mod
    nogpu = types.SimpleNamespace(ok=False, series=None, stop=lambda: None)
    monkeypatch.setattr(gpu_mod, "open_gpu_monitor", lambda cfg, verbose=False: nogpu)
    CONTROL.configure(cfg_for(tmp_path, max_gpu_temp=85.0), "s9")
    assert CONTROL.guard is None and "NOT enforced" in capsys.readouterr().out


# --------------------------------------------------------------------------- from outside: the command line and the page
def _live_lock(tmp_path):
    child = __import__("subprocess").Popen([sys.executable, "-c", "import time\ntime.sleep(60)", "benchmark"])
    (tmp_path / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None,
                                                      "host": status.hostname(), "argv": []}))
    return child


def test_pause_and_resume_requests_need_a_live_run_and_a_paused_one(tmp_path):
    with pytest.raises(RuntimeError, match="no run is going"):
        pause.request(tmp_path, True)
    child = _live_lock(tmp_path)
    try:
        assert "Pause requested" in pause.request(tmp_path, True) and pause.read_control(tmp_path) is True
        assert "Resume requested" in pause.request(tmp_path, False) and pause.read_control(tmp_path) is False
        with pytest.raises(RuntimeError, match="not paused"):
            pause.request(tmp_path, False)
    finally:
        child.kill()
        child.wait()


def test_cli_pause_and_resume_talk_to_the_run_that_is_going(monkeypatch, tmp_path, server, capsys):
    (tmp_path / "results").mkdir()
    child = _live_lock(tmp_path / "results")
    try:
        code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--resume"], datasets=[])   # a live run that is not paused
        assert isinstance(code, str) and "not paused" in code
        code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--pause"], datasets=[])
        assert code == 0 and pause.read_control(tmp_path / "results") is True
        assert "Pause requested" in capsys.readouterr().out
        code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--resume"], datasets=[])
        assert pause.read_control(tmp_path / "results") is False
    finally:
        child.kill()
        child.wait()


def test_resume_of_a_run_lost_while_paused_says_where_it_stopped_and_tidies_up(monkeypatch, tmp_path, server, capsys):
    code, _, first = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[])
    assert code == 0
    res = tmp_path / "results"
    stamp = first["run"]["id"].replace("benchmark_", "")
    doc = json.loads(status.rows_path(res, stamp).read_text())
    doc["state"] = "partial"
    status.rows_path(res, stamp).write_text(json.dumps(doc))
    pause.pause_path(res, stamp).write_text(json.dumps({"reason": "thermal", "phase": "speed", "model": M2,
                                                        "model_i": 2, "models": 2, "completed_models": [M1]}))
    time.sleep(1.1)
    code, _, second = run_cli(monkeypatch, tmp_path, server, ["--speed-only", "--resume"], datasets=[])
    assert code == 0
    assert "was paused (thermal)" in capsys.readouterr().out
    assert not pause.pause_path(res, stamp).exists()


def test_the_page_pauses_and_resumes_a_run_and_offers_resume_for_a_stopped_one(ui):
    u = ui()
    res = u.dir / "results"
    assert _req(u.port, "POST", "/api/pause", {})[0] == 409                  # no run: says so
    child = _live_lock(res)
    try:
        st, _, d = _req(u.port, "POST", "/api/pause", {})
        assert st == 200 and d["pausing"] is True and pause.read_control(res) is True
        assert _req(u.port, "GET", "/api/log?offset=0")[2]["pause_requested"] is True
        assert _req(u.port, "GET", "/api/state")[2]["run"]["pause_requested"] is True
        assert _req(u.port, "POST", "/api/resume", {})[2]["resuming"] is True and pause.read_control(res) is False
        assert _req(u.port, "POST", "/api/resume", {})[0] == 409             # not paused any more
        (res / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None,
                                                     "host": status.hostname(), "argv": ["--batch-probe"]}))
        assert _req(u.port, "POST", "/api/pause", {})[0] == 409              # a batch probe cannot be paused
    finally:
        child.kill()
        child.wait()
        (res / cli.LOCK_NAME).unlink()
    assert _req(u.port, "GET", "/api/state")[2]["outputs"]["resumable"] is False
    (res / "run_20261112_100000.rows.json").write_text(json.dumps({"rows": [], "state": "interrupted", "stamp": "x"}))
    assert _req(u.port, "GET", "/api/state")[2]["outputs"]["resumable"] is True


def test_the_page_has_the_pause_controls():
    html = (ui_server.SCRIPT_DIR / "llmbench" / "ui" / "settings.html").read_text(encoding="utf-8")
    for needle in ('id="pause"', 'id="resume"', 'id="runresume"', "/api/pause", "/api/resume", "pauseWhy"):
        assert needle in html
    assert "'runresume'" in html                                           # locked like the other Run buttons
