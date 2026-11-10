"""v2.3 step 2: prefill at depth - the measurement against the mock server's prompt cache, its guard, the fixed
CSV columns + history migration, the JSON group, deferral and the Agent speed factor."""
import csv
import json

import pytest

import mock_server
from test_pipeline import run_cli


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


def _speed_run(monkeypatch, tmp_path, server, **env):
    mock_server.STATE.pop("prefixes", None)
    env = {"PREFILL_DEPTH": "1024", "AGENT_PROMPT_TOKENS": "0", **env}
    return run_cli(monkeypatch, tmp_path, server, ["--speed-only"], datasets=[], **env)


def test_prefill_at_depth_measured_with_cache_reuse(monkeypatch, tmp_path, server):
    mock_server.STATE["no_reuse"] = {"Mock-30B-A3B-IQ3_XXS.gguf"}       # this server never reuses the prefix
    try:
        code, _, doc = _speed_run(monkeypatch, tmp_path, server)
    finally:
        mock_server.STATE.pop("no_reuse", None)
    assert code == 0
    ok, nore = doc["models"]
    c = ok["context"]
    assert c["prefill_depth_tps"] == pytest.approx(250.0)                  # only the NEW tokens, at depth
    assert c["prefill_depth_tokens"] > 900 and 3000 < c["prefill_depth_new"] < 4500
    assert c["prefill_depth_ttft_s"] == pytest.approx(c["prefill_depth_new"] / 250.0, rel=1e-3)
    # no reuse: a full re-prefill must never be recorded as the depth speed
    assert nore["context"]["prefill_depth_tps"] is None
    log = (tmp_path / "results").glob("benchmark_*.log")
    text = "".join(p.read_text() for p in log)
    assert "prefill at depth not recorded" in text and "reused 15 of" in text
    # the request that measured it carried the cached prompt as its first message, with cache_prompt on
    reqs = [r for r in mock_server.STATE["requests"] if len(r["messages"]) == 3 and r.get("cache_prompt")]
    assert reqs and reqs[0]["messages"][2]["content"].startswith("Tool result (read_file):")
    # CSV has the fixed columns; the timing phase is recorded
    rows = list(csv.DictReader(open(next((tmp_path / "results").glob("benchmark_2*.csv")), newline="")))
    assert "prefill_depth_tps" in rows[0] and float(rows[0]["prefill_depth_tps"]) == pytest.approx(250.0)
    assert "prefill_depth" in ok["timing"]["phases_s"]


def test_prefill_at_depth_skipped_when_it_does_not_fit(monkeypatch, tmp_path, server):
    code, _, doc = _speed_run(monkeypatch, tmp_path, server, PREFILL_DEPTH="65536")    # mock context is 8k
    assert code == 0 and all(m["context"]["prefill_depth_tps"] is None for m in doc["models"])
    text = "".join(p.read_text() for p in (tmp_path / "results").glob("benchmark_*.log"))
    assert "prefill at depth: skipped (64k + 4k does not fit the loaded 8k context)" in text


def test_depth_prompt_reused_from_the_prefill_test(monkeypatch, tmp_path, server):
    """When PREFILL_DEPTH is one of PREFILL_CTX_TESTS its cold prompt is sent once, with cache_prompt on, last."""
    code, _, doc = _speed_run(monkeypatch, tmp_path, server, PREFILL_CTX_TESTS="1024")
    assert code == 0
    # PREFILL_CTX_TESTS snaps to fixed slots; 1024 is not one, so nothing is measured there and the depth test sends
    # its own cold prompt. With a real slot (8k) the prompt would be shared:
    from llmbench.config import load_config
    from llmbench.cli import defer_lengths
    monkeypatch.setenv("PREFILL_CTX_TESTS", "8192,65536")
    monkeypatch.setenv("LONGCTX_LENGTHS", "65536,131072")
    monkeypatch.setenv("PREFILL_DEPTH", "65536")
    cfg = load_config()
    assert defer_lengths(cfg) == ()                     # 64k is measured by the depth test's cold prompt instead
    monkeypatch.setenv("PREFILL_DEPTH", "0")
    assert defer_lengths(load_config()) == (65536,)
    assert doc["models"][0]["context"]["prefill_depth_tps"] == pytest.approx(250.0)


def test_history_csv_migrates_to_the_v23_columns(tmp_path):
    from llmbench.reporting.csv_out import write_csv
    from llmbench.scoring import FIELDS, blank
    old_fields = [f for f in FIELDS if not f.startswith(("prefill_depth", "agent_composite", "agent_longctx",
                                                         "agent_ctx", "agent_borderline", "agent_basis"))]
    p = tmp_path / "benchmark_history_v2.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"model": "Old-9B-Q4_K_M", "status": "ok", "avg_tps": "99.5"})
    r = blank("New-9B-Q4_K_M")
    r.update(prefill_depth_tps=1234.5, agent_ctx_factor=0.9)
    write_csv([r], p, append=True)
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    assert [x["model"] for x in rows] == ["Old-9B-Q4_K_M", "New-9B-Q4_K_M"]
    assert rows[0]["avg_tps"] == "99.5" and rows[0]["prefill_depth_tps"] == ""
    assert rows[1]["prefill_depth_tps"] == "1234.5" and rows[1]["agent_ctx_factor"] == "0.9"
    assert (tmp_path / "benchmark_history_v2.csv.bak").exists()


def test_depth_fields_round_trip_through_json(tmp_path):
    from llmbench.config import load_config
    from llmbench.reload import rows_from_json
    from llmbench.reporting.json_out import _clean, build_report
    from llmbench.scoring import blank, finalize
    cfg = load_config()
    r = blank("M-9B-Q4_K_M")
    r.update(tech_tps=50.0, logic_tps=50.0, prefill_depth_tps=900.0, prefill_depth_ttft_s=4.5,
             prefill_depth_tokens=65000, prefill_depth_new=4100)
    finalize([r], cfg)
    doc = json.loads(json.dumps(_clean(build_report([r], cfg, {"stamp": "x"}))))
    assert doc["models"][0]["context"]["prefill_depth_tps"] == 900.0 and not doc["models"][0]["other"]
    back = rows_from_json(doc)[0]
    assert back["prefill_depth_ttft_s"] == 4.5 and back["agent_ctx_factor"] is None


def test_depth_in_the_agent_speed_factor():
    from llmbench.config import load_config
    from llmbench.scoring import speed_factor
    cfg = load_config()
    slow = {"agent_ttft_s": 10.0, "prefill_depth_tps": 500.0, "avg_tps": 100.0}
    fast = {"agent_ttft_s": 10.0, "prefill_depth_tps": 5000.0, "avg_tps": 100.0}
    assert speed_factor(fast, cfg) > speed_factor(slow, cfg)


# --------------------------------------------------------------------------- PREFILL_DEPTH=auto (default) and load hints
def test_auto_depth_fits_each_model(monkeypatch):
    from types import SimpleNamespace
    from llmbench.config import _prefill_depth, depth_for
    assert _prefill_depth("auto") == (65536, True) and _prefill_depth(None) == (65536, True)
    assert _prefill_depth("32k") == (32768, False) and _prefill_depth("0") == (0, False)
    with pytest.raises(SystemExit):
        _prefill_depth("deep")
    auto = SimpleNamespace(prefill_depth=65536, prefill_depth_auto=True)
    assert depth_for(auto, 65536) == 32768                # a 64k model: 64k + 4k does not fit, 32k does
    assert depth_for(auto, 92160) == depth_for(auto, 131072) == depth_for(auto, 262144) == 65536   # capped at 64k
    assert depth_for(auto, 16384) == 8192 and depth_for(auto, 8192) == 0
    assert depth_for(auto, None) == 65536                 # unknown context: try the cap
    fixed = SimpleNamespace(prefill_depth=65536, prefill_depth_auto=False)
    assert depth_for(fixed, 65536) == 0 and depth_for(fixed, 131072) == 65536
    off = SimpleNamespace(prefill_depth=0, prefill_depth_auto=False)
    assert depth_for(off, 131072) == 0


def test_auto_depth_on_the_mock_server(monkeypatch, tmp_path, server):
    old = mock_server.CTX["n_ctx"]
    mock_server.CTX["n_ctx"] = 20000                       # 16k + 4k does not fit, 8k does
    try:
        code, _, doc = _speed_run(monkeypatch, tmp_path, server, PREFILL_DEPTH="auto")
    finally:
        mock_server.CTX["n_ctx"] = old
    c = doc["models"][0]["context"]
    assert code == 0 and c["prefill_depth_tps"] and 7000 < c["prefill_depth_tokens"] < 8192


def test_agent_score_compares_depth_only_at_one_depth():
    """Prefill is slower deeper in: a model measured at 32k must not beat one measured at 64k on that alone."""
    from llmbench.config import load_config
    from llmbench.scoring import blank, finalize, reference_depth, speed_factor
    cfg = load_config()

    def row(name, depth, tps, ctx):
        r = blank(name)
        r.update(tech_tps=100.0, logic_tps=100.0, vram_gb=12.0, vram_total_gb=16.0, ctx_size=ctx, agent_ttft_s=5.0,
                 prefill_depth_tps=tps, prefill_depth_tokens=int(depth * 0.97), prefill_depth_new=4100)
        return r
    a, b, c = row("A-9B-Q4_K_M", 65536, 1500, 131072), row("B-9B-Q4_K_M", 65536, 1400, 131072), \
        row("C-9B-Q4_K_M", 32768, 3000, 92160)
    assert reference_depth([a, b, c]) == 65536 and reference_depth([c]) == 32768
    finalize([a, b, c], cfg)
    assert c["score_agent"] == pytest.approx(c["agent_quality"] * (1 - 0.3 + 0.3 * speed_factor(c, cfg, use_depth=False)))
    assert "compared at 64k" in c["agent_basis"] and "compared at" not in (a["agent_basis"] or "")


def test_load_failure_hint(tmp_path):
    from types import SimpleNamespace
    from llmbench import loadfail

    class Resp:
        status_code, headers = 200, {"Content-Type": "text/plain"}
        text = ("[INFO] starting llama-server\nggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216 MiB "
                "on device 0: cudaMalloc failed: out of memory\n[INFO] upstream command exited prematurely\n")

    cfg = SimpleNamespace(base_url="http://x", gguf_path_map=(), gguf_search_dirs=(str(tmp_path),))
    s = SimpleNamespace(get=lambda url, **kw: Resp())
    err = 'HTTP 500: {"error":{"message":"unspecific error: upstream command exited prematurely"}}'
    h = loadfail.hint(cfg, s, "II-Medical-8B-1706.Q8_0.gguf", err)
    assert "--fit on" in h and "lower -c" in h and "out of memory" in h
    assert loadfail.describe(cfg, s, "II-Medical-8B-1706.Q8_0.gguf", err)[0] == "Failed to load (out of memory)"
    assert loadfail.hint(cfg, s, "m", "HTTP 400: bad request") is None           # not a load failure

    class Html(Resp):
        headers = {"Content-Type": "text/html"}
    assert loadfail.server_log_lines(cfg, SimpleNamespace(get=lambda url, **kw: Html()), "m") == []
    import gguf_writer
    gguf_writer.dense_model(tmp_path / "Mock-9B-Q4_K_M.gguf", pad_to=50_000)
    f = loadfail.gguf_facts(cfg, "Mock-9B-Q4_K_M.gguf")
    assert f["weights_gb"] > 0 and f.get("n_ctx_train")


def test_load_failure_causes():
    from llmbench import loadfail
    c = lambda *logs: loadfail.classify("HTTP 500: upstream command exited prematurely", logs)[0]
    assert c("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216 MiB: cudaMalloc failed: out of memory") \
        == "out of memory"
    assert c("error: invalid argument: --fit") == "bad launch flag"
    assert c("gguf_init_from_file: failed to open GGUF file '/models/x.gguf'") == "model file not found"
    assert c("llama_model_load: error loading model: unknown model architecture: 'newarch'") == "unsupported model"
    assert c("health check timed out after 120s") == "load timed out"
    assert c().startswith("cause not reported")
    assert loadfail.is_load_failure("HTTP 500: upstream command exited prematurely")
    assert not loadfail.is_load_failure("HTTPConnectionPool: Max retries exceeded (Connection refused)")


def test_failed_load_label_in_the_dashboard():
    import re
    from pathlib import Path
    html = (Path(loadfail_path()).parent / "reporting" / "dashboard.html").read_text()
    assert "startsWith('Failed to load')" in html


def loadfail_path():
    from llmbench import loadfail
    return loadfail.__file__


def test_failed_load_is_labelled_in_log_and_json(monkeypatch, tmp_path, server, capsys):
    mock_server.STATE["load_fail"] = {"Mock-9B-Q4_K_M.gguf"}
    mock_server.STATE["log_text"] = ("[INFO] starting llama-server\nggml_backend_cuda_buffer_type_alloc_buffer: "
                                     "allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory\n")
    try:
        code, _, doc = _speed_run(monkeypatch, tmp_path, server)
    finally:
        mock_server.STATE.pop("load_fail", None)
        mock_server.STATE.pop("log_text", None)
    out = capsys.readouterr().out
    bad = next(m for m in doc["models"] if m["model"] == "Mock-9B-Q4_K_M.gguf")
    assert bad["status"] == "error" and bad["error"].startswith("Failed to load (out of memory): HTTP 500")
    assert "--fit on" in bad["error"] and "cudaMalloc failed" in bad["error"]
    assert "  Failed to load (out of memory)" in out
    assert any(m["status"] == "ok" for m in doc["models"])               # the other model still runs
