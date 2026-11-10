"""Result-cache modes, the sanity check, the cache-aware estimate and --compare (combining runs)."""
import copy
import json

import pytest

import mock_server
from llmbench.config import load_config
from llmbench.evals.cache import SanityCheck, same_answer
from test_pipeline import ev, run_cli
from test_v21 import M1, M2, server  # noqa: F401  (fixture)


def _questions():
    return [r for r in mock_server.STATE["requests"] if r["messages"][0]["content"] != "hi"]


# ---------------------------------------------------------------- sanity check logic ----------------------------
def test_sanity_rate_halves_on_matches_and_resets_on_a_difference():
    s = SanityCheck(80, adaptive=True, seed=1)
    assert s.current_pct("m") == 80
    s.result("m", True)
    s.result("m", True)
    assert s.current_pct("m") == 20
    s.result("m", False)
    assert s.current_pct("m") == 80 and s.stats["m"] == {"checked": 3, "different": 1}
    seq, t = [], SanityCheck(20, adaptive=True, seed=1)        # whole percents, 0 below 2%
    for _ in range(5):
        t.result("m", True)
        seq.append(t.current_pct("m"))
    assert seq == [10, 5, 2, 0, 0] and not any(t.pick("m") for _ in range(200))
    t.result("m", False)
    assert t.current_pct("m") == 20
    fixed = SanityCheck(80, adaptive=False, seed=1)
    fixed.result("m", True)
    assert fixed.current_pct("m") == 80
    assert not any(SanityCheck(0, seed=2).pick("m") for _ in range(100))
    assert all(SanityCheck(100, adaptive=False, seed=2).pick("m") for _ in range(100))


def test_same_answer_rules():
    assert same_answer("qa", {"correct": True, "answer": "B"}, {"correct": True, "answer": " b"})
    assert not same_answer("qa", {"correct": False, "answer": "C"}, {"correct": False, "answer": "D"})
    assert same_answer("code", {"correct": True, "answer": "def f(): return 1"}, {"correct": True, "answer": "x"})
    assert not same_answer("code", {"correct": False}, {"correct": True})


# ---------------------------------------------------------------- cache modes in a run --------------------------
def test_refresh_asks_everything_again_and_keeps_the_new_answers(monkeypatch, tmp_path, server):
    run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"])
    first = len(_questions())
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_RESULT_CACHE="refresh")
    assert code == 0 and len(_questions()) == first and ev(d["models"][0], "gsm8k")["cached"] == 0
    assert d["evaluation"]["result_cache_mode"] == "refresh"
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"])
    assert len(_questions()) == 0 and ev(d["models"][0], "gsm8k")["cached"] == ev(d["models"][0], "gsm8k")["n"]


def test_bad_cache_mode_is_refused(monkeypatch):
    monkeypatch.setenv("EVAL_RESULT_CACHE", "maybe")
    with pytest.raises(SystemExit):
        load_config()


def test_sanity_check_re_asks_and_scores_the_new_answer(monkeypatch, tmp_path, server, capsys):
    run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"])
    capsys.readouterr()
    # 100%, not adaptive: every cached answer is re-asked; the mock is deterministic, so all match
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"],
                         EVAL_SANITY_PCT="100", EVAL_SANITY_ADAPTIVE="false")
    out = capsys.readouterr().out
    m = d["models"][0]["evaluation"]
    assert code == 0 and m["sanity_checked"] == 12 and m["sanity_different"] == 0 and m["cached"] == 0
    assert "sanity check: 100% of cached answers" in out and "24 cached answer(s) re-asked, 0 gave a different" in out
    # the model now answers differently (weak switch): the differences are found, and the NEW answers are scored
    mock_server.STATE["weak"] = {M1}
    try:
        code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"],
                             EVAL_SANITY_PCT="100", EVAL_SANITY_ADAPTIVE="false")
    finally:
        mock_server.STATE["weak"] = set()
    out = capsys.readouterr().out
    m1 = next(x for x in d["models"] if x["model"] == M1)
    assert m1["evaluation"]["sanity_different"] > 0 and ev(m1, "gsm8k")["pct"] == 0
    assert "[sanity]" in out and "the new answer is scored" in out
    # adaptive (default): a model that keeps matching is checked only a few times
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"],
                         EVAL_SANITY_PCT="100")
    m2 = next(x for x in d["models"] if x["model"] == M2)["evaluation"]
    assert 1 <= m2["sanity_checked"] < 12 and m2["cached"] == 12 - m2["sanity_checked"]


def test_estimate_counts_cached_answers(monkeypatch, tmp_path, server, capsys):
    run_cli(monkeypatch, tmp_path, server, [], datasets=["gpqa"], EVAL_PROFILE="standard")
    capsys.readouterr()
    run_cli(monkeypatch, tmp_path, server, ["--estimate"], datasets=["gpqa"], EVAL_PROFILE="standard")
    out = capsys.readouterr().out
    assert "Result cache: on" in out and "cached = answers expected to be reused" in out
    row = next(line for line in out.splitlines() if line.strip().startswith(M1))
    assert "6/96" in row          # the 6 fake GPQA questions are cached; a real standard run asks 96
    run_cli(monkeypatch, tmp_path, server, ["--estimate"], datasets=["gpqa"], EVAL_PROFILE="standard",
            EVAL_RESULT_CACHE="refresh")
    assert "FULL re-run" in capsys.readouterr().out


# ---------------------------------------------------------------- --compare ------------------------------------
def _write_run(tmp_path, cfg, rows, stamp, backend="llama-swap", **cfg_over):
    from llmbench.reporting.json_out import write_json
    from llmbench.scoring import finalize
    cfg.backend = backend
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    finalize(rows, cfg)
    write_json(rows, cfg, {"stamp": stamp, "status": "complete", "hardware": "Test host", "cpu_source": "demo"}, stamp)
    return cfg.output_dir / f"benchmark_{stamp}.json"


@pytest.fixture
def two_runs(monkeypatch, tmp_path):
    from llmbench import compare as cmp
    from llmbench.diagnostics import demo_results
    from llmbench.scoring import finalize
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cmp, "SCRIPT_DIR", tmp_path)
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = demo_results(cfg)
    a_rows = rows[:4]
    b_rows = [rows[4], rows[5]]
    b_rows[0]["model"] = "Qwen3.8-Next-Flash-Q4_K_M.gguf"          # the Strata model
    b_rows[1]["model"] = rows[0]["model"]                         # the same model tested on both backends
    expected = copy.deepcopy(a_rows + b_rows)                     # what ONE run with all six models would score
    for r, src in zip(expected, ["llama-swap 09 Nov 00:45"] * 4 + ["Strata 10 Nov 09:00"] * 2):
        if r["model"] == rows[0]["model"]:
            r["model"] = f"{r['model']} [{src}]"
    a = _write_run(tmp_path, cfg, a_rows, "20261109_004502")
    b = _write_run(tmp_path, cfg, b_rows, "20261110_090000", backend="strata")
    cfg.backend = "llama-swap"
    finalize(expected, cfg)
    return cmp, cfg, a, b, {r["model"]: r for r in expected}


def test_compare_combines_runs_and_rescores_together(two_runs, capsys):
    cmp, cfg, a, b, expected = two_runs
    out, rows = cmp.compare_runs(load_config(), [str(a), str(b)])
    assert out.name.startswith("benchmark_compare_") and (out.parent.parent / "benchmark_compare.html").exists()
    assert {r["source"] for r in rows} == {"llama-swap 09 Nov 00:45", "Strata 10 Nov 09:00"}
    names = [r["model"] for r in rows]
    assert "Qwen3.8-Next-Flash-Q4_K_M.gguf" in names and sum("[" in n for n in names) == 2   # duplicate labelled
    # re-scored over the combined pool: the same scores one run with all six models gives
    for r in rows:
        for k in ("agent_quality", "score_agent", "score_sweet", "eval_avg_pct", "eligible"):
            assert r.get(k) == pytest.approx(expected[r["model"]].get(k)), (r["model"], k)
    html = out.read_text()
    assert '"compare"' in html and "Strata 10 Nov 09:00" in html
    # default selection: the newest llama-swap run + the newest Strata run, llama-swap first
    _, rows2 = cmp.compare_runs(load_config(), None)
    assert rows2[0]["source"].startswith("llama-swap")


def test_compare_refuses_different_eval_settings_unless_forced(two_runs, capsys):
    cmp, cfg, a, b, _ = two_runs
    doc = json.loads(b.read_text())
    doc["evaluation"]["seed"] = 7
    b.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as e:
        cmp.compare_runs(load_config(), [str(a), str(b)])
    assert "refused" in str(e.value) and "eval seed" in capsys.readouterr().out
    out, _ = cmp.compare_runs(load_config(), [str(a), str(b)], force=True)
    assert "settings differ: eval seed" in out.read_text()


def test_compare_needs_two_backends_without_files(monkeypatch, tmp_path):
    from llmbench import compare as cmp
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    with pytest.raises(SystemExit) as e:
        cmp.compare_runs(load_config(), None)
    assert "only one backend" in str(e.value)


# ---------------------------------------------------------------- per-model sampling ----------------------------
def test_temperature_setting_parsing(monkeypatch):
    from llmbench.config import sampling_label, temperature_label, uses_server_sampling
    monkeypatch.setenv("EVAL_TEMPERATURE", "server")
    cfg = load_config()
    assert cfg.eval_temperature is None and uses_server_sampling(cfg, "anything")
    assert "EVAL_TEMPERATURE=server" in temperature_label(cfg)
    monkeypatch.setenv("EVAL_TEMPERATURE", "0")
    monkeypatch.setenv("EVAL_SERVER_SAMPLING_MODELS", "gemma|wicked")
    cfg = load_config()
    assert uses_server_sampling(cfg, "gemma-4-26B.gguf") and uses_server_sampling(cfg, "Wicked-Stuff.gguf")
    assert not uses_server_sampling(cfg, "Qwen3.6.gguf")
    argv = ["llama-server", "--model", "/m.gguf", "--temp", "1.0", "--top-k", "64", "--top-p=0.95", "-c", "8192"]
    assert sampling_label(cfg, "gemma-4.gguf", argv) == "server: temp 1.0, top-k 64, top-p 0.95"
    assert sampling_label(cfg, "gemma-4.gguf", ["llama-server", "-c", "8192"]) == "server: llama-server defaults"
    assert sampling_label(cfg, "Qwen3.6.gguf", argv) == "temp 0"
    for bad in (("EVAL_TEMPERATURE", "warm"), ("EVAL_SERVER_SAMPLING_MODELS", "gemma(")):
        monkeypatch.setenv(*bad)
        with pytest.raises(SystemExit):
            load_config()
        monkeypatch.setenv("EVAL_TEMPERATURE", "0")
        monkeypatch.setenv("EVAL_SERVER_SAMPLING_MODELS", "")


def test_server_sampling_omits_temperature_and_keeps_other_models_cached(monkeypatch, tmp_path, server):
    # v2.3: EVAL_TEMPERATURE defaults to server; this test pins temperature 0 and switches one model to server
    run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_TEMPERATURE="0")
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"],
                         EVAL_SERVER_SAMPLING_MODELS=M1, EVAL_TEMPERATURE="0")
    asked = _questions()
    assert code == 0 and asked and all(r["model"] == M1 for r in asked)          # only M1 was asked again
    assert all("temperature" not in r and r.get("seed") == 42 for r in asked)
    by = {m["model"]: m for m in d["models"]}
    assert by[M1]["evaluation"]["sampling"].startswith("server") and by[M2]["evaluation"]["sampling"] == "temp 0"
    assert ev(by[M2], "gsm8k")["cached"] == ev(by[M2], "gsm8k")["n"] and ev(by[M1], "gsm8k")["cached"] == 0
    assert d["evaluation"]["server_sampling_models"] == M1
    code, _, d = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_TEMPERATURE="server")
    assert d["evaluation"]["temperature"] == "server"
    assert all("temperature" not in r for r in _questions())
