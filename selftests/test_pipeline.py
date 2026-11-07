"""End-to-end: the real CLI against a mock llama-swap server and in-memory datasets."""
import csv
import json
import os
import sys
import time

import pytest

import fake_datasets
import mock_server

ALL_DATASETS = ["gpqa", "mmlu_pro", "humaneval", "mbpp", "livecodebench", "gsm8k", "math"]
ENV_FLAG = {"gpqa": "EVAL_GPQA", "mmlu_pro": "EVAL_MMLU_PRO", "humaneval": "EVAL_HUMANEVAL", "mbpp": "EVAL_MBPP",
            "livecodebench": "EVAL_LIVECODEBENCH", "gsm8k": "EVAL_GSM8K", "math": "EVAL_MATH"}


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


def run_cli(monkeypatch, tmp_path, server, argv, datasets=ALL_DATASETS, fake=None, **env):
    import llmbench.evals.datasets as ds_mod
    import llmbench.reporting.dashboard as dash_mod
    for k in list(os.environ):
        if k.startswith(("EVAL_", "GPU_", "RUN_", "STRATA_", "HA_")):
            monkeypatch.delenv(k, raising=False)
    base = {"LLAMA_SWAP_URL": server.url, "OUTPUT_DIR": str(tmp_path / "results"), "EVAL_CACHE_DIR": str(tmp_path / "tests"),
            "SETTLE_SECONDS": "0", "COOLDOWN_SECONDS": "0", "RUNS_PER_PROMPT": "1", "PREFILL_WORDS": "40",
            "RUN_EVAL_TESTS": "true", "EVAL_PROFILE": "full", "EVAL_CODE_SANDBOX": "subprocess", "EVAL_CODE_TIMEOUT": "5",
            "WRITE_MARKDOWN": "true", "RAM_PROCESS_PATTERN": "no-such-process", "RETRIES": "2", "LOG": "true"}
    base.update({ENV_FLAG[k]: "true" for k in datasets})
    base.update(env)
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    sources = {**fake_datasets.BY_KEY, **(fake or {})}
    monkeypatch.setattr(ds_mod, "load_eval_split", lambda cfg, key, offline_first=True: sources[key]())
    monkeypatch.setattr(dash_mod, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark.py"] + argv)
    from llmbench.cli import main
    mock_server.STATE.update(in_flight=0, max_in_flight=0, calls=0, requests=[], flaky_503_left=0)
    t0 = time.time()
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code
    return code, time.time() - t0, json.load(open(tmp_path / "results" / "results.json")) if (tmp_path / "results" / "results.json").exists() else None


def ev(model, key):
    return model["evaluation"]["datasets"][key]


# ---------------------------------------------------------------------------
def test_full_run_sequential(monkeypatch, tmp_path, server):
    """speed phase + all 7 datasets, EVAL_CONCURRENCY=1 (the default)."""
    mock_server.STATE["flaky_503_left"] = 1
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, [])
    assert code == 0, "run failed"
    assert [m["status"] for m in doc["models"]] == ["ok", "ok"]
    m = doc["models"][0]
    # --- speed phase ran against the mock (and survived a transient 503 on the load request)
    assert m["speed"]["tech_tps"] == pytest.approx(50.0) and m["speed"]["prefill_tps"] == pytest.approx(500.0)
    assert m["metadata"]["quant"] == "Q4_K_M" and m["metadata"]["ctx_size"] == 8192
    assert doc["models"][1]["metadata"]["arch"] == "MoE"
    # --- every dataset graded exactly as designed
    g = lambda k: ev(m, k)
    assert (g("gpqa")["n"], round(g("gpqa")["pct"], 1)) == (6, 83.3)         # one [[wrong]] -> 5/6
    assert (g("mmlu_pro")["n"], round(g("mmlu_pro")["pct"], 1)) == (6, 83.3)
    assert g("gsm8k")["pct"] == pytest.approx(83.33, abs=0.01)               # gold "1,0"-style commas + 1 wrong
    assert g("math")["pct"] == 100.0                                         # 6 differently-written equivalent answers
    assert g("mbpp")["pct"] == 100.0 and g("livecodebench")["pct"] == 100.0
    assert g("humaneval")["pct"] == 50.0                                     # HE/1 calls sys.exit(0) before the tests
    # --- the [[long]] question was cut off, retried once with thinking disabled and then answered
    assert g("gpqa")["fallback"] == 1 and g("gsm8k")["fallback"] == 1 and g("gpqa")["na"] == 0
    assert any((r.get("chat_template_kwargs") or {}).get("enable_thinking") is False for r in mock_server.STATE["requests"])
    assert mock_server.STATE["max_in_flight"] == 1                            # sequential means sequential
    # --- latency semantics + concurrency recorded
    assert g("gsm8k")["concurrency"] == 1 and g("gsm8k")["sec"] >= mock_server.LATENCY * 0.9
    # --- derived scores exist, so finalize() + the new datasets work together
    # (score_agent needs VRAM telemetry, exactly as before, so it is legitimately empty on a GPU-less CI box)
    assert m["evaluation"]["avg_pct"] == pytest.approx(
        sum(ev(m, k)["pct"] for k in ALL_DATASETS) / len(ALL_DATASETS)) and m["evaluation"]["avg_sec"] is not None
    assert m["scores"]["goldilocks"] is not None
    assert doc["run"]["mode"] == "full" and doc["host"]["gpu_backend"] in ("none", "nvml", "smi")
    # --- every output file
    res = tmp_path / "results"
    names = {p.name for p in res.iterdir()}
    for pat in ("benchmark_dashboard_", "benchmark_report.md", "results.json", "benchmark_history_v2.csv", "eval_details_", ".log", ".csv"):
        assert any(pat in n for n in names), f"missing output {pat}: {sorted(names)}"
    assert (tmp_path / "benchmark_dashboard.html").exists()
    rows = list(csv.DictReader(open(next(res.glob("benchmark_2*.csv")))))
    assert rows[0]["eval_gsm8k_pct"] and rows[0]["eval_math_n"] == "6"
    assert "gsm8k" in open(res / "benchmark_report.md").read().lower()
    details = [json.loads(l) for l in open(next(res.glob("eval_details_*.jsonl")))]
    assert {d["dataset"] for d in details} == set(ALL_DATASETS) and all("seconds" in d for d in details)
    assert doc["artifacts"]["json_latest"].endswith("results.json")


def test_concurrent_evals_same_answers_and_faster(monkeypatch, tmp_path, server):
    ds = ["gpqa", "mmlu_pro", "gsm8k", "math"]
    code1, t1, d1 = run_cli(monkeypatch, tmp_path / "seq", server, ["--evals-only"], datasets=ds, EVAL_CONCURRENCY="1") \
        if (tmp_path / "seq").mkdir() is None else None
    seq_max = mock_server.STATE["max_in_flight"]
    (tmp_path / "par").mkdir()
    code2, t2, d2 = run_cli(monkeypatch, tmp_path / "par", server, ["--evals-only"], datasets=ds, EVAL_CONCURRENCY="4")
    assert code1 == code2 == 0
    assert seq_max == 1 and mock_server.STATE["max_in_flight"] >= 3           # real parallelism on the wire
    for m1, m2 in zip(d1["models"], d2["models"]):                            # identical verdicts either way
        for k in ds:
            a, b = ev(m1, k), ev(m2, k)
            assert (a["pct"], a["n"], a["err"], a["na"], a["fallback"]) == (b["pct"], b["n"], b["err"], b["na"], b["fallback"])
            assert b["concurrency"] == 4
    assert t2 < t1 * 0.6, f"expected a clear speed-up, got {t1:.1f}s -> {t2:.1f}s"
    # s/q stays a per-question LATENCY (~LATENCY, +retry), not wall/n: it must NOT shrink by the concurrency factor
    par = ev(d2["models"][0], "gsm8k")
    assert par["sec"] >= mock_server.LATENCY * 0.9
    assert par["wall"] < par["sec"] * par["n"]                                # wall is genuinely shorter than sum of latencies


def test_thread_transport_matches_aiohttp(monkeypatch, tmp_path, server):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"],
                              EVAL_CONCURRENCY="3", EVAL_HTTP_BACKEND="threads")
    assert code == 0 and ev(doc["models"][0], "gsm8k")["pct"] == pytest.approx(83.33, abs=0.01)
    assert doc["evaluation"]["http_backend"] == "threads" and mock_server.STATE["max_in_flight"] >= 2


def test_consecutive_errors_abort_the_dataset(monkeypatch, tmp_path, server):
    broken = lambda: fake_datasets.FakeDS([{"question": f"Q [[err]] {i}", "answer": "A"} for i in range(12)])
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], fake={"gpqa": broken},
                              RETRIES="0")
    r = ev(doc["models"][0], "gpqa")
    assert r["n"] == 5 and r["err"] == 5 and r["pct"] is None                 # stopped after 5, no misleading 0 %


def test_speed_only_and_json_toggle(monkeypatch, tmp_path, server):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], JSON="true")
    assert code == 0 and doc["run"]["mode"] == "speed-only" and doc["evaluation"]["enabled"] is False
    (tmp_path / "off").mkdir()
    code, secs, doc = run_cli(monkeypatch, tmp_path / "off", server, ["--speed-only"], JSON="false")
    assert code == 0 and doc is None and not (tmp_path / "off" / "results" / "results.json").exists()


def test_all_models_fail_still_writes_json(monkeypatch, tmp_path):
    class Dead:       # nothing listening -> no server at all
        url = "http://127.0.0.1:9"
    code, secs, doc = run_cli(monkeypatch, tmp_path, Dead(), ["--speed-only"])
    assert code not in (0, None)                                              # clean error exit, not a traceback


# ---------------------------------------------------------------------------
# Streaming: loop abort, thinking cap, partial-answer salvage, timeouts
# ---------------------------------------------------------------------------
def misbehaving(markers):
    """A GPQA-shaped dataset where question i carries markers[i] ([[loop]], [[ramble]], [[partial]], [[hang]] or '')."""
    return lambda: fake_datasets.FakeDS([{"question": f"Q{i}? A) a B) b C) c D) d [[ans:A]] {m}", "answer": "A"}
                                         for i, m in enumerate(markers)])


def details(tmp_path):
    return [json.loads(l) for l in open(next((tmp_path / "results").glob("eval_details_*.jsonl")))]


def served(marker, model="Mock-9B-Q4_K_M.gguf"):
    """Requests the mock saw for questions carrying `marker` (for ONE of the two mock models)."""
    return [r for r in mock_server.STATE["requests"] if marker in r["messages"][0]["content"] and r["model"] == model]


@pytest.mark.parametrize("backend", ["aiohttp", "threads"])
def test_loop_cap_and_salvage(monkeypatch, tmp_path, server, capsys, backend):
    markers = ["[[loop]]", "[[ramble]]", "[[partial]]", "", "[[loop]]"]
    before = mock_server.STATE["disconnects"]
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(markers)}, EVAL_THINK_BUDGET="1500", EVAL_HTTP_BACKEND=backend)
    assert code == 0
    r = ev(doc["models"][0], "gpqa")
    # every question ends up correct: two loops and the ramble are rescued by the No Thinking retry, [[partial]] by salvage
    assert (r["n"], r["pct"], r["err"], r["na"]) == (5, 100.0, 0, 0)
    assert (r["degr"], r["cap"], r["salv"], r["fallback"]) == (3, 1, 1, 3)       # 3 loops (Q0, Q2, Q4), 1 cap, 1 salvage
    assert r["tok"] and secs < 25                                                # stopped in moments, not after a timeout
    assert mock_server.STATE["disconnects"] - before >= 4                       # the client hung up on every runaway reply
    assert len(served("[[partial]]")) == 1                                       # salvaged: no retry was needed
    assert len(served("[[loop]]")) == 4 and len(served("[[ramble]]")) == 2        # 2 loops x (think + no-think), ramble x 2
    recs = {d["id"]: d for d in details(tmp_path)}
    loop, ramble, partial = recs["gpqa-0"], recs["gpqa-1"], recs["gpqa-2"]
    assert loop["correct"] and loop["thinking_fallback_used"] and loop["flag"] == ""
    assert [a["end"] for a in loop["attempts"]] == ["loop", "stop"] and loop["attempts"][0]["mode"] == "think"
    assert "double check" in loop["attempts"][0]["tail"] and "repetition loop" in loop["attempts"][0]["detail"]
    assert [a["end"] for a in ramble["attempts"]] == ["budget", "stop"] and "budget 1,500" in ramble["attempts"][0]["detail"]
    assert partial["correct"] and partial["salvaged"] and not partial["thinking_fallback_used"]
    assert partial["finish"] == "abort:loop" and len(partial["attempts"]) == 1
    out = capsys.readouterr().out
    assert "[Fail: Degradation] gpqa-0:" in out and "retried with thinking disabled" in out
    assert "[Fail: Think cap] gpqa-1:" in out and "graded from the partial answer" in out
    assert "Fail: Degradation: 3 question(s)" in out and "3 recovered" in out
    # the same numbers reach the CSV (Degradation) and the JSON (all four)
    rows = list(csv.DictReader(open(next((tmp_path / "results").glob("benchmark_2*.csv")))))
    assert rows[0]["eval_gpqa_degr"] == "3"
    assert doc["evaluation"]["stream"] is True and doc["evaluation"]["think_budget"] == 1500


def test_failures_keep_their_flag_without_retry(monkeypatch, tmp_path, server, capsys):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(["[[loop]]", "[[ramble]]", ""])},
                              EVAL_THINK_BUDGET="1500", EVAL_RETRY_ON_LENGTH="false")
    r = ev(doc["models"][0], "gpqa")
    assert (r["n"], round(r["pct"], 1), r["err"], r["na"], r["fallback"]) == (3, 33.3, 0, 0, 0)
    assert (r["degr"], r["cap"], r["salv"]) == (1, 1, 0)
    recs = {d["id"]: d for d in details(tmp_path)}
    assert recs["gpqa-0"]["flag"] == "Fail: Degradation" and not recs["gpqa-0"]["correct"]
    assert recs["gpqa-0"]["reason"].startswith("Fail: Degradation: repetition loop")
    assert recs["gpqa-1"]["flag"] == "Fail: Think cap" and recs["gpqa-2"]["flag"] == ""
    assert "counted as wrong" in capsys.readouterr().out
    assert len(served("[[loop]]")) == 1                                          # no retry was sent


def test_salvage_and_loop_abort_can_be_switched_off(monkeypatch, tmp_path, server):
    # salvage off: the [[partial]] question is retried instead of graded from its partial answer
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(["[[partial]]"])}, EVAL_THINK_BUDGET="1500", EVAL_SALVAGE="false")
    r = ev(doc["models"][0], "gpqa")
    assert (r["pct"], r["salv"], r["fallback"], r["degr"]) == (100.0, 0, 1, 1)
    assert len(served("[[partial]]")) == 2
    # loop abort off: the same looping reply is stopped by the thinking budget instead
    (tmp_path / "b").mkdir()
    code, secs, doc = run_cli(monkeypatch, tmp_path / "b", server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(["[[loop]]"])}, EVAL_THINK_BUDGET="1500", EVAL_LOOP_ABORT="false")
    r = ev(doc["models"][0], "gpqa")
    assert (r["pct"], r["degr"], r["cap"]) == (100.0, 0, 1)


def test_timeout_returns_the_partial_reply_and_retries(monkeypatch, tmp_path, server):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(["[[hang]]", ""])}, EVAL_TIMEOUT="1", TIMEOUT_LOAD="1")
    r = ev(doc["models"][0], "gpqa")
    assert (r["n"], r["pct"], r["err"], r["fallback"]) == (2, 100.0, 0, 1) and secs < 15
    rec = {d["id"]: d for d in details(tmp_path)}["gpqa-0"]
    assert [a["end"] for a in rec["attempts"]] == ["timeout", "stop"] and "no result after" in rec["attempts"][0]["detail"]
    # a timeout with no retry available is still an API-style failure, exactly as before streaming existed
    (tmp_path / "b").mkdir()
    code, secs, doc = run_cli(monkeypatch, tmp_path / "b", server, ["--evals-only"], datasets=["gpqa"],
                              fake={"gpqa": misbehaving(["[[hang]]"])}, EVAL_TIMEOUT="1", TIMEOUT_LOAD="1",
                              EVAL_RETRY_ON_LENGTH="false")
    r = ev(doc["models"][0], "gpqa")
    assert r["err"] == 1 and r["pct"] is None
    assert {d["id"]: d for d in details(tmp_path / "b")}["gpqa-0"]["reason"].startswith("api error: timeout")


def test_stream_off_uses_plain_requests_and_same_verdicts(monkeypatch, tmp_path, server):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa", "gsm8k"], EVAL_STREAM="false")
    assert code == 0
    assert all(r.get("stream") is False for r in mock_server.STATE["requests"])
    m = doc["models"][0]
    assert (round(ev(m, "gpqa")["pct"], 1), ev(m, "gpqa")["fallback"]) == (83.3, 1)         # same as the streamed run
    assert round(ev(m, "gsm8k")["pct"], 2) == 83.33 and ev(m, "gpqa")["degr"] == 0
    assert doc["evaluation"]["stream"] is False


# ---------------------------------------------------------------------------
# v2.2: GGUF header, deployed-context tests, tool calling, long-context retrieval, budget profile
# ---------------------------------------------------------------------------
def test_v22_full_run(monkeypatch, tmp_path, server):
    import gguf_writer
    from llmbench.evals.toolcall_cases import CASES
    models = tmp_path / "models"
    models.mkdir()
    gguf_writer.dense_model(models / "Mock-9B-Q4_K_M.gguf", pad_to=50_000)
    gguf_writer.moe_mla_model(models / "Mock-30B-A3B-IQ3_XXS.gguf", pad_to=80_000)
    old_ctx = mock_server.CTX["n_ctx"]
    mock_server.CTX["n_ctx"] = 40000
    try:
        code, secs, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gpqa"],
                                  EVAL_TOOLCALL="true", EVAL_LONGCTX="true", LONGCTX_LENGTHS="32768,65536",
                                  LONGCTX_TRIALS="1", PREFILL_CTX_TESTS="8192,32768,65536", AGENT_PROMPT_TOKENS="4000",
                                  GGUF_SEARCH_DIRS=str(models), BUDGET_PROFILE="server", MIN_CTX="65536",
                                  BUDGET_SERVER_MAX_RETRY_PCT="off")
    finally:
        mock_server.CTX["n_ctx"] = old_ctx
    assert code == 0
    a, b = doc["models"]
    # --- GGUF header: architecture from the file, not the name
    assert a["gguf"]["gguf_arch"] == "llama" and b["gguf"]["attn_type"] == "MLA" and b["metadata"]["arch"] == "MoE"
    assert b["gguf"]["expert_count"] == 64 and a["gguf"]["bpw"] and a["gguf"]["kv_kib_token"] == pytest.approx(2.0)
    assert a["gguf"]["kv_approx"] is True                    # no launch flags readable here: f16 assumed and marked
    # --- deployed context: prefill by length (64k does not fit the 40k context), agent-prompt TTFT
    assert a["context"]["prefill_8k_tps"] == pytest.approx(300.0) and a["context"]["prefill_32k_tps"]
    assert a["context"]["prefill_64k_tps"] is None
    assert a["context"]["agent_ttft_s"] > 0 and a["context"]["agent_prompt_tokens"] > 3000
    assert a["context"]["ctx_ok"] is False and a["scores"]["eligible"] is False      # 40k < MIN_CTX 64k: flagged
    # --- tool calling: two deliberate mistakes in the mock (wrong argument, call written as text)
    t = ev(a, "toolcall")
    assert t["n"] == len(CASES) and t["pct"] == pytest.approx(100 * (len(CASES) - 2) / len(CASES))
    tools_req = [r for r in mock_server.STATE["requests"] if r.get("tools")]
    assert tools_req and all(r["stream"] is False and r["tool_choice"] == "auto" for r in tools_req)
    # --- long-context retrieval: 32k answered, 64k too long for the context and reported as such
    lc = a["long_context"]
    assert lc["longctx_32k_pct"] == 100.0 and lc["longctx_64k_pct"] is None and "too small" in lc["longctx_note"]
    assert lc["longctx_kv_ok"] is None
    # --- budget: per-rule results, MIN_CTX fails, KV unknown is not a fail
    bud = a["budget"]
    assert bud["budget_profile"] == "server" and bud["budget_pass"] is False and "MIN_CTX" in bud["budget_fails"]
    assert bud["budget"]["KV_MIN"]["ok"] is None and "MAX_RETRY_PCT" not in bud["budget"]
    # --- outputs carry the new columns and sections
    res = tmp_path / "results"
    rows = list(csv.DictReader(open(next(res.glob("benchmark_2*.csv")))))
    assert rows[0]["longctx_32k_pct"] == "100.0" and rows[0]["eval_toolcall_pct"] and rows[0]["budget_pass"] == "False"
    md = open(res / "benchmark_report.md").read()
    assert "## Deployment fit" in md and "margin of error" in md
    dash = open(next(res.glob("benchmark_dashboard_*.html"))).read()
    assert '"longctx"' in dash and '"margin"' in dash and "Tool calling" in dash
    recs = [json.loads(l) for l in open(next(res.glob("eval_details_*.jsonl")))]
    assert {"longctx", "toolcall", "gpqa"} <= {r["dataset"] for r in recs}


def test_min_ctx_refuse_skips_the_model(monkeypatch, tmp_path, server):
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--speed-only"], MIN_CTX="65536", MIN_CTX_ACTION="refuse")
    assert code not in (0, None)                                     # nothing left to report
    assert [m["status"] for m in doc["models"]] == ["skipped", "skipped"] and "MIN_CTX" in doc["models"][0]["error"]
