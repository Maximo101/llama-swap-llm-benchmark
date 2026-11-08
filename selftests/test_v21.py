"""v2.1: faster full runs without less trustworthy rankings.

Every speed-up here is checked for the property that matters: a weaker model must never rank higher because of it.
"""
import csv
import glob
import json
import types

import pytest

import fake_datasets
import mock_server
from llmbench.config import load_config
from llmbench.evals import cache as cachemod
from llmbench.evals import datasets as dsmod
from llmbench.evals import funnel, sandbox
from llmbench.evals.runner import inflight_cap
from llmbench.scoring import blank, finalize
from llmbench.telemetry.host import parse_launch_args
from test_pipeline import details, ev, run_cli, served

M1, M2 = "Mock-9B-Q4_K_M.gguf", "Mock-30B-A3B-IQ3_XXS.gguf"


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    for k in list(__import__("os").environ):
        if k.startswith(("EVAL_", "GPU_", "LLAMA_", "RUN_", "BUDGET_", "MIN_CTX")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    c = load_config()
    c.eval_code_sandbox = "subprocess"
    return c


@pytest.fixture
def ctx_slots():
    """Temporarily give the mock server more context / slots."""
    old = dict(mock_server.CTX)

    def set_(**kw):
        mock_server.CTX.update(kw)
    yield set_
    mock_server.CTX.clear()
    mock_server.CTX.update(old)


def _row(name, pcts, n=96, err=None, tps=60.0, **kw):
    r = blank(name)
    r.update(tech_tps=tps, logic_tps=tps, vram_gb=10.0, vram_total_gb=16.0, offload="none", prefill_tps=2000.0, **kw)
    for k, p in pcts.items():
        e = (err or {}).get(k, 0)
        r.update({f"eval_{k}_pct": p, f"eval_{k}_n": n, f"eval_{k}_err": e, f"eval_{k}_rows": n,
                  f"eval_{k}_sec": 20.0, f"eval_{k}_tok": 1200, f"eval_{k}_ptok": 300, f"eval_{k}_grade_s": 0.1})
    return r


# --------------------------------------------------------------------------- B1: complete dataset sets only
def test_failed_dataset_does_not_lift_a_weaker_model(cfg):
    """v2.0.0 averaged only the datasets that produced a score: a model weaker on every dataset it answered, whose
    hardest dataset failed completely, ranked first (Agent 92.8 vs 88.0)."""
    strong = _row("Strong-14B-Q4_K_M", {"gpqa": 60, "mmlu_pro": 70, "gsm8k": 92})
    weak = _row("Weak-14B-Q4_K_M", {"gpqa": None, "mmlu_pro": 65, "gsm8k": 90}, err={"gpqa": 96})
    finalize([strong, weak], cfg)
    assert strong["score_agent"] is not None and weak["score_agent"] is None
    assert weak["eval_avg_pct"] is None and "GPQA 0/96 answered" in weak["eval_incomplete"]
    assert weak["score_note"].startswith("not ranked: incomplete evals") and weak["eligible"] is False
    # a dataset aborted after a handful of questions is incomplete too
    part = _row("Part-14B-Q4_K_M", {"gpqa": 50, "mmlu_pro": 80, "gsm8k": 95})
    part.update(eval_gpqa_n=8, eval_gpqa_err=5)
    finalize([strong, part], cfg)
    assert part["score_agent"] is None and "GPQA 3/96" in part["eval_incomplete"]
    # complete results are scored exactly as before
    a, b = _row("A-14B-Q4_K_M", {"gpqa": 60, "mmlu_pro": 70}), _row("B-14B-Q4_K_M", {"gpqa": 50, "mmlu_pro": 66})
    finalize([a, b], cfg)
    assert a["eval_avg_pct"] == 65 and b["eval_avg_pct"] == 58 and a["score_agent"] > b["score_agent"]


# --------------------------------------------------------------------------- nested sampling
@pytest.mark.parametrize("n,key", [(198, "gpqa"), (500, "math"), (1319, "gsm8k"), (12032, "mmlu_pro")])
def test_profiles_are_nested(n, key):
    sizes = [6, 11, 24, 43, 96, 196]
    sets = [set(dsmod.sample_indices(n, k, 42, key)) for k in sizes]
    assert all(len(s) == k for s, k in zip(sets, sizes))
    assert all(a <= b for a, b in zip(sets, sets[1:]))              # v2.0.0: MATH-500 quick was not in standard
    assert dsmod.sample_indices(n, 11, 42, key) == dsmod.sample_indices(n, 11, 42, key)   # deterministic
    cand = list(range(0, n, 3))
    got = dsmod.sample_indices(n, 11, 42, key, cand)
    assert set(got) <= set(cand) and len(got) == 11


# --------------------------------------------------------------------------- funnel gate (pure)
def test_gate_keeps_close_models_and_drops_clear_losers():
    def r(name, p, n):
        return {"model": name, "eval_x_pct": p, "eval_x_n": n, "eval_x_err": 0}
    rows = [r("a", 70, 33), r("b", 66, 33), r("c", 63, 33), r("d", 30, 33), r("e", 10, 33)]
    keep, rep, bar = funnel.gate(rows, ["x"], leaders=2)
    assert {"a", "b", "c"} <= keep and not {"d", "e"} & keep           # c is within noise of b: kept
    assert "upper bound" in rep["e"]["why"]
    keep, _, _ = funnel.gate(rows, ["x"], leaders=4)                   # never fewer than EVAL_LEADERS
    assert len(keep) >= 4
    rows.append({"model": "broken", "eval_x_pct": None, "eval_x_n": 10, "eval_x_err": 10})
    keep, rep, _ = funnel.gate(rows, ["x"], leaders=2)
    assert "broken" not in keep and "incomplete" in rep["broken"]["why"]


def test_funnel_run_screens_the_weak_model_and_ranks_only_finalists(monkeypatch, tmp_path, server):
    big = {"gpqa": lambda: fake_datasets.gpqa(30), "gsm8k": lambda: fake_datasets.gsm8k(30)}
    mock_server.STATE["weak"] = {M2}
    try:
        code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa", "gsm8k"], fake=big,
                                  EVAL_PROFILE="funnel", EVAL_FUNNEL_STAGES="test,basic", EVAL_LEADERS="1",
                                  EVAL_RESULT_CACHE="false")
    finally:
        mock_server.STATE["weak"] = set()
    assert code == 0
    strong, weak = doc["models"]
    assert ev(strong, "gpqa")["n"] == 24 and ev(weak, "gpqa")["n"] == 6       # 6 asked of both, 18 more of the leader
    assert weak["evaluation"]["screened"].startswith("screened out at test")
    assert weak["evaluation"]["stage"] == "test" and strong["evaluation"]["stage"] == "basic"
    assert weak["scores"]["score_agent"] is None and weak["scores"]["eligible"] is False
    assert "not ranked" in weak["scores"]["score_note"]
    # nested: the leader was never asked a question twice
    asked = [r["messages"][0]["content"] for r in mock_server.STATE["requests"] if r["model"] == M1
             and "enable_thinking" not in json.dumps(r.get("chat_template_kwargs") or {}) and r["messages"][0]["content"] != "hi"]
    assert len(asked) == len(set(asked)) == 48
    # like-for-like column: both models on the first stage's questions
    assert strong["evaluation"]["stage1_avg_pct"] > weak["evaluation"]["stage1_avg_pct"]
    gates = doc["evaluation"]["funnel"]["gates"]
    assert {g["model"]: g["kept"] for g in gates} == {M1: True, M2: False}


# --------------------------------------------------------------------------- result cache + resume
def test_result_cache_reuses_answers_only_when_nothing_changed(monkeypatch, tmp_path, server):
    def questions():
        return [r for r in mock_server.STATE["requests"] if r["messages"][0]["content"] != "hi"]
    code, _, d1 = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"])
    first = len(questions())
    assert code == 0 and first >= 24
    code, _, d2 = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"])
    assert code == 0 and len(questions()) == 0                               # everything came from the cache
    for a, b in zip(d1["models"], d2["models"]):
        for k in ("gsm8k", "gpqa"):
            assert (ev(a, k)["pct"], ev(a, k)["n"], ev(a, k)["fallback"]) == (ev(b, k)["pct"], ev(b, k)["n"], ev(b, k)["fallback"])
            assert ev(b, k)["cached"] == ev(b, k)["n"]
    assert d2["models"][0]["evaluation"]["cached"] == 12
    newest = max(glob.glob(str(tmp_path / "results" / "eval_details_*.jsonl")), key=lambda p: __import__("os").path.getmtime(p))
    recs = [json.loads(line) for line in open(newest)]
    assert len(recs) == 24 and all(d["cached"] for d in recs)            # the details file is still complete
    # a different setting is a different question: asked again
    code, _, d3 = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_TEMPERATURE="0.5")
    assert len(questions()) >= 12 and ev(d3["models"][0], "gsm8k")["cached"] == 0
    assert glob.glob(str(tmp_path / "results" / "cache" / "*.jsonl"))


def test_cache_keys_and_argv_normalisation(cfg):
    item = {"id": "q1", "prompt": "x"}
    p = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 10, "temperature": 0.0}
    k1 = cachemod.question_key(cfg, "gpqa", item, p, 16384)
    assert k1 == cachemod.question_key(cfg, "gpqa", item, {**p, "model": "other", "stream": True}, 16384)
    assert k1 != cachemod.question_key(cfg, "gpqa", item, {**p, "temperature": 0.6}, 16384)
    assert k1 != cachemod.question_key(cfg, "gpqa", item, p, 4096)
    a = cachemod.normalise_argv(["/app/llama-server", "-m", "/m.gguf", "--port", "5800", "-c", "65536", "--api-key", "s3cret"])
    assert a == ["-m", "/m.gguf", "-c", "65536"]


# --------------------------------------------------------------------------- skip undeployable models
def test_models_failing_min_ctx_are_not_evaluated(monkeypatch, tmp_path, server, capsys):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gpqa"], MIN_CTX="65536",
                           EVAL_RESULT_CACHE="false")
    assert code == 0
    assert all(m["evaluation"]["skipped"] and "below MIN_CTX" in m["evaluation"]["skipped"] for m in doc["models"])
    assert not served("[[ans:"), "no eval question may be sent to a skipped model"
    assert doc["models"][0]["scores"]["score_note"].startswith("evals skipped")
    assert "skipped (EVAL_SKIP_INELIGIBLE=auto)" in capsys.readouterr().out


# --------------------------------------------------------------------------- pipelined grading
def test_grading_overlaps_generation(monkeypatch, tmp_path, server):
    import time as _t
    import llmbench.evals.runner as runner
    real = runner.grade_item

    def slow(cfg, key, item, text, finish=None):
        _t.sleep(0.5)
        return real(cfg, key, item, text, finish)
    monkeypatch.setattr(runner, "grade_item", slow)
    six = lambda: fake_datasets.mbpp(6)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["mbpp"], fake={"mbpp": six},
                           EVAL_RESULT_CACHE="false", EVAL_SANDBOX_WORKERS="2")
    r = ev(doc["models"][0], "mbpp")
    assert code == 0 and r["pct"] == 100.0 and mock_server.STATE["max_in_flight"] == 1
    serial = 6 * (mock_server.LATENCY + 0.5)
    assert r["wall"] < serial * 0.75, f"grading did not overlap generation: {r['wall']}s vs {serial}s serial"
    assert r["sec"] >= 0.5                       # s/q still counts the grading time of each question


# --------------------------------------------------------------------------- batching: caps and s/q basis
def test_inflight_cap(cfg):
    items = [{"prompt": "x" * 300}] * 8
    me = types.SimpleNamespace(slots=4, ctx=65536, kv_unified=None)
    cfg.eval_concurrency = 4
    n, why = inflight_cap(cfg, me, "gpqa", items)
    assert n == 3 and "KV pool" in why                      # 3 x (prompt + 16384 + 4096) fit 64k, 4 do not
    me.kv_unified = False                                   # every slot owns its context: no pool limit
    assert inflight_cap(cfg, me, "gpqa", items)[0] == 4
    me.slots = 2
    assert inflight_cap(cfg, me, "gpqa", items) == (2, "2 server slot(s)")
    me.slots, me.kv_unified, cfg.eval_think_budget = 8, True, 0   # unlimited thinking: one request can fill the pool
    assert inflight_cap(cfg, me, "gpqa", items)[0] == 1
    cfg.eval_concurrency = 1
    assert inflight_cap(cfg, me, "gpqa", items) == (1, "")


def test_parse_kv_unified():
    assert parse_launch_args(["-np", "4", "-kvu"])["kv_unified"] is True
    assert parse_launch_args(["--no-kv-unified"])["kv_unified"] is False
    assert parse_launch_args(["-c", "8192"])["kv_unified"] is None


def test_batching_cannot_change_a_ranking(cfg):
    """Same answers and speed, but one model ran with 4 requests in flight (its measured s/q includes queueing).
    v2.1 ranks both on the single-stream estimate, so they tie; at concurrency 1 the measured s/q is used as before."""
    a = _row("A-14B-Q4_K_M", {"gpqa": 60, "gsm8k": 80}, eval_inflight=4)
    b = _row("B-14B-Q4_K_M", {"gpqa": 60, "gsm8k": 80}, eval_inflight=1)
    a.update(eval_gpqa_sec=70.0, eval_gsm8k_sec=70.0)
    finalize([a, b], cfg)
    assert a["eval_sec_basis"] == b["eval_sec_basis"] == "single"
    assert a["eval_avg_sec"] == b["eval_avg_sec"] == pytest.approx(1200 / 60 + 300 / 2000 + 0.1, abs=0.1)
    assert a["score_agent"] == pytest.approx(b["score_agent"])
    c, d = _row("C-14B-Q4_K_M", {"gpqa": 60}), _row("D-14B-Q4_K_M", {"gpqa": 60})
    d.update(eval_gpqa_sec=40.0)
    finalize([c, d], cfg)
    assert c["eval_sec_basis"] == "measured" and d["eval_avg_sec"] == 40.0 and c["score_agent"] > d["score_agent"]


def test_concurrency_capped_by_server_slots(monkeypatch, tmp_path, server, capsys):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_CONCURRENCY="4",
                           EVAL_RESULT_CACHE="false")
    assert code == 0 and mock_server.STATE["max_in_flight"] == 1         # the mock has 1 slot
    assert ev(doc["models"][0], "gsm8k")["concurrency"] == 1
    assert "capped: 1 server slot(s)" in capsys.readouterr().out


# --------------------------------------------------------------------------- timing report + estimate
def test_timing_report(monkeypatch, tmp_path, server):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gpqa"], EVAL_RESULT_CACHE="false")
    assert code == 0
    t = doc["models"][0]["timing"]
    for phase in ("settle", "load", "speed", "eval_load", "eval:gpqa"):
        assert phase in t["phases_s"], phase
    assert t["time_total_s"] >= t["time_eval_s"] > 0 and t["time_speed_s"] > 0
    run_t = doc["run"]["timing"]
    assert run_t["phase_wall_s"]["speed_s"] > 0 and run_t["phase_wall_s"]["eval_s"] > 0
    assert run_t["eval_by_dataset_s"]["gpqa"] > 0 and run_t["eval_tokens"]["think"] > 0
    rows = list(csv.DictReader(open(next((tmp_path / "results").glob("benchmark_2*.csv")))))
    assert float(rows[0]["time_total_s"]) > 0 and int(rows[0]["eval_tok_think"]) > 0
    dash = open(next((tmp_path / "results").glob("benchmark_dashboard_*.html"))).read()
    assert "Where the time went" in dash and '"timing"' in dash
    # the first question's latency no longer carries the model load: a warm-up request comes first
    reqs = [r for r in mock_server.STATE["requests"] if r["model"] == M1]
    first_eval = next(i for i, r in enumerate(reqs) if "[[ans:" in r["messages"][0]["content"])
    assert reqs[first_eval - 1]["messages"][0]["content"] == "hi" and reqs[first_eval - 1]["max_tokens"] == 1


def test_estimate_uses_earlier_results(monkeypatch, tmp_path, server, capsys):
    run_cli(monkeypatch, tmp_path, server, [], datasets=["gpqa"], EVAL_RESULT_CACHE="false")
    capsys.readouterr()
    code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--estimate"], datasets=["gpqa", "gsm8k"],
                         EVAL_PROFILE="standard", EVAL_RESULT_CACHE="false")
    out = capsys.readouterr().out
    assert code == 0 and "Run-time estimate for 2 model(s) from server" in out and "Total: ~" in out
    assert "tok/s 50 (history)" in out and "measured" in out
    code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--estimate"], datasets=["gpqa"], EVAL_PROFILE="funnel")
    assert "without the funnel" in capsys.readouterr().out


def test_batch_probe(monkeypatch, tmp_path, server, ctx_slots, capsys):
    ctx_slots(slots=4, n_ctx=131072)
    code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--batch-probe"], datasets=["gpqa", "gsm8k"],
                         BATCH_PROBE_LEVELS="1,2,8", BATCH_PROBE_QUESTIONS="8", INCLUDE_MODELS=M1)
    assert code == 0
    doc = json.load(open(next((tmp_path / "results").glob("batch_probe_*.json"))))
    m = doc["models"][0]
    lv = {x["level"]: x for x in m["levels"]}
    assert set(lv) == {1, 2} and m["slots"] == 4                   # 8 > 4 slots: skipped, not faked
    assert lv[2]["agree"] == 1.0 and lv[2]["errors"] == 0 and lv[2]["gain"] > 1.2
    assert m["recommended"] == 2 and "x8: skipped" in capsys.readouterr().out


# --------------------------------------------------------------------------- long context: one shared document
def test_longctx_reuses_one_document(monkeypatch, tmp_path, server, ctx_slots):
    ctx_slots(n_ctx=40000)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], EVAL_LONGCTX="true",
                           LONGCTX_LENGTHS="32768", LONGCTX_TRIALS="3", LONGCTX_KEYS="2", EVAL_RESULT_CACHE="false")
    assert code == 0 and doc["models"][0]["long_context"]["longctx_32k_pct"] == 100.0
    lc = [d for d in details(tmp_path) if d["dataset"] == "longctx" and d["model"] == M1]
    assert len(lc) == 3 and lc[0]["cache_n"] == 0 and all(d["cache_n"] > 0 for d in lc[1:])
    names = [n for d in lc for n in d["asked"]]
    assert len(names) == len(set(names)) == 6                       # each question asks different vaults
    docs = {r["messages"][0]["content"].split("</document>")[0] for r in mock_server.STATE["requests"]
            if r["model"] == M1 and "access code" in r["messages"][0]["content"]}
    assert len(docs) == 1


# --------------------------------------------------------------------------- thinking budgets and budget forcing
def test_budget_forcing_keeps_the_reasoning(monkeypatch, tmp_path, server):
    qs = lambda: fake_datasets.FakeDS([{"question": f"Q{i}? [[ans:B]] {m}", "answer": "B"}
                                       for i, m in enumerate(["[[ramble]]", ""])])
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], fake={"gpqa": qs},
                           EVAL_THINK_BUDGET="1500", EVAL_BUDGET_FORCING="true", EVAL_RESULT_CACHE="false")
    r = ev(doc["models"][0], "gpqa")
    assert code == 0 and (r["pct"], r["cap"], r["forced"], r["fallback"]) == (100.0, 1, 1, 0)
    rec = {d["id"]: d for d in details(tmp_path) if d["model"] == M1}["gpqa-0"]
    assert [a["mode"] for a in rec["attempts"]] == ["think", "forced"] and rec["forced"]
    assert [k for k, _ in mock_server.STATE["native"]][:2] == ["apply-template", "completion"]
    assert len(served("[[ramble]]")) == 1                           # no No-Thinking retry was needed
    assert doc["evaluation"]["budget_forcing"] is True


def test_per_dataset_think_budget(monkeypatch, tmp_path, server):
    qs = lambda: fake_datasets.FakeDS([{"question": "Q? [[ans:A]] [[ramble]]", "answer": "A"}])
    code, secs, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], fake={"gpqa": qs},
                              EVAL_THINK_BUDGET="0", EVAL_THINK_BUDGET_GPQA="1200", EVAL_RESULT_CACHE="false")
    r = ev(doc["models"][0], "gpqa")
    assert code == 0 and r["cap"] == 1 and r["pct"] == 100.0 and secs < 30
    assert doc["evaluation"]["think_budgets"] == {"gpqa": 1200}


# --------------------------------------------------------------------------- LiveCodeBench window, language guard
def test_lcb_since_and_language_guard(cfg):
    rows = fake_datasets.FakeDS([{"contest_date": d, "question_content": "q"} for d in
                                 ("2023-05-07", "2024-11-02", "2025-01-01", "2025-03-30")])
    cfg.eval_lcb_since = "2025-01-01"
    assert dsmod.lcb_candidates(cfg, rows) == [2, 3]
    cfg.eval_lcb_since = "2026-01-01"
    with pytest.raises(SystemExit):
        dsmod.lcb_candidates(cfg, rows)
    dsmod.language_guard("livecodebench", [{"prompt": "Print twice the number."}] * 50)
    with pytest.raises(SystemExit):
        dsmod.language_guard("livecodebench", [{"prompt": "숫자의 두 배를 출력하세요."}] * 5 + [{"prompt": "x"}] * 45)


def test_auto_sandbox_no_longer_falls_back_silently(cfg, monkeypatch):
    monkeypatch.setattr(sandbox, "_docker_ready", lambda c: (False, "docker not found on PATH"))
    cfg.eval_code_sandbox = "auto"
    with pytest.raises(SystemExit) as e:
        sandbox.resolve_sandbox(cfg, run_selftest=False)
    assert "EVAL_CODE_SANDBOX=subprocess" in str(e.value)
    cfg.eval_code_sandbox = "subprocess"
    assert sandbox.resolve_sandbox(cfg, run_selftest=False) == "subprocess"


def test_longctx_at_full_context_fits_and_recovers(monkeypatch, tmp_path, server, ctx_slots):
    """Real run (v2.0): 128k at ctx 131,072 overshot to 131,644-132,331 tokens -> HTTP 400, no result. The length now
    keeps 3% back, and an 'exceeds the available context size' answer shrinks the document once and retries."""
    import llmbench.evals.longctx as lc
    ctx_slots(n_ctx=32768)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], EVAL_LONGCTX="true",
                           LONGCTX_LENGTHS="32768", EVAL_RESULT_CACHE="false")
    assert code == 0 and doc["models"][0]["long_context"]["longctx_32k_pct"] == 100.0
    real = lc.build_haystack                          # now make the words->tokens estimate far too optimistic
    monkeypatch.setattr(lc, "build_haystack", lambda t, tpw, k, seed: real(int(t * 1.25), tpw, k, seed))
    code, _, doc = run_cli(monkeypatch, tmp_path / "b", server, ["--evals-only"], datasets=["gpqa"],
                           EVAL_LONGCTX="true", LONGCTX_LENGTHS="32768", EVAL_RESULT_CACHE="false") \
        if (tmp_path / "b").mkdir() is None else None
    assert code == 0 and doc["models"][0]["long_context"]["longctx_32k_pct"] == 100.0
    assert not doc["models"][0]["long_context"].get("longctx_note")
    asks = [r for r in mock_server.STATE["requests"] if r["model"] == M1 and "access code" in r["messages"][0]["content"]]
    assert len(asks) == 3                             # the first was rejected (too long), rebuilt once, then 2 trials


def test_batch_probe_refuses_a_whole_server(monkeypatch, tmp_path, server, capsys):
    import llmbench.batchprobe as bp
    monkeypatch.setattr(bp, "MAX_MODELS", 1)          # the mock has 2 models
    code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--batch-probe"], datasets=["gsm8k"])
    assert code != 0 and not list((tmp_path / "results").glob("batch_probe_*.json"))
    assert not served("[[ans:"), "nothing may be asked before the model list is accepted"


def test_longctx_model_that_always_thinks_is_not_scored_zero(monkeypatch, tmp_path, server, ctx_slots):
    """Real run (v2.0): LFM2.5 ignores enable_thinking=false; all 400 reply tokens went to reasoning_content, the
    answer was empty and long-ctx scored 0% (25% of its Agent quality). Such a reply is now asked again with room."""
    ctx_slots(n_ctx=40000)
    mock_server.STATE["thinks_always"] = {M1}
    try:
        code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gpqa"], EVAL_LONGCTX="true",
                               LONGCTX_LENGTHS="32768", EVAL_RESULT_CACHE="false")
    finally:
        mock_server.STATE["thinks_always"] = set()
    lc = doc["models"][0]["long_context"]
    assert code == 0 and lc["longctx_32k_pct"] == 100.0 and "asked again" in lc["longctx_note"]
    recs = [d for d in details(tmp_path) if d["dataset"] == "longctx" and d["model"] == M1]
    assert all(d["thinking_retry"] for d in recs)


def test_probe_steady_gain_and_recommendation():
    """Real probe (2026-11-08): 4 questions per dataset at x4 = one batch per dataset, so tokens / wall time showed
    1.45x / 1.22x / 1.26x. With a full queue the gain is in-flight x per-request speed ratio: ~2.7x / 1.9x / 3.3x."""
    from llmbench.batchprobe import recommend, steady_gain
    lfm = [{"level": 1, "n": 16, "wall_s": 276.7, "tok_s": 230.3, "latency_s": 17.4, "acc": 100.0, "agree": 1.0, "errors": 0},
           {"level": 4, "n": 16, "wall_s": 125.5, "tok_s": 333.4, "latency_s": 16.73, "acc": 100.0, "agree": 1.0, "errors": 0}]
    qwen = [{"level": 1, "n": 16, "wall_s": 529.6, "tok_s": 84.3, "latency_s": 33.2, "acc": 100.0, "agree": 1.0, "errors": 0},
            {"level": 4, "n": 16, "wall_s": 423.9, "tok_s": 103.0, "latency_s": 69.88, "acc": 81.2, "agree": 0.812, "errors": 0}]
    assert steady_gain(lfm[1], lfm[0]) == pytest.approx(2.73, abs=0.02)
    assert steady_gain(qwen[1], qwen[0]) == pytest.approx(1.86, abs=0.02)
    for levels in (lfm, qwen):
        for lv in levels:
            lv["steady_gain"] = steady_gain(lv, levels[0])
    assert recommend(lfm) == 4
    assert recommend(qwen) == 4            # 3 flips of 16 is within the noise of a ~90% model; no errors
    qwen[1]["errors"] = 2
    assert recommend(qwen) == 1            # request errors (a KV pool that overflowed) always block a level
    big = [dict(qwen[0], n=48), dict(qwen[1], n=48, errors=0, agree=0.80, acc=95.0)]
    assert recommend(big) == 1             # from 40 questions up, agreement below 90% blocks the level


def test_dashboard_dataset_links_and_column_tools(tmp_path):
    """Header test names link to their dataset page; the table has the column chooser and per-header tooltips."""
    from llmbench.config import EVAL_SPECS
    from llmbench.reporting.dashboard import TEMPLATE_PATH, dataset_url
    assert dataset_url(EVAL_SPECS["mmlu_pro"]) == "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
    assert dataset_url(EVAL_SPECS["gsm8k"], "abc123") == "https://huggingface.co/datasets/openai/gsm8k/tree/abc123"
    assert dataset_url(EVAL_SPECS["toolcall"]).endswith("/llmbench/evals/toolcall_cases.py")
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    for needle in ('id="colpanel"', "data-colall", "data-hide=", "COLINFO", "testLinks()"):
        assert needle in html


def _dash_data(path):
    line = next(l for l in open(path, encoding="utf-8") if l.startswith("const D = "))
    return json.loads(line[len("const D = "):line.index(", META = D.meta")].replace("<\\/", "</"))


def test_reload_rebuilds_the_same_dashboard_from_json(cfg, monkeypatch, tmp_path):
    """--reload must show exactly what the run measured and scored, with the run's settings, not today's .env."""
    from llmbench import reload as rl
    from llmbench.diagnostics import demo_results
    from llmbench.reporting.dashboard import render_dashboard
    from llmbench.reporting.json_out import write_json
    monkeypatch.setattr(rl, "SCRIPT_DIR", tmp_path)            # don't touch the real benchmark_dashboard.html
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_profile = "funnel"
    rows = demo_results(cfg)
    rows[1]["eval_screened"], rows[1]["eval_stage"] = True, "quick"
    finalize(rows, cfg)
    original = tmp_path / "orig.html"
    render_dashboard(rows, cfg, original, "demo data", "Test host")
    write_json(rows, cfg, {"stamp": "20261109_000000", "hardware": "Test host", "cpu_source": "demo data",
                           "status": "complete"}, "20261109_000000")

    monkeypatch.setenv("EVAL_PROFILE", "quick")                # today's .env differs from the run's settings
    out = rl.reload_dashboard(load_config(), str(cfg.output_dir))
    assert out.name == "benchmark_dashboard_20261109_000000_reload.html"
    a, b = _dash_data(original), _dash_data(out)
    a["meta"].pop("generated"), b["meta"].pop("generated")
    assert b["meta"] == a["meta"]
    assert len(a["models"]) == len(b["models"])
    for ra, rb in zip(a["models"], b["models"]):
        for k, v in ra.items():                               # every value the dashboard had is back, unchanged
            assert rb.get(k) == v or (v is None and k not in rb), (ra["model"], k, v, rb.get(k))
    assert (tmp_path / "benchmark_dashboard.html").read_text() == out.read_text()
