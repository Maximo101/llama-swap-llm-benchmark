"""v2.2: GGUF header, launch flags, KV / max-ctx maths, tool calling, long-context retrieval, Agent score, budgets."""
import json
import math

import pytest

import gguf_writer
from llmbench import gguf, modelinfo
from llmbench.config import load_config
from llmbench.evals import longctx, toolcall
from llmbench.evals.toolcall_cases import CASES, TOOLS, build_items
from llmbench.reporting.margin import margin_of_error
from llmbench.scoring import blank, budget_checks, finalize, proxy_quality, vram_fit
from llmbench.telemetry.host import offload_summary, parse_launch_args


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    import os
    for k in list(os.environ):
        if k.startswith(("EVAL_", "GPU_", "LLAMA_", "RUN_", "BUDGET_", "MIN_CTX", "AGENT_", "VRAM_", "LONGCTX_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    return load_config()


# --------------------------------------------------------------------------- GGUF
def test_gguf_dense_gqa(tmp_path):
    p = gguf_writer.dense_model(tmp_path / "m.gguf", vocab=5000, pad_to=200_000)      # vocab > 4096: skipped array
    g = gguf.inspect(p)
    assert g["gguf_arch"] == "llama" and g["attn_type"] == "GQA" and g["expert_count"] is None
    assert g["full_attn_layers"] == 4 and g["n_ctx_train"] == 32768
    assert g["params_total"] == 512 * 5000 + 4 * (512 * 512 + 512 * 2048)
    assert g["file_bytes"] == 200_000 and g["bpw"] == pytest.approx(200_000 * 8 / g["params_total"])
    # 4 layers x 2 KV heads x (64 + 64) dims x bytes
    assert gguf.kv_bytes(g, "f16", "f16")[0] == 4 * 2 * 128 * 2
    assert gguf.kv_bytes(g, "q8_0", "q8_0")[0] == pytest.approx(4 * 2 * 128 * 34 / 32)
    assert gguf.kv_bytes(g, "q8_0", "q4_0")[0] == pytest.approx(4 * 2 * 64 * (34 + 18) / 32)


def test_gguf_moe_mla_is_moe_without_a3b_in_the_name(tmp_path):
    g = gguf.inspect(gguf_writer.moe_mla_model(tmp_path / "GLM-4.7-Flash-30B-UD-IQ3_XXS.gguf"))
    assert g["expert_count"] == 64 and g["expert_used"] == 4 and g["attn_type"] == "MLA"
    experts = 4 * 256 * 128 * 64
    assert g["params_active"] == pytest.approx(g["params_total"] - experts + experts * 4 / 64)
    assert gguf.kv_bytes(g, "f16", "f16")[0] == 4 * (128 + 32) * 2                     # compressed latent only


def test_gguf_sliding_window_and_hybrid(tmp_path):
    # gemma3: the layout is not in the file -> llama.cpp's 5 local : 1 global, marked approximate
    a = "gemma3"
    meta = {"general.architecture": a, f"{a}.block_count": 12, f"{a}.embedding_length": 512,
            f"{a}.attention.head_count": 8, f"{a}.attention.head_count_kv": 4, f"{a}.attention.key_length": 64,
            f"{a}.attention.value_length": 64, f"{a}.attention.sliding_window": 1024}
    g = gguf.describe(meta, [("w", 10)])
    assert (g["full_attn_layers"], g["swa_layers"], g["kv_approx"]) == (2, 10, True) and "SWA" in g["attn_type"]
    per_tok, swa_fixed = gguf.kv_bytes(g, "f16", "f16")
    assert per_tok == 2 * 4 * 128 * 2 and swa_fixed == 10 * 4 * 128 * 2 * 1024
    # an explicit per-layer bool pattern wins and is not approximate
    meta[f"{a}.attention.sliding_window_pattern"] = [True, False]
    g = gguf.describe(meta, [("w", 10)])
    assert (g["full_attn_layers"], g["swa_layers"], g["kv_approx"]) == (6, 6, False)
    # hybrid SSM: per-layer head_count_kv with zeros (lfm2 / nemotron-h style), and qwen3next-style interval
    b = "lfm2"
    g = gguf.describe({"general.architecture": b, f"{b}.block_count": 6, f"{b}.embedding_length": 256,
                       f"{b}.attention.head_count": 8, f"{b}.attention.head_count_kv": [0, 0, 2, 0, 0, 2]}, [])
    assert g["full_attn_layers"] == 2 and g["attn_type"] == "hybrid GQA + SSM"
    c = "qwen3next"
    g = gguf.describe({"general.architecture": c, f"{c}.block_count": 8, f"{c}.embedding_length": 256,
                       f"{c}.attention.head_count": 8, f"{c}.attention.head_count_kv": 2,
                       f"{c}.full_attention_interval": 4}, [])
    assert g["full_attn_layers"] == 2 and g["attn_type"].startswith("hybrid")


def test_gguf_split_files_and_errors(tmp_path):
    assert [p.name for p in gguf.shard_paths(tmp_path / "x-00001-of-00003.gguf")] == \
        ["x-00001-of-00003.gguf", "x-00002-of-00003.gguf", "x-00003-of-00003.gguf"]
    (tmp_path / "bad.gguf").write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(gguf.GGUFError):
        gguf.inspect(tmp_path / "bad.gguf")


def test_find_readable_path_map_and_search(tmp_path):
    d = tmp_path / "host" / "sub"
    d.mkdir(parents=True)
    f = d / "model.gguf"
    f.write_bytes(b"x")
    assert gguf.find_readable("/models/sub/model.gguf", None, [("/models", str(tmp_path / "host"))]) == str(f)
    assert gguf.find_readable("/elsewhere/model.gguf", None, [], [str(tmp_path / "host")]) == str(f)
    assert gguf.find_readable("/nope/missing.gguf") is None
    assert gguf.parse_path_map("/models=/mnt/AI-Models, /a=/b") == [("/models", "/mnt/AI-Models"), ("/a", "/b")]


# --------------------------------------------------------------------------- launch flags, KV, max ctx
def test_launch_flags():
    info = parse_launch_args(["-m", "/models/q.gguf", "-c", "131072", "-ctk", "q8_0", "--cache-type-v=q4_0", "-fa",
                              "on", "-ub", "1024", "-b", "4096", "--mmproj", "/models/mmproj.gguf", "-np", "2",
                              "--n-cpu-moe", "22", "-ot", "exps=CPU", "--jinja", "-ngl", "99"])
    assert (info["ctk"], info["ctv"], info["fa"], info["ubatch"], info["batch"]) == ("q8_0", "q4_0", "on", 1024, 4096)
    assert info["mmproj"] == "/models/mmproj.gguf" and info["parallel"] == 2 and info["jinja"] and info["ctx"] == 131072
    assert info["model_path"] == "/models/q.gguf" and "exps=CPU" in info["ot"]
    assert offload_summary(info) == "n-cpu-moe 22, override-tensor CPU"
    assert parse_launch_args(["-fa", "-c", "4096"])["fa"] == "on"                  # older bare switch
    assert modelinfo.kv_label("q8_0", "q8_0") == "q8_0" and modelinfo.kv_label(None, "q4_0") == "f16/q4_0"
    assert modelinfo.kv_at_least("q8_0", "q8_0") and modelinfo.kv_at_least("q8_0/q4_0", "q8_0") is False
    assert modelinfo.kv_at_least("f16", "q8_0") and modelinfo.kv_at_least(None, "q8_0") is None


def test_max_ctx_at():
    r = {"kv_kib_token": 64, "ctx_size": 65536, "vram_gb": 12.0, "vram_total_gb": 16.0, "offload": "none"}
    # 85% of 16 GB = 13.6 GB -> 1.6 GB spare / 64 KiB per token = 26,214 more tokens
    assert modelinfo.max_ctx_at(r, 85) == (65536 + int(1.6 * 1024 ** 3 / (64 * 1024))) // 1024 * 1024
    assert modelinfo.max_ctx_at({**r, "n_ctx_train": 70000}, 85) == 70000
    assert modelinfo.max_ctx_at({**r, "vram_gb": 15.5}, 85) < 65536                    # already over: less than loaded
    assert modelinfo.max_ctx_at({**r, "offload": "KV on CPU"}, 85) is None


# --------------------------------------------------------------------------- tool calling
def _reply(calls=None, content=""):
    return toolcall.encode_reply(content, [{"function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls or []])


def _item(cid):
    return next(i for i in build_items() if i["id"] == "tool-" + cid)


def test_toolcall_grading():
    it = _item("simple-02")
    ok, pred, why = toolcall.grade_toolcall(it, _reply([("convert_currency", {"amount": "250", "from_currency": "usd",
                                                                               "to_currency": "EUR"})]))
    assert ok and pred == "convert_currency"                                       # "250" == 250, case-insensitive
    ok, _, why = toolcall.grade_toolcall(it, _reply([("convert_currency", {"amount": 25, "from_currency": "USD",
                                                                            "to_currency": "EUR"})]))
    assert not ok and "amount" in why
    ok, _, why = toolcall.grade_toolcall(it, _reply([("convert_currency", {"amount": 250, "from_currency": "USD",
                                                                            "to_currency": "EUR", "fee": 1})]))
    assert not ok and "unknown argument" in why
    # parallel: order does not matter, both needed
    it = _item("parallel-03")
    assert toolcall.grade_toolcall(it, _reply([("get_stock_price", {"ticker": "MSFT"}),
                                               ("get_stock_price", {"ticker": "AAPL"})]))[0]
    assert not toolcall.grade_toolcall(it, _reply([("get_stock_price", {"ticker": "AAPL"})]))[0]
    # irrelevance: no call is the right answer
    it = _item("irrelevance-01")
    assert toolcall.grade_toolcall(it, _reply(content="391"))[0]
    assert not toolcall.grade_toolcall(it, _reply([("get_weather", {"city": "x"})]))[0]
    # a call written as text: wrong in strict mode, graded in lenient mode
    it = _item("simple-09")
    text = toolcall.encode_reply('<tool_call>{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}</tool_call>', [])
    ok, _, why = toolcall.grade_toolcall(it, text, strict=True)
    assert not ok and "--jinja" in why
    assert toolcall.grade_toolcall(it, text, strict=False)[0]
    bad = toolcall.encode_reply("", [{"function": {"name": "get_stock_price", "arguments": "{not json"}}])
    assert not toolcall.grade_toolcall(it, bad)[0]
    # optional arguments: may be left out (None in the allowed list)
    assert toolcall.grade_toolcall(_item("simple-03"), _reply([("set_thermostat", {"room": "Bedroom",
                                                                                   "temperature": 21.0})]))[0]


def test_toolcall_cases_are_consistent():
    for c in CASES:
        for exp in c["expect"]:
            props = TOOLS[exp["name"]]["function"]["parameters"]["properties"]
            assert exp["name"] in c["tools"] and set(exp["args"]) <= set(props), c["id"]
            required = TOOLS[exp["name"]]["function"]["parameters"]["required"]
            assert all(None not in exp["args"].get(r, [0]) for r in required), c["id"]
    assert {c["category"] for c in CASES} == {"simple", "multiple", "parallel", "irrelevance"}


# --------------------------------------------------------------------------- long context
def test_longctx_prompt_and_scoring():
    prompt, asked, decoy = longctx.build_prompt(3000, 1.2, 4, "42:x:0")
    assert len(asked) == 4 and len(decoy) == 4 and not set(asked) & set(decoy)
    for n, c in {**asked, **decoy}.items():
        assert f"vault {n} is {c}" in prompt
    assert all(n in prompt.split("<document>")[1].split("</document>")[1] for n in asked)     # asked after the text
    good = "\n".join(f"Vault {n}: {c}" for n, c in asked.items())
    assert longctx.score_reply(good, asked, decoy) == 1.0
    names = list(asked)
    half = f"Vault {names[0]}: {asked[names[0]]}\nVault {names[1]}: {decoy[list(decoy)[0]]}"
    assert longctx.score_reply(half, asked, decoy) == 0.25
    assert longctx.score_reply("<think>" + good + "</think>nothing", asked, decoy) == 0.0


# --------------------------------------------------------------------------- scores
def _row(name, tps=100.0, vram=60.0, ev=None, n=10, err=0, fb=0, **kw):
    r = blank(name)
    kw.setdefault("prefill_tps", 3000.0)
    r.update(tech_tps=tps, logic_tps=tps, vram_gb=vram * 0.16, vram_total_gb=16.0, offload="none", **kw)
    if ev is not None or err:
        r.update(eval_mmlu_pro_pct=ev, eval_mmlu_pro_n=n, eval_mmlu_pro_err=err, eval_mmlu_pro_fallback=fb,
                 eval_mmlu_pro_sec=20.0)
    return r


def test_failed_evals_get_no_agent_score(cfg):
    """v2.1 bug: a model whose evals all errored got the params proxy (thousands) and ranked first."""
    rs = [_row("Good-27B-Q4_K_M", ev=90), _row("Ok-9B-Q4_K_M", ev=70),
          _row("Broken-35B-A3B-IQ2_M", vram=48, n=5, err=5)]
    finalize(rs, cfg)
    by = {r["model"]: r for r in rs}
    assert by["Broken-35B-A3B-IQ2_M"]["score_agent"] is None
    assert "5 request errors" in by["Broken-35B-A3B-IQ2_M"]["score_note"]
    assert 0 < by["Ok-9B-Q4_K_M"]["score_agent"] < by["Good-27B-Q4_K_M"]["score_agent"] <= 100


def test_retry_penalty(cfg):
    rs = [_row("A-9B-Q4_K_M", ev=80, n=20, fb=0), _row("B-9B-Q4_K_M", ev=80, n=20, fb=10)]
    finalize(rs, cfg)
    a, b = rs
    assert (b["eval_retry_n"], b["eval_retry_pct"]) == (10, 50.0)
    assert b["eval_quality"] == pytest.approx(80 - cfg.agent_fallback_penalty * 0.5) and a["eval_quality"] == 80
    assert b["agent_quality"] == pytest.approx(a["agent_quality"] * (1 - cfg.agent_fallback_penalty / 100 * 0.5))


def test_proxy_no_evals_prefers_quality_not_empty_vram(cfg):
    """Speed-only run: v2.1 ranked a 35B-A3B IQ2_M at 48% VRAM first because the proxy multiplied by free VRAM."""
    small = _row("Qwen3.8-35B-A3B-IQ2_M", tps=71, vram=48, prefill_tps=1585.0)
    dense = _row("Qwen3.8-27B-Q4_K_M", tps=55, vram=80, prefill_tps=1636.0)
    rs = [small, dense]
    finalize(rs, cfg)
    assert dense["score_agent"] > small["score_agent"] and dense["score_agent"] <= 100
    assert "proxy" in small["score_note"]
    assert proxy_quality({"params_b": 35, "active_b": 3, "quant": "IQ2_M"}) < proxy_quality({"params_b": 27, "quant": "Q4_K_M"})


def test_prefill_counts_in_agent_speed(cfg):
    fast = _row("F-9B-Q4_K_M", ev=80, prefill_32k_tps=4000.0)
    slow = _row("S-9B-Q4_K_M", ev=80, prefill_32k_tps=400.0)
    finalize([fast, slow], cfg)
    assert fast["agent_quality"] == slow["agent_quality"] and fast["score_agent"] > slow["score_agent"]


def test_vram_fit_floor(cfg):
    assert vram_fit(85, cfg) == 1.0 and vram_fit(40, cfg) == pytest.approx(0.85 + 0.15 * 40 / 80)
    assert vram_fit(0, cfg) == pytest.approx(cfg.vram_underuse_floor) and vram_fit(94, cfg) == pytest.approx(0.5)


# --------------------------------------------------------------------------- budgets, min ctx, margin
def test_budget_profile(monkeypatch, cfg):
    monkeypatch.setenv("BUDGET_PROFILE", "server")
    monkeypatch.setenv("BUDGET_SERVER_MIN_TPS", "60")
    monkeypatch.setenv("BUDGET_SERVER_MAX_RETRY_PCT", "off")
    c = load_config()
    assert c.min_ctx == 65536 and c.budget["rules"]["KV_MIN"] == "q8_0" and "MAX_RETRY_PCT" not in c.budget["rules"]
    ok = _row("Ok-9B-Q4_K_M", ev=80, ctx_size=131072, kv_type="q8_0", vram_delta_gb=10.0)
    kv = _row("Kv-9B-Q4_K_M", ev=80, ctx_size=131072, kv_type="q4_0", vram_delta_gb=10.0)
    big = _row("Big-9B-Q4_K_M", ev=80, ctx_size=131072, kv_type="f16", vram_delta_gb=14.5)
    short = _row("Short-9B-Q4_K_M", ev=80, ctx_size=32768, kv_type="q8_0", vram_delta_gb=10.0, ctx_ok=False)
    slow = _row("Slow-9B-Q4_K_M", tps=30, ev=80, ctx_size=131072, kv_type="q8_0", vram_delta_gb=10.0)
    rs = [ok, kv, big, short, slow]
    finalize(rs, c)
    assert ok["budget_pass"] and ok["eligible"]
    assert not kv["budget_pass"] and "KV_MIN" in kv["budget_fails"] and not kv["eligible"]
    assert not big["budget_pass"] and big["budget"]["VRAM"]["ok"] is False          # 14.5 + 2.0 > 16
    assert not short["budget_pass"] and short["budget"]["MIN_CTX"]["ok"] is False and not short["eligible"]
    assert slow["budget"]["MIN_TPS"]["ok"] is False
    assert budget_checks(blank("x"), c)["VRAM"]["ok"] is None                        # not measured is not a fail


def test_min_ctx_flag_and_refuse(cfg):
    from llmbench.speed import check_min_ctx
    cfg.min_ctx = 65536
    r = blank("m")
    r["ctx_size"] = 32768
    assert check_min_ctx(cfg, r) is False and r["status"] == "ok" and r["ctx_ok"] is False
    cfg.min_ctx_action = "refuse"
    r = blank("m")
    r["ctx_size"] = 32768
    check_min_ctx(cfg, r)
    assert r["status"] == "skipped" and "MIN_CTX" in r["error"]


def test_margin_of_error(cfg):
    cfg.eval_profile = "test"
    rs = [{"eval_mmlu_pro_n": 6, "eval_humaneval_n": 6, "eval_mbpp_n": 6}]
    m = margin_of_error(cfg, rs)
    assert m["rough"] and round(m["per_dataset"]["mmlu_pro"]["moe"]) == 40 and round(m["average"]["moe"]) == 23
    cfg.eval_profile = "standard"
    assert not margin_of_error(cfg, [{"eval_gpqa_n": 96}])["rough"]
    assert margin_of_error(cfg, [{}]) is None


def test_prefill_and_longctx_length_slots(monkeypatch, cfg):
    monkeypatch.setenv("PREFILL_CTX_TESTS", "8k, 32768, 100000")
    monkeypatch.setenv("LONGCTX_LENGTHS", "64k")
    c = load_config()
    assert c.prefill_ctx_tests == (8192, 32768) and c.longctx_lengths == (65536,)
