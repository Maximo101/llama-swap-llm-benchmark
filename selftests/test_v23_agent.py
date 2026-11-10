"""v2.3 step 1: Agent score - context minimum + curve, common-length long-ctx, composite, tool minimum, tie zone,
tool calling in full at every stage, speed factor, the funnel gate on the same composite, --compare guard."""
import json

import pytest

from llmbench import agent
from llmbench.compare import check, scoring_generation
from llmbench.config import funnel_stages, load_config, rows_for
from llmbench.evals import funnel
from llmbench.scoring import agent_notes, blank, finalize, speed_factor


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    import os
    for k in list(os.environ):
        if k.startswith(("EVAL_", "GPU_", "LLAMA_", "RUN_", "BUDGET_", "MIN_", "AGENT_", "VRAM_", "LONGCTX_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    return load_config()


def _row(name, mmlu=90.0, he=90.0, tools=100.0, lc=None, ctx=131072, n=43, tool_n=32, fb=0, tps=150.0, ttft=4.0,
         sec=30.0, vram=85.0):
    r = blank(name)
    r.update(tech_tps=tps, logic_tps=tps, vram_gb=16.0 * vram / 100, vram_total_gb=16.0, ctx_size=ctx,
             agent_ttft_s=ttft, offload="none")
    for k, p, nn in (("mmlu_pro", mmlu, n), ("humaneval", he, n), ("toolcall", tools, tool_n)):
        if p is None:
            continue
        r.update({f"eval_{k}_pct": p, f"eval_{k}_n": nn, f"eval_{k}_rows": nn, f"eval_{k}_sec": sec,
                  f"eval_{k}_err": 0, f"eval_{k}_fallback": fb if k == "mmlu_pro" else 0})
    for l, p in (lc or {}).items():
        r[f"longctx_{l}_pct"] = p
    return r


# --------------------------------------------------------------------------- context
def test_ctx_curve_parse_and_interpolate():
    c = agent.parse_ctx_curve("131072:1.0,92160:0.9,65536:0.75")
    assert c == ((65536, 0.75), (92160, 0.9), (131072, 1.0))
    assert agent.parse_ctx_curve("128k:1, 64k:0.8") == ((65536, 0.8), (131072, 1.0))
    assert agent.parse_ctx_curve("") == () and agent.parse_ctx_curve("off") == ()
    f = lambda x: agent.ctx_factor(x, c)
    assert f(131072) == 1.0 and f(262144) == 1.0 and f(92160) == 0.9 and f(65536) == 0.75
    assert f(98304) == pytest.approx(0.9158, abs=1e-3) and f(77000) == pytest.approx(0.81, abs=0.005)
    assert f(None) is None and agent.ctx_factor(65536, ()) is None
    for bad in ("131072", "131072:1.5", "x:1", "0:0.5"):
        with pytest.raises(ValueError):
            agent.parse_ctx_curve(bad)


def test_ctx_curve_bad_value_stops_load(monkeypatch):
    monkeypatch.setenv("AGENT_CTX_CURVE", "131072:2")
    with pytest.raises(SystemExit):
        load_config()


def test_ctx_min_and_curve_in_the_score(cfg):
    full = _row("Full-27B-Q4_K_M", ctx=131072)
    ninety = _row("Ninety-27B-Q4_K_M", ctx=92160)
    small = _row("Small-27B-Q4_K_M", ctx=32768)
    finalize([full, ninety, small], cfg)
    assert small["score_agent"] is None and "AGENT_CTX_MIN 64k" in small["score_note"]
    f90 = 0.97 + 0.03 * (92160 - 78848) / (131072 - 78848)                 # default curve: 77k 0.97 .. 128k 1.0
    assert ninety["agent_ctx_factor"] == pytest.approx(f90) and full["agent_ctx_factor"] == 1.0
    # identical quality and speed: the (mild) context factor alone separates them, after the tie zone
    assert ninety["agent_quality"] == pytest.approx(f90 * full["agent_quality"])
    assert ninety["score_agent"] == pytest.approx(f90 * full["score_agent"])
    assert agent.ctx_factor(65536, cfg.agent_ctx_curve) == pytest.approx(0.95)
    # a small GPU sets its own minimum and anchors
    cfg.agent_ctx_min, cfg.agent_ctx_curve = 32768, agent.parse_ctx_curve("65536:1.0,32768:0.8")
    small = _row("Small-27B-Q4_K_M", ctx=32768)
    finalize([small], cfg)
    assert small["score_agent"] is not None and small["agent_ctx_factor"] == pytest.approx(0.8)


def test_unknown_ctx_is_not_penalised(cfg):
    a, b = _row("A-9B-Q4_K_M", ctx=None), _row("B-9B-Q4_K_M", ctx=131072)
    finalize([a, b], cfg)
    assert a["agent_ctx_factor"] is None and a["score_agent"] == pytest.approx(b["score_agent"])


# --------------------------------------------------------------------------- long-ctx over common lengths
def test_longctx_compared_over_common_lengths(cfg):
    """v2.2 averaged only the lengths a model could hold: a 90k model skipped the hardest 128k test and looked
    better than a 128k model that ran it. v2.3 compares everyone on the lengths all ranked models ran."""
    big = _row("Big-27B-Q4_K_M", ctx=131072, lc={"64k": 100.0, "128k": 50.0})
    mid = _row("Mid-27B-Q4_K_M", ctx=92160, lc={"64k": 100.0})
    finalize([big, mid], cfg)
    assert big["agent_longctx_pct"] == mid["agent_longctx_pct"] == 100.0       # 64k only, for both
    assert big["agent_composite_pct"] == pytest.approx(mid["agent_composite_pct"])
    assert "long-ctx 25% (64k)" in big["agent_basis"]
    # Big finds only 50% at 128k: it works like a 64k model (usable 64k, factor 0.95); Mid passed every length it
    # can hold, so its 90k counts (factor ~0.98)
    assert big["agent_ctx_usable"] == 65536 and mid["agent_ctx_usable"] == 92160
    assert mid["score_agent"] > big["score_agent"]
    assert big["longctx_avg_pct"] == 75.0                                      # per-length results stay as they are
    # when every ranked model ran 128k it counts
    big2 = _row("Big2-27B-Q4_K_M", ctx=131072, lc={"64k": 100.0, "128k": 100.0})
    finalize([big, big2], cfg)
    assert "(64k, 128k)" in big["agent_basis"] and big["agent_longctx_pct"] == 75.0 and big2["score_agent"] > big["score_agent"]


def test_usable_context_from_the_longest_length_passed(cfg):
    run = [65536, 131072]
    u = lambda **kw: agent.usable_ctx(_row("X-9B-Q4_K_M", **kw), cfg, run)
    assert u(ctx=131072, lc={"64k": 100.0, "128k": 75.0}) == (131072, None)          # 75% passes
    assert u(ctx=131072, lc={"64k": 100.0, "128k": 62.5})[0] == 65536                 # 5 of 8 does not
    assert u(ctx=92160, lc={"64k": 87.5}) == (92160, None)          # passed every length it holds: 90k counts
    use, why = u(ctx=131072, lc={"64k": 62.5, "128k": 100.0})
    assert use == 0 and "62% at 64k" in why                         # a failed shorter length is not skipped over
    use, why = u(ctx=131072, lc={"128k": 100.0})
    assert use == 0 and "no long-ctx result at 64k" in why          # a missing result is not a pass
    assert agent.usable_ctx(_row("X-9B-Q4_K_M", ctx=131072), cfg, []) == (131072, None)   # no long-ctx in the run
    cfg.agent_ctx_pass_pct = 0
    assert u(ctx=131072, lc={"64k": 10.0}) == (131072, None)        # AGENT_CTX_PASS_PCT=0: the loaded context


def test_failing_64k_means_no_agent_score(cfg):
    """The user's rule: a model must pass 64k (AGENT_CTX_MIN) to get an Agent score; a model with no long-ctx
    result is not ranked, and the others keep the long-ctx part of the composite."""
    a = _row("A-27B-Q4_K_M", lc={"64k": 100.0, "128k": 100.0})
    b = _row("B-27B-Q4_K_M", lc={"64k": 50.0, "128k": 25.0})
    c = _row("C-27B-Q4_K_M", lc=None)                       # long-ctx failed (server down)
    finalize([a, b, c], cfg)
    assert b["score_agent"] is None and "long-ctx 50% at 64k is below AGENT_CTX_PASS_PCT 75%" in b["score_note"]
    assert c["score_agent"] is None and "no long-ctx result at 64k" in c["score_note"]
    assert a["score_agent"] is not None and a["agent_longctx_pct"] == 100.0 and "long-ctx left out" not in a["agent_basis"]


def test_model_without_longctx_drops_the_blend_with_a_note(cfg):
    cfg.agent_ctx_pass_pct = 0                              # usable context = loaded context (v2.3 before 2026-11-10)
    a = _row("A-27B-Q4_K_M", lc={"64k": 100.0, "128k": 100.0})
    b = _row("B-27B-Q4_K_M", lc={"64k": 50.0, "128k": 25.0})
    c = _row("C-27B-Q4_K_M", lc=None)                       # long-ctx failed (server down)
    finalize([a, b, c], cfg)
    assert a["agent_longctx_pct"] is None and "long-ctx left out" in a["agent_basis"] and "C-27B-Q4_K_M" in a["agent_basis"]
    assert agent_notes([a, b, c]) == ["long-ctx left out of the Agent score: no long-ctx result for C-27B-Q4_K_M"]
    assert a["agent_composite_pct"] == pytest.approx(b["agent_composite_pct"])   # evals 70 / tools 30 only
    # without C the blend is back and B's weak retrieval counts
    finalize([a, b], cfg)
    assert a["agent_longctx_pct"] == 100.0 and b["agent_longctx_pct"] == 37.5 and a["score_agent"] > b["score_agent"]


# --------------------------------------------------------------------------- composite, weights, tie zone
def test_composite_weights(cfg):
    w = agent.weights(cfg, True, True, True)
    assert w == pytest.approx({"evals": 0.45, "tools": 0.30, "longctx": 0.25})
    assert agent.weights(cfg, True, True, False) == pytest.approx({"evals": 0.70, "tools": 0.30})
    assert agent.weights(cfg, True, False, False) == pytest.approx({"evals": 1.0})
    assert agent.weights(cfg, False, True, True) == pytest.approx({"tools": 0.3 / 0.55, "longctx": 0.25 / 0.55})
    a = _row("A-9B-Q4_K_M", mmlu=80, he=90, tools=75, lc={"64k": 100.0})
    finalize([a, _row("B-9B-Q4_K_M", tools=90, lc={"64k": 100.0})], cfg)
    assert a["agent_composite_pct"] == pytest.approx(0.45 * 85 + 0.30 * 75 + 0.25 * 100)


def test_tie_zone_from_the_composite_noise(cfg):
    """32 tool cases and 8 long-ctx keys are noisy: the tie zone (1 SE of the best composite) is wider than the
    v2.2 one computed from the total question count."""
    best = _row("Best-9B-Q4_K_M", mmlu=95, he=95, tools=100, lc={"64k": 100.0})
    w = agent.weights(cfg, True, True, True)
    _, se = agent.composite(best, w, ["mmlu_pro", "humaneval"], True, ["64k"], 8)
    old = agent.se_tie(96.0, 43 * 2 + 32)
    assert se > old > 1.0
    # a model just inside that zone ties at full credit; one well below does not
    near = _row("Near-9B-Q4_K_M", mmlu=95, he=95, tools=100 - 100 * se / 0.30 * 0.9 / 100 * 1.0, lc={"64k": 100.0})
    far = _row("Far-9B-Q4_K_M", mmlu=80, he=80, tools=80, lc={"64k": 75.0})
    finalize([best, near, far], cfg)
    assert near["agent_quality"] == pytest.approx(100.0) and far["agent_quality"] < 60


def test_min_toolcall_and_borderline(cfg):
    good = _row("Good-9B-Q4_K_M", tools=100)
    low = _row("Low-9B-Q4_K_M", tools=65.625)                                 # 21/32
    edge = _row("Edge-9B-Q4_K_M", tools=75.0)                                 # 24/32, within +/-17 of 70
    finalize([good, low, edge], cfg)
    assert low["score_agent"] is None and "MIN_TOOLCALL_PCT 70%" in low["score_note"] and low["agent_borderline"]
    assert edge["score_agent"] is not None and edge["agent_borderline"] is True
    assert good["agent_borderline"] is False
    cfg.min_toolcall_pct = 0                                                  # 0 = off
    finalize([good, low], cfg)
    assert low["score_agent"] is not None and low["agent_borderline"] is None


def test_retry_penalty_and_ctx_after_the_tie_zone(cfg):
    a = _row("A-9B-Q4_K_M", fb=0)
    b = _row("B-9B-Q4_K_M", fb=43, ctx=92160)                                 # every MMLU question retried
    finalize([a, b], cfg)
    share = 43 / (43 * 2 + 32)
    f90 = agent.ctx_factor(92160, cfg.agent_ctx_curve)
    assert b["agent_quality"] == pytest.approx(a["agent_quality"] * (1 - 0.10 * share) * f90)


def test_stale_eval_quality_from_older_json_is_cleared(cfg):
    """--compare / rescoring reads rows back from JSON: a stored eval_quality must not keep an incomplete model in
    the Agent pool (v2.2.1 crashed on the v2.0 run; a silent rank would be worse)."""
    a, b = _row("A-9B-Q4_K_M"), _row("B-9B-Q4_K_M", he=None)
    b.update(eval_quality=88.0, eval_humaneval_rows=None)
    a["eval_humaneval_rows"] = 43
    finalize([a, b], cfg)
    assert b["eval_quality"] is None and b["score_agent"] is None and b["eval_incomplete"]


def test_without_evals_the_proxy_gets_the_context_rules(cfg):
    big = blank("m-27B-Q4_K_M.gguf")
    big.update(tech_tps=50.0, logic_tps=50.0, vram_gb=12.0, vram_total_gb=16.0, ctx_size=92160)
    small = blank("m-14B-Q4_K_M.gguf")
    small.update(tech_tps=50.0, logic_tps=50.0, vram_gb=12.0, vram_total_gb=16.0, ctx_size=32768)
    finalize([big, small], cfg)
    assert small["score_agent"] is None and "AGENT_CTX_MIN" in small["score_note"]
    assert big["agent_quality"] == pytest.approx(100 * agent.ctx_factor(92160, cfg.agent_ctx_curve)) and "proxy" in big["score_note"]


# --------------------------------------------------------------------------- speed factor
def test_speed_factor_weights(cfg):
    r = {"agent_ttft_s": 15.0, "eval_avg_sec": 60.0, "avg_tps": 200.0}
    assert speed_factor(r, cfg) == pytest.approx(0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 1.0)
    r["prefill_depth_tps"] = 20000 / cfg.agent_ttft_ref                   # the at-depth credit halves here
    assert speed_factor(r, cfg) == pytest.approx(0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 1.0)
    r["prefill_depth_tps"] = 1e9                                           # instant: at-depth part 1.0, weight 0.6
    assert speed_factor(r, cfg) == pytest.approx(0.5 * (0.6 * 1 + 0.4 * 0.5) + 0.15 + 0.2)
    only = {"prefill_32k_tps": 4000.0, "avg_tps": 200.0}                   # neither agent test: longest prefill test
    assert speed_factor(only, cfg) == pytest.approx(1.0)


# --------------------------------------------------------------------------- tool calling in full at every stage
def test_tool_calling_runs_in_full_at_every_stage(cfg):
    from llmbench.evals.datasets import prepare_eval_data
    from llmbench.evals.runner import _stage_ids
    cfg.eval_profile, cfg.eval_funnel_final, cfg.eval_datasets = "funnel", "moderate", ("toolcall",)
    assert rows_for("toolcall", 11) is None and rows_for("mmlu_pro", 11) == 11
    data = prepare_eval_data(cfg)
    assert len(data["toolcall"]) == 32
    ids = _stage_ids(cfg, data, funnel_stages(cfg))["toolcall"]
    assert [len(s) for s in ids] == [32, 32, 32]


# --------------------------------------------------------------------------- the funnel gate ranks on the composite
def test_gate_uses_the_agent_composite(cfg):
    keys = ["mmlu_pro", "toolcall"]
    # v2.2 gave tools half the weight here (mean of 2 datasets); the composite gives them 0.30
    a = _row("A-9B-Q4_K_M", mmlu=100, he=None, tools=71.875, n=96)
    b = _row("B-9B-Q4_K_M", mmlu=85, he=None, tools=100, n=96)
    finalize([a, b], cfg)
    _, rep_old, _ = funnel.gate([a, b], keys, leaders=1)
    _, rep_new, _ = funnel.gate([a, b], keys, leaders=1, cfg=cfg)
    assert rep_old["B-9B-Q4_K_M"]["score"] > rep_old["A-9B-Q4_K_M"]["score"]
    assert rep_new["A-9B-Q4_K_M"]["score"] > rep_new["B-9B-Q4_K_M"]["score"]
    assert rep_new["A-9B-Q4_K_M"]["score"] == pytest.approx(a["agent_composite_pct"])
    keys = ["mmlu_pro", "humaneval", "toolcall"]
    a = _row("A-9B-Q4_K_M", lc={"64k": 100.0}, n=96)
    b = _row("B-9B-Q4_K_M", lc={"64k": 100.0}, n=96)
    # a model that cannot be an agent (ctx below the minimum, no 64k result) is still gated, on the parts it has
    c = _row("C-9B-Q4_K_M", ctx=32768, lc=None, n=96)
    keep, rep, _ = funnel.gate([a, b, c], keys, leaders=3, cfg=cfg)
    assert rep["C-9B-Q4_K_M"]["score"] is not None and "C-9B-Q4_K_M" in keep


# --------------------------------------------------------------------------- --compare guard
def test_compare_refuses_pre_and_post_v23(cfg):
    assert scoring_generation("2.2.1") == scoring_generation("2.1.0") == "pre-2.3"
    assert scoring_generation("2.3.0") == scoring_generation("2.10.1") == scoring_generation("3.0") == "2.3+"
    ev = {"datasets": {"mmlu_pro": {"rows": 43}}, "seed": 42}
    old, new = {"tool": {"version": "2.2.1"}, "evaluation": ev}, {"tool": {"version": "2.3.0"}, "evaluation": ev}
    problems, _ = check([old, new], ["a", "b"])
    assert any("before and after v2.3" in p for p in problems)
    problems, _ = check([new, json.loads(json.dumps(new))], ["a", "b"])
    assert not problems
