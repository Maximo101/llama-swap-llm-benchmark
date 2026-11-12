"""v2.3.3: hard rules (no Agent score) decided outside the funnel, the reducible-error gate, the three rank states."""
import pytest

from llmbench.config import load_config
from llmbench.evals import funnel
from llmbench.reporting.json_out import _model_entry
from llmbench.scoring import finalize

from test_v23_agent import _row, cfg  # noqa: F401  (the fixture and the row builder of the v2.3 tests)

KEYS = ["mmlu_pro", "humaneval", "toolcall"]
LC = {"64k": 100.0, "128k": 100.0}


def _good(name, mmlu, n=43, **kw):
    return _row(name, mmlu=mmlu, he=mmlu, tools=100.0, lc=dict(LC), n=n, **kw)


def _flint(n=11):
    """100% on every eval, then 64k 50% and 128k 0% on long-context: the real 'flint' case."""
    return _row("flint-9B-Q4_K_M", mmlu=100.0, he=100.0, tools=90.6, lc={"64k": 50.0, "128k": 0.0}, n=n)


def test_long_context_failure_is_a_hard_rule_not_a_screen(cfg):
    rows = [_good("A-9B-Q4_K_M", 90, n=11), _good("B-9B-Q4_K_M", 85, n=11), _flint()]
    keep, rep, bar = funnel.gate(rows, KEYS, leaders=2, cfg=cfg)
    f = rep["flint-9B-Q4_K_M"]
    assert f["state"] == "no_agent" and "flint-9B-Q4_K_M" not in keep
    assert "hard rule" in f["why"] and "long-ctx" in f["why"].lower().replace("long-context", "long-ctx")
    assert f.get("score") is None                         # never compared: no estimate, no margin of error
    # and its 100% evals did not raise the leaders' bar
    _, rep2, bar2 = funnel.gate(rows[:2], KEYS, leaders=2, cfg=cfg)
    assert bar == pytest.approx(bar2)


def test_hard_fail_continue_keeps_the_model_but_still_not_in_the_bar(cfg):
    cfg.eval_hard_fail = "continue"
    rows = [_good("A-9B-Q4_K_M", 90, n=11), _good("B-9B-Q4_K_M", 85, n=11), _flint()]
    keep, rep, bar = funnel.gate(rows, KEYS, leaders=2, cfg=cfg)
    assert "flint-9B-Q4_K_M" in keep and rep["flint-9B-Q4_K_M"]["state"] == "no_agent"
    assert bar == pytest.approx(funnel.gate(rows[:2], KEYS, leaders=2, cfg=cfg)[2])


def test_tool_calling_below_the_minimum_is_a_hard_rule(cfg):
    weak = _row("T-9B-Q4_K_M", mmlu=95, he=95, tools=40.0, lc=dict(LC), n=11)
    keep, rep, _ = funnel.gate([_good("A-9B-Q4_K_M", 90, n=11), weak], KEYS, leaders=1, cfg=cfg)
    assert rep["T-9B-Q4_K_M"]["state"] == "no_agent" and "tool calling" in rep["T-9B-Q4_K_M"]["why"]


def test_gate_uses_the_reducible_error_and_reports_decimal_gaps(cfg):
    a, b = _good("A-9B-Q4_K_M", 90), _good("B-9B-Q4_K_M", 60)
    keep, rep, bar = funnel.gate([a, b], KEYS, leaders=1, cfg=cfg)
    ra, rb = rep["A-9B-Q4_K_M"], rep["B-9B-Q4_K_M"]
    assert ra["se_red"] < ra["se"]                        # tools / long-context error is not counted at a gate
    assert ra["state"] == "keep" and rb["state"] == "screened" and "B-9B-Q4_K_M" not in keep
    assert rb["gap"] < 0 and rb["gap"] == pytest.approx(rb["ub"] - bar)
    assert f"{-rb['gap']:.2f} below the leaders' bar {bar:.2f}" in rb["why"]


def test_close_models_stay_and_the_margin_is_honoured(cfg):
    a, b = _good("A-9B-Q4_K_M", 90, n=11), _good("B-9B-Q4_K_M", 84, n=11)
    keep, rep, _ = funnel.gate([a, b], KEYS, leaders=1, cfg=cfg)
    assert "B-9B-Q4_K_M" in keep                          # within the noise of 11 questions
    c, d = _good("C-9B-Q4_K_M", 90), _good("D-9B-Q4_K_M", 80)
    cfg.eval_funnel_margin = 0.0
    _, rep0, _ = funnel.gate([c, d], KEYS, leaders=1, cfg=cfg)
    cfg.eval_funnel_margin = 10.0
    keep10, rep10, _ = funnel.gate([c, d], KEYS, leaders=1, cfg=cfg)
    assert rep10["D-9B-Q4_K_M"]["ub"] == pytest.approx(rep0["D-9B-Q4_K_M"]["ub"] + 10.0)
    assert "D-9B-Q4_K_M" in keep10


def test_never_fewer_than_the_leaders(cfg):
    rows = [_good(f"M{i}-9B-Q4_K_M", 90 - 12 * i) for i in range(4)]
    keep, _, _ = funnel.gate(rows, KEYS, leaders=3, cfg=cfg)
    assert len(keep) >= 3


def test_three_rank_states_after_finalize(cfg):
    ranked = _good("A-9B-Q4_K_M", 90)
    screened = _good("S-9B-Q4_K_M", 60)
    screened["eval_screened"] = "screened out at quick: upper bound 60.00 below the bar 80.00"
    flint = _flint(n=43)
    flint["eval_screened"] = "screened out at quick: upper bound 91.6 < leaders' bar 91.6"   # a v2.3.2-style result
    finalize([ranked, screened, flint], cfg)
    assert ranked["rank_state"] == "ranked" and ranked["score_agent"] is not None
    assert screened["rank_state"] == "screened" and "screened out" in screened["rank_reason"]
    assert flint["rank_state"] == "no_agent" and flint["score_note"].startswith("no Agent score")
    assert screened["score_agent"] is None and flint["score_agent"] is None


def test_rank_state_is_in_the_json_scores_group(cfg):
    r = _good("A-9B-Q4_K_M", 90)
    finalize([r], cfg)
    sc = _model_entry(r)["scores"]
    assert sc["rank_state"] == "ranked" and "rank_reason" in sc


def test_hard_fail_setting_is_validated(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("EVAL_HARD_FAIL", raising=False)
    c = load_config()
    assert c.eval_hard_fail == "stop" and c.eval_funnel_margin == 3.0
    monkeypatch.setenv("EVAL_HARD_FAIL", "maybe")
    with pytest.raises(SystemExit):
        load_config()
    monkeypatch.setenv("EVAL_HARD_FAIL", "Continue")
    monkeypatch.setenv("EVAL_FUNNEL_MARGIN", "5")
    c = load_config()
    assert c.eval_hard_fail == "continue" and c.eval_funnel_margin == 5.0
