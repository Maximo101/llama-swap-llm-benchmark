"""EVAL_THINK_BUDGET=auto (the default): a share of each model's own context, so models with a big context get room to
think and every model keeps the same share. A number or 0 still fixes it for every model."""
import types

import pytest

import fake_datasets
import mock_server
from llmbench import config
from llmbench.config import AUTO_BUDGET, auto_think_budget, load_config, think_budget, think_budget_label
from llmbench.evals import runner
from test_pipeline import ev, run_cli


def cfg(budget=AUTO_BUDGET, pct=25.0, per=None, mn=4096, mx=60.0):
    return types.SimpleNamespace(eval_think_budget=budget, eval_think_budget_pct=pct, eval_think_budgets=per or {},
                                 eval_think_budget_min=mn, eval_think_budget_max_pct=mx)


def test_auto_budget_is_a_share_of_the_context_rounded_down_to_1024():
    assert auto_think_budget(cfg(), 131072) == 32768            # 128k: the 32,768 asked for
    assert auto_think_budget(cfg(), 65536) == 16384             # 64k: what the fixed default used to be
    assert auto_think_budget(cfg(), 90112) == 22528
    assert auto_think_budget(cfg(), 100000) == 24576            # 25,000 rounded down to a multiple of 1,024
    assert auto_think_budget(cfg(pct=20.0), 131072) == 25600
    assert auto_think_budget(cfg(), 8192) == config.THINK_AUTO_FLOOR      # never starved
    assert auto_think_budget(cfg(), None) == config.THINK_AUTO_FALLBACK == 16384      # context unknown
    assert auto_think_budget(cfg(), "n/a") == 16384


def test_floor_and_ceiling_bound_the_auto_budget():
    assert auto_think_budget(cfg(), 8192) == 4096               # 25% is 2,048: the floor lifts it
    assert auto_think_budget(cfg(), 16384) == 4096              # 25% is exactly the floor
    assert auto_think_budget(cfg(), 4096) == 2048               # ceiling 60% = 2,457 wins over the floor: room is left to answer
    assert auto_think_budget(cfg(), 131072) == 32768            # in between: the share
    assert auto_think_budget(cfg(pct=80.0, mx=90.0), 131072) == 104448    # a user who raises the share and the ceiling
    assert auto_think_budget(cfg(mn=8192), 16384) == 8192       # a raised floor
    assert auto_think_budget(cfg(mn=100000), 131072) == 77824   # a floor above the ceiling is held to 60% (78,643 -> 76 x 1,024)
    assert auto_think_budget(cfg(pct=50.0, mx=40.0), 131072) == 52224       # ceiling 40% beats the 50% share
    for ctx in (2048, 4096, 8192, 131072, 1048576):             # never more than the ceiling, never zero
        assert 1 <= auto_think_budget(cfg(), ctx) <= ctx * 0.6
    assert config.think_budget_basis(cfg(), 131072) == "25% of 131,072 ctx"
    assert config.think_budget_basis(cfg(), 8192).startswith("floor 4,096")
    assert config.think_budget_basis(cfg(), 4096).startswith("ceiling 60%")


def test_floor_and_ceiling_settings_are_validated(monkeypatch):
    c = _load(monkeypatch)
    assert c.eval_think_budget_min == 4096 and c.eval_think_budget_max_pct == 60.0
    c = _load(monkeypatch, EVAL_THINK_BUDGET_MIN="8192", EVAL_THINK_BUDGET_MAX_PCT="50", EVAL_THINK_BUDGET_PCT="30")
    assert (c.eval_think_budget_min, c.eval_think_budget_max_pct, c.eval_think_budget_pct) == (8192, 50.0, 30.0)
    for var, bad in (("EVAL_THINK_BUDGET_MIN", "100"), ("EVAL_THINK_BUDGET_MAX_PCT", "95"), ("EVAL_THINK_BUDGET_MAX_PCT", "5")):
        with pytest.raises(SystemExit, match=var):
            _load(monkeypatch, **{var: bad})
    with pytest.raises(SystemExit, match="cannot be above"):
        _load(monkeypatch, EVAL_THINK_BUDGET_PCT="70")           # share 70 > the default ceiling 60


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="Node.js not installed")
def test_settings_page_tooltip_matches_python(tmp_path):
    """The settings page works the budget out in JS: it must agree with auto_think_budget for every size it lists."""
    import json, re, subprocess
    from pathlib import Path
    from llmbench.ui import server
    html = (Path(server.__file__).with_name("settings.html")).read_text(encoding="utf-8")
    lines = [re.search(r"const %s = .*" % n, html).group(0) for n in ("isCtxKey", "CTX_TIP", "isThinkKey", "thinkFor")]
    tip = re.search(r"const thinkTip = .*?\n(?=const tipOf)", html, re.S).group(0)
    tips = re.search(r"const tipOf = .*", html).group(0)
    sizes = [4096, 8192, 16384, 32768, 65536, 98304, 131072, 262144, 524288, 1048576]
    js = "\n".join(lines) + "\n" + tip + tips + f"""
var V = {{EVAL_THINK_BUDGET_PCT: '30', EVAL_THINK_BUDGET_MIN: '8192', EVAL_THINK_BUDGET_MAX_PCT: '50'}};
console.log(JSON.stringify({{keys: ['EVAL_THINK_BUDGET','EVAL_THINK_BUDGET_PCT','EVAL_THINK_BUDGET_MIN','EVAL_THINK_BUDGET_MAX_PCT','EVAL_THINK_BUDGET_GSM8K','EVAL_THINK_BUDGET_PCTX'].map(k => !!tipOf(k)),
  f: {json.dumps(sizes)}.map(c => thinkFor(c, 30, 8192, 50)), tip: tipOf('EVAL_THINK_BUDGET_PCT')}}))""".replace("\\n", "\n")
    f = tmp_path / "t.js"; f.write_text(js)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["keys"] == [True, True, True, True, False, False]
    assert out["f"] == [auto_think_budget(cfg(pct=30.0, mn=8192, mx=50.0), c) for c in sizes]
    assert "131072 (128k) -> 38912" in out["tip"] and "30%" in out["tip"] and "8192" in out["tip"]


def test_a_number_or_zero_is_the_same_for_every_model_and_a_dataset_override_wins():
    assert think_budget(cfg(16384), "gsm8k", 131072) == 16384 and think_budget(cfg(16384), "gsm8k", 8192) == 16384
    assert think_budget(cfg(0), "gsm8k", 131072) == 0                                  # unlimited
    c = cfg(AUTO_BUDGET, per={"gsm8k": 2048, "gpqa": AUTO_BUDGET, "math": 0})
    assert think_budget(c, "gsm8k", 131072) == 2048 and think_budget(c, "math", 131072) == 0
    assert think_budget(c, "gpqa", 131072) == 32768 and think_budget(c, "mbpp", 65536) == 16384
    assert think_budget_label(AUTO_BUDGET, cfg()) == "auto (25% of each model's context)"
    assert think_budget_label(0) == "unlimited" and think_budget_label(16384) == "16,384 tokens"


def _load(monkeypatch, **env):
    for k in list(__import__("os").environ):
        if k.startswith(("EVAL_THINK", "MAX_GPU", "GPU_TEMP")):
            monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "ENV_FILE", config.SCRIPT_DIR / "no-such.env")
    return load_config()


def test_settings_default_to_auto_at_25_percent_and_are_validated(monkeypatch):
    c = _load(monkeypatch)
    assert c.eval_think_budget == AUTO_BUDGET and c.eval_think_budget_pct == 25.0 and c.max_gpu_temp == 85.0 and c.eval_sanity_pct == 0.0
    assert _load(monkeypatch, EVAL_THINK_BUDGET="auto").eval_think_budget == AUTO_BUDGET
    assert _load(monkeypatch, EVAL_THINK_BUDGET="16384").eval_think_budget == 16384
    assert _load(monkeypatch, EVAL_THINK_BUDGET="unlimited").eval_think_budget == 0
    assert _load(monkeypatch, EVAL_THINK_BUDGET_PCT="20").eval_think_budget_pct == 20.0
    assert _load(monkeypatch, EVAL_THINK_BUDGET_GSM8K="auto").eval_think_budgets == {"gsm8k": AUTO_BUDGET}
    for bad in ("2", "90"):
        with pytest.raises(SystemExit, match="EVAL_THINK_BUDGET_PCT"):
            _load(monkeypatch, EVAL_THINK_BUDGET_PCT=bad)
    with pytest.raises(SystemExit, match="MAX_GPU_TEMP"):
        _load(monkeypatch, MAX_GPU_TEMP="20")
    with pytest.raises(SystemExit, match="GPU_TEMP_RESUME_DELTA"):
        _load(monkeypatch, MAX_GPU_TEMP="85", GPU_TEMP_RESUME_DELTA="70")
    c = _load(monkeypatch, MAX_GPU_TEMP="85")
    assert c.max_gpu_temp == 85.0 and c.gpu_temp_resume_delta == 10.0 and c.gpu_temp_max_restarts == 3


def test_a_bigger_context_means_a_bigger_request_and_fewer_in_flight():
    """The KV pool is shared: with a budget that grows with the context, fewer requests fit at once. Say so."""
    item = {"prompt": "x" * 3000}
    c = types.SimpleNamespace(eval_think_budget=AUTO_BUDGET, eval_think_budget_pct=25.0, eval_think_budgets={},
                              eval_max_tokens_qa=None, eval_max_tokens_code=None, eval_concurrency=8)
    small = runner.per_request_tokens(c, "gsm8k", item, 65536)
    big = runner.per_request_tokens(c, "gsm8k", item, 131072)
    assert big > small and 21000 < small < 22000 and 37000 < big < 39000     # 131072 // 37928 = 3 in flight (6 before)


@pytest.fixture(scope="module")
def server():
    s = mock_server.MockServer().start()
    yield s
    s.stop()


def test_a_default_run_uses_auto_and_says_what_each_model_got(monkeypatch, tmp_path, server, capsys):
    monkeypatch.delenv("EVAL_THINK_BUDGET", raising=False)
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gsm8k"])
    assert code == 0
    e = doc["evaluation"]
    assert e["think_budget"] == "auto" and e["think_budget_pct"] == 25.0
    out = capsys.readouterr().out
    assert "thinking budget auto (25% of each model's context)" in out
    assert "    thinking budget 4,096 tokens (floor 4,096; 25% of 8,192 ctx is lower)" in out      # the per-model line (mock ctx 8192)
    assert e["think_budget_min"] == 4096 and e["think_budget_max_pct"] == 60.0
    assert ev(doc["models"][0], "gsm8k")["n"] > 0


def test_a_fixed_budget_is_still_exported_as_a_number(monkeypatch, tmp_path, server):
    code, _, doc = run_cli(monkeypatch, tmp_path, server, [], datasets=["gsm8k"], EVAL_THINK_BUDGET="1500")
    assert doc["evaluation"]["think_budget"] == 1500 and doc["evaluation"]["think_budget_pct"] is None
