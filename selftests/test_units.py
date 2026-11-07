"""Unit tests for the pure / isolated parts: telemetry (against a fake NVML), parsing, math grading, sandbox, JSON."""
import json
import math
import sys
import time
import types

import pytest

from llmbench.common import TimeSeries
from llmbench.config import load_config
from llmbench.evals import answers, mathcheck, sandbox
from llmbench.reporting.json_out import _clean, build_report
from llmbench.scoring import blank, finalize, parse_active, parse_params, parse_quant
from llmbench.telemetry import gpu as gpumod


# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg(monkeypatch, tmp_path):
    for k in list(__import__("os").environ):
        if k.startswith(("EVAL_", "GPU_", "LLAMA_", "RUN_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    c = load_config()
    c.eval_code_sandbox = "subprocess"
    return c


def fake_nvml(devices=2, unsupported_power=(), init_error=False):
    m = types.ModuleType("pynvml")
    counter = {"n": 0}
    GiB = 1024 ** 3

    class Err(Exception):
        pass
    m.NVMLError = Err
    m.NVML_TEMPERATURE_GPU = 0

    def init():
        if init_error:
            raise Err("driver not loaded")
    m.nvmlInit, m.nvmlShutdown = init, lambda: None
    m.nvmlDeviceGetCount = lambda: devices
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetHandleByUUID = lambda u: (_ for _ in ()).throw(Err("no uuid"))
    m.nvmlDeviceGetHandleByPciBusId = lambda u: (_ for _ in ()).throw(Err("no pci"))

    def power(h):
        if h in unsupported_power:
            raise Err("not supported")
        return 100_000 * (h + 1)
    m.nvmlDeviceGetPowerUsage = power
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=(h + 1) * GiB, total=8 * GiB)
    m.nvmlDeviceGetTemperature = lambda h, kind: 50 + h
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=10 * (h + 1))

    def energy(h):
        counter["n"] += 1000
        return counter["n"]
    m.nvmlDeviceGetTotalEnergyConsumption = energy
    m.nvmlDeviceGetName = lambda h: b"NVIDIA GeForce RTX 5060 Ti"
    return m


# ------------------------------- telemetry ---------------------------------
def test_timeseries_window_and_fallback():
    ts = TimeSeries()
    assert ts.window(0, 10) == []
    for t in (1.0, 2.0, 3.0, 4.0):
        ts.append(t * 10, t)
    assert ts.window(2.0, 3.0) == [20.0, 30.0]
    assert ts.window(5.0, 6.0) == [40.0]            # nothing inside -> latest sample, so short requests still get a reading
    assert len(ts) == 4


def test_nvml_aggregates_devices(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(2))
    cfg.gpu_poll_ms = 100
    mon = gpumod.NvmlGpuMonitor(cfg)
    assert mon.start() and mon.ok and mon.backend == "nvml"
    time.sleep(0.35)
    s = mon.window(time.time() - 0.3, time.time())[-1]
    assert s["gpu_w"] == pytest.approx(300.0)        # 100 W + 200 W summed
    assert s["vram_gb"] == pytest.approx(3.0) and s["vram_total_gb"] == pytest.approx(16.0)
    assert s["gpu_temp"] == 51 and s["gpu_util"] == 20      # max across GPUs
    assert len(mon.series) >= 3                      # sampled on its own thread
    mon.stop()


def test_nvml_gpu_index_selects_one_gpu(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(2))
    cfg.gpu_index = "1"
    mon = gpumod.NvmlGpuMonitor(cfg)
    assert mon.start()
    assert mon.window(0, time.time() + 1)[-1]["gpu_w"] == pytest.approx(200.0)
    mon.stop()


def test_nvml_unsupported_power_does_not_break(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(1, unsupported_power=(0,)))
    mon = gpumod.NvmlGpuMonitor(cfg)
    assert mon.start()
    s = mon.window(0, time.time() + 1)[-1]
    assert s["gpu_w"] is None and s["vram_gb"] == pytest.approx(1.0)
    mon.stop()


def test_nvml_energy_counter_is_opt_in(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(2))
    cfg.gpu_energy_counter = False
    off = gpumod.NvmlGpuMonitor(cfg)
    assert off.start() and off.energy_mj() is None
    off.stop()
    cfg.gpu_energy_counter = True
    on = gpumod.NvmlGpuMonitor(cfg)
    assert on.start()
    a, b = on.energy_mj(), on.energy_mj()
    assert a is not None and b > a                   # monotonic cumulative counter
    on.stop()


def test_backend_selection_and_fallbacks(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(1, init_error=True))
    monkeypatch.setattr(gpumod.SmiGpuMonitor, "start", lambda self: False)      # no nvidia-smi either
    cfg.gpu_backend = "auto"
    mon = gpumod.open_gpu_monitor(cfg)
    assert not mon.ok and mon.backend == "none" and mon.window(0, 1) == []     # graceful: blank GPU columns
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(1))
    assert gpumod.open_gpu_monitor(cfg).backend == "nvml"
    monkeypatch.setattr(gpumod.SmiGpuMonitor, "start", lambda self: True)
    cfg.gpu_backend = "smi"
    assert gpumod.open_gpu_monitor(cfg).backend == "smi"


def test_gpu_inventory_uses_nvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(2))
    inv = gpumod.gpu_inventory()
    assert inv == [("NVIDIA GeForce RTX 5060 Ti", 8.0)] * 2


# ------------------------------ config / misc ------------------------------
def test_fallback_dotenv_strips_inline_comments(monkeypatch, tmp_path):
    import builtins
    import llmbench.config as config
    (tmp_path / ".env").write_text('RUNS_PER_PROMPT=3            # median is reported\nCURRENCY_SYMBOL="€"  # q\n'
                                   "HF_TOKEN=\nTOKEN_X=abc#notacomment\n")
    monkeypatch.setattr(config, "SCRIPT_DIR", tmp_path)
    for k in ("RUNS_PER_PROMPT", "CURRENCY_SYMBOL", "TOKEN_X"):
        monkeypatch.delenv(k, raising=False)
    real_import = builtins.__import__

    def no_dotenv(name, *a, **k):
        if name == "dotenv":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    config._load_dotenv()
    import os
    assert os.environ["RUNS_PER_PROMPT"] == "3" and os.environ["CURRENCY_SYMBOL"] == "€"
    assert os.environ["TOKEN_X"] == "abc#notacomment"       # '#' not preceded by whitespace is part of the value


def test_model_name_parsing():
    n = "Qwen3-30B-A3B-IQ3_XXS.gguf"
    assert parse_params(n) == 30 and parse_active(n) == 3 and parse_quant(n) == "IQ3_XXS"


# --------------------------------- answers ---------------------------------
@pytest.mark.parametrize("text,letters,want", [
    ("blah\nAnswer: C", "ABCD", "C"), ("The answer is (B).", "ABCD", "B"), ("\\boxed{D}", "ABCD", "D"),
    ("<think>Answer: A</think>\nAnswer: B", "ABCD", "B"), ("Answer: Z", "ABCD", None), ("no idea", "ABCD", None)])
def test_extract_choice(text, letters, want):
    assert answers.extract_choice(text, letters) == want


@pytest.mark.parametrize("text,gold,numeric_fallback,want", [
    ("so\nAnswer: 18", "18", True, True),
    ("Answer: $\\boxed{1,000}$.", "1000", True, True),
    ("Answer: \\dfrac{1}{2}", "0.5", False, True),
    ("Answer: 3/4", "\\frac{3}{4}", False, True),
    ("Answer: x = 5", "5", False, True),
    ("Answer: 12 cm", "12", False, True),
    ("Answer: (1, 2)", "\\left(1, 2\\right)", False, True),
    ("Answer: 7", "8", False, False),
    ("Answer: 0.30000001", "0.3", False, True),
    ("<think>Answer: 99</think>The result is 42", "42", True, True),
    ("We compute 3 + 4 = 7 and then 7 * 2", "14", False, False),   # no explicit answer and no fallback -> not graded as right
])
def test_math_grading(text, gold, numeric_fallback, want):
    pred = mathcheck.extract_math_answer(text, numeric_fallback=numeric_fallback)
    assert (pred is not None and mathcheck.math_equal(pred, gold)) is want


def test_truncated_reply_never_gets_last_number_fallback(cfg):
    from llmbench.evals.grading import grade_item
    item = {"gold": "14"}
    ok, pred, why = grade_item(cfg, "gsm8k", item, "first 7 then 14", finish="length")
    assert not ok and pred is None
    ok, pred, why = grade_item(cfg, "gsm8k", item, "first 7 then 14", finish="stop")
    assert ok


# --------------------------------- sandbox ---------------------------------
def test_sandbox_selftest_all_pass(cfg):
    sandbox.resolve_sandbox(cfg, run_selftest=False)
    results = sandbox.selftest(cfg)
    failed = [d for d, ok in results if not ok]
    assert not failed, failed


def test_sandbox_modes(cfg):
    sandbox.resolve_sandbox(cfg, run_selftest=False)
    run = lambda *a: sandbox.run_in_sandbox(cfg, *a)
    assert run("assert 2 * 2 == 4\n", "script", [], 5) == (True, "")
    ok, why = run("import os\nos._exit(0)\n", "script", [], 5)          # quits before the end-of-program sentinel
    assert not ok and "exited before" in why
    ok, why = run("while True: pass\n", "script", [], 1)
    assert not ok and "timeout" in why
    t = [{"input": "4\n", "output": "8"}, {"input": "5\n", "output": "10"}]
    assert run("print(int(input()) * 2)\n", "stdin", t, 5)[0]
    ok, why = run("print(int(input()) * 3)\n", "stdin", t, 5)
    assert not ok and "wrong answer on test 0" in why
    assert run("import json\nprint(json.dumps([1.0, 2]))\n", "call", [{"input": "", "output": "[1, 2.0]"}], 5)[0]


def test_sandbox_does_not_leak_environment(cfg, monkeypatch):
    monkeypatch.setenv("LLAMA_SWAP_API_KEY", "sekrit-token")
    sandbox.resolve_sandbox(cfg, run_selftest=False)
    code = "import os\nassert 'sekrit-token' not in str(dict(os.environ)), 'leaked'\n"
    assert sandbox.run_in_sandbox(cfg, code, "script", [], 5) == (True, "")


# ----------------------------------- json ----------------------------------
def test_json_clean_and_strict(cfg):
    assert _clean({"a": float("nan"), "b": [float("inf"), 1.5], "c": "x"}) == {"a": None, "b": [None, 1.5], "c": "x"}
    r = blank("m-7B-Q4_K_M.gguf")
    r.update(tech_tps=10.0, logic_tps=float("nan"), eval_gsm8k_pct=81.0, eval_gsm8k_n=10, eval_gsm8k_conc=4, mystery_field=7)
    cfg.api_key, cfg.ha_token = "secret-api-key", "ha-token-123"
    finalize([r], cfg)
    doc = _clean(build_report([r], cfg, {"stamp": "x", "started": time.time() - 5, "finished": time.time()}))
    text = json.dumps(doc, allow_nan=False)                 # raises on NaN/Infinity
    assert "secret-api-key" not in text and "ha-token-123" not in text
    m = doc["models"][0]
    assert m["evaluation"]["datasets"]["gsm8k"]["concurrency"] == 4
    assert m["other"] == {"mystery_field": 7}               # unmapped fields are kept, not dropped
    assert doc["schema_version"] == 1 and doc["run"]["models_ok"] == 1


# ------------------------------- agent score -------------------------------
def _agent_row(name, pct, sec, tps, vram_pct, n=72):
    r = blank(name)
    r.update(tech_tps=tps, logic_tps=tps, vram_gb=16.0 * vram_pct / 100, vram_total_gb=16.0,
             eval_mmlu_pro_pct=pct, eval_mmlu_pro_n=n, eval_mmlu_pro_sec=sec)
    return r


def test_agent_score_is_quality_first(cfg):
    rows = [_agent_row("best-7B-Q4_K_M.gguf", 94.0, 50.0, 80.0, 70),        # most accurate, slow
            _agent_row("fast-7B-Q4_K_M.gguf", 91.0, 15.0, 190.0, 97),       # ~1 std. error behind, fastest, VRAM nearly full
            _agent_row("quick-7B-Q4_K_M.gguf", 70.0, 5.0, 200.0, 40),       # far weaker but quickest and leanest
            _agent_row("slow-7B-Q4_K_M.gguf", 91.0, 55.0, 70.0, 70)]        # same accuracy as "fast", slower
    finalize(rows, cfg)
    s = {r["model"].split("-")[0]: r["score_agent"] for r in rows}
    assert s["fast"] > s["slow"]                      # equal accuracy: speed decides
    assert s["fast"] > 0.7 * s["best"]                # a full VRAM no longer wrecks the score
    assert s["quick"] < min(s["best"], s["fast"], s["slow"])   # speed never lifts a clearly weaker model
    assert rows[0]["agent_quality"] == 100.0          # the best model sits at full accuracy credit
    assert all(0 <= r["score_agent"] <= 100 for r in rows)


def test_agent_score_without_evals_uses_params_proxy(cfg):
    # v2.2: the proxy is on the same 0-100 scale as the eval-based score (relative to the best model of the run)
    r = blank("m-7B-Q4_K_M.gguf")
    r.update(tech_tps=50.0, logic_tps=50.0, vram_gb=8.0, vram_total_gb=16.0)
    s = blank("m-14B-Q4_K_M.gguf")
    s.update(tech_tps=50.0, logic_tps=50.0, vram_gb=8.0, vram_total_gb=16.0)
    finalize([r, s], cfg)
    assert s["agent_quality"] == 100.0 and 0 < r["agent_quality"] < 100 and 0 < r["score_agent"] < s["score_agent"]
    assert "proxy" in r["score_note"]


# --------------------------- VRAM sweet spot -------------------------------
def test_vram_fit_shape(cfg):
    from llmbench.scoring import vram_fit
    assert vram_fit(None, cfg) is None
    assert vram_fit(80, cfg) == vram_fit(85, cfg) == vram_fit(90, cfg) == 1.0       # default band 80-90%
    assert 0.5 < vram_fit(60, cfg) < 1.0                                           # under-use is a mild loss
    assert vram_fit(94, cfg) == 0.5 and vram_fit(97.7, cfg) < 0.25                 # over 94% is "too tight"
    assert vram_fit(100, cfg) == 0.0
    vals = [vram_fit(p, cfg) for p in (91, 92, 93, 94, 96, 99)]
    assert vals == sorted(vals, reverse=True)                                      # monotonic above the band


def test_sweet_spot_prefers_headroom(cfg):
    roomy = _agent_row("roomy-7B-Q4_K_M.gguf", 90.0, 20.0, 120.0, 86)
    tight = _agent_row("tight-7B-Q4_K_M.gguf", 90.0, 18.0, 150.0, 97)             # a bit faster, but VRAM nearly full
    finalize([roomy, tight], cfg)
    assert tight["score_agent"] > roomy["score_agent"]                            # capability alone prefers "tight"
    assert roomy["score_sweet"] > tight["score_sweet"]                            # the sweet spot prefers headroom
    assert roomy["vram_fit"] == 1.0 and tight["vram_fit"] < 0.3
