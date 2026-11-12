"""v2.3 step 3: colour tiers (COLOUR_*), model families (DASH_FAMILIES), and the dashboard's view features."""
import json
import re
import shutil
import subprocess

import pytest

from llmbench import envspec
from llmbench.config import load_config
from llmbench.reporting import colours
from llmbench.reporting.dashboard import TEMPLATE_PATH, render_dashboard
from llmbench.scoring import blank, finalize


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    import os
    for k in list(os.environ):
        if k.startswith(("EVAL_", "COLOUR_", "DASH_", "AGENT_", "MIN_", "BUDGET_", "VRAM_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    return load_config()


# --------------------------------------------------------------------------- colour scales
def test_default_scales_match_the_agreed_tiers(cfg):
    assert cfg.colour_style == "bands" and cfg.colour_mode == "absolute"
    assert cfg.colour_scales["GEN"] == (30, 40, 50, 80, 100)
    assert cfg.colour_scales["PREFILL"] == (500, 1000, 1500, 2500, 4000)
    assert all(len(t) == 5 for t in cfg.colour_scales.values())
    for m, (_, d, *_r) in colours.SCALES.items():           # 'low' scales fall, 'high' scales rise
        t = cfg.colour_scales[m]
        assert list(t) == sorted(t, reverse=(d == "low")), m


def test_threshold_parsing():
    assert colours.parse_thresholds("GEN", "10, 15,20,30,40") == (10, 15, 20, 30, 40)
    assert colours.parse_thresholds("RETRY", "30;20;10;5;1") == (30, 20, 10, 5, 1)
    for metric, raw in (("GEN", "1,2,3"), ("GEN", "50,40,30,20,10"), ("SQ", "10,20,30,40,50"), ("GEN", "a,b,c,d,e"),
                        ("NOPE", "1,2,3,4,5")):
        with pytest.raises(ValueError):
            colours.parse_thresholds(metric, raw)
    sc = colours.scales_from_env({"COLOUR_GEN": "10,15,20,30,40"})
    assert sc["GEN"] == (10, 15, 20, 30, 40) and sc["PREFILL"] == (500, 1000, 1500, 2500, 4000)


def test_colour_settings_load_and_fail_clearly(monkeypatch, cfg):
    monkeypatch.setenv("COLOUR_STYLE", "smooth")
    monkeypatch.setenv("COLOUR_MODE", "relative")
    monkeypatch.setenv("COLOUR_GEN", "10,15,20,30,40")
    c = load_config()
    assert (c.colour_style, c.colour_mode, c.colour_scales["GEN"]) == ("smooth", "relative", (10, 15, 20, 30, 40))
    for k, v in (("COLOUR_STYLE", "stripes"), ("COLOUR_MODE", "rel"), ("COLOUR_GEN", "10,20")):
        monkeypatch.setenv(k, v)
        with pytest.raises(SystemExit):
            load_config()
        monkeypatch.delenv(k)


def test_check_env_knows_the_colour_family(tmp_path):
    p = tmp_path / ".env"
    p.write_text("COLOUR_GENN=1,2,3,4,5\nCOLOUR_PREFILL=1,2\nCOLOUR_SQ=90,60,30,15,5\nCOLOUR_STYLE=smooth\n",
                 encoding="utf-8")
    rep = envspec.check_env(p, environ={})
    unknown = {k: s for k, _, _, s in rep.unknown}
    assert unknown == {"COLOUR_GENN": "COLOUR_GEN?"}
    assert {k for k, *_ in rep.invalid} == {"COLOUR_PREFILL"}


# --------------------------------------------------------------------------- families
REAL = ["agent-gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf", "gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf", "Air-26B-A4B.gguf",
        "Wicked-Stuff-26B-A4B-v3.0-Final.i1-IQ3_XS.gguf", "Qwen3.6-35B-A3B-UD-IQ3_XXS-MTP.gguf",
        "Qwen3.8-27B-UD-IQ3_XXS.gguf", "Ornith-1.5-35B-A3B-Q4_K_M.gguf", "LFM2.5-24B-A2B-UD-IQ3_XXS.gguf",
        "endless-frontier_BigBang-v1-35B-A3B-Q4_0.gguf", "GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf"]


def test_families_and_patterns():
    fams = colours.parse_families('"Gemma 4=*gemma-4*|Air*|Wicked*; BigBang=*BigBang*"')
    assert fams == (("Gemma 4", ("*gemma-4*", "Air*", "Wicked*")), ("BigBang", ("*BigBang*",)))
    g = [n for n in REAL if any(colours.matches(n, p) for p in fams[0][1])]
    assert g == REAL[:4]                                            # *gemma-4* finds agent-gemma-4... too
    assert colours.matches("agent-gemma-4-26B", "GEMMA") and not colours.matches("Qwen3.6", "gemma")
    assert colours.matches("Qwen3.6-35B", "qwen3.?-*") and not colours.matches("xQwen3.6", "qwen*")
    for bad in ("Gemma", "=*x*", "Name="):
        with pytest.raises(ValueError):
            colours.parse_families(bad)
    assert [colours.auto_family(n) for n in REAL[4:]] == ["Qwen", "Qwen", "Ornith", "LFM", "endless", "GrootTeam"]


def test_dash_families_setting(monkeypatch, cfg):
    monkeypatch.setenv("DASH_FAMILIES", "Gemma 4=*gemma-4*|Air*")
    assert load_config().dash_families == (("Gemma 4", ("*gemma-4*", "Air*")),)
    monkeypatch.setenv("DASH_FAMILIES", "broken")
    with pytest.raises(SystemExit):
        load_config()


# --------------------------------------------------------------------------- the dashboard
def _render(cfg, tmp_path, rows=None):
    rows = rows or []
    if not rows:
        for i, n in enumerate(REAL[:6]):
            r = blank(n)
            r.update(tech_tps=40.0 + 30 * i, logic_tps=40.0 + 30 * i, vram_gb=12.0, vram_total_gb=16.0, ctx_size=131072)
            rows.append(r)
    finalize(rows, cfg)
    out = tmp_path / "d.html"
    render_dashboard(rows, cfg, out)
    html = out.read_text(encoding="utf-8")
    meta = json.loads(re.search(r"const D = (\{.*?\}), META = D\.meta", html, re.S).group(1))["meta"]
    return html, meta


def test_dashboard_meta_carries_view_and_agent_settings(monkeypatch, cfg, tmp_path):
    cfg.dash_families = (("Gemma 4", ("*gemma-4*",)),)
    cfg.colour_scales = dict(cfg.colour_scales, GEN=(10.0, 15.0, 20.0, 30.0, 40.0))
    html, meta = _render(cfg, tmp_path)
    assert meta["colours"]["style"] == "bands" and meta["colours"]["scales"]["GEN"]["t"] == [10, 15, 20, 30, 40]
    assert meta["colours"]["scales"]["SQ"]["dir"] == "low" and meta["families"] == [{"name": "Gemma 4", "patterns": ["*gemma-4*"]}]
    assert meta["min_toolcall_pct"] == 70 and meta["agent_ctx_min"] == 65536 and meta["prefill_depth"] == 65536
    assert meta["agent_ctx_curve"] == [[65536, 0.95], [78848, 0.97], [131072, 1.0]] and meta["agent_ctx_pass_pct"] == 75


def test_template_has_the_v23_view_features():
    t = TEMPLATE_PATH.read_text(encoding="utf-8")
    for needle in ('id="viewbar"', 'id="helppop"', ":root[data-theme=\"light\"]", "prefers-color-scheme",
                   "llmbench.view.v1", "function tierOf", "function smoothColor", "function vramTier", "FAMILIES",
                   'id="fampat"', "SPEEDBYSETUP", "not measurements", "\\u26A0", "AGENT_CTX_CURVE"):
        assert needle in t, needle
    # every new table column has a tooltip / chooser group
    for k in ("agent_composite_pct", "agent_longctx_pct", "agent_ctx_factor", "prefill_depth_tps", "prefill_depth_ttft_s"):
        assert re.search(rf"\n {k}:\['", t), k
    # Chart.js can't read CSS variables: no dark-only text colour may be left in the chart code
    script = t.split("<script>", 1)[1]
    assert "'#e6edf3'" not in script.replace("text:'#e6edf3'", "")


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_dashboard_javascript_parses(cfg, tmp_path):
    html, _ = _render(cfg, tmp_path)
    js = html.split("<script>", 2)[-1].rsplit("</script>", 1)[0]
    (tmp_path / "d.js").write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(process.argv[1],'utf8'))",
                        str(tmp_path / "d.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- v2.3.2 leaderboard, table and chart views
def test_template_has_the_v232_view_features():
    t = TEMPLATE_PATH.read_text(encoding="utf-8")
    for needle in ('id="lbsec"', "function renderLeaderboard", "LBKEYS = {score:['score_agent'", 'id="hiddenbox"',
                   "data-restoreall", "state.excluded.has(m.model)", "function bindResize", "llmbench.colWidths.v1",
                   "Chart.register(topTicks)", "function fitAxis", "data-fmin", "Funnel final: ${esc(cap(FUNNEL_FINAL))}",
                   "if (!chart.isDatasetVisible(di)) return;"):
        assert needle in t, needle
    # the leaderboard sits above the winners and the charts, in its own full-width card
    assert t.index('id="lbsec"') < t.index('id="podium"') < t.index('id="charts"')
    # chart order: prefill|memory, speed|VRAM, power|prefill-by-length, then the rest
    ch = t.split("const CH = ", 1)[1].split(";\n", 1)[0]
    order = re.findall(r"\['(\w+)',", ch)
    assert order == ["pg", "mem", "tps", "vram", "pwr", "pctx", "eff", "jt", "ttft", "lat", "quant"], order
    # the leaderboard ranks by figures the run stored: nothing is blended or re-scored in the page
    lb = t.split("function renderLeaderboard", 1)[1].split("\n}\n", 1)[0]
    assert "sort((a,b) => (b[k]-a[k])" in lb and "score_agent *" not in lb and "agent_quality *" not in lb


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_fit_axis_follows_the_data(tmp_path):
    """Scatter axes: 80..100% starts near the lowest point, never below 0 or above 100; the data always fits."""
    t = TEMPLATE_PATH.read_text(encoding="utf-8")
    fn = "function fitAxis" + t.split("function fitAxis", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    cases = [[[80, 91.5, 100], 0, 100], [[12.7, 10.8, 14.5], 0, None], [[57.6, 93.6], 0, 100], [[5], 0, None],
             [[0.4, 2.1], 0, None], [[99.2, 99.9], 0, 100], [[], 0, 100], [[None, 70], 0, 100]]
    (tmp_path / "f.js").write_text(fn + f"console.log(JSON.stringify({json.dumps(cases)}.map(c=>fitAxis(c[0],c[1]??undefined,c[2]??undefined))))",
                                   encoding="utf-8")
    r = subprocess.run(["node", str(tmp_path / "f.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out[0] == {"min": 75, "max": 100} and out[6] == {}
    for (vals, lo, hi), b in zip(cases, out):
        v = [x for x in vals if x is not None]
        if not v:
            continue
        assert b["min"] <= min(v) and b["max"] >= max(v), (vals, b)
        assert b["min"] >= lo and (hi is None or b["max"] <= hi), (vals, b)
        assert b["max"] - b["min"] <= max(4 * ((max(v) - min(v)) or 1), 2), (vals, b)     # no wide empty margin
    assert out[1]["min"] > 0          # VRAM 10.8..14.5 GB does not start at 0
