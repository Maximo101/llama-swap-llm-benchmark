"""Changes after v2.3.3: LONGCTX 90k slot and default, MAX_GPU_TEMP 85, and the EVAL_SANITY_PCT guide (default stays 0)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from llmbench import agent
from llmbench.config import LONGCTX_SLOTS
from llmbench.reporting import json_out
from llmbench.scoring import FIELDS


def test_90k_is_a_long_context_slot_between_64k_and_128k():
    assert list(LONGCTX_SLOTS.items()) == [(32768, "32k"), (65536, "64k"), (92160, "90k"), (131072, "128k")]
    assert "longctx_90k_pct" in FIELDS and "longctx_90k_pct" in json_out._GROUPS["long_context"]
    rows = [{"model": "a", "longctx_64k_pct": 100.0, "longctx_90k_pct": 87.5, "longctx_128k_pct": 75.0},
            {"model": "b", "longctx_64k_pct": 100.0, "longctx_90k_pct": 87.5}]
    assert agent.common_lengths(rows)[0] == ["64k", "90k"] and agent.run_lc_lengths(rows) == [65536, 92160, 131072]


def test_a_model_loaded_at_90112_is_not_asked_the_92160_length_and_keeps_its_loaded_context():
    cfg = type("C", (), {"agent_ctx_pass_pct": 75.0})()
    row = {"ctx_size": 90112, "longctx_64k_pct": 100.0}
    assert agent.usable_ctx(row, cfg, [65536, 92160, 131072]) == (90112, None)
    row92 = {"ctx_size": 92160, "longctx_64k_pct": 100.0}                    # holds 90k, has no result: not passed
    assert agent.usable_ctx(row92, cfg, [65536, 92160, 131072])[0] == 65536
    row92["longctx_90k_pct"] = 87.5
    assert agent.usable_ctx(row92, cfg, [65536, 92160, 131072]) == (92160, None)


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_sanity_guide_shows_the_halving_sequence_from_the_current_value(tmp_path):
    from llmbench.ui import server
    html = (Path(server.__file__).with_name("settings.html")).read_text(encoding="utf-8")
    guide = re.search(r"const sanityGuide = .*?\n};\n", html, re.S).group(0)
    js = ("var V = {};\nconst shown = e => e.v; const ent = k => ({v: V[k]});\n" + guide +
          "\nconst out = {};\nfor (const v of ['20', '100', '5', '0', '']) { V.EVAL_SANITY_PCT = v; "
          "out[v] = (sanityGuide().match(/<b>([^<]*&rarr;[^<]*|[^<]*%)<\\/b>\\. Each new run/) || [])[1]; }\n"
          "console.log(JSON.stringify(out));")
    f = tmp_path / "g.js"; f.write_text(js)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["20"] == "20% &rarr; 10% &rarr; 5% &rarr; 2% &rarr; 0%"
    assert out["100"] == "100% &rarr; 50% &rarr; 25% &rarr; 12% &rarr; 6% &rarr; 3% &rarr; 0%"
    assert out["5"] == "5% &rarr; 2% &rarr; 0%" and out["0"] == "20% &rarr; 10% &rarr; 5% &rarr; 2% &rarr; 0%" and out[""] == "20% &rarr; 10% &rarr; 5% &rarr; 2% &rarr; 0%"


# ---- v2.3.5: LLAMA_SWAP_* -> SERVER_*, STRATA_* -> SECONDARY_* (old names keep working) ----
RENAMED = {"LLAMA_SWAP_URL": "SERVER_URL", "LLAMA_SWAP_API_KEY": "SERVER_API_KEY", "STRATA_URL": "SECONDARY_URL",
           "STRATA_SWAP_URL": "SECONDARY_URL", "STRATA_API_KEY": "SECONDARY_API_KEY", "STRATA_MODEL": "SECONDARY_MODEL",
           "STRATA_PARAMS_B": "SECONDARY_PARAMS_B", "STRATA_ACTIVE_B": "SECONDARY_ACTIVE_B",
           "STRATA_QUANT": "SECONDARY_QUANT", "STRATA_CTX": "SECONDARY_CTX"}


def _clean(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith(("SERVER_", "SECONDARY_", "LLAMA_SWAP_", "STRATA_")):
            monkeypatch.delenv(k, raising=False)
    from llmbench import config
    config._LEGACY_WARNED.clear()


def test_old_setting_names_still_work_and_say_so(monkeypatch, capsys):
    from llmbench.config import load_config
    _clean(monkeypatch)
    for k, v in {"LLAMA_SWAP_URL": "http://old:8000/", "LLAMA_SWAP_API_KEY": "k1", "STRATA_URL": "http://sec:9000",
                 "STRATA_MODEL": "m", "STRATA_PARAMS_B": "35", "STRATA_ACTIVE_B": "3", "STRATA_QUANT": "Q4", "STRATA_CTX": "65536",
                 "STRATA_API_KEY": "k2"}.items():
        monkeypatch.setenv(k, v)
    c = load_config()
    assert (c.base_url, c.api_key, c.strata_url, c.strata_api_key, c.strata_model) == ("http://old:8000", "k1", "http://sec:9000", "k2", "m")
    assert (c.strata_params_b, c.strata_active_b, c.strata_quant, c.strata_ctx) == (35.0, 3.0, "Q4", 65536)
    out = capsys.readouterr().out
    assert "LLAMA_SWAP_URL is now called SERVER_URL" in out and "STRATA_CTX is now called SECONDARY_CTX" in out
    _clean(monkeypatch)
    monkeypatch.setenv("STRATA_SWAP_URL", "http://older:1")                   # the v2.3 name
    assert load_config().strata_url == "http://older:1"


def test_the_new_name_wins_when_both_are_set_and_nothing_is_said(monkeypatch, capsys):
    from llmbench.config import load_config
    _clean(monkeypatch)
    monkeypatch.setenv("LLAMA_SWAP_URL", "http://old:1")
    monkeypatch.setenv("SERVER_URL", "http://new:2")
    c = load_config()
    assert c.base_url == "http://new:2" and "is now called" not in capsys.readouterr().out


def test_check_env_and_the_page_helpers_know_the_old_names(tmp_path):
    from llmbench import envspec
    from llmbench.ui import server
    assert set(RENAMED) == set(envspec.LEGACY) - set()           # every old name is mapped, and only those
    assert all(v in {e.key for e in envspec.parse_example()} for v in set(RENAMED.values()))
    p = tmp_path / ".env"
    p.write_text("LLAMA_SWAP_URL=http://x:1\nSTRATA_URL=http://y:2\n", encoding="utf-8")
    rep = envspec.check_env(p, environ={})
    assert {(a, b) for a, b, _ in rep.legacy} == {("LLAMA_SWAP_URL", "SERVER_URL"), ("STRATA_URL", "SECONDARY_URL")}
    assert server._setting({"LLAMA_SWAP_URL": "http://x:1"}, "SERVER_URL") == "http://x:1"
    assert server._pick({"STRATA_URL": "http://y:2"}, "SECONDARY_URL") == "http://y:2"
    assert envspec.old_names("SECONDARY_URL") == ["STRATA_SWAP_URL", "STRATA_URL"]


def test_saving_the_settings_renames_old_keys_in_the_env_file(tmp_path):
    from llmbench import envfile
    text = envfile.render_env({"LLAMA_SWAP_URL": "http://x:1", "STRATA_API_KEY": "s3"})
    lines = text.splitlines()
    assert "SERVER_URL=http://x:1" in lines[next(i for i, l in enumerate(lines) if l.startswith("SERVER_URL="))]
    assert any(l.startswith("SECONDARY_API_KEY=s3") for l in lines)
    assert not any(l.startswith(("LLAMA_SWAP_URL=", "STRATA_API_KEY=")) for l in lines)       # only comments name them
