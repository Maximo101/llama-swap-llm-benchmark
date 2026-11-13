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
