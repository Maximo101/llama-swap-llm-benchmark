"""v2.3 step 4: the settings page (python benchmark.py --ui), .env writing, favourites, --env / --validate, run lock."""
import http.client
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench import cli, config, envfile, envspec
from llmbench.config import SCRIPT_DIR, load_config
from llmbench.ui import server

BENCH = str(SCRIPT_DIR / "benchmark.py")


# --------------------------------------------------------------------------- choices and kinds
@pytest.mark.parametrize("key", sorted(envspec.CHOICES))
def test_every_choice_loads_and_others_are_refused(key, monkeypatch, tmp_path):
    """The page's drop-downs offer exactly what a run accepts: each choice loads, anything else stops the run and
    is marked by --check-env / the page."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    for c in envspec.CHOICES[key]:
        monkeypatch.setenv(key, c)
        load_config()
        assert envspec._value_problem(c, "choice", key) is None
    monkeypatch.setenv(key, "not-a-choice")
    with pytest.raises(SystemExit):
        load_config()
    assert envspec._value_problem("not-a-choice", "choice", key)


def test_secrets_and_choice_kinds():
    code = envspec.code_settings()
    for k in envspec.SECRETS:
        assert code[k][1] == "secret", k
    for k in envspec.CHOICES:
        assert code[k][1] == "choice", k
    keys = {e.key for e in envspec.parse_example()}
    assert {"UI_PORT", "UI_HOST", "UI_TOKEN"} <= keys
    assert code["UI_PORT"][0] == 41010


# --------------------------------------------------------------------------- writing .env
def _sample():
    return {"LLAMA_SWAP_URL": "http://10.0.0.5:8080", "HF_TOKEN": "hf_abc#def", "EVAL_PROFILE": "thorough",
            "MODEL_FILTER": "gemma|qwen", "DASH_FAMILIES": "Gemma=*gemma-4*;BigBang=*bigbang*",
            "COLOUR_GEN": "20,30,40,60,80", "AGENT_CTX_CURVE": "131072:1.0,92160:0.9,65536:0.75",
            "INCLUDE_MODELS": "all", "BUDGET_SERVER_VRAM_RESERVE_GB": "1.5", "SOME_OLD_THING": "it's kept"}


def test_write_env_round_trips_with_python_dotenv(tmp_path):
    dotenv = pytest.importorskip("dotenv")
    p = tmp_path / ".env"
    p.write_text("OLD=1\n")
    vals = _sample()
    bak = envfile.write_env(vals, p)
    assert bak == tmp_path / ".env.bak" and bak.read_text() == "OLD=1\n"
    got = {k: v for k, v in dotenv.dotenv_values(p).items() if v is not None}
    for k, v in vals.items():
        assert got.get(k) == v, k
    ours = envfile.read_values(p)
    for k, v in vals.items():
        assert ours.get(k) == v, k
    text = p.read_text()
    assert text.startswith("# Written by the settings page")
    assert "# --- Not in .env.example" in text and "SOME_OLD_THING=" in text
    lines = text.splitlines()                                  # a family member sits right after its family line
    i = next(n for n, ln in enumerate(lines) if ln.startswith("COLOUR_GEN="))
    assert "COLOUR_<METRIC>" in "\n".join(lines[max(0, i - 4):i])
    # an untouched setting keeps its shipped (commented or active) line
    example = envspec.EXAMPLE_PATH.read_text().splitlines()
    gold = [ln for ln in example if ln.startswith(("GOLDILOCKS_MODE", "# GOLDILOCKS_MODE", "#GOLDILOCKS_MODE"))]
    assert all(g in lines for g in gold)


def test_write_env_refuses_values_it_cannot_write(tmp_path):
    with pytest.raises(envfile.EnvValueError):
        envfile.write_env({"MODEL_FILTER": "a\nb"}, tmp_path / ".env")
    with pytest.raises(envfile.EnvValueError):
        envfile.write_env({"MODEL_FILTER": "it's \"x\""}, tmp_path / ".env")
    assert not (tmp_path / ".env").exists()


def test_defaults_keep_secrets_only(tmp_path):
    cur = {"HF_TOKEN": "hf_x", "EVAL_PROFILE": "deep", "LLAMA_SWAP_API_KEY": ""}
    d = envfile.defaults_with_secrets(cur)
    assert d["HF_TOKEN"] == "hf_x" and "LLAMA_SWAP_API_KEY" not in d
    assert d.get("EVAL_PROFILE") == envfile.shipped_values()["EVAL_PROFILE"]


# --------------------------------------------------------------------------- favourites
def test_favourite_holds_only_changes_and_no_secrets(tmp_path):
    shipped = envfile.shipped_values()
    vals = dict(shipped, EVAL_PROFILE="deep", HF_TOKEN="hf_secret", COLOUR_STYLE="smooth")
    p = envfile.save_favourite(tmp_path, "night", vals, "2.3.0")
    text = p.read_text()
    assert "hf_secret" not in text and "HF_TOKEN" not in text
    fav, then = envfile.read_favourite(p)
    assert fav == {"EVAL_PROFILE": "deep", "COLOUR_STYLE": "smooth"}
    assert then["COLOUR_STYLE"] == shipped["COLOUR_STYLE"]
    assert envfile.list_favourites(tmp_path) == ["night"]
    for bad in ("../x", "a/b", "", ".hidden", "x" * 41):
        with pytest.raises(envfile.EnvValueError):
            envfile.favourite_path(tmp_path, bad)


def test_favourite_onto_a_newer_release(tmp_path):
    """Renamed settings follow LEGACY, removed ones are listed, a changed shipped value the user also changed is a
    conflict to choose, new settings keep the new defaults, secrets come from the current .env."""
    shipped = envfile.shipped_values()
    p = tmp_path / "old.env"
    p.write_text("# favourite from an older version\n"
                 "STRATA_SWAP_URL=http://strata:9000      # shipped: \n"
                 "GONE_SETTING=5                          # shipped: 3\n"
                 f"EVAL_PROFILE=deep                       # shipped: not-{shipped['EVAL_PROFILE']}\n"
                 f"COLOUR_MODE=relative                    # shipped: {shipped['COLOUR_MODE']}\n"
                 "HF_TOKEN=from_the_favourite\n")
    res = envfile.apply_favourite(p, {"HF_TOKEN": "hf_now"})
    v = res["values"]
    assert ("STRATA_SWAP_URL", "STRATA_URL") in [tuple(x) for x in res["renamed"]] and v["STRATA_URL"] == "http://strata:9000"
    assert [tuple(x) for x in res["removed"]] == [("GONE_SETTING", "5")]
    assert [c["key"] for c in res["conflicts"]] == ["EVAL_PROFILE"] and v["EVAL_PROFILE"] == "deep"
    assert v["COLOUR_MODE"] == "relative"
    assert v["HF_TOKEN"] == "hf_now"
    assert v["AGENT_CTX_MIN"] == shipped["AGENT_CTX_MIN"]       # a v2.3 setting the old favourite never had


# --------------------------------------------------------------------------- the server
@pytest.fixture
def ui(tmp_path, monkeypatch):
    """A settings page on an ephemeral port over a private .env (UI_TOKEN set when `token` is given)."""
    for k in list(os.environ):
        if k.startswith(("UI_", "EVAL_", "COLOUR_", "HF_", "OUTPUT_DIR")):
            monkeypatch.delenv(k, raising=False)
    env = tmp_path / "my.env"
    env.write_text(f"OUTPUT_DIR={tmp_path / 'results'}\nHF_TOKEN=hf_topsecret\nEVAL_PROFILE=quick\n")
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.setattr(server, "configs_dir", lambda: tmp_path / "configs")
    servers = []

    def start(token=""):
        cfg = SimpleNamespace(host="127.0.0.1", port=0, token=token, lan=bool(token))
        srv = server.UIServer(cfg)
        cfg.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return SimpleNamespace(srv=srv, port=cfg.port, env=env, dir=tmp_path)
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _req(port, method, path, body=None, host=None, headers=None, json_headers=True):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Host": host or f"127.0.0.1:{port}"}
    if body is not None and json_headers:
        h.update({"Content-Type": "application/json", "X-LLMB-UI": "1"})
    h.update(headers or {})
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read()
    try:
        out = json.loads(data)
    except ValueError:
        out = data.decode("utf-8", "replace")
    return r.status, dict(r.getheaders()), out


def test_host_header_guard(ui):
    u = ui()
    assert _req(u.port, "GET", "/", host="evil.example.com")[0] == 403          # DNS rebinding
    assert _req(u.port, "GET", "/", host=f"127.0.0.1:{u.port + 1}")[0] == 403    # another port
    st, hd, page = _req(u.port, "GET", "/")
    assert st == 200 and "frame-ancestors 'none'" in hd["Content-Security-Policy"] and hd["X-Frame-Options"] == "DENY"
    assert "Local LLM Benchmark Suite" in page
    assert _req(u.port, "GET", "/", host=f"localhost:{u.port}")[0] == 200
    assert server.host_ok("[::1]:41010", 41010) and server.host_ok("192.168.1.20:41010", 41010)
    assert not server.host_ok("attacker.test:41010", 41010) and not server.host_ok("", 41010)


def test_state_masks_secrets_and_creates_main(ui):
    u = ui()
    st, _, s = _req(u.port, "GET", "/api/state")
    assert st == 200
    assert "hf_topsecret" not in json.dumps(s)
    assert s["values"]["HF_TOKEN"] == "" and s["secrets_set"]["HF_TOKEN"] is True
    assert s["created_main"] and s["favourites"] == ["main"]
    _, _, again = _req(u.port, "GET", "/api/state")                   # announced once, not after every save
    assert again["created_main"] is False and again["favourites"] == ["main"]
    main = (u.dir / "configs" / "main.env").read_text()
    assert "EVAL_PROFILE=quick" in main and "hf_topsecret" not in main
    keys = {e["key"] for e in s["entries"]}
    assert {"AGENT_CTX_MIN", "PREFILL_DEPTH", "COLOUR_STYLE", "UI_PORT"} <= keys
    e = next(e for e in s["entries"] if e["key"] == "COLOUR_STYLE")
    assert e["choices"] == ["bands", "smooth"] and e["kind"] == "choice"
    assert _req(u.port, "POST", "/api/secret", {"key": "HF_TOKEN"})[2]["value"] == "hf_topsecret"
    assert _req(u.port, "POST", "/api/secret", {"key": "EVAL_PROFILE"})[0] == 400


def test_post_needs_the_page_header_and_same_origin(ui):
    u = ui()
    body = {"values": {"EVAL_PROFILE": "deep"}}
    assert _req(u.port, "POST", "/api/check", body, json_headers=False,
                headers={"Content-Type": "application/json"})[0] == 403                  # no X-LLMB-UI
    assert _req(u.port, "POST", "/api/check", body, headers={"Origin": "http://evil.example.com"})[0] == 403
    assert _req(u.port, "POST", "/api/check", body, json_headers=False,
                headers={"X-LLMB-UI": "1", "Content-Type": "text/plain"})[0] == 403      # a simple form post
    assert _req(u.port, "POST", "/api/check", body, headers={"Origin": f"http://127.0.0.1:{u.port}"})[0] == 200
    assert _req(u.port, "PUT", "/api/save", body)[0] == 405
    assert _req(u.port, "GET", "/results/..%2f.env")[0] == 404
    assert _req(u.port, "GET", "/results/my.env")[0] == 404


def test_check_and_save(ui):
    u = ui()
    st, _, rep = _req(u.port, "POST", "/api/check", {"values": {"COLOUR_STYLE": "wavy", "UI_PORT": "abc"}})
    assert st == 200 and set(rep["problems"]) >= {"COLOUR_STYLE", "UI_PORT"}
    before = u.env.read_text()
    st, _, out = _req(u.port, "POST", "/api/save", {"values": {"COLOUR_STYLE": "wavy"}})
    assert out["saved"] is False and u.env.read_text() == before
    vals = {"OUTPUT_DIR": str(u.dir / "results"), "EVAL_PROFILE": "deep", "COLOUR_MODE": "relative", "HF_TOKEN": ""}
    st, _, out = _req(u.port, "POST", "/api/save", {"values": vals})
    assert st == 200 and out["saved"], out
    now = envfile.read_values(u.env)
    assert now["EVAL_PROFILE"] == "deep" and now["COLOUR_MODE"] == "relative"
    assert now["HF_TOKEN"] == "hf_topsecret"                            # not named in "secrets": kept
    assert (u.dir / "my.env.bak").read_text() == before
    st, _, out = _req(u.port, "POST", "/api/save", {"values": vals, "secrets": {"HF_TOKEN": ""}})
    assert out["saved"] and not envfile.read_values(u.env).get("HF_TOKEN")   # cleared on purpose
    st, _, out = _req(u.port, "POST", "/api/save", {"values": vals, "secrets": {"HF_TOKEN": "hf_new"}})
    assert envfile.read_values(u.env)["HF_TOKEN"] == "hf_new"


def test_return_to_defaults_keeps_tokens(ui):
    u = ui()
    st, _, out = _req(u.port, "POST", "/api/defaults", {})
    assert out["saved"]
    now = envfile.read_values(u.env)
    assert now["HF_TOKEN"] == "hf_topsecret"
    assert now.get("EVAL_PROFILE") == envfile.shipped_values()["EVAL_PROFILE"]


def test_fixed_commands_only(ui, monkeypatch):
    u = ui()
    assert _req(u.port, "POST", "/api/cmd", {"cmd": "rm"})[0] == 400
    assert _req(u.port, "POST", "/api/cmd", {"cmd": "compare", "files": ["/etc/passwd", "x.json"]})[0] == 400
    (u.dir / "results" / "benchmark_20261101_101010.json").write_text("{}")
    (u.dir / "results" / "benchmark_20261102_101010.json").write_text("{}")
    argv = server.command_argv("compare", {"files": ["benchmark_20261101_101010.json",
                                                     "benchmark_20261102_101010.json"], "force": True})
    assert argv[0] == "--compare" and argv[-1] == "--force" and all(Path(a).is_absolute() for a in argv[1:-1])
    with pytest.raises(ValueError):
        server.command_argv("compare", {"files": ["benchmark_20261101_101010.json; rm -rf /"]})


def test_estimate_uses_the_unsaved_values_via_env(ui, monkeypatch):
    u = ui()
    seen = {}

    def fake_run(argv, **kw):
        i = argv.index("--env")
        p = Path(argv[i + 1])
        seen.update(argv=argv, text=p.read_text(), mode=stat.S_IMODE(p.stat().st_mode))
        return SimpleNamespace(returncode=0, stdout="estimate ok\n", stderr="")
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    st, _, out = _req(u.port, "POST", "/api/cmd", {"cmd": "estimate", "values": {"EVAL_PROFILE": "ultra"}})
    assert st == 200 and out["output"] == "estimate ok\n"
    assert seen["argv"][-1] == "--estimate" and seen["mode"] == 0o600
    assert "EVAL_PROFILE=ultra" in seen["text"] and "HF_TOKEN=hf_topsecret" in seen["text"]
    tmpfile = Path(seen["argv"][seen["argv"].index("--env") + 1])
    assert not tmpfile.exists()                                        # removed afterwards
    out = server.run_command(["--reload"], None, 10)                   # saved settings: --env <the page's file>
    assert seen["argv"][-3:] == ["--env", str(u.env), "--reload"] or seen["argv"][-2:] == [str(u.env), "--reload"]


def test_favourite_endpoints(ui):
    u = ui()
    _req(u.port, "GET", "/api/state")
    st, _, out = _req(u.port, "POST", "/api/fav/save", {"name": "night", "values": {"EVAL_PROFILE": "deep",
                                                                                     "HF_TOKEN": "leak?"}})
    assert out["favourites"] == ["main", "night"]
    assert "leak" not in (u.dir / "configs" / "night.env").read_text()
    assert _req(u.port, "POST", "/api/fav/save", {"name": "../../evil", "values": {}})[0] == 400
    st, _, out = _req(u.port, "POST", "/api/fav/load", {"name": "night"})
    assert out["values"]["EVAL_PROFILE"] == "deep" and out["values"].get("HF_TOKEN", "") == ""
    st, _, out = _req(u.port, "POST", "/api/fav/delete", {"name": "night"})
    assert out["favourites"] == ["main"]


def test_lan_mode_needs_the_token(ui):
    u = ui(token="s3cret-token")
    assert _req(u.port, "GET", "/api/state")[0] == 401
    assert _req(u.port, "GET", "/")[0] == 401
    assert _req(u.port, "GET", "/?token=wrong")[0] == 401
    assert _req(u.port, "POST", "/api/check", {"values": {}})[0] == 401
    st, hd, _ = _req(u.port, "GET", "/?token=s3cret-token")
    assert st == 303 and hd["Location"] == "/"
    ck = hd["Set-Cookie"]
    assert "HttpOnly" in ck and "SameSite=Strict" in ck
    cookie = ck.split(";")[0]
    assert _req(u.port, "GET", "/api/state", headers={"Cookie": cookie})[0] == 200
    assert _req(u.port, "POST", "/api/check", {"values": {}}, headers={"Cookie": cookie})[0] == 200
    assert _req(u.port, "GET", "/api/state", headers={"Cookie": "llmb_ui=nope"})[0] == 401


def test_page_settings_default_to_localhost(tmp_path, monkeypatch):
    for k in ("UI_HOST", "UI_PORT", "UI_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("EVAL_PROFILE=quick\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    s = server.page_settings()
    assert (s.host, s.port, s.token, s.lan) == ("127.0.0.1", 41010, "", False)
    env.write_text("UI_HOST=0.0.0.0\nUI_PORT=41999\n")
    s = server.page_settings()
    assert s.lan and s.port == 41999
    monkeypatch.setenv("UI_HOST", "127.0.0.1")                          # the shell wins, as for a run
    assert not server.page_settings().lan


# --------------------------------------------------------------------------- run lock, start / stop
def _sleeper():
    # "benchmark" in the command line, like a real run (lock_info checks it against pid reuse)
    return subprocess.Popen([sys.executable, "-c", "import time\ntry:\n time.sleep(60)\nexcept KeyboardInterrupt:\n pass",
                             "benchmark"])


def test_run_lock(tmp_path):
    assert cli.pid_alive(os.getpid()) and not cli.pid_alive(0) and not cli.pid_alive(2 ** 22 + 12345)
    cfg = SimpleNamespace(output_dir=tmp_path)
    assert cli.lock_info(cfg) is None
    p = cli.run_lock(cfg, tmp_path / "x.log")
    info = json.loads(p.read_text())
    assert info["pid"] == os.getpid() and info["log"].endswith("x.log")
    cli.release_lock(p)
    assert not p.exists()
    (tmp_path / cli.LOCK_NAME).write_text(json.dumps({"pid": 2 ** 22 + 12345}))   # stale: no such process
    assert cli.lock_info(tmp_path) is None
    child = _sleeper()
    try:
        (tmp_path / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None}))
        assert cli.lock_info(tmp_path)["pid"] == child.pid
    finally:
        child.kill()
        child.wait()
    assert cli.lock_info(tmp_path) is None


@pytest.mark.skipif(os.name != "posix", reason="SIGINT to another process is POSIX")
def test_stop_sends_ctrl_c_and_one_run_at_a_time(ui, monkeypatch):
    u = ui()
    child = _sleeper()
    try:
        (u.dir / "results" / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None}))
        st, _, out = _req(u.port, "GET", "/api/run")
        assert out["running"] and out["pid"] == child.pid
        assert _req(u.port, "POST", "/api/run", {"mode": "speed"})[0] == 409          # refuses a second run
        assert _req(u.port, "POST", "/api/stop", {})[2]["stopping"] is True
        assert child.wait(timeout=10) is not None
    finally:
        if child.poll() is None:
            child.kill()
    assert _req(u.port, "POST", "/api/run", {"mode": "rm -rf"})[0] == 400


def test_log_follows_the_run(ui):
    u = ui()
    child = _sleeper()
    log = u.dir / "results" / "benchmark_x.log"
    log.write_text("line one\n")
    try:
        (u.dir / "results" / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now",
                                                                     "log": str(log)}))
        out = _req(u.port, "GET", "/api/log?offset=0")[2]
        assert out["running"] and out["text"] == "line one\n"
        with open(log, "a") as fh:
            fh.write("line two\n")
        out = _req(u.port, "GET", f"/api/log?offset={out['offset']}")[2]
        assert out["text"] == "line two\n"
        with open(log, "a") as fh:
            fh.write("Script Finished\n")
    finally:
        child.kill()
        child.wait()
    out = _req(u.port, "GET", f"/api/log?offset={out['offset']}")[2]        # the run is gone: still its last lines
    assert not out["running"] and out["text"] == "Script Finished\n" and out["log"] == str(log)


# --------------------------------------------------------------------------- CLI: --env, --validate
def test_cli_env_and_validate(tmp_path):
    good = tmp_path / "good.env"
    good.write_text(f"OUTPUT_DIR={tmp_path / 'r'}\nEVAL_PROFILE=quick\nCOLOUR_STYLE=smooth\n")
    r = subprocess.run([sys.executable, BENCH, "--env", str(good), "--validate"], capture_output=True, text=True,
                       timeout=120, cwd=tmp_path)
    assert r.returncode == 0 and "Settings OK" in r.stdout, r.stdout + r.stderr
    bad = tmp_path / "bad.env"
    bad.write_text(f"OUTPUT_DIR={tmp_path / 'r'}\nCOLOUR_STYLE=wavy\n")
    r = subprocess.run([sys.executable, BENCH, "--env", str(bad), "--validate"], capture_output=True, text=True,
                       timeout=120, cwd=tmp_path)
    assert r.returncode != 0 and "COLOUR_STYLE" in r.stdout + r.stderr
    r = subprocess.run([sys.executable, BENCH, "--env", str(tmp_path / "missing.env"), "--validate"],
                       capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert r.returncode == 2 and "not found" in r.stderr
    r = subprocess.run([sys.executable, BENCH, "--env", str(bad), "--check-env"], capture_output=True, text=True,
                       timeout=120, cwd=tmp_path)
    assert "COLOUR_STYLE" in r.stdout


def test_settings_page_parses_under_node(tmp_path):
    import shutil
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    import re
    html = (Path(server.__file__).with_name("settings.html")).read_text()
    js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    f = tmp_path / "page.js"
    f.write_text(js)
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- v2.3 settings-page guides and defaults
def test_gpu_bandwidth_table_and_fallback():
    from llmbench import gpubw
    assert gpubw.lookup("NVIDIA GeForce RTX 5070 Ti", 15.9)[6] == 896
    assert gpubw.lookup("NVIDIA GeForce RTX 5070", 11.9)[6] == 672             # not the Ti row
    assert gpubw.lookup("NVIDIA GeForce RTX 4070 Ti SUPER", 16)[6] == 672
    assert gpubw.lookup("NVIDIA GeForce RTX 4070 SUPER", 12)[6] == 504
    assert gpubw.lookup("NVIDIA GeForce RTX 3080", 10)[6] == 760 and gpubw.lookup("RTX 3080", 12)[6] == 912
    assert gpubw.lookup("NVIDIA GeForce RTX 3060 Ti", 8)[6] == 448 and gpubw.lookup("RTX 3060", 12)[6] == 360
    assert gpubw.lookup("Radeon RX 7900 XTX", 24) is None
    assert gpubw.resolve("auto", [("NVIDIA GeForce RTX 5090", 31.8)]) == (1792.0, "RTX 5090 (detected)")
    assert gpubw.resolve("", [])[0] == 670.0 and "no GPU" in gpubw.resolve("", [])[1]
    assert gpubw.resolve("auto", [("Radeon RX 7900 XTX", 24)])[0] == 670.0
    assert gpubw.resolve("500", [("NVIDIA GeForce RTX 5090", 31.8)]) == (500.0, "ESTIMATE_GPU_BW_GBS")   # a number wins
    assert gpubw.valid("auto") is None and gpubw.valid("672") is None and gpubw.valid("fast")
    assert envspec._value_problem("fast", "str", "ESTIMATE_GPU_BW_GBS")


def test_new_defaults(monkeypatch, tmp_path):
    """EVAL_PROFILE funnel, EVAL_TEMPERATURE and TEMPERATURE server (llama-swap's sampling), bandwidth auto."""
    for k in ("EVAL_PROFILE", "EVAL_TEMPERATURE", "TEMPERATURE", "ESTIMATE_GPU_BW_GBS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "none.env")
    (tmp_path / "none.env").write_text(f"OUTPUT_DIR={tmp_path}\n")
    cfg = load_config()
    assert cfg.eval_profile == "funnel" and cfg.eval_temperature is None and cfg.temperature is None
    code = envspec.code_settings()
    assert code["ESTIMATE_GPU_BW_GBS"][0] == "auto"
    ex = {e.key: e for e in envspec.parse_example()}
    assert ex["ESTIMATE_GPU_BW_GBS"].part == 1 and ex["ESTIMATE_GPU_BW_GBS"].active
    monkeypatch.setenv("TEMPERATURE", "0.3")
    assert load_config().temperature == 0.3
    monkeypatch.setenv("TEMPERATURE", "warm")
    with pytest.raises(SystemExit):
        load_config()


def test_guides_in_state(ui):
    u = ui()
    g = _req(u.port, "GET", "/api/state")[2]["guides"]
    assert g["recommended"]["EVAL_PROFILE"] == "funnel"
    quick = next(p for p in g["profiles"] if p["name"] == "quick")
    assert quick["rows"] == 11 and quick["tests"]["humaneval"]["margin"] == 29.5
    assert quick["tests"]["toolcall"]["n"] == 32                       # tool calling: always all 32 cases
    full = next(p for p in g["profiles"] if p["name"] == "full")
    assert full["tests"]["gpqa"]["n"] == 198
    mod = next(f for f in g["finals"] if f["final"] == "moderate")
    assert [s["name"] for s in mod["stages"]] == ["quick", "basic", "moderate"]
    assert g["longctx_n"] == 8 and g["gpu"]["fallback"] == 670.0 and len(g["gpu"]["table"]) >= 14


def test_batch_probe_from_the_page(ui, monkeypatch):
    seen = {}

    class P:
        pid = 4242

        def poll(self):
            return None
    monkeypatch.setattr(server.subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv) or P())
    u = ui()
    st, _, out = _req(u.port, "POST", "/api/run", {"mode": "probe"})
    assert st == 200 and seen["argv"][-1] == "--batch-probe"
    assert _req(u.port, "GET", "/api/run")[2]["running"]               # followed like a run until it ends
    server._CHILD.update(proc=None)


def test_models_list_reads_gguf_headers(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    import gguf_writer
    d = tmp_path / "models"
    d.mkdir()
    gguf_writer.moe_mla_model(d / "Mock-30B-A3B-IQ3_XXS.gguf", pad_to=80_000)
    monkeypatch.delenv("GGUF_SEARCH_DIRS", raising=False)
    monkeypatch.delenv("GGUF_PATH_MAP", raising=False)
    g = server.gguf_meta("Mock-30B-A3B-IQ3_XXS.gguf", {"GGUF_SEARCH_DIRS": str(d)})
    assert g["src"] == "GGUF header" and g["size_gb"] > 0 and g["params_b"] and g["n_ctx_train"] and g["arch"]
    last = {"gguf_gb": 12.0, "params_b": 29.9, "active_b": 3.9, "n_ctx_train": 202752, "arch": "deepseek2", "bpw": 3.4}
    g = server.gguf_meta("Other-30B-A3B.gguf", {}, last)                 # not readable here: the last run's figures
    assert g["src"] == "last run" and g["size_gb"] == 12.0 and g["active_b"] == 3.9
    g = server.gguf_meta("Qwen3.8-27B-UD-IQ3_XXS.gguf", {})              # nothing known: parameters from the name
    assert g["src"] == "model name" and g["params_b"] == 27 and g["size_gb"] is None


# --------------------------------------------------------------------------- page on another PC than llama-swap
def test_llama_swap_on_another_machine():
    from llmbench import locality
    assert locality.server_is_local("http://localhost:8000")[0] is True
    assert locality.server_is_local("http://127.0.0.1:8080")[0] is True
    import socket
    assert locality.server_is_local(f"http://{socket.gethostname()}:8000")[0] in (True, None)
    assert locality.server_is_local("http://192.0.2.10:8000")[0] is False          # TEST-NET: never this machine
    assert "another machine" in locality.remote_warning("http://192.0.2.10:8000")
    assert locality.remote_warning("http://localhost:8000") is None


def test_latest_copy_that_cannot_be_written(tmp_path, capsys):
    from llmbench.reporting.dashboard import copy_latest
    src = tmp_path / "results" / "benchmark_dashboard_x.html"
    src.parent.mkdir()
    src.write_text("<html>")
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)                                             # like a root-owned file on a share
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as root: permissions are not enforced")
        out = copy_latest(src, locked / "benchmark_dashboard.html")
    finally:
        os.chmod(locked, 0o700)
    assert "not updated" in str(out) and "could not update" in capsys.readouterr().out
    assert copy_latest(src, tmp_path / "ok.html") == tmp_path / "ok.html"


def test_page_serves_the_newest_dashboard_and_warns_when_remote(ui, monkeypatch):
    u = ui()
    res = u.dir / "results"
    old = server.SCRIPT_DIR / "benchmark_dashboard.html"
    (res / "benchmark_dashboard_20990101_000000.html").write_text("<html>newest</html>")
    os.utime(res / "benchmark_dashboard_20990101_000000.html", (4102444800, 4102444800))
    assert server.newest_dashboard("dashboard").name == "benchmark_dashboard_20990101_000000.html"
    st, _, body = _req(u.port, "GET", "/dashboard")
    assert st == 200 and "newest" in body
    (res / "benchmark_compare_20990101_000000.html").write_text("<html>cmp</html>")
    os.utime(res / "benchmark_compare_20990101_000000.html", (4102444800, 4102444800))
    assert "cmp" in _req(u.port, "GET", "/compare")[2]
    assert _req(u.port, "GET", "/api/state")[2]["remote"] is None or "another machine" in \
        _req(u.port, "GET", "/api/state")[2]["remote"]
    u.env.write_text(u.env.read_text() + "LLAMA_SWAP_URL=http://192.0.2.10:8000\n")
    assert "another machine" in _req(u.port, "GET", "/api/state")[2]["remote"]


def test_ui_host_url_is_cut_to_the_address(tmp_path, monkeypatch):
    from llmbench.ui import server
    for raw, want in [("http://192.168.2.200", "192.168.2.200"), ("http://192.168.2.200:41010/", "192.168.2.200"),
                      ("192.168.2.200:41010", "192.168.2.200"), ("0.0.0.0", "0.0.0.0"), ("[::1]:41010", "::1"),
                      ("::", "::"), ("https://myserver.local/x?y", "myserver.local")]:
        assert server.clean_host(raw) == want, raw
    from llmbench.envspec import _value_problem
    assert "192.168.2.200" in _value_problem("http://192.168.2.200", "str", "UI_HOST")
    assert _value_problem("0.0.0.0", "str", "UI_HOST") is None


def test_bind_hint_names_the_cause():
    import errno
    import socket
    from llmbench.ui.server import bind_hint
    assert "not an address of this machine" in bind_hint(socket.gaierror(-2, "Name or service not known"), "x")
    assert "not an address of this machine" in bind_hint(OSError(errno.EADDRNOTAVAIL, "no"), "10.9.9.9")
    assert "UI_PORT" in bind_hint(OSError(errno.EADDRINUSE, "in use"), "0.0.0.0")


def test_failed_save_note_is_replaced_and_cleared():
    """A 'Not saved' message carries the tag 'save', so the next save result replaces it."""
    html = (Path(__file__).resolve().parents[1] / "llmbench" / "ui" / "settings.html").read_text()
    assert "function clearNotes(tag)" in html
    assert "Fix the values marked in red first.'}`, true, 'save')" in html
    assert ".`, false, 'save'); await load();" in html


def test_demo_run_never_touches_real_outputs(tmp_path):
    """--demo writes to OUTPUT_DIR/demo: results.json, --reload, --estimate and the page keep the real run."""
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "results"
    out.mkdir()
    (out / "results.json").write_text('{"real": 1}')
    latest = root / "benchmark_dashboard.html"
    before = latest.read_bytes() if latest.exists() else None
    env = {**os.environ, "OUTPUT_DIR": str(out)}                # the shell wins over .env
    r = subprocess.run([sys.executable, str(root / "benchmark.py"), "--demo"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "results.json").read_text() == '{"real": 1}'
    assert not list(out.glob("benchmark_*.json")) and list((out / "demo").glob("benchmark_*.json"))
    assert (latest.read_bytes() if latest.exists() else None) == before
    assert "not replaced (demo run)" in r.stdout


# --------------------------------------------------------------------------- a run that fails at once is reported as failed
def test_failed_run_is_reported_not_called_ended(ui):
    from llmbench.ui import server
    u = ui()
    out_dir = u.dir / "results"
    log = out_dir / "benchmark_20261110_160728.log"
    log.write_text("[eval] The 'datasets' package is missing. Run: pip install -r requirements.txt\n")
    old = {"started": "2026-11-10 16:07:20", "state": "complete", "log": None}
    (out_dir / "run_status.json").write_text(json.dumps(old))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    server._CHILD.update(proc=done, started="2026-11-10 16:07:28")
    try:
        # the older run's 'complete' is not the run this page just started
        st = _req(u.port, "GET", "/api/run")[2]
        assert not st["running"] and "status" not in st and st["exited"] == 0
        (out_dir / "run_status.json").write_text(json.dumps({"started": "2026-11-10 16:07:28", "state": "failed",
                                                             "phase": "starting", "log": str(log)}))
        st = _req(u.port, "GET", "/api/run")[2]
        assert not st["running"] and st["status"]["state"] == "failed" and st["log"] == str(log)
        lg = _req(u.port, "GET", "/api/log?offset=0")[2]
        assert "datasets' package is missing" in lg["text"] and lg["status"]["state"] == "failed"
    finally:
        server._CHILD.update(proc=None, started=None)


def test_check_flags_missing_datasets(monkeypatch, capsys):
    import builtins
    from llmbench import config, diagnostics
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "datasets":
            raise ImportError(name)
        return real(name, *a, **k)
    cfg = load_config()
    cfg.eval_datasets = ["mmlu_pro", "toolcall"]
    monkeypatch.setattr(builtins, "__import__", fake)
    diagnostics._check_eval_client(cfg)
    assert "NOT AVAILABLE" in capsys.readouterr().out


# --------------------------------------------------------------------------- v2.3.2 run protection
@pytest.mark.skipif(os.name != "posix", reason="uses the POSIX run lock helpers of the tests above")
def test_save_and_defaults_are_refused_while_a_run_is_active(ui):
    """.env is not rewritten under a live run, whoever asks (the page locks its buttons; this is the server side)."""
    u = ui()
    vals = {"OUTPUT_DIR": str(u.dir / "results"), "EVAL_PROFILE": "deep"}
    before = u.env.read_text()
    child = _sleeper()
    try:
        (u.dir / "results" / cli.LOCK_NAME).write_text(json.dumps({"pid": child.pid, "started": "now", "log": None}))
        st, _, out = _req(u.port, "POST", "/api/save", {"values": vals})
        assert st == 409 and out["error"].startswith("Run active"), out
        st, _, out = _req(u.port, "POST", "/api/defaults", {})
        assert st == 409 and out["error"].startswith("Run active"), out
        assert u.env.read_text() == before and not (u.dir / "my.env.bak").exists()
        assert _req(u.port, "POST", "/api/check", {"values": vals})[0] == 200          # reading and checking still work
        st, _, log = _req(u.port, "GET", "/api/log?offset=0")                         # no log file yet: still names the run
        assert log["running"] and log["pid"] == child.pid and log["started"] == "now"
    finally:
        child.kill()
        child.wait()
    st, _, out = _req(u.port, "POST", "/api/save", {"values": vals})                    # the run is over: saved
    assert st == 200 and out["saved"], out
    assert envfile.read_values(u.env)["EVAL_PROFILE"] == "deep"


def test_settings_page_has_the_run_lock_and_context_tips():
    html = (Path(server.__file__).with_name("settings.html")).read_text(encoding="utf-8")
    assert ("Suggested context sizes: 4096 (4k), 8192 (8k), 16384 (16k), 32768 (32k), 65536 (64k), 98304 (96k), "
            "131072 (128k), 262144 (256k), 524288 (512k), 1048576 (1M)") in html
    assert "Existing run will stop, are you sure?" in html and "toast('Run active', t)" in html
    assert "#runst.running{color:var(--red);font-weight:700}" in html
    for b in ("save", "defaults", "favload", "favsave", "favdel", "reload", "runall"):
        assert f"'{b}'" in html.split("const LOCK_IDS = ", 1)[1].split("]", 1)[0], b


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="Node.js not installed")
def test_context_tip_goes_on_context_sizes_only(tmp_path):
    """isCtxKey: token-size settings get the tip; percentages, actions and switches that merely mention ctx do not."""
    import re
    html = (Path(server.__file__).with_name("settings.html")).read_text(encoding="utf-8")
    fn = re.search(r"const isCtxKey = .*", html).group(0)
    yes = ["AGENT_CTX_MIN", "AGENT_CTX_CURVE", "LONGCTX_LENGTHS", "PREFILL_CTX_TESTS", "PREFILL_DEPTH",
           "AGENT_PROMPT_TOKENS", "MIN_CTX", "STRATA_CTX", "BUDGET_SERVER_MIN_CTX"]
    no = ["AGENT_CTX_PASS_PCT", "MAX_CTX_VRAM_PCT", "MIN_CTX_ACTION", "EVAL_LONGCTX", "LONGCTX_KV_MIN", "LONGCTX_KEYS",
          "EVAL_THINK_BUDGET", "MAX_TOKENS_TECH", "UI_TOKEN", "HF_TOKEN"]
    f = tmp_path / "k.js"
    f.write_text(fn + f"\nconsole.log(JSON.stringify([{json.dumps(yes)}.filter(k=>!isCtxKey(k)), {json.dumps(no)}.filter(isCtxKey)]))")
    r = subprocess.run(["node", str(f)], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == [[], []], r.stdout + r.stderr


# --------------------------------------------------------------------------- Strata fallback in the models list
def test_models_list_falls_back_to_strata(monkeypatch):
    calls = []

    def fake(url, key, strata=False):
        calls.append((url, key, strata))
        if "8000" in url:
            raise ConnectionError("refused")
        return ["Qwen3.8-Next-Flash"]
    monkeypatch.setattr(server, "_fetch_model_ids", fake)
    vals = {"LLAMA_SWAP_URL": "http://h:8000", "STRATA_URL": "http://h:8080", "STRATA_API_KEY": "k"}
    d = server.list_server_models(vals)
    assert d["backend"] == "Strata" and d["url"] == "http://h:8080" and "llama-swap did not answer" in d["note"]
    assert [m["id"] for m in d["models"]] == ["Qwen3.8-Next-Flash"]
    assert calls[-1] == ("http://h:8080", "k", True)
    # llama-swap answering wins, no note
    monkeypatch.setattr(server, "_fetch_model_ids", lambda u, k, strata=False: ["a", "b"])
    d = server.list_server_models(vals)
    assert d["backend"] == "llama-swap" and d["note"] is None and len(d["models"]) == 2
    # no Strata configured: the llama-swap error is shown as before
    monkeypatch.setattr(server, "_fetch_model_ids", fake)
    d = server.list_server_models({"LLAMA_SWAP_URL": "http://h:8000"})
    assert d["models"] == [] and "refused" in d["error"]
