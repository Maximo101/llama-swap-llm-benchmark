"""v2.2.1: .env.example is the documented list of settings; --check-env compares a user's .env with it."""
import os
import sys

import pytest

from llmbench import config, envspec


def _entries():
    return envspec.parse_example()


def test_example_documents_every_setting_the_code_reads():
    entries = _entries()
    documented = {e.key for e in entries if not e.family}
    families = [e for e in entries if e.family]
    code = envspec.code_settings()
    missing = sorted(k for k in code if k not in documented and not any(f.pattern().match(k) for f in families))
    assert not missing, f"read by the code but not in .env.example: {missing}"
    extra = sorted(k for k in documented if k not in code)
    assert not extra, f"in .env.example but never read by the code: {extra}"


def test_example_defaults_match_the_code():
    code = envspec.code_settings()
    wrong = [(e.key, e.default, code[e.key][0]) for e in _entries()
             if not e.family and e.key in code and not envspec.same_default(e.default, *code[e.key])]
    assert not wrong, f"documented default differs from the code: {wrong}"


def test_example_lines_follow_the_format():
    raw = envspec.EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    entries = _entries()
    assert {e.part for e in entries} == {1, 2, 3}
    keys = [e.key for e in entries]
    assert len(keys) == len(set(keys)), "a setting is listed twice"
    for e in entries:
        line = raw[e.line - 1]
        if e.active:
            assert "[default:" in line, f"active line without [default: ...]: {e.key}"
            assert not e.value.startswith("#"), e.key
        else:
            assert "[default:" not in line, f"commented line must show its default as the value: {e.key}"
        assert e.part == 3 or e.active or e.default == "" or e.family, \
            f"PART 1/2 settings with a non-empty default should be active lines: {e.key}"
        assert e.desc or e.key in ("STRATA_API_KEY", "MAX_TOKENS_TECH", "MAX_TOKENS_LOGIC", "RETRIES",
                                   "EVAL_CODE_MEM_MB", "EVAL_CODE_DOCKER_IMAGE", "CURRENCY_SYMBOL",
                                   "TECH_PROMPT", "LOGIC_PROMPT") or e.key.endswith("_REVISION"), \
            f"no description: {e.key}"
    assert all(e.part == 3 for e in entries if e.key.endswith("_REVISION"))


def test_example_reads_the_same_with_python_dotenv():
    dotenv = pytest.importorskip("dotenv")
    ours = {k: v for k, v, _ in envspec.read_env_file(envspec.EXAMPLE_PATH)}
    assert dotenv.dotenv_values(envspec.EXAMPLE_PATH) == ours
    assert ours["EXCLUDE_MODELS"] == "" and ours["EVAL_SERVER_SAMPLING_MODELS"] == ""


def test_example_values_load(monkeypatch):
    for k, v, _ in envspec.read_env_file(envspec.EXAMPLE_PATH):
        monkeypatch.setenv(k, v)
    cfg = config.load_config()
    assert cfg.eval_profile == "funnel" and cfg.eval_funnel_final == "moderate"
    assert set(cfg.eval_datasets) == {"mmlu_pro", "humaneval", "mbpp", "toolcall"}
    assert cfg.budget["name"] == "server" and cfg.exclude_models == () and cfg.eval_server_sampling == ""
    assert cfg.prefill_ctx_tests == (8192, 32768, 65536) and cfg.longctx_lengths == (65536, 131072)


def test_dotenv_empty_value_with_comment_is_the_comment(tmp_path):
    p = tmp_path / ".env"
    p.write_text('A=   # c\nB=""   # c\nC=x  # c\nD=\nexport E=1\n', encoding="utf-8")
    assert envspec.read_env_file(p) == [("A", "# c", 1), ("B", "", 2), ("C", "x", 3), ("D", "", 4), ("E", "1", 5)]


def test_check_env_finds_problems(tmp_path):
    p = tmp_path / ".env"
    p.write_text("\n".join([
        "LLAMA_SWAP_URL=http://server:8000",
        "EVAL_TOOLCAL=true",                        # typo
        "EVAL_STREAM=tue",                          # bad boolean
        "AGENT_ACC_TOL=   # empty = 1 SE",          # dotenv reads the comment
        "EVAL_CONCURRENCY=four",                    # not a number
        "EVAL_PROFILE=quick",
        "EVAL_PROFILE=funnel",                      # duplicate, last wins
        "BUDGET_PROFILE=server",
        "BUDGET_SERVER_VRAM_RESERVE_GB=1.75",
        "BUDGET_SERVER_MIN_TSP=40",                 # unknown rule
        "BUDGET_MAIN_MIN_TPS=30",                   # other profile: a note
        "EVAL_THINK_BUDGET_GSM8K=2048",
        "EVAL_THINK_BUDGET_GSM8=2048",              # unknown dataset
        "STRATA_SWAP_URL=http://x",                 # legacy name
        "EVAL_SEED=42",                             # = default
        "GPU_POLL_MS=300",                          # changed, optional
    ]) + "\n", encoding="utf-8")
    rep = envspec.check_env(p, environ={"EVAL_SEED": "7"})
    unknown = {k: s for k, _, _, s in rep.unknown}
    assert unknown["EVAL_TOOLCAL"] == "EVAL_TOOLCALL?"
    assert "BUDGET_SERVER_MIN_TSP" in unknown and "MIN_TPS" in unknown["BUDGET_SERVER_MIN_TSP"]
    assert unknown["EVAL_THINK_BUDGET_GSM8"] == "EVAL_THINK_BUDGET_GSM8K?"
    invalid = {k for k, *_ in rep.invalid}
    assert invalid == {"EVAL_STREAM", "AGENT_ACC_TOL", "EVAL_CONCURRENCY"}
    assert rep.duplicates == [("EVAL_PROFILE", [6, 7], "funnel")]
    assert rep.legacy == [("STRATA_SWAP_URL", "STRATA_URL", 14)]
    assert rep.shell == [("EVAL_SEED", "7", "42")]
    assert any("BUDGET_MAIN_MIN_TPS" in n for n in rep.info)
    assert any("BUDGET_SERVER_VRAM_RESERVE_GB=1.75" in n and "2" in n for n in rep.info)
    missing = {e.key for e in rep.missing}
    assert "RUN_EVAL_TESTS" in missing and "LLAMA_SWAP_URL" not in missing and "GPU_POLL_MS" not in missing
    changed = {k: d for k, _, d, _ in rep.changed}
    assert changed["GPU_POLL_MS"] == "200" and changed["LLAMA_SWAP_URL"] == "http://localhost:8080"
    assert ("EVAL_SEED", "42") in rep.at_default
    assert envspec.print_report(rep) == 1


def test_check_env_clean_copy_of_the_example(tmp_path, capsys):
    p = tmp_path / ".env"
    p.write_text(envspec.EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    rep = envspec.check_env(p, environ={})
    assert rep.problems == 0 and not rep.missing and not rep.duplicates
    assert envspec.print_report(rep) == 0
    assert "No problems found" in capsys.readouterr().out
    assert envspec.print_report(envspec.check_env(tmp_path / "nope.env", environ={})) == 1


def test_fallback_reader_last_line_wins_like_python_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLMB_T_DUP=1\nLLMB_T_DUP=2\nLLMB_T_Q=\"a # b\"  # c\n", encoding="utf-8")
    monkeypatch.setattr(config, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(envspec, "SCRIPT_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "dotenv", None)          # import fails -> built-in reader
    monkeypatch.delenv("LLMB_T_DUP", raising=False)
    monkeypatch.delenv("LLMB_T_Q", raising=False)
    config._load_dotenv()
    import os
    assert os.environ["LLMB_T_DUP"] == "2" and os.environ["LLMB_T_Q"] == "a # b"
    monkeypatch.delenv("LLMB_T_DUP")
    monkeypatch.delenv("LLMB_T_Q")


def test_comment_only_value_is_read_as_empty_by_the_tool(tmp_path, monkeypatch):
    from llmbench import config, envfile
    p = tmp_path / ".env"
    p.write_text('UI_TOKEN=   # token for the page   [default: empty]\nHASH=#abc\nQ="# kept"\nX=1  # c\n', encoding="utf-8")
    assert envspec.read_env_file(p, blank_comments=True) == [("UI_TOKEN", "", 1), ("HASH", "#abc", 2),
                                                             ("Q", "# kept", 3), ("X", "1", 4)]
    assert envfile.read_values(p)["UI_TOKEN"] == ""                          # the settings page sees it empty too
    for k in ("UI_TOKEN", "HASH", "Q", "X"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "ENV_FILE", p)
    config._load_dotenv()
    assert os.environ["UI_TOKEN"] == "" and os.environ["HASH"] == "#abc" and os.environ["X"] == "1"
    for k in ("UI_TOKEN", "HASH", "Q", "X"):
        monkeypatch.delenv(k, raising=False)
