"""v2.3.6: the eval runner had its second half pasted in twice (a merge of the pause work onto the v2.3.3 funnel).
Python keeps the LAST definition of a name, so the run used the older EvalPhase._gate: hard-rule models were logged as
'drop' and stored as 'screened out', and `eval_hard_fail` was never set on their rows."""
import ast
import contextlib
import io
from pathlib import Path

from llmbench.evals.runner import EvalPhase

from test_v23_agent import _row, cfg  # noqa: F401  (fixture + row builder)

PKG = Path(__file__).resolve().parent.parent / "llmbench"


def _duplicates(body):
    seen, dup = set(), []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                dup.append(f"{node.name} (line {node.lineno})")
            seen.add(node.name)
            if isinstance(node, ast.ClassDef):
                dup += [f"{node.name}.{d}" for d in _duplicates(node.body)]
    return dup


def test_no_module_defines_a_function_or_class_twice():
    bad = {}
    for path in sorted(PKG.rglob("*.py")):
        dup = _duplicates(ast.parse(path.read_text(encoding="utf-8")).body)
        if dup:
            bad[str(path.relative_to(PKG))] = dup
    assert not bad, f"defined twice (the later one silently wins): {bad}"


def test_the_run_uses_the_v233_gate(cfg):
    lc = {"64k": 100.0, "128k": 100.0}
    a = _row("A-9B-Q4_K_M", mmlu=90, he=90, tools=100.0, lc=dict(lc), n=11)
    b = _row("B-9B-Q4_K_M", mmlu=85, he=85, tools=100.0, lc=dict(lc), n=11)
    flint = _row("flint-9B-Q4_K_M", mmlu=100.0, he=100.0, tools=90.6, lc={"64k": 50.0, "128k": 0.0}, n=11)
    rows = [a, b, flint]
    cfg.eval_leaders = 2
    phase = EvalPhase.__new__(EvalPhase)               # only what _gate reads
    phase.cfg, phase.results, phase.keys = cfg, rows, ["mmlu_pro", "humaneval", "toolcall"]
    phase.stages, phase.funnel_report = [("quick", 11), ("basic", 24)], []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        alive = phase._gate(rows, 0)
    log = out.getvalue()
    assert [r["model"] for r in alive] == ["A-9B-Q4_K_M", "B-9B-Q4_K_M"]
    assert "NO AGENT SCORE" in log and "KEEP" in log and "margin 3" in log
    assert flint.get("eval_hard_fail", "").startswith("long-ctx") and not flint.get("eval_screened")
    assert {g["model"]: g["state"] for g in phase.funnel_report}["flint-9B-Q4_K_M"] == "no_agent"
