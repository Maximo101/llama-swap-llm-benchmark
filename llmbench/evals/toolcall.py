"""Grading for the tool-calling eval (AST-style matching, as in BFCL)."""
from __future__ import annotations

import json
import re

_TEXT_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def encode_reply(content, tool_calls):
    """The runner hands graders a string; tool replies are passed as this JSON."""
    return json.dumps({"content": content or "", "tool_calls": tool_calls or []}, ensure_ascii=False)


def _norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)                # "20" and 20 are the same argument
        except ValueError:
            return s.lower().rstrip(".")
    if isinstance(v, list):
        return sorted((_norm(x) for x in v), key=repr)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _same(a, b):
    a, b = _norm(a), _norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    return a == b


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    v = json.loads(raw)
    if not isinstance(v, dict):
        raise ValueError("arguments are not a JSON object")
    return v


def native_calls(tool_calls):
    """[(name, args dict)] from message.tool_calls. Raises ValueError for unparsable arguments."""
    out = []
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        out.append((fn.get("name") or "", _parse_args(fn.get("arguments"))))
    return out


def text_calls(content):
    """Tool calls written into the text instead of the API field (Hermes <tool_call> tags or a JSON block)."""
    found = []
    for m in list(_TEXT_CALL.finditer(content or "")) or list(_JSON_BLOCK.finditer(content or "")):
        try:
            v = json.loads(m.group(1))
        except ValueError:
            continue
        for c in v if isinstance(v, list) else [v]:
            if isinstance(c, dict) and c.get("name"):
                args = c.get("arguments", c.get("parameters", {}))
                try:
                    found.append((c["name"], _parse_args(args)))
                except ValueError:
                    continue
    return found


def _match_one(exp, name, args, schema):
    """'' if the call matches the expectation, else why not."""
    if name != exp["name"]:
        return f"called {name or '?'}, expected {exp['name']}"
    props = (schema.get(name) or {}).get("parameters", {}).get("properties", {})
    unknown = [k for k in args if k not in props]
    if unknown:
        return f"{name}: unknown argument(s) {', '.join(unknown)}"
    for key, allowed in exp["args"].items():
        if key not in args:
            if None in allowed:
                continue
            return f"{name}: missing {key}"
        if not any(a is not None and _same(args[key], a) for a in allowed):
            return f"{name}: {key}={args[key]!r}, expected one of {[a for a in allowed if a is not None]!r}"
    return ""


def _match_all(expect, calls, schema):
    """Every expected call matched by a distinct actual call (order does not matter)."""
    if len(calls) != len(expect):
        return f"{len(calls)} call(s), expected {len(expect)}"
    left = list(calls)
    for exp in expect:
        reasons = []
        for i, (name, args) in enumerate(left):
            why = _match_one(exp, name, args, schema)
            if not why:
                left.pop(i)
                break
            reasons.append(why)
        else:
            return reasons[0] if len(reasons) == 1 else "; ".join(dict.fromkeys(reasons))
    return ""


def grade_toolcall(item, text, strict=True):
    """-> (correct, summary of what was called, reason)."""
    try:
        reply = json.loads(text)
    except (TypeError, ValueError):
        reply = {"content": text or "", "tool_calls": []}
    schema = {t["function"]["name"]: t["function"] for t in item["tools"]}
    try:
        calls, mode = native_calls(reply.get("tool_calls")), "native"
    except ValueError as e:
        return False, None, f"invalid JSON in tool-call arguments ({e})"
    if not calls:
        found = text_calls(reply.get("content"))
        if found and strict:
            return False, ", ".join(n for n, _ in found), \
                "tool call written as text, not as a native tool_call (check --jinja and the chat template)"
        calls, mode = found, "text"
    pred = ", ".join(n for n, _ in calls) or "(no call)"
    expect = item["expect"]
    if not expect:
        return (not calls), pred, ("" if not calls else "called a tool for a question that needs none")
    if not calls:
        return False, pred, "no tool call"
    why = _match_all(expect, calls, schema)
    return (not why), pred + ("" if mode == "native" else " [text]"), why
