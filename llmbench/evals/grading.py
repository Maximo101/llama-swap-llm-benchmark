"""Turning a model reply into a verdict: program assembly for each code dataset + dispatch."""
from __future__ import annotations

import base64
import io
import json
import pickle
import random
import re
import zlib

from ..config import EVAL_SPECS
from .answers import extract_choice, extract_code, strip_main_guard
from .mathcheck import extract_math_answer, math_equal
from .sandbox import run_in_sandbox

def humaneval_program(row, code):
    code = strip_main_guard(code)
    entry = row["entry_point"]
    if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", code):
        # Keep the prompt's imports/helper functions; stub its own (unfinished) function, then let the
        # model's full definition override the stub.
        body = row["prompt"].rstrip() + "\n    pass\n\n" + code
    else:                                   # the model returned just a function body: continue the prompt
        body = row["prompt"] + code
    return body + "\n\n" + row["test"] + f"\n\ncheck({entry})\n"


def mbpp_program(row, code):
    return ((row.get("test_setup_code") or "") + "\n" + strip_main_guard(code) + "\n\n"
            + "\n".join(row["test_list"]) + "\n")


class _SafeUnpickler(pickle.Unpickler):
    """LiveCodeBench stores private tests as base64(zlib(pickle(json_string))). A pickle of a plain string
    needs no classes, so every global lookup is refused: a malicious pickle cannot run code."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _decode_lcb_private(raw):
    if not raw:
        return []
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        obj = _SafeUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).load()
        return json.loads(obj) if isinstance(obj, (str, bytes)) else obj
    except Exception:
        return []


def lcb_tests(cfg, item):
    """Public tests + (a seeded sample of) the private tests, decoded once and cached on the item."""
    if item["tests"] is None:
        pub = json.loads(item["row"].get("public_test_cases") or "[]")
        try:
            raw = item["ds"].select_columns(["private_test_cases"])[item["idx"]]["private_test_cases"]
        except Exception:
            raw = ""
        priv = _decode_lcb_private(raw)
        cap = cfg.eval_lcb_max_tests
        if cap and len(pub) + len(priv) > cap:
            pub = pub[:cap]
            priv = random.Random(f"{cfg.eval_seed}:{item['id']}").sample(priv, min(len(priv), cap - len(pub)))
        item["tests"] = [{"input": t["input"], "output": t["output"]} for t in pub + priv
                         if isinstance(t, dict) and "input" in t and "output" in t]
    return item["tests"]


_LCB_PRELUDE = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
                "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
                "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
                "from operator import *\nfrom typing import *\nfrom builtins import *\n"
                "import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, "
                "itertools, functools, operator, sys, json\n")


def lcb_program(item, code):
    if item["mode"] != "call":
        return code
    future = [l for l in code.splitlines() if l.strip().startswith("from __future__")]
    body = "\n".join(l for l in code.splitlines() if l not in future)
    return ("\n".join(future) + "\n" + _LCB_PRELUDE + body + "\n\n"
            "_args = [json.loads(_l) for _l in sys.stdin.read().split('\\n') if _l.strip() != '']\n"
            f"print(json.dumps(Solution().{item['func']}(*_args)))\n")


def grade_item(cfg, key, item, text, finish=None):
    """Returns (correct, parsed_answer_or_None, reason). This is CPU/IO work (regexes, and for code a sandbox run
    that can take seconds), so the async runner calls it from a worker thread, never on the event loop.
    `finish` is the server's finish_reason: a reply that was cut off ("length") never gets the last-number
    fallback for GSM8K, because the last number of an unfinished chain of thought is not an answer."""
    kind = EVAL_SPECS[key]["kind"]
    if kind == "tool":
        from .toolcall import grade_toolcall
        return grade_toolcall(item, text, cfg.eval_toolcall_strict)
    if kind == "qa":
        pred = extract_choice(text, item["letters"])
        if pred is None:
            return False, None, "no answer found"
        return pred == item["gold"], pred, ("" if pred == item["gold"] else f"expected {item['gold']}")
    if kind == "math":
        pred = extract_math_answer(text, numeric_fallback=(key == "gsm8k" and finish != "length"))
        if pred is None:
            return False, None, "no answer found"
        ok = math_equal(pred, item["gold"])
        return ok, pred, ("" if ok else f"expected {item['gold']}")
    t = cfg.eval_code_timeout
    if key == "humaneval":
        code = extract_code(text, item["row"]["entry_point"])
        ok, why = run_in_sandbox(cfg, humaneval_program(item["row"], code), "script", [], t)
    elif key == "mbpp":
        code = extract_code(text)
        ok, why = run_in_sandbox(cfg, mbpp_program(item["row"], code), "script", [], t)
    else:
        code = extract_code(text)
        tests = lcb_tests(cfg, item)
        if not tests:
            return False, None, "no test cases available"
        ok, why = run_in_sandbox(cfg, lcb_program(item, code), item["mode"], tests, t)
    return ok, None, why
