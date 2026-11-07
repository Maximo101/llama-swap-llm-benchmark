"""Evaluation datasets: loading, sampling, prompts and item builders."""
from __future__ import annotations

import json
import random
import re
import string
import sys
import time

from ..config import EVAL_SPECS, PROFILE_ROWS
from .sandbox import resolve_sandbox

def load_eval_split(cfg, key, offline_first=True):
    """Load one dataset split with cache_dir=cfg.eval_cache_dir (./tests). Normal runs try the local cache
    first (no network, milliseconds) and only download if the dataset is not cached yet."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[eval] The 'datasets' package is missing. Run: pip install -r requirements.txt")
    spec, cache = EVAL_SPECS[key], str(cfg.eval_cache_dir)
    ds = None
    if offline_first:
        try:
            from datasets import DownloadConfig
            ds = load_dataset(spec["repo"], spec.get("config"), cache_dir=cache,
                              download_config=DownloadConfig(local_files_only=True))
        except Exception:
            ds = None
    if ds is None:
        if offline_first:
            print(f"  [eval] {spec['label']} is not in {cache} yet - downloading it now "
                  "(run --download-datasets next time to do this up front)")
        ds = load_dataset(spec["repo"], spec.get("config"), cache_dir=cache)
    if isinstance(ds, dict):                       # DatasetDict
        ds = ds[spec["split"]] if spec["split"] in ds else ds[next(iter(ds))]
    return ds


def download_datasets(cfg):
    """--download-datasets: cache every dataset marked true in .env into ./tests and exit."""
    if not cfg.eval_datasets:
        print("No datasets are enabled. Set at least one of "
              + ", ".join(s["env"] for s in EVAL_SPECS.values() if not s.get("builtin")) + " to true in .env.")
        return
    cfg.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching {len(cfg.eval_datasets)} dataset(s) into {cfg.eval_cache_dir}")
    failed = []
    for key in cfg.eval_datasets:
        spec = EVAL_SPECS[key]
        if spec.get("builtin"):
            print(f"  {spec['label']}: built into the tool, nothing to download")
            continue
        print(f"  {spec['label']} ({spec['repo']}) ...", flush=True)
        if key == "livecodebench":
            print("    note: LiveCodeBench is large (about 4 GB); this can take a while")
        try:
            ds = load_eval_split(cfg, key, offline_first=False)
            print(f"    ok: {len(ds)} rows in split '{spec['split']}'")
        except SystemExit:
            raise
        except Exception as e:
            failed.append(key)
            print(f"    FAILED: {e.__class__.__name__}: {e}")
    if failed:
        sys.exit(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
    print("\nDone. Normal runs now read these from local disk.")


def sample_indices(n, k, seed, key):
    """Deterministic sample, so every model sees exactly the same rows."""
    if k is None or k >= n:
        return list(range(n))
    return sorted(random.Random(f"{seed}:{key}").sample(range(n), k))


# ----- prompts --------------------------------------------------------------

_QA_FORMAT = ("Think step by step, then finish your reply with a final line in exactly this form:\n"
              "Answer: X\nwhere X is a single letter ({letters}).")


def prompt_gpqa(row):
    # The dataset's question text already contains its four options.
    return (f"Answer the following multiple choice question. The options are labelled A, B, C and D.\n"
            f"{_QA_FORMAT.format(letters='A, B, C or D')}\n\n{row['question'].strip()}")


def prompt_mmlu_pro(row):
    letters = string.ascii_uppercase[:len(row["options"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    cat = (row.get("category") or "general knowledge").strip()
    return (f"The following is a multiple choice question about {cat}.\n"
            f"{_QA_FORMAT.format(letters=', '.join(letters[:-1]) + ' or ' + letters[-1])}\n\n"
            f"Question: {row['question'].strip()}\nOptions:\n{opts}")


def prompt_humaneval(row):
    return ("Complete the following Python function. Reply with the complete function (signature included, plus any "
            "imports it needs) in a single ```python code block. Do not include tests or example usage.\n\n"
            f"```python\n{row['prompt'].rstrip()}\n```")


def prompt_mbpp(row):
    tests = "\n".join(row["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {row['text'].strip()} "
            f"Your code should pass these tests:\n\n{tests}\n\n"
            "Reply with the complete Python code in a single ```python code block. Do not include the tests.")


def prompt_lcb(row, mode, func):
    if mode == "call":
        fmt = ("Use the following starter code and complete it:\n```python\n"
               f"{(row.get('starter_code') or '').rstrip()}\n```")
    else:
        fmt = "Read the input from standard input and write the answer to standard output."
    return ("You are an expert competitive programmer. Solve the following problem in Python 3.\n\n"
            f"### Problem: {row['question_title']}\n{row['question_content'].strip()}\n\n### Format\n{fmt}\n\n"
            "Reply with the complete solution in a single ```python code block.")


def prompt_gsm8k(row):
    return ("Solve the following grade-school math word problem. Think step by step, then finish your reply with a "
            "final line in exactly this form:\nAnswer: <number>\nwhere <number> is the final numeric answer only "
            "(no units, no commas).\n\n"
            f"Problem: {row['question'].strip()}")


def prompt_math(row):
    return ("Solve the following math problem. Think step by step, then finish your reply with a final line in "
            "exactly this form:\nAnswer: <answer>\nwhere <answer> is the final simplified result only, written in "
            "LaTeX where appropriate (for example \\frac{3}{4} or (1, 2)), without units.\n\n"
            f"Problem: {row['problem'].strip()}")


# ----- item builders --------------------------------------------------------

def _lcb_mode(row):
    """('call', func_name) for LeetCode-style problems, ('stdin', None) for AtCoder/Codeforces ones."""
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except ValueError:
        meta = {}
    func = meta.get("func_name") if isinstance(meta, dict) else None
    if not func and (row.get("starter_code") or "").strip():
        m = re.search(r"def\s+(\w+)\s*\(\s*self", row["starter_code"])
        func = m.group(1) if m else None
    return ("call", func) if func and (row.get("starter_code") or "").strip() else ("stdin", None)


def build_items(cfg, key, ds, idxs):
    items = []
    if key == "livecodebench":
        # keep the (huge, compressed) private tests out of memory: read only the light columns here
        light = ds.select_columns(["question_title", "question_content", "starter_code", "metadata",
                                   "public_test_cases", "question_id"])
        for i in idxs:
            row = light[i]
            mode, func = _lcb_mode(row)
            items.append({"id": str(row["question_id"]), "prompt": prompt_lcb(row, mode, func), "row": row,
                          "ds": ds, "idx": i, "mode": mode, "func": func, "tests": None})
        return items
    for i in idxs:
        row = ds[i]
        if key == "gpqa":
            gold = str(row["answer"]).strip().upper()[:1]
            items.append({"id": f"gpqa-{i}", "prompt": prompt_gpqa(row), "gold": gold, "letters": "ABCD"})
        elif key == "mmlu_pro":
            letters = string.ascii_uppercase[:len(row["options"])]
            gold = str(row.get("answer") or "").strip().upper()[:1]
            if gold not in letters and row.get("answer_index") is not None:
                gold = letters[int(row["answer_index"])]
            items.append({"id": str(row.get("question_id", i)), "prompt": prompt_mmlu_pro(row), "gold": gold,
                          "letters": letters})
        elif key == "humaneval":
            items.append({"id": row["task_id"], "prompt": prompt_humaneval(row), "row": row})
        elif key == "mbpp":
            items.append({"id": f"mbpp-{row['task_id']}", "prompt": prompt_mbpp(row), "row": row})
        elif key == "gsm8k":          # reference solution ends with "#### <answer>"
            gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
            items.append({"id": f"gsm8k-{i}", "prompt": prompt_gsm8k(row), "gold": gold})
        elif key == "math":
            items.append({"id": str(row.get("unique_id") or f"math-{i}"), "prompt": prompt_math(row),
                          "gold": str(row["answer"]).strip()})
    return items


def prepare_eval_data(cfg):
    """Load + sample every enabled dataset up front, so a missing package, dataset or cache fails
    before the (long) speed test rather than after it."""
    rows, data, info = PROFILE_ROWS[cfg.eval_profile], {}, {}
    cfg.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading evaluation datasets (profile {cfg.eval_profile}: "
          f"{'all rows' if rows is None else f'{rows} rows each'}; cache {cfg.eval_cache_dir})")
    for key in cfg.eval_datasets:
        spec = EVAL_SPECS[key]
        t = time.time()
        if spec.get("builtin"):
            from .toolcall_cases import build_items as builtin_items
            pool = builtin_items()
            data[key] = [pool[i] for i in sample_indices(len(pool), rows, cfg.eval_seed, key)]
            info[key] = {"rows": len(data[key]), "available": len(pool)}
            print(f"  {spec['label']}: {len(data[key])} of {len(pool)} cases (built in)")
            continue
        try:
            ds = load_eval_split(cfg, key)
            idxs = sample_indices(len(ds), rows, cfg.eval_seed, key)
            data[key] = build_items(cfg, key, ds, idxs)
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"[eval] Could not load {spec['label']} ({spec['repo']}): {e.__class__.__name__}: {e}\n"
                     "        Check the network, or run: python benchmark.py --download-datasets")
        info[key] = {"rows": len(data[key]), "available": len(ds)}
        print(f"  {spec['label']}: {len(data[key])} of {len(ds)} rows ({time.time() - t:.1f}s)")
    cfg.eval_info = info
    if any(EVAL_SPECS[k]["kind"] == "code" for k in cfg.eval_datasets):
        cfg.eval_sandbox_resolved = resolve_sandbox(cfg)
    return data
