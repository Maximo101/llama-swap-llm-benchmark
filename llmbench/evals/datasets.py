"""Evaluation datasets: loading, sampling, prompts and item builders."""
from __future__ import annotations

import json
import random
import re
import string
import sys
import time

from ..config import EVAL_SPECS, rows_for, stage_rows
from .sandbox import resolve_sandbox

_HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
HANGUL_MAX_SHARE = 0.01     # refuse a dataset when more than 1% of the sampled questions contain Korean text

def load_eval_split(cfg, key, offline_first=True):
    """Load one dataset split with cache_dir=cfg.eval_cache_dir (./tests). Normal runs try the local cache
    first (no network, milliseconds) and only download if the dataset is not cached yet."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[eval] The 'datasets' package is missing. Run: pip install -r requirements.txt")
    spec, cache = EVAL_SPECS[key], str(cfg.eval_cache_dir)
    rev = (cfg.eval_revisions or {}).get(key)          # EVAL_<DATASET>_REVISION: a pinned commit / tag
    extra = {"revision": rev} if rev else {}
    ds = None
    if offline_first:
        try:
            from datasets import DownloadConfig
            ds = load_dataset(spec["repo"], spec.get("config"), cache_dir=cache,
                              download_config=DownloadConfig(local_files_only=True), **extra)
        except Exception:
            ds = None
    if ds is None:
        if offline_first:
            print(f"  [eval] {spec['label']} is not in {cache} yet - downloading it now "
                  "(run --download-datasets next time to do this up front)")
        ds = load_dataset(spec["repo"], spec.get("config"), cache_dir=cache, **extra)
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


def question_order(candidates, seed, key):
    """One seeded permutation of a dataset's rows. Every profile takes a PREFIX of it, so a smaller profile's rows are
    always a subset of a larger one's (nested samples): the funnel and the result cache build on that.
    (v2.0.0 drew each profile independently, so e.g. MATH-500 'quick' was not a subset of 'standard'.)"""
    order = list(candidates)
    random.Random(f"{seed}:{key}").shuffle(order)
    return order


def sample_indices(n, k, seed, key, candidates=None):
    """Deterministic, nested sample: the first k of question_order(), sorted. Every model sees exactly the same rows.
    `candidates` restricts the pool (e.g. LiveCodeBench problems after EVAL_LCB_SINCE)."""
    pool = list(range(n)) if candidates is None else list(candidates)
    if k is None or k >= len(pool):
        return sorted(pool)
    return sorted(question_order(pool, seed, key)[:k])


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


def lcb_candidates(cfg, ds):
    """EVAL_LCB_SINCE: indices of LiveCodeBench problems whose contest_date is on/after the date, or None (all).
    LiveCodeBench's own method is to score only problems released after a model's training cutoff; older problems
    may have been seen in training and inflate newer models' scores."""
    if not cfg.eval_lcb_since:
        return None
    try:
        dates = ds.select_columns(["contest_date"])
        vals = [str(dates[i]["contest_date"] or "")[:10] for i in range(len(ds))]
    except Exception:
        print("  [warn] LiveCodeBench: no contest_date column, EVAL_LCB_SINCE ignored")
        return None
    keep = [i for i, d in enumerate(vals) if d and d >= cfg.eval_lcb_since]
    if not keep:
        sys.exit(f"[eval] EVAL_LCB_SINCE={cfg.eval_lcb_since}: no LiveCodeBench problem is that recent "
                 f"(newest: {max(vals) if vals else 'unknown'})")
    return keep


def language_guard(key, items):
    """Refuse a dataset whose sampled questions are in Korean (a mirror can carry translated text; the bzantium copy's
    card still mentions a Korean translation inherited from bzantium/ko-livecodebench). Scores on translated prompts
    would not be comparable with anything else."""
    if not items:
        return
    hits = sum(1 for it in items if _HANGUL.search(it.get("prompt") or ""))
    if hits / len(items) > HANGUL_MAX_SHARE:
        sys.exit(f"[eval] {EVAL_SPECS[key]['label']}: {hits} of {len(items)} sampled questions contain Korean text. "
                 "This copy is not the English benchmark; pin another revision (EVAL_<DATASET>_REVISION) or use an "
                 "English source.")


def prepare_eval_data(cfg, rows=0):
    """Load + sample every enabled dataset up front, so a missing package, dataset or cache fails
    before the (long) speed test rather than after it. Loads the LARGEST stage's rows (the funnel's final stage);
    each item carries `_row`, its index in the dataset, so smaller (nested) stages can be selected later."""
    rows = stage_rows(cfg) if rows == 0 else rows          # rows=None: every row; an int: that many (batch probe)
    data, info, pools = {}, {}, {}
    cfg.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    label = "funnel up to " + cfg.eval_funnel_final if cfg.eval_profile == "funnel" else cfg.eval_profile
    print(f"Loading evaluation datasets (profile {label}: "
          f"{'all rows' if rows is None else f'{rows} rows each'}; cache {cfg.eval_cache_dir})")
    for key in cfg.eval_datasets:
        spec = EVAL_SPECS[key]
        t = time.time()
        if spec.get("builtin"):
            from .toolcall_cases import build_items as builtin_items
            pool = builtin_items()
            idxs = sample_indices(len(pool), rows_for(key, rows), cfg.eval_seed, key)
            data[key] = [{**pool[i], "_row": i} for i in idxs]
            info[key] = {"rows": len(data[key]), "available": len(pool)}
            pools[key] = (len(pool), None)
            print(f"  {spec['label']}: {len(data[key])} of {len(pool)} cases (built in)")
            continue
        try:
            ds = load_eval_split(cfg, key)
            cand = lcb_candidates(cfg, ds) if key == "livecodebench" else None
            idxs = sample_indices(len(ds), rows, cfg.eval_seed, key, cand)
            data[key] = build_items(cfg, key, ds, idxs)
            for it, i in zip(data[key], idxs):
                it["_row"] = i
            language_guard(key, data[key])
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"[eval] Could not load {spec['label']} ({spec['repo']}): {e.__class__.__name__}: {e}\n"
                     "        Check the network, or run: python benchmark.py --download-datasets")
        avail = len(cand) if cand is not None else len(ds)
        info[key] = {"rows": len(data[key]), "available": avail,
                     "revision": (cfg.eval_revisions or {}).get(key),
                     "since": cfg.eval_lcb_since if key == "livecodebench" and cand is not None else None}
        pools[key] = (len(ds), cand)
        print(f"  {spec['label']}: {len(data[key])} of {avail} rows"
              + (f" released since {cfg.eval_lcb_since}" if info[key]["since"] else "")
              + (f", revision {info[key]['revision']}" if info[key]["revision"] else "")
              + f" ({time.time() - t:.1f}s)")
    cfg.eval_info = info
    cfg.eval_pools = pools
    if any(EVAL_SPECS[k]["kind"] == "code" for k in cfg.eval_datasets):
        cfg.eval_sandbox_resolved = resolve_sandbox(cfg)
    return data
