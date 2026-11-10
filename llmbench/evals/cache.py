"""Per-question result cache: resume a crashed run, and rerun without re-asking questions that cannot have changed.

A cached answer is reused only when EVERYTHING that could change it is identical:

  model fingerprint   the model file (size + SHA-256 of its first and last MiB), the server's launch command
                      (normalised: port / host / threads / logging / API key removed), llama.cpp's build_info, the
                      loaded n_ctx and slots, a hash of the chat template, the model id, and the hardware string
                      (latencies are reused too, so a different GPU must not share a cache)
  question key        dataset, question id, the exact request body (prompt, max_tokens, temperature, tools,
                      EVAL_EXTRA_BODY), the thinking budget, loop abort / salvage / retry / budget-forcing switches,
                      code-grading settings and GRADER_VERSION (bump it whenever grading logic changes)

If the model file cannot be read from this machine (evals-only from another host, Strata) the fingerprint falls back
to what the server reports; that is weaker (a file swapped under the same path and name would not be noticed), so a
warning is printed. The speed tests are never cached: they measure the machine as it is now.

Storage: one append-only JSONL file per model fingerprint in EVAL_RESULT_CACHE_DIR (default results/cache, which is
git-ignored). Lines hold the same record as eval_details_*.jsonl, so GPQA answers live here too: never publish it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

GRADER_VERSION = "2.1.0"        # NOT the tool version; bump only when answer extraction / grading / the question prompts change

_DROP_FLAGS = {"--port", "--host", "-t", "--threads", "-tb", "--threads-batch", "--api-key", "--api-key-file",
               "--log-file", "--metrics", "--slots", "--props", "--alias", "-a", "--timeout", "-to", "--verbose",
               "-v", "--log-verbosity", "-lv", "--log-prefix", "--log-timestamps", "--log-colors", "--no-webui",
               "--webui", "--ssl-key-file", "--ssl-cert-file", "--path", "--threads-http"}
_VALUELESS = {"--metrics", "--slots", "--props", "--verbose", "-v", "--log-prefix", "--log-timestamps", "--no-webui",
              "--webui"}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_identity(path, mib=1):
    """size + SHA-256 of the first and last `mib` MiB: fast on a 20 GB file, and any re-quantisation changes it."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read(mib << 20))
            if size > (mib << 21):
                fh.seek(-(mib << 20), os.SEEK_END)
                h.update(fh.read(mib << 20))
        return f"{size}:{h.hexdigest()}"
    except OSError:
        return None


def normalise_argv(argv):
    """Launch tokens that can change a model's answers, with run-specific noise removed."""
    if not argv:
        return None
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in _DROP_FLAGS:
            skip = a not in _VALUELESS
            continue
        if i == 0 and not a.startswith("-") and ("llama-server" in a or a.endswith("server")):
            continue                                   # the binary's path
        out.append(a)
    return out


def model_fingerprint(r, props, hardware):
    """-> (hex digest, strong: bool). `r` is the result row (it may carry _gguf_path / _launch_argv from modelinfo)."""
    props = props or {}
    ident = file_identity(r["_gguf_path"]) if r.get("_gguf_path") else None
    parts = {"model": r["model"], "file": ident, "argv": normalise_argv(r.get("_launch_argv")),
             "build": props.get("build_info"), "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
             "slots": props.get("total_slots"), "model_path": props.get("model_path"),
             "template": hashlib.sha256((props.get("chat_template") or "").encode()).hexdigest()[:16],
             "hardware": hardware or ""}
    return _sha(parts)[:24], ident is not None


def question_key(cfg, key, item, payload, think_budget):
    body = {k: v for k, v in payload.items() if k not in ("model", "stream", "stream_options")}
    settings = {"budget": think_budget, "loop": cfg.eval_loop_abort, "salvage": cfg.eval_salvage,
                "retry": cfg.eval_retry_on_length, "stream": cfg.eval_stream, "forcing": cfg.eval_budget_forcing,
                "strict": cfg.eval_toolcall_strict, "lcb_tests": cfg.eval_lcb_max_tests,
                "code_timeout": cfg.eval_code_timeout, "grader": GRADER_VERSION}
    return _sha({"dataset": key, "id": item["id"], "body": body, "settings": settings})


class ResultCache:
    """Thread-safe append-only store. lookup() returns a copy of the cached record or None."""

    def __init__(self, root):
        self.root = Path(root)
        self._mem: dict = {}            # fingerprint -> {qkey: record}
        self._lock = threading.Lock()

    def _path(self, model, fp):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:80]
        return self.root / f"{slug}__{fp}.jsonl"

    def _load(self, model, fp):
        if fp in self._mem:
            return self._mem[fp]
        recs = {}
        p = self._path(model, fp)
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        recs[obj["qkey"]] = obj["rec"]
                    except (ValueError, KeyError, TypeError):
                        continue                # a half-written last line after a crash
        self._mem[fp] = recs
        return recs

    def lookup(self, model, fp, qkey):
        with self._lock:
            rec = self._load(model, fp).get(qkey)
        return dict(rec) if rec is not None else None

    def store(self, model, fp, qkey, rec):
        if rec.get("api_error"):
            return                              # a server failure is not an answer: ask again next time
        with self._lock:
            self._load(model, fp)[qkey] = rec
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self._path(model, fp), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"qkey": qkey, "rec": rec}, ensure_ascii=False, default=str) + "\n")

    def count(self, model, fp):
        with self._lock:
            return len(self._load(model, fp))


def same_answer(kind, fresh, cached):
    """Does a re-asked question give the same result as its cached answer? Code and tool calls: same pass/fail (the
    code text itself may differ harmlessly). Multiple choice and math: same extracted answer and same verdict."""
    if bool(fresh.get("correct")) != bool(cached.get("correct")):
        return False
    if kind in ("code", "tool"):
        return True
    norm = lambda a: re.sub(r"\s+", "", str(a)).lower() if a is not None else None
    return norm(fresh.get("answer")) == norm(cached.get("answer"))


class SanityCheck:
    """EVAL_SANITY_PCT: re-ask a random share of the answers that would come from the cache, to check they still hold.

    Each model starts at EVAL_SANITY_PCT. With EVAL_SANITY_ADAPTIVE, every re-asked answer that matches the cached one
    halves that model's rate, rounded down to a whole percent, and below 2% it becomes 0 (no more checks of that model
    in this run; the next run starts again at EVAL_SANITY_PCT). A different answer puts it back to EVAL_SANITY_PCT:
    a model that keeps answering the same is checked less and less, one that changes keeps being checked at the full
    rate. Example from 20%: 20 -> 10 -> 5 -> 2 -> 0. The fresh answer is the one
    scored and stored (it is the newest measurement under identical settings). The random picks differ per run, so
    repeated runs check different questions."""

    def __init__(self, pct, adaptive=True, seed=None):
        self.base = max(0.0, min(100.0, pct)) / 100
        self.adaptive = adaptive
        self.seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
        self._rng = __import__("random").Random(self.seed)
        self.rate: dict = {}                # model -> current probability
        self.stats: dict = {}               # model -> {"checked", "different"}

    def pick(self, model):
        rate = self.rate.setdefault(model, self.base)
        return rate > 0 and self._rng.random() < rate

    def result(self, model, same):
        st = self.stats.setdefault(model, {"checked": 0, "different": 0})
        st["checked"] += 1
        st["different"] += not same
        if self.adaptive:
            if same:
                pct = math.floor(100 * self.rate.get(model, self.base) / 2 + 1e-9)
                self.rate[model] = pct / 100 if pct >= 2 else 0.0
            else:
                self.rate[model] = self.base

    def current_pct(self, model):
        return 100 * self.rate.get(model, self.base)
