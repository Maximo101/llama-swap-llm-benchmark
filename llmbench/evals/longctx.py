"""Long-context retrieval check: several "needles" hidden in a long filler text, all asked for at once (multi-key).

For each length in LONGCTX_LENGTHS (default 64k and 128k tokens) ONE haystack is built that hides
LONGCTX_TRIALS x LONGCTX_KEYS vault codes at spread-out depths, plus as many decoy vaults that are never asked for.
The model then gets LONGCTX_TRIALS questions about that same document, each asking for a different set of
LONGCTX_KEYS vaults. Score = share of asked codes returned correctly, over all questions of that length.

v2.1 - WHY ONE HAYSTACK: the document is the prompt's prefix and identical for every question of a length, so after
the first question llama-server reuses it from its prompt cache and only the short question is processed. v2.0.0
built a fresh 64k / 128k document for every trial and paid the full prefill each time (the single most expensive
request of a run). The questions share a document, so they are not independent samples; each asks different needles
at different depths, and a vault asked in one question is a decoy for the others.

The FIRST question of a length is a cold, unique prompt of that length, so its prompt tokens/s is a prefill-at-length
measurement. When the speed phase deferred that length (PREFILL_CTX_TESTS also lists it, see cli.py), the value is
written to prefill_<len>_tps here instead of sending a second, separate long prompt.

What this measures: whether the model (at the KV-cache precision it is deployed with) can still find and keep apart
facts deep in a long context. It does not measure reasoning over that context. A length that does not fit the loaded
context is skipped and reported ("ctx too small"), so a model loaded at 90k gets a 64k result and no 128k result.

KV-cache precision: the benchmark cannot change the server's -ctk / -ctv (they are launch flags in llama-swap's
config). It reads them, and LONGCTX_KV_MIN (default q8_0) says how low they may go: a model whose KV type is below it is
flagged (LONGCTX_KV_ACTION=flag, the default) or not tested (skip). Lower the setting (e.g. q4_0) to accept such runs.
"""
from __future__ import annotations

import random
import re
import time

from ..client import post_chat
from ..common import BenchError, add_time, avg
from ..config import LONGCTX_SLOTS, PREFILL_SLOTS
from ..modelinfo import kv_at_least
from .answers import strip_think

_WORDS = ("river stone copper window engine garden signal marble orbit harvest paper lantern thunder velvet compass "
          "anchor meadow circuit whisper granite ember canvas harbor quartz timber mirror falcon cobalt prairie summit "
          "voyage lattice monsoon plume fabric glacier beacon orchard ledger saddle crater tundra pillar").split()
_NAMES = ("Amber Basalt Cedar Delta Echo Falcon Garnet Harbor Indigo Juniper Kestrel Lumen Maple Nimbus Onyx Pebble "
          "Quill Raven Sable Tundra Umber Vesper Willow Xenon Yarrow Zephyr").split()
_SUFFIX = ("", "Ridge", "Point", "Hollow")
_OVER = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")         # 104 distinct one-word vault names (LONGCTX_KEYS x TRIALS x 2 <= 104)


def _sentence(rng):
    n = rng.randint(8, 16)
    w = [rng.choice(_WORDS) for _ in range(n)]
    return "The " + " ".join(w) + "."


def build_haystack(tokens, tpw, n_needles, seed):
    """-> (document, {name: code} needles to ask about, {name: code} decoys). `tpw` = tokens per word."""
    rng = random.Random(seed)
    pool = [n + s for s in _SUFFIX for n in _NAMES]
    names = rng.sample(pool, 2 * n_needles)
    asked = {n: f"{rng.randint(100000, 999999)}" for n in names[:n_needles]}
    decoy = {n: f"{rng.randint(100000, 999999)}" for n in names[n_needles:]}
    words_total = max(200, int(tokens / (tpw * 1.08)))           # punctuation makes sentences ~8% denser than words
    filler, count = [], 0
    while count < words_total:
        s = _sentence(rng)
        filler.append(s)
        count += len(s.split())
    needles = [(n, c) for n, c in {**asked, **decoy}.items()]
    rng.shuffle(needles)
    k = len(needles)
    for i, (n, c) in enumerate(needles):          # spread over 5%..95% depth, jittered
        depth = 0.05 + 0.9 * (i + rng.uniform(0.2, 0.8)) / k
        filler.insert(int(depth * len(filler)), f"Remember this: the access code for vault {n} is {c}.")
    doc = ("Below is a long document. Somewhere in it are access codes for several vaults.\n\n<document>\n"
           + " ".join(filler) + "\n</document>\n\n")
    return doc, asked, decoy


def question(names):
    return (f"Give the access code for each of these vaults: {', '.join(names)}. Reply with one line per vault in the "
            "form 'Vault <name>: <code>' and nothing else.")


def build_prompt(tokens, tpw, n_keys, seed):
    """One complete prompt asking for every hidden needle -> (prompt, asked, decoys)."""
    doc, asked, decoy = build_haystack(tokens, tpw, n_keys, seed)
    return doc + question(list(asked)), asked, decoy


def score_reply(text, asked, decoy):
    """Share of asked vaults whose code is given. A line naming the vault decides; if the model wrote bare codes, a
    code counts when it appears and no decoy code does."""
    body = strip_think(text)
    hit = 0
    any_decoy = any(c in body for c in decoy.values())
    for name, code in asked.items():
        m = re.search(rf"\b{re.escape(name)}\b\D{{0,40}}?(\d{{6}})", body, re.I)
        if m:
            hit += m.group(1) == code
        elif code in body and not any_decoy:
            hit += 1
    return hit / len(asked)


THINKING_RETRY_TOKENS = 4096   # a model that ignores enable_thinking=false spends a 400-token reply on thinking


def _ask(cfg, session, model, prompt, timeout, max_tokens=None):
    """-> (reply text, prompt tokens, timings dict). A reply with no visible answer because the model used the whole
    reply budget for thinking is asked again (same prompt, so the document comes from the prompt cache) with room to
    finish: v2.0 scored every LFM2.5 reply as 0% because its chat template ignores enable_thinking=false and all 400
    tokens went to reasoning_content, leaving an empty answer."""
    text, used, timings, finish, reasoning = _ask_once(cfg, session, model, prompt, timeout, max_tokens)
    if not strip_think(text).strip() and (reasoning or finish == "length") and (max_tokens or 0) < THINKING_RETRY_TOKENS:
        text, used2, timings2, finish, reasoning = _ask_once(cfg, session, model, prompt, timeout, THINKING_RETRY_TOKENS)
        timings = {**timings2, "first_prompt_per_second": timings.get("prompt_per_second"), "thinking_retry": True}
        used = used or used2
    return text, used, timings


def _ask_once(cfg, session, model, prompt, timeout, max_tokens=None):
    p = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
         "temperature": 0.0, "max_tokens": max_tokens or (400 if cfg.longctx_no_think else THINKING_RETRY_TOKENS),
         "cache_prompt": True}
    if cfg.eval_extra_body:
        p.update(cfg.eval_extra_body)
    p["stream"] = False                         # EVAL_EXTRA_BODY must not switch the reader's mode (v2.0.0 bug B6)
    if cfg.longctx_no_think:
        p["chat_template_kwargs"] = {**(p.get("chat_template_kwargs") or {}), "enable_thinking": False}
    data = post_chat(cfg, session, p, timeout)
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "", (data.get("usage") or {}).get("prompt_tokens"), data.get("timings") or {},
            ch.get("finish_reason"), len(msg.get("reasoning_content") or ""))


def run_longctx(cfg, session, r, ctx, sink=None):
    """Fill r['longctx_<len>_pct'], r['longctx_kv_ok'], r['longctx_note'] (and deferred prefill_<len>_tps).
    Blocking (call from a thread)."""
    model, notes = r["model"], []
    ok = kv_at_least(r.get("kv_type"), cfg.longctx_kv_min)
    r["longctx_kv_ok"] = ok
    if ok is False:
        msg = f"KV {r.get('kv_type')} is below LONGCTX_KV_MIN {cfg.longctx_kv_min}"
        if cfg.longctx_kv_action == "skip":
            r["longctx_note"] = msg + " (skipped)"
            print(f"    Long-ctx   skipped: {msg}")
            return
        notes.append(msg)
    tpw = r.get("tok_per_word") or 1.25
    t_all = time.time()
    for n in cfg.longctx_lengths:
        label = LONGCTX_SLOTS[n]
        if ctx and n > ctx:
            notes.append(f"{label}: ctx {ctx // 1024}k too small")
            continue
        # Room for the question and the reply. The words -> tokens estimate can be ~1.5% off, so when the test length
        # is the whole loaded context keep 3% back: v2.0 asked for 130,472 tokens at ctx 131,072 and got prompts of
        # 131,644-132,331 tokens (HTTP 400, no 128k result).
        target = (min(n, ctx) if ctx else n) - 600
        if ctx and n >= ctx:
            target = ctx - max(1024, int(0.03 * ctx))
        trials, keys = cfg.longctx_trials, cfg.longctx_keys
        timeout = max(cfg.eval_timeout, n / 100 + 120)
        scores, toks, cached, t0 = [], [], [], time.time()
        first, retried = None, False
        for attempt in range(2):            # one rebuild if the server still says the prompt is too long
            doc, needles, decoys = build_haystack(target, tpw, trials * keys, f"{cfg.eval_seed}:{n}")
            names = list(needles)
            try:
                first = _ask(cfg, session, model, doc + question(names[:keys]), timeout)
                break
            except BenchError as e:
                over = _OVER.search(str(e))
                if attempt == 0 and over:
                    want, avail = int(over.group(1)), int(over.group(2))
                    target = int(target * (avail - 1024) / want)
                    continue
                notes.append(f"{label}: {e}")
                break
        if first is None:
            continue
        for trial in range(trials):
            ask_names = names[trial * keys:(trial + 1) * keys]
            asked = {k: needles[k] for k in ask_names}
            others = {**decoys, **{k: v for k, v in needles.items() if k not in asked}}
            try:
                text, used, timings = first if trial == 0 else _ask(cfg, session, model, doc + question(ask_names),
                                                                    timeout)
            except BenchError as e:
                notes.append(f"{label}: {e}")
                break
            if used:
                toks.append(used)
            if trial == 0:
                slot = PREFILL_SLOTS.get(n)
                pps = timings.get("first_prompt_per_second") or timings.get("prompt_per_second")
                if slot and pps and n in cfg.defer_prefill and r.get(f"prefill_{slot}_tps") is None:
                    r[f"prefill_{slot}_tps"] = pps          # this cold prompt IS the prefill-at-length test
            if timings.get("cache_n") is not None:
                cached.append(timings["cache_n"])
            if timings.get("thinking_retry"):
                retried = True
            s = score_reply(text, asked, others)
            scores.append(s)
            if sink:
                sink({"model": model, "dataset": "longctx", "id": f"longctx-{label}-{trial}", "correct": s == 1.0,
                      "score": s, "prompt_tokens": used, "cache_n": timings.get("cache_n"), "asked": asked,
                      "thinking_retry": bool(timings.get("thinking_retry")), "response": text[:1000]})
        if retried:
            notes.append(f"{label}: thinking used the whole 400-token reply, asked again with "
                         f"{THINKING_RETRY_TOKENS:,} tokens (the model ignores enable_thinking=false)")
        if scores:
            r[f"longctx_{label}_pct"] = 100 * avg(scores)
            k = f"{avg(toks) / 1024:.0f}k tok" if toks else "? tok"
            reuse = f", prefix reused {max(cached) / 1024:.0f}k" if cached and max(cached) > 1024 else ""
            print(f"    Long-ctx   {label:<4} {100 * avg(scores):5.1f}%  ({len(scores)} x {keys} keys, {k}, "
                  f"{time.time() - t0:.0f}s, KV {r.get('kv_type') or '?'}{reuse})")
    add_time(r, "longctx", time.time() - t_all)
    r["longctx_note"] = "; ".join(notes) or None
    if notes:
        print("    Long-ctx   note: " + "; ".join(notes))
