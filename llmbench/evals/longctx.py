"""Long-context retrieval check: several "needles" hidden in a long filler text, all asked for at once (multi-key).

For each length in LONGCTX_LENGTHS (default 64k and 128k tokens) the model gets LONGCTX_TRIALS prompts. Each prompt
hides LONGCTX_KEYS vault codes at spread-out depths, plus the same number of decoy vaults that are NOT asked for, and
asks for the asked vaults' codes. Score = share of asked codes returned correctly, over all trials of that length.

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
from ..common import BenchError, avg
from ..config import LONGCTX_SLOTS
from ..modelinfo import kv_at_least
from .answers import strip_think

_WORDS = ("river stone copper window engine garden signal marble orbit harvest paper lantern thunder velvet compass "
          "anchor meadow circuit whisper granite ember canvas harbor quartz timber mirror falcon cobalt prairie summit "
          "voyage lattice monsoon plume fabric glacier beacon orchard ledger saddle crater tundra pillar").split()
_NAMES = ("Amber Basalt Cedar Delta Echo Falcon Garnet Harbor Indigo Juniper Kestrel Lumen Maple Nimbus Onyx Pebble "
          "Quill Raven Sable Tundra Umber Vesper Willow Xenon Yarrow Zephyr").split()


def _sentence(rng):
    n = rng.randint(8, 16)
    w = [rng.choice(_WORDS) for _ in range(n)]
    return "The " + " ".join(w) + "."


def build_prompt(tokens, tpw, n_keys, seed):
    """-> (prompt, {name: code} asked, {name: code} decoys). `tpw` = tokens per word, measured on this model."""
    rng = random.Random(seed)
    names = rng.sample(_NAMES, 2 * n_keys)
    asked = {n: f"{rng.randint(100000, 999999)}" for n in names[:n_keys]}
    decoy = {n: f"{rng.randint(100000, 999999)}" for n in names[n_keys:]}
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
    ask = ", ".join(asked)
    prompt = ("Below is a long document. Somewhere in it are access codes for several vaults.\n\n<document>\n"
              + " ".join(filler) + "\n</document>\n\n"
              f"Give the access code for each of these vaults: {ask}. Reply with one line per vault in the form "
              "'Vault <name>: <code>' and nothing else.")
    return prompt, asked, decoy


def score_reply(text, asked, decoy):
    """Share of asked vaults whose code is given. A line naming the vault decides; if the model wrote bare codes, a
    code counts when it appears and no decoy code does."""
    body = strip_think(text)
    hit = 0
    any_decoy = any(c in body for c in decoy.values())
    for name, code in asked.items():
        m = re.search(rf"{name}\D{{0,40}}?(\d{{6}})", body, re.I)
        if m:
            hit += m.group(1) == code
        elif code in body and not any_decoy:
            hit += 1
    return hit / len(asked)


def _ask(cfg, session, model, prompt, timeout):
    p = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
         "temperature": 0.0, "max_tokens": 400 if cfg.longctx_no_think else 4096}
    if cfg.eval_extra_body:
        p.update(cfg.eval_extra_body)
    if cfg.longctx_no_think:
        p["chat_template_kwargs"] = {**(p.get("chat_template_kwargs") or {}), "enable_thinking": False}
    data = post_chat(cfg, session, p, timeout)
    ch = (data.get("choices") or [{}])[0]
    return ((ch.get("message") or {}).get("content") or ""), (data.get("usage") or {}).get("prompt_tokens")


def run_longctx(cfg, session, r, ctx, sink=None):
    """Fill r['longctx_<len>_pct'], r['longctx_kv_ok'], r['longctx_note']. Blocking (call from a thread)."""
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
    for n in cfg.longctx_lengths:
        label = LONGCTX_SLOTS[n]
        if ctx and n > ctx:
            notes.append(f"{label}: ctx {ctx // 1024}k too small")
            continue
        target = (min(n, ctx) if ctx else n) - 600                  # room for the question and the reply
        scores, toks, t0 = [], [], time.time()
        for trial in range(cfg.longctx_trials):
            prompt, asked, decoy = build_prompt(target, tpw, cfg.longctx_keys, f"{cfg.eval_seed}:{n}:{trial}")
            try:
                text, used = _ask(cfg, session, model, prompt, max(cfg.eval_timeout, n / 100 + 120))
            except BenchError as e:
                notes.append(f"{label}: {e}")
                break
            if used:
                toks.append(used)
                tpw = tpw * target / used if used > target * 1.02 else tpw     # prompt came out too big: shrink
            s = score_reply(text, asked, decoy)
            scores.append(s)
            if sink:
                sink({"model": model, "dataset": "longctx", "id": f"longctx-{label}-{trial}", "correct": s == 1.0,
                      "score": s, "prompt_tokens": used, "asked": asked, "response": text[:1000]})
        if scores:
            r[f"longctx_{label}_pct"] = 100 * avg(scores)
            k = f"{avg(toks) / 1024:.0f}k tok" if toks else "? tok"
            print(f"    Long-ctx   {label:<4} {100 * avg(scores):5.1f}%  ({len(scores)} x {cfg.longctx_keys} keys, {k}, "
                  f"{time.time() - t0:.0f}s, KV {r.get('kv_type') or '?'})")
    r["longctx_note"] = "; ".join(notes) or None
    if notes:
        print("    Long-ctx   note: " + "; ".join(notes))
