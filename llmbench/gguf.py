"""GGUF metadata: architecture, parameter counts, bits per weight and the KV-cache size per token.

Only the header is read (metadata key/values and the tensor table); tensor data is never touched, so this takes well
under a second even for a 20 GB file. Pure Python, no `gguf` package needed. Supports GGUF v2 and v3 and split files
(`*-00001-of-00003.gguf`: every shard is read and the sizes and tensor counts are summed).

WHY: the filename is a poor source of truth. GLM-4.7-Flash (30B-A3B MoE with Multi-head Latent Attention) has no
"A3B" in its name and was reported as Dense; the KV cache type, the number of full-attention layers and sliding-window
layers decide how much context fits, and none of that is in a filename.

KV bytes per token (what one more token of context costs in VRAM), per full-attention layer:
    standard / GQA / MQA:  n_head_kv * (key_length * bytes(K type) + value_length * bytes(V type))
    MLA (kv_lora_rank set): (kv_lora_rank + rope dims) * bytes(K type)    (llama.cpp caches the compressed latent)
Sliding-window layers hold at most `sliding_window` tokens, so they are a fixed cost, not a per-token one.
Recurrent / SSM layers (head_count_kv = 0 in the per-layer array, or not on the full_attention_interval) hold a
fixed-size state and are not counted. Where the sliding-window layout is not stored in the file, llama.cpp's built-in
pattern for that architecture is used and the estimate is marked approximate.
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path

# ---------------------------------------------------------------------------- #
# Low-level reader
# ---------------------------------------------------------------------------- #

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_MAX_KEEP_ARRAY = 4096          # longer arrays (tokenizer vocab, merges) are skipped, only their length is kept

# Bytes per element of the KV cache types llama.cpp accepts for -ctk / -ctv (block size 32 for the quantised ones).
KV_TYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                 "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
# Lower number = lower precision. Used to compare a KV type against a required minimum (e.g. "at least q8_0").
KV_PRECISION = {"f32": 9, "f16": 8, "bf16": 8, "q8_0": 7, "q5_1": 6, "q5_0": 5, "q4_1": 4, "iq4_nl": 3, "q4_0": 3}

# llama.cpp's hard-coded sliding-window layouts, for files that do not store the pattern. period N = every N-th layer
# (the last one of each group) uses full attention, the others use the sliding window.
_SWA_PERIOD = {"gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4, "gpt-oss": 2, "llama4": 4, "exaone4": 4,
               "gemma-embedding": 6}


class GGUFError(Exception):
    pass


class _Reader:
    def __init__(self, fh, version):
        self.f, self.version = fh, version

    def _unpack(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("unexpected end of file")
        return struct.unpack(fmt, b)[0]

    def u32(self):
        return self._unpack("<I")

    def u64(self):
        return self._unpack("<Q") if self.version >= 2 else self._unpack("<I")

    def string(self):
        n = self.u64()
        if n > 1 << 24:
            raise GGUFError("string too long (corrupt header?)")
        return self.f.read(n).decode("utf-8", "replace")

    def skip_string(self):
        self.f.seek(self.u64(), os.SEEK_CUR)

    def value(self, vtype):
        if vtype in _SCALAR:
            return self._unpack(_SCALAR[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.u32(), self.u64()
            if count > _MAX_KEEP_ARRAY:
                self.skip_array(etype, count)
                return _Skipped(count)
            return [self.value(etype) for _ in range(count)]
        raise GGUFError(f"unknown metadata value type {vtype}")

    def skip_array(self, etype, count):
        if etype in _SCALAR:
            self.f.seek(struct.calcsize(_SCALAR[etype]) * count, os.SEEK_CUR)
        elif etype == _STRING:
            for _ in range(count):
                self.skip_string()
        else:
            for _ in range(count):
                self.value(etype)


class _Skipped:
    """Placeholder for a long array that was not loaded (only its length is known)."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def read_header(path):
    """-> (metadata dict, [(tensor name, element count)]). Raises GGUFError / OSError."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = struct.unpack("<I", fh.read(4))[0]
        if version not in (1, 2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        rd = _Reader(fh, version)
        n_tensors, n_kv = rd.u64(), rd.u64()
        meta = {}
        for _ in range(n_kv):
            key = rd.string()
            meta[key] = rd.value(rd.u32())
        tensors = []
        for _ in range(n_tensors):
            name = rd.string()
            dims = [rd.u64() for _ in range(rd.u32())]
            rd.u32()                                  # ggml type
            rd.u64()                                  # data offset
            n = 1
            for d in dims:
                n *= d
            tensors.append((name, n))
    return meta, tensors


_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def shard_paths(path):
    """Every shard of a split GGUF (the given path alone if it is not split)."""
    p = Path(path)
    m = _SPLIT_RE.match(p.name)
    if not m:
        return [p]
    total = int(m.group(3))
    return [p.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, total + 1)]


# ---------------------------------------------------------------------------- #
# Interpretation
# ---------------------------------------------------------------------------- #

def _get(meta, arch, key, default=None):
    v = meta.get(f"{arch}.{key}")
    return default if v is None or isinstance(v, _Skipped) else v


def _per_layer(v, n_layers, default=None):
    """A metadata value that may be a scalar or a per-layer array -> list of n_layers values."""
    if isinstance(v, list) and v:
        return [v[i] if i < len(v) else v[-1] for i in range(n_layers)]
    return [default if v is None else v] * n_layers


def _layer_kinds(meta, arch, n_layers, head_kv):
    """-> (list of 'full' | 'swa' | 'none' per layer, approximate?)."""
    kinds = ["full" if (h or 0) > 0 else "none" for h in head_kv]
    approx = False
    interval = _get(meta, arch, "full_attention_interval")
    if isinstance(interval, int) and interval > 1:               # qwen3next / qwen3.5-style hybrids
        kinds = ["full" if (i + 1) % interval == 0 and kinds[i] != "none" else "none" for i in range(n_layers)]
    window = _get(meta, arch, "attention.sliding_window")
    if not window:
        return kinds, approx
    pattern = _get(meta, arch, "attention.sliding_window_pattern")
    if isinstance(pattern, list) and pattern:                    # per-layer bools: True = sliding window
        swa = [bool(pattern[i % len(pattern)]) for i in range(n_layers)]
    else:
        period = pattern if isinstance(pattern, int) and pattern > 0 else _SWA_PERIOD.get(arch)
        if period is None:              # a window is set but the layout is unknown: llama.cpp applies it to all layers
            period, approx = 0, True
        elif not isinstance(pattern, int):
            approx = True
        swa = [True] * n_layers if period == 0 else [(i + 1) % period != 0 for i in range(n_layers)]
    return [("swa" if s and k == "full" else k) for k, s in zip(kinds, swa)], approx


def _attention_type(meta, arch, head_kv, kinds, n_head):
    if _get(meta, arch, "attention.kv_lora_rank"):
        base = "MLA"
    else:
        hk = max((h for h, k in zip(head_kv, kinds) if k != "none" and h), default=None)
        if hk is None:
            return "SSM / recurrent"
        base = "MHA" if n_head and hk >= n_head else "MQA" if hk == 1 else "GQA"
    if "none" in kinds:
        base = f"hybrid {base} + SSM"
    if "swa" in kinds:
        base += " + SWA"
    return base


def _params(tensors, n_expert, n_used):
    """(total params, active params or None) from the tensor table. Routed expert tensors (`*_exps`) count only
    n_used / n_expert towards the active figure; shared experts (`*_shexp`) and everything else always count."""
    total = sum(n for _, n in tensors)
    if not n_expert or n_expert <= 1:
        return total, None
    experts = sum(n for name, n in tensors if "_exps" in name)
    if not experts:
        return total, None
    return total, total - experts + experts * (n_used or 1) / n_expert


def kv_bytes(info, ctk="f16", ctv="f16"):
    """(bytes per token over the full-attention layers, fixed bytes for the sliding-window layers) for KV types ctk/ctv.
    None when the architecture is not understood."""
    if not info or info.get("_kv") is None:
        return None, None
    bk = KV_TYPE_BYTES.get((ctk or "f16").lower(), 2.0)
    bv = KV_TYPE_BYTES.get((ctv or "f16").lower(), 2.0)
    per_full, per_swa_tok = 0.0, 0.0
    for layer in info["_kv"]:
        if layer["mla"] is not None:
            b = layer["mla"] * bk
        else:
            b = layer["heads"] * (layer["k"] * bk + layer["v"] * bv)
        if layer["kind"] == "full":
            per_full += b
        elif layer["kind"] == "swa":
            per_swa_tok += b
    return per_full, per_swa_tok * (info.get("swa_window") or 0)


def describe(meta, tensors):
    """Metadata + tensor table -> the facts the benchmark reports."""
    arch = str(meta.get("general.architecture") or "unknown")
    n_layers = int(_get(meta, arch, "block_count", 0) or 0)
    n_head = _get(meta, arch, "attention.head_count")
    n_head = max(n_head) if isinstance(n_head, list) and n_head else n_head
    n_embd = _get(meta, arch, "embedding_length")
    head_kv = _per_layer(_get(meta, arch, "attention.head_count_kv"), n_layers, n_head)
    n_expert = _get(meta, arch, "expert_count")
    n_used = _get(meta, arch, "expert_used_count")
    total, active = _params(tensors, n_expert, n_used)
    out = {"gguf_arch": arch, "n_layers": n_layers or None, "n_ctx_train": _get(meta, arch, "context_length"),
           "expert_count": n_expert if n_expert and n_expert > 1 else None,
           "expert_used": n_used if n_expert and n_expert > 1 else None,
           "params_total": total or None, "params_active": active, "kv_approx": False, "_kv": None}
    if not n_layers:
        return out
    kinds, approx = _layer_kinds(meta, arch, n_layers, head_kv)
    out["attn_type"] = _attention_type(meta, arch, head_kv, kinds, n_head)
    out["full_attn_layers"] = kinds.count("full")
    out["swa_layers"] = kinds.count("swa") or None
    out["swa_window"] = _get(meta, arch, "attention.sliding_window") if "swa" in kinds else None
    out["kv_approx"] = approx
    head_dim = (n_embd // n_head) if n_embd and n_head else None
    k_len = _per_layer(_get(meta, arch, "attention.key_length"), n_layers, head_dim)
    v_len = _per_layer(_get(meta, arch, "attention.value_length"), n_layers, head_dim)
    lora = _get(meta, arch, "attention.kv_lora_rank")
    rope = _get(meta, arch, "rope.dimension_count") or 0
    if not lora and None in k_len:
        return out                                     # head size unknown: cannot size the cache
    out["_kv"] = [{"kind": kinds[i], "heads": head_kv[i] or 0, "k": k_len[i] or 0, "v": v_len[i] or 0,
                   "mla": (lora + rope) if lora else None} for i in range(n_layers)]
    return out


def inspect(path):
    """Read a (possibly split) GGUF -> describe() dict plus file_bytes, bpw. Raises OSError / GGUFError."""
    shards = shard_paths(path)
    meta, tensors, size = None, [], 0
    for p in shards:
        m, t = read_header(p)
        meta = meta or m                                # architecture metadata lives in the first shard
        tensors += t
        size += os.path.getsize(p)
    info = describe(meta, tensors)
    info["file_bytes"] = size
    info["bpw"] = size * 8 / info["params_total"] if info.get("params_total") else None
    return info


# ---------------------------------------------------------------------------- #
# Finding the file the server actually loaded
# ---------------------------------------------------------------------------- #

def parse_path_map(raw):
    """GGUF_PATH_MAP="/models=/mnt/AI-Models,/data=/srv/llm" -> [("/models", "/mnt/AI-Models"), ...]."""
    out = []
    for part in (raw or "").split(","):
        if "=" in part:
            a, b = part.split("=", 1)
            if a.strip() and b.strip():
                out.append((a.strip().rstrip("/"), b.strip().rstrip("/")))
    return out


def candidate_paths(model_path, pid=None, path_map=(), search_dirs=()):
    """Where a path seen by the server (often inside a Docker container) can be read from this host, in order:
    as is; through /proc/<pid>/root (the server's own filesystem view; needs root or the same user); with
    GGUF_PATH_MAP prefixes swapped; by file name under GGUF_SEARCH_DIRS."""
    if not model_path:
        return []
    p = str(model_path)
    out = [p]
    if pid:
        out.append(f"/proc/{pid}/root{p if p.startswith('/') else '/' + p}")
    for src, dst in path_map:
        if p == src or p.startswith(src + "/"):
            out.append(dst + p[len(src):])
    name = os.path.basename(p)
    for d in search_dirs:
        base = Path(d)
        direct = base / name
        out.append(str(direct))
        try:
            out += [str(x) for x in base.rglob(name)][:3]
        except OSError:
            pass
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def find_readable(model_path, pid=None, path_map=(), search_dirs=()):
    for c in candidate_paths(model_path, pid, path_map, search_dirs):
        try:
            if os.path.isfile(c) and os.access(c, os.R_OK):
                return c
        except OSError:
            continue
    return None
