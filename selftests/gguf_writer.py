"""Write small synthetic GGUF headers (no real tensor data) for the GGUF-reader tests."""
import struct

U32, I32, F32, BOOL, STR, ARR, U64 = 4, 5, 6, 7, 8, 9, 10


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _val(v):
    if isinstance(v, bool):
        return struct.pack("<I", BOOL) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", U32) + struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<I", F32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", STR) + _s(v)
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            return struct.pack("<IIQ", ARR, STR, len(v)) + b"".join(_s(x) for x in v)
        if all(isinstance(x, bool) for x in v):
            return struct.pack("<IIQ", ARR, BOOL, len(v)) + b"".join(struct.pack("<?", x) for x in v)
        return struct.pack("<IIQ", ARR, I32, len(v)) + b"".join(struct.pack("<i", x) for x in v)
    raise TypeError(v)


def write_gguf(path, meta, tensors, pad_to=0):
    """meta: {key: value}; tensors: [(name, [dims])]. pad_to: final file size in bytes (stands in for weights)."""
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(meta))
    for k, v in meta.items():
        out += _s(k) + _val(v)
    off = 0
    for name, dims in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", 1, off)
        off += 1
    if pad_to > len(out):
        out += b"\0" * (pad_to - len(out))
    with open(path, "wb") as fh:
        fh.write(out)
    return path


def dense_model(path, layers=4, heads=8, kv_heads=2, embd=512, ctx_train=32768, vocab=2000, pad_to=0, extra=None):
    meta = {"general.architecture": "llama", "llama.block_count": layers, "llama.context_length": ctx_train,
            "llama.embedding_length": embd, "llama.attention.head_count": heads,
            "llama.attention.head_count_kv": kv_heads, "tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab)]}
    meta.update(extra or {})
    tensors = [("token_embd.weight", [embd, vocab])]
    for i in range(layers):
        tensors += [(f"blk.{i}.attn_q.weight", [embd, embd]), (f"blk.{i}.ffn_up.weight", [embd, embd * 4])]
    return write_gguf(path, meta, tensors, pad_to)


def moe_mla_model(path, layers=4, experts=64, used=4, pad_to=0):
    """deepseek2-style: MLA (kv_lora_rank) + routed experts + a shared expert, no 'A3B' needed in the name."""
    a = "deepseek2"
    meta = {"general.architecture": a, f"{a}.block_count": layers, f"{a}.context_length": 131072,
            f"{a}.embedding_length": 256, f"{a}.attention.head_count": 8, f"{a}.attention.head_count_kv": 8,
            f"{a}.attention.kv_lora_rank": 128, f"{a}.rope.dimension_count": 32, f"{a}.attention.key_length": 96,
            f"{a}.attention.value_length": 64, f"{a}.expert_count": experts, f"{a}.expert_used_count": used}
    tensors = [("token_embd.weight", [256, 1000])]
    for i in range(layers):
        tensors += [(f"blk.{i}.attn_kv_a_mqa.weight", [256, 160]),
                    (f"blk.{i}.ffn_up_exps.weight", [256, 128, experts]),
                    (f"blk.{i}.ffn_up_shexp.weight", [256, 128])]
    return write_gguf(path, meta, tensors, pad_to)
