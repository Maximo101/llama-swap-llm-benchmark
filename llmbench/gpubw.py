"""GPU memory bandwidth for --estimate (ESTIMATE_GPU_BW_GBS).

`auto` (the default) looks the detected GPU up in GPU_TABLE; an unknown or undetected GPU uses FALLBACK_GBS
(670 GB/s, about a 12-16 GB mid-range card). A number in ESTIMATE_GPU_BW_GBS always wins.
Bandwidths are the vendors' published memory specifications (bus width x memory data rate); the "use" column is
a rough guide for local LLMs. Only the bandwidth is used by the estimate.
"""
from __future__ import annotations

FALLBACK_GBS = 670.0

# (match key, label, arch, VRAM, bus, memory, GB/s, use). Longest matching key wins ("rtx 4070 ti super" before
# "rtx 4070"); `vram_gb` picks between variants that share a name (RTX 3080 10 GB vs 12 GB).
GPU_TABLE = [
    ("rtx 3060 ti",       "RTX 3060 Ti",          "Ampere",       "8 GB",   "256-bit", "GDDR6",  448,  "Fast for its size; 8 GB limits it to small models"),
    ("rtx 3060",          "RTX 3060 12GB",        "Ampere",       "12 GB",  "192-bit", "GDDR6",  360,  "Budget local AI entry point (12 GB is excellent for 7B/8B models at 4-bit)"),
    ("rtx 3070 ti",       "RTX 3070 Ti",          "Ampere",       "8 GB",   "256-bit", "GDDR6X", 608,  "Fast, but 8 GB limits it to small models"),
    ("rtx 3070",          "RTX 3070",             "Ampere",       "8 GB",   "256-bit", "GDDR6",  448,  "Fast, but 8 GB limits it to small models"),
    ("rtx 3080|10",       "RTX 3080 10GB",        "Ampere",       "10 GB",  "320-bit", "GDDR6X", 760,  "Solid 1440p/4K gaming; fast inference, limited by 10 GB"),
    ("rtx 3080|12",       "RTX 3080 12GB / 3080 Ti", "Ampere",    "12 GB",  "384-bit", "GDDR6X", 912,  "Fast inference, limited by 12 GB"),
    ("rtx 3090 ti",       "RTX 3090 Ti",          "Ampere",       "24 GB",  "384-bit", "GDDR6X", 1008, "Classic workstation choice; 24 GB fits 14B-32B models easily"),
    ("rtx 3090",          "RTX 3090",             "Ampere",       "24 GB",  "384-bit", "GDDR6X", 936,  "Classic workstation choice; 24 GB fits 14B-32B models easily"),
    ("rtx 4060 ti",       "RTX 4060 Ti 8/16GB",   "Ada Lovelace", "8/16 GB", "128-bit", "GDDR6", 288,  "16 GB is a high-VRAM budget option; low bandwidth limits token speed"),
    ("rtx 4060",          "RTX 4060 8GB",         "Ada Lovelace", "8 GB",   "128-bit", "GDDR6",  272,  "1080p gaming; tight VRAM for modern LLMs"),
    ("rtx 4070 ti super", "RTX 4070 Ti Super",    "Ada Lovelace", "16 GB",  "256-bit", "GDDR6X", 672,  "Sweet spot for 16 GB VRAM and fast memory"),
    ("rtx 4070 ti",       "RTX 4070 Ti",          "Ada Lovelace", "12 GB",  "192-bit", "GDDR6X", 504,  "Strong 1440p gaming; smooth local AI for mid-sized models"),
    ("rtx 4070",          "RTX 4070 / 4070 Super", "Ada Lovelace", "12 GB", "192-bit", "GDDR6X", 504,  "Strong 1440p gaming; smooth local AI for mid-sized models"),
    ("rtx 4080 super",    "RTX 4080 Super",       "Ada Lovelace", "16 GB",  "256-bit", "GDDR6X", 736,  "Premium 4K gaming; fast inference"),
    ("rtx 4080",          "RTX 4080",             "Ada Lovelace", "16 GB",  "256-bit", "GDDR6X", 716,  "Premium 4K gaming; fast inference"),
    ("rtx 4090",          "RTX 4090",             "Ada Lovelace", "24 GB",  "384-bit", "GDDR6X", 1008, "Pre-Blackwell flagship; the reference for fast 24 GB local AI"),
    ("rtx 5060 ti",       "RTX 5060 Ti 8/16GB",   "Blackwell",    "8/16 GB", "128-bit", "GDDR7", 448,  "16 GB on a budget; GDDR7 helps token speed"),
    ("rtx 5060",          "RTX 5060",             "Blackwell",    "8 GB",   "128-bit", "GDDR7",  448,  "Entry level; 8 GB limits it to small models"),
    ("rtx 5070 ti",       "RTX 5070 Ti",          "Blackwell",    "16 GB",  "256-bit", "GDDR7",  896,  "High-performance 16 GB option with near-1 TB/s bandwidth"),
    ("rtx 5070",          "RTX 5070",             "Blackwell",    "12 GB",  "192-bit", "GDDR7",  672,  "Efficient mid-range with GDDR7 throughput"),
    ("rtx 5080",          "RTX 5080",             "Blackwell",    "16 GB",  "256-bit", "GDDR7",  960,  "Very fast token generation and top-tier 4K gaming"),
    ("rtx 5090",          "RTX 5090",             "Blackwell",    "32 GB",  "512-bit", "GDDR7",  1792, "Local AI powerhouse; 32 GB runs larger models at very high speed"),
]


def _norm(name):
    return " ".join(name.lower().replace("nvidia", " ").replace("geforce", " ").split())


def lookup(name, vram_gb=None):
    """The GPU_TABLE row for a GPU name (NVML / nvidia-smi spelling), or None."""
    n = _norm(name or "")
    best = None
    for row in GPU_TABLE:
        key, _, variant = row[0].partition("|")
        if key not in n:
            continue
        if variant and vram_gb is not None and abs(vram_gb - float(variant)) > 1.5:
            continue
        if best is None or len(key) > len(best[0].partition("|")[0]):
            best = row
    return best


def detected():
    """[(name, GB)] of the local GPUs (NVML / nvidia-smi), [] when none or not readable."""
    try:
        from .telemetry.gpu import gpu_inventory
        return gpu_inventory()
    except Exception:
        return []


def resolve(raw=None, gpus=None):
    """(GB/s, where it came from) for ESTIMATE_GPU_BW_GBS: a number wins; `auto` / empty looks up the first detected
    GPU that is in GPU_TABLE; otherwise FALLBACK_GBS."""
    raw = (raw or "").strip().lower()
    if raw and raw != "auto":
        return float(raw), "ESTIMATE_GPU_BW_GBS"
    gpus = detected() if gpus is None else gpus
    for name, gb in gpus:
        row = lookup(name, gb)
        if row:
            return float(row[6]), f"{row[1]} (detected)"
    if gpus:
        return FALLBACK_GBS, f"{gpus[0][0]} is not in the table: {FALLBACK_GBS:g} GB/s assumed"
    return FALLBACK_GBS, f"no GPU detected here: {FALLBACK_GBS:g} GB/s assumed"


def valid(raw):
    """Problem text for an ESTIMATE_GPU_BW_GBS value, or None."""
    raw = (raw or "").strip().lower()
    if not raw or raw == "auto":
        return None
    try:
        if float(raw) > 0:
            return None
    except ValueError:
        pass
    return "must be auto or a bandwidth in GB/s (e.g. 672)"
