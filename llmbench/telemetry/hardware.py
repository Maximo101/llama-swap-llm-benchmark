"""Hardware description for the dashboard header."""
from __future__ import annotations

import re
from pathlib import Path

from .gpu import gpu_inventory

def detect_hardware(cfg) -> str:
    """'CPU · RAM · GPU' for the dashboard header. HW_CPU / HW_RAM / HW_GPU override auto-detection
    (auto-detect cannot know e.g. the RAM generation, so set HW_RAM='64 GB DDR5' if you want it shown)."""
    cpu = cfg.hw_cpu
    if not cpu:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        if cpu:
            cpu = re.sub(r"\((?:R|TM)\)", "", cpu)
            cpu = re.sub(r"\s+@.*$", "", cpu)
            cpu = re.sub(r"\bCPU\b", "", cpu)
            cpu = re.sub(r"\s+", " ", cpu).strip()
    ram = cfg.hw_ram
    if not ram:
        try:
            kb = next(float(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                      if l.startswith("MemTotal"))
            gb, step = kb / 1024 ** 2, None
            step = 8 if gb >= 24 else 4
            ram = f"{max(step, round(gb / step) * step)} GB RAM"
        except (OSError, StopIteration, ValueError):
            pass
    gpu = cfg.hw_gpu
    if not gpu:
        # NVML first (no subprocess), nvidia-smi only if the bindings are missing.
        names = [f"{n.replace('NVIDIA ', '').replace('GeForce ', '')} {round(gb)} GB" for n, gb in gpu_inventory()]
        gpu = " + ".join(names) or None
    return " · ".join(x for x in (cpu, ram, gpu) if x)
