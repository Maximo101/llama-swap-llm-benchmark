"""GPU telemetry.

Two interchangeable backends, selected by GPU_BACKEND (auto | nvml | smi):

  * NvmlGpuMonitor  - talks to the driver through NVML (pip install nvidia-ml-py; imported as `pynvml`).
                      One in-process thread, direct library calls: no child process, no CSV text parsing, no
                      "[N/A]" string handling, real exceptions, and access to the cumulative energy counter.
  * SmiGpuMonitor   - the original persistent `nvidia-smi -lms N` stream. Kept ONLY as a fallback for machines
                      where the NVML Python bindings are not installed.

Both expose the same small interface, so the rest of the benchmark does not care which one is active:

    mon.ok            -> bool, True when samples are flowing
    mon.backend       -> "nvml" | "smi" | "none"
    mon.window(t0,t1) -> list of sample dicts {gpu_w, vram_gb, vram_total_gb, gpu_temp, gpu_util}
    mon.energy_mj()   -> cumulative GPU energy in millijoules (NVML only, opt-in) or None
    mon.stop()
"""
from __future__ import annotations

import subprocess
import threading
import time

from ..common import TimeSeries


class _Monitor:
    backend = "none"

    def __init__(self, cfg):
        self.cfg = cfg
        self.series = TimeSeries()
        self.ok = False
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:                 # pragma: no cover - overridden
        return False

    def window(self, t0, t1):
        return self.series.window(t0, t1)

    def energy_mj(self):
        return None

    def stop(self):
        self._halt.set()


# --------------------------------------------------------------------------- #
# NVML
# --------------------------------------------------------------------------- #

class NvmlGpuMonitor(_Monitor):
    """Polls NVML every GPU_POLL_MS. Multiple GPUs are aggregated (power/VRAM summed, temp/util = max)
    unless GPU_INDEX selects some. GPU_INDEX accepts comma-separated indices, UUIDs or PCI bus ids."""

    backend = "nvml"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._nv = None
        self._handles: list = []
        self._energy_ok = False

    # -- setup -------------------------------------------------------------
    def _resolve_handles(self):
        nv = self._nv
        if not self.cfg.gpu_index:
            return [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        handles = []
        for tok in (t.strip() for t in str(self.cfg.gpu_index).split(",") if t.strip()):
            if tok.isdigit():
                handles.append(nv.nvmlDeviceGetHandleByIndex(int(tok)))
                continue
            for getter in (nv.nvmlDeviceGetHandleByUUID, nv.nvmlDeviceGetHandleByPciBusId):
                for arg in (tok, tok.encode()):       # older bindings want bytes, newer ones accept str
                    try:
                        handles.append(getter(arg))
                        break
                    except Exception:
                        continue
                else:
                    continue
                break
        return handles

    def start(self) -> bool:
        try:
            import pynvml as nv                      # provided by `pip install nvidia-ml-py`
        except ImportError:
            return False
        try:
            nv.nvmlInit()
            self._nv = nv
            self._handles = self._resolve_handles()
        except Exception:                            # NVMLError_* (no driver, no permission, bad index...)
            return False
        if not self._handles:
            return False
        if self.cfg.gpu_energy_counter:
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(self._handles[0])    # Volta or newer only
                self._energy_ok = True
            except Exception:
                self._energy_ok = False
        first = self._sample()
        if first is None:
            return False
        self.series.append(first)
        self.ok = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nvml-poll")
        self._thread.start()
        return True

    # -- sampling ----------------------------------------------------------
    def _query(self, fn, *args):
        """One NVML call -> value, or None when this GPU does not support it (e.g. power on some laptops)."""
        try:
            return fn(*args)
        except Exception:
            return None

    def _sample(self):
        nv, rows = self._nv, []
        for h in self._handles:
            mw = self._query(nv.nvmlDeviceGetPowerUsage, h)                          # milliwatts
            mem = self._query(nv.nvmlDeviceGetMemoryInfo, h)                         # bytes
            temp = self._query(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
            util = self._query(nv.nvmlDeviceGetUtilizationRates, h)
            rows.append((None if mw is None else mw / 1000.0,
                         None if mem is None else mem.used / 1024 ** 3,
                         None if mem is None else mem.total / 1024 ** 3,
                         temp, None if util is None else util.gpu))
        if all(v is None for r in rows for v in r):
            return None

        def total(i):
            xs = [r[i] for r in rows if r[i] is not None]
            return sum(xs) if xs else None

        def peak(i):
            xs = [r[i] for r in rows if r[i] is not None]
            return max(xs) if xs else None
        return {"gpu_w": total(0), "vram_gb": total(1), "vram_total_gb": total(2),
                "gpu_temp": peak(3), "gpu_util": peak(4)}

    def _loop(self):
        interval = self.cfg.gpu_poll_ms / 1000
        nxt = time.monotonic()
        while not self._halt.is_set():
            s = self._sample()
            if s is not None:
                self.series.append(s)
            # Fixed cadence: sleep until the next tick instead of `interval` after the previous query, so
            # query time does not stretch the sampling period.
            nxt += interval
            self._halt.wait(max(0.0, nxt - time.monotonic()))

    # -- exact energy ------------------------------------------------------
    def energy_mj(self):
        """Sum of the per-GPU cumulative energy counters (mJ). Reading it before and after a request gives the
        exact energy of that request, independent of the polling rate. None unless GPU_ENERGY_COUNTER=true and
        the GPU supports it."""
        if not self._energy_ok:
            return None
        total = 0
        for h in self._handles:
            v = self._query(self._nv.nvmlDeviceGetTotalEnergyConsumption, h)
            if v is None:
                return None
            total += v
        return total

    def stop(self):
        super().stop()
        if self._nv is not None:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# nvidia-smi fallback (the original implementation, now on a TimeSeries)
# --------------------------------------------------------------------------- #

class SmiGpuMonitor(_Monitor):
    """One persistent `nvidia-smi -lms N` process; a reader thread timestamps every line."""

    backend = "smi"
    QUERY = "index,power.draw,memory.used,memory.total,temperature.gpu,utilization.gpu"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.proc = None
        self._latest: dict = {}

    def start(self) -> bool:
        cmd = ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
               "-lms", str(self.cfg.gpu_poll_ms)]
        if self.cfg.gpu_index:
            cmd += ["-i", str(self.cfg.gpu_index)]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1)
        except OSError:
            return False
        self._thread = threading.Thread(target=self._read, daemon=True, name="smi-read")
        self._thread.start()
        deadline = time.time() + 6
        while time.time() < deadline:
            if len(self.series):
                self.ok = True
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(float(parts[0]))
                w, used, total, temp, util = (float(x) for x in parts[1:6])
            except ValueError:                       # "[N/A]" fields
                continue
            self._latest[idx] = (w, used, total, temp, util)
            v = list(self._latest.values())
            self.series.append({"gpu_w": sum(x[0] for x in v), "vram_gb": sum(x[1] for x in v) / 1024,
                                "vram_total_gb": sum(x[2] for x in v) / 1024,
                                "gpu_temp": max(x[3] for x in v), "gpu_util": max(x[4] for x in v)})

    def stop(self):
        super().stop()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class NoGpuMonitor(_Monitor):
    """Placeholder when no GPU telemetry is available: every window is empty, so GPU columns stay blank."""
    backend = "none"


# --------------------------------------------------------------------------- #
# Selection + helpers
# --------------------------------------------------------------------------- #

def open_gpu_monitor(cfg, verbose=False):
    """Start the monitor chosen by GPU_BACKEND. auto = NVML first, nvidia-smi if NVML is unavailable.
    Always returns an object with the monitor interface; check `.ok`."""
    order = {"nvml": [NvmlGpuMonitor], "smi": [SmiGpuMonitor]}.get(cfg.gpu_backend, [NvmlGpuMonitor, SmiGpuMonitor])
    for cls in order:
        mon = cls(cfg)
        if mon.start():
            return mon
        mon.stop()
        if verbose:
            print(f"  [gpu] {cls.backend} backend unavailable")
    return NoGpuMonitor(cfg)


def gpu_inventory():
    """[(name, total_GB)] for every visible GPU. NVML first, `nvidia-smi` as the fallback. [] when none."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            out = []
            for i in range(nv.nvmlDeviceGetCount()):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                name = nv.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                out.append((name, nv.nvmlDeviceGetMemoryInfo(h).total / 1024 ** 3))
            return out
        finally:
            nv.nvmlShutdown()
    except Exception:
        pass
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
        out = []
        for line in text.strip().splitlines():
            name, mib = [x.strip() for x in line.rsplit(",", 1)]
            out.append((name, float(mib) / 1024))
        return out
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
