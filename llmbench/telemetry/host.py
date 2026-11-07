"""Host-side telemetry: CPU power (RAPL / Home Assistant), CPU util, RAM, inference-server process."""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import requests

from ..common import TimeSeries, avg

class HostMeter:
    """CPU package power (RAPL energy counters), CPU utilisation (/proc/stat), RAM (/proc/meminfo).
    RAPL energy_uj is often root-only; run as root (e.g. the Unraid terminal) or relax permissions."""

    def __init__(self, cfg):
        self.ram = RamSampler(cfg)
        self.domains = []
        for d in sorted(Path(cfg.rapl_path).glob("intel-rapl:*")):
            if re.fullmatch(r"intel-rapl:\d+", d.name):     # package domains only, no sub-zones
                try:
                    int((d / "energy_uj").read_text())
                    self.domains.append((d / "energy_uj", int((d / "max_energy_range_uj").read_text())))
                except (OSError, ValueError):
                    pass

    @property
    def rapl_ok(self): return bool(self.domains)

    @staticmethod
    def _stat():
        try:
            f = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
            return sum(f), f[3] + f[4]
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _mem_gb():
        try:
            m = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
            kb = lambda k: float(m[k].split()[0])
            return (kb("MemTotal") - kb("MemAvailable")) / 1024 ** 2
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _mem_total_gb():
        try:
            m = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
            return float(m["MemTotal"].split()[0]) / 1024 ** 2
        except (OSError, ValueError, KeyError):
            return None

    def snap(self):
        energy = []
        for path, mx in self.domains:
            try:
                energy.append((int(path.read_text()), mx))
            except (OSError, ValueError):
                energy.append(None)
        return {"t": time.time(), "energy": energy, "stat": self._stat()}

    def diff(self, a, b) -> dict:
        out, dt = {}, b["t"] - a["t"]
        if self.domains and dt > 0 and None not in a["energy"] + b["energy"]:
            uj = 0
            for (e0, mx), (e1, _) in zip(a["energy"], b["energy"]):
                uj += e1 - e0 + (mx if e1 < e0 else 0)      # counter wrap
            out["cpu_w"] = uj / 1e6 / dt
        if a["stat"] and b["stat"] and b["stat"][0] > a["stat"][0]:
            out["cpu_util"] = 100 * (1 - (b["stat"][1] - a["stat"][1]) / (b["stat"][0] - a["stat"][0]))
        out["sys_mem_gb"] = self._mem_gb()
        return out


_FA_VALUES = ("on", "off", "auto", "1", "0", "true", "false", "enabled", "disabled")


def parse_launch_args(argv) -> dict:
    """Read the llama-server launch flags that decide what lives in system RAM instead of VRAM, and how big the KV
    cache is: -ngl, --n-cpu-moe / --cpu-moe, -ot, -nkvo, -c, -ctk / -ctv, -fa, -ub / -b, --mmproj, -np, -m, --jinja."""
    toks = []
    for a in argv:
        toks.extend(a.split("=", 1) if a.startswith("--") and "=" in a else [a])
    info = {"ngl": None, "n_cpu_moe": None, "cpu_moe": False, "no_kv_offload": False, "ot_cpu": False, "ctx": None,
            "ctk": None, "ctv": None, "fa": None, "ubatch": None, "batch": None, "mmproj": None, "parallel": None,
            "model_path": None, "jinja": False, "ot": None, "swa_full": False}
    for i, a in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        try:
            if a in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
                info["ngl"] = 999 if nxt.lower() == "all" else int(nxt)
            elif a in ("--n-cpu-moe", "-ncmoe"):
                info["n_cpu_moe"] = int(nxt)
            elif a in ("--cpu-moe", "-cmoe"):
                info["cpu_moe"] = True
            elif a in ("--no-kv-offload", "-nkvo"):
                info["no_kv_offload"] = True
            elif a in ("-ot", "--override-tensor"):
                info["ot_cpu"] = info["ot_cpu"] or "cpu" in nxt.lower()
                info["ot"] = (info["ot"] + ";" if info["ot"] else "") + nxt[:120]
            elif a in ("-c", "--ctx-size"):
                info["ctx"] = int(nxt)
            elif a in ("-ctk", "--cache-type-k"):
                info["ctk"] = nxt.lower()
            elif a in ("-ctv", "--cache-type-v"):
                info["ctv"] = nxt.lower()
            elif a in ("-fa", "--flash-attn"):
                # newer llama.cpp: -fa on|off|auto; older: a bare switch
                info["fa"] = nxt.lower() if nxt.lower() in _FA_VALUES else "on"
                info["fa"] = {"1": "on", "true": "on", "enabled": "on", "0": "off", "false": "off",
                              "disabled": "off"}.get(info["fa"], info["fa"])
            elif a in ("--no-flash-attn", "-nofa"):
                info["fa"] = "off"
            elif a in ("-ub", "--ubatch-size"):
                info["ubatch"] = int(nxt)
            elif a in ("-b", "--batch-size"):
                info["batch"] = int(nxt)
            elif a in ("--mmproj", "-mm"):
                info["mmproj"] = nxt
            elif a in ("-np", "--parallel"):
                info["parallel"] = int(nxt)
            elif a in ("-m", "--model"):
                info["model_path"] = nxt
            elif a == "--jinja":
                info["jinja"] = True
            elif a == "--swa-full":
                info["swa_full"] = True
        except ValueError:
            pass
    return info


def offload_summary(info) -> str:
    """Human-readable list of CPU-offload flags; '' means the model was launched fully on the GPU."""
    parts = []
    if info["n_cpu_moe"]:
        parts.append(f"n-cpu-moe {info['n_cpu_moe']}")
    if info["cpu_moe"]:
        parts.append("cpu-moe")
    if info["ngl"] is not None and info["ngl"] < 99:
        parts.append(f"ngl {info['ngl']}")
    if info["no_kv_offload"]:
        parts.append("KV on CPU")
    if info["ot_cpu"]:
        parts.append("override-tensor CPU")
    return ", ".join(parts)


def find_listener_pids(port) -> set:
    """PIDs of processes LISTENing on a TCP port, whatever they are called (needs root to see other users' fds)."""
    inodes = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(parts[9])
    pids = set()
    if not inodes:
        return pids
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{d}/fd"):
                try:
                    link = os.readlink(f"/proc/{d}/fd/{fd}")
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    pids.add(int(d))
                    break
        except OSError:
            continue
    return pids


class RamSampler(threading.Thread):
    """Every RAM_POLL_MS: system RAM in use (MemTotal - MemAvailable, the figure htop / Home Assistant show)
    and the resident memory (RSS) of the inference-server process(es), matched by executable name
    (RAM_PROCESS_PATTERN). The process RSS is the best available estimate of how much of the model and
    cache lives in system RAM instead of VRAM. Processes in Docker are visible from the host's /proc."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.interval = cfg.ram_poll_ms / 1000
        self.pattern = re.compile(cfg.ram_process_pattern)
        self.page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        self.series, self.pids, self._scanned = TimeSeries(), set(), 0.0
        self.listen_port = None      # set in Strata mode: find the server by the port it listens on
        self._halt, self._lock = threading.Event(), threading.Lock()

    def find_processes(self) -> set:
        if self.listen_port:      # Strata: no known executable name, so use the listening socket
            def comm(p):
                try:
                    return Path(f"/proc/{p}/comm").read_text().strip()
                except OSError:
                    return ""
            pids = {p for p in find_listener_pids(self.listen_port) if comm(p) != "docker-proxy"}
            if pids:
                return pids
        found = set()
        try:
            names = os.listdir("/proc")
        except OSError:
            return found
        for d in names:
            if not d.isdigit() or int(d) == os.getpid():
                continue
            try:
                argv0 = Path(f"/proc/{d}/cmdline").read_bytes().split(b"\0")[0].decode("utf-8", "ignore")
            except OSError:
                continue
            if argv0 and self.pattern.search(os.path.basename(argv0)):
                found.add(int(d))
        return found

    def launch_info(self):
        """Launch flags of the (largest) inference-server process, or None if unreadable."""
        with self._lock:
            pids = list(self.pids)
        best = max(pids, key=lambda p: self.rss_gb([p]) or 0, default=None)
        if best is None:
            return None
        try:
            argv = Path(f"/proc/{best}/cmdline").read_bytes().decode("utf-8", "ignore").split("\0")
        except OSError:
            return None
        info = parse_launch_args([a for a in argv[1:] if a])
        info["pid"] = best
        return info

    def rescan(self):
        found = self.find_processes()
        with self._lock:
            self.pids, self._scanned = found, time.time()

    def rss_gb(self, pids=None):
        with self._lock:
            pids = list(self.pids) if pids is None else pids
        total, seen = 0, False
        for pid in pids:
            try:
                total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * self.page
                seen = True
            except (OSError, ValueError, IndexError):
                with self._lock:
                    self.pids.discard(pid)
        return total / 1024 ** 3 if seen else None

    def run(self):
        while not self._halt.is_set():
            if time.time() - self._scanned > 3:
                self.rescan()
            self.series.append({"ram_gb": HostMeter._mem_gb(), "proc_gb": self.rss_gb()})
            self._halt.wait(self.interval)

    def window(self, t0, t1):
        return self.series.window(t0, t1)

    def stop(self): self._halt.set()


class HaCpuPoller(threading.Thread):
    """Optional CPU power from a Home Assistant sensor (used only when RAPL is unavailable).
    HA sensors update slowly, so values lag behind short requests."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg, self.samples, self._halt = cfg, [], threading.Event()
        self.headers = {"Authorization": f"Bearer {cfg.ha_token}"}

    def run(self):
        url = f"{self.cfg.ha_url}/api/states/{self.cfg.ha_cpu_entity}"
        while not self._halt.is_set():
            try:
                r = requests.get(url, headers=self.headers, timeout=3)
                if r.ok:
                    self.samples.append((time.time(), float(r.json().get("state"))))
            except (requests.RequestException, ValueError, TypeError):
                pass
            self._halt.wait(self.cfg.ha_interval)

    def window_avg(self, t0, t1):
        xs = [w for t, w in self.samples if t0 - self.cfg.ha_interval * 2 <= t <= t1]
        return avg(xs) if xs else (self.samples[-1][1] if self.samples else None)

    def stop(self): self._halt.set()
