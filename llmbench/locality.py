"""Is the llama-swap server on THIS machine? (v2.3)

The speed test reads GPU power, VRAM, CPU power and RAM on the machine the benchmark runs on. When SERVER_URL
points at another machine (e.g. the settings page started on a desktop PC from a share of the server's folder), those
readings belong to the wrong machine: VRAM / power / the budget checks would be the desktop's. Evals don't depend on
it (they only talk HTTP), so an evals-only run from another machine is fine.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _local_ips():
    ips = {"127.0.0.1", "::1"}
    try:
        name = socket.gethostname()
        ips.add(name.lower())
        for fam, *_rest, addr in socket.getaddrinfo(name, None):
            ips.add(addr[0])
    except OSError:
        pass
    return ips


def server_is_local(url):
    """(True | False | None, host). None = can't tell (host doesn't resolve). Uses only local lookups; the route
    check (which local address would talk to the host) sends no packet."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return None, host
    if host in ("localhost", "host.docker.internal") or host.endswith(".localhost"):
        return True, host
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(host, None)}
    except OSError:
        return None, host
    if any(ipaddress.ip_address(a.split("%")[0]).is_loopback for a in addrs):
        return True, host
    local = _local_ips()
    if addrs & local:
        return True, host
    for a in addrs:                                  # the address this machine would use to reach the host is the host
        try:
            fam = socket.AF_INET6 if ":" in a else socket.AF_INET
            with socket.socket(fam, socket.SOCK_DGRAM) as s:
                s.connect((a, 9))
                if s.getsockname()[0] == a:
                    return True, host
        except OSError:
            continue
    return False, host


def remote_warning(url):
    """A warning text when llama-swap runs on another machine, else None."""
    local, host = server_is_local(url)
    if local is not False:
        return None
    return (f"llama-swap ({host}) is on another machine. The speed test measures GPU power, VRAM, CPU power and RAM "
            f"on THIS machine, so those figures (and the VRAM budget checks) would be wrong. Run the speed test on "
            f"the machine with the GPU (for the settings page there: UI_HOST=0.0.0.0, then open it from here with "
            f"the printed token link). Evals only (--evals-only) are fine from here.")
