"""Runs INSIDE the sandbox (Docker container or restricted subprocess). Not imported by the benchmark itself:
sandbox.py copies this file next to the generated program and executes it with `python -I runner.py`.

Protocol
  stdin  : "<nonce>\\n<json job>"   (the job is NEVER written to disk, so the untrusted program cannot read the
           expected outputs; the nonce authenticates our result line)
  stdout : exactly one line  @@RESULT@@<nonce>:<json result>
  The untrusted program runs as a child with its stdout/stderr redirected to files, and (POSIX) in its own
  process group so a timeout kills everything it spawned, not just the direct child.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile

# --- Hardening 1: become non-dumpable (PR_SET_DUMPABLE=0). On Linux this makes /proc/<this pid>/{mem,environ,fd/*}
# unreadable to other processes of the same user - including the untrusted child. Without it the child could open
# /proc/<ppid>/fd/1 and print a forged "@@RESULT@@..." line into our stdout pipe.
try:
    import ctypes
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
except Exception:
    pass

# --- Hardening 2: resource limits (inherited by the child)
try:
    import resource
    _mem = int(os.environ.get("LLMB_MEM_MB", "2048")) * 1024 * 1024
    for _lim, _val in ((resource.RLIMIT_AS, _mem), (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
                       (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(_lim, (_val, _val))
        except (ValueError, OSError):
            pass
except ImportError:
    pass

# --- Job description from stdin
_raw = sys.stdin.buffer.read()
_nonce, _, _body = _raw.partition(b"\n")
NONCE = _nonce.decode("ascii", "replace")
spec = json.loads(_body)

PROG = os.path.abspath("prog.py")
SCRATCH = tempfile.mkdtemp()
TIMEOUT = float(spec["timeout"])
SENTINEL = spec.get("sentinel")


def _kill_group(p):
    try:
        if hasattr(os, "killpg"):
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()
    except (OSError, ProcessLookupError):
        pass


def run(stdin_text):
    """Execute prog.py once. Returns (returncode, stdout, stderr, timed_out)."""
    op, ep = os.path.join(SCRATCH, "out"), os.path.join(SCRATCH, "err")
    with open(op, "wb") as fo, open(ep, "wb") as fe:
        p = subprocess.Popen([sys.executable, "-I", PROG], stdin=subprocess.PIPE, stdout=fo, stderr=fe,
                             cwd=SCRATCH, start_new_session=True)
        try:
            p.communicate(input=stdin_text.encode("utf-8", "replace"), timeout=TIMEOUT)
            rc, to = p.returncode, False
        except subprocess.TimeoutExpired:
            _kill_group(p)
            try:
                p.communicate(timeout=5)
            except Exception:
                pass
            rc, to = -9, True
        else:
            _kill_group(p)            # reap anything the program left running in the background
    out = open(op, "rb").read(4000000).decode("utf-8", "replace")
    err = open(ep, "rb").read(20000).decode("utf-8", "replace")
    return rc, out, err, to


def num_eq(u, v):
    if u == v:
        return True
    try:
        if u.lstrip("+-").isdigit() and v.lstrip("+-").isdigit():
            return int(u) == int(v)
        fu, fv = float(u), float(v)
    except ValueError:
        return False
    return abs(fu - fv) <= 1e-6 * max(1.0, abs(fv))


def same_text(out, exp):
    a = [l.rstrip() for l in out.strip().splitlines()]
    b = [l.rstrip() for l in exp.strip().splitlines()]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty) or not all(num_eq(u, v) for u, v in zip(tx, ty)):
            return False
    return True


def json_eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_eq(a[k], b[k]) for k in a)
    return a == b


def same_json(out, exp):
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if not lines:
        return False
    try:
        return json_eq(json.loads(lines[-1]), json.loads(exp))
    except ValueError:
        return lines[-1].strip() == exp.strip()


mode = spec["mode"]
res = {"ok": False, "reason": ""}
if mode == "script":
    rc, out, err, to = run("")
    if to:
        res["reason"] = "timeout"
    elif rc != 0:
        res["reason"] = (err.strip().splitlines() or ["exit %d" % rc])[-1][:200]
    elif SENTINEL and SENTINEL not in out:
        # Exit code 0 but the line appended after the tests never printed: the program quit early
        # (sys.exit / exit / os._exit at module level) before the tests ran. That is not a pass.
        res["reason"] = "program exited before the tests finished"
    else:
        res["ok"] = True
else:
    tests, passed = spec["tests"], 0
    res["total"] = len(tests)
    for i, t in enumerate(tests):
        rc, out, err, to = run(t["input"])
        if to:
            res["reason"] = "timeout on test %d" % i
            break
        if rc != 0:
            res["reason"] = "runtime error on test %d: %s" % (i, (err.strip().splitlines() or ["exit %d" % rc])[-1][:160])
            break
        if not (same_json(out, t["output"]) if mode == "call" else same_text(out, t["output"])):
            res["reason"] = "wrong answer on test %d" % i
            break
        passed += 1
    else:
        res["ok"] = True
    res["passed"] = passed
print("@@RESULT@@" + NONCE + ":" + json.dumps(res), flush=True)
