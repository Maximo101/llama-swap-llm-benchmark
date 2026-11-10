"""Sandboxed execution of model-written code (HumanEval, MBPP, LiveCodeBench).

SECURITY MODEL - generated code is untrusted.

  docker      A throw-away container per program: no network, read-only root filesystem, non-root user,
              all capabilities dropped, no-new-privileges, memory/CPU/pid/file-descriptor limits, a private
              tmpfs, an init process to reap children, and a label so leftovers can always be found and removed.
              This IS a security boundary (as far as Docker is).
  subprocess  A child Python with a stripped environment (your API keys are NOT passed on), a scratch directory,
              rlimits and - on Linux, when `unshare -rn` works - no network. It still runs as YOUR user and can
              read your files. This is NOT a security boundary against deliberately hostile code.

Defence in depth that applies to both modes:
  * the job (tests + expected outputs) is sent over stdin, never written to a file the program could read;
  * the result line carries a per-run random nonce and the runner is non-dumpable, so the program cannot forge it;
  * for HumanEval/MBPP a unique sentinel is appended AFTER the tests; a program that calls sys.exit(0) early
    exits cleanly but never prints it, so it is graded as a failure instead of a pass;
  * a timeout kills the whole process group (and `docker kill`s the container), not just the direct child.
"""
from __future__ import annotations

import atexit
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

LABEL = "llmbench-sandbox"
RUNNER_SRC = Path(__file__).with_name("sandbox_runner.py").read_text(encoding="utf-8")

_NETLESS = None          # cached probe: can we drop network access with `unshare -rn`?


def _can_unshare():
    global _NETLESS
    if _NETLESS is None:
        _NETLESS = False
        if sys.platform.startswith("linux") and shutil.which("unshare"):
            try:                                  # probed as the user the programs will run as
                _NETLESS = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10,
                                          **drop_privileges()).returncode == 0
            except (OSError, subprocess.SubprocessError):
                _NETLESS = False
    return _NETLESS


NOBODY = 65534
_DROP = None             # cached: Popen arguments that run the child as an unprivileged user (when we are root)


def drop_privileges():
    """v2.3.1, subprocess sandbox: when the benchmark runs as root (Unraid User Scripts, a root shell), code written
    by a model must not. -> Popen keyword arguments that start the child as user/group 65534 ("nobody"), or {} when
    we are not root or cannot change user (then a warning is printed once: the code runs as root)."""
    global _DROP
    if _DROP is None:
        _DROP = {}
        if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
            kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []}
            try:
                if subprocess.run([sys.executable, "-I", "-c", "pass"], capture_output=True, timeout=20,
                                  **kw).returncode == 0:
                    _DROP = kw
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            if not _DROP:
                print("  [warn] code sandbox: running as root and could not switch to an unprivileged user, so "
                      "model-written code runs AS ROOT. Use EVAL_CODE_SANDBOX=docker.")
    return _DROP


def sandbox_workers(cfg):
    """How many programs may run at once. Each Docker container is limited to 1 CPU, so default to a few."""
    return cfg.eval_sandbox_workers or max(1, min(4, os.cpu_count() or 1))


# --------------------------------------------------------------------------- #
# Docker housekeeping
# --------------------------------------------------------------------------- #

def cleanup_orphans():
    """Remove any container this tool started and left behind (Ctrl-C, crash, kill -9 of an earlier run)."""
    if not shutil.which("docker"):
        return
    try:
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}=1"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass


def _docker_ready(cfg):
    """(ok, why-not). Pulls the image once if it is missing."""
    if not shutil.which("docker"):
        return False, "docker not found on PATH"
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "the docker daemon is not reachable"
        if subprocess.run(["docker", "image", "inspect", cfg.eval_code_docker_image],
                          capture_output=True, timeout=30).returncode == 0:
            return True, ""
        print(f"  [eval] pulling {cfg.eval_code_docker_image} for the code sandbox ...")
        if subprocess.run(["docker", "pull", cfg.eval_code_docker_image], timeout=600).returncode == 0:
            return True, ""
        img = cfg.eval_code_docker_image
        tip = "" if img == "python:3.11-slim" else ", or remove the line for the default python:3.11-slim"
        return False, (f"could not pull the sandbox image {img}: check EVAL_CODE_DOCKER_IMAGE in .env "
                       f"(a locally built image must exist on this machine: docker image ls{tip})")
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


# --------------------------------------------------------------------------- #
# Running one program
# --------------------------------------------------------------------------- #

def _docker_cmd(cfg, name, td):
    mem = max(256, cfg.eval_code_mem_mb)
    return ["docker", "run", "--rm", "-i", "--name", name, "--label", f"{LABEL}=1",
            "--network", "none",
            "--memory", f"{mem}m", "--memory-swap", f"{mem}m", "--cpus", "1", "--pids-limit", "256",
            "--ulimit", "nofile=256:256", "--ulimit", "core=0",
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--init",
            "-e", f"LLMB_MEM_MB={mem}", "-v", f"{td}:/work:ro", "-w", "/work",
            cfg.eval_code_docker_image, "python", "-I", "runner.py"]


def _subprocess_cmd(cfg, td):
    tmp = os.path.join(td, "tmp")
    os.makedirs(tmp, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": td, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp,
           "LLMB_MEM_MB": str(cfg.eval_code_mem_mb), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "LANG": "C.UTF-8"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    cmd = [sys.executable, "-I", "runner.py"]
    if _can_unshare():
        cmd = ["unshare", "-rn"] + cmd
    return cmd, env


def run_in_sandbox(cfg, program, mode, tests, per_test_timeout):
    """Run `program`. mode: 'script' (pass = exit 0 AND the end-of-program sentinel printed), 'stdin' or 'call'
    (graded per test). Thread-safe: every call uses its own temp dir, container name and nonce.
    Returns (ok, reason)."""
    runs = max(1, len(tests)) if mode != "script" else 1
    total = per_test_timeout * runs + 20                  # + container start-up / image warm-up allowance
    nonce, sentinel = secrets.token_hex(16), "@@PASS-" + secrets.token_hex(16)
    if mode == "script":
        # os.write(1, ...) bypasses a redefined print()/sys.stdout and is flushed immediately
        program += f"\n\n__import__('os').write(1, b'{sentinel}\\n')\n"
    job = json.dumps({"mode": mode, "tests": tests, "timeout": per_test_timeout,
                      "sentinel": sentinel if mode == "script" else None}).encode()
    payload = nonce.encode() + b"\n" + job
    docker = cfg.eval_sandbox_resolved == "docker"
    with tempfile.TemporaryDirectory(prefix="llmbench_") as td:
        os.chmod(td, 0o755)                               # readable by the unprivileged container user
        for fname, text in (("prog.py", program), ("runner.py", RUNNER_SRC)):
            fp = os.path.join(td, fname)
            with open(fp, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
            os.chmod(fp, 0o644)
        name = f"llmbench-{uuid.uuid4().hex[:12]}"
        drop = {}
        if docker:
            cmd, env = _docker_cmd(cfg, name, td), None
        else:
            cmd, env = _subprocess_cmd(cfg, td)
            drop = drop_privileges()
            if drop:                                      # its scratch folder must be writable by that user
                try:
                    os.chown(os.path.join(td, "tmp"), NOBODY, NOBODY)
                except OSError:
                    pass
        proc = subprocess.Popen(cmd, cwd=td, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=(os.name == "posix"), **drop)
        try:
            out, err = proc.communicate(input=payload, timeout=total)
        except subprocess.TimeoutExpired:
            if docker:
                subprocess.run(["docker", "kill", name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
            except (OSError, ProcessLookupError):
                pass
            proc.communicate()
            return False, "sandbox timeout"
    prefix = "@@RESULT@@" + nonce + ":"
    for line in reversed(out.decode("utf-8", "replace").splitlines()):
        if line.startswith(prefix):
            res = json.loads(line[len(prefix):])
            return bool(res.get("ok")), res.get("reason", "")
    tail = (err.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1][:200]
    return False, f"sandbox error: {tail}"


# --------------------------------------------------------------------------- #
# Choosing a sandbox and proving it works
# --------------------------------------------------------------------------- #

def selftest(cfg):
    """Run known programs through the sandbox. Returns [(description, passed)]. Used by resolve_sandbox() and --check
    so a broken Docker setup is caught before a long run instead of silently scoring every answer 0 %."""
    forge = ("import os\n"
             "fd = os.open('/proc/%d/fd/1' % os.getppid(), os.O_WRONLY)\n"
             "os.write(fd, b'@@RESULT@@guess:{\"ok\": true}\\n')\n"
             "raise SystemExit(1)\n")
    net = ("import socket, sys\n"
           "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    sys.exit(1)\n"
           "except OSError:\n    pass\n")
    cases = [("passing program is accepted", "assert 1 + 1 == 2\n", 5, True),
             ("failing assert is rejected", "assert 1 + 1 == 3\n", 5, False),
             ("early sys.exit(0) is NOT a pass", "import sys\nsys.exit(0)\n", 5, False),
             ("infinite loop is stopped by the timeout", "while True:\n    pass\n", 2, False),
             ("forged result line is ignored", forge, 5, False)]
    out = []
    for label, code, to, expect in cases:
        try:
            ok, _ = run_in_sandbox(cfg, code, "script", [], to)
        except Exception:
            ok = None
        out.append((label, ok is not None and ok == expect))
    try:
        ok, _ = run_in_sandbox(cfg, net, "script", [], 8)
        out.append(("outbound network is blocked", bool(ok)))
    except Exception:
        out.append(("outbound network is blocked", False))
    return out


def resolve_sandbox(cfg, run_selftest=True):
    """Decide between Docker and plain subprocess (EVAL_CODE_SANDBOX), verify it works, print what will be used."""
    want, choice = cfg.eval_code_sandbox, "subprocess"
    if want in ("auto", "docker"):
        ok, why = _docker_ready(cfg)
        if ok:
            choice = "docker"
        elif want == "docker":
            sys.exit(f"[eval] EVAL_CODE_SANDBOX=docker but {why}.")
        else:
            # v2.1: 'auto' no longer falls back silently. A long unattended run would otherwise execute untrusted,
            # model-written code as your user with only a warning in the log.
            sys.exit(f"[eval] Docker is unavailable ({why}), so model-written code cannot be isolated.\n"
                     "        Fix Docker, or set EVAL_CODE_SANDBOX=subprocess to accept running generated code as your "
                     "user\n        (timeout + rlimits" + (", no network" if _can_unshare() else "") + "; NOT a "
                     "security boundary), or disable the code datasets (HumanEval, MBPP, LiveCodeBench).")
    cfg.eval_sandbox_resolved = choice
    if choice == "docker":
        cleanup_orphans()
        atexit.register(cleanup_orphans)
        print("  [eval] code sandbox: Docker (no network, read-only, non-root, memory/CPU/pid limits)")
        try:                                              # which image exactly: a tag such as python:3.11-slim moves
            img = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", cfg.eval_code_docker_image],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if img:
                cfg.eval_sandbox_image_id = img
                print(f"  [eval] sandbox image {cfg.eval_code_docker_image} = {img[:19]}"
                      + ("" if "@sha256:" in cfg.eval_code_docker_image else
                         " (a tag can change: pin it with EVAL_CODE_DOCKER_IMAGE=<image>@sha256:<digest>)"))
        except (OSError, subprocess.SubprocessError):
            pass
    elif want == "subprocess":
        if drop_privileges():
            print("  [eval] code sandbox: this run is root, so generated code runs as the unprivileged user 'nobody'")
        print("  [eval] code sandbox: subprocess (timeout + rlimits" + (", no network" if _can_unshare() else "")
              + "); not a security boundary against hostile code - use EVAL_CODE_SANDBOX=docker for that")
    if run_selftest:
        results = selftest(cfg)
        bad = [d for d, passed in results if not passed]
        if bad and choice == "docker":
            sys.exit("[eval] Docker sandbox self-test failed: " + "; ".join(bad))
        if bad:
            note = " (expected: a plain subprocess cannot isolate the network)" if bad == ["outbound network is blocked"] else ""
            print("  [warn] sandbox self-test: " + "; ".join(bad) + note)
        else:
            print(f"  [eval] sandbox self-test: {len(results)}/{len(results)} checks passed")
    return choice
