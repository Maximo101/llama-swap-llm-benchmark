"""Command line entry point and run orchestration.

A run is a sequence of small phases (each its own function, so each can be read, tested and changed alone):

    detect backend -> load eval data (fail fast) -> speed phase -> [early dashboard] -> eval phase -> outputs
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from . import status
from .client import (ctx_from_details, detect_backend, is_outage, list_models, probe_strata, strata_details,
                     strata_discover, wait_ready)
from .common import add_time, fmt_hr_m_s, stamp_line, now
from .config import EVAL_SPECS, load_config
from .diagnostics import check, demo_results
from .evals.datasets import download_datasets, prepare_eval_data
from .evals.runner import run_evals
from .reporting.csv_out import write_csv
from .reporting.dashboard import write_dashboards
from .reporting.json_out import json_paths, write_json
from .reporting.markdown import render_markdown
from .scoring import agent_notes, blank, finalize
from .speed import bench_model
from .telemetry import HaCpuPoller, HostMeter, detect_hardware, open_gpu_monitor

_DATASET_ENVS = ", ".join(s["env"] for s in EVAL_SPECS.values())


# --------------------------------------------------------------------------- #
# Console / log plumbing
# --------------------------------------------------------------------------- #

def run_footer(cfg, info, t_start, finished_ok):
    """Closing banner: when the run ended, how long it took, how many models, which profile and which tests ran."""
    when = datetime.now().astimezone()
    bar = "=" * 50
    lines = [bar,
             f"Benchmark {'complete' if finished_ok else 'stopped'} at: {when:%a %b} {when.day:>2} {when:%H:%M:%S} "
             f"{when.tzname() or ''} {when:%Y}",
             f"Time taken to run benchmark: {fmt_hr_m_s(now() - t_start)}"]
    results = info.get("results") or []
    if info.get("demo"):
        lines.append("Demo run: synthetic data, no real tests were run")
    elif results:
        ran = [EVAL_SPECS[k]["short"] for k in EVAL_SPECS if any(r.get(f"eval_{k}_n") for r in results)]
        if any(r.get("longctx_avg_pct") is not None or r.get("longctx_note") for r in results):
            ran.append("Long-ctx")
        lines.append(f"{len(results)} Models" + (f" using EVAL_PROFILE '{cfg.eval_profile}'" if ran else ""))
        tests = (["Speed test"] if info.get("speed") else []) + ran
        if tests:
            lines.append("Running benchmarking tests: " + ", ".join(tests))
    lines += [bar, f"Script Finished {when:%b %d, %Y}  {when:%H:%M.%S}"]
    return "\n".join(lines)


class Tee:
    """Copy everything written to a stream into the log file as well. Live '\\r' progress updates are collapsed the
    way a terminal would show them, so the log keeps one clean line per result instead of every counter tick."""

    def __init__(self, stream, fh):
        self.stream, self.fh, self._pending = stream, fh, ""

    def write(self, s):
        self.stream.write(s)
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self.fh.write(line.split("\r")[-1].rstrip() + "\n")
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()

    def close_log(self):
        if self._pending.strip():
            self.fh.write(self._pending.split("\r")[-1].rstrip() + "\n")
            self._pending = ""
        self.fh.flush()

    def __getattr__(self, name):        # isatty(), encoding, fileno()... come from the real stream
        return getattr(self.stream, name)


# --------------------------------------------------------------------------- #
# Run phases
# --------------------------------------------------------------------------- #

@dataclass
class RunState:
    """Everything the phases hand to each other. `results` is mutated in place (never rebound) because
    main() reads the same list for the closing banner even when a phase raises."""
    results: list = field(default_factory=list)
    cpu_source: str = "n/a"
    hardware: str = ""
    details_path: Path | None = None
    interrupted: bool = False
    prior: dict = field(default_factory=dict)      # --resume: {model: row} of the run being continued
    save: object = None                            # saves the run's rows (status.save_rows); None for a demo
    fresh_evals: bool = True                       # --resume: this run evaluates again (earlier eval figures dropped)


def _load_eval_data(cfg):
    """Load + sample the enabled datasets BEFORE the (long) speed test, so a missing package, dataset or cache
    fails immediately rather than hours later."""
    if not cfg.run_eval_tests:
        cfg.eval_longctx = False
        return {}
    if not cfg.eval_datasets:
        if not cfg.eval_longctx:
            print(f"[warn] RUN_EVAL_TESTS=true but no dataset is enabled ({_DATASET_ENVS}, EVAL_LONGCTX); "
                  "skipping the evals.")
        return {}
    return prepare_eval_data(cfg)


def _has_evals(cfg, eval_data):
    return bool(eval_data) or (cfg.run_eval_tests and cfg.eval_longctx and bool(cfg.longctx_lengths))


def _start_telemetry(cfg, session, models):
    """Start GPU / CPU / RAM sampling. Returns (gpu, host, ha, cpu_source)."""
    gpu = open_gpu_monitor(cfg)
    cfg.gpu_backend_used = gpu.backend
    host = HostMeter(cfg)
    host.ram.start()
    if cfg.backend == "strata":
        host.ram.listen_port = urlparse(cfg.base_url).port or 80
        found = strata_discover(cfg, session)
        cfg.strata_details = strata_details(found)
        cfg.strata_ctx_detected = ctx_from_details(cfg.strata_details)
        print(f"Strata: read {len(cfg.strata_details)} info endpoint(s); context "
              f"{cfg.strata_ctx or cfg.strata_ctx_detected or 'unknown'}")
    ha = None
    if host.rapl_ok:
        cpu_source = "RAPL package power"
    elif cfg.ha_url and cfg.ha_token and cfg.ha_cpu_entity:
        ha = HaCpuPoller(cfg)
        ha.start()
        cpu_source = "Home Assistant sensor (polled, may lag)"
    else:
        cpu_source = "unavailable (no RAPL / Home Assistant)"
    gpu_label = {"nvml": "NVML (pynvml)", "smi": "nvidia-smi stream"}.get(gpu.backend, "UNAVAILABLE")
    print(f"Found {len(models)} model(s). GPU telemetry: {gpu_label}; CPU power: {cpu_source}")
    if cfg.run_speed_tests:
        from .locality import remote_warning
        far = remote_warning(cfg.base_url)
        if far:
            print(f"\n{'!' * 50}\n[warn] {far}\n{'!' * 50}\n")
    if not gpu.ok:
        print("  [warn] no GPU data: run this on the GPU host with the NVIDIA driver loaded "
              "(pip install nvidia-ml-py, or have nvidia-smi on PATH; try --check).")
    return gpu, host, ha, cpu_source


def resumable(row, need_speed=True):
    """--resume: can this row of the earlier run be kept? Its model loaded and, for a run with speed tests, was
    measured. Everything else (failed to load, internal error, never reached) is benchmarked again."""
    return bool(row) and row.get("status") == "ok" and (not need_speed or row.get("avg_tps") is not None)


def restored(model, row, fresh_evals=True):
    """A row of the earlier run, on today's column set, marked so the eval phase keeps its long-context result.
    `fresh_evals`: this run evaluates again (from the result cache where it can), so the earlier run's eval figures,
    skip / screening notes and eval timings are dropped rather than mixed with the new ones."""
    r = blank(model)
    for k, v in row.items():
        if k == "model" or (fresh_evals and k.startswith("eval_")):
            continue
        r[k] = v
    if fresh_evals and isinstance(r.get("timing"), dict):
        r["timing"] = {k: v for k, v in r["timing"].items() if not k.startswith("eval")}
    r["_resumed"] = True
    return r


def _speed_phase(cfg, session, models, st, info):
    """Benchmark every model one request at a time. Returns True if the user interrupted."""
    keep = {m: st.prior[m] for m in models if resumable(st.prior.get(m))}
    if keep:
        print(f"Resuming: {len(keep)} of {len(models)} model(s) keep their speed test from run "
              f"{info.get('resumed_from')}; {len(models) - len(keep)} to benchmark")
    if keep and len(keep) == len(models):           # nothing to measure: no telemetry, no unloading
        st.results.extend(restored(m, keep[m], st.fresh_evals) for m in models)
        st.cpu_source = info.get("prior_cpu_source") or "kept from the resumed run"
        info["speed"] = True
        return False
    gpu, host, ha, st.cpu_source = _start_telemetry(cfg, session, models)
    state = {"warned": False}
    print("Starting Speed Test")
    info["speed"] = True
    stamp_line("Speed test started")
    interrupted = False
    try:
        for i, m in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {m}")
            status.update(force=True, phase="speed", model=m, model_i=i, models=len(models))
            if m in keep:
                print("  kept from the resumed run (speed test finished there)")
                st.results.append(restored(m, keep[m], st.fresh_evals))
                continue
            for attempt in range(2):
                try:
                    r = bench_model(cfg, session, gpu, host, ha, m, state)
                except Exception as e:           # BenchError is handled inside; this is a bug or an odd server reply
                    traceback.print_exc()
                    r = blank(m)
                    r.update(status="error", error=f"internal error: {e.__class__.__name__}: {e}")
                    break
                # v2.3.1: the server was not there (a restart, not this model's fault): wait for it, test once more
                if attempt or r["status"] != "error" or not is_outage(r["error"]) \
                        or str(r["error"]).startswith("Failed to load") or not wait_ready(cfg, session):
                    break
                print("  benchmarking this model again: the server did not answer")
            if r["status"] == "skipped":
                print("  SKIPPED: " + r["error"])
            elif r["status"] != "ok":
                print("  " + ("" if r["error"].startswith("Failed to load") else "ERROR: ") + r["error"])
            else:
                print(f"  tech {r['tech_tps']:.1f} / logic {r['logic_tps']:.1f} tok/s"
                      + (f" | GPU {r['gpu_w']:.0f} W, VRAM {r['vram_gb']:.1f} GB" if r["gpu_w"] and r["vram_gb"] else "")
                      + (f" | CPU {r['cpu_w']:.0f} W" if r["cpu_w"] else ""))
            st.results.append(r)
            if st.save:
                st.save()                      # everything measured so far survives a crash, a kill or a power cut
            if i < len(models):
                time.sleep(cfg.cooldown)
                add_time(r, "cooldown", cfg.cooldown)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted: writing partial results.")
    finally:                                   # telemetry is stopped BEFORE the evals so they cannot disturb it
        gpu.stop()
        host.ram.stop()
        if ha:
            ha.stop()
    print("\nSpeed Test Complete")
    stamp_line("Speed test finished")
    return interrupted


def _eval_phase(cfg, session, st, eval_data, stamp, info):
    t0 = now()
    try:
        st.details_path, info["funnel"] = run_evals(cfg, session, st.results, eval_data, stamp, st.hardware,
                                                    on_model=st.save)
    except KeyboardInterrupt:
        st.interrupted = True
        print("\nInterrupted: writing partial results (answers so far are in the result cache; rerun to resume).")
    except Exception as e:                       # a bug or an unforeseen server reply: keep everything measured so far
        st.interrupted = True
        info["error"] = f"eval phase stopped: {e.__class__.__name__}: {e}"
        traceback.print_exc()
        print("\n[error] the evaluation phase stopped on an internal error: writing the results obtained so far "
              "(answers are in the result cache; rerun to resume).")
    finally:
        info.setdefault("phases", {})["eval_s"] = round(now() - t0, 1)


def defer_lengths(cfg):
    """Prefill lengths the long-context test can measure instead. Not the PREFILL_DEPTH length: the prefill-at-depth
    test sends that cold prompt in the speed phase anyway (v2.3), so it measures it there."""
    return tuple(n for n in cfg.prefill_ctx_tests if n in cfg.longctx_lengths and n != cfg.prefill_depth)


def _defer_prefill(cfg, eval_data):
    """Prefill lengths that the long-context test measures anyway (its first, cold prompt of that length): the speed
    phase skips them instead of sending a second, separate 64k / 128k prompt."""
    if _has_evals(cfg, eval_data) and cfg.eval_longctx:
        cfg.defer_prefill = defer_lengths(cfg)
        if cfg.defer_prefill:
            print("Prefill at " + ", ".join(f"{n // 1024}k" for n in cfg.defer_prefill)
                  + " is taken from the long-context test's first prompt (no separate request)")


def _early_dashboard(cfg, st, stamp):
    """Safety net for the long eval phase: save a speed-test-only dashboard now. If the evals crash or the run is
    killed it is still there; when the evals finish, the final dashboard replaces it."""
    try:
        finalize(st.results, cfg)
        early, _ = write_dashboards(cfg, st.results, stamp, st.cpu_source, st.hardware, partial=True)
        print(f"Speed-test dashboard saved (replaced when the evals finish): {early}")
    except Exception as e:                      # never let this stop the evaluations
        print(f"[warn] could not save the speed-test dashboard: {e.__class__.__name__}: {e}")


def _mode(args, evals_only):
    return "demo" if args.demo else "evals-only" if evals_only else "speed-only" if args.speed_only else "full"


def previous_speeds(cfg):
    """{model: facts} from the newest finished run (OUTPUT_DIR/results.json), read BEFORE this run replaces it."""
    try:
        doc = json.loads((cfg.output_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for m in doc.get("models") or []:
        sp, md, gg = m.get("speed") or {}, m.get("metadata") or {}, m.get("gguf") or {}
        if m.get("status") == "ok" and sp.get("avg_tps"):
            out[m.get("model")] = {"tps": sp["avg_tps"], "build": sp.get("build_info"), "ctx": md.get("ctx_size"),
                                   "gb": gg.get("gguf_gb"), "offload": md.get("offload"), "run": (doc.get("run") or {}).get("id")}
    return out


def speed_changes(results, prev, stamp=None):
    """v2.3.1: tokens/s against the previous run of the SAME model file at the same context and offload. Fills
    tps_prev / tps_change_pct / build_prev and returns the lines worth printing (a change of 5% or more). This is
    what shows a llama.cpp update, a driver or a launch-flag change that made a model slower."""
    lines = []
    for r in results:
        p = prev.get(r["model"])
        if not p or r["status"] != "ok" or not r.get("avg_tps") or p.get("run") == stamp or r.get("_resumed"):
            continue
        same = (p["ctx"] == r.get("ctx_size") and p.get("offload") == r.get("offload")
                and (p["gb"] is None or r.get("gguf_gb") is None or abs(p["gb"] - r["gguf_gb"]) < 0.001))
        if not same:
            continue
        r["tps_prev"], r["build_prev"] = p["tps"], p.get("build")
        r["tps_change_pct"] = round(100 * (r["avg_tps"] - p["tps"]) / p["tps"], 1)
        if abs(r["tps_change_pct"]) >= 5:
            build = (f"; llama.cpp build {p['build']} -> {r.get('build_info')}"
                     if p.get("build") and r.get("build_info") and p["build"] != r["build_info"] else "")
            lines.append(f"  {r['model']}: {p['tps']:.1f} -> {r['avg_tps']:.1f} tok/s ({r['tps_change_pct']:+.1f}%){build}")
    return lines


def _write_outputs(args, cfg, st, stamp, info, evals_only):
    results = st.results
    if not args.demo and not evals_only:
        changed = speed_changes(results, previous_speeds(cfg), stamp)
        if changed:
            print("\nGeneration speed changed by 5% or more since the previous run (same file, context, offload):")
            print("\n".join(changed))
    ok = [r for r in results if r["status"] == "ok"]
    csv_path = cfg.output_dir / f"benchmark_{stamp}.csv"
    artifacts = {"log": str(info["log_path"]) if info.get("log_path") else None}
    info.update(stamp=stamp, hardware=st.hardware, cpu_source=st.cpu_source, mode=_mode(args, evals_only),
                status="interrupted" if st.interrupted else "complete", artifacts=artifacts)
    if info.get("resumed_from"):
        artifacts["resumed_from"] = info["resumed_from"]
    artifacts["rows"] = str(status.rows_path(cfg.output_dir, stamp)) if st.save else None

    def export_json():
        if cfg.json_enabled:
            info["finished"] = now()
            per_run, latest = json_paths(cfg, stamp)
            artifacts["json"], artifacts["json_latest"] = str(per_run), str(latest)     # listed inside the file itself
            write_json(results, cfg, info, stamp)
            return [f"JSON:      {latest}"]
        return []

    if not args.demo and not ok:
        info["status"] = "failed"
        written = export_json()
        if cfg.csv_enabled:
            write_csv(results, csv_path)
            sys.exit(f"\nNo model completed successfully, so no dashboard was written (errors are in {csv_path})."
                     + "".join(f"\n{w}" for w in written))
        sys.exit("\nNo model completed successfully, so no dashboard was written (CSV output is off; see the log).")
    if not args.demo and not evals_only and ok and all((r["vram_gb"] or 0) < 0.5 for r in ok):
        print("\n[warn] VRAM looks ~0 for every model: GPU telemetry is missing or from the wrong GPU (see GPU_INDEX).")
    out = []
    if cfg.csv_enabled:                      # CSV=false in .env skips both the per-run CSV and the history CSV
        write_csv(results, csv_path)
        if not args.demo:
            write_csv(results, cfg.output_dir / "benchmark_history_v2.csv", append=True)
        out.append(f"CSV:       {csv_path}")
        artifacts["csv"] = str(csv_path)
    html_stamped, html_latest = write_dashboards(cfg, results, stamp, st.cpu_source, st.hardware,
                                                 update_latest=not args.demo)
    out += [f"Dashboard: {html_stamped}", f"Latest:    {html_latest}"]
    artifacts["dashboard"] = str(html_stamped)
    if st.details_path:
        out.append(f"Eval log:  {st.details_path}")
        artifacts["eval_details"] = str(st.details_path)
    if cfg.write_markdown:
        md_path = cfg.output_dir / "benchmark_report.md"
        render_markdown(results, cfg, md_path)
        out.append(f"Markdown:  {md_path}")
        artifacts["markdown"] = str(md_path)
    out += export_json()                     # last, so its "artifacts" section lists everything above
    print("\n" + "\n".join(out))


def load_prior(cfg, which):
    """--resume: (rows document, its path) of the run to continue, or exit with what to do."""
    path = status.find_rows(cfg.output_dir, which or None)
    if not path:
        sys.exit(f"[error] --resume: no run to resume in {cfg.output_dir}"
                 + (f" matching {which!r}" if which else "") + ". Runs made by v2.3.1 or later leave a "
                 "run_<stamp>.rows.json there; an older run cannot be resumed (its eval answers are still in the "
                 "result cache, so a normal run only repeats the speed tests).")
    try:
        doc = status.load_rows(path)
    except (OSError, ValueError) as e:
        sys.exit(f"[error] --resume: cannot read {path}: {e}")
    return doc, path


def _run(args, cfg, stamp, info):
    """-> exit code: 0 complete, 2 stopped early by an error (partial results written), 130 stopped by Ctrl+C / Stop."""
    st = RunState()
    if args.resume is not None and not args.demo:
        doc, path = load_prior(cfg, args.resume)
        st.prior = {r.get("model"): r for r in doc["rows"] if isinstance(r, dict) and r.get("model")}
        info.update(resumed_from=doc.get("stamp"), prior_cpu_source=doc.get("cpu_source"))
        print(f"Resuming run {doc.get('stamp')} ({doc.get('state')}, saved {doc.get('saved')}, {len(st.prior)} model "
              f"row(s)) from {path}")
        if doc.get("base_url") and doc["base_url"] != cfg.base_url:
            print(f"[warn] that run used {doc['base_url']}, this one uses {cfg.base_url}: only resume a run on the "
                  "server it was made on")
        if doc.get("host") and doc["host"] != status.hostname():
            print(f"[warn] that run was made on {doc['host']}, this is {status.hostname()}: kept speed results are "
                  "that machine's")
    if not args.demo:
        st.save = lambda state="partial": status.save_rows(
            cfg, stamp, st.results, state, {"hardware": st.hardware, "cpu_source": st.cpu_source,
                                            "resumed_from": info.get("resumed_from")})
    info["results"] = st.results              # shared with main(): the closing banner reads models/tests from it
    evals_only = (args.evals_only or not cfg.run_speed_tests) and not args.demo
    if evals_only:
        cfg.run_eval_tests = True

    if args.demo:
        st.results.extend(demo_results(cfg))
        st.cpu_source, info["demo"] = "demo data", True
        st.hardware = detect_hardware(cfg) or "Demo system (synthetic data)"
    else:
        session, models = detect_backend(cfg)
        status.update(force=True, phase="starting", models=len(models))
        st.hardware = "" if evals_only else detect_hardware(cfg)
        eval_data = _load_eval_data(cfg)
        try:                                     # one line: what --estimate predicts for this run (never fatal)
            from .estimate import estimate
            estimate(cfg, brief=True)
        except Exception:
            pass
        if evals_only:
            # API-client mode: no GPU / RAPL / RAM sampling, nothing local is read.
            st.cpu_source = "not measured (evals-only run)"
            if not _has_evals(cfg, eval_data):
                sys.exit(f"[error] Nothing to do: enable at least one dataset ({_DATASET_ENVS}) or EVAL_LONGCTX in .env.")
            st.results.extend(restored(m, st.prior[m], True) if resumable(st.prior.get(m), need_speed=False) else blank(m)
                              for m in models)
            print(f"Evals-only run: {len(models)} model(s); speed and hardware telemetry are skipped.")
            _eval_phase(cfg, session, st, eval_data, stamp, info)
        else:
            _defer_prefill(cfg, eval_data)
            st.fresh_evals = _has_evals(cfg, eval_data)
            t0 = now()
            st.interrupted = _speed_phase(cfg, session, models, st, info)
            info.setdefault("phases", {})["speed_s"] = round(now() - t0, 1)
            # Evaluations run AFTER the speed tests, once telemetry is stopped so they cannot disturb it.
            if _has_evals(cfg, eval_data) and not st.interrupted and any(r["status"] == "ok" for r in st.results):
                _early_dashboard(cfg, st, stamp)
                _eval_phase(cfg, session, st, eval_data, stamp, info)

    if not st.results:
        sys.exit("No results to write.")
    finalize(st.results, cfg)
    for note in agent_notes(st.results):               # e.g. long-ctx left out of the Agent composite this run
        print(f"[note] {note}")
    status.update(force=True, phase="writing results", model=None, dataset=None, done=None, total=None)
    try:
        _write_outputs(args, cfg, st, stamp, info, evals_only)
    finally:
        if st.save:                                    # kept after a finished run too: --resume can repair it
            st.save("interrupted" if st.interrupted else "complete")
    if info.get("error"):
        return 2
    return 130 if st.interrupted else 0


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #

LOCK_NAME = ".benchmark_run.json"


def notify(cfg, info, stamp, code, seconds, demo=False):
    """NOTIFY_URL (v2.3.1): POST a short JSON summary when a run ends, e.g. to a Home Assistant webhook. Model names,
    scores and counts only: no settings, no tokens, no answers. Never fatal."""
    url = getattr(cfg, "notify_url", None)
    if not url or demo:
        return
    results = info.get("results") or []
    ranked = sorted((r for r in results if r.get("score_agent") is not None), key=lambda r: -r["score_agent"])
    body = {"tool": "llama-swap-llm-benchmark", "run": stamp, "host": status.hostname(),
            "status": {0: "complete", 2: "partial", 130: "interrupted"}.get(code, "failed"), "exit_code": code,
            "duration_s": round(seconds), "error": info.get("error"),
            "models": len(results), "models_ok": sum(1 for r in results if r.get("status") == "ok"),
            "failed": [r["model"] for r in results if r.get("status") == "error"][:20],
            "top": [{"model": r["model"], "score_agent": round(r["score_agent"], 1)} for r in ranked[:3]]}
    try:
        import requests
        requests.post(url, json=body, timeout=10)
    except Exception as e:
        print(f"[warn] NOTIFY_URL: could not send the summary ({e.__class__.__name__})")


def pid_alive(pid):
    """Is process `pid` running? (Never os.kill(pid, 0) on Windows: there signal 0 is CTRL_C_EVENT.)"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True                                                 # alive, owned by another user
    except OSError:
        return False
    return True


REMOTE_LOCK_MAX_AGE = 1800      # a lock written on another machine counts as live while it is younger than this


def lock_info(cfg_or_dir):
    """The live run's lock {pid, started, log, argv, host}, or None (no lock, or a stale one: the process is gone).
    v2.3.1: a lock written on ANOTHER machine (OUTPUT_DIR on a share) cannot be checked by pid. The run refreshes the
    lock file's modification time with every status update, so it is live while that is recent."""
    d = getattr(cfg_or_dir, "output_dir", cfg_or_dir)
    p = Path(d) / LOCK_NAME
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if info.get("host") and info["host"] != status.hostname():
        try:
            fresh = time.time() - p.stat().st_mtime < REMOTE_LOCK_MAX_AGE
        except OSError:
            fresh = False
        return {**info, "remote": True} if fresh else None
    if not pid_alive(pid):
        return None
    cmd = Path(f"/proc/{pid}/cmdline")
    if cmd.exists() and b"benchmark" not in cmd.read_bytes() and b"llmbench" not in cmd.read_bytes():
        return None                                     # the pid was reused by something else
    return info


def run_lock(cfg, log_path, exclusive=False, force=False):
    """v2.3: mark this run in OUTPUT_DIR so the settings page sees it (and can stop it), wherever it was started.
    v2.3.1: with `exclusive` (a run that measures speed) another live run stops this one, unless --force: two runs on
    one GPU make each other's timings meaningless. Other runs are still only warned."""
    other = lock_info(cfg)
    if other and (other.get("remote") or int(other.get("pid", 0)) != os.getpid()):
        where = f" on {other['host']}" if other.get("remote") else ""
        what = f"another benchmark run (pid {other['pid']}{where}, started {other.get('started')}) is still running"
        if exclusive and not force:
            sys.exit(f"[error] {what}: two runs on one GPU disturb each other's measurements, so this one was not "
                     "started. Stop the other run (the settings page, or Ctrl+C where it runs), or add --force to "
                     f"start anyway. If that run is gone, delete {cfg.output_dir / LOCK_NAME}.")
        print(f"[warn] {what}: two runs on one GPU disturb each other's measurements")
    p = cfg.output_dir / LOCK_NAME
    try:
        p.write_text(json.dumps({"pid": os.getpid(), "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "log": str(log_path) if log_path else None, "argv": sys.argv[1:],
                                 "host": status.hostname()}), encoding="utf-8")
    except OSError:
        return None
    return p


def release_lock(p):
    try:
        if p and json.loads(p.read_text(encoding="utf-8")).get("pid") == os.getpid():
            p.unlink()
    except (OSError, ValueError):
        pass


def _parser():
    ap = argparse.ArgumentParser(description="Benchmark every model on a llama-swap server (run on the GPU host).")
    ap.add_argument("--check", action="store_true", help="test telemetry sources and exit")
    ap.add_argument("--probe-strata", action="store_true", help="list what Strata's API exposes and save it")
    ap.add_argument("--demo", action="store_true", help="build a dashboard from synthetic data (no server)")
    ap.add_argument("--evals-only", action="store_true",
                    help="skip the speed/telemetry tests and run only the evaluations (works from any machine "
                         "that can reach the API; same as RUN_SPEED_TESTS=false RUN_EVAL_TESTS=true)")
    ap.add_argument("--speed-only", action="store_true",
                    help="run only the speed/hardware tests and skip the evaluations completely "
                         "(same as RUN_SPEED_TESTS=true RUN_EVAL_TESTS=false)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models the server offers (GET /v1/models) and exit")
    ap.add_argument("--download-datasets", action="store_true",
                    help="download and cache the eval datasets marked true in .env into ./tests, then exit")
    ap.add_argument("--estimate", action="store_true",
                    help="predict how long a run with the current .env would take (from earlier runs' JSON in "
                         "OUTPUT_DIR where available, else typical speeds), then exit; nothing is benchmarked")
    ap.add_argument("--batch-probe", action="store_true",
                    help="measure what answering several eval questions at once gains on this GPU: the same "
                         "questions at BATCH_PROBE_LEVELS requests in flight, per selected model; then exit")
    ap.add_argument("--reload", nargs="?", const="", metavar="JSON",
                    help="rebuild the dashboard from the newest finished run (OUTPUT_DIR/results.json, or the "
                         "given benchmark_<stamp>.json / folder) with the current dashboard code, then exit; "
                         "nothing is benchmarked or rescored (for working on the dashboard)")
    ap.add_argument("--compare", nargs="*", metavar="JSON",
                    help="combine finished runs into one dashboard, e.g. a Strata run with a llama-swap run (no "
                         "files: the newest run of each backend in OUTPUT_DIR); scores are re-computed over all models "
                         "together; nothing is benchmarked")
    ap.add_argument("--force", action="store_true",
                    help="with --compare: combine runs even when their eval settings differ; with a run: start "
                         "although another benchmark run is still going")
    ap.add_argument("--resume", nargs="?", const="", metavar="RUN",
                    help="continue a run that stopped (crash, kill, power cut, Ctrl+C), or repair a finished one: "
                         "models whose speed test finished are kept, the others are benchmarked, and the result "
                         "cache answers every eval question already asked. RUN = a stamp (20261110_103848) or a "
                         "run_<stamp>.rows.json; without it the newest run in OUTPUT_DIR")
    ap.add_argument("--models", metavar="LIST",
                    help="only these models, comma-separated, * and ? as wildcards (INCLUDE_MODELS for this run)")
    ap.add_argument("--exclude", metavar="LIST", help="leave these models out (EXCLUDE_MODELS for this run)")
    ap.add_argument("--profile", metavar="NAME", help="eval profile for this run (EVAL_PROFILE), e.g. quick or funnel")
    ap.add_argument("--output-dir", metavar="DIR", help="write the results there (OUTPUT_DIR for this run)")
    ap.add_argument("--env", metavar="FILE",
                    help="read the settings from FILE instead of .env next to benchmark.py (e.g. a favourite, or the "
                         "settings page's unsaved settings for --estimate)")
    ap.add_argument("--validate", action="store_true",
                    help="check that the settings load (every value the run would stop on), print OK or the error, "
                         "then exit; with --env FILE checks that file")
    ap.add_argument("--ui", action="store_true",
                    help="open the settings page: edit .env, favourites, estimate, reload / compare, open the latest "
                         "dashboard, run / stop (port UI_PORT, default 41010; localhost unless UI_HOST is set)")
    ap.add_argument("--check-env", nargs="?", const="", metavar="ENV",
                    help="check .env (or the given file) against .env.example: unknown or misspelt settings, values "
                         "that won't work (e.g. 'tue'), settings new in this version, and what differs from the "
                         "built-in defaults; nothing is changed or run")
    return ap


def _utility_command(args, cfg):
    """--check / --probe-strata / --list-models / --download-datasets / --estimate / --batch-probe: run it and report
    True if one was requested."""
    from .batchprobe import batch_probe
    from .estimate import estimate
    for flag, fn in ((args.check, check), (args.probe_strata, probe_strata),
                     (args.list_models, list_models), (args.download_datasets, download_datasets),
                     (args.estimate, estimate), (args.batch_probe, batch_probe)):
        if flag:
            fn(cfg)
            return True
    return False


def _open_log(cfg, log_path):
    """Tee stdout/stderr into the log file (LOG=false in .env turns it off). Returns (file handle | None)."""
    if not cfg.log_enabled:
        return None
    try:
        fh = open(log_path, "w", encoding="utf-8")
    except OSError as e:
        print(f"[warn] could not create the log file {log_path}: {e}")
        return None
    sys.stdout, sys.stderr = Tee(sys.stdout, fh), Tee(sys.stderr, fh)
    return fh


def main():
    ap = _parser()
    args = ap.parse_args()
    if args.speed_only and args.evals_only:
        ap.error("--speed-only and --evals-only cannot be used together")

    for flag, name in ((args.models, "INCLUDE_MODELS"), (args.exclude, "EXCLUDE_MODELS"),
                       (args.profile, "EVAL_PROFILE"), (args.output_dir, "OUTPUT_DIR")):
        if flag:                                 # the shell environment wins over .env, so this is all it takes
            os.environ[name] = flag
    if args.env:
        from . import config as _config
        p = Path(args.env).expanduser()
        if not p.is_file():
            ap.error(f"--env: {p} not found")
        _config.ENV_FILE = p.resolve()
    if args.check_env is not None:                 # before load_config, so .env doesn't hide shell overrides
        from .envspec import check_env_cli
        sys.exit(check_env_cli(args.check_env or (args.env or None)))
    if args.ui:                                    # before load_config: the page must never load .env into its own
        from .ui.server import serve               # environment (its commands run with the file being edited)
        sys.exit(serve())
    if hasattr(signal, "SIGBREAK"):              # Windows: the settings page's Stop sends Ctrl+Break; act as Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    cfg = load_config()
    if args.validate:
        print(f"Settings OK ({args.env or '.env'})")
        return
    if args.speed_only:
        cfg.run_eval_tests, cfg.run_speed_tests = False, True
    if args.compare is not None:
        from .compare import compare_runs
        compare_runs(cfg, args.compare, force=args.force)
        return
    if args.reload is not None:
        from .reload import reload_dashboard
        reload_dashboard(cfg, args.reload or None)
        return
    if args.batch_probe:                     # loads models like a run: log file + run lock (the settings page follows it)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.output_dir / f"batch_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        real_out, real_err = sys.stdout, sys.stderr
        fh = _open_log(cfg, log_path)
        lock = run_lock(cfg, log_path if fh else None, exclusive=True, force=args.force)
        try:
            _utility_command(args, cfg)
        except KeyboardInterrupt:
            print("\nBatch probe stopped (Ctrl+C).")
        finally:
            release_lock(lock)
            sys.stdout, sys.stderr = real_out, real_err
            if fh:
                fh.close()
        return
    if _utility_command(args, cfg):
        return
    if args.demo:          # synthetic data goes to its own folder: never results.json, --reload, --estimate or the page
        cfg.output_dir = cfg.output_dir / "demo"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = now()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path, real_out, real_err = cfg.output_dir / f"benchmark_{stamp}.log", sys.stdout, sys.stderr
    log_fh = _open_log(cfg, log_path)
    bar = "=" * 50
    code, info = 0, {"started": t_start, "log_path": log_path if log_fh else None}
    try:
        lock = run_lock(cfg, log_path if log_fh else None, force=args.force,
                        exclusive=not args.demo and cfg.run_speed_tests and not args.evals_only)
    except SystemExit as e:                  # another run holds the lock: say so, leave its files alone
        if log_fh:
            for s in (sys.stdout, sys.stderr):
                if isinstance(s, Tee):
                    s.close_log()
            sys.stdout, sys.stderr = real_out, real_err
            log_fh.close()
            try:
                log_path.unlink()
            except OSError:
                pass
        raise
    if not args.demo:
        status.start(cfg.output_dir, touch=lock, stamp=stamp, log=str(log_path) if log_fh else None,
                     mode=_mode(args, args.evals_only or not cfg.run_speed_tests), phase="starting")
    try:
        print(f"{bar}\nBenchmark started: {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}"
              + (f"\nLog file: {log_path}" if log_fh else "") + f"\n{bar}")
        code = _run(args, cfg, stamp, info) or 0
    except SystemExit as e:                  # keep the exit message in the log too
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            code = 1
        elif e.code:
            code = e.code
    except KeyboardInterrupt:
        print("\nInterrupted.")
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        status.finish({0: "complete", 2: "partial", 130: "interrupted"}.get(code, "failed"))
        release_lock(lock)
        notify(cfg, info, stamp, code, now() - t_start, args.demo)
        print(run_footer(cfg, info, t_start, code == 0))
        if log_fh:
            for s in (sys.stdout, sys.stderr):
                if isinstance(s, Tee):
                    s.close_log()
            sys.stdout, sys.stderr = real_out, real_err
            log_fh.close()
    if code:
        sys.exit(code)
