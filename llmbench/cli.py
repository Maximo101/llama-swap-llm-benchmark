"""Command line entry point and run orchestration.

A run is a sequence of small phases (each its own function, so each can be read, tested and changed alone):

    detect backend -> load eval data (fail fast) -> speed phase -> [early dashboard] -> eval phase -> outputs
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .client import (ctx_from_details, detect_backend, list_models, probe_strata, strata_details,
                     strata_discover)
from .common import add_time, fmt_hr_m_s, stamp_line
from .config import EVAL_SPECS, load_config
from .diagnostics import check, demo_results
from .evals.datasets import download_datasets, prepare_eval_data
from .evals.runner import run_evals
from .reporting.csv_out import write_csv
from .reporting.dashboard import write_dashboards
from .reporting.json_out import json_paths, write_json
from .reporting.markdown import render_markdown
from .scoring import blank, finalize
from .speed import bench_model
from .telemetry import HaCpuPoller, HostMeter, detect_hardware, open_gpu_monitor

_DATASET_ENVS = ", ".join(s["env"] for s in EVAL_SPECS.values())


# --------------------------------------------------------------------------- #
# Console / log plumbing
# --------------------------------------------------------------------------- #

def run_footer(cfg, info, t_start, finished_ok):
    """Closing banner: when the run ended, how long it took, how many models, which profile and which tests ran."""
    now = datetime.now().astimezone()
    bar = "=" * 50
    lines = [bar,
             f"Benchmark {'complete' if finished_ok else 'stopped'} at: {now:%a %b} {now.day:>2} {now:%H:%M:%S} "
             f"{now.tzname() or ''} {now:%Y}",
             f"Time taken to run benchmark: {fmt_hr_m_s(time.time() - t_start)}"]
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
    lines += [bar, f"Script Finished {now:%b %d, %Y}  {now:%H:%M.%S}"]
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
    if not gpu.ok:
        print("  [warn] no GPU data: run this on the GPU host with the NVIDIA driver loaded "
              "(pip install nvidia-ml-py, or have nvidia-smi on PATH; try --check).")
    return gpu, host, ha, cpu_source


def _speed_phase(cfg, session, models, st, info):
    """Benchmark every model one request at a time. Returns True if the user interrupted."""
    gpu, host, ha, st.cpu_source = _start_telemetry(cfg, session, models)
    state = {"warned": False}
    print("Starting Speed Test")
    info["speed"] = True
    stamp_line("Speed test started")
    interrupted = False
    try:
        for i, m in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {m}")
            r = bench_model(cfg, session, gpu, host, ha, m, state)
            if r["status"] == "skipped":
                print("  SKIPPED: " + r["error"])
            elif r["status"] != "ok":
                print("  ERROR: " + r["error"])
            else:
                print(f"  tech {r['tech_tps']:.1f} / logic {r['logic_tps']:.1f} tok/s"
                      + (f" | GPU {r['gpu_w']:.0f} W, VRAM {r['vram_gb']:.1f} GB" if r["gpu_w"] and r["vram_gb"] else "")
                      + (f" | CPU {r['cpu_w']:.0f} W" if r["cpu_w"] else ""))
            st.results.append(r)
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
    t0 = time.time()
    try:
        st.details_path, info["funnel"] = run_evals(cfg, session, st.results, eval_data, stamp, st.hardware)
    except KeyboardInterrupt:
        st.interrupted = True
        print("\nInterrupted: writing partial results (answers so far are in the result cache; rerun to resume).")
    finally:
        info.setdefault("phases", {})["eval_s"] = round(time.time() - t0, 1)


def _defer_prefill(cfg, eval_data):
    """Prefill lengths that the long-context test measures anyway (its first, cold prompt of that length): the speed
    phase skips them instead of sending a second, separate 64k / 128k prompt."""
    if _has_evals(cfg, eval_data) and cfg.eval_longctx:
        cfg.defer_prefill = tuple(n for n in cfg.prefill_ctx_tests if n in cfg.longctx_lengths)
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


def _write_outputs(args, cfg, st, stamp, info, evals_only):
    results = st.results
    ok = [r for r in results if r["status"] == "ok"]
    csv_path = cfg.output_dir / f"benchmark_{stamp}.csv"
    artifacts = {"log": str(info["log_path"]) if info.get("log_path") else None}
    info.update(stamp=stamp, hardware=st.hardware, cpu_source=st.cpu_source, mode=_mode(args, evals_only),
                status="interrupted" if st.interrupted else "complete", artifacts=artifacts)

    def export_json():
        if cfg.json_enabled:
            info["finished"] = time.time()
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
    html_stamped, html_latest = write_dashboards(cfg, results, stamp, st.cpu_source, st.hardware)
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


def _run(args, cfg, stamp, info):
    st = RunState()
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
            st.results.extend(blank(m) for m in models)
            print(f"Evals-only run: {len(models)} model(s); speed and hardware telemetry are skipped.")
            _eval_phase(cfg, session, st, eval_data, stamp, info)
        else:
            _defer_prefill(cfg, eval_data)
            t0 = time.time()
            st.interrupted = _speed_phase(cfg, session, models, st, info)
            info.setdefault("phases", {})["speed_s"] = round(time.time() - t0, 1)
            # Evaluations run AFTER the speed tests, once telemetry is stopped so they cannot disturb it.
            if _has_evals(cfg, eval_data) and not st.interrupted and any(r["status"] == "ok" for r in st.results):
                _early_dashboard(cfg, st, stamp)
                _eval_phase(cfg, session, st, eval_data, stamp, info)

    if not st.results:
        sys.exit("No results to write.")
    finalize(st.results, cfg)
    _write_outputs(args, cfg, st, stamp, info, evals_only)


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #

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

    cfg = load_config()
    if args.speed_only:
        cfg.run_eval_tests, cfg.run_speed_tests = False, True
    if _utility_command(args, cfg):
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path, real_out, real_err = cfg.output_dir / f"benchmark_{stamp}.log", sys.stdout, sys.stderr
    log_fh = _open_log(cfg, log_path)
    bar = "=" * 50
    code, info = 0, {"started": t_start, "log_path": log_path if log_fh else None}
    try:
        print(f"{bar}\nBenchmark started: {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}"
              + (f"\nLog file: {log_path}" if log_fh else "") + f"\n{bar}")
        _run(args, cfg, stamp, info)
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
        print(run_footer(cfg, info, t_start, code == 0))
        if log_fh:
            for s in (sys.stdout, sys.stderr):
                if isinstance(s, Tee):
                    s.close_log()
            sys.stdout, sys.stderr = real_out, real_err
            log_fh.close()
    if code:
        sys.exit(code)
