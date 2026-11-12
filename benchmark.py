#!/usr/bin/env python3
"""
Local LLM Benchmark Suite (run it ON the machine that has the GPU)
==================================================================
Thin launcher: all of the code lives in the ./llmbench package next to this file, so
`python benchmark.py ...` works exactly as before (same flags, same .env, same results/ folder).

    python benchmark.py --check              # verify telemetry sources + code sandbox before a long run
    python benchmark.py                      # run the benchmark (+ evals if RUN_EVAL_TESTS=true)
    python benchmark.py --demo               # build a dashboard from synthetic data
    python benchmark.py --list-models        # print the models the server offers, then exit
    python benchmark.py --download-datasets  # cache the eval datasets enabled in .env into ./tests, then exit
    python benchmark.py --speed-only         # speed + hardware telemetry only
    python benchmark.py --evals-only         # evaluations only (works from any machine that can reach the API)
    python benchmark.py --help               # all CLI flags

Package layout (see README): config.py . client.py . speed.py . scoring.py . telemetry/ (gpu, host, hardware)
. evals/ (datasets, answers, mathcheck, sandbox, grading, transport, runner) . reporting/ (csv, markdown,
dashboard, json) . diagnostics.py . cli.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # make ./llmbench importable from any working directory

from llmbench.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
