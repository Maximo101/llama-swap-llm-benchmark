# Llama-Swap LLM Benchmark Suite

> **Note:** This is a temporary, interim v1.5 release documenting the newly added dataset evaluation benchmark tests. A fully refactored, cleaner version (v2.0) is currently in development.

An automated, real-load benchmark for LLMs served by [llama-swap](https://github.com/mostlygeek/llama-swap).

For every model on your server it measures generation speed (Technical + Logic prompts), prefill speed, load time, Time To First Token (TTFT), prompt-cache efficiency and context size, while tracking GPU power, VRAM, temperature and CPU power. From those it derives energy per token, efficiency and (optionally) electricity cost, then builds a self-contained interactive dashboard. Following the speed tests, it can now run comprehensive dataset-based evaluations to grade reasoning and coding capabilities.

<img width="1591" height="648" alt="d1" src="https://github.com/user-attachments/assets/71a7cf23-fca5-42f5-b89d-afb3708fd2f4" />

## 📦 Features

* **Real-load telemetry:** One persistent `nvidia-smi -lms` stream (GPU power, utilization, temperature, VRAM) plus CPU power from Intel/AMD RAPL, sampled during exactly the window of each request.
* **Idle baseline per model:** Every model is unloaded first, the idle GPU is sampled, then the model is cold-loaded. This gives load time and attributes VRAM and power to the model under test.
* **Prefill and cache tests:** A long, unique prompt measures prompt-processing speed with the cache disabled. A cold-then-warm pair measures how much of a follow-up prompt is reused from the KV cache (a multi-turn chat proxy).
* **RAM and spill detection:** Peak system RAM (% used), the inference server's own RAM, the RAM change versus the idle baseline, and an estimated "RAM spill" for models that don't fit in VRAM. Works for Strata too.
* **Dataset Evaluations (NEW in v1.5):** Runs QA accuracy (GPQA Diamond, MMLU-Pro) and code pass@1 (HumanEval, MBPP, LiveCodeBench) tests. Code generation is graded by executing the output in an isolated Docker or subprocess sandbox.
* **Context tracking:** Records the context size each model was loaded with, because VRAM and prefill speed depend on it.
* **Energy and cost:** Watt-hours, joules per token, tokens/sec per watt, and optional cost per 1M tokens / 1,000 queries at your peak and off-peak electricity rates.
* **Goldilocks & Agentic ranking:** Scores models by parameter count × average TPS to surface the best size/speed trade-off, and includes an Agentic Reasoner score factoring in eval quality and VRAM economy.
* **Strata fallback:** If llama-swap is down, the single model on a Strata server is benchmarked instead.
* **Interactive dashboard:** Filters, sortable and baseline-comparable table, ten charts (including new Accuracy & Pass Rate graphs) and a radar per podium model.
* **Rich outputs:** Per-run CSV, appended history CSV, optional Markdown report, per-question eval JSONL logs, and the HTML dashboard.

<img width="1591" height="712" alt="d77" src="https://github.com/user-attachments/assets/541c5569-7074-4c7c-91ed-28400c5a0b4d" />

## 🛠️ Prerequisites

* **Python 3.8+**
* **An NVIDIA GPU, and the script must run on the machine that hosts it.** GPU telemetry comes from a local `nvidia-smi` stream.
* **llama-swap** running and reachable over HTTP.
* **CPU power (optional):** RAPL energy counters (Intel, and AMD on Linux 5.8+). Reading them usually needs root. A Home Assistant power sensor can be used as a fallback.

## 🔬 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Maximo101/llama-swap-llm-benchmark.git](https://github.com/Maximo101/llama-swap-llm-benchmark.git)
   cd llama-swap-llm-benchmark

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```


*(Note: `datasets` and `huggingface_hub` are now required for evaluation testing).*
3. Configure your environment:
```bash
cp .env.example .env

```


Edit `.env` to match your setup.

## 🚀 Usage

**1. Verify telemetry sources**

```bash
python benchmark.py --check

```

**2. Run the full benchmark (Speed + Evals)**

```bash
python benchmark.py

```

**3. Run specific test suites (NEW in v1.5)**

```bash
# Run only the speed and hardware telemetry tests
python benchmark.py --speed-only

# Skip speed tests and run only the dataset evaluations
python benchmark.py --evals-only

```

**4. Cache datasets locally**
Download the enabled datasets upfront to avoid network delays during the run:

```bash
python benchmark.py --download-datasets

```

**5. Inspect Strata's API (only if you use the Strata fallback)**

```bash
python benchmark.py --probe-strata

```

**6. Build a demo dashboard (no server needed)**

```bash
python benchmark.py --demo

```

**7. Help - Lists all flags**

```bash
python benchmark.py --help

```

## What happens for each model

1. All models are unloaded (`POST /api/models/unload`) and the GPU settles.
2. The idle GPU power and VRAM are sampled (baseline).
3. The model is cold-loaded (load time) and its context size is read.
4. Technical, Logic and Prefill tests run, `RUNS_PER_PROMPT` times each. The median is reported. The prompt cache is disabled for these.
5. The cache test sends a long prompt cold, then again with a follow-up appended.
6. After a short cooldown, the next model starts.
7. **(NEW)** Once all speed tests conclude, the script runs the Evaluation tests one model at a time based on the active datasets and `.env` profile.

## Configuration

All settings live in `.env`. See `.env.example` for the full annotated list. The main options:

```ini
# --- llama-swap ---
LLAMA_SWAP_URL=http://localhost:8080

# --- Benchmark behaviour ---
RUNS_PER_PROMPT=3            # median is reported
TIMEOUT_TECH=60
TIMEOUT_LOGIC=300
TIMEOUT_LOAD=300
MAX_TOKENS_TECH=300
MAX_TOKENS_LOGIC=500
PREFILL_WORDS=1200           # long-prompt prefill test (~1.5k tokens). 0 disables

# --- Evaluations (accuracy / pass-rate tests, run AFTER the speed tests) ---
RUN_EVAL_TESTS=true
EVAL_GPQA=false
EVAL_MMLU_PRO=true
EVAL_HUMANEVAL=true
EVAL_MBPP=true
EVAL_LIVECODEBENCH=false

# Profile dictates how many rows are sampled per dataset (test, quick, basic, standard, full)
EVAL_PROFILE=quick

# Sandbox mode for executing generated code (auto | docker | subprocess)
EVAL_CODE_SANDBOX=auto

# Agentic score weighting (higher = time matters less)
AGENT_SEC_REF=60

# --- Output ---
WRITE_MARKDOWN=true

```

## 📊 Outputs

All results are saved to `OUTPUT_DIR` (default: `results/` next to the script).

* `benchmark_<timestamp>.csv`: Raw telemetry and benchmark data for the current run.
* `benchmark_history_v2.csv`: Every run appended over time.
* `eval_details_<timestamp>.jsonl`: Per-question evaluation logs containing the prompts, model responses, and grading logic.
* `benchmark_dashboard.html`: The interactive, self-contained dashboard.
* `benchmark_report.md`: Markdown report with the top 3, speed overview, full comparison, and quantization analysis.

## The dashboard

* **Header:** Model count, date, your host hardware (CPU, RAM, GPU) and the power sources used.
* **Filters:** A sticky toggle bar for quantization, Dense/MoE, parameter-size bucket, "Hide VRAM ≥95%" and "Hide failed".
* **Top 3 Goldilocks podium:** Gold/silver/bronze cards ranking Overall Balance, Chat Driver, and Agentic Reasoner (which factors in eval quality).
* **Charts:** Generation speed, VRAM vs speed, power efficiency, energy per token, TTFT leaderboard, latency vs throughput, prefill vs generation, active vs idle GPU power, speed by quantization, and **Accuracy & Pass rates (NEW)**.
* **Full model comparison:** Columns with no data in a run are hidden automatically. Click a row to set it as a baseline, and every other row shows its +/- % against it.
* **Quantization analysis and methodology:** Per-quant averages, plus the exact definitions used for this run.

## Metric definitions

| Metric | How it is measured |
| --- | --- |
| Tech / Logic TPS | `timings.predicted_per_second` (median of `RUNS_PER_PROMPT`); falls back to completion tokens / wall time |
| TTFT | Approximated by `timings.prompt_ms` (prompt-processing time) |
| Prefill t/s, Prefill tok | `timings.prompt_per_second` on the long prompt, and that prompt's size in tokens |
| Load s | Time for the first request after unloading all models (cold load) |
| Ctx | Context size the model was loaded with, read from llama-server `/props` via llama-swap's `/upstream/<model>/props` |
| Cache hit % / Cache x | Share of the follow-up prompt reused from the KV cache (`cache_n`), and how much faster its prompt processing was than the cold prompt |
| RAM vs idle GB | Increase in system RAM over a baseline sampled with all models unloaded (llama-swap only) |
| System RAM | Absolute system RAM in use (`MemTotal - MemAvailable`), peak during the tests. Other dashboards may use another formula or sample at another moment |
| Server RAM GB | Peak resident memory (RSS) of the inference-server process. Always includes overhead (CUDA context, pinned buffers, mapped pages) |
| CPU offload | Read from the server's launch command: `-ngl` below 99, `--n-cpu-moe`, `--cpu-moe`, `--no-kv-offload`, `-ot ...CPU`. `none` means launched fully on the GPU |
| RAM spill GB | 0 for models launched fully on the GPU. For models with CPU offload: Server RAM minus the typical overhead. If launch flags can't be read (Strata), Server RAM when it is at least `RAM_SPILL_THRESHOLD_GB`. An estimate |
| GPU W / Idle W / Peak W | Mean power during generation, power with no model loaded, and the maximum |
| CPU W | RAPL package power (or the Home Assistant sensor) during generation |
| Energy Wh | (mean GPU W + mean CPU W) x request time, for one Technical plus one Logic request |
| J/token | Energy divided by generated tokens |
| TPS/W | Average TPS divided by (GPU W + CPU W) |
| Cost | Measured GPU + CPU energy x your electricity rate. Excludes the rest of the system. "Per 1k queries" uses the average of the Technical and Logic requests, so it scales with `MAX_TOKENS_*` |
| Goldilocks | params (B) x average TPS, using total parameters parsed from the filename |
| **Agent Score (NEW)** | `quality × (100 - VRAM%) × ln(avg TPS)`. Quality uses average eval score modified by seconds per question (s/q). |

## Troubleshooting

* **GPU telemetry unavailable / VRAM about 0:** The script must run on the GPU host with `nvidia-smi` on the PATH. Inside a container it needs the NVIDIA runtime. Run `python benchmark.py --check`.
* **CPU power missing:** RAPL counters are often root-only, and are not visible inside most containers. Run as root, point `RAPL_PATH` at the counters, or configure the Home Assistant fallback.
* **Load time and VRAM delta are blank:** The unload call (`/api/models/unload`) failed. Check your llama-swap version.
* **Ctx column is blank:** The context size could not be read from `/upstream/<model>/props` or `/running`. The benchmark itself is unaffected.
* **Server RAM / RAM spill is blank:** No process matched `RAM_PROCESS_PATTERN`. Run `python benchmark.py --check`, which lists the largest processes by RAM, and set the pattern to your server's executable name. The script must run on the host (as root) to see processes inside Docker.
* **A model shows RAM spill but should be fully in VRAM:** Check its `CPU offload` column. Spill is only reported when the launch flags offload to the CPU. If a model has no offload flags but high Server RAM, it is probably runtime overhead or leftover mapped file pages, and it is not counted as spill. If you rely on llama.cpp's automatic fitting instead of explicit flags, the script cannot see that; set explicit `-ngl` / `--n-cpu-moe` values.
* **Parameter count or quantization shows wrong or unknown:** Both are parsed from the model filename. Models without a `NNB` tag can't be ranked by Goldilocks.
* **Results in the wrong folder:** Remove `OUTPUT_DIR` from `.env`, or set an absolute path. Relative values are resolved against the script's folder.
* **Results are not comparable between models:** Context size and KV-cache type can differ per model in your llama-swap config and affect speed and VRAM. Compare models with a similar Ctx.
* **Docker/Sandbox issues during evals:** If evaluation coding tests fail due to docker restrictions, ensure Docker is properly installed or adjust `EVAL_CODE_SANDBOX` in `.env`.

## Dashboard Example
