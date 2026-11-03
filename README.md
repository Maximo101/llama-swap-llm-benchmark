# Llama-Swap LLM Benchmark Suite

An automated, real-load benchmark for LLMs served by [llama-swap](https://github.com/mostlygeek/llama-swap).

For every model on your server it measures generation speed (Technical + Logic prompts), prefill speed, load time, Time To First Token (TTFT), prompt-cache efficiency and context size, while tracking GPU power, VRAM, temperature and CPU power. From those it derives energy per token, efficiency and (optionally) electricity cost, then builds a self-contained interactive dashboard.

<img width="1591" height="648" alt="d1" src="https://github.com/user-attachments/assets/71a7cf23-fca5-42f5-b89d-afb3708fd2f4" />

## 📦 Features

* **Real-load telemetry:** One persistent `nvidia-smi -lms` stream (GPU power, utilization, temperature, VRAM) plus CPU power from Intel/AMD RAPL, sampled during exactly the window of each request.
* **Idle baseline per model:** Every model is unloaded first, the idle GPU is sampled, then the model is cold-loaded. This gives load time and attributes VRAM and power to the model under test.
* **Prefill and cache tests:** A long, unique prompt measures prompt-processing speed with the cache disabled. A cold-then-warm pair measures how much of a follow-up prompt is reused from the KV cache (a multi-turn chat proxy).
* **RAM and spill detection:** Peak system RAM (% used), the inference server's own RAM, the RAM change versus the idle baseline, and an estimated "RAM spill" for models that don't fit in VRAM. Works for Strata too.
* **Context tracking:** Records the context size each model was loaded with, because VRAM and prefill speed depend on it.
* **Energy and cost:** Watt-hours, joules per token, tokens/sec per watt, and optional cost per 1M tokens / 1,000 queries at your peak and off-peak electricity rates.
* **Goldilocks ranking:** Scores models by parameter count × average TPS to surface the best size/speed trade-off for your hardware.
* **Strata fallback:** If llama-swap is down, the single model on a Strata server is benchmarked instead. Its quantization and context size are read from Strata's API, the server process is found by the port it listens on, and a collapsible "Strata details" panel shows what its API reports.
* **Interactive dashboard:** Filters, sortable and baseline-comparable table, nine charts and a radar per podium model (see below).
* **Rich outputs:** Per-run CSV, appended history CSV, optional Markdown report (Obsidian-friendly) and the HTML dashboard.

<img width="1591" height="712" alt="d77" src="https://github.com/user-attachments/assets/541c5569-7074-4c7c-91ed-28400c5a0b4d" />

## 🛠️ Prerequisites

* **Python 3.8+**
* **An NVIDIA GPU, and the script must run on the machine that hosts it.** GPU telemetry comes from a local `nvidia-smi` stream.
* **llama-swap** running and reachable over HTTP.
* **CPU power (optional):** RAPL energy counters (Intel, and AMD on Linux 5.8+). Reading them usually needs root. A Home Assistant power sensor can be used as a fallback.

## 🔬 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maximo101/llama-swap-llm-benchmark.git
   cd llama-swap-llm-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   `requests` is required. `python-dotenv` is optional; without it the script uses a built-in `.env` reader.

3. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to match your setup.

## 🚀 Usage

**1. Verify telemetry sources**

Before a long run, check that the script can reach llama-swap, `nvidia-smi` and your CPU power source:
```bash
python benchmark.py --check
```
It also prints the hardware description that will appear in the dashboard header.

**2. Run the benchmark**
```bash
python benchmark.py
```

**3. Inspect Strata's API (only if you use the Strata fallback)**
```bash
python benchmark.py --probe-strata
```
Lists the endpoints Strata's own web UI uses and saves the raw responses to `results/strata_probe.json`. Only GET requests are made, and anything that could change state (unload, chat, history) is skipped. Review the file before sharing it.

**4. Build a demo dashboard (no server needed)**
```bash
python benchmark.py --demo
```

The script resolves `.env` and `results/` relative to **its own location, not the current directory**, so you can run it from anywhere, for example from a cron job or an Unraid User Script:
```bash
python /path/to/llama-swap-llm-benchmark/benchmark.py
```

Press Ctrl+C at any time to stop. Results gathered so far are still written out.

## What happens for each model

1. All models are unloaded (`POST /api/models/unload`) and the GPU settles.
2. The idle GPU power and VRAM are sampled (baseline).
3. The model is cold-loaded (load time) and its context size is read.
4. Technical, Logic and Prefill tests run, `RUNS_PER_PROMPT` times each. The median is reported. The prompt cache is disabled for these.
5. The cache test sends a long prompt cold, then again with a follow-up appended.
6. After a short cooldown, the next model starts.

A model that fails (timeout, HTTP error, crash) is recorded as an error and the run continues.

## Configuration

All settings live in `.env`. See `.env.example` for the full annotated list. The main options:

```ini
# --- llama-swap ---
LLAMA_SWAP_URL=http://localhost:8080
# LLAMA_SWAP_API_KEY=

# --- Optional fallback: a single Strata endpoint, used only if llama-swap is not reachable ---
# STRATA_URL=http://localhost:8080
# STRATA_API_KEY=
# STRATA_MODEL=qwen3.8-Next-Flash # model name sent in requests (default: first id from /v1/models, else "strata")
# STRATA_PARAMS_B=125             # facts about the model, for ranking/radar (it has no filename to parse)
# STRATA_ACTIVE_B=3.8             # active params if MoE, else leave unset
# STRATA_QUANT=IQ3_S              # optional: otherwise parsed from Strata's own model id
# STRATA_CTX=32768                # optional: otherwise read from Strata's API (see --probe-strata)

# --- GPU ---
# Fallback VRAM total (GB) if nvidia-smi doesn't report it
VRAM_LIMIT_GB=16
# Benchmark one GPU only (nvidia-smi index). Empty = aggregate all GPUs.
# GPU_INDEX=0
GPU_POLL_MS=200

# --- Benchmark behaviour ---
RUNS_PER_PROMPT=3            # median is reported
TIMEOUT_TECH=60
TIMEOUT_LOGIC=300
TIMEOUT_LOAD=300
MAX_TOKENS_TECH=300
MAX_TOKENS_LOGIC=500
TEMPERATURE=0.2
RETRIES=2
SETTLE_SECONDS=5             # wait after unloading, then sample idle baseline
COOLDOWN_SECONDS=5           # pause between models
PREFILL_WORDS=1200           # long-prompt prefill test (~1.5k tokens). 0 disables
CACHE_TEST=true              # cold vs warm prompt-cache test (needs PREFILL_WORDS > 0)
# MODEL_FILTER=Qwen|gemma    # regex; only benchmark matching model ids
# TECH_PROMPT=Explain quantum computing in 3 sentences.
# LOGIC_PROMPT=Outline a 101 Physics Handbook. Create a 3-question quiz.

# --- Electricity cost (optional; adds $/1M tokens and $/1k queries columns) ---
# Price per kWh in your currency. Set OFFPEAK too to get a Peak / Off-peak toggle on the dashboard.
# ELECTRICITY_RATE_PEAK=0.60
# ELECTRICITY_RATE_OFFPEAK=0.35
CURRENCY_SYMBOL=$

# --- Output ---
# Leave unset to write to a "results" folder next to benchmark.py (regardless of where you run it from).
# A relative value is also resolved against the script's folder, not the current directory.
# OUTPUT_DIR=results
WRITE_MARKDOWN=true
```

## 📊 Outputs

All results are saved to `OUTPUT_DIR` (default: `results/` next to the script).

* `benchmark_<timestamp>.csv`: Raw telemetry and benchmark data for the current run.
* `benchmark_history_v2.csv`: Every run appended over time.
* `benchmark_dashboard.html`: The interactive, self-contained dashboard. Chart.js is loaded from the cdnjs CDN, so open it with internet access.
* `benchmark_report.md`: Markdown report with the top 3, speed overview, full comparison and quantization analysis.

## The dashboard

* **Header:** Model count, date, your host hardware (CPU, RAM, GPU) and the power sources used.
* **Filters:** A sticky toggle bar for quantization, Dense/MoE, parameter-size bucket, "Hide VRAM ≥95%" and "Hide failed". Filters apply to the podium, every chart and the tables.
* **Top 3 Goldilocks podium:** Gold/silver/bronze cards, each with a radar chart of Speed, Efficiency, Reasoning (parameters) and VRAM economy.
* **Charts:** Generation speed, VRAM vs speed (bubble size = parameters), power efficiency, energy per token, TTFT leaderboard, latency vs throughput, prefill vs generation (log scale), active vs idle GPU power, and speed by quantization. Charts grow so every model name is visible.
* **Full model comparison:** Columns with no data in a run (for example Load s and RAM delta on Strata, or cost without electricity rates) are hidden automatically, and a note lists them.
   Vertical scrolling with a frozen header, and frozen Model / Params / Quant / Load / VRAM columns while the rest scroll horizontally. Click a column to sort low to high, click again for high to low. Click a row to set it as a baseline, and every other row shows its +/- % against it (green is better, red is worse). Click it again to clear. VRAM at 95% or more is highlighted as a warning.
* **Quantization analysis and methodology:** Per-quant averages, plus the exact definitions used for this run.

## Metric definitions

| Metric | How it is measured |
|---|---|
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
| RAM spill GB | 0 for models launched fully on the GPU. For models with CPU offload: Server RAM minus the typical overhead (median Server RAM of this run's fully-GPU models, or `RAM_OVERHEAD_GB`). If launch flags can't be read (Strata), Server RAM when it is at least `RAM_SPILL_THRESHOLD_GB`. An estimate |

| GPU W / Idle W / Peak W | Mean power during generation, power with no model loaded, and the maximum |
| CPU W | RAPL package power (or the Home Assistant sensor) during generation |
| Energy Wh | (mean GPU W + mean CPU W) x request time, for one Technical plus one Logic request |
| J/token | Energy divided by generated tokens |
| TPS/W | Average TPS divided by (GPU W + CPU W) |
| Cost | Measured GPU + CPU energy x your electricity rate. Excludes the rest of the system. "Per 1k queries" uses the average of the Technical and Logic requests, so it scales with `MAX_TOKENS_*` |
| Goldilocks | params (B) x average TPS, using total parameters parsed from the filename |

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

## Dashboard Example

<img width="1591" height="734" alt="d2" src="https://github.com/user-attachments/assets/a4921473-e69d-43fb-b896-31be8a3d7c87" />

<img width="1591" height="688" alt="d3" src="https://github.com/user-attachments/assets/7bea5c96-8a96-4d5d-85b5-5f6b49e4c865" />

<img width="1591" height="682" alt="d4" src="https://github.com/user-attachments/assets/c88f23a5-d446-42e4-be82-ecb9ce6d8ddb" />

<img width="1591" height="830" alt="d5" src="https://github.com/user-attachments/assets/111a3cea-0f20-4ec3-9bec-701bd91b681a" />

<img width="1591" height="774" alt="d6" src="https://github.com/user-attachments/assets/ee818ccd-ae13-43bf-8f95-ba5ad21f67b5" />

<img width="1591" height="712" alt="d78" src="https://github.com/user-attachments/assets/1b8f7eb2-16f9-48e6-889b-1661096c2fc6" />
