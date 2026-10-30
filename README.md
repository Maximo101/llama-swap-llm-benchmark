# Llama-Swap LLM Benchmark Suite

An automated, real-load benchmark for LLMs served by [llama-swap](https://github.com/mostlygeek/llama-swap).

For every model on your server it measures generation speed (Technical + Logic prompts), prefill speed, load time, Time To First Token (TTFT), prompt-cache efficiency and context size, while tracking GPU power, VRAM, temperature and CPU power. From those it derives energy per token, efficiency and (optionally) electricity cost, then builds a self-contained interactive dashboard.

![Llama-swap Benchmark Dashboard](image_016cc2.png)

## 📦 Features

* **Real-load telemetry:** One persistent `nvidia-smi -lms` stream (GPU power, utilization, temperature, VRAM) plus CPU power from Intel/AMD RAPL, sampled during exactly the window of each request.
* **Idle baseline per model:** Every model is unloaded first, the idle GPU is sampled, then the model is cold-loaded. This gives load time and attributes VRAM and power to the model under test.
* **Prefill and cache tests:** A long, unique prompt measures prompt-processing speed with the cache disabled. A cold-then-warm pair measures how much of a follow-up prompt is reused from the KV cache (a multi-turn chat proxy).
* **Context tracking:** Records the context size each model was loaded with, because VRAM and prefill speed depend on it.
* **Energy and cost:** Watt-hours, joules per token, tokens/sec per watt, and optional cost per 1M tokens / 1,000 queries at your peak and off-peak electricity rates.
* **Goldilocks ranking:** Scores models by parameter count × average TPS to surface the best size/speed trade-off for your hardware.
* **Interactive dashboard:** Filters, sortable and baseline-comparable table, nine charts and a radar per podium model (see below).
* **Rich outputs:** Per-run CSV, appended history CSV, optional Markdown report (Obsidian-friendly) and the HTML dashboard.

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

**3. Build a demo dashboard (no server needed)**
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

# --- GPU ---
VRAM_LIMIT_GB=16            # fallback if nvidia-smi doesn't report total VRAM
# GPU_INDEX=0               # benchmark one GPU only; empty = aggregate all GPUs
GPU_POLL_MS=200

# --- CPU power ---
RAPL_PATH=/sys/class/powercap
# Fallback if RAPL isn't readable (Home Assistant; polled, so it lags short requests)
# HA_URL=http://homeassistant.local:8123
# HA_TOKEN=                 # long-lived token; keep it in .env only
# HA_CPU_POWER_ENTITY=sensor.my_server_cpu_power
HA_POLL_INTERVAL=1

# --- Benchmark behaviour ---
RUNS_PER_PROMPT=3           # median is reported
TIMEOUT_TECH=60
TIMEOUT_LOGIC=300
TIMEOUT_LOAD=300
MAX_TOKENS_TECH=300
MAX_TOKENS_LOGIC=500
TEMPERATURE=0.2
RETRIES=2
SETTLE_SECONDS=5            # wait after unloading, then sample idle baseline
COOLDOWN_SECONDS=5          # pause between models
PREFILL_WORDS=1200          # long-prompt prefill test (~1.5k tokens); 0 disables
CACHE_TEST=true             # cold vs warm prompt-cache test (needs PREFILL_WORDS > 0)
# MODEL_FILTER=Qwen|gemma   # regex; only benchmark matching model ids

# --- Electricity cost (optional) ---
# ELECTRICITY_RATE_PEAK=0.60      # price per kWh
# ELECTRICITY_RATE_OFFPEAK=0.35   # set both to get a Peak / Off-peak toggle
CURRENCY_SYMBOL=$

# --- Dashboard header: host hardware (auto-detected; set to override) ---
# HW_CPU=Intel Core
# HW_RAM=32 GB DDR4
# HW_GPU=RTX 5060 Ti 16 GB

# --- Ranking ---
GOLDILOCKS_MODE=linear      # linear = params_B x TPS, log = ln(params_B) x TPS
GOLDILOCKS_MAX_VRAM_PCT=100 # models above this % of VRAM are excluded from the podium

# --- Output ---
# OUTPUT_DIR=results        # default: a "results" folder next to benchmark.py.
                            # Relative paths are relative to the script, not the current directory.
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
* **Full model comparison:** Vertical scrolling with a frozen header, and frozen Model / Params / Quant / Load / VRAM columns while the rest scroll horizontally. Click a column to sort low to high, click again for high to low. Click a row to set it as a baseline, and every other row shows its +/- % against it (green is better, red is worse). Click it again to clear. VRAM at 95% or more is highlighted as a warning.
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
* **Parameter count or quantization shows wrong or unknown:** Both are parsed from the model filename. Models without a `NNB` tag can't be ranked by Goldilocks.
* **Results in the wrong folder:** Remove `OUTPUT_DIR` from `.env`, or set an absolute path. Relative values are resolved against the script's folder.
* **Results are not comparable between models:** Context size and KV-cache type can differ per model in your llama-swap config and affect speed and VRAM. Compare models with a similar Ctx.

## Dashboard Example


