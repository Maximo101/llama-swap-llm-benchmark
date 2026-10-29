# Llama-Swap Benchmark Suite

An automated, real-load benchmark suite for LLMs running on a `llama-swap` server.

This tool measures generation speed (Technical + Logic prompts), prefill speed, load times, and Time To First Token (TTFT). It simultaneously tracks GPU power, VRAM utilization, CPU power (via RAPL or Home Assistant), and calculates energy efficiency (Joules per token).

![Llama-swap Benchmark Dashboard](image_016cc2.png)

## Features

* **Comprehensive Telemetry:** Tracks generation speed alongside hardware metrics (GPU power/utilization/temperature/VRAM via `nvidia-smi` and CPU power via Intel/AMD RAPL).
* **Idle Baseline Calculation:** Unloads models before testing to sample an idle baseline, ensuring VRAM and power metrics are accurately attributed to the specific model being tested.
* **Goldilocks Ranking:** Automatically scores and ranks models based on a "Goldilocks" metric (Parameter size × TPS) to help you find the most efficient model for your hardware.
* **Rich Outputs:** Generates a detailed CSV history, an optional Markdown report, and a self-contained, interactive HTML dashboard using Chart.js.

## Prerequisites

* **Python 3.8+**
* **NVIDIA GPU:** This script must be run *on the machine hosting the GPU*, as it relies on a persistent `nvidia-smi -lms` stream for hardware telemetry.
* **llama-swap:** A running instance of `llama-swap` accessible via HTTP.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maximo101/llama-swap-benchmark.git
   cd llama-swap-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to match your setup. **Never commit your `.env` file.**

## Configuration Variables

The benchmark is highly configurable via the `.env` file. Below are the key configuration options:

```ini
# --- llama-swap ---
LLAMA_SWAP_URL=http://localhost:8080
# LLAMA_SWAP_API_KEY=

# --- GPU ---
VRAM_LIMIT_GB=16
# GPU_INDEX=0            # Benchmark one GPU only. Empty = aggregate all GPUs.
GPU_POLL_MS=200

# --- CPU power ---
RAPL_PATH=/sys/class/powercap
# Fallback if RAPL isn't readable (Home Assistant)
# HA_URL=http://homeassistant.local:8123
# HA_TOKEN=              # long-lived token; keep it in .env only
# HA_CPU_POWER_ENTITY=sensor.my_server_cpu_power
HA_POLL_INTERVAL=1

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

# --- Output ---
OUTPUT_DIR=./results
WRITE_MARKDOWN=true
```

## Usage

**1. Verify Telemetry Sources**
Before running a long benchmark, verify that the script can successfully read from `llama-swap`, `nvidia-smi`, and your CPU power source:
```bash
python benchmark.py --check
```

**2. Run the Benchmark**
Execute the standard benchmark across all available models:
```bash
python benchmark.py
```

**3. Generate Demo Dashboard**
If you want to test the dashboard generation without running models, you can build it using synthetic data:
```bash
python benchmark.py --demo
```

## Outputs

All results are saved to the directory specified in `OUTPUT_DIR` (default: `./results`).
* `benchmark_<timestamp>.csv`: Raw telemetry and benchmark data for the current run.
* `benchmark_history_v2.csv`: An appended history of all benchmark runs over time.
* `benchmark_dashboard.html`: An interactive, self-contained Chart.js dashboard analyzing speed, VRAM limits, and power efficiency.
* `benchmark_report.md`: A summarized Markdown report detailing the top 3 models and full dataset comparisons.
