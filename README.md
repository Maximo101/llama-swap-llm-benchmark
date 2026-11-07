# Llama-Swap LLM Benchmark Suite

An automated, real-load benchmark for LLMs served by [llama-swap](https://github.com/mostlygeek/llama-swap).

For every model on your server it measures generation speed (Technical + Logic prompts), prefill speed, load time, Time To First Token (TTFT), prompt-cache efficiency and context size, while tracking GPU power, VRAM, temperature and CPU power. From those it derives energy per token, efficiency and (optionally) electricity cost, then builds a self-contained interactive dashboard. Following the speed tests, it runs dataset-based evaluations that grade knowledge, math and coding ability, optionally with several questions in flight at once.

<img width="1337" height="864" alt="d1" src="https://github.com/user-attachments/assets/7cd334dc-d9c7-41fd-84f9-82c06125027c" />

<img width="1273" height="828" alt="s1" src="https://github.com/user-attachments/assets/6e4bc882-24b3-4f53-ac0e-9e369e3f7635" />

<img width="1337" height="856" alt="m1" src="https://github.com/user-attachments/assets/17ea4e3a-bdf8-4830-9ff1-1e79c8da15eb" />

## 📦 Features

* **Real-load telemetry:** GPU power, utilization, temperature and VRAM read in-process through NVML (`pynvml`), plus CPU power from Intel/AMD RAPL, sampled during exactly the window of each request. If the NVML bindings are not installed, it falls back to a persistent `nvidia-smi -lms` stream.
* **Idle baseline per model:** Every model is unloaded first, the idle GPU is sampled, then the model is cold-loaded. This gives load time and attributes VRAM and power to the model under test.
* **Prefill and cache tests:** A long, unique prompt measures prompt-processing speed with the cache disabled. A cold-then-warm pair measures how much of a follow-up prompt is reused from the KV cache (a multi-turn chat proxy).
* **RAM and spill detection:** Peak system RAM (% used), the inference server's own RAM, the RAM change versus the idle baseline, and an estimated "RAM spill" for models that don't fit in VRAM. Works for Strata too.
* **Dataset evaluations:** Knowledge QA accuracy (GPQA Diamond, MMLU-Pro), **math accuracy (GSM8K, MATH-500, NEW in v2.0)** and code pass@1 (HumanEval, MBPP, LiveCodeBench). Generated code is graded by executing it in an isolated Docker or subprocess sandbox.
* **Streamed evaluations with loop abort, thinking cap and partial-answer salvage (NEW in v2.1):** Eval replies are read as a stream, so a model that falls into a repetition loop (`Fail: Degradation`) or thinks past its budget (`Fail: Think cap`) is stopped within seconds instead of after a 10-minute timeout. Whatever it had written is kept, a reply whose visible answer is already complete is graded as it is, and anything else gets one retry with thinking disabled. See [Streaming, loop abort and the thinking cap](#streaming-loop-abort-and-the-thinking-cap).
* **Concurrent evaluations (NEW in v2.0):** `EVAL_CONCURRENCY` keeps several questions in flight at once (asyncio + aiohttp, with a thread-pool fallback) for servers that batch requests. The default of `1` is fully sequential.
* **Hardened code sandbox (NEW in v2.0):** Docker runs as a non-root user with no network, a read-only filesystem, dropped capabilities and memory/CPU/pid limits. A self-test runs at start-up and in `--check`, programs that exit early do not count as passes, and leftover containers are cleaned up automatically.
* **Context tracking:** Records the context size each model was loaded with, because VRAM and prefill speed depend on it.
* **Energy and cost:** Watt-hours, joules per token, tokens/sec per watt, and optional cost per 1M tokens / 1,000 queries at your peak and off-peak electricity rates. Optionally uses the GPU's hardware energy counter for exact per-request energy.
* **Click-to-highlight:** click a model name anywhere (table row, winner cards, quantization table, chart labels, bars or dots) and it turns red everywhere and becomes the table baseline. Click it again, or press Reset / Clear, to remove it.
* **Goldilocks & Agentic ranking:** Scores models by parameter count × average TPS to surface the best size/speed trade-off, and includes an Agentic Reasoner score factoring in eval quality and VRAM economy.
* **Strata fallback:** If llama-swap is down, the single model on a Strata server is benchmarked instead.
* **Interactive dashboard:** Filters, sortable and baseline-comparable table, charts (including Accuracy & Pass Rate graphs for knowledge, math and coding) and a radar per podium model.
* **Rich outputs:** Per-run CSV, appended history CSV, **structured JSON (`results.json`, NEW in v2.0)**, optional Markdown report, per-question eval JSONL logs, a run log, and the HTML dashboard.

<img width="1410" height="769" alt="ss2" src="https://github.com/user-attachments/assets/f3790d98-cba0-4f8e-a990-e2dd9ed06c4d" />

<img width="1410" height="769" alt="ss1" src="https://github.com/user-attachments/assets/d8247de4-6966-47d8-8aaf-fea03f130cee" />

<img width="1735" height="444" alt="dd5" src="https://github.com/user-attachments/assets/a0156de2-2569-408f-81aa-56ae17912a34" />

## 🛠️ Prerequisites

* **Python 3.9+**
* **An NVIDIA GPU, and the script must run on the machine that hosts it.** GPU telemetry comes from the local NVIDIA driver (NVML, or `nvidia-smi` as a fallback). Evaluation-only runs (`--evals-only`) work from any machine that can reach the API.
* **llama-swap** running and reachable over HTTP.
* **CPU power (optional):** RAPL energy counters (Intel, and AMD on Linux 5.8+). Reading them usually needs root. A Home Assistant power sensor can be used as a fallback.
* **Docker (optional, recommended for evals):** Used to sandbox model-written code. Without it, code runs in a restricted subprocess, which is not a security boundary.
* **llama-server `--parallel N` (only if you use `EVAL_CONCURRENCY` > 1):** The server must be able to serve requests in parallel for concurrency to speed anything up.

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
   *(Note: `datasets` is required for evaluation testing and brings in `huggingface_hub` automatically. `nvidia-ml-py` (NVML) and `aiohttp` are optional at runtime: without them the tool falls back to `nvidia-smi` and a thread pool.)*

3. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to match your setup.

### Project layout

`python benchmark.py` is a small launcher; the code lives in the `llmbench/` package next to it. The `.env` file, `results/` folder and `tests/` dataset cache are all resolved relative to `benchmark.py`, never the current directory.

```text
benchmark.py            launcher (all CLI flags)
llmbench/
  config.py             .env loading, Config, dataset specs
  client.py             llama-swap / Strata HTTP, backend detection, model selection
  speed.py              speed, prefill and cache tests for one model
  scoring.py            CSV schema, derived metrics, Goldilocks / Agent scores
  telemetry/            gpu.py (NVML + nvidia-smi), host.py (RAPL, RAM), hardware.py
  evals/                datasets, answers, mathcheck, sandbox, grading, transport, streaming, runner
  reporting/            csv_out, markdown, json_out, dashboard (+ dashboard.html template)
  diagnostics.py        --demo and --check
  cli.py                run orchestration
selftests/              pytest suite with a mock llama-swap server (no GPU needed)
```

## 🚀 Usage

**1. Verify telemetry sources**

```bash
python benchmark.py --check
```

Tests llama-swap, both GPU backends (NVML and `nvidia-smi`), the async client, RAPL, RAM, the inference process and, if a code dataset is enabled, the code sandbox self-test.

**2. Run the full benchmark (Speed + Evals)**

```bash
python benchmark.py
```

**3. Run specific test suites**

```bash
# Run only the speed and hardware telemetry tests
python benchmark.py --speed-only

# Skip speed tests and run only the dataset evaluations
python benchmark.py --evals-only
```

**4. Choose which models run**

```bash
# Print what the server offers; [x] marks the models that will run
python benchmark.py --list-models
```

Narrow the list with `INCLUDE_MODELS`, `EXCLUDE_MODELS` and `MODEL_FILTER` in `.env`.

**5. Cache datasets locally**
Download the enabled datasets upfront to avoid network delays during the run:

```bash
python benchmark.py --download-datasets
```

**6. Inspect Strata's API (only if you use the Strata fallback)**

```bash
python benchmark.py --probe-strata
```

**7. Build a demo dashboard (no server needed)**

```bash
python benchmark.py --demo
```

**8. Help - Lists all flags**

```bash
python benchmark.py --help
```

**9. Run the self-tests (optional, no GPU or server needed)**

```bash
pip install pytest
python -m pytest selftests/
```

## What happens for each model

1. All models are unloaded (`POST /api/models/unload`) and the GPU settles.
2. The idle GPU power and VRAM are sampled (baseline).
3. The model is cold-loaded (load time) and its context size is read.
4. Technical, Logic and Prefill tests run, `RUNS_PER_PROMPT` times each. The median is reported. The prompt cache is disabled for these.
5. The cache test sends a long prompt cold, then again with a follow-up appended.
6. After a short cooldown, the next model starts.
7. Once all speed tests conclude, telemetry is stopped and the script runs the evaluation tests one model at a time, for the active datasets and the `.env` profile. Within a dataset, up to `EVAL_CONCURRENCY` questions run at once. Replies are streamed. If a reply is stopped (repetition loop, thinking cap or timeout) and its visible answer is already complete, it is graded as it is; otherwise, and also when a reply is cut off by the token limit, the question is asked once more with thinking disabled.

## Configuration

All settings live in `.env`. See `.env.example` for the full annotated list. The main options:

```ini
# --- llama-swap ---
LLAMA_SWAP_URL=http://localhost:8080

# --- GPU telemetry ---
GPU_BACKEND=auto             # auto (NVML, else nvidia-smi) | nvml | smi
GPU_POLL_MS=200
GPU_ENERGY_COUNTER=false     # true = exact per-request GPU energy from NVML (Volta or newer)

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
EVAL_GSM8K=false             # NEW: grade-school math word problems
EVAL_MATH=false              # NEW: MATH-500 competition problems

# Profile dictates how many rows are sampled per dataset
# (test, quick, basic, moderate, standard, thorough, deep, ultra, full)
EVAL_PROFILE=quick

# Questions in flight at once (1 = sequential). Needs llama-server --parallel N, N >= this value
EVAL_CONCURRENCY=1
# EVAL_HTTP_BACKEND=auto     # auto | aiohttp | threads
# EVAL_SANDBOX_WORKERS=0     # parallel code executions (0 = auto)

# NEW in v2.1: streamed evals (speed tests are never streamed)
# EVAL_STREAM=true           # false = old behaviour: no loop abort, no salvage, no live thinking budget
# EVAL_THINK_BUDGET=16384    # max thinking tokens per question (0 = unlimited) -> "Fail: Think cap"
# EVAL_LOOP_ABORT=true       # stop repetition loops -> "Fail: Degradation"
# EVAL_SALVAGE=true          # grade the complete visible answer of a stopped reply
# EVAL_TEMPERATURE=0         # reasoning models often loop at 0; see Troubleshooting
# EVAL_TIMEOUT=600           # safety net per request

# Sandbox mode for executing generated code (auto | docker | subprocess)
EVAL_CODE_SANDBOX=auto

# Agentic score (quality first, speed second)
AGENT_SEC_REF=60        # s/q at which the time factor halves (higher = time matters less)
# AGENT_ACC_TOL=          # points below the best eval score still counted as a tie (empty = 1 standard error)
# AGENT_ACC_SPAN=40       # points under the tie zone at which the accuracy credit reaches 0
# AGENT_SPEED_WEIGHT=0.4  # most of the score speed can remove (0 = accuracy only)
# AGENT_TPS_REF=200       # tokens/s that earns the full token-speed credit

# --- Output ---
WRITE_MARKDOWN=true
JSON=true                    # NEW: results.json + benchmark_<timestamp>.json
```

## 📊 Outputs

All results are saved to `OUTPUT_DIR` (default: `results/` next to the script).

* `benchmark_<timestamp>.csv`: Raw telemetry and benchmark data for the current run.
* `benchmark_history_v2.csv`: Every run appended over time. If new columns are added (for example when you enable GSM8K or MATH), older rows are migrated automatically and a `.bak` copy is kept.
* `results.json` and `benchmark_<timestamp>.json`: Structured export for dashboards, databases and notebooks. `results.json` is always the newest run and is written atomically, so it is safe to poll. See [JSON export](#json-export).
* `eval_details_<timestamp>.jsonl`: Per-question evaluation logs containing the model responses and grading logic. Since v2.1 each line also has `flag` (`Fail: Degradation` / `Fail: Think cap` / empty), `salvaged`, `tokens`, and `attempts`: one entry per request sent, with how it ended (`stop`, `length`, `loop`, `budget`, `timeout`, `error`), its tokens and seconds, and for stopped attempts the reason and the last ~600 characters the model wrote.
* `benchmark_dashboard_<timestamp>.html`: The interactive, self-contained dashboard for this run. The newest one is also copied next to the script as `benchmark_dashboard.html`.
* `benchmark_report.md`: Markdown report with the top 3, speed overview, full comparison, quantization analysis and eval results.
* `benchmark_<timestamp>.log`: Everything printed during the run, with start/end time and time taken.

### JSON export

`results.json` has these top-level sections (`schema_version` 1):

| Section | Contents |
| --- | --- |
| `run` | Run id, start/end time, duration, status (`complete`, `interrupted`, `failed`), mode, backend, model counts |
| `host` | Hardware, platform, Python version, GPU backend used, CPU power source |
| `config` | The settings used for the run (API keys and tokens are never written) |
| `evaluation` | Profile, seed, temperature, concurrency, HTTP backend, code sandbox, streaming settings (`stream`, `think_budget`, `loop_abort`, `salvage`), and per-dataset row counts |
| `artifacts` | Paths of every file written for the run, including the JSON itself |
| `models[]` | One entry per model: `metadata`, `speed`, `gpu`, `cpu`, `memory`, `energy`, `scores`, `evaluation` (per dataset: `pct`, `n`, `err`, `na`, `fallback`, `sec`, `trunc`, `degr`, `cap`, `salv`, `tok`, `wall`, `concurrency`), and `other` for any column not in a group |

Missing measurements are `null`, never `NaN`. A JSON file is also written when every model fails, so errors are machine-readable.

## The dashboard

* **Header:** Model count, date, your host hardware (CPU, RAM, GPU) and the power sources used.
* **Filters:** A sticky toggle bar for quantization, Dense/MoE, parameter-size bucket, "Hide VRAM ≥95%" and "Hide failed".
* **Top 3 Goldilocks podium:** Gold/silver/bronze cards ranking Overall Balance, Chat Driver, and Agentic Reasoner (which factors in eval quality).
* **Charts:** Generation speed, VRAM vs speed, power efficiency, energy per token, TTFT leaderboard, latency vs throughput, prefill vs generation, active vs idle GPU power, speed by quantization, and **Accuracy & Pass rates** (knowledge, math and coding shown as separate groups).
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
| Energy Wh | (mean GPU W + mean CPU W) x request time, for one Technical plus one Logic request. With `GPU_ENERGY_COUNTER=true` the GPU part is the hardware energy counter delta instead of mean power x time |
| J/token | Energy divided by generated tokens |
| TPS/W | Average TPS divided by (GPU W + CPU W) |
| Cost | Measured GPU + CPU energy x your electricity rate. Excludes the rest of the system. "Per 1k queries" uses the average of the Technical and Logic requests, so it scales with `MAX_TOKENS_*` |
| Goldilocks | params (B) x average TPS, using total parameters parsed from the filename |
| QA accuracy (GPQA, MMLU-Pro) | The model ends with `Answer: X`; the last such letter (or `\boxed{}`, or a final line that is just a letter) is compared with the dataset answer |
| **Math accuracy (GSM8K, MATH-500) (NEW)** | The model ends with `Answer: <answer>`. The last `Answer:` line is used, else the last `\boxed{}`, else (GSM8K only, and never for a cut-off reply) the last number. Answers are normalised (units, `\dfrac`, thousands separators, `x=` prefixes) and match when the strings are equal or both are numerically equal (fractions, decimals) to 1e-6. Answers that are only equivalent after algebra count as wrong, so scores are a slight lower bound |
| Code pass@1 (HumanEval, MBPP, LiveCodeBench) | The last Python block of the reply is executed in the sandbox. HumanEval uses the task's `check()`, MBPP its `test_list` asserts, LiveCodeBench public tests plus up to `EVAL_LCB_MAX_TESTS` more (stdin/stdout or starter-code function). A program that exits before its tests finish fails |
| N/A | A reply that was cut off by the token limit and has no usable answer, even after the No Thinking retry. Counted as wrong, tracked separately |
| **Fail: Degradation (NEW in v2.1)** | The model's reply fell into a repetition loop (the same block of text over and over, or the same few sentences recycled) and was stopped. Counted per question in `degr` (CSV `eval_<dataset>_degr`), even when the retry then answered correctly; a question that still has no answer afterwards is wrong. Detection is conservative (exact repetition only), so a numbered enumeration is never mistaken for a loop |
| **Fail: Think cap (NEW in v2.1)** | The reply used more than `EVAL_THINK_BUDGET` tokens of thinking and was stopped. Counted in `cap` (JSON). Thinking is recognised when the server splits it out (`reasoning_content`) or when it sits between `<think>` tags |
| **Salvaged (NEW in v2.1)** | A stopped reply whose visible answer already contained a final answer (QA, math) or a complete code block, so it was graded as it is. Counted in `salv`. Thinking text is never used for salvage |
| **tok (NEW in v2.1)** | Mean tokens generated per question over all attempts (exact when the server reports usage, otherwise estimated). In the JSON only |
| **s/q** | Mean latency of one question: request, retries, timeouts and grading. It is deliberately not dataset wall time divided by questions, so concurrent runs don't make every model look faster. With `EVAL_CONCURRENCY` > 1 it is latency under load. Real elapsed time is stored as `eval_<dataset>_wall` |
| **Agent Score** | 0-100, quality first. **Accuracy credit:** eval average vs the best of the run; within the noise margin of the best (`AGENT_ACC_TOL`, default 1 standard error) = 100, then a quadratic fall to 0 at `AGENT_ACC_SPAN` points below. **Speed factor** (0-1): 50% `1/(1+s/q ÷ AGENT_SEC_REF)`, 50% `ln(TPS)/ln(AGENT_TPS_REF)`. `score = credit × (1 - W + W × speed)` with `W = AGENT_SPEED_WEIGHT` (0.4). Speed ranks models of equal accuracy but never lifts a clearly weaker one. s/q counts every retry and aborted attempt. VRAM is not part of it. Without evals: `params_B × (100 - VRAM%) × ln(TPS)` |
| **VRAM fit %** | 100% when VRAM use is inside `VRAM_SWEET_LOW`-`VRAM_SWEET_HIGH` (default 80-90%); falls gently below it, to 50% at `VRAM_SWEET_MAX` (94%) and 0 at 100% |
| **Sweet spot** | `Agent score × VRAM fit`: the most capable model that also uses the GPU sensibly (some headroom, nothing wasted). Ranked in the podium's "VRAM sweet spot" row |
| **VRAM tuning hint** | Under the band with CPU offload: room to lower `--n-cpu-moe`. Over the band: raise `--n-cpu-moe` or cut ctx / KV cache. Tip: add several llama-swap entries of one model with different `--n-cpu-moe` values and the Sweet spot ranks them |

## Streaming, loop abort and the thinking cap

Reasoning models can fall into endless loops, and a non-streaming request returns nothing until the model has finished, so a looping reply used to run into the 10-minute timeout and the benchmark learned nothing about it. Since v2.1 the evaluation phase reads replies as a stream (the speed tests do not), which allows:

1. **Loop abort:** a reply that repeats itself is stopped within seconds and flagged `Fail: Degradation`. A line is printed as it happens, for example `[Fail: Degradation] gpqa-10: repetition loop (a 59-character block repeated 67+ times) after ~1,058 tokens (7s) -> retried with thinking disabled`.
2. **Thinking cap:** `EVAL_THINK_BUDGET` (default 16384 tokens) limits the thinking per question. The server is also told `max_tokens <= budget + 4096`.
3. **Partial-answer salvage:** a stopped reply whose *visible* answer is already complete is graded as it is. Otherwise the question is retried once with thinking disabled (`EVAL_RETRY_ON_LENGTH`), and if that fails too it counts as wrong and keeps its flag.
4. **Evidence:** the tail of every stopped reply is kept in `eval_details_*.jsonl` (`attempts[].tail`), and the per-dataset summary shows how many questions were affected and how many were recovered.
5. **No runaway generation:** closing the connection makes llama-server stop generating. (A non-streaming request that times out is not reliably cancelled, so the abandoned generation could keep the GPU busy and slow the next question down.)

The default budget is a judgement call: it keeps a run's duration predictable, but it also means that a model which needs more than about 16k tokens of thinking for a question is scored as if it had not answered that way. Every cap hit is counted, so you can see how many questions were affected and raise the budget if it matters to you (`EVAL_THINK_BUDGET=32768`, or `0` for unlimited). Set `EVAL_STREAM=false` to get the v2.0 behaviour back.

## Troubleshooting

* **GPU telemetry unavailable / VRAM about 0:** The script must run on the GPU host with the NVIDIA driver loaded. Install `nvidia-ml-py` for NVML, or have `nvidia-smi` on the PATH for the fallback. Inside a container it needs the NVIDIA runtime. Run `python benchmark.py --check`, which tries both backends and shows samples from each.
* **CPU power missing:** RAPL counters are often root-only, and are not visible inside most containers. Run as root, point `RAPL_PATH` at the counters, or configure the Home Assistant fallback.
* **Load time and VRAM delta are blank:** The unload call (`/api/models/unload`) failed. Check your llama-swap version.
* **Ctx column is blank:** The context size could not be read from `/upstream/<model>/props` or `/running`. The benchmark itself is unaffected.
* **Server RAM / RAM spill is blank:** No process matched `RAM_PROCESS_PATTERN`. Run `python benchmark.py --check`, which lists the largest processes by RAM, and set the pattern to your server's executable name. The script must run on the host (as root) to see processes inside Docker.
* **A model shows RAM spill but should be fully in VRAM:** Check its `CPU offload` column. Spill is only reported when the launch flags offload to the CPU. If a model has no offload flags but high Server RAM, it is probably runtime overhead or leftover mapped file pages, and it is not counted as spill. If you rely on llama.cpp's automatic fitting instead of explicit flags, the script cannot see that; set explicit `-ngl` / `--n-cpu-moe` values.
* **Parameter count or quantization shows wrong or unknown:** Both are parsed from the model filename. Models without a `NNB` tag can't be ranked by Goldilocks.
* **Results in the wrong folder:** Remove `OUTPUT_DIR` from `.env`, or set an absolute path. Relative values are resolved against the script's folder.
* **Results are not comparable between models:** Context size and KV-cache type can differ per model in your llama-swap config and affect speed and VRAM. Compare models with a similar Ctx.
* **Docker/Sandbox issues during evals:** The sandbox self-test runs at the start of the eval phase and in `--check`. If Docker fails it, `EVAL_CODE_SANDBOX=auto` falls back to a subprocess with a warning, while `docker` stops with the reason. Make sure the Docker daemon is running and your user can use it, or set `EVAL_CODE_SANDBOX=subprocess` (only for models you trust). Containers left behind by an interrupted run are removed automatically on the next start.
* **Evals are no faster with `EVAL_CONCURRENCY` above 1:** The server is serving requests one at a time. Start llama-server with `--parallel N` (N at least `EVAL_CONCURRENCY`) and make sure llama-swap routes concurrent requests to the same model. Note that llama-server splits the context between its slots, so each question gets a smaller token budget; if many replies are cut off, raise `-c`.
* **Many `Fail: Degradation` lines, or questions taking 10 minutes:** Reasoning models often loop under greedy decoding (`EVAL_TEMPERATURE=0`). Use the sampling settings the model's authors recommend (often 0.6 to 1.0), or switch thinking off for that model with `EVAL_EXTRA_BODY`. Heavily quantized models (IQ2/IQ3) loop more. The run prints a warning when more than 25% of a dataset's questions loop.
* **Many `Fail: Think cap` lines:** The model genuinely thinks for a long time. Raise `EVAL_THINK_BUDGET` (the run then takes longer), or accept the cap as part of the score.
* **After a timeout the next request is unusually slow (about half speed):** In the llama-swap log, requests that follow a 499 status (client closed) can decode much more slowly. With non-streaming requests the abandoned generation may keep running on the server. Streaming (the default since v2.1) closes the connection explicitly; to check on your build, run `curl http://<server>/slots` right after an abort and confirm no slot is still processing.
* **Eval scores aren't comparable with older runs:** Keep `EVAL_CONCURRENCY=1` for runs you want to compare with earlier results, because s/q (and therefore the Agent score) is measured under load when it is higher. The thinking cap (v2.1) can also lower the score of a model that needs very long reasoning; compare runs with the same `EVAL_THINK_BUDGET`. Enabling GSM8K or MATH also changes the eval average, since it averages every enabled dataset.
* **`ModuleNotFoundError: llmbench`:** Run `benchmark.py` from the repository as downloaded; the `llmbench/` folder must sit next to it.
* **A dataset fails to load:** Run `python benchmark.py --download-datasets` with internet access once; later runs read from the local `tests/` cache. Setting `HF_TOKEN` in `.env` avoids anonymous rate limits.

## Dashboard Example

<img width="1423" height="901" alt="d3" src="https://github.com/user-attachments/assets/ffa2a8b5-be96-4f86-8772-83ea3a3e0e6b" />

<img width="1423" height="829" alt="d4" src="https://github.com/user-attachments/assets/2461467f-5eeb-4b2f-a736-24517382d892" />

<img width="1423" height="829" alt="d5" src="https://github.com/user-attachments/assets/fd744301-9bc5-4d1c-b879-1f114400f15c" />

<img width="1728" height="768" alt="d6" src="https://github.com/user-attachments/assets/370e02b3-2847-419d-8b3e-e7f237bf9d21" />

<img width="1735" height="507" alt="d7" src="https://github.com/user-attachments/assets/3de1fb15-7c94-4553-85ac-da50e3c00784" />

<img width="1337" height="784" alt="m3" src="https://github.com/user-attachments/assets/ef9d120b-a624-42fc-8b8d-d2aeb9d35e13" />

<img width="1337" height="756" alt="m4" src="https://github.com/user-attachments/assets/36ced3d9-6cf8-4996-8041-adbb06a1bb1b" />

<img width="1337" height="772" alt="m5" src="https://github.com/user-attachments/assets/1ca0ddcb-e13d-4c29-983c-f22885f6e1dc" />


<img width="1424" height="783" alt="s2" src="https://github.com/user-attachments/assets/32724f33-a04a-4be4-8267-02352270dd73" />

<img width="1424" height="783" alt="s3" src="https://github.com/user-attachments/assets/ec82207e-3cb0-4751-acee-3bced3560ab6" />

<img width="1424" height="783" alt="s4" src="https://github.com/user-attachments/assets/4ec1c74d-4896-4a3d-bdf2-20d361c522e2" />

<img width="1424" height="783" alt="s5" src="https://github.com/user-attachments/assets/da11daf3-0377-4c84-9365-ab513e39055a" />

<img width="1568" height="516" alt="s6" src="https://github.com/user-attachments/assets/183a5493-a059-4984-ab77-265ea08c67ab" />

<img width="1571" height="749" alt="s7" src="https://github.com/user-attachments/assets/35b75f0b-97f0-4bcb-9703-7b63da477cfa" />




