# Llama-Swap LLM Benchmark Suite

An automated, real-load benchmark for LLMs served by [llama-swap](https://github.com/mostlygeek/llama-swap).

For every model on your server it measures generation speed (Technical + Logic prompts), prefill speed at the prompt lengths you actually deploy, agent-prompt time to first token, load time, prompt-cache efficiency and context size, while tracking GPU power, VRAM, temperature and CPU power. It reads what the server really loaded (launch flags and the GGUF header: KV-cache type, bits per weight, experts, KV cost per token). From those it derives energy per token, efficiency and (optionally) electricity cost. After the speed tests it runs dataset evaluations that grade knowledge, math, coding, tool calling and long-context retrieval, then ranks the models for real use on that machine and builds a self-contained interactive dashboard.

A full run of ~28 models is dominated by the GPU generating eval answers. v2.1 adds the tools to make that much shorter without making the ranking less trustworthy: a confidence-gated funnel, a per-question result cache with resume, batched evaluation with a measured probe, skipping models that cannot be deployed anyway, a single shared long-context document, and a per-phase timing report with `--estimate`. See [Making a full run faster](#making-a-full-run-faster).

<img width="1337" height="864" alt="d1" src="https://github.com/user-attachments/assets/7cd334dc-d9c7-41fd-84f9-82c06125027c" />

<img width="1273" height="828" alt="s1" src="https://github.com/user-attachments/assets/6e4bc882-24b3-4f53-ac0e-9e369e3f7635" />

<img width="1337" height="856" alt="m1" src="https://github.com/user-attachments/assets/17ea4e3a-bdf8-4830-9ff1-1e79c8da15eb" />

## 📦 Features

* **Real-load telemetry:** GPU power, utilization, temperature and VRAM read in-process through NVML (`pynvml`), plus CPU power from Intel/AMD RAPL, sampled during exactly the window of each request. If the NVML bindings are not installed, it falls back to a persistent `nvidia-smi -lms` stream.
* **Idle baseline per model:** Every model is unloaded first, the idle GPU is sampled, then the model is cold-loaded. This gives load time and attributes VRAM and power to the model under test.
* **Prefill and cache tests:** A long, unique prompt measures prompt-processing speed with the cache disabled. A cold-then-warm pair measures how much of a follow-up prompt is reused from the KV cache (a multi-turn chat proxy).
* **RAM and spill detection:** Peak system RAM (% used), the inference server's own RAM, the RAM change versus the idle baseline, and an estimated "RAM spill" for models that don't fit in VRAM. Works for Strata too.
* **Dataset evaluations:** Knowledge QA accuracy (GPQA Diamond, MMLU-Pro), math accuracy (GSM8K, MATH-500), code pass@1 (HumanEval, MBPP, LiveCodeBench, optionally limited to problems released after a date) and BFCL-style tool calling. Generated code is graded by executing it in an isolated Docker sandbox.
* **Long-context retrieval:** several needles and decoys hidden in one 32k / 64k / 128k document, at the KV-cache precision the model is deployed with.
* **What the server really loaded:** KV-cache type, flash attention, `-ub` / `-b`, slots and unified KV, CPU offload and mmproj come from the llama-server command line; architecture, bits per weight, experts, attention layout and KV bytes per token from the GGUF header. From these: KV GB at the loaded context and the largest context that would fit at a chosen share of VRAM.
* **Deployed-context speed:** prefill tok/s at 8k-128k prompt tokens and time to first token for an agent-shaped prompt (a tool catalogue plus a long history), because a model that is fast on short prompts can be slow where an agent actually works.
* **Per-machine budgets:** `BUDGET_PROFILE` rules (minimum context, VRAM reserve for other services, minimum KV precision, spill, speed, accuracy, retries) give every model a pass/fail per rule; failing models stay in the table but off the podium, and their evals can be skipped.
* **Streamed evaluations with loop abort, thinking cap and partial-answer salvage:** a reply that falls into a repetition loop (`Fail: Degradation`) or thinks past its budget (`Fail: Think cap`) is stopped within seconds instead of after a 10-minute timeout. A reply whose visible answer is already complete is graded as it is; otherwise the reasoning can be closed and the answer asked for (budget forcing, opt-in), or the question is retried once with thinking disabled. See [Streaming, loop abort and the thinking cap](#streaming-loop-abort-and-the-thinking-cap).
* **Honest margins:** every eval score carries its 95% margin of error; small profiles are flagged as a rough guide, ties within the noise are treated as ties, and a model whose evals are incomplete is not ranked.
* **Faster full runs:** funnel profile with confidence-interval gates, result cache and resume, batched evaluation capped to what the server's slots and KV pool can hold (with `--batch-probe` to measure the gain first), grading pipelined behind generation, one shared long-context document per length, and skipping the evals of models that cannot be deployed. A timing report shows where every minute went and `--estimate` predicts a run before it starts.
* **Hardened code sandbox:** Docker runs as a non-root user with no network, a read-only filesystem, dropped capabilities and memory/CPU/pid limits. A self-test runs at start-up and in `--check`, programs that exit early do not count as passes, and leftover containers are cleaned up automatically. Without Docker the run stops unless you explicitly accept an unisolated subprocess.
* **Context tracking:** Records the context size each model was loaded with, because VRAM and prefill speed depend on it.
* **Energy and cost:** Watt-hours, joules per token, tokens/sec per watt, and optional cost per 1M tokens / 1,000 queries at your peak and off-peak electricity rates. Optionally uses the GPU's hardware energy counter for exact per-request energy.
* **Click-to-highlight:** click a model name anywhere (table row, winner cards, quantization table, chart labels, bars or dots) and it turns red everywhere and becomes the table baseline. Click it again, or press Reset / Clear, to remove it.
* **Rankings for real use:** Goldilocks (parameters × average TPS), Chat (speed × free VRAM), Agent (quality first, speed second, no VRAM) and Sweet spot (Agent × how well the model fits the GPU). See [Metric definitions](#metric-definitions).
* **Strata fallback:** If llama-swap is down, the single model on a Strata server is benchmarked instead.
* **Interactive dashboard:** Filters, sortable and baseline-comparable table, charts (including Accuracy & Pass Rate graphs for knowledge, math and coding) and a radar per podium model. Hover any eval column or dataset name for a short explanation of that test.
* **Rich outputs:** Per-run CSV, appended history CSV, structured JSON (`results.json`), optional Markdown report, per-question eval JSONL logs, a run log, and the HTML dashboard.

<img width="1410" height="769" alt="ss2" src="https://github.com/user-attachments/assets/f3790d98-cba0-4f8e-a990-e2dd9ed06c4d" />

<img width="1410" height="769" alt="ss1" src="https://github.com/user-attachments/assets/d8247de4-6966-47d8-8aaf-fea03f130cee" />

<img width="1735" height="444" alt="dd5" src="https://github.com/user-attachments/assets/a0156de2-2569-408f-81aa-56ae17912a34" />

## 🛠️ Prerequisites

* **Python 3.9+**
* **An NVIDIA GPU, and the script must run on the machine that hosts it.** GPU telemetry comes from the local NVIDIA driver (NVML, or `nvidia-smi` as a fallback). Evaluation-only runs (`--evals-only`) work from any machine that can reach the API.
* **llama-swap** running and reachable over HTTP.
* **CPU power (optional):** RAPL energy counters (Intel, and AMD on Linux 5.8+). Reading them usually needs root. A Home Assistant power sensor can be used as a fallback.
* **Docker (needed for the code datasets):** Used to sandbox model-written code (HumanEval, MBPP, LiveCodeBench). Without Docker the run stops with an explanation; set `EVAL_CODE_SANDBOX=subprocess` to accept running generated code as your user in a restricted subprocess, which is not a security boundary.
* **llama-server slots (only if you use `EVAL_CONCURRENCY` > 1):** `-np N` with N at least `EVAL_CONCURRENCY`, and preferably `--kv-unified` so every request keeps the full context. See [Batched evaluation](#batched-evaluation-eval_concurrency).

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
  modelinfo.py, gguf.py  launch flags, /props and the GGUF header (own reader)
  scoring.py            CSV schema, derived metrics, Goldilocks / Chat / Agent / Sweet-spot scores, budgets
  telemetry/            gpu.py (NVML + nvidia-smi), host.py (RAPL, RAM, launch flags), hardware.py
  evals/                datasets, answers, mathcheck, sandbox, grading, transport, streaming, runner,
                        funnel (CI gates), cache (result cache), longctx, toolcall
  reporting/            csv_out, markdown, json_out, dashboard (+ dashboard.html template), margin
  estimate.py           --estimate
  batchprobe.py         --batch-probe
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

**7. Predict how long a run will take**

```bash
python benchmark.py --estimate
```

Uses earlier runs' JSON in `results/` (measured tok/s, prefill speed, seconds and tokens per question) and any batch-probe result; models without history get a rough guess from their size and quantisation. Models that `EVAL_SKIP_INELIGIBLE` would skip (context below `MIN_CTX`, a failed budget rule) get no eval time. A one-line estimate is also printed at the start of every run.

**8. Measure what batched evaluation gains on your GPU**

```bash
INCLUDE_MODELS=model-a,model-b,model-c python benchmark.py --batch-probe
```

See [Batched evaluation](#batched-evaluation-eval_concurrency).

**9. Resume an interrupted run**

Just run it again with the same `.env`. Every answered question is kept in the result cache (`results/cache/`), so only what is missing is asked; the speed tests are always measured again.

**10. Build a demo dashboard (no server needed)**

```bash
python benchmark.py --demo
```

**11. Help - Lists all flags**

```bash
python benchmark.py --help
```

**12. Run the self-tests (optional, no GPU or server needed)**

```bash
pip install pytest
python -m pytest selftests/
```

## What happens for each model

**Speed phase** (every model, one request at a time, with telemetry):

1. All models are unloaded (`POST /api/models/unload`) and the GPU settles.
2. The idle GPU power and VRAM are sampled (baseline).
3. The model is cold-loaded (load time). Its context size, launch flags and GGUF header are read, and `MIN_CTX` is checked.
4. Technical, Logic and Prefill tests run, `RUNS_PER_PROMPT` times each. The median is reported. The prompt cache is disabled for these.
5. The cache test sends a long prompt cold, then again with a follow-up appended.
6. Prefill at each `PREFILL_CTX_TESTS` length (one cold, unique prompt each) and the agent-prompt TTFT (`AGENT_PROMPT_TOKENS`). A length that the long-context test also uses is taken from that test's first prompt instead of being sent twice.
7. After a short cooldown, the next model starts.

**Eval phase** (after all speed tests, telemetry stopped):

1. Models that already fail a deployment rule (`MIN_CTX`, the budget profile, `GOLDILOCKS_MAX_VRAM_PCT`) are skipped (`EVAL_SKIP_INELIGIBLE`).
2. For each model: a warm-up request loads it (timed separately, so the load is never part of a question's latency), `/props` is read (context, slots, chat template), the long-context test runs on one shared document per length, then the datasets.
3. Within a dataset, up to `EVAL_CONCURRENCY` questions are in flight, capped to what the server's slots and KV pool can hold. Replies are streamed; grading (including sandboxed code) runs while the next question is generating.
4. A stopped reply (repetition loop, thinking cap, timeout) whose visible answer is already complete is graded as it is; otherwise, and also when a reply is cut off by the token limit, the question is asked once more with thinking disabled (or, with `EVAL_BUDGET_FORCING=true`, the reasoning is closed and the answer asked for).
5. Every answer is stored in the result cache. With `EVAL_PROFILE=funnel` the models clearly below the leaders are dropped after each stage and the rest answer the next stage's new questions.

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
EVAL_GSM8K=false             # grade-school math word problems
EVAL_MATH=false              # MATH-500 competition problems
EVAL_TOOLCALL=false          # 32 built-in BFCL-style tool-calling cases
EVAL_LONGCTX=true            # long-context retrieval at LONGCTX_LENGTHS (default 64k, 128k)
# EVAL_LCB_SINCE=2025-01-01  # LiveCodeBench: only problems released on/after this date (contamination)

# Rows per dataset: test, quick, basic, moderate, standard, thorough, deep, ultra, full - or funnel
EVAL_PROFILE=funnel          # to choose a model; test / quick for a first look
# EVAL_FUNNEL_FINAL=standard # the profile the funnel's finalists finish at
# EVAL_LEADERS=5             # minimum number of models kept at every funnel gate
# EVAL_RESULT_CACHE=true     # reuse answers already obtained with identical settings (resume, reruns)
# EVAL_SKIP_INELIGIBLE=auto  # no evals for models that fail MIN_CTX / the budget profile

# Questions in flight at once (1 = sequential). Needs llama-server -np N (and ideally --kv-unified)
EVAL_CONCURRENCY=1
# EVAL_HTTP_BACKEND=auto     # auto | aiohttp | threads
# EVAL_SANDBOX_WORKERS=0     # parallel code executions (0 = auto)

# Streamed evals (speed tests are never streamed)
# EVAL_STREAM=true           # false = no loop abort, no salvage, no live thinking budget
# EVAL_THINK_BUDGET=16384    # max thinking tokens per question (0 = unlimited) -> "Fail: Think cap"
# EVAL_THINK_BUDGET_GSM8K=   # per-dataset override (opt-in; changes what is measured)
# EVAL_BUDGET_FORCING=false  # at the cap: close the thinking and ask for the answer instead of a No-Thinking retry
# EVAL_LOOP_ABORT=true       # stop repetition loops -> "Fail: Degradation"
# EVAL_SALVAGE=true          # grade the complete visible answer of a stopped reply
# EVAL_TEMPERATURE=0         # reasoning models often loop at 0; see Troubleshooting
# EVAL_TIMEOUT=600           # safety net per request

# Sandbox for executing generated code: auto / docker (needs Docker) | subprocess (not isolated)
EVAL_CODE_SANDBOX=auto

# Agent score (quality first, speed second; see Metric definitions)
# AGENT_SPEED_WEIGHT=0.3     # most of the score speed can remove (0 = accuracy only)
# AGENT_ACC_TOL=             # points below the best eval score still counted as a tie (empty = 1 standard error)
# AGENT_ACC_SPAN=40          # points under the tie zone at which the accuracy credit reaches 0
# AGENT_FALLBACK_PENALTY=10  # % of quality lost if EVERY question needed the No-Thinking retry
# AGENT_LONGCTX_WEIGHT=0.25  # share of the quality figure from long-context retrieval
# AGENT_SEC_REF=60           # s/q at which the s/q credit halves
# AGENT_TTFT_REF=15          # agent-prompt TTFT (s) at which its credit halves
# AGENT_PREFILL_REF=4000     # prefill tok/s that earns the full prefill credit (when TTFT is not measured)
# AGENT_TPS_REF=200          # tokens/s that earns the full token-speed credit

# --- Deployment fit ---
PREFILL_CTX_TESTS=8192,32768,65536   # prefill tok/s at these prompt lengths (8k 16k 32k 64k 128k)
AGENT_PROMPT_TOKENS=20000            # TTFT of an agent-shaped prompt (0 = off)
MIN_CTX_ACTION=flag                  # a model below MIN_CTX: flag (kept off the podium) | refuse (not benchmarked)
BUDGET_PROFILE=server                # server | main | your own name | none (rules: see .env.example)
GGUF_INSPECT=true                    # read the model file's header
# GGUF_PATH_MAP=/models=/mnt/AI-Models   # container path -> host path, when the file is not readable as-is

# --- Output ---
WRITE_MARKDOWN=true
JSON=true                    # results.json + benchmark_<timestamp>.json
```

## Making a full run faster

On a 16 GB card with ~28 models, 94-98% of a full run is the GPU generating (or reading) tokens, and most of what it generates in the evals is thinking. Loads, cooldowns and sandbox start-up are a few percent. Single-request token speed is fixed by the hardware, so the levers are: ask fewer questions of models that are clearly losing, generate fewer tokens, and run several questions at once on the same card. None of them may let a weaker model rank higher, and each one below says how that is guaranteed.

| Lever | Setting | Typical saving | Why the ranking stays honest |
| --- | --- | --- | --- |
| Timing report + estimate | always on; `--estimate` | measures the rest | `results.json` (`run.timing`, `models[].timing`), CSV `time_*_s`, dashboard "Where the time went" |
| Skip undeployable models | `EVAL_SKIP_INELIGIBLE=auto` | the share of models that fail a rule | they are off the podium already; the Agent "bar" only ever came from eligible models |
| Funnel | `EVAL_PROFILE=funnel` | ~40-50% of eval work (more when the field is spread out) | a model is only dropped when its upper bound is below the leaders' lower bound; screened models are not ranked against the finalists |
| Batched evaluation | `EVAL_CONCURRENCY`, `--batch-probe` | eval time / 1.8-3 on dense models fully on the GPU (less for offloaded MoEs) | requests capped to the server's slots and KV pool; with concurrency the Agent score uses the single-stream s/q estimate |
| Result cache + resume | `EVAL_RESULT_CACHE=true` | reruns only ask what changed | an answer is reused only when the model file, launch flags, build, prompt, sampling and grader are identical |
| Pipelined grading | always on | the sandbox time (a few %) | grading is unchanged, it just runs while the next question generates |
| One long-context document per length | always on | about half of the long-context time, plus the separate 64k/128k prefill prompt | prefill tok/s depends on length, not content; questions ask different needles |
| Thinking budgets | `EVAL_THINK_BUDGET_<DATASET>`, `EVAL_BUDGET_FORCING` | 15-35% of tokens (model-dependent) | **opt-in only**: this changes what is measured (accuracy within a budget), recorded in the JSON and dashboard |

### Funnel (`EVAL_PROFILE=funnel`)

Every model answers the first stage (`quick`, 11 questions per dataset). After each stage a gate keeps every model whose upper bound (one-sided, `EVAL_FUNNEL_CONF`, default 95%) reaches the lower bound of the `EVAL_LEADERS`-th best model, and never fewer than `EVAL_LEADERS` models. The survivors answer the next stage (`basic`, `moderate`, ... up to `EVAL_FUNNEL_FINAL`, default `standard`); questions are nested, so each stage only asks the new ones. The compared figure is the one the Agent score ranks on (dataset average, blended with long-context retrieval when every model has it).

A plain "keep the top 5 after `quick`" was rejected: in a simulation with a crowded top (six models within six points) it screened out the true best model in 18-33% of runs. The confidence gates lost it in at most 0.1% of runs and picked as well as running everyone at the final profile. A screened-out model keeps its stage score (with its larger margin of error) and the column **First-stage avg** compares every model on the same first-stage questions, but it is **not ranked** and never on the podium.

Note: v2.1 samples questions as nested prefixes of one seeded order per dataset, so every profile's questions are a subset of the next one's. v2.0 drew each profile independently, so per-question results are not row-for-row comparable with v2.0 runs (the statistical precision is the same).

### Batched evaluation (`EVAL_CONCURRENCY`)

Generating tokens on one GPU is limited by memory bandwidth, so serving several requests in one batch raises the total tokens per second. llama-server needs slots for this (`-np N`). By default `-c` is the **total** context and each slot gets `c / N`; with `--kv-unified` (the default in current llama.cpp when `-np` is left at auto) all slots share one pool and each request may use the full context. Without it, `-np 4 -c 131072` gives every request 32k, which fails a 64k `MIN_CTX`.

The runner caps the requests in flight per model: never more than the server's slots, and with a shared pool never more than fit at their worst case (prompt + `max_tokens`, which the thinking budget limits to budget + 4,096). The cap is printed when it applies.

Measure before relying on it:

```bash
INCLUDE_MODELS=small-dense,mid-dense,offloaded-moe BATCH_PROBE_LEVELS=1,2,4,8 python benchmark.py --batch-probe
```

The probe asks the same `BATCH_PROBE_QUESTIONS` (default 32) questions at each level and reports aggregate tok/s, the **steady gain** over one request, per-question latency, accuracy, how many answers **agree** with the one-request answers (llama.cpp is not batch-invariant, so a few greedy answers can flip), request errors (a pool that overflowed) and peak VRAM. The steady gain is the throughput while the batch is full (Little's law: requests in flight × tokens per second of latency, relative to one request), so one long reply finishing alone at the end does not hide the gain. It recommends the smallest level that reaches 90% of the best steady gain with no errors and accuracy within the margin of error; agreement is only a rule (at least 90%) when 40 or more questions were asked, because with fewer a handful of chaotic flips is noise. Every probe answer is written to `batch_probe_<timestamp>_details.jsonl`. The probe reads context, slots and KV layout from the server, refuses more than 4 models (name them with `INCLUDE_MODELS`) and warns when other models are still being served (stop any other run first). `--estimate` uses the result, and models of the same offload class without a probe borrow the median gain of those with one.

**s/q under concurrency:** measured latency then includes waiting behind the other requests, and the in-flight cap differs per model. So when any model ran with more than one request in flight, the Agent score uses the **single-stream estimate** for every model: tokens generated ÷ the speed test's tok/s + prompt tokens ÷ its prefill tok/s + grading time (`EVAL_SQ_BASIS=auto`). With one request at a time the measured latency is used, exactly as before.

### Result cache and resume

Every answered question is stored in `results/cache/` (one JSONL file per model fingerprint). A cached answer is reused only when everything that could change it is identical: the model file (size and SHA-256 of its first and last MiB), the server's launch command (port, host, threads, logging and API key removed), llama.cpp's `build_info`, the loaded context and slots, the chat template, the hardware, the exact request (prompt, `max_tokens`, temperature, tools, `EVAL_EXTRA_BODY`), the thinking budget and retry/salvage/forcing switches, code-grading settings and the grader version. Change any of them and the question is asked again. If the model file is not readable from where the benchmark runs, the fingerprint falls back to what the server reports and says so. API errors are never cached. The speed tests are never cached. Delete the folder, or set `EVAL_RESULT_CACHE=false`, to ask everything again. The cache holds model replies: do not publish it after a GPQA run.

### Recommended setup for choosing a model

```ini
EVAL_PROFILE=funnel
EVAL_FUNNEL_FINAL=standard
EVAL_LEADERS=5
EVAL_SKIP_INELIGIBLE=auto
BUDGET_PROFILE=server            # or main
EVAL_CONCURRENCY=1               # raise after --batch-probe, with -np / --kv-unified in llama-swap
```

Keep GPQA and LiveCodeBench for decisive runs: together they are about half of a full-profile run's time.

## 📊 Outputs

All results are saved to `OUTPUT_DIR` (default: `results/` next to the script).

* `benchmark_<timestamp>.csv`: Raw telemetry and benchmark data for the current run.
* `benchmark_history_v2.csv`: Every run appended over time. If new columns are added (for example when you enable GSM8K or MATH), older rows are migrated automatically and a `.bak` copy is kept.
* `results.json` and `benchmark_<timestamp>.json`: Structured export for dashboards, databases and notebooks. `results.json` is always the newest run and is written atomically, so it is safe to poll. See [JSON export](#json-export).
* `eval_details_<timestamp>.jsonl`: Per-question evaluation logs containing the model responses and grading logic. Each line has the verdict, `flag` (`Fail: Degradation` / `Fail: Think cap` / empty), `salvaged`, `forced`, `api_error`, `tokens` and `think_tokens`, `prompt_tokens`, `seconds` and `grade_s`, the funnel `stage`, `cached`, and `attempts`: one entry per request sent, with how it ended (`stop`, `length`, `loop`, `budget`, `timeout`, `error`), its mode (`think`, `forced`, `no-think`), its tokens and seconds, and for stopped attempts the reason and the last ~600 characters the model wrote. Long-context lines include the reused prefix (`cache_n`).
* `cache/`: the result cache (see [Result cache and resume](#result-cache-and-resume)).
* `batch_probe_<timestamp>.json`: written by `--batch-probe`.
* `benchmark_dashboard_<timestamp>.html`: The interactive, self-contained dashboard for this run. The newest one is also copied next to the script as `benchmark_dashboard.html`.
* `benchmark_report.md`: Markdown report with the top 3, speed overview, full comparison, quantization analysis and eval results.
* `benchmark_<timestamp>.log`: Everything printed during the run, with start/end time and time taken.

### JSON export

`results.json` has these top-level sections (`schema_version` 1):

| Section | Contents |
| --- | --- |
| `run` | Run id, start/end time, duration, status (`complete`, `interrupted`, `failed`), mode, backend, model counts, `timing` (wall time of the speed and eval phases, every model's phases summed, eval time per dataset, thinking and answer tokens) |
| `host` | Hardware, platform, Python version, GPU backend used, CPU power source |
| `config` | The settings used for the run (API keys and tokens are never written) |
| `evaluation` | Profile, seed, temperature, concurrency, HTTP backend, code sandbox, streaming settings (`stream`, `think_budget`, `think_budgets`, `budget_forcing`, `loop_abort`, `salvage`), `sq_basis`, `result_cache`, `skip_ineligible`, `funnel` (stages, leaders, confidence and every gate decision), margin of error, long-context settings, and per dataset the row counts, pinned `revision` and LiveCodeBench `since` |
| `artifacts` | Paths of every file written for the run, including the JSON itself |
| `models[]` | One entry per model: `metadata`, `speed`, `gpu`, `cpu`, `memory`, `energy`, `scores`, `launch`, `gguf`, `context`, `long_context`, `budget`, `timing` (seconds per phase), `evaluation` (`incomplete`, `skipped`, `screened`, `stage`, `stage1_avg_pct`, `in_flight`, `sec_basis`, `cached`, tokens, and per dataset: `pct`, `n`, `rows`, `err`, `na`, `fallback`, `sec`, `sec1`, `trunc`, `degr`, `cap`, `salv`, `forced`, `tok`, `think`, `answer`, `ptok`, `grade_s`, `cached`, `wall`, `concurrency`), and `other` for any column not in a group |

Missing measurements are `null`, never `NaN`. A JSON file is also written when every model fails, so errors are machine-readable.

## The dashboard

* **Header:** Model count, date, your host hardware (CPU, RAM, GPU) and the power sources used.
* **Filters:** A sticky toggle bar for quantization, Dense/MoE, parameter-size bucket, "Hide VRAM ≥95%" and "Hide failed".
* **Top 3 Goldilocks podium:** Gold/silver/bronze cards ranking Overall Balance, Chat Driver, and Agentic Reasoner (which factors in eval quality).
* **Charts:** Generation speed, VRAM vs speed, power efficiency, energy per token, TTFT leaderboard, latency vs throughput, prefill vs generation, active vs idle GPU power, speed by quantization, and **Accuracy & Pass rates** (knowledge, math and coding shown as separate groups).
* **Full model comparison:** Columns with no data in a run are hidden automatically. Click a row to set it as a baseline, and every other row shows its +/- % against it. Models that are not ranked (incomplete evals, screened out by the funnel) or whose evals were skipped carry a badge; hover it for the reason.
* **Where the time went:** time per model and phase (speed tests, eval load, long context, each dataset) as a stacked chart, the totals, and the thinking vs answer tokens generated.
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
| Goldilocks | params (B) x average TPS, using total parameters from the GGUF header (else the filename) |
| KV type, FA, -ub / -b, Slots, Unified KV, CPU offload, mmproj | Read from the llama-server command line (`/proc/<pid>/cmdline`, else llama-swap's `/running`). KV type defaults to f16 when no `-ctk/-ctv` flag is given. Unified KV is blank when `-np` is left at auto (current llama.cpp then enables it) |
| BPW, GGUF GB, Experts, Attention, Full-attn layers, Train ctx | From the GGUF file header (own reader; the file is found through `/proc/<pid>/root`, `GGUF_PATH_MAP` or `GGUF_SEARCH_DIRS`) |
| KV KiB/tok, KV GB @ctx | KV-cache bytes per token from the header's layer/head layout and the KV type (sliding-window layers counted at their window), times the loaded context |
| Max ctx @X% | The context that would fit with VRAM at `MAX_CTX_VRAM_PCT` of the GPU: loaded context + (X% of VRAM - measured VRAM) / KV bytes per token, capped at the training context |
| Prefill 8k...128k t/s | `timings.prompt_per_second` for one cold, unique prompt of that length (`PREFILL_CTX_TESTS`). A length the long-context test also uses is taken from its first, cold prompt |
| Agent TTFT s / Next-turn TTFT s | Time to first token for an agent-shaped prompt of `AGENT_PROMPT_TOKENS` (a 16-tool catalogue plus a long history), cold, and for the next turn of the same conversation (prefix cached) |
| Long-ctx % | Share of asked vault codes returned, over `LONGCTX_TRIALS` questions about one document of that length, each asking `LONGCTX_KEYS` different vaults among as many decoys; asked with thinking off. A model that thinks anyway and spends its whole reply on reasoning is asked once more with 4,096 tokens (noted, and `thinking_retry` in the details). Flagged when the KV type is below `LONGCTX_KV_MIN` |
| Tool calling % | Built-in BFCL-style cases (simple, multiple, parallel, irrelevance) sent as OpenAI `tools` requests, graded on function name and arguments |
| Budget columns | One pass / fail / ? per `BUDGET_PROFILE` rule. VRAM rule: model VRAM + `VRAM_RESERVE_GB` <= GPU total x `MAX_VRAM_PCT`. `?` = not measured, never a fail. A failing model stays off the podium (`BUDGET_PODIUM_PASS_ONLY`) |
| Margin of error | 95% binomial interval, worst case: ±98/√n points per dataset (`test` ±40, `quick` ±30, `basic` ±20, `moderate` ±15, `standard` ±10, `thorough` ±7). `test` and `quick` are flagged as a rough guide |
| QA accuracy (GPQA, MMLU-Pro) | The model ends with `Answer: X`; the last such letter (or `\boxed{}`, or a final line that is just a letter) is compared with the dataset answer |
| **Math accuracy (GSM8K, MATH-500)** | The model ends with `Answer: <answer>`. The last `Answer:` line is used, else the last `\boxed{}`, else (GSM8K only, and never for a cut-off reply) the last number. Answers are normalised (units, `\dfrac`, thousands separators, `x=` prefixes) and match when the strings are equal or both are numerically equal (fractions, decimals) to 1e-6. Answers that are only equivalent after algebra count as wrong, so scores are a slight lower bound |
| Code pass@1 (HumanEval, MBPP, LiveCodeBench) | The last Python block of the reply is executed in the sandbox. HumanEval uses the task's `check()`, MBPP its `test_list` asserts, LiveCodeBench public tests plus up to `EVAL_LCB_MAX_TESTS` more (stdin/stdout or starter-code function). A program that exits before its tests finish fails |
| N/A | A reply that was cut off by the token limit and has no usable answer, even after the No Thinking retry. Counted as wrong, tracked separately |
| **Fail: Degradation** | The model's reply fell into a repetition loop (the same block of text over and over, or the same few sentences recycled) and was stopped. Counted per question in `degr` (CSV `eval_<dataset>_degr`), even when the retry then answered correctly; a question that still has no answer afterwards is wrong. Detection is conservative (exact repetition only), so a numbered enumeration is never mistaken for a loop |
| **Fail: Think cap** | The reply used more than `EVAL_THINK_BUDGET` (or `EVAL_THINK_BUDGET_<DATASET>`) tokens of thinking and was stopped. Counted in `cap` (JSON). Thinking is recognised when the server splits it out (`reasoning_content`) or when it sits between `<think>` tags |
| **Salvaged** | A stopped reply whose visible answer already contained a final answer (QA, math) or a complete code block, so it was graded as it is. Counted in `salv`. Thinking text is never used for salvage |
| **Forced** | A question answered by budget forcing (`EVAL_BUDGET_FORCING=true`): at the thinking cap the reasoning was closed and the model asked for its answer. Counted in `forced` |
| **tok, think, answer** | Mean tokens generated per question over all attempts (exact when the server reports usage, otherwise estimated), and the dataset's total thinking and answer tokens. In the JSON (and CSV totals `eval_tok_think` / `eval_tok_answer`) |
| **s/q** | Mean latency of one question: request, retries, timeouts and grading (never the model load: a warm-up request comes first). It is deliberately not dataset wall time divided by questions. With requests in flight it includes queueing, so `sec1` (JSON) is also given: the single-stream estimate, tokens ÷ speed-test tok/s + prompt tokens ÷ prefill tok/s + grading. Real elapsed time is `eval_<dataset>_wall` |
| **Eval avg %** | Mean of the dataset scores. Only for a model with a score on **every** dataset of the run (at least 90% of each dataset's questions answered without an API error); otherwise the model is **not ranked** and the reason is shown ("incomplete evals: GPQA 0/96 answered") |
| **Retries, Eval − retry penalty** | Share of questions that needed the No-Thinking retry, and the eval average minus `AGENT_FALLBACK_PENALTY` × that share |
| **Agent Score** | 0-100, quality first. **1. Quality** = the eval average, blended `AGENT_LONGCTX_WEIGHT` (25%) with long-context retrieval when every model has it, compared with the best *eligible* model of the run: within its noise margin (`AGENT_ACC_TOL`, default 1 standard error) = a tie at 100, below that a quadratic fall to 0 at `AGENT_ACC_SPAN` (40) points under the tie zone. Then × (1 − `AGENT_FALLBACK_PENALTY`% × the share of questions retried), applied after the tie zone so it separates tied models. **2. Speed factor** (0-1): prompt processing 40% (`1/(1 + agent TTFT ÷ AGENT_TTFT_REF)`, else `ln(longest prefill tok/s) / ln(AGENT_PREFILL_REF)`), s/q 35% (`1/(1 + s/q ÷ AGENT_SEC_REF)`), generation 25% (`ln(TPS) / ln(AGENT_TPS_REF)`); missing parts are left out and the rest re-weighted. `score = quality × (1 − W + W × speed)` with `W = AGENT_SPEED_WEIGHT` (0.3), so speed separates models of equal quality but never lifts a clearly weaker one. VRAM is not part of it. A model whose evals all failed, are incomplete, were skipped, or that the funnel screened out gets **no** Agent score. **Without evals in the run:** a proxy from dense-equivalent size (a MoE counts as √(total × active)) with a penalty below ~4.5 bits per weight, relative to the largest, also 0-100 and marked as a proxy |
| **Funnel stage, First-stage avg** | The last funnel stage a model answered, and every model's average on the first stage's questions only (like-for-like across finalists and screened models) |
| **In flight** | Requests a model actually had in flight during the evals (after the slot / KV-pool cap). Shown when `EVAL_CONCURRENCY` > 1 |
| **Time min** | Everything this model took in the run (speed tests, loads, long context, evals). Per phase in the JSON |
| **VRAM fit %** | 100% when VRAM use is inside `VRAM_SWEET_LOW`-`VRAM_SWEET_HIGH` (default 80-90%); falls gently below it, to 50% at `VRAM_SWEET_MAX` (94%) and 0 at 100% |
| **Sweet spot** | `Agent score × VRAM fit`: the most capable model that also uses the GPU sensibly (some headroom, nothing wasted). Ranked in the podium's "VRAM sweet spot" row |
| **VRAM tuning hint** | Under the band with CPU offload: room to lower `--n-cpu-moe`. Over the band: raise `--n-cpu-moe` or cut ctx / KV cache. Tip: add several llama-swap entries of one model with different `--n-cpu-moe` values and the Sweet spot ranks them |

## Streaming, loop abort and the thinking cap

Reasoning models can fall into endless loops, and a non-streaming request returns nothing until the model has finished, so a looping reply used to run into the 10-minute timeout and the benchmark learned nothing about it. Since v2.1 the evaluation phase reads replies as a stream (the speed tests do not), which allows:

1. **Loop abort:** a reply that repeats itself is stopped within seconds and flagged `Fail: Degradation`. A line is printed as it happens, for example `[Fail: Degradation] gpqa-10: repetition loop (a 59-character block repeated 67+ times) after ~1,058 tokens (7s) -> retried with thinking disabled`.
2. **Thinking cap:** `EVAL_THINK_BUDGET` (default 16384 tokens) limits the thinking per question; `EVAL_THINK_BUDGET_<DATASET>` (e.g. `EVAL_THINK_BUDGET_GSM8K=2048`) overrides it for one dataset. The server is also told `max_tokens <= budget + 4096`.
3. **Partial-answer salvage:** a stopped reply whose *visible* answer is already complete is graded as it is. Otherwise the question is retried once with thinking disabled (`EVAL_RETRY_ON_LENGTH`), and if that fails too it counts as wrong and keeps its flag.
4. **Budget forcing (opt-in, `EVAL_BUDGET_FORCING=true`):** at the thinking cap the reasoning written so far is kept, the thinking block is closed, and the model is asked for its answer (through llama-server's `/apply-template` and `/completion`, for chat templates that use `<think>` tags). This keeps the work already done instead of throwing it away for a No-Thinking retry. Method from *s1: Simple test-time scaling* (Muennighoff et al., 2025). It changes what is measured compared with the retry, so compare runs with the same setting. llama-server's own `--reasoning-budget` launch flag is an alternative on the server side.
5. **Evidence:** the tail of every stopped reply is kept in `eval_details_*.jsonl` (`attempts[].tail`), and the per-dataset summary shows how many questions were affected and how many were recovered.
6. **No runaway generation:** closing the connection makes llama-server stop generating. (A non-streaming request that times out is not reliably cancelled, so the abandoned generation could keep the GPU busy and slow the next question down.)

The default budget is a judgement call: it keeps a run's duration predictable, but it also means that a model which needs more than about 16k tokens of thinking for a question is scored as if it had not answered that way. Every cap hit is counted, so you can see how many questions were affected and raise the budget if it matters to you (`EVAL_THINK_BUDGET=32768`, or `0` for unlimited). Per-dataset budgets are opt-in for the same reason: a tight budget measures "accuracy within N thinking tokens", which can reorder models (a strong model that reasons at length can drop below a terse weaker one). The budgets in use are recorded in `results.json` and shown in the dashboard's methodology. Set `EVAL_STREAM=false` to switch streaming off.

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
* **Docker/Sandbox issues during evals:** The sandbox self-test runs when the datasets load and in `--check`. If Docker is unavailable or fails it, the run stops with the reason (v2.0 fell back to a subprocess with only a warning, which in an unattended run meant executing untrusted code as your user). Make sure the Docker daemon is running and your user can use it, or set `EVAL_CODE_SANDBOX=subprocess` (only for models you trust). Containers left behind by an interrupted run are removed automatically on the next start.
* **Evals are no faster with `EVAL_CONCURRENCY` above 1:** The run prints `in flight: 1 (capped: ...)` with the reason. Give llama-server slots (`-np N`, N at least `EVAL_CONCURRENCY`) and `--kv-unified`, and check llama-swap's `concurrencyLimit` for that model (when it is reached llama-swap answers 429 instead of queueing). With a shared KV pool, raise `-c` or lower `EVAL_THINK_BUDGET` if the cap is the pool. Run `--batch-probe` to see the real gain.
* **A model shows "not ranked":** hover the badge. *Incomplete evals*: a dataset failed or stopped early (often a chat-template / `--jinja` problem); fix it and rerun (the result cache keeps everything else). *Screened out*: the funnel found it clearly below the leaders; its first-stage score is still shown. *Evals skipped*: it fails a deployment rule (set `EVAL_SKIP_INELIGIBLE=false` to evaluate it anyway).
* **A rerun finishes instantly / did not ask anything:** the result cache had every answer. Set `EVAL_RESULT_CACHE=false` (or delete `results/cache/`) to measure again.
* **Many `Fail: Degradation` lines, or questions taking 10 minutes:** Reasoning models often loop under greedy decoding (`EVAL_TEMPERATURE=0`). Use the sampling settings the model's authors recommend (often 0.6 to 1.0), or switch thinking off for that model with `EVAL_EXTRA_BODY`. Heavily quantized models (IQ2/IQ3) loop more. The run prints a warning when more than 25% of a dataset's questions loop.
* **Many `Fail: Think cap` lines:** The model genuinely thinks for a long time. Raise `EVAL_THINK_BUDGET` (the run then takes longer), or accept the cap as part of the score.
* **After a timeout the next request is unusually slow (about half speed):** In the llama-swap log, requests that follow a 499 status (client closed) can decode much more slowly. With non-streaming requests the abandoned generation may keep running on the server. Streaming (the default since v2.1) closes the connection explicitly; to check on your build, run `curl http://<server>/slots` right after an abort and confirm no slot is still processing.
* **Eval scores aren't comparable with older runs:** v2.1 samples nested questions, so a profile's questions differ from v2.0's (scores move within the margin of error); the long-context test uses one document per length; `EVAL_LCB_SINCE` restricts LiveCodeBench to newer (harder) problems. The thinking cap can also lower the score of a model that needs very long reasoning; compare runs with the same `EVAL_THINK_BUDGET`. Enabling GSM8K or MATH changes the eval average, since it averages every enabled dataset.
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

## 🙏 Credits, datasets and licences

This tool does not include or redistribute any benchmark dataset. Each one is downloaded from Hugging Face at run time (or with `--download-datasets`) into the local `tests/` cache, and is used under the licence and terms set by its authors. All credit for the questions, problems, test cases and reference answers belongs to them. If you publish results produced with these datasets, please cite the original papers below.

The licence of this repository covers its own code only. It does not re-license any dataset, and the dataset licences do not change the licence of this code.

### Evaluation datasets

| Eval | Original authors and paper | Original source | Copy this tool downloads | Licence |
| --- | --- | --- | --- | --- |
| **GPQA Diamond** | Rein, Hou, Stickland, Petty, Pang, Dirani, Michael, Bowman. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark* (COLM 2024). [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) | [idavidrein/gpqa](https://github.com/idavidrein/gpqa) · [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (gated) | [fingertap/GPQA-Diamond](https://huggingface.co/datasets/fingertap/GPQA-Diamond) (community copy of the 198-question Diamond subset) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), plus the authors' request below |
| **MMLU-Pro** | Wang et al. (TIGER-Lab). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* (NeurIPS 2024 Datasets and Benchmarks). [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) | [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT. Built from MMLU (Hendrycks et al., [arXiv:2009.03300](https://arxiv.org/abs/2009.03300), MIT), TheoremQA (MIT), SciBench (MIT) and STEM websites |
| **HumanEval** | Chen et al. (OpenAI). *Evaluating Large Language Models Trained on Code*, 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) | [openai/human-eval](https://github.com/openai/human-eval) | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | MIT |
| **MBPP** | Austin et al. (Google Research). *Program Synthesis with Large Language Models*, 2021. [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **LiveCodeBench** | Jain, Han, Gu, Li, Yan, Zhang, Wang, Solar-Lezama, Sen, Stoica. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*, 2024. [arXiv:2403.07974](https://arxiv.org/abs/2403.07974) · [livecodebench.github.io](https://livecodebench.github.io/) | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) (code: MIT) · [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) | [bzantium/livecodebench](https://huggingface.co/datasets/bzantium/livecodebench) (a copy of `code_generation_lite` in plain JSONL, because the original's loading script no longer works with current `datasets`; default config `release_latest`, 1,055 problems from May 2023 to April 2025) | The dataset card says "cc" without naming a specific Creative Commons licence, so the exact terms are **unclear**. The problems come from LeetCode, AtCoder and Codeforces, whose own terms apply to the problem statements. The original repository's code is MIT |
| **GSM8K** | Cobbe et al. (OpenAI). *Training Verifiers to Solve Math Word Problems*, 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) | [openai/grade-school-math](https://github.com/openai/grade-school-math) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main`) | MIT |
| **MATH-500** | The 500 test problems held out by Lightman et al. (OpenAI), *Let's Verify Step by Step*, 2023 ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)), from the MATH dataset by Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset* (NeurIPS 2021, [arXiv:2103.03874](https://arxiv.org/abs/2103.03874)) | [openai/prm800k](https://github.com/openai/prm800k) (`math_splits`) · [hendrycks/math](https://github.com/hendrycks/math) | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | MATH is MIT. The Hugging Face card states no licence of its own; the terms of MATH and openai/prm800k apply |

#### LiveCodeBench: which copy, and which problems

* **Language:** the `bzantium/livecodebench` card's field list says the question text is "translated to Korean". That line appears to be left over from the same author's separate `bzantium/ko-livecodebench`; the Hugging Face viewer shows English problem text for this copy. To be safe, the tool refuses a dataset when more than 1% of the sampled questions contain Korean text.
* **Pinning:** set `EVAL_LIVECODEBENCH_REVISION` (or `EVAL_GPQA_REVISION`, ... for any dataset) to a commit hash of the Hugging Face repository so a later change to a community copy cannot silently change your scores or the tests the sandbox runs. The revision is recorded in `results.json`.
* **Contamination:** LiveCodeBench's own method is to score only problems released after a model's training cutoff. `EVAL_LCB_SINCE=2025-01-01` keeps only problems from that date; older problems may have been seen in training and inflate newer models' scores. Scores with and without the window are not comparable.
* `EVAL_LCB_MAX_TESTS` (default 20) grades on the public tests plus a seeded sample of the private ones, which is a little more lenient than the official harness.

#### GPQA: please honour the authors' request

The GPQA authors ask users **not to reveal examples from the dataset in plain text or images online**, to reduce the risk of the questions leaking into model training data. The official Hugging Face copy is gated behind that agreement. The copy this tool downloads is not gated, but the request still applies. Please accept the terms on [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) if you enable `EVAL_GPQA`, and in practice:

* **Do not publish `eval_details_*.jsonl` from a run with GPQA enabled**, nor the result cache (`results/cache/`). Each line holds the model's full reply (which often restates the question) and the reference answer.
* **Do not commit the `tests/` dataset cache or the `results/` folder.** The included `.gitignore` excludes both, so `git add .` will not pick them up.
* Aggregate scores (percentages, charts, the dashboard and `results.json`) contain no questions and are fine to share.

### Methods this tool adapts (no data copied)

| Test | Credit | Licence of the original |
| --- | --- | --- |
| **Tool calling** (`EVAL_TOOLCALL`) | Categories (simple, multiple, parallel, irrelevance) and the AST-style argument matching follow the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) by the Gorilla team at UC Berkeley ([ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)). The 32 test cases in `llmbench/evals/toolcall_cases.py` were written for this tool; no BFCL data is used, and the scores are not comparable with the BFCL leaderboard | Apache 2.0 |
| **Long-context retrieval** (`EVAL_LONGCTX`) | The needle-in-a-haystack test by Greg Kamradt ([gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)), and the multi-key variant from *RULER: What's the Real Context Size of Your Long-Context Language Models?* by Hsieh et al. (NVIDIA, COLM 2024, [arXiv:2404.06654](https://arxiv.org/abs/2404.06654), [NVIDIA/RULER](https://github.com/NVIDIA/RULER)). The haystack, keys and decoys are generated by this tool at run time | See each repository |
| **Margin of error, funnel gates** | The standard 95% binomial interval (worst case, ±98/√n percentage points); the funnel's one-sided bounds use the normal approximation with each dataset's binomial standard error | n/a |
| **Budget forcing** (`EVAL_BUDGET_FORCING`) | Closing the thinking at a token budget and asking for the answer, from *s1: Simple test-time scaling* by Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès and Hashimoto, 2025 ([arXiv:2501.19393](https://arxiv.org/abs/2501.19393), [simplescaling/s1](https://github.com/simplescaling/s1)). Only the idea is used; no code or data | Apache 2.0 (s1 repository) |

### Software

| Project | Used for | Licence |
| --- | --- | --- |
| [llama-swap](https://github.com/mostlygeek/llama-swap) by mostlygeek | The model-swapping proxy this tool benchmarks through | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org) | llama-server, its `/props`, `/apply-template`, `/completion` and `timings` fields, and the [GGUF file format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) read by `llmbench/gguf.py` (own implementation, no code copied) | MIT |
| [Chart.js](https://www.chartjs.org/) | Dashboard charts, loaded from cdnjs when the dashboard is opened (not bundled) | MIT |
| [Hugging Face `datasets`](https://github.com/huggingface/datasets) / `huggingface_hub` | Downloading and caching the evaluation datasets | Apache 2.0 |
| [Requests](https://github.com/psf/requests) | HTTP client | Apache 2.0 |
| [aiohttp](https://github.com/aio-libs/aiohttp) | Concurrent evaluation requests | Apache 2.0 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` loading | BSD 3-Clause |
| [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) (NVIDIA) | NVML GPU telemetry (`pynvml`) | BSD 3-Clause |

Dependencies are installed by you from PyPI and are not redistributed here. Each project's licence applies to it.

If you are an author of one of these works and would like the credit worded differently, or a dataset removed, please open an issue.
