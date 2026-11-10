# 🛟 llama-swap-llm-benchmark v2.3.1: runs that survive trouble, `--resume`, honest timings, a harder settings page

A long run used to have three ways to lose or distort its results when something went wrong: an unexpected error threw away the JSON and CSV, a llama-swap restart cost the models tested in that window their ranking, and one failed dataset quietly changed which latency every model was scored on. This release fixes all three, adds `--resume`, and tightens the settings page and the code sandbox.

**The Agent score changes slightly in one situation** (a run with batched evals in which a dataset failed completely for some model). *How old and new results compare* below shows what that does, using the 9 November run.

## ✨ Key New Features

### 🛟 Nothing measured is lost, and `--resume`
* **Rows are saved after every model.** `results/run_<stamp>.rows.json` is rewritten after each model of the speed phase and each model of every eval stage. A crash, `kill -9` or a power cut loses at most the model being tested.
* **`python benchmark.py --resume`** continues the newest run in `OUTPUT_DIR` (or `--resume 20261110_103848`):
  * models whose speed test finished are kept and not benchmarked again;
  * models that failed, or were never reached, are benchmarked;
  * the result cache answers every eval question already asked, so only the missing ones are sent;
  * a long-context result that is complete is kept; a missing one is run.
* **It also repairs a finished run.** A finished run keeps its rows file, so `--resume` re-tests only what went wrong in it: a model that failed to load, lost its long-context test, or has incomplete evals.
* The resumed run gets its own stamp and files; `artifacts.resumed_from` in the JSON names the run it continued. Runs made before v2.3.1 have no rows file and cannot be resumed (their eval answers are still in the result cache).
* **An unexpected internal error no longer discards the run.** In the speed phase the model is marked `error` and the next one runs. In the eval phase everything measured so far is written (status `interrupted`).
* **SIGTERM is handled like Ctrl+C** (Unraid "abort script", `docker stop`, systemd): partial results are written.
* **Exit codes:** `0` complete, `2` stopped early by an error with partial results written, `130` stopped by Ctrl+C / Stop / SIGTERM, `1` could not run.

### 🔌 llama-swap goes away mid-run
* **`SERVER_WAIT_SECONDS`** (default 600; 0 = off). When requests fail because the server is not there (refused, dropped, 5xx), the run waits for it to list its models again, reloads the model, and:
  * asks the failed questions, and the ones never asked because the dataset stopped, once more;
  * repeats a long-context test that failed on a connection error;
  * benchmarks a model again whose speed test failed that way.
* Timeouts are the model's own behaviour and are not asked again. Re-asked questions are counted in `eval_reasked_n` and marked `"reasked": true` in `eval_details_*.jsonl`.

### ⏱️ Timings measured over the right window
* **A retry is no longer inside the measurement.** Latency, energy, J/token and GPU watts of a speed request are taken over the attempt that answered, not over a failed attempt plus its backoff. `speed_retries` counts them.
* **Durations use a clock that cannot step.** Every interval and telemetry window uses a monotonic clock; an NTP correction during a long run no longer shifts them.
* **"TTFT ms" is now "Prompt ms".** The column was always llama-server's prompt-processing time, not what a client waits. The evals now record what the client sees through llama-swap, per dataset in the JSON: `ttft` (median seconds to the first token), `ttfa` (to the first visible answer token, after the thinking) and `ttfa95`. Display only: no score uses them.
* **Spread between runs.** `tps_samples` keeps every speed run and **TPS spread %** shows how far apart they were.
* **How tokens/s was obtained** (`tps_basis`): `server` from llama-server's timings, `wall` when a backend reports none (that figure includes prompt processing).
* **Change since the last run.** **TPS vs last %** compares with the previous run of the same model file at the same context and offload; changes of 5% or more are printed with the llama.cpp builds (`build_info`).
* A deferred prefill figure (taken from the long-context test) is recorded only when the prompt was cold; very short requests no longer read a telemetry sample from before they started.

### 📟 Following a run
* **`results/run_status.json`**: phase, model i of N, dataset, questions done, started / updated, host, pid. The settings page shows it above the log; a script or a Home Assistant sensor can read it.
* **`NOTIFY_URL`**: a short JSON summary is POSTed there when a run ends (status, duration, models, failed models, top three). No settings, tokens or answers.
* **New flags for one-off runs:** `--models`, `--exclude`, `--profile`, `--output-dir` (the same as setting `INCLUDE_MODELS`, `EXCLUDE_MODELS`, `EVAL_PROFILE`, `OUTPUT_DIR` for that run).
* **A second speed run is refused** while another run is going (`--force` starts anyway). The lock now carries the host name and a heartbeat, so a run started on another machine over a share is seen as live. Evals-only runs are still only warned.
* Model names such as `…-E4B-…` (effective parameters) are read as 4B when the name has no plain size.

### 📊 Dashboard
* **Chart.js 4.4.1 is written into every dashboard.** It opens without internet access and no longer loads a script from a CDN. Each file grows by about 200 KB.

## 🔧 Fixes
* One dataset with no real answers switched the whole run from the single-stream s/q estimate to measured latency (see below).
* Answers cached at one concurrency and reused in a run at another: the s/q basis now follows the concurrency the answers were measured at.
* The Home Assistant power poller kept every sample for the whole run and scanned them all per request.
* NVML was shut down while its polling thread could still be reading.
* **A run that failed at once was reported as "The run ended".** A Full run started from the settings page under a Python without the `datasets` package died in its first second and the page said nothing. The page now shows a red "The run failed" with the last lines of the log (amber for a run stopped early, green for a finished one), and `--check` prints `Hugging Face datasets: NOT AVAILABLE` with the install command when an eval needs it. If you use a virtual environment, start `--ui` from it: the page runs the benchmark with the Python that started the page.
* **`KEY=   # comment` is read as empty.** python-dotenv reads an empty value followed by a comment as the text of the comment (a `UI_TOKEN=   # ...` line became the token). The benchmark and the settings page now read it as empty. `--check-env` still flags the line; write `KEY=""` or put the comment on its own line.

## 🔒 Security
* **Settings page on the LAN:** a `UI_TOKEN` shorter than 16 characters is refused at start (delete the line and a random one is generated). Ten wrong tokens from one address lock that address out for five minutes.
* **Dashboards served by the settings page are sandboxed** (`Content-Security-Policy: sandbox`): their scripts cannot reach the page's cookie or API.
* **`.env`, `.env.bak` and `.env.tmp` are created readable by their owner only** (0600), whatever the umask.
* **The page only reads a run log inside `OUTPUT_DIR`** (`benchmark_*.log`, `batch_probe_*.log`); the path in the lock file is no longer trusted.
* **Subprocess sandbox as root:** model-written code is started as the unprivileged user `nobody`. If that is not possible a warning says the code runs as root.
* The run prints the sandbox image's ID and reminds you that a tag can change; pin one with `EVAL_CODE_DOCKER_IMAGE=<image>@sha256:<digest>`.
* `run_<stamp>.rows.json` never stores the value of `--api-key`.

## ⚖️ How old and new results compare

**What changed.** With batched evals the Agent score uses the single-stream s/q estimate, so queueing cannot change a ranking. v2.3.0 fell back to *measured* latency for every model when any model had a dataset in which every request failed. Now a dataset with no real answers is ignored for that decision.

**When it applies.** Only to runs with `EVAL_CONCURRENCY` above 1 in which some model lost a whole dataset. Runs without such a failure score exactly as before, and so do evals-only runs. One more case: a run at concurrency 1 that reuses answers cached at a higher concurrency now uses the single-stream estimate too, because those cached latencies include queueing.

**The 9 November 00:45 run** (14 ranked models, concurrency 4, llama-swap down for a few minutes), rescored with v2.3.0 and v2.3.1 scoring at default settings:

| Model | v2.3.0 | v2.3.1 | Rank |
| --- | --- | --- | --- |
| Groot-19B-A3B Q4_K_M | 93.8 | 94.6 | 1 → 1 |
| Ornith-1.5 IQ3_XXS-IQ2_S | 92.0 | 92.5 | 2 → 2 |
| gemma-4-26B-A4B IQ3_XXS | 88.8 | 90.4 | 3 → 3 |
| Ornith-1.5 Q5_K-IQ4_XS | 86.5 | 87.9 | 4 → 4 |
| Ornith-1.5 Q4_K_M | 86.2 | 87.6 | 5 → 5 |
| Qwen3.8-27B GSQ-RCO IQ3_XXS | 85.1 | 86.0 | 6 → 6 |
| Qwen3.8-27B UD IQ3_XXS | 84.7 | 85.8 | 7 → 7 |
| BigBang-v1 Q4_0 | 82.9 | 84.0 | 9 → 8 |
| BigBang-v1 IQ3_XXS | 83.0 | 83.7 | 8 → 9 |
| BigBang-v1 Q4_K_M | 81.3 | 82.7 | 10 → 10 |
| LFM2.5 Q4_K_XL | 78.2 | 79.7 | 11 → 11 |
| GLM-4.7-Flash Q4_K_XL | 73.1 | 75.0 | 13 → 12 |
| Wicked-Stuff IQ3_XS | 73.2 | 74.4 | 12 → 13 |
| GLM-4.7-Flash IQ3_XXS | 72.5 | 73.7 | 14 → 14 |

Every score rises by 0.5 to 1.9 points, because queueing no longer counts as slowness. Two adjacent pairs that were 0.1 apart swap. The models that were not ranked before are not ranked now, for the same reasons.

**Nothing else in scoring changed.** `GRADER_VERSION` is unchanged, so the result cache stays valid. Questions lost to an outage are now asked again instead of counted wrong; that can only raise the score of the model that was hit, toward what it would have scored without the outage.

## 🔄 Updating
1. Copy the files over your install. Your `.env`, `results/` and cache are untouched.
2. `python benchmark.py --check-env` lists the two new settings.
3. **If your settings page is open to the LAN with a short `UI_TOKEN`, it will not start** until the token has 16 characters or the line is deleted.
4. A cron or User Script that tests the exit code: Ctrl+C / Stop now ends with 130, an early stop with 2.

## ⚙️ Configuration
| Setting | Default | What it does |
| --- | --- | --- |
| `SERVER_WAIT_SECONDS` | `600` | How long to wait for llama-swap to answer again mid-run before giving up on the failed requests; 0 = off |
| `NOTIFY_URL` | empty | POST a JSON summary there when a run ends |

## Not in this release
The three structural changes from the v2.3.0 review are left for v2.4, because they touch every module and should be agreed first: scoring as a pure function of the measurements (the base for `--rescore` and the score preview), a run-context object instead of rewriting `Config` during a run, and one shared retry policy for the four HTTP paths.
