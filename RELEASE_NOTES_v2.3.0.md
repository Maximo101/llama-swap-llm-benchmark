# 🚀 llama-swap-llm-benchmark v2.3.0: an Agent score built for agents, prefill at depth, a clearer dashboard, and a settings page

The Agent score now ranks models the way an agent uses them: it requires enough context, prefers more, scores tool calling as its own part, and compares long-context retrieval on the same questions for every model. A new speed test measures what an agent waits for on every turn: new tokens on top of a long, cached conversation. The dashboard gets six colour tiers, a light theme, model families and a Help page. A settings page (`python benchmark.py --ui`) edits `.env`, keeps favourite setups, and starts or stops runs from the browser.

**The Agent score changes in this release.** *How old and new results compare* below shows what that does to rankings, using two real runs.

## ✨ Key New Features

### 🤖 Agent score for agentic use
* **Usable context, a minimum and a mild preference.** The Agent score now uses the context a model can actually work with, not just the one it was loaded with. The long-context lengths of the run are walked upwards; every length that fits the loaded context must reach `AGENT_CTX_PASS_PCT` (75%). If all of them pass, the usable context is the loaded context (a 90k model that finds 100% at 64k counts as 90k). If one fails, the usable context is the longest length passed before it (a 128k model that finds 12% at 128k works like a 64k model). A length without a result does not count as passed. A model whose usable context is below `AGENT_CTX_MIN` (default 64k) gets **no Agent score**, so a model must pass the 64k test to be ranked. Above it, a **mild** context factor from `AGENT_CTX_CURVE` multiplies the score: 128k and up 1.0, 77k 0.97, 64k 0.95, linear in between (90k ≈ 0.98). The three settings are in PART 1, because what an agent needs depends on your setup: a very smart model that only fits with a short context can still be ranked by lowering `AGENT_CTX_MIN` and adding an anchor. `AGENT_CTX_PASS_PCT=0` uses the loaded context; `AGENT_CTX_MIN=0` and `AGENT_CTX_CURVE=off` switch the rest off. New column **Usable ctx**.
* **Composite quality.** The Agent score now combines three parts:
  * evals (every dataset except tool calling) 45%;
  * tool calling 30% (`AGENT_TOOL_WEIGHT`);
  * long-context retrieval 25% (`AGENT_LONGCTX_WEIGHT`).

  A part the run doesn't have gives its weight to the evals. Before, tool calling was one dataset among many in the eval average, and long-context retrieval was only blended when every model had it.
* **Long-context retrieval on the same questions for every model.** Only the lengths that every ranked model ran count. v2.2 averaged each model over its own lengths, so a 90k model that skipped the hard 128k test could beat a 128k model that passed it only partly. A model with no long-context result at 64k is not ranked (see above), so the others keep the long-context part. Only with `AGENT_CTX_PASS_PCT=0` can a ranked model lack long-context; then it is left out for everyone and the dashboard says why.
* **Tool-calling minimum.** Below `MIN_TOOLCALL_PCT` (default 70%), a model gets no Agent score: it can't reliably drive an agent. A model within its margin of error of the minimum is marked **≈ borderline**. Tool calling now runs **in full** (all 32 cases) at every profile and funnel stage, so its margin is the same everywhere.
* **Tie zone from the composite's own noise.** A model within one standard error of the best composite counts as a tie at 100. The standard error comes from each part's question count; tool calling has only 32 cases, so it carries more noise than a large eval set. The funnel's confidence gates rank on the same composite.
* **Speed factor.** It is now weighted prompt processing 50%, seconds per question 30%, generation 20% (before: 40 / 35 / 25). Prompt processing uses prefill at depth (60%) and the cold agent-prompt TTFT (40%), whichever exist. `AGENT_SPEED_WEIGHT` stays 0.3: speed can still take off at most 30% and never lifts a clearly weaker model.
* New columns **Composite**, **Long-ctx (Agent)**, **Usable ctx** and **Ctx factor**, plus a note above the table with the weights the run used.

### ⏱️ Prefill at depth (`PREFILL_DEPTH`, default `auto`)
* The speed phase keeps a long conversation in llama-server's prompt cache: with `auto`, per model the largest of 8k / 16k / 32k / 64k that fits the loaded context with the 4k turn: a model loaded at 64k is measured at 32k, every 90k+ model at 64k (a number fixes the depth; 0 switches the test off). It then sends ~4,096 new tokens on top (a tool result) and measures only those new tokens. This is the wait on every turn of a long agent session, which the cold 8k–128k prefill tests don't show.
* New columns **Prefill @depth t/s** and **@depth TTFT s** (CSV, JSON `context.prefill_depth_*`, Markdown). Older history CSVs are migrated.
* The result is recorded only when the server reused at least 90% of the cached prefix. When it didn't, the run says how much was reused. New column **Depth** shows the depth each model was measured at.
* It is part of the Agent score's speed factor: the credit halves at about 1,333 tok/s. Prefill is slower deeper into a context, so the score only compares models measured at the same depth (the one most models had); a model measured at another depth counts its cold agent-prompt TTFT instead, with a note.

### 🎨 Dashboard: colours, light theme, families, Help
* **Six tiers per metric**, worst to best: red with a ⚠ sign, pink, amber, green, blue, purple.
  * Generation tok/s: 30 / 40 / 50 / 80 / 100.
  * Prefill tok/s: 500 / 1,000 / 1,500 / 2,500 / 4,000.
  * Accuracy and scores: 50 / 60 / 70 / 80 / 90.
  * Reversed scales for times, retries and power.

  VRAM keeps its sweet-spot colours. The worst tier always shows ⚠, so colour is never the only signal.
* **Bands or Smooth** (flat tiers, or a gradient between them) and **Absolute or Relative** (tiers spread over this run's own range, so a slow GPU still shows which models are faster). These are toggles at the top right and apply to the table and all charts. `COLOUR_STYLE` and `COLOUR_MODE` set the defaults; `COLOUR_<METRIC>=a,b,c,d,e` sets your own thresholds (e.g. `COLOUR_GEN=10,15,20,30,40` for a small GPU).
* **Light and dark theme.** Auto follows your system. The light theme has its own tier colours, each with at least 4.5:1 contrast on white.
* **Model families.** Chips come from `DASH_FAMILIES` (`Gemma=*gemma-4*;BigBang=*bigbang*`) and automatically from model names. A wildcard box takes patterns: `*gemma-4*` also finds `agent-gemma-4-26B…`.
* **? Help** lists every threshold for the run, explains the toggles, and has a rough speed-by-setup table to read the tiers against your hardware.
* "How scores work" describes the v2.3 Agent score.

### 🖥️ Settings page (`python benchmark.py --ui`)
* Shows every `.env` setting by PART and section, with its description, default, `.env.example` value and allowed choices. Values are checked as you type.
* **Save** checks the values, writes `.env` in the `.env.example` layout and keeps the old file as `.env.bak`. **Return to defaults** asks first and keeps your tokens.
* Tokens and keys are masked; **show** fetches one on request.
* **Favourites** live in `configs/<name>.env`. They hold only the settings that differ from `.env.example`, never secrets. Your current `.env` is saved as `main` on the first visit. Loading a favourite into a newer version follows renamed settings and asks you when both sides changed a value.
* **Models** lists the server's models with tick boxes for include / exclude, their file size, parameters, trained context and architecture (GGUF header, else the last run), and last run's VRAM, context and Agent score.
* On opening, the page lists the server's models and runs `--check` (once per browser session).
* **Estimate** (with unsaved settings), **Run --check again**, **Reload dashboard**, **Compare**, **Open latest dashboard**, **? Help**.
* **Run and Stop.** Full run, speed only or evals only. The page follows the run's log and picks it back up after a reload. Stop is Ctrl+C for the run, so partial results are written.
* Port 41010, localhost only by default. See Security below.
* Editing `.env` by hand and running from the command line or Unraid User Scripts work exactly as before.

### 🧭 New defaults and guides
* **`EVAL_PROFILE=funnel`** is the default and the recommended profile (was `quick`). Every model answers `quick`; confidence gates drop the clearly weaker ones; the rest go on up to `EVAL_FUNNEL_FINAL`.
* **`EVAL_TEMPERATURE=server`** and **`TEMPERATURE=server`** are the defaults (were 0 and 0.2): no temperature is sent, so every model uses the sampling in its llama-swap config, as you deploy it. Put each model's recommended sampling in its llama-swap command.
* **`ESTIMATE_GPU_BW_GBS=auto`**, now in PART 1 (was 900 in PART 3). It looks up the detected NVIDIA GPU in a table of 22 common cards (RTX 3060 to 5090), else uses 670 GB/s; a number overrides it.
* **Guides on the settings page:**
  * `EVAL_PROFILE`: why the funnel, how it works, the margin of error per test for every profile, and an Estimate button;
  * `EVAL_FUNNEL_FINAL`: the stages and the finalists' margin for each choice;
  * `ESTIMATE_GPU_BW_GBS`: the GPU table (click a row to use it);
  * both temperatures.
* A **Batch probe** button starts `--batch-probe` from the page. `--batch-probe` now writes a log (`batch_probe_<stamp>.log`) and the run lock, like a run.

### 🧰 Command line
* `--env FILE`: use another complete settings file instead of `.env` (works with every flag).
* `--validate`: check that the settings load; prints OK or the error.
* `--ui`: the settings page.
* A run now marks itself in `OUTPUT_DIR/.benchmark_run.json`, so the settings page can see and stop it. A second run started on the same GPU prints a warning.

## 🔧 Fixes
* **llama-swap on another machine:** `--check`, the start of a run with speed tests, and the settings page warn when `LLAMA_SWAP_URL` points at another machine. The speed test reads GPU, VRAM, power and RAM locally, so it must run on the machine with the GPU; evals-only runs are fine from elsewhere. The page asks before starting a speed run.
* **`--reload`, `--compare` and runs no longer fail when the "latest" copy next to `benchmark.py` can't be written** (e.g. root-owned on a share). They warn and keep the dashboard in `results/`; the settings page opens the newest one. The page's "Open the combined dashboard" link after Compare now works.
* **A model that fails to load** is flagged **Failed to load (cause)** in the log and on the dashboard (badge with the full hint). Causes: out of memory, bad launch flag, model file not found, unsupported model, load timed out, or not reported. Each comes with its fix; out of memory suggests `--fit on` (with `-c` / `-ngl` removed) or a lower `-c`. It shows llama-swap's own error lines and, when the GGUF is readable, its size, trained context and KV cost.
* **`--reload` / `--compare` of an older run.** Rescoring a v2.0 JSON could crash on a stale `eval_quality`; `finalize()` now clears it first.
* **Docs: interrupted runs.** The v2.1 release notes said an interrupted run can't be reloaded. A run stopped with Ctrl+C writes its JSON, CSV and dashboard with the models finished so far, and `--reload` works on it. Only a hard kill (`kill -9`, a closed terminal, a power cut) leaves nothing to reload. The v2.1 notes and the README are corrected.
* `GOLDILOCKS_MODE` is now checked (`linear` or `log`); a typo used to fall back silently.
* `UI_HOST` written as a URL (`http://192.168.2.200`) failed with "Name or service not known". It is now cut down to the address with a note, `--check-env` and the settings page flag it, and the error names the real cause: not this machine's address, port in use, or port not allowed. The LAN token is saved to `.env` only once the page has opened.
* `--demo` wrote its synthetic results into `results/`: `results.json`, a `benchmark_<stamp>.json` and the latest dashboard. A following `--reload` rebuilt the demo instead of the last real run, and `--estimate` and the settings page read the demo files. The demo now writes to `results/demo/` and leaves the latest dashboard alone.
* New `requirements-dev.txt` (the normal requirements plus pytest) for running the self-tests.

## 🔒 Security (settings page)
* Binds `127.0.0.1` unless `UI_HOST` is set. For the LAN, a token is required: it is generated on the first start, saved as `UI_TOKEN`, and kept as an HttpOnly, SameSite=Strict cookie after the first visit.
* Requests whose `Host` isn't an IP address, `localhost` or this machine's name are refused (DNS rebinding).
* Every change needs the page's own header and a matching `Origin`, so other websites in your browser can't drive it.
* Only a fixed set of the tool's own commands can run. The page never takes a program or a file path, and runs to compare are picked from its own list.
* Unsaved settings for Estimate go to a private temporary file (mode 0600) that is deleted straight after.
* Plain HTTP: on the LAN the token, and any token you reveal, travel unencrypted. Fine on a home network; use a reverse proxy with HTTPS for more, and don't forward the port to the internet.

## ⚖️ How old and new results compare
**Measurements and the result cache are unchanged** as long as the eval settings are. `GRADER_VERSION` and the cache key are untouched, so cached answers are reused. **One exception: the new default `EVAL_TEMPERATURE=server`.** A `.env` that sets `EVAL_TEMPERATURE=0` (as the v2.2 `.env.example` did) keeps every cached answer. Switching to `server` changes every eval request, so every question is asked again, and `--compare` won't combine the new run with temperature-0 runs (`--force` overrides). Scores under sampling also vary a little more from run to run; the margin of error still applies. Two things do take more time:
* Tool calling now asks all 32 cases at every profile, so a `quick` run asks 21 more short questions per model.
* Prefill at depth adds a 4k follow-up on a cached prompt per model (up to 64k). That prompt doubles as the prefill test of the same length when one is set, and it is sent on its own otherwise. Roughly 10–65 s per model, depending on the GPU and the depth.

**The Agent score changes**, and with it the Agent podium and the Sweet spot. `--compare` refuses to mix runs from before and after v2.3 (`--force` overrides). The old scores were rescored with the v2.2.1 code, which reproduced every stored Agent score exactly, and with v2.3. Neither run had prefill at depth, so the speed factor uses the cold agent TTFT for both.

**Run of 9 Nov (20 models, `funnel`).** Composite: evals 45% · tools 30% · long-ctx 25% (64k, the length every ranked model ran).

| New # | Old # | Model | Loaded | Usable | LC 64k/128k | Tools % | Old Agent | New Agent | Δ | Ctx factor |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | Groot-19B-A3B-261023-Q4_K_M | 128k | 128k | 100/100 | 97 | 85.9 | 93.8 | +7.9 | 1.00 |
| 2 | 1 | Ornith-1.5-35B-A3B-AD-IQ3_XXS-IQ2_S | 90k | 90k | 100/– | 97 | 94.3 | 92.0 | −2.3 | 0.98 |
| 3 | 3 | gemma-4-26B-A4B-it-UD-IQ3_XXS | 128k | 128k | 100/88 | 100 | 88.6 | 88.8 | +0.2 | 1.00 |
| 4 | 5 | Ornith-1.5-35B-A3B-AD-Q5_K-IQ4_XS | 128k | 128k | 100/100 | 100 | 87.2 | 86.5 | −0.7 | 1.00 |
| 5 | 12 | Ornith-1.5-35B-A3B-Q4_K_M | 128k | 128k | 100/100 | 97 | 81.0 | 86.2 | +5.2 | 1.00 |
| 6 | 4 | Qwen3.8-27B-GSQ-RCO-IQ3_XXS | 90k | 90k | 100/– | 100 | 87.3 | 85.1 | −2.2 | 0.98 |
| 7 | 7 | Qwen3.8-27B-UD-IQ3_XXS | 90k | 90k | 100/– | 100 | 86.7 | 84.7 | −2.1 | 0.98 |
| 8 | 15 | BigBang-v1-35B-A3B-IQ3_XXS | 128k | 128k | 100/100 | 96 | 73.2 | 83.0 | +9.8 | 1.00 |
| 9 | 13 | BigBang-v1-35B-A3B-Q4_0 | 128k | 128k | 100/100 | 96 | 75.2 | 82.9 | +7.7 | 1.00 |
| 10 | 14 | BigBang-v1-35B-A3B-Q4_K_M | 128k | 128k | 100/100 | 96 | 73.8 | 81.3 | +7.6 | 1.00 |
| 11 | 2 | LFM2.5-24B-A2B-UD-Q4_K_XL | 128k | 64k | 88/12 | 100 | 90.0 | 78.2 | −11.8 | 0.95 |
| 12 | 11 | Wicked-Stuff-26B-A4B-v3.0-Final.i1-IQ3_XS | 128k | 64k | 88/50 | 100 | 81.2 | 73.2 | −8.0 | 0.95 |
| 13 | 16 | GLM-4.7-Flash-30B-A3B-UD-Q4_K_XL | 96k | 96k | 100/– | 97 | 63.0 | 73.1 | +10.1 | 0.98 |
| 14 | 17 | GLM-4.7-Flash-30B-A3B-UD-IQ3_XXS | 64k | 64k | 100/– | 97 | 62.4 | 72.5 | +10.0 | 0.95 |
| – | 6 | flint-40B-A2.5B-UD-IQ3_XXS | 128k | – | 50/38 | 92 | 86.7 | no score¹ |  | – |
| – | 9 | LFM2.5-24B-A2B-UD-IQ3_XXS | 128k | – | 62/12 | 97 | 83.4 | no score¹ |  | – |
| – | 10 | Qwen3.6-35B-A3B-UD-Q4_K_M | 128k | – | –/– | 97 | 81.3 | no score¹ |  | – |

¹ No Agent score: flint (50% at 64k) and LFM2.5 IQ3_XXS (62% at 64k) are below `AGENT_CTX_PASS_PCT` 75%. Qwen3.6-35B-A3B-UD-Q4_K_M has no long-context result: llama-swap went down during its turn, so it is not ranked until it is rerun (the cache keeps its other answers).

**Run of 7 Nov (v2.0 data).** Every ranked model ran 64k, so the composite is evals 45% · tools 30% · long-ctx 25% over 64k.

| New # | Old # | Model | Loaded | Usable | LC 64k | Old Agent | New Agent | Δ |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | Groot-19B-A3B-261023-Q4_K_M | 128k | 128k | 100 | 83.6 | 90.7 | +7.0 |
| 2 | 1 | Ornith-1.5-35B-A3B-AD-IQ3_XXS-IQ2_S | 64k | 64k | 100 | 94.7 | 89.7 | −5.0 |
| 3 | 3 | Qwen3.6-35B-A3B-UD-IQ3_XXS-MTP | 128k | 128k | 100 | 79.7 | 83.4 | +3.7 |
| 4 | 4 | Empero-Qwen3.8-9B-Q4_K_M | 88k | 88k | 100 | 65.0 | 69.2 | +4.2 |
| 5 | 6 | GLM-4.7-Flash-30B-A3B-UD-Q4_K_XL | 96k | 96k | 100 | 56.5 | 62.4 | +5.8 |
| 6 | 5 | Nemotron-3.5-Lightning-30B-A3B-Q4_0 | 128k | 64k | 100 | 63.6 | 60.0 | −3.7 |
| 7 | 7 | GLM-4.7-Flash-30B-A3B-UD-IQ3_XXS | 64k | 64k | 100 | 56.1 | 59.4 | +3.2 |
| – | 8 | LFM2.5-24B-A2B-UD-IQ3_XXS | 128k | – | 0 | 10.3 | no score¹ |  |
| – | 9 | LFM2.5-24B-A2B-UD-Q4_K_XL | 128k | – | 0 | 8.2 | no score¹ |  |

¹ Both LFM2.5 builds found 0% at 64k in the v2.0 run (a harness artefact fixed in v2.1: replies were all reasoning), so they are below `AGENT_CTX_PASS_PCT`.

**What moved, and why it is honest:**
* **Long context that doesn't work no longer counts.** LFM2.5 Q4_K_XL (#2 → #11, −11.8) and Wicked (−8.0) are loaded at 128k but find only 12% and 50% at 128k, so they rank as 64k models. flint and LFM2.5 IQ3_XXS fail even 64k and get no Agent score; under v2.2 they ranked #6 and #9.
* **90k models lose only a little.** Ornith IQ3_XXS-IQ2_S and both Qwen3.8-27B at 90k lose about 2 points (factor 0.98) and stay near the top. Ornith at 64k in the older run loses 5.0 (0.95).
* **Models with full long-context retrieval move up.** Groot (#8 → #1), Ornith Q4_K_M (+5.2) and the three BigBangs (+7.6 to +9.8) all find 100% at 64k and 128k. In v2.2 long-context was left out of every score in this run because one model lacked it. Now it counts for 25%, and these models' composites sit closer to the best. GLM-4.7-Flash (+10) gains for the same reason: 100% at 64k. Its usable context is 96k / 64k, so it is ranked behind models with similar quality at 128k.
* **No model gains from missing data.** A model without a long-context result is not ranked, instead of everyone losing the long-context part. In v2.2 a 90k model's 64k-only average (100%) beat a 128k model's 64k + 128k average; now everyone is compared on the same lengths, and failing a longer length lowers the usable context.
* **Small changes** of up to ±1 (gemma-4, Ornith Q5_K) come from the reweighting (tools 30%) and the composite-based tie zone, within the scores' own noise. Models equal on quality are still separated by speed, by at most 30%.

**Defaults agreed for this release:** `AGENT_CTX_MIN=65536`, `AGENT_CTX_PASS_PCT=75`, `AGENT_CTX_CURVE=131072:1.0,78848:0.97,65536:0.95` (agreed 2026-11-10: usable context, mild preference), `AGENT_TOOL_WEIGHT=0.30`, `AGENT_LONGCTX_WEIGHT=0.25` (unchanged), `EVAL_PROFILE=funnel` (was quick), `EVAL_TEMPERATURE=server` (was 0), `TEMPERATURE=server` (was 0.2), `ESTIMATE_GPU_BW_GBS=auto` (was 900), `MIN_TOOLCALL_PCT=70`, `AGENT_SPEED_WEIGHT=0.3` (unchanged), speed factor 0.5 / 0.3 / 0.2, `PREFILL_DEPTH=auto`, `COLOUR_STYLE=bands`, `COLOUR_MODE=absolute`, `UI_PORT=41010`, localhost only.

Other display changes: the retry warning (⚠) now shows above 25% retried (was 20%), the red tier of the new retry scale; set `COLOUR_RETRY` to change it.

## 🔄 Updating
1. Copy the changed files over your install. `llmbench/ui/`, `llmbench/agent.py`, `llmbench/envfile.py`, `llmbench/gpubw.py`, `llmbench/reporting/colours.py` and `configs/` are new.
2. Run `python benchmark.py --check-env`. It lists the new PART 1 settings `AGENT_CTX_MIN`, `AGENT_CTX_PASS_PCT` and `AGENT_CTX_CURVE`, and the PART 2 setting `PREFILL_DEPTH`. Or run `python benchmark.py --ui`: the page shows the same list, and **Save** adds them for you (your old file stays as `.env.bak`).
3. Decide whether 64k is the right minimum for your agents. If a model has to run at 32k (a large, smart model that only fits that way), set `AGENT_CTX_MIN=32768` and add a `32768:…` point to `AGENT_CTX_CURVE`; with long-context tests in the run, it must still pass every length it can hold.
4. **Sampling:** your `.env` keeps whatever `EVAL_TEMPERATURE` it has. To move to the new default, set `EVAL_TEMPERATURE=server` once each model's llama-swap command carries its recommended sampling; expect one full re-ask of the evals.
5. `--reload` of an older run keeps the scores it stored. `--compare` of two older runs rescores both with the v2.3 rules (without prefill at depth, which they didn't measure).

## ⚙️ Configuration
* **New in PART 1:** `AGENT_CTX_MIN`, `AGENT_CTX_PASS_PCT`, `AGENT_CTX_CURVE`; `ESTIMATE_GPU_BW_GBS` moved here from PART 3.
* **Changed defaults:** `EVAL_PROFILE` funnel, `EVAL_TEMPERATURE` and `TEMPERATURE` server, `ESTIMATE_GPU_BW_GBS` auto (see above).
* **New in PART 2:** `PREFILL_DEPTH`.
* **New in PART 3:**
  * `AGENT_TOOL_WEIGHT`, `MIN_TOOLCALL_PCT` (Agent score);
  * `COLOUR_STYLE`, `COLOUR_MODE`, `COLOUR_<METRIC>` (GEN, PREFILL, ACC, SCORE, TTFT_MS, AGENT_TTFT, DEPTH_TTFT, SQ, RETRY, POWER), `DASH_FAMILIES` (dashboard view);
  * `UI_PORT`, `UI_HOST`, `UI_TOKEN` (settings page).
* `.env.example` lists 153 settings plus three families (141 before). `--check-env` and the settings page know the allowed values of every setting that has a fixed set.
* **New command-line options:** `--ui`, `--env FILE`, `--validate`.
* No third-party code or data was added. The settings page uses only Python's standard library; the dashboard still loads Chart.js from cdnjs.
