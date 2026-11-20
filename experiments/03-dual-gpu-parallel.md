# Experiment 3: many seeds in parallel on two GPUs (layer split), 32B model

**Status: code written and self-tested with fake workers (no GPU). Never run on real GPUs.** The first thing to do
on the 2-GPU machine is the "First run" checklist below.

Goal: get many more seeds of "vanilla BO vs LLM agent" per hour than the one-seed-at-a-time loop of experiments 1-2,
using a 32B Qwen model across two RTX 3090s (2 x 24 GiB), with a design in which a single process owns every log and
the figure.

## 1. The idea: layer (pipeline) split with shared weights

* The model's layers are divided between the GPUs: GPU 0 holds the first half of the layers **and the matching half
  of every KV cache**, GPU 1 the second half. This is llama.cpp's default multi-GPU mode (`split-mode layer`), which
  Ollama uses when a model spans GPUs.
* Only each token's **hidden state** crosses the GPU boundary (hidden size x 2 bytes, about 10 KiB per token for a
  5120-wide model). PCIe is enough; NVLink is not needed.
* Weights are loaded **once** and shared by all concurrent requests. Each concurrent agent is a "slot" with its own KV
  cache (`OLLAMA_NUM_PARALLEL=N`, per-slot context `OLLAMA_CONTEXT_LENGTH`). So memory = weights + N x KV(ctx).
* `OLLAMA_SCHED_SPREAD=1` forces the spread over all GPUs even if the model would fit on one.

## 2. How many instances fit: `scripts/dual_gpu/dg_capacity.py`

```
usable  = gpus * (vram * (1-safety) - overhead - desktop)       # overhead 1.2 GiB, desktop 0.5 GiB per GPU (defaults)
free    = usable - weights
kv_slot = 2 * layers * kv_heads * head_dim * bytes_per_element * ctx
max     = floor(free / kv_slot)                                  # comfortable = floor((1-0.15) * free / kv_slot)
```

KV element size: f16 = 2 B, q8_0 = 34/32 B, q4_0 = 18/32 B. Verified architecture for Qwen2.5-32B-class models
(read from Ollama `/api/show`): 64 layers, 8 KV heads, head dim 128 => **256 KiB/token f16, 136 KiB/token q8_0**.
Qwen3-32B has the same shape (from memory; confirm with `--model` once pulled).

```bash
python scripts/dual_gpu/dg_capacity.py --model qwen3:32b            # reads Ollama + nvidia-smi
python scripts/dual_gpu/dg_capacity.py --preset qwen3-32b-q4 --ctx 16384 --kv-type q8_0
```

Example (2 x 24 GiB, ~18.8 GiB Q4 weights; 24.4 GiB left for KV caches). Slots as max / comfortable:

| context | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 8K | 12 / 10 | 22 / 19 | 43 / 36 |
| 16K | 6 / 5 | 11 / 9 | 21 / 18 |
| 32K | 3 / 2 | 5 / 4 | 10 / 9 |
| 64K | 1 / 1 | 2 / 2 | 5 / 4 |

With 32.5 GiB Q8 weights (the `deepseek-r1:32b-qwen-distill-q8_0` profile in `scripts/_gpu_profile.sh`), 16K q8_0 gives
5 / 4. The calculator also sizes the CPU side: BO workers needed to stay ahead of the agents
(`ceil(agents * t_bo / t_agent)`, with placeholder run times of 10 and 30 minutes -- **measure and pass real ones**)
versus workers that fit (`(cpus - 2 - agents*2) // bo_threads`).

Caveats:
* Throughput does not scale linearly with slots: the GPUs' compute is shared. `--agents auto` therefore caps at 6.
  Memory says how many *fit*; only measurement says how many are *worth running*. Start with 3-4.
* An **untrimmed** agent history grows ~1.3K tokens per evaluation (~69K tokens for 50 evaluations). A 16K window
  will overflow around evaluation 10-12. Either use a large window with fewer slots (64K q8_0 = 2 slots) or add
  history trimming to `sara/agent.py` (not implemented). Experiment 1 saw the *first* forced continuation at a
  median of evaluation 10 (not necessarily caused by overflow; unresolved), so check the forced-continuation markers.
* Hybrid models (Qwen3.5/3.6, arch `qwen35`) have a smaller KV and a per-slot recurrent state; the calculator's
  standard-attention formula is then an upper bound.

## 3. Architecture: one writer, no locks (`scripts/dual_gpu/orchestrator.py`)

```
BO workers (CPU subprocesses)  ─┐
                                ├─ stdout pipes → supervisor coroutines → asyncio.Queue → WRITER coroutine
Agent workers (GPU, via Ollama)─┘                 (also summarise the                      (only code that writes
                                                   finished run, read-only)                 logs, csv, status, figure)
```

* A worker is `python -m benchmarks.run_blind_baseline` (BO) or `python -m benchmarks.run_blind_test` (agent). It
  writes only inside its own run directory `<out>/runs/seed<N>/{vanilla,agent}/`. No two workers share a directory.
* Workers never see the log or figure paths. They "send their results" by exiting; the supervisor then reads the
  finished run directory read-only (status, regret trace from the hidden answer key, first forced continuation) and
  puts one message on the queue. The writer coroutine is the **only** writer of `events.log`, `workers/*.log`,
  `results.csv`, `failures.jsonl`, `status.json` and the figure. Single process + single thread of control => no file
  locks are needed. `status.json` and the PNG are written to a temp file and renamed (`os.replace`) so a viewer never
  sees a half-written file.
* **The figure is re-rendered only when a seed is paired** (its BO run and its agent run both completed), plus once at
  start-up if a resumed session already has pairs. A seed where either run failed is **excluded** (equal seeds on every
  line); failures are counted in `failures.jsonl`, `status.json` and the figure footer. A snapshot per paired-seed
  count is kept in `figures/snapshots/`.
* **Logs update throughout:** every start/finish/failure is logged immediately, and every `--heartbeat` seconds
  (default 30) `status.json` and `events.log` get a heartbeat with each running run's evaluation count and GPU
  utilisation/memory (`nvidia-smi`).
* **BO stays ahead:** BO workers pull seeds in order and may start seed `s` only while `s < next_agent_seed +
  --bo-lead` (default 3 x agents), so BO results are normally waiting when the agent for that seed finishes.
* Resume: completed runs are reloaded; unfinished ones are *moved* to `<out>/_stale/` (never deleted).
* Stop: Ctrl+C / SIGTERM kills workers' process trees, marks them `INTERRUPTED` (not counted as failures), and unloads
  the model. `--hours H` stops *starting* new seeds after H hours and lets in-flight runs finish.

### The figure (`dg_plot.py`)

Mean best-so-far regret +/- 1 standard error across paired seeds for both conditions, faint per-seed trajectories,
an **X on each agent trajectory at its first forced continuation** (the evaluation count when the harness first had
to tell the model to keep going), and a **dashed vertical line at the median** of those. The first-forced index is
derived from `trace.jsonl` (`forced_continuation` marker timestamps) vs the trials' `observed_at`, so the agent code
is unchanged.

## 4. Running it

```bash
# 0. install + pull
python -m pip install -e . matplotlib && ollama pull qwen3:32b

# 1. size it (prints the table above for your model and GPUs)
python scripts/dual_gpu/dg_capacity.py --model qwen3:32b --ctx 16384 --kv-type q8_0

# 2. start a separate layer-split Ollama on :11435 (does not touch an existing :11434 server)
python scripts/dual_gpu/start_ollama.py --agents 4 --ctx 16384 --kv-type q8_0 --model qwen3:32b --check
#    ^ --check loads the model, sends one request and prints per-GPU memory. BOTH GPUs must hold a share.
python scripts/dual_gpu/start_ollama.py --agents 4 --ctx 16384 --kv-type q8_0 &      # the real server

# 3. smoke test of the orchestrator logic (no GPU): copies finished runs from experiment 1
python scripts/dual_gpu/selftest.py

# 4. first real run: tiny, to measure real timings
python scripts/dual_gpu/orchestrator.py --model qwen3:32b --agents 2 --bo-workers 2 --max-seeds 2 --budget 20 \
       --out results/dual_gpu_probe

# 5. the real thing
python scripts/dual_gpu/orchestrator.py --model qwen3:32b --agents 4 --bo-workers 3 --hours 12
#    (--agents/--bo-workers accept "auto")
```

Useful flags: `--freedom {pinned,paper,all}` (default `paper`), `--hint {none,vague}` (default `vague`) -- these defaults
scale experiment 2 up; `--freedom pinned --hint none` reproduces experiment 1's condition. `--bench`, `--budget`,
`--start-seed` (42, to pair with experiments 1-2), `--max-seeds`, `--extra-body '{...}'` (pass-through to the LLM
request, e.g. to switch off thinking if the server honours it), `--bash-timeout` (600), `--ctx/--kv-type` (only
used by `auto` sizing; the server's real values come from `start_ollama.py`).

Outputs (default `--out results/dual_gpu`):

```
runs/seed<N>/{vanilla,agent}/       per-run data (state.json, trace.jsonl, run_meta.json, _answers/)
logs/events.log                     timestamped human log (also printed)
logs/status.json                    live snapshot: running runs + eval counts, GPU stats, paired/excluded seeds
logs/results.csv                    one row per finished run (status, n_evals, best_regret, first_forced_idx, ...)
logs/failures.jsonl                 failed runs
logs/workers/<cond>-seed<N>.log     raw worker output
figures/regret_dual_gpu.png         latest figure;  figures/snapshots/regret_nNNN.png  one per paired-seed count
```

## 5. First run checklist (things not verified without the real hardware)

1. `start_ollama.py --check`: confirm both GPUs show memory (layer split working) and `ollama ps` shows the model fully
   in VRAM. If only one GPU is used, `OLLAMA_SCHED_SPREAD` is not taking effect on your Ollama version.
2. Confirm the per-slot context: in the Ollama server log look for `n_slots = N` and `n_ctx_slot = <ctx>` (Ollama
   multiplies the allocation by `OLLAMA_NUM_PARALLEL`). Also check the `KV buffer size` lines match the calculator.
3. Verify the capacity defaults (1.2 GiB overhead/GPU, 0.5 GiB desktop/GPU) against `nvidia-smi` with the model
   loaded; adjust `--overhead-gib/--desktop-gib`.
4. Run the probe (step 4), read `logs/events.log` and `results.csv` for real BO and agent durations, then re-run
   `dg_capacity.py --t-bo-min ... --t-agent-min ...` and choose `--agents/--bo-workers`.
5. Qwen3 models default to thinking mode; if runs are slow or replies empty, try `--extra-body` or a non-thinking tag.
6. On Windows the orchestrator shims `python3` for `./oracle`; the agent's bash tool needs Git Bash (see README).

## 6. Test coverage

`scripts/dual_gpu/selftest.py` runs the real orchestrator with fake workers (`dg_worker_fake.py`) on real experiment-1
data: 6 seeds, 3 agent + 2 BO slots, one injected agent failure. It checks: results rows, exactly one logged failure,
failed seed excluded from the pairs, one figure update per paired seed (5), snapshots, heartbeat lines, no temp files,
per-worker logs, and that no log/figure files appear inside worker run directories. **Not tested:** real Ollama,
real GPUs, real BO/agent workers launched by the orchestrator, Ctrl+C handling on Windows.
