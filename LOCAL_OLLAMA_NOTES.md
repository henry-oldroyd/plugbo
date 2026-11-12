# Local Ollama experiments: what has been done so far

Branch: `ollama-local` (fork: `henry-oldroyd/plugbo`, upstream: `richardcsuwandi/plugbo`).
Written 2026-11-12 from the 4090 machine, for picking this up on the 2x3090 machine.
This is an independent re-implementation of Meta's agentic BO paper (see README); the
notes below are only about running it against local Ollama models.

## 1. What was added (nothing in the BO logic was changed)

| File | Purpose |
|---|---|
| `scripts/_ollama_env.sh` | Sourced by the other `run_ollama_*.sh`. Sets `PROVIDER=ollama`, `MODEL` (default `qwen3-coder:30b`, override with `OLLAMA_MODEL`), `BASE_URL` (override with `OLLAMA_BASE_URL`). Preflight: refuses to run under WSL bash, checks Python modules import, checks the Ollama server and model exist, sends a tiny test completion. Shims `python3` to `python` on Windows. Exit trap unloads the model from VRAM. `SKIP_PREFLIGHT=1` bypasses the checks. |
| `scripts/run_ollama_synthetic.sh` | One benchmark (default `hartmann6`), budget 20, backends `vanilla,sara-lenz`. |
| `scripts/run_ollama_sweep.sh` | branin + hartmann6 x blind + revealed-shift. |
| `scripts/run_ollama_bolt.sh` | BoLT LoRA experiment (needs `pip install -e '.[bolt]'`). |
| `scripts/run_ollama_smoke.sh` | Quick check: budget 12, seed 42, branin + hartmann6. Writes to `results/logs/ollama-smoke`. |
| `scripts/run_ollama_batch.sh` | Many seeds x benchmarks. Env: `SEEDS`, `BENCHES`, `BUDGET` (default 50), `BACKENDS`, `DISCLOSURES`, `OUT`. Finished runs are skipped, so Ctrl-C and rerun resumes. Writes to `results/logs/ollama-batch/<model>/<bench>-<disclosure>/seed<N>`. |
| `scripts/ollama.ps1` | PowerShell launcher that runs the above under Git Bash (not WSL): `.\scripts\ollama.ps1 smoke \| batch \| synthetic \| sweep \| bolt`. |
| `scripts/ollama_pipeline.ps1` | Full Windows pipeline: restart Ollama with context 32768 + flash attention + q8 KV cache, pull model, start `plugbo-viz` on :8765, smoke test, batch, final report. Foreground by default (Ctrl+C aborts). `-Background` exists but is not recommended. |
| `llm/ollama_unload.py` | `unload_model()` (never raises) and `python -m llm.ollama_unload MODEL` to free VRAM, same as `ollama stop`. |
| `.gitattributes` | `*.sh text eol=lf`. |

## 2. Fixes to existing code (all needed on Windows)

- **Line endings:** `core.autocrlf=true` had turned the repo's `.sh` scripts into CRLF, which broke
  the shebang (`bash\r`). The five existing scripts were converted back to LF and `.gitattributes` added.
  They show as modified in git only because of this.
- **Symlink (`benchmarks/sandbox.py`):** `Path.symlink_to` fails on Windows without Developer Mode
  (WinError 1314). It now falls back to copying the oracle file into the sandbox.
- **Agent bash tool (`sara/tools.py`):** it used `subprocess.run(shell=True)`, i.e. `cmd.exe` on Windows,
  where `./oracle '<json>'` cannot work. On Windows it now runs commands under Git Bash
  (`C:/Program Files/Git/bin/bash.exe`, or the `PLUGBO_BASH` env var). Linux/macOS behaviour is `/bin/sh -c`, as before.

## 3. Gotchas hit on the 4090 Windows machine

- PowerShell's `bash` is **WSL**, which uses a different Python with none of the packages. Use Git Bash
  (`& "C:\Program Files\Git\bin\bash.exe" ...`) or `scripts/ollama.ps1`.
- `python3` on Windows is a Microsoft Store stub; the env script shims it.
- Running `./x.sh` in PowerShell opens it in an editor instead of running it.
- `ollama_pipeline.ps1 -Background` first failed because the script path contains spaces and was unquoted
  (fixed). The Ollama readiness check also timed out once at 30 s; it now waits up to 2 minutes on `127.0.0.1`.
  Both fixes were pushed but the background mode itself was not re-tested.
- **Ollama settings used:** `OLLAMA_CONTEXT_LENGTH=32768`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`.
  The pipeline script sets these for its own `ollama serve`; on a plain Ollama install set them yourself and restart it.
- There is no command that purges VRAM in general. Only `ollama stop <model>` (or the unload helper) for Ollama's own use.

## 4. What was actually run (on the 4090, 24 GB)

All runs: benchmark `hartmann6` (6-D, minimise, true optimum -3.3224), blind disclosure, seed-shared
Sobol warm-up of 7 points. "Regret" = best value found minus the true optimum, so lower is better.

**Run 1: `qwen3-coder:30b`, budget 20, seed 42 (default context)**

| Backend | Evals used | Best regret |
|---|---|---|
| vanilla | 20/20 | 1.13 |
| sara-lenz | 14/20 (agent stopped early) | 1.16 |

The best point in both was the warm-up point #7; neither method improved on it meaningfully. The sara-lenz
agent's final report claimed it "completed the budget" and "successfully identified a strong local minimum",
which was not true.

**Run 2: `qwen3.6:27b`, smoke test (budget 12) and then batch (budget 50), seeds 42-44, context 32768**

The pipeline script was run in the foreground. Smoke test: all four runs (branin and hartmann6, vanilla and
sara-lenz) completed. Batch results at the time of writing:

| Seed | Backend | Evals used | Best regret | Status |
|---|---|---|---|---|
| 42 | vanilla | 50/50 | 0.045 | completed |
| 42 | sara-lenz | 13/50 | 1.16 | **stopped early** |
| 43 | vanilla | 50/50 | 1.20 | completed |
| 43 | sara-lenz | 14/50 | 1.71 | **stopped early** |
| 44 | vanilla | 50/50 | not summarised | completed |
| 44 | sara-lenz | in progress | not yet | running |

Numbers are from `summary.json` (made by `python -m benchmarks.summarize_compare --root <seed dir>`);
seeds 43 and 44 were still running when this was written.

## 5. What this does and does not show

- It works end-to-end locally: preflight, smoke test and batch run, results readable in `plugbo-viz`.
- **The sara-lenz agent stops at about 13-14 evaluations out of 50 on both models**
  (and at 14/20 in run 1). With a 32K context and the bigger model it still happened, so a too-small
  context is **not** confirmed as the cause. Why the agent ends its run early is unexplained. Read
  the agent trace in `plugbo-viz` for a stopped run before changing models again.
- Because of that, vanilla vs sara-lenz is not a fair comparison yet: vanilla used its full 50 evaluations,
  the agent about 14. Three seeds is also far too few to say anything about which method is better.
- The "larger model" run on the 3090 machine has not been tried, and Qwen3.6-27B was the only candidate checked.
  The model recommendations came from third-party web articles and were not verified.

## 6. Setting up the 2x3090 machine

```bash
git clone -b ollama-local https://github.com/henry-oldroyd/plugbo.git
cd plugbo
python -m pip install -e .          # add '.[bolt]' for the LoRA experiment
```

Then:
1. Install Ollama; pull the larger model. Set the three `OLLAMA_*` variables above and restart Ollama.
2. Ollama splits a model over both GPUs automatically. Alternatively run two servers (one per GPU, ports 11434
   and 11435) with different `CUDA_VISIBLE_DEVICES`/`OLLAMA_HOST`, and point each batch at one via `OLLAMA_BASE_URL`.
   Give each batch different seeds, so they do not write to the same folders.
3. Smoke test, then batch (Linux/macOS):

```bash
export OLLAMA_MODEL=<big-model>
./scripts/run_ollama_smoke.sh
SEEDS="42 43 44 45 46" BENCHES="hartmann6" BUDGET=50 ./scripts/run_ollama_batch.sh
plugbo-viz --root results/logs/ollama-batch
```

On Windows use `.\scripts\ollama_pipeline.ps1 -Model <big-model>`; check its `-Context` default for a
two-GPU setup. The Git Bash path and `python3` shim only matter on Windows.

`.env` and `results/` are gitignored, so run logs stay on each machine. To compare across machines, copy the
`results/logs/ollama-batch/<model>` folders by hand.
