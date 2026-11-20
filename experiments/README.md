# Local-LLM experiments on PlugBO: index and handover

This folder documents a series of experiments run on top of PlugBO (an independent re-implementation of Meta's
agentic Bayesian-optimization paper, arXiv 2608.00316) using **local models served by Ollama**. It is written so
that another person or agent can read it, re-run everything, and extend it. The upstream repository's own docs are
in `README.md` and `docs/`; this folder only covers what was added on the `ollama-local` branch.

## The question

On a blind Hartmann-6 problem (renamed parameters, unit cube, shifted optimum, budget 50 evaluations, 7 shared Sobol
warm-up points), does an LLM agent (Sara) driving the BoTorch backend (lenz) find better points than the plain
backend (vanilla BO)? Regret is the best value found minus the hidden true optimum (-3.32237); lower is better.

## The experiments

| # | Doc | What differs | Status |
|---|---|---|---|
| 1 | [01-pinned-vs-vanilla.md](01-pinned-vs-vanilla.md) | Agent may not change the surrogate/acquisition (pinned). Model `qwen3.6:27b` on one RTX 4090. 65 seeds. | Done. Vanilla clearly better (final regret 0.33 vs 1.36). |
| 2 | [02-free-reconfiguration.md](02-free-reconfiguration.md) | Agent may reconfigure acqf, surrogate (incl. CAKE), bounds, objectives, constraints, plus a vague true hint. Same model/GPU. | Running at time of writing (seeds 42, 43, ...). |
| 3 | [03-dual-gpu-parallel.md](03-dual-gpu-parallel.md) | Many seeds in parallel on 2 GPUs (layer split), `qwen 32B`, single-writer orchestrator. | Code written and self-tested with fake workers; **not yet run on real GPUs**. |

## Conditions and names you will see

* `vanilla` ("LENZ-ONLY" in the viewer): BoTorch loop, fixed Matern GP, `noisy_logei`, no LLM.
* `sara-lenz`: agent, optimizer pinned (experiment 1).
* `sara-lenz-free`, `sara-lenz-free-all`, `sara-lenz-free-hint`: agent may reconfigure (experiment 2/3).
  `-all` also allows the region/sampler/belief plugin verbs; `-hint` adds the vague true hint.
* "forced continuation": our fix (`SARA_ENFORCE_BUDGET=1`, `sara/agent.py`). When the model ends its turn with no
  tool call while evaluations remain, the harness sends "budget not exhausted, keep going". Capped at 30 per run.
  Without it the agent ended runs after 7-17 of 50 evaluations (empty replies were treated as a final report).

## Setup (either machine)

```bash
git clone -b ollama-local https://github.com/henry-oldroyd/plugbo.git && cd plugbo
python -m pip install -e .          # torch, gpytorch, botorch, openai, ...
python -m pip install matplotlib    # for the plots (already present on the 4090 machine)
ollama pull <model>                 # e.g. qwen3.6:27b (experiments 1-2) or a 32B model (experiment 3)
```

Windows gotchas (all already handled in the scripts, but worth knowing):
* PowerShell's `bash` is **WSL**; the scripts need **Git Bash** (`C:\Program Files\Git\bin\bash.exe`). Use
  `scripts/ollama.ps1` or call Git Bash explicitly.
* `core.autocrlf=true` turns `.sh` files into CRLF and breaks the shebang; `.gitattributes` forces LF.
* `python3` is a Store stub; the scripts shim it to `python`.
* Symlinks need privileges; `benchmarks/sandbox.py` falls back to copying the oracle.
* The agent's bash tool uses `cmd.exe` by default; `sara/tools.py` now runs Git Bash on Windows.

## Where things are

| Path | Contents |
|---|---|
| `scripts/_ollama_env.sh`, `run_ollama_*.sh`, `ollama*.ps1` | Experiment 1/2 runners (see docs 01, 02) |
| `scripts/ollama_free_loop.ps1` | Experiment 2 seed loop (Windows) |
| `scripts/dual_gpu/` | Experiment 3 (cross-platform Python): capacity calculator, orchestrator, plot, Ollama launcher, self-test |
| `benchmarks/run_blind_test.py` | Agent runner; `--freedom`, `--hint` added for experiment 2/3 |
| `sara/agent.py` | `SARA_ENFORCE_BUDGET` (forced continuation); `sara/tools.py`: `SARA_BASH_TIMEOUT`, Git Bash on Windows |
| `results/` | **Git-ignored.** Run data, logs, figures. Copy between machines by hand. |

## What is and is not committed

Code and docs are committed. `results/`, `.env` and generated PDFs are not. Run data from the 4090 machine
(`results/logs/ollama-batch`, `results/logs/ollama-free`, ...) exists only there.

## Open questions worth knowing

1. Why does the agent return empty replies (no tool call)? First occurrence is at a median of evaluation 10, so a
   full context window is not the whole story (see 01). Not yet tested across models.
2. Does reconfiguration freedom (experiment 2) change the picture? In a short smoke test the agent used `set-acqf`.
3. Does a larger/better-behaved model (experiment 3) close the gap to vanilla BO?
