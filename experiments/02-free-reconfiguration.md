# Experiment 2: free reconfiguration (+ optional vague hint)

Follow-up to [experiment 1](01-pinned-vs-vanilla.md). Same benchmark, budget, model (`qwen3.6:27b`) and GPU (one
RTX 4090); the difference is what the agent is allowed to do.

## What changed in the agent's freedom

Experiment 1 appended to the agent's task: "Do not call `set-acqf` or `set-surrogate`... keep them fixed". These were
only *instructions*; lenz does not block them. Experiment 2 replaces that with a directive that **allows**:

* `set-acqf` (noisy_logei, logei, pi, ucb, sobol), `set-surrogate` (`fixed` Matern GP or `cake`, an LLM-evolved
  kernel), `set-bounds`, `set-objectives`, `set-constraints` (`--freedom paper`, the default), and additionally
  `set-region` (TuRBO), `set-sampler` (LLAMBO), `set-belief` (pi-BO prior) with `--freedom all`.

It still forbids `lenz create` (would wipe the shared warm-up), and tells the agent that its score is the best true
`y` (minimized) inside the original [0,1] space, so the objective direction must not change. Nothing prevents it
technically; it is a prompt-level rule, as in experiment 1.

Optional `--hint vague` appends true, generic background (smooth, noise-free, multimodal with several local minima;
no location or scale). The text is `HINTS["vague"]` in `benchmarks/run_blind_test.py`. It is meant to give the agent
something to reconfigure *for* (e.g. noise-free => `logei`; multimodal => explore).

## Code changes (all backward compatible; defaults reproduce experiment 1)

| File | Change |
|---|---|
| `benchmarks/run_blind_test.py` | `--freedom {pinned,paper,all}` (default `pinned`), `--hint {none,vague}`; `_free_directive`; both recorded in `run_meta.json` |
| `scripts/_run_lib.sh` | backends `sara-lenz-free`, `sara-lenz-free-all`, `sara-lenz-free-hint` (blind only) |
| `sara/tools.py` | `SARA_BASH_TIMEOUT` env var (default 120 s) because CAKE/LLAMBO call an LLM inside lenz |
| `scripts/ollama_free_loop.ps1` | the seed loop (below) |
| `scripts/make_free_group.py`, `plot_free_comparison.py`, `log_failures.py` | grouping, plot, failure log |

## How it was run

`scripts/ollama_free_loop.ps1 -HintOnly` (PowerShell, Windows). For each seed it:

1. starts vanilla BO (CPU) as a separate process, **in parallel** with the agent (GPU) for the same seed;
2. runs the agent leg (`sara-lenz-free-hint`) with `SARA_ENFORCE_BUDGET=1` and a 600 s command timeout;
3. records failed runs in `results/logs/ollama-free/failures.jsonl`;
4. rebuilds `results/logs/analysis-free-hartmann6/` (completed runs only) and re-draws
   `results/figures/regret_free_vs_vanilla.png` (+ snapshot per seed count in `results/figures/free_snapshots/`).

Seeds where the agent fails are excluded from every line so all lines use equal seeds; the plot footer shows the
failure count. Experiment 1's pinned-agent runs (`results/logs/ollama-batch`) are read-only and drawn as a reference
line. Other switches: `-Hint` (hint agent *in addition to* the unhinted free agent), `-Freedom all`, `-MaxSeeds`,
`-Hours`, `-SmokeTest` (budget 12, scratch folders). Re-running resumes: unfinished runs are deleted and redone.

Never run two loops at once (same GPU/model), and do not run it alongside `ollama_seed_loop.ps1`.

## Observations so far

* In a 12-evaluation smoke test the agent made no reconfiguration calls.
* In the first full run (seed 42) the agent called `set-acqf logei` (reason given: "smooth, deterministic ...
  no observation noise" -- it used the hint) and later `set-acqf sobol` (no explanation), after which several
  evaluations landed in poor regions. One seed proves nothing; read the figure footer for the running tally of which
  reconfiguration verbs were used.
* The agent's final written report was empty in these runs (the model ends with no text).

## Limitations

Same as experiment 1 (one benchmark, one model, quantized 27B, uncontrolled sampling parameters), plus: the agent can
narrow `set-bounds` so that the optimum is excluded (allowed; regret is still scored against the true optimum);
`set-constraints` is meaningless here (no constraint metric exists on this benchmark).
