# Experiment 1: vanilla BO vs LLM agent on Hartmann-6 (local Qwen, optimizer pinned)

Figure: `results/figures/regret_seed_distribution.png` (regenerate: `python scripts/make_comparison_group.py && python scripts/plot_seed_distribution.py`).
Numbers below were computed 2026-11-20 from the runs on disk (seeds 42-106; runs happened 2026-11-13/14).

## 1. One-paragraph summary

We compared two ways of minimising a blind 6-D test function (Hartmann-6) with a budget of 50 evaluations,
using the PlugBO framework (an independent re-implementation of Meta's agentic-BO paper): **vanilla Bayesian
optimisation** (a fixed algorithm, no LLM) and **an LLM agent** (Qwen3.6 27B running locally in Ollama) that
drives the same BO backend and decides what to evaluate. Both start from the same 7 seeded Sobol points per seed.
The plot shows the mean best-so-far regret across seeds with +/- 1 standard error of the mean. Result:
vanilla ends at mean regret 0.33 (n=65 seeds); the agent ends at 1.36 (n=63 seeds). Vanilla is clearly better here.
The agent also mostly fails to use its whole budget (55 of 63 runs ended early, median 30 evaluations), but even
the 8 runs that used all 50 evaluations did worse than vanilla on the same seeds (1.39 vs 0.42).

## 2. What was tested

- **Benchmark:** Hartmann-6. A smooth 6-D function on the unit cube, sum of 4 negative Gaussian-like bumps, 6 local
  minima, known global minimum -3.32237. Minimise.
- **Blind setup (disclosure = `blind`):** parameters renamed to random 4-letter names (e.g. `zycz`, `xkpw`), space is
  the unit cube [0,1]^6, the optimum location is **shifted**, and the problem text is generic ("an unknown function").
  The agent is told not to read the oracle's source or fit the function. This stops the LLM recalling the textbook
  optimum. The true answer key is stored separately and used only to score regret afterwards.
- **Budget:** 50 evaluations per run. The first **7** are a shared Sobol warm-up (= d+1), seeded, **identical for
  vanilla and the agent on the same seed**, and they count toward the 50.
- **Metric: simple regret.** Best value found so far minus the true optimum (-3.32237), looked up from the hidden key.
  Lower is better; 0 = found the optimum. Plotted as best-so-far vs evaluation number.
- **Seeds:** each seed fixes the Sobol warm-up (and the function shift). 65 vanilla runs (seeds 42-106) and 63 agent
  runs (the same seeds minus 80 and 94, which crashed; see 6).

## 3. The two conditions

**Vanilla BO (`vanilla`, shown in the viewer as "LENZ-ONLY")**

- Plain BoTorch loop through the `lenz` backend. Per step: fit a Gaussian process (fixed Matern kernel) to all
  points so far, choose the next point by maximising the `noisy_logei` (noisy log expected improvement) acquisition,
  evaluate, repeat.
- No LLM. Runs on CPU, a few minutes per run. No agent decisions.

**LLM agent (`sara-lenz`, "Sara + lenz")**

- Same backend and same pinned settings: fixed Matern surrogate, `noisy_logei`, box region, no prior. The agent is
  *forbidden* to change surrogate/acquisition (`set-acqf`, `set-surrogate`).
- Sara is an LLM with two tools, `bash` and `read`. It calls lenz commands (`suggest`, `diagnostics`, `score`,
  `submit`, `observe`, ...) and runs `./oracle '<config-json>'` itself. Its system prompt tells it to use domain
  knowledge, propose its own candidates, and *prefer its own reasoning over lenz's ranking when justified*. So it may
  follow or override the BO suggestion at every step.
- Model: **qwen3.6:27b** via Ollama on one RTX 4090 (24 GB), Q4-class weights (~17 GB). Context length 32768,
  flash attention on, q8 KV cache (set when the pipeline restarted Ollama; context showed 32768 at the start).
  Sampling settings left at Ollama defaults (not tuned or recorded).
- Cost per run (from the traces): about 22 minutes, ~2.5M input tokens (mostly prompt-cache reads), ~20K output tokens.
- Safety caps in the harness: max 400 agent steps, 120 s timeout per bash command.

**Forced budget ("forced to continue") - our modification**

- In the first runs the agent ended after only 7-17 of 50 evaluations. In the two traces inspected, the model's last
  message was an **empty reply with no tool call**, which the loop treated as a final report. (This was not
  verified for every run.)
- Fix, in `sara/agent.py` (`SARA_ENFORCE_BUDGET=1`, on by default in the Ollama batch script): if the model replies
  without a tool call while evaluations remain, drop the empty turn and send a message "budget not exhausted, keep
  going - call a tool". Capped at **30** such nudges per run so a stuck model cannot loop forever.
- Off by default in the code, so original behaviour is preserved. The "first stop" runs (5 seeds, before the fix) are
  kept in `results/backup/...` but are not in the final plot.

## 4. How the experiment was run (pipeline)

- `scripts/ollama_seed_loop.ps1` loops over seeds. For each seed it runs vanilla (CPU, in a separate process) in
  parallel with the agent (GPU), then rebuilds the comparison group and the plot. Separate folders per condition,
  so they are separate experiments.
- Runs write to `results/logs/ollama-batch/qwen3.6_27b/hartmann6-blind/seed<N>/{vanilla,sara-lenz}/sandbox_<id>/`.
  Agent sandboxes contain `trace.jsonl` (full conversation incl. tool calls and `forced_continuation` markers),
  `state.json` (trial log), `run_meta.json` (status, settings, token usage), `context.md` (the problem text shown).
- `scripts/make_comparison_group.py` copies completed runs into `results/logs/analysis-hartmann6/<condition>/` so the
  viewer can aggregate across seeds (completed runs only; one run per seed, the latest completed).
- `scripts/plot_seed_distribution.py` draws the figure.

## 5. How to read the plot

- X: evaluation number 1-50. Y: best-so-far regret (lower is better).
- Thick line = mean over seeds. Shaded band = mean +/- 1 **standard error of the mean** (std / sqrt(n)). This is
  uncertainty about the *mean*, not the spread of individual runs; thin faint lines are individual seeds.
- First 7 evaluations coincide because the warm-up points are shared.
- **Agent curves are carried forward after the agent stopped** (held at its last best value). So the agent's tail
  past ~30 evaluations is not measured data. The dotted vertical line marks the median stopping point (31).
- Different n per line: 65 vs 63 because two agent runs crashed (see 6). The old "first stop" line had n=5 and was removed.

## 6. Key numbers (mean +/- SE across seeds)

| Evaluations | Vanilla (n=65) | LLM agent, forced (n=63) |
|---|---|---|
| 7 (warm-up only) | 2.45 +/- 0.07 | 2.44 +/- 0.08 |
| 20 | 1.49 +/- 0.08 | 1.78 +/- 0.08 |
| 30 | 0.89 +/- 0.07 | 1.44 +/- 0.09 |
| 40 | 0.50 +/- 0.07 | 1.38 +/- 0.09 |
| 50 | 0.33 +/- 0.06 | 1.36 +/- 0.09 |

Paired comparison on the same 63 seeds (agent minus vanilla regret; positive = agent worse):

- at 20 evals: +0.29 +/- 0.09, agent better in 24/63 seeds
- at 30 evals: +0.55 +/- 0.10, agent better in 14/63
- at 50 evals: +1.03 +/- 0.11, agent better in 6/63

Agent run behaviour:

- 8 of 63 runs used the full 50 evaluations (seeds 52, 60, 63, 66, 69, 75, 89, 99). 55 ended early: median 30
  evaluations (range 21-47). Every early ender had used all 30 allowed nudges, i.e. the nudge cap, not the agent's
  choice, ended those runs.
- Full-budget runs: agent final mean regret 1.39 vs vanilla 0.42 on the same seeds. Early-ended runs: agent 1.36 vs
  vanilla 0.32. So early stopping alone does not explain the gap.
- Two agent runs crashed mid-run (seeds 80 and 94, status `failed`; cause not investigated) and are excluded; one run (seed 107)
  was incomplete when the loop was stopped.

## 7. Limitations and things not to overclaim

- One benchmark (Hartmann-6), one budget (50), one model (a 27B open model, quantised), one set of prompts. Meta's paper
  used much stronger frontier models; this is **not** a replication of its results and says nothing about them.
- Agent sampling parameters were not controlled or recorded. LLM runs are stochastic; seeds fix the warm-up, not the LLM.
- The agent's 30-nudge cap is a harness choice of ours, so "agent regret at 50" mixes search quality with the cap.
- Cause of the empty replies is unknown (model, Ollama chat template, or context). Not tested across models.
  Added later: the FIRST forced continuation happens at a median of evaluation 10 (range 7-27, 28 seeds), when the
  prompt is only ~15K tokens, so a full 32K context window cannot be the (sole) cause of the first empty reply. The
  server log does show prompts reaching 31-32.6K tokens of the 32,768 window later in runs, so overflow may explain
  some of the later ones. Unresolved.
- Seed 42 has two agent runs (concurrent jobs overlapped); the aggregator uses one per seed (latest completed).
- No compute/token penalty is applied: regret is versus evaluation count only.
- Standard error of the mean assumes independent seeds; seeds also share function shift via seed, so vanilla and agent on
  the same seed are paired, not independent (hence the paired comparison).
- The agent's bash tool was changed to run through Git Bash on Windows, and the oracle is a copy not a symlink
  (Windows). Neither should change results, but they differ from the upstream setup.

## 8. What the paper says about stopping (arXiv 2608.00316)

- Section 4: `stop` is a normal agent action (terminate and return the incumbent). Evaluations are the only thing
  that consumes budget; all methods get the same budget (Hartmann 100 in the paper).
- The only mention of early stopping is Appendix C.2: "Opus 4.8 tends to stop earlier than GPT 5.5." (a cost
  comparison, no explanation, no stats on unused budget). The paper reports no stopping rule and no scoring rule for
  runs that stop early. Our empty-reply early stops are a different, local-setup phenomenon.

## 9. Likely questions and short answers

- *Why does vanilla beat the agent?* The agent chooses its own points and can override the optimiser; in the
  traces it sometimes picked poor points. With a mid-sized local model it is not adding value over the plain algorithm here.
- *Why isn't a longer-running agent always better?* Best-so-far is non-increasing within one run, but different runs
  are separate random trajectories; the agent is not "vanilla plus extra steps".
- *Is the agent's flat tail real?* No, carried forward after it stopped (see 5).
- *What is the uncertainty?* +/- 1 standard error of the mean across seeds; 65 and 63 seeds.
- *What are the seeds?* Different shifted problem instances / warm-up designs; same seed = same warm-up for both conditions.
- *Why "blind"?* To stop the LLM recalling the known Hartmann-6 optimum.
- *What would strengthen it?* More benchmarks and budgets (Branin, Ackley, LoRA/BoLT), larger models on the 2x3090
  machine, checking the empty-reply cause, and recording sampling settings.
