# LoRA hyperparameter optimization (BoLT)

The mixed-type experiment is LoRA hyperparameter optimization on **BoLT**
(Black-box Optimization for LLM Tasks)[^7]. There is no textbook optimum to
recall — the oracle is a deterministic emulator of expensive LLM fine-tuning
runs.

The search space is always revealed: seven mixed continuous, integer, and
categorical variables. `--context` only changes the story the agent reads,
not the space:

| `--context` | What the agent reads |
|---|---|
| `domain` (default) | Real LoRA/Qwen names and a short task description |
| `generic` | Names, types, and bounds only — no domain prose |
| `misleading` | False LoRA folklore (dropout near 0.05, `lora_target = 0`, few layers) presented as known-good defaults |

## Running it

```bash
pip install -e '.[bolt]'
./scripts/run_bolt.sh
./scripts/run_bolt.sh --backend vanilla,cake
./scripts/run_bolt.sh --context generic
```

Provider and model come from `.env` or the `PROVIDER` / `MODEL` environment
variables. Completed and in-flight legs are skipped, so a run can be resumed
by re-invoking the same command.

Plots land in `compare.html` next to the run, or open the
[run viewer](run-viewer.md) (`plugbo-viz`).

[^7]: Chew, Chen, Hemachandra, Low. *BoLT: A Benchmark to Democratize Black-box Optimization Research for Expensive LLM Tasks.* arXiv:2605.17000, 2026.
