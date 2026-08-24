# Experiments

## Synthetic benchmark harness

Textbook functions such as Hartmann and Ackley have published optima that an
LLM can recall. The synthetic harness therefore uses an **anti-memorization
sandbox**: renamed parameters, a unit cube, and a shifted optimum. Scoring
uses a hidden answer key outside the sandbox.

List available functions:

```bash
python3 -c "from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))"
```

Run a benchmark:

```bash
./scripts/run_synthetic.sh hartmann6
./scripts/run_synthetic.sh hartmann6 --disclosure revealed
./scripts/run_synthetic.sh hartmann6 --disclosure revealed-shift
./scripts/run_synthetic.sh hartmann6 --backend sara-lenz --disclosure all
./scripts/run_synthetic.sh hartmann6 --warmup 7
```

### Disclosure levels

`--disclosure` controls how much identity the agent sees:

| Level | What the agent sees | What it tests |
|---|---|---|
| `blind` (default) | Renamed parameters, unit cube, shifted optimum, generic problem text | Search ability, not retrieval |
| `revealed-shift` | Real name and bounds, same shift as the blind run | Search is still required |
| `revealed` | The textbook problem, unshifted | Whether evaluation 1 is already the known optimum |
| `all` | Runs the three levels above | Needs a single `--backend` |

### Backends

`--backend` is a comma list: `vanilla`, `cake`, `turbo`, `sara-lenz`,
`sara-lenz-cake`, `sara-only`, or `all` for `vanilla` + `sara-lenz` +
`sara-lenz-cake`.

Every backend uses the same seeded Sobol warm-start, defaulting to `d+1`
evaluations when a seed is set. Pass `--warmup N` to override it for all
selected backends.

`gp_sample<dim>` is a no-prior control (a fresh GP sample path, not in
`REGISTRY`). Run it the same way as `hartmann6`.

Next: [LoRA hyperparameter optimization (BoLT)](bolt.md), or view results in
the [run viewer](run-viewer.md).
