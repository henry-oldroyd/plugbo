# Quickstart

## Vanilla loop (no LLM)

`lenz` runs a standalone BoTorch BO loop. It owns the trial log and the
posterior; nothing here calls an LLM.

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --acqf noisy_logei

lenz suggest --state ./state.json
lenz submit --state ./state.json --config '{"x1":1.0,"x2":2.0}' --metrics '{"y":12.3}'
lenz incumbent --state ./state.json
```

`create` defines the search space and objectives, `suggest` proposes the next
config, `submit` records an observed outcome, and `incumbent` reports the
best trial so far.

## Agent in the loop

`sara run` puts an LLM agent (Sara) in front of the same backend. Sara reads
problem context, calls an evaluation command, and drives `lenz` through the
loop.

```bash
sara run \
  --provider anthropic --model claude-opus-5 \
  --context examples/branin/context.md \
  --eval "python3 examples/branin/eval.py" \
  --budget 30 \
  --workdir ./results/logs/branin-1
```

- `--context` is problem markdown (see `examples/branin/context.md` for the
  shape).
- `--eval` is a command that takes one config JSON on argv and prints
  metrics.
- `--no-lenz` drops the backend entirely: Sara proposes configs and calls the
  oracle herself, without a BoTorch posterior.

### Providers

| `--provider` | Notes |
|---|---|
| `anthropic` | Needs `ANTHROPIC_API_KEY` |
| `openai` | Needs `OPENAI_API_KEY` |
| `openai-compatible` | Requires `--base-url` |
| `ollama` | `http://localhost:11434/v1` by default |

## Occupying a plugin slot

Any slot (surrogate, region, prior, sampler) can be occupied with or without
the agent:

```bash
lenz set-surrogate --state ./state.json --surrogate cake
lenz set-region --state ./state.json --policy turbo
lenz set-belief --state ./state.json --prior '{"x":{"dist":"normal","mu":0.3,"sigma":0.1}}'
lenz set-sampler --state ./state.json --sampler llambo
```

Or occupy a slot at `create` time:

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --surrogate cake --budget 30
```

CAKE and LLAMBO inherit Sara's provider and model by default. Pass
`--kernel-llm-*` or `--sampler-llm-*` only to use a different model for that
plugin. TuRBO and πBO do not call an LLM.

See the [Plugins](../concepts/plugins.md) page for the full slot model, and
the [plugin reference](../reference/cake.md) pages for per-plugin details.

Full command reference lives in `sara/prompts/LENZ_REF.md` in the repo.
