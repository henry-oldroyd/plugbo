# LLAMBO (sampler slot)

**Cite:** Liu et al., ICLR 2024[^5].

[Paper](https://arxiv.org/abs/2402.03921){ .md-button }
[Code](https://github.com/tennisonliu/LLAMBO){ .md-button }

Does not replace the GP. Use it to propose candidates from natural-language
context and recent trials, then `score` them against lenz.

## LLM configuration

Sampling uses Sara's LLM by default. Pass `--sampler-llm-*` or
`llambo set-llm` only to use a different model:

```bash
lenz llambo set-llm --state ./state.json --provider openai --model gpt-4.1
lenz llambo warmstart --state ./state.json --n 5 --context ./context.md
lenz llambo sample --state ./state.json --n 8
```

## Making it the default sampler

To make every `suggest` go through LLAMBO instead of BoTorch acquisition
optimization:

```bash
lenz set-sampler --state ./state.json --sampler llambo
```

Strongest early in a run, when observations are scarce. After the GP is
trustworthy, switch back to `botorch`.

[^5]: Liu, Astorga, Seedat, van der Schaar. *Large Language Models to Enhance Bayesian Optimization.* ICLR 2024.
