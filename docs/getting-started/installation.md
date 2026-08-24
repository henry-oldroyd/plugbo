# Installation

PlugBO is not on PyPI. Install it from source (Python 3.10+).

## Editable install

```bash
git clone https://github.com/richardcsuwandi/plugbo.git
cd plugbo
pip install -e .
```

That pulls `torch`, `gpytorch`, `botorch`, and the LLM clients (`anthropic`,
`openai`). Copy `.env.example` to `.env` if you will call an LLM:

```bash
cp .env.example .env
```

```
# Anthropic (Claude)
ANTHROPIC_API_KEY=

# OpenAI (or any OpenAI-compatible endpoint: vLLM, Together, Groq, LM Studio, ...)
OPENAI_API_KEY=
OPENAI_BASE_URL=

# Ollama (local models). Defaults to http://localhost:11434/v1 if unset.
OLLAMA_BASE_URL=
```

## Dev install

Adds test dependencies:

```bash
pip install -e ".[dev]"
```

## LoRA HPO extra

Needed only for the [BoLT benchmark](../guides/bolt.md) (LoRA hyperparameter
optimization emulator):

```bash
pip install -e ".[bolt]"
```

## Console scripts

The editable install registers these entry points:

| Command | Package | Purpose |
|---|---|---|
| `lenz` | `lenz.cli:main` | BoTorch backend CLI |
| `sara` | `sara.cli:main` | Agent-in-the-loop CLI |
| `sara-blind-test` | `benchmarks.run_blind_test:main` | Blind synthetic-benchmark runner |
| `plugbo-viz` | `viz.server:main` | Run viewer |
| `plugbo-viz-merged` | `viz.merged_server:main` | Merged run viewer |

Next: the [quickstart](quickstart.md).
