# Contributing

Bug reports, new BO plugins, and benchmarks are welcome. By participating,
you agree to the [Code of Conduct](https://github.com/richardcsuwandi/plugbo/blob/main/CODE_OF_CONDUCT.md).

Open a GitHub issue first if the change adds a slot, a new experiment
backend, or a large dependency. Use the bug template for failures and the
proposal template for new methods or functions. Do not paste API keys or
`.env`.

This repository is an independent re-implementation of Sara and lenz. The
plugin protocol is the main extension. Keep Sara's tools as `bash` and
`read` — new capability should appear as a `lenz` verb, not a second agent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # only if you will call an LLM
```

Python 3.10+. Do not commit `.env`, `state.json`, or anything under
`results/`.

## Tests

Run the default, no-network checks for the code you touched:

```bash
pytest -m "not slow"
./scripts/smoke_plugins.sh
```

`pytest` covers unit tests. The smoke script drives a short `lenz` CLI loop
so a plugin that unit-tests still has to work as a verb.

Extra smoke flags (`--live`, `--baseline`, and any later ones) are optional.
Use them only when your change hits that layer (LLM-backed plugins, the
scripted BO loop). The script header documents what each flag currently
does. `--live` needs API keys.

Add or extend tests next to the existing ones for that layer (`tests/`). A
plugin PR should prove occupancy and the hook. A benchmark PR should prove
the function values and that identity does not leak into agent-visible
files.

## Adding a BO plugin

A method occupies one slot of the live policy and stores its own blob under
`frame.plugins[<name>]`. Core `lenz` never imports plugin internals — it
calls hooks, and the plugin registers CLI verbs. See
[Plugins](concepts/plugins.md) for the slot model and the full walkthrough
in [`CONTRIBUTING.md`](https://github.com/richardcsuwandi/plugbo/blob/main/CONTRIBUTING.md).

## Links

- [`CONTRIBUTING.md`](https://github.com/richardcsuwandi/plugbo/blob/main/CONTRIBUTING.md) — full setup, plugin protocol, adding a function
- [Issues](https://github.com/richardcsuwandi/plugbo/issues) — bugs and proposals
- [Code of Conduct](https://github.com/richardcsuwandi/plugbo/blob/main/CODE_OF_CONDUCT.md)
