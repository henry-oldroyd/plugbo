# Plugins

`lenz` exposes four slots. Each slot has a default that needs no plugin, and
an optional occupant that upgrades it:

| Module | Role | Default | Occupant | Commands |
|---|---|---|---|---|
| `sara` | search agent | — | — | `sara run` |
| `lenz` | trial log, posterior, acquisition | BoTorch loop | — | `create`, `suggest`, `submit`, `incumbent` |
| Surrogate | GP | fixed Matérn | [**CAKE**](../reference/cake.md) | `set-surrogate`, `evolve-kernels`, `kernel-population` |
| Region | search bounds | box | [**TuRBO**](../reference/turbo.md) | `set-region`, `set-bounds`, `turbo status` |
| Prior | belief | none | [**πBO**](../reference/pibo.md) | `set-belief` |
| Sampler | candidates | BoTorch | [**LLAMBO**](../reference/llambo.md) | `set-sampler`, `llambo sample` |

Occupying a slot is always a `set-*` verb (`set-surrogate`, `set-region`,
`set-belief`, `set-sampler`). Plugins may register additional verbs on top of
that. `lenz plugins` lists installed modules.

```bash
lenz plugins
```

## How a slot is occupied

```bash
lenz set-surrogate --state ./state.json --surrogate cake
```

- Method state lives in `state.json` under `plugins`, not on the live shelf.
- Reconfiguring a slot never discards trials — the trial log stays intact
  regardless of which plugin is active.

## Adding a new plugin

A method occupies one slot and stores its own blob under
`frame.plugins[<name>]`. Core `lenz` never imports plugin internals — it
calls hooks, and the plugin registers CLI verbs.

To add a method:

1. Implement `LenzPlugin` (`lenz/plugins/base.py`) in `lenz/plugins/`.
2. Ship a `{name}.md` prompt file beside it — this is what Sara reads to
   learn how to use the plugin (see the existing
   [reference pages](../reference/cake.md), which are generated from these
   files).
3. Register the plugin in `lenz/plugins/registry.py`.

Open a GitHub issue before adding a new slot, a large dependency, or a new
experiment backend — see [Contributing](../contributing.md) for the full
process.
