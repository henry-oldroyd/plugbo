# TuRBO (region slot)

**Cite:** Eriksson et al., NeurIPS 2019[^3].

[Paper](https://arxiv.org/abs/1910.01739){ .md-button }
[Code](https://github.com/uber-research/TuRBO){ .md-button }

Switch with `lenz set-region --policy turbo`. Subsequent `suggest` calls
(without `--bounds` / `--around`) optimize the acquisition function inside a
hyperrectangle trust region. The region is centered on the incumbent. Length
doubles after a streak of improvements and halves after a streak of
failures. Below a minimum length the region restarts.

You own *mode* (enable, disable, override). TuRBO owns the counters.

```bash
lenz turbo init --state ./state.json
lenz turbo status --state ./state.json
lenz turbo override --state ./state.json --length 0.4 --center '{"x": 0.3}'
lenz set-region --state ./state.json --policy box
```

Prefer this over improvised `set-bounds` when you want principled local
search. Still use `suggest --around` for a one-shot local probe that does
not persist.

[^3]: Eriksson, Pearce, Gardner, Turner, Poloczek. *Scalable Global Optimization via Local Bayesian Optimization.* NeurIPS 2019.
