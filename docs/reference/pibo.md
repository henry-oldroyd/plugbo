# πBO (prior slot)

**Cite:** Hvarfner et al., ICLR 2022[^4].

[Paper](https://arxiv.org/abs/2204.11051){ .md-button }
[Code](https://github.com/piboauthors/PiBO-Spearmint){ .md-button }

Compile a belief from context into a factorized distribution, then fold it
into the acquisition:

$$\alpha_\pi(x) = \alpha(x) \cdot \pi(x)^{\beta / (t+1)}$$

The exponent decays so a wrong prior fades as data arrives.

```bash
lenz set-belief --state ./state.json --prior '{
  "lr": {"dist": "lognormal", "mu": -7.0, "sigma": 1.0},
  "dropout": {"dist": "beta", "a": 2, "b": 8}
}' --decay-beta 10
```

Supported `dist` values: `uniform`, `normal` (`mu`, `sigma`), `lognormal`
(`mu`, `sigma`), `beta` (`a`, `b` on the range scaled to `[0,1]`),
`categorical` (`probs` map). Name a numeric distribution — do not pass
adjectives such as "aggressive".

Clear a prior:

```bash
lenz set-belief --state ./state.json --prior '{}' --clear
```

[^4]: Hvarfner, Stoll, Souza, Lindauer, Hutter, Nardi. *πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization.* ICLR 2022.
