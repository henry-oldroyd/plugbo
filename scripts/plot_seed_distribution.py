"""Mean +/- standard-error 'sausage' plot of best-so-far regret across seeds.

  python scripts/plot_seed_distribution.py [group_dir] [out.png]
Reads the group built by scripts/make_comparison_group.py (one folder per condition).
A run that ended early is held flat at its last best value (same convention as plugbo-viz),
so the right-hand part of an agent curve is carried forward, not measured; dotted
verticals mark the median evaluation count at which the agent stopped.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.aggregate import condition_seed_traces

group = Path(sys.argv[1] if len(sys.argv) > 1 else "results/logs/analysis-hartmann6")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/figures/regret_seed_distribution.png")
N = 50
STYLE = {  # label, colour (reference palette slots 1-3), order
    "vanilla-no-LLM": ("Vanilla BO (no LLM)", "#2a78d6"),
    "agent-first-stop": ("Agent, first stop (before fix)", "#1baf7a"),
    "agent-forced-continue": ("Agent, forced to continue", "#eb6834"),
}
SHORT = {"agent-first-stop": "first-stop agent", "agent-forced-continue": "forced agent"}
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"

fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150, facecolor=BG)
ax.set_facecolor(BG)
x = np.arange(1, N + 1)
for key, (label, color) in STYLE.items():
    traces = condition_seed_traces(group / key)
    if not traces:
        continue
    rows, stops = [], []
    for seed, tr in sorted(traces.items()):
        stops.append(len(tr))
        tr = tr[:N] + [tr[-1]] * max(0, N - len(tr))
        rows.append(tr)
        ax.plot(x, tr, color=color, lw=0.7, alpha=0.25, zorder=1)
    a = np.array(rows)
    mean = a.mean(0)
    se = a.std(0, ddof=1) / np.sqrt(len(a))
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.28, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.2, zorder=3, label=f"{label}  (n={len(a)} seeds)")
    ms = int(st.median(stops))
    if ms < N:
        ax.axvline(ms, color=color, lw=1, ls=":", zorder=1)
        ax.text(ms + 0.4, 3.25, f"{SHORT[key]}\nmedian stop @{ms}", color=MUTED, fontsize=8, va="top")

ax.set_xlim(1, N)
ax.set_ylim(0, 3.3)
ax.set_xlabel("Evaluations (7 shared Sobol warm-up points first)", color=MUTED)
ax.set_ylabel("Best-so-far regret (lower is better)", color=MUTED)
ax.set_title("Hartmann-6, qwen3.6:27b: regret across 5 seeds (mean ± 1 standard error)",
             loc="left", color=INK, fontsize=12, fontweight="bold")
ax.grid(axis="y", color="#e4e3df", lw=0.8)
ax.tick_params(colors=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c2")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93), labelcolor=INK)
fig.text(0.01, 0.01, "Thin lines = individual seeds. Agent curves are held flat after the agent stopped (carried forward, not measured).",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=BG)
print("wrote", out)
