"""Trajectory plot: BO vs agent across paired seeds, with first-forced-continuation markers.

    render(pairs, out_path, ...)  ->  stats dict

`pairs` maps seed -> {"bo": [regret...], "agent": [regret...], "ff": int | None}.
Only the single writer process (the orchestrator) calls this; the file is written to a temp path and
atomically renamed so a viewer never sees a half-written PNG.
"""
from __future__ import annotations

import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, BG, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BO_COLOR, AGENT_COLOR = "#2a78d6", "#eb6834"  # reference-palette slots 1 and 2


def _pad(tr: list[float], n: int) -> list[float]:
    return (list(tr) + [tr[-1]] * n)[:n]


def render(pairs: dict[int, dict], out_path: Path, title: str, footer: str = "", budget: int = 50,
           seed_lines: bool = True) -> dict:
    seeds = sorted(pairs)
    x = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)

    stats = {"n": len(seeds)}
    for key, color, label in (("bo", BO_COLOR, "Vanilla BO (no LLM)"), ("agent", AGENT_COLOR, "LLM agent")):
        arr = np.array([_pad(pairs[s][key], budget) for s in seeds])
        mean = arr.mean(0)
        se = arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(budget)
        if seed_lines:
            for row in arr:
                ax.plot(x, row, color=color, lw=0.6, alpha=0.16, zorder=1)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.25, lw=0, zorder=2)
        ax.plot(x, mean, color=color, lw=2.4, zorder=3, label=f"{label}  (n={len(arr)} seeds)")
        stats[f"{key}_final_mean"], stats[f"{key}_final_se"] = float(mean[-1]), float(se[-1])

    # first forced continuation: an X on each agent trajectory, and the median as a vertical line
    ff_x, ff_y = [], []
    for s in seeds:
        idx = pairs[s].get("ff")
        tr = pairs[s]["agent"]
        if idx and 1 <= idx <= len(tr):
            ff_x.append(idx)
            ff_y.append(tr[idx - 1])
    stats["n_with_forced"] = len(ff_x)
    if ff_x:
        ax.scatter(ff_x, ff_y, marker="x", s=34, linewidths=1.3, color="#8a2d0e", zorder=5,
                   label=f"first forced continuation ({len(ff_x)}/{len(seeds)} runs)")
        med = statistics.median(ff_x)
        stats["median_first_forced"] = med
        ax.axvline(med, color=AGENT_COLOR, ls="--", lw=1.4, zorder=2)
        ax.text(med + 0.4, ax.get_ylim()[1] * 0.97, f"median first forced\ncontinuation: eval {med:g}",
                color=MUTED, fontsize=8, va="top")
    else:
        stats["median_first_forced"] = None

    ax.set_xlim(1, budget)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Evaluations (shared Sobol warm-up points first)", color=MUTED)
    ax.set_ylabel("Best-so-far regret (lower is better)", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c2")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK, fontsize=9)
    if footer:
        fig.text(0.01, 0.012, footer, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp, facecolor=BG)
    plt.close(fig)
    os.replace(tmp, out_path)
    return stats
