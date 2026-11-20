"""Mean +/- standard-error regret across seeds: vanilla BO vs pinned agent (experiment 1) vs free agent.

Reads the group built by make_free_group.py and saves the figure (plus a per-seed-count snapshot).
All three lines use the SAME seeds (those where vanilla and the free agent both finished), so the
comparison is paired. Also reports how often the free agent actually reconfigured lenz.

  python scripts/plot_free_comparison.py [--group G] [--fig out.png]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.aggregate import condition_seed_traces

N = 50
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"
STYLE = {  # reference-palette slots 1-3
    "vanilla-no-LLM": ("Vanilla BO (no LLM)", "#2a78d6"),
    "agent-pinned": ("Agent, optimizer pinned (experiment 1)", "#eb6834"),
    "agent-free": ("Agent, may reconfigure optimizer", "#1baf7a"),
    "agent-free-hint": ("Agent, may reconfigure + vague hint", "#eda100"),
}
SET_RE = re.compile(r"lenz\s+(set-[a-z]+|llambo\s+[a-z\-]+|turbo\s+[a-z\-]+)")


def failure_counts(free_root: Path) -> Counter:
    """Failed run attempts per condition, from failures.jsonl (written by scripts/log_failures.py)."""
    log, c = free_root / "failures.jsonl", Counter()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c[json.loads(line)["condition"]] += 1
    return c


def reconfig_usage(free_root: Path, model_tag: str, bench: str, cond: str = "sara-lenz-free") -> tuple[int, Counter, Counter]:
    """(#free-agent runs, runs using each verb, total calls per verb), from the agents' traces."""
    base = free_root / model_tag / f"{bench}-blind"
    runs, per_run, calls = 0, Counter(), Counter()
    for tr in base.glob(f"seed*/{cond}/sandbox_*/trace.jsonl"):
        meta = tr.parent / "run_meta.json"
        try:
            if json.loads(meta.read_text()).get("status") != "completed":
                continue
        except (OSError, json.JSONDecodeError):
            continue
        runs += 1
        used = set()
        for line in tr.read_text(encoding="utf-8").splitlines():
            for c in json.loads(line).get("tool_calls", []) or []:
                if c["name"] != "bash":
                    continue
                for m in SET_RE.finditer(c["arguments"].get("cmd", "")):
                    verb = re.sub(r"\s+", " ", m.group(1))
                    calls[verb] += 1
                    used.add(verb)
        per_run.update(used)
    return runs, per_run, calls


def pad(tr: list[float]) -> list[float]:
    return (tr + [tr[-1]] * N)[:N]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="results/logs/analysis-free-hartmann6")
    ap.add_argument("--fig", default="results/figures/regret_free_vs_vanilla.png")
    ap.add_argument("--free-root", default="results/logs/ollama-free")
    ap.add_argument("--model-tag", default="qwen3.6_27b")
    ap.add_argument("--bench", default="hartmann6")
    ap.add_argument("--seed-lines", action="store_true", help="also draw faint per-seed lines")
    a = ap.parse_args()
    group, fig_path = Path(a.group), Path(a.fig)

    traces = {k: condition_seed_traces(group / k) if (group / k).is_dir() else {} for k in STYLE}
    # The "primary" agent is the one this run is about; seeds where it failed/never finished are dropped from
    # EVERY line, so all lines use exactly the same seeds.
    primary = "agent-free-hint" if traces["agent-free-hint"] else "agent-free"
    cond_dir = "sara-lenz-free-hint" if primary == "agent-free-hint" else "sara-lenz-free"
    common = sorted(set(traces["vanilla-no-LLM"]) & set(traces[primary]))
    if not common:
        print("no seeds finished yet for both vanilla and the agent; nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(1, N + 1)
    stats = {}
    for key, (label, color) in STYLE.items():
        seeds = [s for s in common if s in traces[key]]
        if len(seeds) < 2:
            continue
        arr = np.array([pad(traces[key][s]) for s in seeds])
        mean = arr.mean(0)
        se = arr.std(0, ddof=1) / np.sqrt(len(arr))
        if a.seed_lines:
            for row in arr:
                ax.plot(x, row, color=color, lw=0.6, alpha=0.2, zorder=1)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.25, lw=0, zorder=2)
        ax.plot(x, mean, color=color, lw=2.2, zorder=3, label=f"{label}  (n={len(arr)} seeds)")
        stops = [len(traces[key][s]) for s in seeds]
        stats[key] = (mean[-1], se[-1], float(np.median(stops)))

    ax.set_xlim(1, N)
    ax.set_xlabel("Evaluations (7 shared Sobol warm-up points first)", color=MUTED)
    ax.set_ylabel("Best-so-far regret (lower is better)", color=MUTED)
    ax.set_title(f"Hartmann-6, qwen3.6:27b: mean regret over {len(common)} paired seeds (± 1 SE of the mean)",
                 loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)

    runs, per_run, calls = reconfig_usage(Path(a.free_root), a.model_tag, a.bench, cond_dir)
    if runs:
        used = ", ".join(f"{v} {n}/{runs}" for v, n in per_run.most_common()) or "none"
        note = f"Agent reconfiguration (runs using each verb): {used}."
    else:
        note = "Agent: no finished runs yet."
    fails = failure_counts(Path(a.free_root))
    fail_note = "  Failed runs excluded from all lines: " + (
        ", ".join(f"{k} {v}" for k, v in sorted(fails.items())) if fails else "0"
    ) + "."
    stop_note = ""
    if primary in stats:
        stop_note = f"  Agent median evaluations used: {stats[primary][2]:.0f}/50 (curves held flat after an agent stops)."
    fig.text(0.01, 0.012, note + stop_note + fail_note, color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, facecolor=BG)
    snap = fig_path.parent / "free_snapshots" / f"{fig_path.stem}_n{len(common):03d}.png"
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fig_path, snap)
    print(f"wrote {fig_path}  (+ snapshot {snap.name})")
    for key, (m, se, med) in stats.items():
        print(f"  {STYLE[key][0]:45s} final regret {m:.2f} +/- {se:.2f}   median evals used {med:.0f}")
    if runs:
        print(f"  reconfiguration ({cond_dir}, {runs} runs): runs using each verb = {dict(per_run)}; total calls = {dict(calls)}")
    print(f"  failed runs logged: {dict(fails)}")


if __name__ == "__main__":
    main()
