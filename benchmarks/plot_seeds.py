"""Multi-seed comparison plots: median and inter-quartile band of best-so-far regret across seeds.

Reads the layout written by `scripts/ollama_experiments.sh` (and `run_ollama_batch.sh`):

    <root>/<benchmark>-<disclosure>/seed<N>/<backend>/sandbox_*/...

and writes (with matplotlib; `pip install -e '.[plots]'`) one PNG per benchmark plus a self-contained
HTML page that embeds them, with, per benchmark,
a log-regret chart (every seed as a faint line, the median as a solid line, the 25-75% band shaded)
and a table of final regret per backend. Pass several `--root`s (e.g. one per model) to overlay
them: the model-independent `vanilla` backend is taken from the first root that has it.

    python3 -m benchmarks.plot_seeds --root results/logs/ollama-experiments/qwen3.6_27b
    python3 -m benchmarks.plot_seeds --root .../qwen3.6_27b --root .../deepseek-r1_32b-qwen-distill-q8_0 --out cmp.html

Honesty rules: an agent run that stopped early is NOT padded with its last value. The median/band at
evaluation i uses only the runs that reached i, and is drawn only while at least half of that backend's
runs did; the table reports how many runs there are, how many used the full budget, and the median
number of evaluations used.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
import statistics
from pathlib import Path

from .plot_compare import COLORS, regret_trace

REGRET_FLOOR = 1e-6  # log-scale floor: a regret of exactly 0 would otherwise be -inf
_SEED_RE = re.compile(r"^seed(\d+)$")


# ---------------------------------------------------------------- data collection (touches the disk)
def collect_runs(root: Path) -> dict[str, dict[str, dict[int, list[float]]]]:
    """{benchmark-disclosure: {backend: {seed: best-so-far regret trace}}} for one model root."""
    out: dict[str, dict[str, dict[int, list[float]]]] = {}
    for group in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        for seed_dir in sorted(p for p in group.iterdir() if p.is_dir()):
            m = _SEED_RE.match(seed_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            for cond in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                trace = regret_trace(cond, seed=seed)
                if trace:
                    out.setdefault(group.name, {}).setdefault(cond.name, {})[seed] = trace
    return out


def merge_roots(roots: list[Path], names: dict[str, str] | None = None) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Combine several model roots. With one root labels are plain backend names. With several, `vanilla`
    keeps its name (taken from the first root that has it) and other backends become 'backend [model]'.
    `names` maps a root directory name (or "vanilla") to the label to show instead."""
    names = names or {}
    merged: dict[str, dict[str, dict[int, list[float]]]] = {}
    for root in roots:
        for group, backends in collect_runs(root).items():
            for backend, runs in backends.items():
                if backend == "vanilla":
                    label = names.get("vanilla", "vanilla")
                elif len(roots) == 1:
                    label = names.get(root.name, backend)
                else:
                    label = names.get(root.name, f"{backend} [{root.name}]")
                if backend == "vanilla" and label in merged.get(group, {}):
                    continue
                merged.setdefault(group, {})[label] = runs
    return merged


# ---------------------------------------------------------------- statistics (pure)
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]."""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def aggregate(runs: dict[int, list[float]]) -> dict:
    """Median / quartiles per evaluation index over the runs that reached it (see module docstring)."""
    traces = [t for _, t in sorted(runs.items())]
    n_runs = len(traces)
    max_len = max(len(t) for t in traces)
    median, lo, hi, counts = [], [], [], []
    for i in range(max_len):
        vals = [t[i] for t in traces if len(t) > i]
        counts.append(len(vals))
        median.append(statistics.median(vals))
        lo.append(percentile(vals, 0.25))
        hi.append(percentile(vals, 0.75))
    n_shown = sum(1 for c in counts if c * 2 >= n_runs)  # counts only shrink, so this is a prefix
    finals = [t[-1] for t in traces]
    lengths = [len(t) for t in traces]
    # Mean +- standard error needs every run at every index, so a run that stopped early is carried forward at
    # its last best-so-far value (its incumbent; NOT a measurement). The plot says so.
    padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
    mean = [statistics.fmean(col) for col in zip(*padded)]
    se = [(statistics.stdev(col) / math.sqrt(n_runs)) if n_runs > 1 else 0.0 for col in zip(*padded)]
    return {
        "mean": mean, "se": se, "final_mean": mean[-1], "final_se": se[-1],
        "n_runs": n_runs,
        "seeds": sorted(runs),
        "traces": traces,
        "median": median[:n_shown], "lo": lo[:n_shown], "hi": hi[:n_shown],
        "final_median": statistics.median(finals),
        "final_q25": percentile(finals, 0.25),
        "final_q75": percentile(finals, 0.75),
        "evals_median": statistics.median(lengths),
        "evals_max": max_len,
        "n_full_budget": sum(1 for n in lengths if n >= max_len),
    }


# ---------------------------------------------------------------- rendering (matplotlib)
def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: this runs on servers over SSH
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plot_seeds: pip install -e '.[plots]'") from e
    return plt


def render_figure(group: str, series: dict[str, dict], stat: str = "mean", title: str | None = None):
    """One benchmark, faint line per seed. stat="mean": bold mean with a +-1 standard-error band on a linear
    axis (early-stopped runs carried forward) and a dotted "median stop" line for backends that ended early.
    stat="median": bold median with a 25-75% band on a log axis, early-stopped runs not padded."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=110)
    n_max = max(s["evals_max"] for s in series.values())
    for k, (label, s) in enumerate(series.items()):
        color = COLORS[k % len(COLORS)]
        floor = REGRET_FLOOR if stat == "median" else 0.0
        for t in s["traces"]:  # every seed, drawn only where it was actually measured
            ax.plot(range(1, len(t) + 1), [max(v, floor) for v in t], color=color, lw=0.9, alpha=0.22)
        if stat == "mean":
            xs = range(1, len(s["mean"]) + 1)
            lo = [max(m - e, 0.0) for m, e in zip(s["mean"], s["se"])]
            hi = [m + e for m, e in zip(s["mean"], s["se"])]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax.plot(xs, s["mean"], color=color, lw=2.6, label=f"{label} (n={s['n_runs']} seeds)")
            if s["evals_median"] < n_max:
                ax.axvline(s["evals_median"], color=color, ls=":", lw=1.2)
                ax.annotate(f"median stop\n@{s['evals_median']:g}", (s["evals_median"], 1.0),
                            xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                            va="top", fontsize=8, color=color)
        else:
            m = len(s["median"])
            xs = range(1, m + 1)
            if m:
                ax.fill_between(xs, [max(v, floor) for v in s["lo"]], [max(v, floor) for v in s["hi"]],
                                color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, [max(v, floor) for v in s["median"]], color=color, lw=2.4, label=f"{label} (n={s['n_runs']})")
    if stat == "median":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(1, n_max)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best regret so far" + (" (log)" if stat == "median" else " (lower is better)"))
    ax.set_title(title or (f"{group}: mean ± 1 standard error across seeds" if stat == "mean"
                           else f"{group}: median and 25-75% across seeds"))
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    foot = ("Thin lines = individual seeds. Runs that stopped early are carried forward at their last best value (not measured)."
            if stat == "mean" else "Thin lines = individual seeds. Runs that ended early are not padded.")
    fig.text(0.01, 0.01, foot, fontsize=7.5, alpha=0.7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    _plt().close(fig)
    return buf.getvalue()


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def build_html(merged: dict[str, dict[str, dict[int, list[float]]]], title: str,
               fig_dir: Path | None = None, stat: str = "mean") -> tuple[str, dict]:
    """Returns (html, summary-json-able dict). With `fig_dir`, also writes <fig_dir>/<group>.png."""
    summary: dict = {}
    sections, overview_rows = [], []
    for group, backends in merged.items():
        series = {label: aggregate(runs) for label, runs in backends.items()}
        summary[group] = {
            label: {k: s[k] for k in ("n_runs", "seeds", "n_full_budget", "evals_median",
                                      "final_median", "final_q25", "final_q75", "final_mean", "final_se")}
            for label, s in series.items()
        }
        png = _png_bytes(render_figure(group, series, stat=stat))
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            (fig_dir / f"{group}.png").write_bytes(png)
        img = f'<img alt="{html.escape(group)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
        rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{s['n_runs']}</td><td>{s['n_full_budget']}/{s['n_runs']}</td>"
            f"<td>{_fmt(s['evals_median'])}</td><td>{_fmt(s['final_mean'])} &plusmn; {_fmt(s['final_se'])}</td>"
            f"<td>{_fmt(s['final_median'])} ({_fmt(s['final_q25'])} &ndash; {_fmt(s['final_q75'])})</td></tr>"
            for label, s in series.items()
        )
        sections.append(
            f"<h2>{html.escape(group)}</h2>{img}"
            f"<table><tr><th>backend</th><th>runs</th><th>full budget</th><th>median evals</th>"
            f"<th>final regret, mean &plusmn; SE</th><th>final regret, median (25&ndash;75%)</th></tr>{rows}</table>"
        )
        overview_rows.append((group, min(series, key=lambda b: series[b]["final_mean" if stat == "mean" else "final_median"]), series))
    labels = sorted({b for g in merged.values() for b in g})
    overview = ""
    if merged:
        head = "".join(f"<th>{html.escape(b)}</th>" for b in labels)
        body = "".join(
            f"<tr><td>{html.escape(g)}</td>"
            + "".join(f"<td>{_fmt(sr[b]['final_mean'] if stat == 'mean' else sr[b]['final_median'])} (n={sr[b]['n_runs']})</td>" if b in sr else "<td></td>" for b in labels)
            + f"<td>{html.escape(best)}</td></tr>"
            for g, best, sr in overview_rows
        )
        overview = (f"<h2>Overview: final best regret, {'mean' if stat == 'mean' else 'median'} over seeds (lower is better)</h2>"
                    f"<table><tr><th>benchmark</th>{head}<th>lowest</th></tr>{body}</table>")
    note = ("<p class=note>Faint lines: individual seeds. "
            + ("Bold line: mean across seeds; band: &plusmn;1 standard error of the mean (SE = sd/&radic;n). Runs that stopped "
               "early are carried forward at their last best value, so the late part of an early-stopping backend is "
               "extrapolated, not measured; the dotted line marks the median stopping point. "
               if stat == "mean" else
               "Bold line: median across seeds; band: 25&ndash;75%. Runs that ended early are not padded; the median/band "
               "is drawn only while at least half the runs reached that evaluation. ")
            + "Few seeds means wide uncertainty: read the band, not just the line.</p>")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 24px; background: #fff; color: #111;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  h1 {{ font-size: 16px; margin: 0 0 8px; }} h2 {{ font-size: 14px; margin: 28px 0 8px; }}
  img {{ max-width: 100%; height: auto; }} .note {{ opacity: .7; max-width: 900px; font-size: 12px; }}
  table {{ border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
  td, th {{ padding: 3px 12px; border-bottom: 1px solid rgba(128,128,128,.35); text-align: left; }}
</style></head>
<body><h1>{html.escape(title)}</h1>{note}
{overview if merged else "<p>No scoreable runs found yet.</p>"}
{"".join(sections)}
</body></html>
"""
    return page, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="model root containing <benchmark>-<disclosure>/seed<N>/<backend>; repeat to overlay several")
    ap.add_argument("--out", default=None, help="output HTML (default: <first root>/seeds.html)")
    ap.add_argument("--title", default="Multi-seed comparison: best regret vs evaluations")
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="mean: mean +- 1 standard error (default); median: median and 25-75%% band, log axis")
    ap.add_argument("--name", action="append", default=[], metavar="KEY=LABEL",
                    help="display label for a root directory name (or 'vanilla'); repeatable")
    args = ap.parse_args()
    roots = [Path(r) for r in args.root]
    for r in roots:
        if not r.is_dir():
            ap.error(f"--root {r} is not a directory")
    out = Path(args.out) if args.out else roots[0] / "seeds.html"
    names = dict(n.split("=", 1) for n in args.name)
    page, summary = build_html(merge_roots(roots, names), args.title, fig_dir=out.with_suffix(""), stat=args.stat)
    out.write_text(page)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}, {out.with_suffix('.json').name} and PNGs in {out.with_suffix('')}/ ({len(summary)} benchmark group(s))")


if __name__ == "__main__":
    main()
