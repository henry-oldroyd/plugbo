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


def merge_roots(roots: list[Path]) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Combine several model roots. With one root labels are plain backend names. With several, `vanilla`
    keeps its name (taken from the first root that has it) and other backends become 'backend [model]'."""
    merged: dict[str, dict[str, dict[int, list[float]]]] = {}
    for root in roots:
        for group, backends in collect_runs(root).items():
            for backend, runs in backends.items():
                label = backend if (len(roots) == 1 or backend == "vanilla") else f"{backend} [{root.name}]"
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
    return {
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


def render_figure(group: str, series: dict[str, dict]):
    """One benchmark: faint line per seed, bold median, shaded 25-75% band, log y-axis."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=110)
    for k, (label, s) in enumerate(series.items()):
        color = COLORS[k % len(COLORS)]
        for t in s["traces"]:  # every seed; NOT padded past where the run ended
            ax.plot(range(1, len(t) + 1), [max(v, REGRET_FLOOR) for v in t], color=color, lw=0.9, alpha=0.25)
        m = len(s["median"])
        if m:
            xs = range(1, m + 1)
            ax.fill_between(xs, [max(v, REGRET_FLOOR) for v in s["lo"]], [max(v, REGRET_FLOOR) for v in s["hi"]],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, [max(v, REGRET_FLOOR) for v in s["median"]], color=color, lw=2.4,
                    label=f"{label} (n={s['n_runs']})")
        else:
            ax.plot([], [], color=color, lw=2.4, label=f"{label} (n={s['n_runs']})")
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best regret so far (log)")
    ax.set_title(group)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return fig


def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    _plt().close(fig)
    return buf.getvalue()


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def build_html(merged: dict[str, dict[str, dict[int, list[float]]]], title: str,
               fig_dir: Path | None = None) -> tuple[str, dict]:
    """Returns (html, summary-json-able dict). With `fig_dir`, also writes <fig_dir>/<group>.png."""
    summary: dict = {}
    sections, overview_rows = [], []
    for group, backends in merged.items():
        series = {label: aggregate(runs) for label, runs in backends.items()}
        summary[group] = {
            label: {k: s[k] for k in ("n_runs", "seeds", "n_full_budget", "evals_median",
                                      "final_median", "final_q25", "final_q75")}
            for label, s in series.items()
        }
        png = _png_bytes(render_figure(group, series))
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            (fig_dir / f"{group}.png").write_bytes(png)
        img = f'<img alt="{html.escape(group)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
        rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{s['n_runs']}</td><td>{s['n_full_budget']}/{s['n_runs']}</td>"
            f"<td>{_fmt(s['evals_median'])}</td><td>{_fmt(s['final_median'])}</td>"
            f"<td>{_fmt(s['final_q25'])} &ndash; {_fmt(s['final_q75'])}</td></tr>"
            for label, s in series.items()
        )
        sections.append(
            f"<h2>{html.escape(group)}</h2>{img}"
            f"<table><tr><th>backend</th><th>runs</th><th>full budget</th><th>median evals</th>"
            f"<th>final regret (median)</th><th>25&ndash;75%</th></tr>{rows}</table>"
        )
        overview_rows.append((group, min(series, key=lambda b: series[b]["final_median"]), series))
    labels = sorted({b for g in merged.values() for b in g})
    overview = ""
    if merged:
        head = "".join(f"<th>{html.escape(b)}</th>" for b in labels)
        body = "".join(
            f"<tr><td>{html.escape(g)}</td>"
            + "".join(f"<td>{_fmt(sr[b]['final_median'])} (n={sr[b]['n_runs']})</td>" if b in sr else "<td></td>" for b in labels)
            + f"<td>{html.escape(best)}</td></tr>"
            for g, best, sr in overview_rows
        )
        overview = (f"<h2>Overview: final best regret, median over seeds (lower is better)</h2>"
                    f"<table><tr><th>benchmark</th>{head}<th>lowest median</th></tr>{body}</table>")
    note = ("<p class=note>Faint lines: individual seeds. Solid line: median across seeds. Shaded: 25&ndash;75% band. "
            "Runs that ended early are not padded; the median/band is drawn only while at least half the runs reached "
            "that evaluation. Few seeds means wide uncertainty: read the band, not just the line.</p>")
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
    args = ap.parse_args()
    roots = [Path(r) for r in args.root]
    for r in roots:
        if not r.is_dir():
            ap.error(f"--root {r} is not a directory")
    out = Path(args.out) if args.out else roots[0] / "seeds.html"
    page, summary = build_html(merge_roots(roots), args.title, fig_dir=out.with_suffix(""))
    out.write_text(page)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}, {out.with_suffix('.json').name} and PNGs in {out.with_suffix('')}/ ({len(summary)} benchmark group(s))")


if __name__ == "__main__":
    main()
