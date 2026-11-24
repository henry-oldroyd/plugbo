#!/usr/bin/env python3
"""Build results/report/tooluse_report.pdf: explains regret_free_vs_vanilla.png using agent tool-use logs."""
from __future__ import annotations

import collections
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).parent))
from tooluse_stats import CATS, SET_VERBS, load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
REGRET = ROOT / "results" / "figures" / "regret_free_vs_vanilla.png"
COL = {"suggest": "#1f77b4", "submit": "#2ca02c", "oracle": "#d62728", "reconfigure (set-*)": "#9467bd",
       "inspect": "#8c564b", "other": "#7f7f7f", "forced reprompt": "#ff7f0e"}
ALLC = CATS[:-1] + ["forced reprompt"]
BINS = [(7, 12), (12, 17), (17, 22), (22, 27), (27, 32), (32, 37), (37, 42), (42, 50)]
data = load_all()
names = list(data)
EXPCOL = ["#e8683a", "#f0a000"]


def tot(runs, c):
    return sum(r["tot"].get(c, 0) for r in runs)


def lab(lo, hi):
    return f"{lo}-{hi - 1}" if hi < 50 else f"{lo}-50"


def save(fig, name):
    fig.tight_layout()
    p = OUT / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def binned(runs):
    rows = {}
    for lo, hi in BINS:
        alive = [r for r in runs if r["evals"] > lo]
        d = {c: sum(cnt.get(c, 0) for r in alive for ev, cnt in r["per_eval"].items() if lo <= ev < hi) for c in ALLC}
        rows[(lo, hi)] = (len(alive), d)
    return rows


def fig_evals():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, c in zip(names, EXPCOL):
        e = [r["evals"] for r in data[n]]
        ax[0].hist(e, bins=range(7, 52, 3), alpha=.6, label=f"{n} (n={len(e)})", color=c)
        xs = list(range(8, 51))
        ax[1].plot(xs, [100 * sum(v >= x for v in e) / len(e) for x in xs], label=n, color=c, lw=2)
    ax[0].set_xlabel("evaluations used (incl. 7 warm-up)")
    ax[0].set_ylabel("runs")
    ax[0].set_title("How far each agent run got")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("evaluation number")
    ax[1].set_ylabel("% of runs still going")
    ax[1].set_title("Runs still active at each evaluation")
    ax[1].legend(fontsize=8)
    return save(fig, "c_evals.png")


def fig_totals():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, n in enumerate(names):
        b = 0
        for c in ALLC:
            v = tot(data[n], c) / len(data[n])
            ax.bar(i, v, bottom=b, color=COL[c], label=c if i == 0 else None)
            b += v
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("commands per run (mean)")
    ax.set_title("What the agent spends its commands on")
    ax.legend(fontsize=8, ncol=2)
    return save(fig, "c_totals.png")


def fig_time():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, n in zip(axs, names):
        b = binned(data[n])
        xs = [lab(*k) for k in b]
        for c in ALLC:
            ys = [(b[k][1][c] / max(b[k][0], 1)) / (k[1] - k[0]) for k in b]
            ax.plot(xs, ys, marker="o", color=COL[c], label=c, lw=1.8)
        ax.set_title(n, fontsize=10)
        ax.set_xlabel("evaluations completed when command was issued")
        ax.tick_params(axis="x", labelsize=7)
    axs[0].set_ylabel("commands per active run per evaluation")
    axs[0].legend(fontsize=7)
    return save(fig, "c_time.png")


def fig_share():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, n in zip(axs, names):
        b = binned(data[n])
        xs = [lab(*k) for k in b]
        bot = [0] * len(b)
        for c in ALLC:
            ys = [100 * b[k][1][c] / max(sum(b[k][1].values()), 1) for k in b]
            ax.bar(xs, ys, bottom=bot, color=COL[c], label=c)
            bot = [x + y for x, y in zip(bot, ys)]
        ax.set_title(n, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("evaluations completed when command was issued")
    axs[0].set_ylabel("% of commands")
    axs[0].legend(fontsize=7, loc="lower left")
    return save(fig, "c_share.png")


def fig_q():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    qa = sorted({q for n in names for r in data[n] for q in r["qs"]})
    w = .38
    for i, (n, c) in enumerate(zip(names, EXPCOL)):
        t = collections.Counter()
        for r in data[n]:
            t.update(r["qs"])
        s = sum(t.values()) or 1
        ax.bar([j + i * w for j in range(len(qa))], [100 * t[q] / s for q in qa], w, label=n, color=c)
    ax.set_xticks([j + w / 2 for j in range(len(qa))])
    ax.set_xticklabels(qa)
    ax.set_xlabel("--q (points requested per suggest call)")
    ax.set_ylabel("% of suggest calls")
    ax.set_title("Batch size of suggest calls")
    ax.legend(fontsize=8)
    return save(fig, "c_q.png")


def fig_verbs():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    v = collections.Counter()
    for r in data[names[1]]:
        v.update(r["verbs"])
    ks = [k for k in SET_VERBS if v[k]] or ["(none)"]
    ax.barh(ks, [v[k] for k in ks], color="#9467bd")
    ax.invert_yaxis()
    for i, k in enumerate(ks):
        ax.text(v[k], i, f" {v[k]}", va="center", fontsize=8)
    ax.set_title("Reconfiguration commands used (Exp 2, all runs)")
    ax.set_xlabel("calls")
    return save(fig, "c_verbs.png")


charts = dict(evals=fig_evals(), totals=fig_totals(), time=fig_time(), share=fig_share(), q=fig_q(), verbs=fig_verbs())

e2 = [r["evals"] for r in data[names[1]]]
n2 = len(e2)
full2 = sum(x >= 50 for x in e2)
fc2 = [r["tot"].get("forced reprompt", 0) for r in data[names[1]]]
cap2 = sum(x >= 30 for x in fc2)
tq = collections.Counter()
for r in data[names[1]]:
    tq.update(r["qs"])
batch_share = 100 * sum(v for q, v in tq.items() if q > 1) / max(sum(tq.values()), 1)

ss = getSampleStyleSheet()
H1, H2 = ss["Title"], ss["Heading2"]
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=6)
SM = ParagraphStyle("sm", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))


def img(p, w=17):
    from PIL import Image as PI
    iw, ih = PI.open(p).size
    return Image(str(p), width=w * cm, height=w * cm * ih / iw)


def table(rows, widths=None, fs=8):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                           ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    return t


S = [
    Paragraph("Why does the agent fall behind vanilla BO?", H1),
    Paragraph("Explaining the regret chart from the agents' tool-use logs (Hartmann-6, qwen3.6:27b, budget 50)", ss["Heading3"]),
    Paragraph("<b>Aim.</b> The chart below is the headline result: vanilla Bayesian optimisation reaches a final regret of about 0.35, "
              "while the agents stall around 1.3-1.4. This document uses the agents' command logs to explain the shape of the chart. "
              "Everything here is descriptive: it shows <i>what the agents did</i>, not proof of why it hurt performance.", B),
    img(REGRET, 17),
    Paragraph("Regret = best value found minus the hidden true optimum (lower is better). Lines are means over paired seeds, shaded band = "
              "+/-1 standard error of the mean. All lines start from the same 7 Sobol warm-up points per seed.", SM),
    Paragraph("How to read it, in four steps", H2),
    Paragraph("<b>1. Evaluations 1-20: the three lines are the same.</b> The first 7 points are shared, and for the next ~13 evaluations a 6-D "
              "Gaussian process has so little data that it is a weak model, so no strategy has much of an edge.", B),
    Paragraph("<b>2. Evaluations ~20-50: vanilla keeps improving, the agents flatten.</b> Vanilla refits its GP and picks the next point after "
              "<i>every</i> evaluation, all 50 times. The agent lines level off around evaluation 30.", B),
    Paragraph(f"<b>3. The flat tail is largely runs that ended.</b> The agent is told to use 50 evaluations but nothing hard-enforces it. "
              f"Only {full2} of {n2} free+hint runs ({100 * full2 / n2:.0f}%) reached 50 evaluations; the median run stopped at "
              f"{st.median(e2):.0f} (including the 7 warm-up points). When a run stops, the plot carries its best value forward as a flat line "
              f"to evaluation 50, so a growing share of the orange line after ~30 is 'runs that are no longer searching'. The chart effectively "
              f"compares 50 vanilla evaluations to about {st.mean(e2):.0f} agent evaluations.", B),
    Paragraph(f"<b>4. Why they stop.</b> When the model ends a turn without a command, the harness replies 'budget not exhausted, keep going' "
              f"(a <i>forced reprompt</i>), capped at 30 per run. {cap2} of {n2} runs ({100 * cap2 / n2:.0f}%) used all 30, so the cap, rather than "
              f"the model deciding it was finished, usually ended the run. The sections below show how the commands look over the course of a run.", B),
    Paragraph("Caveat: this does not show whether the agents would have caught vanilla with a full 50 evaluations. The recommended test is to "
              "raise or remove the reprompt cap and rerun.", B),
    PageBreak(),
    Paragraph("1. How far did the agents get?", H2),
    img(charts["evals"], 17),
    Paragraph("Left: histogram of evaluations used per run (the 7 warm-up points are always included). Right: percentage of runs still going "
              "at each evaluation number. Vanilla BO is always at 100% to evaluation 50.", SM),
]


def row(n, runs):
    e = [r["evals"] for r in runs]
    return [n, len(runs), f"{st.median(e):.0f}", f"{st.mean(e):.1f}", min(e), max(e), sum(x >= 50 for x in e),
            sum(r["tot"].get("forced reprompt", 0) >= 30 for r in runs)]


S += [table([["Experiment", "runs", "median evals", "mean evals", "min", "max", "reached 50", "hit cap (30)"]] +
            [row(n, data[n]) for n in names], [4.6 * cm, 1.3 * cm, 2 * cm, 2 * cm, 1.1 * cm, 1.1 * cm, 2 * cm, 2.6 * cm]),
      Paragraph("Run counts are every logged run (the regret chart uses only paired seeds, n=63 and n=166), so figures differ slightly.", SM),
      Paragraph("2. What the agent spends its commands on", H2),
      img(charts["totals"], 13)]
rows = [["command"] + [f"{n} total" for n in names] + [f"{n} per run" for n in names]]
for c in ALLC:
    rows.append([c] + [f"{tot(data[n], c):,}" for n in names] + [f"{tot(data[n], c) / len(data[n]):.1f}" for n in names])
S += [table(rows, [3.4 * cm, 3.6 * cm, 3.6 * cm, 3.4 * cm, 3.4 * cm], 7),
      Paragraph("<b>suggest</b> = lenz suggest (ask the optimiser for candidate points). <b>submit</b> = lenz submit (record an evaluated point). "
                "<b>oracle</b> = ./oracle (the real evaluation; this is what spends budget). <b>reconfigure</b> = any lenz set-*. "
                "<b>inspect</b> = status/trials/best etc. <b>forced reprompt</b> = the harness's 'keep going' message (not a command by the agent). "
                "One shell call can hold several commands, so these are counts of commands, not calls.", SM),
      PageBreak(),
      Paragraph("3. Commands over time (like the paper's frequency-over-time plots)", H2),
      Paragraph("X-axis: evaluations completed when the command was issued (7 = just after warm-up). Y-axis: commands per still-active run per "
                "evaluation step, so lines are not pulled down just because runs end.", B),
      img(charts["time"], 17), Spacer(1, 6), img(charts["share"], 17),
      Paragraph("Second figure: the same data as a share of all commands in each phase.", SM),
      Paragraph("<b>What the time plot shows.</b> (a) Oracle and submit calls stay near one per evaluation throughout, as they must. "
                "(b) <b>suggest</b> calls fall steadily and are near zero after about evaluation 35: late in a run the agent is still "
                "evaluating points but rarely asks the optimiser for new ones, so those points come from an earlier batch or from the agent's own "
                "choice (the logs here do not say which; checking that is a follow-up). (c) <b>Forced reprompts</b> are rare early, then rise above "
                "one per evaluation from about evaluation 17-31: the model keeps ending its turn without acting and has to be pushed. "
                "The reprompt budget (30 per run) is used up in this window, which is why runs end around evaluation 30. "
                "(d) Reconfiguration (Exp 2) happens almost entirely in the first few evaluations.", B)]
for n in names:
    b = binned(data[n])
    rr = [["evals", "active runs"] + ALLC] + [[lab(lo, hi), b[(lo, hi)][0]] + [b[(lo, hi)][1][c] for c in ALLC] for lo, hi in BINS]
    S += [Paragraph(f"Raw counts: {n}", B), table(rr, [1.8 * cm, 2.2 * cm] + [2.0 * cm] * len(ALLC), 6.5)]
S += [Paragraph("Counts are totals in each evaluation window across runs still active (active = ended after the window started).", SM),
      PageBreak(),
      Paragraph("4. Batch suggestions", H2),
      img(charts["q"], 12),
      Paragraph(f"In Exp 2, {batch_share:.0f}% of suggest calls asked for more than one point (q&gt;1). A batch is chosen jointly without seeing "
                "the results of its own members, whereas vanilla BO refits after every point. That is one plausible (untested) reason the agent "
                "gains less per evaluation.", B)]
qa = sorted({q for n in names for r in data[n] for q in r["qs"]})
cq = []
for n in names:
    t = collections.Counter()
    for r in data[n]:
        t.update(r["qs"])
    cq.append(t)
S += [table([["--q"] + [f"{n} (calls)" for n in names]] + [[q] + [cq[i][q] for i in range(len(names))] for q in qa], [2 * cm, 5 * cm, 5 * cm]),
      Paragraph("5. Reconfiguration (Exp 2 only)", H2),
      img(charts["verbs"], 11)]
v = collections.Counter()
used = collections.Counter()
for r in data[names[1]]:
    v.update(r["verbs"])
    for k in r["verbs"]:
        used[k] += 1
S += [table([["verb", "total calls", f"runs using it (of {n2})"]] + [[k, v[k], used[k]] for k in SET_VERBS if v[k]], [4 * cm, 3 * cm, 4 * cm]),
      Paragraph("Almost every free run changed the acquisition function (set-acqf); the other levers were rarely touched. In single traces "
                "the agent switched to logei (citing the hint that the function is noise-free) and to sobol (random exploration).", B),
      Paragraph("6. Caveats", H2),
      Paragraph("One benchmark (Hartmann-6), one quantised local model, one RTX 4090. Exp 1 and Exp 2 differ in more than optimiser freedom "
                "(Exp 2 adds the hint and a longer command timeout), and both used forced continuation. Counts come from regular-expression "
                "matching on logged commands. Nothing here isolates a cause; the main testable follow-up is rerunning with the reprompt cap "
                "removed so that every run reaches 50 evaluations.", B)]
SimpleDocTemplate(str(OUT / "tooluse_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="Tool-use report").build(S)
print(OUT / "tooluse_report.pdf")
