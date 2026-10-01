#!/usr/bin/env python3
"""
plot_style_report.py

Report figures for the v2 conversational style features by gender dyad
(all 418 episodes), from src/stat_analysis/style_dyad_tests_v2.py output:

  style_rates                 hedges, boosters, factuality per 1,000 words:
                              host and guest median per dyad with 95%
                              bootstrap CI (three panels, shared dyad axis)
  style_questions_deference   host direct questions per 1,000 words (median,
                              95% CI) and the share of episodes in which host
                              and guest use deference at all
  style_rates_iqr,            the same two figures with the interquartile
  style_questions_deference_iqr  range instead of the CI (descriptive versions)

Usage:
  python src/conversational_style/plot_style_report.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
LABELS = {"MALE->MALE": "Male host\nmale guest", "MALE->FEMALE": "Male host\nfemale guest",
          "FEMALE->MALE": "Female host\nmale guest", "FEMALE->FEMALE": "Female host\nfemale guest"}
HOST, GUEST = "#2a78d6", "#eb6834"                 # categorical slots 1 and 2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
Y = np.arange(len(DYADS))[::-1]
OFF = 0.14                                         # host above, guest below the dyad line


def style(ax, labels=True):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(Y)
    if labels:
        ax.set_yticklabels([LABELS[d] for d in DYADS])
    else:
        ax.tick_params(axis="y", labelleft=False)   # shared axis: hide, don't clear, the labels
    ax.set_ylim(Y.min() - 0.55, Y.max() + 0.55)


INTERVALS = {"ci": ("median_ci_low", "median_ci_high", "Median per 1,000 words (95% CI)"),
             "iqr": ("q1", "q3", "Median per 1,000 words (IQR)")}


def ci_points(ax, s, metric, color, dy_off, interval="ci"):
    lo, hi, _ = INTERVALS[interval]
    t = s[s.metric == metric].set_index("dyad").loc[DYADS]
    ax.errorbar(t["median"], Y + dy_off, xerr=[t["median"] - t[lo], t[hi] - t["median"]],
                fmt="o", color=color, ecolor=color, elinewidth=1.8, capsize=3, capthick=1.4, markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)


def legend(fig, handles):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.2, 0.995), ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.4, borderaxespad=0)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)


def marker(color, label):
    return Line2D([], [], marker="o", color=color, lw=1.8, markersize=6, markeredgecolor="white", label=label)


def rates_figure(s, path, interval="ci"):
    panels = [("hy_hedge", "Hedges", 12), ("hy_booster", "Boosters", 9), ("ck_factuality", "Factuality", 3)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.12})
    for i, (ax, (f, title, xmax)) in enumerate(zip(axes, panels)):
        ci_points(ax, s, f"host_{f}_per_1k", HOST, OFF, interval)
        ci_points(ax, s, f"guest_{f}_per_1k", GUEST, -OFF, interval)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        style(ax, labels=(i == 0))
    fig.supxlabel(INTERVALS[interval][2], fontsize=9, color=INK2, y=0.02)
    legend(fig, [marker(HOST, "Host"), marker(GUEST, "Guest")])
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.16)
    save(fig, path)


def questions_deference_figure(s, path, interval="ci"):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True,
                               gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1]})
    ci_points(a, s, "host_ck_direct_question_per_1k", HOST, 0, interval)
    a.set_xlim(0, 8 if interval == "ci" else 10)
    a.set_title("Host direct questions", loc="left", fontsize=9.5, color=INK)
    a.set_xlabel(INTERVALS[interval][2])
    style(a)

    h = 0.26                                      # < 2 * OFF, so host and guest bars do not touch
    for role, color, off in [("host", HOST, OFF), ("guest", GUEST, -OFF)]:
        t = s[s.metric == f"{role}_ck_deference_per_1k"].set_index("dyad").loc[DYADS]
        b.barh(Y + off, t.share_nonzero, height=h, color=color, edgecolor="white", linewidth=1.2)
        for yi, v in zip(Y + off, t.share_nonzero):
            b.text(v + 0.012, yi, f"{100 * v:.0f}%", va="center", ha="left", color=INK2, fontsize=7.5)
    b.set_xlim(0, 0.75)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_title("Deference", loc="left", fontsize=9.5, color=INK)
    b.set_xlabel("Episodes in which it is used")
    style(b, labels=False)
    legend(fig, [Patch(color=HOST, label="Host"), Patch(color=GUEST, label="Guest")])
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.17)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Report figures: style features by dyad")
    ap.add_argument("--summary", default="results/stat_analysis/conversational_style/style_dyad_tests_v2_summary.csv")
    ap.add_argument("--fig_dir", default="results/figures/conversational_style")
    args = ap.parse_args()
    s = pd.read_csv(args.summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    rates_figure(s, f"{args.fig_dir}/style_rates")
    questions_deference_figure(s, f"{args.fig_dir}/style_questions_deference")
    rates_figure(s, f"{args.fig_dir}/style_rates_iqr", interval="iqr")
    questions_deference_figure(s, f"{args.fig_dir}/style_questions_deference_iqr", interval="iqr")
    print(f"Saved {args.fig_dir}/style_rates, style_questions_deference and their _iqr versions (.pdf/.png)")


if __name__ == "__main__":
    main()
