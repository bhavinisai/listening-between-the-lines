#!/usr/bin/env python3
"""
plot_dialogue_act_report.py

Report figures for dialogue acts by gender dyad (all 418 episodes), from
src/stat_analysis/dialogue_act_dyad_tests_v2.py output, in the same layout as
plot_style_report.py: host and guest median share of segments per dyad with
95% bootstrap CIs.

  dialogue_acts_main    ask, answer, say
  dialogue_acts_minor   acknowledge, intent, reply_yes

Usage:
  python src/dialogue_act/plot_dialogue_act_report.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from plot_style_report import GUEST, HOST, INK, INK2, MUTED, OFF, ci_points, legend, marker, save, style  # noqa: E402


def acts_figure(s, panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.24})
    for i, (ax, (act, title, xmax)) in enumerate(zip(axes, panels)):
        ci_points(ax, s, f"host_{act}", HOST, OFF)
        ci_points(ax, s, f"guest_{act}", GUEST, -OFF)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        style(ax, labels=(i == 0))
    fig.supxlabel("Median share of the speaker's segments (95% CI)", fontsize=9, color=INK2, y=0.02)
    legend(fig, [marker(HOST, "Host"), marker(GUEST, "Guest")])
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.16)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Report figures: dialogue acts by dyad")
    ap.add_argument("--summary", default="results/stat_analysis/dialogue_act/dialogue_act_dyad_tests_v2_summary.csv")
    ap.add_argument("--fig_dir", default="results/figures/dialogue_act")
    args = ap.parse_args()
    s = pd.read_csv(args.summary)
    s["metric"] = s.role.str.lower() + "_" + s.act
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    acts_figure(s, [("ask", "Ask (questions)", 0.35), ("answer", "Answer", 0.6), ("say", "Say (statements)", 0.7)],
                f"{args.fig_dir}/dialogue_acts_main")
    acts_figure(s, [("acknowledge", "Acknowledge", 0.08), ("intent", "Intent", 0.06),
                    ("reply_yes", "Reply yes", 0.04)], f"{args.fig_dir}/dialogue_acts_minor")
    print(f"Saved {args.fig_dir}/dialogue_acts_main.pdf/.png and dialogue_acts_minor.pdf/.png")


if __name__ == "__main__":
    main()
