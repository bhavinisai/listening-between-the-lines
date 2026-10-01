#!/usr/bin/env python3
"""
plot_dialogue_act_composition.py

Report figure: how host and guest speech is divided between dialogue acts in
each host->guest gender dyad (all 418 episodes). One 100% stacked bar per
dyad and speaker: ask, answer, say, and all other acts combined.

Shares are the mean over episodes of each speaker's share of segments with
that act (from results/dialogue_acts/all_418/dialogue_act_rates_by_episode_role.csv),
so each bar sums to 100%; medians would not.

Usage:
  python src/dialogue_act/plot_dialogue_act_composition.py
  python src/dialogue_act/plot_dialogue_act_composition.py --width 5.5 --height 3.2 --dpi 200 \
      --out results/figures/dialogue_act/dialogue_acts_composition_small
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
DYAD_LABEL = {"MALE->MALE": "Male host, male guest", "MALE->FEMALE": "Male host, female guest",
              "FEMALE->MALE": "Female host, male guest", "FEMALE->FEMALE": "Female host, female guest"}
ACTS = [("ask", "Ask", "#2a78d6"), ("answer", "Answer", "#eb6834"), ("say", "Say", "#1baf7a"),
        ("other", "Other acts", "#c9c8c2")]          # categorical slots 1-3, neutral grey for the rest
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"


def main():
    ap = argparse.ArgumentParser(description="Dialogue-act composition by dyad and speaker")
    ap.add_argument("--rates", default="results/dialogue_acts/all_418/dialogue_act_rates_by_episode_role.csv")
    ap.add_argument("--dyads", default="results/features/episode_dyads.csv")
    ap.add_argument("--out", default="results/figures/dialogue_act/dialogue_acts_composition")
    ap.add_argument("--width", type=float, default=7.2, help="Figure width in inches")
    ap.add_argument("--height", type=float, default=3.9, help="Figure height in inches")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    args = ap.parse_args()

    r = pd.read_csv(args.rates).merge(pd.read_csv(args.dyads)[["episode_id", "dyad"]], on="episode_id")
    r = r[r.role.isin(["HOST", "GUEST"]) & r.dyad.isin(DYADS)]
    labels = [c for c in r.columns if c not in ("episode_id", "role", "dyad")]
    r["other"] = r[[c for c in labels if c not in ("ask", "answer", "say")]].sum(axis=1)
    comp = r.groupby(["dyad", "role"])[["ask", "answer", "say", "other"]].mean()
    comp = comp.div(comp.sum(axis=1), axis=0)          # guard against rounding
    comp.to_csv(f"{args.out}.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    bar_width_in = args.width - 2.0 - 0.35              # plot width in inches (see margins below)
    ypos, ylabels = [], []
    y, h = 0.0, 0.34
    for d in DYADS:
        for role in ["HOST", "GUEST"]:
            left = 0.0
            for key, _, color in ACTS:
                v = comp.loc[(d, role), key]
                ax.barh(y, v, left=left, height=h, color=color, edgecolor="white", linewidth=1.5)
                if v * bar_width_in >= 0.28:           # label only segments wide enough to hold it
                    ax.text(left + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                            color="white" if key != "other" else INK2, fontweight="bold")
                left += v
            ypos.append(y)
            ylabels.append(role.title())
            y -= 0.42
        y -= 0.34                                        # gap between dyads
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8, color=INK2)
    for d, (yh, yg) in zip(DYADS, zip(ypos[0::2], ypos[1::2])):
        # fixed offset in points left of the Host/Guest labels, so it holds at any figure size
        ax.annotate(DYAD_LABEL[d].replace(", ", "\n"), xy=(0, (yh + yg) / 2),
                    xycoords=("axes fraction", "data"), xytext=(-44, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of the speaker's segments (mean over episodes)", color=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in ACTS], loc="upper left",
               bbox_to_anchor=(2.0 / args.width, 0.995), ncol=4, frameon=False, fontsize=8,
               handlelength=1.2)
    # margins in inches, converted to figure fractions
    fig.subplots_adjust(left=2.0 / args.width, right=1 - 0.35 / args.width,
                        top=1 - 0.38 / args.height, bottom=0.5 / args.height)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi)
    print(comp.round(3).to_string())
    print(f"Saved {args.out}.pdf/.png and {args.out}.csv")


if __name__ == "__main__":
    main()
