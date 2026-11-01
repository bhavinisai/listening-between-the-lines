#!/usr/bin/env python3
"""
plot_interruption_report.py

Report figures for the interruption analysis by gender dyad (episodes 331-418),
from results/interruptions/ (detect_interruptions.py output):

  interruption_rates       successful interruptions, unsuccessful interruptions
                           and backchannels made by the host and by the guest,
                           per 10 minutes of the other speaker's speech: median
                           per dyad with 95% bootstrap CI (three panels)
  interruption_rates_iqr   the same with the interquartile range (descriptive)
  overlap_type_shares      share of each overlap type among all overlap events
                           started by hosts and by guests (pooled episodes)

Style matches src/conversational_style/plot_style_report.py.

Usage:
  python src/interruptions/plot_interruption_report.py
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
TYPE_LABELS = {"successful_interruption": "Successful\ninterruption",
               "unsuccessful_interruption": "Unsuccessful\ninterruption",
               "terminal_overlap": "Terminal\noverlap", "backchannel": "Backchannel",
               "simultaneous_start": "Simultaneous\nstart"}
XLABEL = {"ci": "Median per 10 min of the other speaker's speech (95% CI)",
          "iqr": "Median per 10 min of the other speaker's speech (IQR)"}


def style(ax, labels=True):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(Y)
    if labels:
        ax.set_yticklabels([LABELS[d] for d in DYADS])
        ax.set_ylabel("Gender dyad (host → guest)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(Y.min() - 0.55, Y.max() + 0.55)


def median_interval(x, interval, rng, n_boot=5000):
    x = np.asarray(x.dropna())
    med = np.median(x)
    if interval == "iqr":
        return med, np.quantile(x, 0.25), np.quantile(x, 0.75)
    boots = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return med, np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def points(ax, d, metric, color, dy_off, interval, rng):
    stats = np.array([median_interval(d.loc[d.dyad == dy, metric], interval, rng) for dy in DYADS])
    med, lo, hi = stats.T
    ax.errorbar(med, Y + dy_off, xerr=[med - lo, hi - med], fmt="o", color=color, ecolor=color,
                elinewidth=1.8, capsize=3, capthick=1.4, markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    return hi.max()


def legend(fig, handles):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.24, 0.995), ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.4, borderaxespad=0)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)


def marker(color, label):
    return Line2D([], [], marker="o", color=color, lw=1.8, markersize=6, markeredgecolor="white", label=label)


def rates_figure(d, path, interval="ci", seed=0):
    rng = np.random.default_rng(seed)
    panels = [("successful_interruption", "Successful interruptions"),
              ("unsuccessful_interruption", "Unsuccessful interruptions"),
              ("backchannel", "Backchannels")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.12})
    for i, (ax, (k, title)) in enumerate(zip(axes, panels)):
        top = max(points(ax, d, f"host_{k}_per10", HOST, OFF, interval, rng),
                  points(ax, d, f"guest_{k}_per10", GUEST, -OFF, interval, rng))
        ax.set_xlim(0, np.ceil(top / 5) * 5 + 1)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        style(ax, labels=(i == 0))
    fig.supxlabel(XLABEL[interval], fontsize=9, color=INK2, y=0.02)
    legend(fig, [marker(HOST, "Made by host"), marker(GUEST, "Made by guest")])
    fig.subplots_adjust(left=0.19, right=0.98, top=0.84, bottom=0.16)
    save(fig, path)


def type_shares_figure(events, path):
    events = events[events.overlapper_role.isin(["HOST", "GUEST"])]
    types = list(TYPE_LABELS)
    share = (events.groupby("overlapper_role").type.value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=types))
    n = events.overlapper_role.value_counts()
    y = np.arange(len(types))[::-1]
    h = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for role, color, off in [("HOST", HOST, h / 2 + 0.02), ("GUEST", GUEST, -h / 2 - 0.02)]:
        ax.barh(y + off, share.loc[role], height=h, color=color, edgecolor="white", linewidth=1.2)
        for yi, v in zip(y + off, share.loc[role]):
            ax.text(v + 0.008, yi, f"{100 * v:.0f}%", va="center", ha="left", color=INK2, fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABELS[t] for t in types])
    ax.set_xlim(0, 0.7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of overlap events started by that speaker")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend(fig, [Patch(color=HOST, label=f"Started by host (n = {n['HOST']:,})"),
                 Patch(color=GUEST, label=f"Started by guest (n = {n['GUEST']:,})")])
    fig.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.17)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Report figures: interruptions by dyad")
    ap.add_argument("--rates", default="results/interruptions/interruption_rates_by_episode.csv")
    ap.add_argument("--events", default="results/interruptions/interruption_events.csv")
    ap.add_argument("--qc", default="results/interruptions/episode_qc.csv")
    ap.add_argument("--fig_dir", default="results/figures/interruptions")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})

    d = pd.read_csv(args.rates)
    d = d[d.dyad.isin(DYADS)]
    # same episodes as the rates: drop those whose speaker mapping failed QC
    qc = pd.read_csv(args.qc)
    ok = set(qc.loc[qc.mapping_ok.astype(bool), "episode_id"])
    events = pd.read_csv(args.events)
    events = events[events.episode_id.isin(ok) & events.episode_id.isin(d.episode_id)]

    rates_figure(d, f"{args.fig_dir}/interruption_rates")
    rates_figure(d, f"{args.fig_dir}/interruption_rates_iqr", interval="iqr")
    type_shares_figure(events, f"{args.fig_dir}/overlap_type_shares")
    print(f"Saved {args.fig_dir}/interruption_rates(_iqr) and overlap_type_shares (.pdf/.png) "
          f"from {len(d)} episodes, {len(events):,} events")


if __name__ == "__main__":
    main()
