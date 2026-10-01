#!/usr/bin/env python3
"""
plot_speaking_time_report.py

Report figures for host/guest speaking time by gender dyad:
  speaking_time_minutes             median host and guest speaking minutes per
                                    episode (stacked bars; host and guest
                                    medians are computed separately)
  speaking_time_share_distribution  host share of speaking time: episodes (dots)
                                    with a box of the median and IQR per dyad
                                    (descriptive; no tests)
  speaking_time_share               host share of speaking time: episodes (dots)
                                    and the median per dyad with its 95%
                                    bootstrap CI (speaking_time_dyad_tests_v2.py)

All 418 episodes. Inputs are the outputs of src/speaking_time/speaking_time_v2.py and
src/stat_analysis/speaking_time_dyad_tests_v2.py.

Usage:
  python src/speaking_time/plot_speaking_time_report.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
LABELS = {"MALE->MALE": "Male host\nmale guest", "MALE->FEMALE": "Male host\nfemale guest",
          "FEMALE->MALE": "Female host\nmale guest", "FEMALE->FEMALE": "Female host\nfemale guest"}
HOST, GUEST = "#2a78d6", "#eb6834"                 # categorical slots 1 and 2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
XMAX_SHARE = 0.80
FIGSIZE = (6.0, 2.8)
Y = np.arange(len(DYADS))[::-1]


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(Y)
    ax.set_yticklabels([LABELS[d] for d in DYADS])


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)


def minutes_figure(ep, path):
    bd = ep.groupby("dyad")[["host_min", "guest_min"]].median().loc[DYADS]
    host_m, guest_m = bd.host_min.to_numpy(), bd.guest_min.to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    h = 0.56
    # 2px surface gap between the host and guest segments
    ax.barh(Y, host_m, height=h, color=HOST, edgecolor="white", linewidth=1.5, label="Host")
    ax.barh(Y, guest_m, left=host_m, height=h, color=GUEST, edgecolor="white", linewidth=1.5, label="Guest")
    for yi, hm, gm in zip(Y, host_m, guest_m):
        ax.text(hm / 2, yi, f"{hm:.0f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(hm + gm / 2, yi, f"{gm:.0f}", ha="center", va="center", color="white", fontsize=8,
                fontweight="bold")
    ax.set_xlim(0, 70)
    ax.set_xlabel("Median speaking time per episode (minutes)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8,
              handlelength=1.2, borderaxespad=0.3)
    style(ax)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.88, bottom=0.18)
    save(fig, path)


def share_distribution_figure(ep, path):
    """Descriptive view: per-dyad box (median, IQR, whiskers to 1.5 IQR) over the episodes."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(1)
    data = []
    for yi, d in zip(Y, DYADS):
        s = ep.loc[ep.dyad == d, "host_time_share"].to_numpy()
        data.append(s)
        s = s[s <= XMAX_SHARE]
        ax.scatter(s, yi + rng.uniform(-0.18, 0.18, len(s)), s=7, color=MUTED, alpha=0.3, linewidths=0, zorder=1)
        ax.text(np.median(data[-1]), yi + 0.3, f"{100 * np.median(data[-1]):.1f}%", va="bottom", ha="center",
                color=INK, fontsize=8, zorder=5)
    ax.boxplot(data, positions=Y, orientation="horizontal", widths=0.34, showfliers=False, patch_artist=True, zorder=3,
               boxprops={"facecolor": "none", "edgecolor": HOST, "linewidth": 1.6},
               medianprops={"color": HOST, "linewidth": 2.4},
               whiskerprops={"color": HOST, "linewidth": 1.2}, capprops={"color": HOST, "linewidth": 1.2})
    ax.set_xlim(0, XMAX_SHARE)
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.75)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Host share of total speaking time")
    style(ax)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.18)
    save(fig, path)


def share_figure(ep, sm, path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(1)
    clipped = 0
    for yi, d in zip(Y, DYADS):
        s = ep.loc[ep.dyad == d, "host_time_share"].to_numpy()
        clipped += int((s > XMAX_SHARE).sum())
        s = s[s <= XMAX_SHARE]
        ax.scatter(s, yi + rng.uniform(-0.18, 0.18, len(s)), s=7, color=MUTED, alpha=0.3, linewidths=0, zorder=1)
        r = sm.loc[d]
        med, lo, hi = r.host_share_median, r.host_share_median_ci_low, r.host_share_median_ci_high
        ax.errorbar([med], [yi], xerr=[[med - lo], [hi - med]], fmt="o",
                    color=HOST, ecolor=HOST, elinewidth=2, capsize=4, capthick=1.6, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.text(med, yi + 0.3, f"{100 * med:.1f}%", va="bottom", ha="center", color=INK, fontsize=8, zorder=5)
    ax.set_xlim(0, XMAX_SHARE)
    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.75)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Host share of total speaking time")
    style(ax)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.18)
    save(fig, path)
    return clipped


def main():
    ap = argparse.ArgumentParser(description="Report figures: speaking time by dyad")
    ap.add_argument("--episodes", default="results/dyads/speaking_time_dyads_v2.csv")
    ap.add_argument("--summary", default="results/stat_analysis/speaking_time/speaking_time_dyad_tests_v2_summary.csv")
    ap.add_argument("--fig_dir", default="results/figures/speaking_time")
    args = ap.parse_args()

    ep = pd.read_csv(args.episodes)
    ep = ep[ep.dyad.isin(DYADS)]
    sm = pd.read_csv(args.summary).set_index("dyad").loc[DYADS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    minutes_figure(ep, f"{args.fig_dir}/speaking_time_minutes")
    share_distribution_figure(ep, f"{args.fig_dir}/speaking_time_share_distribution")
    clipped = share_figure(ep, sm, f"{args.fig_dir}/speaking_time_share")
    print(f"Saved {args.fig_dir}/speaking_time_minutes, speaking_time_share_distribution and "
          f"speaking_time_share (.pdf/.png) "
          f"({clipped} episode(s) above {XMAX_SHARE:.0%} not drawn in the share figure)")


if __name__ == "__main__":
    main()
