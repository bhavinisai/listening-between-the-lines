#!/usr/bin/env python3
"""
Publication figures (PNG, 300 dpi) for the guest-gender analysis, from the
CSVs written by gender_analysis.py. Light-mode, white background.

    fig1_type_shares_by_guest_gender.png   shares with episode-bootstrap 95% CI
    fig2_forest_guest_gender.png           odds ratios, episode-clustered vs host-fixed-effects
    fig3_sensitivity_personal_professional.png   robustness variants
    fig4_within_host_differences.png       per-host female-minus-male guest differences
    fig5_misclassification_corrected.png   observed vs corrected host-standardised difference
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = ["closed", "open", "leading", "personal", "professional", "challenge"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
Z_BONF = 2.638   # two-sided z for alpha = 0.05/6
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, d, name):
    fig.savefig(os.path.join(d, name), dpi=300, facecolor="white")
    plt.close(fig)


def fig1(a, d):
    ci = pd.read_csv(f"{a}/shares_guest_bootci.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(T))[::-1]
    for g, col, off, lab in [("female", BLUE, 0.17, "Female guests"), ("male", ORANGE, -0.17, "Male guests")]:
        s = ci[ci.group == g].set_index("type").loc[T]
        ax.errorbar(s.share * 100, y + off, xerr=[(s.share - s.lo) * 100, (s.hi - s.share) * 100], fmt="o", ms=5,
                    color=col, ecolor=col, elinewidth=1.4, capsize=2.5, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([t.capitalize() for t in T])
    ax.set_xlabel("Share of host questions (%)\npoint = pooled share; bars = 95% CI (episode bootstrap)")
    ax.set_title("Question-type distribution by guest gender", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, d, "fig1_type_shares_by_guest_gender.png")


def fig2(a, d):
    g = pd.read_csv(f"{a}/gee_main.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(T))[::-1]
    for model, col, off, lab, fill in [("M1_episode_cluster", MUTED, 0.17, "Pooled (episode-clustered)", "none"),
                                       ("M2_hostFE_episode_cluster", BLUE, -0.17, "Within-host (host fixed effects)", BLUE)]:
        s = g[g.model == model].set_index("type").loc[T]
        z = np.log(s.OR.values); se = s.se.values
        lo, hi = np.exp(z - Z_BONF * se), np.exp(z + Z_BONF * se)
        ax.errorbar(s.OR, y + off, xerr=[s.OR - lo, hi - s.OR], fmt="o", ms=5, mfc=fill, mec=col, color=col, ecolor=col,
                    elinewidth=1.4, capsize=2.5, label=lab)
    ax.axvline(1, color=INK, lw=0.9); ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6, 2.0]); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6", "2.0"])
    ax.set_yticks(y); ax.set_yticklabels([t.capitalize() for t in T])
    ax.set_xlabel("Odds ratio, female vs male guest (>1 = asked more often of women)\nbars = Bonferroni-adjusted 99.2% CI")
    ax.set_title("Guest-gender effect on question type", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, d, "fig2_forest_guest_gender.png")


def fig3(a, d):
    th = pd.read_csv(f"{a}/sens_threshold.csv"); so = pd.read_csv(f"{a}/sens_soft.csv"); mi = pd.read_csv(f"{a}/sens_mi.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, t in zip(axes, ["personal", "professional"]):
        rows = []
        for thr in [0.0, 0.5, 0.6, 0.7]:
            r = th[(th.type == t) & (th.threshold == thr)].iloc[0]
            lab = "All predictions" if thr == 0 else f"Confidence ≥ {thr:g}  (n={r.n_questions/1000:.1f}k)"
            rows.append((lab, r.OR, r.lo, r.hi))
        r = so[so.type == t].iloc[0]; rows.append(("Soft labels (probabilities)", r.OR, r.lo, r.hi))
        r = mi[mi.type == t].iloc[0]; rows.append(("Multiple imputation (30 draws)", r.OR, r.lo, r.hi))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, o, lo, hi) in zip(y, rows):
            ax.errorbar(o, yi, xerr=[[o - lo], [hi - o]], fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=2.5)
        ax.axvline(1, color=INK, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(t.capitalize(), loc="left", fontsize=10, fontweight="bold"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Odds ratio, female vs male guest")
    fig.suptitle("Robustness of the personal / professional findings (host fixed effects; bars = 95% CI)", x=0.01, y=1.04, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, d, "fig3_sensitivity_personal_professional.png")


def fig4(a, d):
    w = pd.read_csv(f"{a}/within_host.csv")
    if w.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    hosts = (w.drop_duplicates("host").sort_values("n_ep_female", ascending=False).host.tolist())
    for ax, t in zip(axes, ["personal", "professional"]):
        s = w[w.type == t].set_index("host").loc[hosts]
        y = np.arange(len(hosts))[::-1]
        ax.errorbar(s["diff"] * 100, y, xerr=[(s["diff"] - s.lo) * 100, (s.hi - s["diff"]) * 100], fmt="o", ms=5, color=BLUE,
                    ecolor=BLUE, elinewidth=1.4, capsize=2.5)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([f"{h}  ({int(r.n_ep_female)}F/{int(r.n_ep_male)}M)" for h, r in s.iterrows()])
        ax.set_title(t.capitalize(), loc="left", fontsize=10, fontweight="bold"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Female − male guests, mean per-episode share (pp)")
    fig.suptitle("Within-host differences (episodes per gender in brackets; bars = 95% bootstrap CI)", x=0.01, y=1.04, ha="left",
                 fontsize=10, fontweight="bold")
    save(fig, d, "fig4_within_host_differences.png")


def fig5(a, d):
    c = pd.read_csv(f"{a}/sens_corrected.csv").set_index("type").loc[T]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(T))[::-1]
    for col, off, lab, k, lo, hi, fill in [(MUTED, 0.17, "Observed (classifier output)", "observed_diff", "obs_lo", "obs_hi", "none"),
                                           (BLUE, -0.17, "Corrected for classifier error", "corrected_diff", "cor_lo", "cor_hi", BLUE)]:
        ax.errorbar(c[k] * 100, y + off, xerr=[(c[k] - c[lo]) * 100, (c[hi] - c[k]) * 100], fmt="o", ms=5, mfc=fill, mec=col,
                    color=col, ecolor=col, elinewidth=1.4, capsize=2.5, label=lab)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([t.capitalize() for t in T])
    ax.set_xlabel("Female − male guests, host-standardised share difference (pp; 95% bootstrap CI)")
    ax.set_title("Effect of classifier error on the gender difference", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, d, "fig5_misclassification_corrected.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(a.analysis_dir, a.fig_dir)
    print("[OK] figures ->", a.fig_dir)


if __name__ == "__main__":
    main()
