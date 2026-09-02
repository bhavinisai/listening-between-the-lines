#!/usr/bin/env python3
"""
Publication-ready figures for the topic-control analysis (topic_control.py).
Reads results/topic_control_turns.csv, results/balanced_200_episodes.csv,
and results/topic_control_stats.csv, and writes PNG (300dpi) + vector PDF
figures to results/figures/.

Usage:
    python src/make_topic_control_figures.py
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Validated colorblind-safe categorical pair (adjacent slots 1-2 of the
# dataviz skill's reference palette -- worst adjacent CVD Delta-E 9.1 light).
BLUE = "#2a78d6"    # male
ORANGE = "#eb6834"  # female
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

GENDER_COLOR = {"male": BLUE, "female": ORANGE}


def wald_ci(p, n, z=1.96):
    se = np.sqrt(p * (1 - p) / n)
    return z * se


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {name}.png / {name}.pdf")


def load_mixed_turns():
    turns = pd.read_csv(os.path.join(REPO, "results", "topic_control_turns.csv"))
    episodes = pd.read_csv(os.path.join(REPO, "results", "balanced_200_episodes.csv"))
    dyads = episodes.set_index("episode_id")["dyad"]
    turns["dyad"] = turns["episode_id"].map(dyads)
    mixed = turns[turns["dyad"].isin(["MALE->FEMALE", "FEMALE->MALE"])].copy()
    return mixed


def fig1_headline(mixed, chi2_p):
    rate = mixed.groupby("initiator_gender")["followed_up"].agg(["mean", "count"])
    rate = rate.reindex(["female", "male"])
    fig, ax = plt.subplots(figsize=(4.2, 4.5))

    x = np.arange(len(rate))
    heights = rate["mean"].values
    ns = rate["count"].values
    errs = [wald_ci(p, n) for p, n in zip(heights, ns)]
    colors = [GENDER_COLOR[g] for g in rate.index]

    bars = ax.bar(x, heights, width=0.55, color=colors, yerr=errs,
                   capsize=4, error_kw={"ecolor": MUTED, "elinewidth": 1.2})

    for xi, h, e, n in zip(x, heights, errs, ns):
        ax.text(xi, h + e + 0.012, f"{h:.1%}\n(n={n:,})", ha="center",
                va="bottom", fontsize=9.5, color=INK)

    # significance bracket
    y_top = max(h + e for h, e in zip(heights, errs)) + 0.05
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.01, y_top + 0.01, y_top], color=INK, lw=1)
    ax.text(0.5, y_top + 0.015, sig_stars(chi2_p), ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(["Female initiator", "Male initiator"])
    ax.set_ylabel("Follow-up rate")
    ax.set_ylim(0, y_top + 0.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Topic follow-up rate by initiator gender\n(mixed-gender dyads)", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig1_followup_rate_by_gender")


def fig2_confound(mixed):
    grp = mixed.groupby(["initiator_role", "initiator_gender"])["followed_up"].agg(["mean", "count"])
    roles = ["GUEST", "HOST"]
    genders = ["female", "male"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    x = np.arange(len(roles))

    for i, g in enumerate(genders):
        heights, errs = [], []
        for r in roles:
            if (r, g) in grp.index:
                p, n = grp.loc[(r, g)]
            else:
                p, n = np.nan, 0
            heights.append(p)
            errs.append(wald_ci(p, n) if n > 0 else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, heights, width=width, color=GENDER_COLOR[g],
                       yerr=errs, capsize=4, error_kw={"ecolor": MUTED, "elinewidth": 1.1},
                       label=g.capitalize())
        for xi, h, e, r in zip(x + offset, heights, errs, roles):
            n = grp.loc[(r, g), "count"] if (r, g) in grp.index else 0
            ax.text(xi, h + e + 0.012, f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Guest turns", "Host turns"])
    ax.set_ylabel("Follow-up rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Follow-up rate by initiator gender AND role\n(the apparent gender gap tracks host/guest role)", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig2_gender_by_role_confound")


def fig3_forest(stats_df):
    rows = []
    label_map = {
        "ols_similarity_is_male_initiator": "Male initiator",
        "ols_similarity_is_host_initiator": "Host initiator",
        "ols_similarity_initiator_word_count": "Initiator word count\n(per word)",
    }
    for test_name, label in label_map.items():
        row = stats_df[stats_df["test"] == test_name].iloc[0]
        m = re.search(r"coef=(-?[\d.]+),\s*se=([\d.]+)", row["notes"])
        coef, se = float(m.group(1)), float(m.group(2))
        rows.append({"label": label, "coef": coef, "ci": 1.96 * se, "p": row["p_value"]})

    df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)  # top-to-bottom order

    fig, ax = plt.subplots(figsize=(6, 4.2))
    y = np.arange(len(df))
    ax.errorbar(df["coef"], y, xerr=df["ci"], fmt="o", color=BLUE,
                ecolor=MUTED, elinewidth=1.4, capsize=4, markersize=7)
    ax.axvline(0, color=GRID, linewidth=1, linestyle="--")

    for yi, coef, ci, p in zip(y, df["coef"], df["ci"], df["p"]):
        ax.text(coef, yi + 0.22, f"{coef:+.4f} ({sig_stars(p)})", ha="center",
                fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_ylim(-0.6, len(df) - 1 + 0.9)
    ax.set_xlabel("Effect on outgoing similarity (OLS coefficient, 95% CI)")
    ax.set_title("What predicts topic follow-up strength?\n(controlling for the other two variables)",
                 fontsize=12, pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig3_regression_forest_plot")


def fig4_robustness(mixed):
    thresholds = np.arange(0.02, 0.32, 0.02)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for g in ["female", "male"]:
        sub = mixed[mixed["initiator_gender"] == g]["similarity"].values
        rates = [(sub > t).mean() for t in thresholds]
        ax.plot(thresholds, rates, marker="o", markersize=4, linewidth=2,
                color=GENDER_COLOR[g], label=g.capitalize())

    ax.set_xlabel("Follow-up similarity threshold")
    ax.set_ylabel("Follow-up rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Male > female follow-up gap holds across thresholds\n(robustness check)", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig4_robustness_across_thresholds")


def main():
    mixed = load_mixed_turns()
    stats_df = pd.read_csv(os.path.join(REPO, "results", "topic_control_stats.csv"))
    chi2_p = float(stats_df.loc[stats_df["test"] == "chi_square_followup_by_gender", "p_value"].iloc[0])

    fig1_headline(mixed, chi2_p)
    fig2_confound(mixed)
    fig3_forest(stats_df)
    fig4_robustness(mixed)


if __name__ == "__main__":
    main()
