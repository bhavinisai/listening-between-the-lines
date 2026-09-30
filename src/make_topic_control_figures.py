#!/usr/bin/env python3
"""
Figures for the topic-control analysis (topic_control.py): one plot per
image, minimal text. Reads results/topic_control_turns.csv,
results/topic_control_stats.csv and (if present)
results/topic_control_embeddings_turns.csv; writes PNG (300dpi) to
results/figures/.

    fig1a  follow-up rate by initiator gender, pooled turn-pairs
    fig1b  follow-up rate by initiator gender, per episode (paired)
    fig2   follow-up rate by initiator gender within host / guest role
    fig3a  GEE odds ratios (clustered by episode)
    fig3b  mixed-effects coefficients on follow-up similarity
    fig4a  follow-up rate across fixed similarity thresholds
    fig4b  male - female gap across fixed similarity thresholds
    fig5   host vs. guest follow-up rate in all four dyads
    fig6   male - female gap, TF-IDF vs. sentence embeddings

Usage:
    python src/make_topic_control_figures.py
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT_DIR = os.path.join(RESULTS, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Colorblind-safe categorical pairs: adjacent slots 1-2 (gender) and 3-4
# (role) of the dataviz skill's reference palette.
BLUE = "#2a78d6"    # male
ORANGE = "#eb6834"  # female
AQUA = "#1baf7a"    # host
YELLOW = "#eda100"  # guest
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e4e3dc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titlepad": 12,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

GENDER_COLOR = {"female": ORANGE, "male": BLUE}
GENDERS = ["female", "male"]
MIXED = ["MALE->FEMALE", "FEMALE->MALE"]
DYAD_ORDER = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
DYAD_LABEL = {"MALE->MALE": "M → M", "MALE->FEMALE": "M → F",
              "FEMALE->MALE": "F → M", "FEMALE->FEMALE": "F → F"}
PCT = lambda v, _: f"{v:.0%}"


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def new_fig(w=5.5, h=4.2, grid_axis="y"):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {name}.png")


def label_bars(ax, bars, values, tops):
    for b, v, t in zip(bars, values, tops):
        ax.text(b.get_x() + b.get_width() / 2, t + 0.01, f"{v:.0%}",
                ha="center", va="bottom", fontsize=10, color=INK)


def stat_row(stats_df, test):
    return stats_df[stats_df["test"] == test].iloc[0]


def coef_se(row):
    m = re.search(r"coef=(-?[\d.]+),\s*se=([\d.]+)", row["notes"])
    return float(m.group(1)), float(m.group(2))


def episode_gender_rates(df):
    ep = df.groupby(["episode_id", "initiator_gender"])["followed_up"].mean().unstack()
    return ep.dropna(subset=["male", "female"])


# --------------------------------------------------------------------------
def fig1a_pooled(mixed):
    rate = mixed.groupby("initiator_gender")["followed_up"].agg(["mean", "count"]).reindex(GENDERS)
    errs = 1.96 * np.sqrt(rate["mean"] * (1 - rate["mean"]) / rate["count"])
    fig, ax = new_fig(4.5, 4.2)
    for xi, g in enumerate(GENDERS):
        bar = ax.bar(xi, rate.loc[g, "mean"], width=0.6, color=GENDER_COLOR[g], yerr=errs[g],
                     capsize=4, error_kw={"ecolor": INK, "elinewidth": 1},
                     label=f"{g.capitalize()} initiator")
        label_bars(ax, bar, [rate.loc[g, "mean"]], [rate.loc[g, "mean"] + errs[g]])
    ax.set_title("Follow-up rate")
    ax.set_xticks([0, 1], ["Female", "Male"])
    ax.set_xlabel("Initiator gender")
    ax.set_ylabel("Topics followed up")
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig1a_followup_pooled")


def fig1b_per_episode(mixed):
    ep = episode_gender_rates(mixed)
    fig, ax = new_fig(4.5, 4.2)
    for _, r in ep.iterrows():
        ax.plot([0, 1], [r["female"], r["male"]], color=GRID, lw=0.8, zorder=1)
    for xi, g in enumerate(GENDERS):
        ax.scatter(np.full(len(ep), xi), ep[g], s=12, color=GENDER_COLOR[g], alpha=0.6,
                   lw=0, zorder=2, label=f"{g.capitalize()} initiator")
    means = [ep[g].mean() for g in GENDERS]
    ax.plot([0, 1], means, color=INK, lw=2, marker="D", markersize=7, zorder=3, label="Mean")
    for xi, m in enumerate(means):
        ax.text(xi + (0.12 if xi else -0.12), m, f"{m:.0%}", ha="left" if xi else "right",
                va="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Follow-up rate per episode (n = {len(ep)})")
    ax.set_xticks([0, 1], ["Female", "Male"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("Initiator gender")
    ax.set_ylabel("Topics followed up")
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper center", ncol=3, fontsize=9, handletextpad=0.3, columnspacing=1)
    save(fig, "fig1b_followup_per_episode")


def fig2_role(mixed):
    ep = (mixed.groupby(["episode_id", "initiator_role", "initiator_gender"])["followed_up"]
          .mean().reset_index())
    roles = ["HOST", "GUEST"]
    fig, ax = new_fig(5.5, 4.2)
    width, x = 0.36, np.arange(len(roles))
    for i, g in enumerate(GENDERS):
        stats = [mean_ci(ep[(ep["initiator_role"] == r) & (ep["initiator_gender"] == g)]["followed_up"])
                 for r in roles]
        means, errs = zip(*stats)
        bars = ax.bar(x + (i - 0.5) * width, means, width=width - 0.03, color=GENDER_COLOR[g],
                      yerr=errs, capsize=4, error_kw={"ecolor": INK, "elinewidth": 1},
                      label=f"{g.capitalize()} initiator")
        label_bars(ax, bars, means, np.array(means) + np.array(errs))
    ax.set_title("Follow-up rate by role and gender")
    ax.set_xticks(x, ["Host", "Guest"])
    ax.set_xlabel("Initiator role")
    ax.set_ylabel("Topics followed up (episode mean)")
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper right")
    save(fig, "fig2_followup_by_role_and_gender")


TERMS = [("is_male_initiator", "Male initiator"),
         ("is_host_initiator", "Host initiator"),
         ("initiator_word_count", "+100 words")]


def forest(stats_df, prefix, transform, ref, xlabel, title, name, log=False):
    fig, ax = new_fig(5.5, 3.4, grid_axis="x")
    y = np.arange(len(TERMS))[::-1]
    for yi, (term, _) in zip(y, TERMS):
        row = stat_row(stats_df, f"{prefix}_{term}")
        c, s = coef_se(row)
        k = 100 if term == "initiator_word_count" else 1
        est, lo, hi = (transform(v * k) for v in (c, c - 1.96 * s, c + 1.96 * s))
        sig = row["p_value"] < 0.05
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.5, zorder=1)
        ax.scatter([est], [yi], s=55, zorder=2, color=INK if sig else "white",
                   edgecolor=INK, linewidth=1.5)
    ax.axvline(ref, color=MUTED, lw=1, ls="--", zorder=0)
    if log:
        ax.set_xscale("log")
        ax.set_xticks([0.75, 1, 1.5, 2, 3])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.set_yticks(y, [t[1] for t in TERMS])
    ax.set_ylim(-0.6, len(TERMS) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.scatter([], [], s=55, color=INK, label="p < 0.05")
    ax.scatter([], [], s=55, color="white", edgecolor=INK, linewidth=1.5, label="Not significant")
    ax.plot([], [], color=INK, lw=1.5, label="95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    save(fig, name)


def fig3a_gee(stats_df):
    forest(stats_df, "gee_followedup", np.exp, 1, "Odds ratio of follow-up (log scale)",
           "Predictors of follow-up (GEE)", "fig3a_gee_odds_ratios", log=True)


def fig3b_mixedlm(stats_df):
    forest(stats_df, "mixedlm_similarity", lambda v: v, 0, "Change in follow-up similarity",
           "Predictors of follow-up similarity (mixed model)", "fig3b_mixedlm_coefficients")


def threshold_curves(mixed, n_boot=500):
    thresholds = np.round(np.arange(0.01, 0.31, 0.01), 2)
    eps = mixed["episode_id"].unique()
    idx = {e: i for i, e in enumerate(eps)}
    above = {g: np.zeros((len(eps), len(thresholds))) for g in GENDERS}
    total = {g: np.zeros(len(eps)) for g in GENDERS}
    for (e, g), sims in mixed.groupby(["episode_id", "initiator_gender"])["similarity"]:
        above[g][idx[e]] = (sims.values[:, None] > thresholds[None, :]).sum(axis=0)
        total[g][idx[e]] = len(sims)
    rate = {g: above[g].sum(0) / total[g].sum() for g in GENDERS}
    rng = np.random.default_rng(42)
    boot = {g: [] for g in GENDERS}
    for _ in range(n_boot):
        s = rng.integers(0, len(eps), len(eps))
        for g in GENDERS:
            boot[g].append(above[g][s].sum(0) / total[g][s].sum())
    return thresholds, rate, {g: np.array(v) for g, v in boot.items()}


def fig4a_thresholds(thresholds, rate, boot):
    fig, ax = new_fig(5.5, 4.2)
    for g in GENDERS:
        lo, hi = np.percentile(boot[g], [2.5, 97.5], axis=0)
        ax.fill_between(thresholds, lo, hi, color=GENDER_COLOR[g], alpha=0.2, lw=0)
        ax.plot(thresholds, rate[g], lw=2, color=GENDER_COLOR[g], label=f"{g.capitalize()} initiator")
    ax.fill_between([], [], [], color=MUTED, alpha=0.2, lw=0, label="95% CI")
    ax.set_title("Follow-up rate across thresholds")
    ax.set_xlabel("Similarity threshold for a follow-up")
    ax.set_ylabel("Topics followed up")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper right")
    save(fig, "fig4a_followup_across_thresholds")


def fig4b_gap(thresholds, rate, boot):
    diff = (boot["male"] - boot["female"]) * 100
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    fig, ax = new_fig(5.5, 4.2)
    ax.fill_between(thresholds, lo, hi, color=BLUE, alpha=0.2, lw=0, label="95% CI")
    ax.plot(thresholds, (rate["male"] - rate["female"]) * 100, lw=2, color=BLUE,
            label="Male − female gap")
    ax.axhline(0, color=MUTED, lw=1, ls="--", label="No gap")
    ax.set_title("Raw gender gap across thresholds")
    ax.set_xlabel("Similarity threshold for a follow-up")
    ax.set_ylabel("Male − female (percentage points)")
    ax.set_ylim(-1, 8)
    ax.legend(loc="upper right")
    save(fig, "fig4b_gender_gap_across_thresholds")


def fig5_dyads(turns):
    ep = (turns[turns["dyad"].isin(DYAD_ORDER)]
          .groupby(["dyad", "initiator_role", "episode_id"])["followed_up"].mean().reset_index())
    fig, ax = new_fig(6.5, 4.2)
    width, x = 0.38, np.arange(len(DYAD_ORDER))
    for i, (r, color) in enumerate([("HOST", AQUA), ("GUEST", YELLOW)]):
        stats = [mean_ci(ep[(ep["dyad"] == d) & (ep["initiator_role"] == r)]["followed_up"])
                 for d in DYAD_ORDER]
        means, errs = zip(*stats)
        bars = ax.bar(x + (i - 0.5) * width, means, width=width - 0.03, color=color,
                      yerr=errs, capsize=3, error_kw={"ecolor": INK, "elinewidth": 1},
                      label=f"{r.capitalize()} initiator")
        label_bars(ax, bars, means, np.array(means) + np.array(errs))
    ax.set_title("Follow-up rate by dyad and role")
    ax.set_xticks(x, [DYAD_LABEL[d] for d in DYAD_ORDER])
    ax.set_xlabel("Dyad (host gender → guest gender)")
    ax.set_ylabel("Topics followed up (episode mean)")
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper center", ncol=2)
    save(fig, "fig5_followup_by_dyad_and_role")


def fig6_measures(mixed):
    emb_path = os.path.join(RESULTS, "topic_control_embeddings_turns.csv")
    if not os.path.exists(emb_path):
        print("[SKIP] fig6: no embeddings turns file")
        return
    emb = pd.read_csv(emb_path)
    emb = emb[emb["dyad"].isin(MIXED)]
    rows = [("TF-IDF\n(shared words)", episode_gender_rates(mixed)),
            ("Embeddings\n(shared meaning)", episode_gender_rates(emb))]
    fig, ax = new_fig(5.5, 3.0, grid_axis="x")
    y = np.arange(len(rows))[::-1]
    for yi, (_, ep) in zip(y, rows):
        m, ci = mean_ci((ep["male"] - ep["female"]) * 100)
        ax.plot([m - ci, m + ci], [yi, yi], color=BLUE, lw=2, zorder=1)
        ax.scatter([m], [yi], s=60, color=BLUE, zorder=2)
    ax.plot([], [], color=BLUE, lw=2, marker="o", label="Mean gap ± 95% CI")
    ax.axvline(0, color=MUTED, lw=1, ls="--", label="No gap")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("Male − female follow-up rate per episode (pp)")
    ax.set_title("Gender gap under two similarity measures")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, "fig6_gender_gap_by_similarity_measure")


def main():
    turns = pd.read_csv(os.path.join(RESULTS, "topic_control_turns.csv"))
    mixed = turns[turns["dyad"].isin(MIXED)].copy()
    stats_df = pd.read_csv(os.path.join(RESULTS, "topic_control_stats.csv"))

    fig1a_pooled(mixed)
    fig1b_per_episode(mixed)
    fig2_role(mixed)
    fig3a_gee(stats_df)
    fig3b_mixedlm(stats_df)
    curves = threshold_curves(mixed)
    fig4a_thresholds(*curves)
    fig4b_gap(*curves)
    fig5_dyads(turns)
    fig6_measures(mixed)


if __name__ == "__main__":
    main()
