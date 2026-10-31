#!/usr/bin/env python3
"""
Step 3: revised topic-control figures (PNG, 300 dpi) for Report 2 / capstone.

    fig1_followup_by_role_all_dyads.png        replaces Report 1 Fig. 12
    fig2_followup_by_gender_within_role.png    replaces Report 1 Fig. 10
    fig3_within_episode_gender_difference.png  replaces Report 1 Fig. 9
    fig4_gee_original_model.png                replaces Report 1 Fig. 11
    fig5_gender_estimate_sensitivity.png       new (sensitivity, specs A-F)
    fig6_followup_by_reply_length.png          new (measurement diagnostic)

Colors: categorical slots 1-4 of the dataviz reference palette, as in
Report 1 (blue = male, orange = female, aqua = host, yellow = guest).
Run after 02_sensitivity_and_tables.py.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import FIGURES, MIXED, ORIG_TURNS, SEED, TABLES, load_original_mixed  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, POINT = "#1a1a1a", "#5f5e5a", "#e4e3dc", "#9a9890"
GENDER_COLOR = {"female": ORANGE, "male": BLUE}
DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
DYAD_LABEL = {"MALE->MALE": "Male host\nmale guest", "MALE->FEMALE": "Male host\nfemale guest",
              "FEMALE->MALE": "Female host\nmale guest", "FEMALE->FEMALE": "Female host\nfemale guest"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300,
})
PCT = lambda v, _: f"{v:.0f}%"


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]", name)


def mean_ci_t(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def bar_labels(ax, xs, tops, vals):
    for x, t, v in zip(xs, tops, vals):
        ax.text(x, t + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)


def fig1(turns):
    ep = (turns[turns.dyad.isin(DYADS)].groupby(["dyad", "initiator_role", "episode_id"])
          ["followed_up"].mean().reset_index())
    ep["followed_up"] *= 100
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    x, w = np.arange(len(DYADS)), 0.36
    for i, (role, col, lab) in enumerate([("HOST", AQUA, "Host initiated"), ("GUEST", YELLOW, "Guest initiated")]):
        st = [mean_ci_t(ep[(ep.dyad == d) & (ep.initiator_role == role)]["followed_up"]) for d in DYADS]
        means, errs = map(np.array, zip(*st))
        xs = x + (i - 0.5) * w
        ax.bar(xs, means, width=w - 0.04, color=col, label=lab, yerr=errs, capsize=3,
               error_kw={"ecolor": INK, "elinewidth": 1})
        bar_labels(ax, xs, means + errs, means)
    n = ep[ep.initiator_role == "HOST"].groupby("dyad").size()
    ax.set_xticks(x, [f"{DYAD_LABEL[d]}\n(n = {n[d]})" for d in DYADS])
    ax.set_ylabel("Topics followed up (episode mean)")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Follow-up rate by initiator role")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig1_followup_by_role_all_dyads.png")


def fig2(m):
    ep = (m.groupby(["episode_id", "initiator_role", "initiator_gender"])["followed_up"]
          .mean().reset_index())
    ep["followed_up"] *= 100
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    x, w = np.arange(2), 0.36
    for i, g in enumerate(["female", "male"]):
        st = [mean_ci_t(ep[(ep.initiator_role == r) & (ep.initiator_gender == g)]["followed_up"])
              for r in ["HOST", "GUEST"]]
        means, errs = map(np.array, zip(*st))
        xs = x + (i - 0.5) * w
        ax.bar(xs, means, width=w - 0.04, color=GENDER_COLOR[g], label=f"{g.capitalize()} initiator",
               yerr=errs, capsize=3, error_kw={"ecolor": INK, "elinewidth": 1})
        bar_labels(ax, xs, means + errs, means)
    ax.set_xticks(x, ["Host initiated", "Guest initiated"])
    ax.set_ylabel("Topics followed up (episode mean)")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Follow-up rate by gender and role")
    ax.legend(loc="upper right")
    save(fig, "fig2_followup_by_gender_within_role.png")


def fig3(m):
    pe = m.groupby(["episode_id", "dyad", "initiator_gender"])["followed_up"].mean().unstack().reset_index()
    pe["diff"] = (pe["male"] - pe["female"]) * 100
    groups = [("All mixed-gender\nepisodes", pe, INK),
              ("Male host,\nfemale guest", pe[pe.dyad == "MALE->FEMALE"], INK),
              ("Female host,\nmale guest", pe[pe.dyad == "FEMALE->MALE"], INK)]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    for k, (lab, df, col) in enumerate(groups):
        y = len(groups) - 1 - k
        jit = rng.uniform(-0.18, 0.18, len(df))
        ax.scatter(df["diff"], y + jit, s=12, color=POINT, alpha=0.7, lw=0, zorder=2)
        mu, h = mean_ci_t(df["diff"])
        ax.errorbar(mu, y, xerr=h, fmt="D", color=col, ms=7, capsize=4, lw=2, zorder=3,
                    markeredgecolor="white", markeredgewidth=1)
        ax.text(mu, y + 0.3, f"{mu:+.1f} pp".replace("-", "−"), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(groups))[::-1], [f"{g[0]}\n(n = {len(g[1])})" for g in groups])
    ax.set_ylim(-0.6, len(groups) - 0.25)
    ax.set_xlabel("Male − female follow-up rate within the episode (percentage points)")
    lim = np.ceil(pe["diff"].abs().max() / 10) * 10
    ax.set_xlim(-lim, lim)
    ax.set_title("Within-episode gender difference in follow-up")
    ax.scatter([], [], s=12, color=POINT, label="One episode")
    ax.errorbar([], [], xerr=[], fmt="D", color=INK, label="Mean and 95% CI")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2)
    save(fig, "fig3_within_episode_gender_difference.png")


def forest_axis(ax, ests, los, his, labels, filled, xlabel, ticks):
    y = np.arange(len(labels))[::-1]
    for yi, e, lo, hi, f in zip(y, ests, los, his, filled):
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.6, zorder=2)
        ax.scatter([e], [yi], s=48, zorder=3, color=INK if f else "white", edgecolor=INK, lw=1.5)
        ax.text(e, yi + 0.2, f"{e:.2f}", va="bottom", ha="center", fontsize=8.5, color=INK)
    ax.axvline(1, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig4(sens_full):
    A = sens_full[sens_full.spec == "A"].iloc[0]
    tab = pd.read_csv(os.path.join(TABLES, "table12_corrected.csv"))
    wc = tab[tab.Predictor.str.startswith("Initiator turn length")]["GEE OR [95% CI]"].iloc[0]
    e, rest = wc.split(" [")
    lo, hi = rest.rstrip("]").split(", ")
    ests = [A.male_OR, A.host_OR, float(e)]
    los, his = [A.male_lo, A.host_lo, float(lo)], [A.male_hi, A.host_hi, float(hi)]
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    forest_axis(ax, ests, los, his, ["Male initiator\n(vs female)", "Host initiator\n(vs guest)",
                                     "Initiator turn\n+100 words"], [True] * 3,
                "Odds ratio of follow-up (log scale)", [0.5, 0.75, 1, 1.5, 2, 3, 4])
    ax.set_xlim(0.6, 4.2)
    ax.set_title("Original model (GEE): predictors of follow-up")
    save(fig, "fig4_gee_original_model.png")


def fig5(sens_full):
    s = sens_full[sens_full.spec.isin(list("ABCDEF"))].copy()
    names = {"A": "A  Original model", "B": "B  No length control", "C": "C  Log initiator length",
             "D": "D  Log initiator + reply length", "E": "E  Equal episode weights",
             "F": "F  Host-clustered SE*"}
    labels = [names[c] for c in s.spec]
    filled = [c != "F" for c in s.spec]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    forest_axis(axes[0], s.male_OR, s.male_lo, s.male_hi, labels, filled,
                "Odds ratio, male vs female initiator", [0.6, 0.8, 1, 1.25])
    axes[0].set_xlim(0.6, 1.4)
    axes[0].set_title("Gender")
    forest_axis(axes[1], s.host_OR, s.host_lo, s.host_hi, labels, filled,
                "Odds ratio, host vs guest initiator", [1, 1.5, 2, 3, 4])
    axes[1].set_xlim(0.85, 4.3)
    axes[1].set_title("Role")
    axes[1].tick_params(axis="y", length=0)
    axes[0].scatter([], [], s=48, color=INK, label="Episode-clustered 95% CI")
    axes[0].scatter([], [], s=48, color="white", edgecolor=INK, lw=1.5, label="*Host-clustered (4 male hosts)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.55, -0.06), ncol=2)
    fig.suptitle("Sensitivity of the follow-up estimates", fontweight="bold", fontsize=11.5, x=0.55)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig5_gender_estimate_sensitivity.png")


def fig6(m):
    m = m.copy()
    edges = np.quantile(m.responder_word_count, [0, .2, .4, .6, .8, 1])
    m["q"] = pd.qcut(m.responder_word_count, 5, labels=False)
    eps = m.episode_id.unique()
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for role, col, lab in [("HOST", AQUA, "Host initiated"), ("GUEST", YELLOW, "Guest initiated")]:
        sub = m[m.initiator_role == role]
        agg = sub.groupby(["episode_id", "q"])["followed_up"].agg(["sum", "size"]).unstack(fill_value=0)
        agg = agg.reindex(eps, fill_value=0)
        S, N = agg["sum"].values, agg["size"].values
        point = S.sum(0) / N.sum(0) * 100
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(eps), len(eps))
            boots.append(S[i].sum(0) / N[i].sum(0) * 100)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        xs = np.arange(5)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(xs, point, color=col, lw=2, marker="o", ms=6, label=lab,
                markeredgecolor="white", markeredgewidth=1)
    labs = []
    for k in range(5):
        lo_, hi_ = int(edges[k]) if k == 0 else int(edges[k]) + 1, int(edges[k + 1])
        labs.append(f"{lo_}–{hi_}" if k < 4 else f"{lo_}+")
    ax.set_xticks(range(5), labs)
    ax.set_xlabel("Length of the reply (words, quintiles)")
    ax.set_ylabel("Topics followed up (pooled)")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Follow-up rate by reply length")
    ax.legend(loc="upper left")
    save(fig, "fig6_followup_by_reply_length.png")


def main():
    turns = pd.read_csv(ORIG_TURNS)
    m = load_original_mixed()
    sens = pd.read_csv(os.path.join(TABLES, "sensitivity_full.csv"))
    fig1(turns)
    fig2(m)
    fig3(m)
    fig4(sens)
    fig5(sens)
    fig6(m)


if __name__ == "__main__":
    main()
