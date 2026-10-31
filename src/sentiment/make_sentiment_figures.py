#!/usr/bin/env python3
"""
Figures for the sentiment / emotional-dynamics analysis. One plot per image,
PNG 300 dpi, written to results/figures/sentiment/.

    fig1  gold-set accuracy and macro-F1 by model (vs. majority baseline)
    fig2  confusion matrix of the best valence model (episode-grouped CV)
    fig3  mean valence and arousal by dyad and role (per-episode means)
    fig4  mirroring: responder valence vs. previous-turn valence, by previous speaker gender
    fig5  per-episode mirroring correlation by previous speaker gender
    fig6  escalation: arousal-rise and valence-drop step rates; escalation runs per 100 turns
    fig7  recovery: next-turn lift after a negative turn, and who delivers the first non-negative turn

Usage:
    python src/sentiment/make_sentiment_figures.py [--res results/sentiment] [--out results/figures/sentiment]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # male, female, host, guest
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3dc"
GCOL = {"male": BLUE, "female": ORANGE}
RCOL = {"HOST": AQUA, "GUEST": YELLOW}
DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
DLAB = ["M→M", "M→F", "F→M", "F→F"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlesize": 12,
    "axes.titlepad": 12, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 10,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)
    print("wrote", name)


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0


def fig_models(res, out):
    m = pd.read_csv(f"{res}/valence_eval_models.csv")
    m = m[m.kind.isin(["zero-shot", "baseline"])].copy()
    rows = [dict(label=r.model + "\n(zero-shot)" if r.kind == "zero-shot" else "always\n'neutral'",
                 accuracy=r.accuracy, lo=r.ci_lo, hi=r.ci_hi, f1=r.macro_f1, kappa=r.kappa) for r in m.itertuples()]
    s = f"{res}/ft_cv/summary.csv"
    if os.path.exists(s):
        gold = pd.read_csv(f"{res}/valence_eval_best_preds.csv") if False else None
        for r in pd.read_csv(s).itertuples():
            n = 400
            p = r.accuracy; h = 1.96 * np.sqrt(p * (1 - p) / n)
            rows.append(dict(label="fine-tuned\n" + r.config.replace("cardiff_", "").replace("_lr", "\nlr"),
                             accuracy=p, lo=p - h, hi=p + h, f1=r.macro_f1, kappa=r.kappa))
    d = pd.DataFrame(rows)
    d["label"] = d.label.str.replace("\n", " ")
    ft = d[d.label.str.startswith("fine-tuned")].nlargest(3, "f1")  # best three fine-tuned configs only
    d = pd.concat([d[~d.label.str.startswith("fine-tuned")], ft]).sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(d) + 1.2)); ax.grid(axis="y", visible=False)
    y = np.arange(len(d)); h = 0.38
    ax.barh(y - h / 2, d.accuracy, h, color=BLUE, label="accuracy (95% CI)",
            xerr=[d.accuracy - d.lo, d.hi - d.accuracy], error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.barh(y + h / 2, d.f1, h, color=ORANGE, label="macro-F1")
    base = d[d.label.str.startswith("always")].accuracy.iloc[0]
    ax.axvline(base, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=9); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("score on gold set (n=400); dashed = always-'neutral' accuracy"); ax.legend(loc="lower right")
    save(fig, out, "fig1_gold_model_comparison")


def fig_confusion(res, out):
    s = f"{res}/ft_cv/summary.csv"
    if not os.path.exists(s):
        return
    best = pd.read_csv(s).config.iloc[0]
    x = pd.read_csv(f"{res}/ft_cv/{best}.csv")
    lab = ["negative", "neutral", "positive"]
    y = x.valence.map({l: i for i, l in enumerate(lab)}); p = x[["p_neg", "p_neu", "p_pos"]].values.argmax(1)
    cm = confusion_matrix(y, p); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.grid(False)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_yticks(range(3)); ax.set_yticklabels(lab)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold (hand label)"); ax.set_title(f"{best}", fontsize=9)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    save(fig, out, "fig2_confusion_best_valence")


def fig_tone(res, out):
    e = pd.read_csv(f"{res}/episode_role_tone.csv"); e = e[e.dyad.isin(DYADS)]
    for m, name, yl in [("valence", "fig3a_valence_by_dyad_role", "mean valence (−1 negative … +1 positive)"),
                        ("arousal", "fig3b_arousal_by_dyad_role", "mean arousal (0 calm … 1 intense)")]:
        fig, ax = plt.subplots(figsize=(7, 4.4)); w = 0.38
        for k, role in enumerate(["HOST", "GUEST"]):
            v = [e[(e.dyad == d) & (e.role == role)][m] for d in DYADS]
            ax.bar(np.arange(4) + (k - .5) * w, [x.mean() for x in v], w, color=RCOL[role], label=role.lower(),
                   yerr=[ci(x) for x in v], error_kw=dict(ecolor=INK, lw=1, capsize=2))
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{l}\n(n={(e.dyad == d).sum() // 2})" for l, d in zip(DLAB, DYADS)])
        ax.set_xlabel("host → guest gender"); ax.set_ylabel(yl); ax.legend()
        save(fig, out, name)


def fig_mirror(res, out):
    p = pd.read_csv(f"{res}/adjacent_pairs.csv"); p = p[p.prev_gender.isin(GCOL) & p.resp_role.isin(RCOL)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)
    for ax, role in zip(axes, ["GUEST", "HOST"]):
        for g, c in GCOL.items():
            d = p[(p.resp_role == role) & (p.prev_gender == g)]
            b = pd.cut(d.v_prev, np.linspace(-1, 1, 11))
            t = d.groupby(b, observed=True).agg(x=("v_prev", "mean"), y=("v_cur", "mean"), n=("v_cur", "size"),
                                                 se=("v_cur", lambda s: s.std() / np.sqrt(len(s)))); t = t[t.n >= 30]
            ax.errorbar(t.x, t.y, 1.96 * t.se, color=c, marker="o", ms=5, lw=2, capsize=2,
                        label=f"previous speaker {g}")
        ax.set_title(f"{role.lower()} responding"); ax.set_xlabel("previous-turn valence")
    axes[0].set_ylabel("responder's valence"); axes[0].legend()
    save(fig, out, "fig4_mirroring_binned")


def fig_rho(res, out):
    r = pd.read_csv(f"{res}/mirroring_episode_rho.csv"); r = r[r.tone == "valence"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3)); w = 0.35
    for k, g in enumerate(["male", "female"]):
        data = [r[(r.resp_role == role) & (r.prev_gender == g)].rho.dropna() for role in ["GUEST", "HOST"]]
        bp = ax.boxplot(data, positions=np.arange(2) + (k - .5) * (w + .05), widths=w, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=GCOL[g], alpha=.8, edgecolor=MUTED)
        ax.plot([], [], "s", color=GCOL[g], label=f"previous speaker {g}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["guest responding\n(to host)", "host responding\n(to guest)"])
    ax.axhline(0, color=MUTED, lw=.8); ax.set_ylabel("per-episode Spearman ρ (valence)"); ax.legend()
    save(fig, out, "fig5_mirroring_episode_rho")


def fig_escalation(res, out):
    p = pd.read_csv(f"{res}/adjacent_pairs.csv"); p = p[p.prev_gender.isin(GCOL) & p.resp_role.isin(RCOL)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, (y, t) in zip(axes, [("esc_step", "arousal rises ≥ 0.5 SD"), ("neg_step", "valence drops ≥ 0.5 SD")]):
        for k, role in enumerate(["HOST", "GUEST"]):
            v = [p[(p.resp_role == role) & (p.prev_gender == g)][y] for g in ["male", "female"]]
            ax.bar(np.arange(2) + (k - .5) * .38, [100 * x.mean() for x in v], .38, color=RCOL[role],
                   label=f"{role.lower()} responding", yerr=[100 * ci(x) for x in v], error_kw=dict(ecolor=INK, lw=1, capsize=2))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["previous speaker\nmale", "previous speaker\nfemale"]); ax.set_title(t)
    axes[0].set_ylabel("% of adjacent turn pairs"); axes[0].legend()
    save(fig, out, "fig6a_escalation_steps")
    fig, ax = plt.subplots(figsize=(7, 4.2)); w = .38
    for k, (kind, c, lab) in enumerate([("arousal", BLUE, "arousal-rise runs"), ("valence", ORANGE, "valence-fall runs")]):
        f = f"{res}/escalation_rate_{kind}.csv"
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f); v = [e[e.dyad == d].rate_per_100 for d in DYADS]
        ax.bar(np.arange(4) + (k - .5) * w, [x.mean() for x in v], w, color=c, label=lab,
               yerr=[ci(x) for x in v], error_kw=dict(ecolor=INK, lw=1, capsize=2))
    ax.set_xticks(range(4)); ax.set_xticklabels(DLAB); ax.set_xlabel("host → guest gender")
    ax.set_ylabel("runs of ≥3 turns per 100 turns"); ax.legend()
    save(fig, out, "fig6b_escalation_runs_by_dyad")


def fig_recovery(res, out):
    ev = pd.read_csv(f"{res}/recovery_events.csv"); ev = ev[ev.resp_gender.isin(GCOL)]
    fig, ax = plt.subplots(figsize=(6.4, 4.3)); w = .38
    for k, g in enumerate(["male", "female"]):
        v = [ev[(ev.resp_role == r) & (ev.resp_gender == g)].immediate for r in ["HOST", "GUEST"]]
        ax.bar(np.arange(2) + (k - .5) * w, [100 * x.mean() for x in v], w, color=GCOL[g], label=f"responder {g}",
               yerr=[100 * ci(x) for x in v], error_kw=dict(ecolor=INK, lw=1, capsize=2))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["host responds\n(after guest negativity)", "guest responds\n(after host negativity)"])
    ax.set_ylabel("% of negative turns lifted on the next turn"); ax.legend()
    save(fig, out, "fig7a_recovery_next_turn")
    e = ev[ev.dyad.isin(DYADS)]
    ct = pd.crosstab(e.dyad, e.recoverer_role, normalize="index").reindex(DYADS)
    fig, ax = plt.subplots(figsize=(7, 4.2)); bottom = np.zeros(4)
    for role, c in [("HOST", AQUA), ("GUEST", YELLOW), ("none", GRID)]:
        if role in ct:
            ax.bar(range(4), 100 * ct[role], .6, bottom=bottom, color=c, edgecolor="white", linewidth=2,
                   label={"none": "no recovery in 3 turns"}.get(role, role.lower()))
            bottom += 100 * ct[role].values
    ax.set_xticks(range(4)); ax.set_xticklabels(DLAB); ax.set_xlabel("host → guest gender")
    ax.set_ylabel("% of negative turns"); ax.set_title("who delivers the first non-negative turn", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=3)
    save(fig, out, "fig7b_who_recovers_by_dyad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/sentiment")
    ap.add_argument("--out", default="results/figures/sentiment")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(f"{a.res}/valence_eval_models.csv"):
        fig_models(a.res, a.out); fig_confusion(a.res, a.out)
    for f in (fig_tone, fig_mirror, fig_rho, fig_escalation, fig_recovery):
        f(a.res, a.out)


if __name__ == "__main__":
    main()
