#!/usr/bin/env python3
"""
Sentiment report figures (redesigned: short titles, minimal clutter, PNG only at 300 dpi), regenerated entirely from
saved results. No training, no new experiments, no tuning, no held-out test / human-gold evaluation.

    python src/sentiment/figures/make_figures.py

Output:  results/sentiment/figures/*.png   (snake_case)      results/sentiment/figures/data/*.csv  (plotted numbers)

Sources (read-only): results/sentiment/llm_scaling_exp/, llm_scaling_stratum/, llm_ensemble_check/, llm_labels3/, audit/
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

R = "results/sentiment"
SC, ST, EN, AU, L3 = f"{R}/llm_scaling_exp", f"{R}/llm_scaling_stratum", f"{R}/llm_ensemble_check", f"{R}/audit", f"{R}/llm_labels3"
OUT = f"{R}/figures"
DATA = f"{OUT}/data"
os.makedirs(DATA, exist_ok=True)

CLASSES = {"valence": ["Negative", "Neutral", "Positive"], "arousal": ["Low", "Medium", "High"]}
SIZES = [1783, 3000, 5000]
RETAINED = {"valence": (1783, 2), "arousal": (5000, 2)}          # (training size, seed) of the retained checkpoints
TITLE = {"valence": "Valence", "arousal": "Arousal"}
DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
DYAD_LABEL = {"MALE->MALE": "Male → Male", "MALE->FEMALE": "Male → Female", "FEMALE->MALE": "Female → Male", "FEMALE->FEMALE": "Female → Female"}
MIN_SUPPORT = 5                                                   # a class with fewer true turns makes a subgroup macro-F1 "unreliable"

C = dict(blue="#0072B2", verm="#D55E00", green="#009E73", orange="#E69F00", sky="#56B4E9", pink="#CC79A7", grey="#6B6B6B", lgrey="#BBBBBB", black="#000000")
TASK_COL = {"valence": C["blue"], "arousal": C["verm"]}
W = 6.5                                                           # inches: fits a Word page at 100% scale

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": False, "legend.frameon": False, "lines.linewidth": 1.3})

log_sc = {(r["task"], r["size"], r["seed"]): r for r in json.load(open(f"{SC}/run_log.json"))}
log_st = {(r["cond"], r["seed"]): r for r in json.load(open(f"{ST}/run_log.json"))}
FORBIDDEN = re.compile(r"seed\s?\d|_\d{4}|checkpoint|llm_ft|models/|results/|\.npy|\.csv", re.I)


def finish(fig, name, title, handles=None, ncol=None, legend_title=None):
    """Title (checked for technical strings), optional shared legend, PNG at 300 dpi."""
    assert not FORBIDDEN.search(title), title
    for ax in fig.axes:
        assert not FORBIDDEN.search(ax.get_title()), ax.get_title()
    fig.suptitle(title, fontsize=11, fontweight="bold")
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles), title=legend_title, title_fontsize=8.5)
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)


def hgrid(ax):
    ax.grid(axis="y", color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)


def thousands(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))


def val_labels(task):
    return pd.read_csv(f"{SC}/val_labels_{task}.csv")


def best_probs(task, size, seed):
    return np.load(f"{SC}/valprobs_{task}_{size}_seed{seed}.npy")[log_sc[(task, size, seed)]["best_epoch"] - 1]


def ablation_probs(cond, seed):
    if cond == "base" and seed < 3:
        return best_probs("valence", 1783, seed)
    return np.load(f"{ST}/valprobs_{cond}_seed{seed}.npy")[log_st[(cond, seed)]["best_epoch"] - 1]


def ablation_macro(cond, seed):
    return (log_sc[("valence", 1783, seed)] if (cond == "base" and seed < 3) else log_st[(cond, seed)])["val_best"]["macro_f1"]


SEED_H = Line2D([], [], marker="o", ls="", mfc="white", mec=C["grey"], ms=5, label="Individual seed")


# ===================================================================== performance_by_training_size
def performance_by_training_size():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True)
    for ax, task, ylim in zip(axes, ("valence", "arousal"), ((0.70, 0.83), (0.46, 0.67))):
        means, sds = [], []
        for size in SIZES:
            vals = [log_sc[(task, size, s)]["val_best"]["macro_f1"] for s in (0, 1, 2)]
            for s, v in zip((0, 1, 2), vals):
                rows.append(dict(task=task, size=size, seed=s, n_train=log_sc[(task, size, s)]["n_train"], val_macro_f1=v))
                ax.scatter(size + (s - 1) * 75, v, s=22, facecolors="white", edgecolors=C["grey"], linewidths=0.9, zorder=3)
            means.append(np.mean(vals)); sds.append(np.std(vals, ddof=1))
        ax.errorbar(SIZES, means, yerr=sds, color=TASK_COL[task], marker="o", ms=5.5, capsize=3, lw=1.5, elinewidth=1.1, zorder=4)
        ax.set_xticks(SIZES); ax.set_xticklabels([f"{s:,}" for s in SIZES]); ax.set_xlim(1300, 5500); ax.set_ylim(*ylim)
        ax.set_xlabel("Training Examples"); ax.set_ylabel("Macro-F1"); ax.set_title(TITLE[task]); hgrid(ax)
    h = [SEED_H, Line2D([], [], color="black", marker="o", ms=5, label="Mean ± SD")]
    pd.DataFrame(rows).to_csv(f"{DATA}/performance_by_training_size.csv", index=False)
    finish(fig, "performance_by_training_size", "Performance by Training Size", h)


# ===================================================================== valence_confusion_matrix / arousal_confusion_matrix
def confusion_matrix_figure(task):
    size, seed = RETAINED[task]
    y = val_labels(task).y.values
    cm = confusion_matrix(y, best_probs(task, size, seed).argmax(1), labels=[0, 1, 2])
    cn = cm / cm.sum(1, keepdims=True)
    rows = [dict(task=task, true_label=CLASSES[task][i], predicted_label=CLASSES[task][j], count=int(cm[i, j]), row_percent=float(cn[i, j] * 100))
            for i in range(3) for j in range(3)]
    pd.DataFrame(rows).to_csv(f"{DATA}/{task}_confusion_matrix.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.9), constrained_layout=True)
    im = ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cn[i, j] * 100:.0f}%)", ha="center", va="center", fontsize=9.5, color="white" if cn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES[task]); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES[task])
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03); cb.set_label("Share of True Label"); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["0%", "50%", "100%"])
    finish(fig, f"{task}_confusion_matrix", f"{TITLE[task]} Confusion Matrix")


# ===================================================================== valence_data_comparison / valence_ablation_results
COND_COL = {"base": C["grey"], "A_targeted": C["green"], "B_random": C["orange"]}
COND_TICK = {"base": "Baseline", "A_targeted": "+ Targeted", "B_random": "+ Random"}


def ablation_seed_table():
    y = pd.read_csv(f"{ST}/val_labels_valence.csv").y.values
    rows = []
    for cond in COND_COL:
        for seed in range(5):
            p = ablation_probs(cond, seed).argmax(1)
            pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
            rows.append(dict(condition=cond, seed=seed, macro_f1=ablation_macro(cond, seed), neg_recall=rc[0], neg_precision=pr[0], neg_f1=f[0]))
    return pd.DataFrame(rows)


def valence_data_comparison():
    T = ablation_seed_table(); T.to_csv(f"{DATA}/valence_data_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax, metric, title, ylab in zip(axes, ("macro_f1", "neg_recall"), ("Macro-F1", "Negative-Class Recall"), ("Macro-F1", "Recall")):
        for i, cond in enumerate(COND_COL):
            v = T[T.condition == cond][metric].values
            m, sd = v.mean(), v.std(ddof=1)
            ax.bar(i, m, width=0.6, color=COND_COL[cond], alpha=0.9, zorder=2)
            ax.errorbar(i, m, yerr=sd, color="black", capsize=3.5, lw=1.0, zorder=4)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=14, facecolors="white", edgecolors="black", linewidths=0.7, zorder=5)
            ax.text(i, max(v.max(), m + sd) + 0.03, f"{m:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(list(COND_TICK.values()))
        ax.set_ylim(0, 1.0); ax.set_ylabel(ylab); ax.set_title(title); hgrid(ax)
    h = [Line2D([], [], color="black", marker="_", ms=7, ls="", label="Mean ± SD"), Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=5, label="Individual seed")]
    finish(fig, "valence_data_comparison", "Valence Data Comparison", h)


def forest(ax, rows, xlim):
    """rows: ('__header__', text) group headers or (label, diff, lo, hi, colour) points."""
    y, ylab, ypos = 0.0, [], []
    for r in rows:
        if r[0] == "__header__":
            ax.text(xlim[0], y - 0.05, r[1], fontsize=9, fontweight="bold", va="center", ha="left")
            y -= 0.7
            continue
        lab, d, lo, hi, col = r
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ms=5.5, capsize=3, lw=1.4, zorder=3)
        ylab.append(lab); ypos.append(y); y -= 0.75
    ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab); ax.set_ylim(y + 0.25, 0.4); ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)


def valence_ablation_results():
    B = pd.read_csv(f"{ST}/bootstrap_differences.csv"); B.to_csv(f"{DATA}/valence_ablation_results.csv", index=False)
    comps = [("A_targeted - base", "Targeted − Baseline", C["green"]), ("B_random - base", "Random − Baseline", C["orange"]), ("A_targeted - B_random", "Targeted − Random", C["blue"])]
    rows = []
    for mk, mname in (("macro_f1", "Macro-F1"), ("neg_r", "Negative Recall"), ("neg_p", "Negative Precision")):
        rows.append(("__header__", mname))
        for key, lab, col in comps:
            r = B[(B.comparison == key) & (B.metric == mk)].iloc[0]
            rows.append((lab, r["diff"], r.ci_lo, r.ci_hi, col))
    fig, ax = plt.subplots(figsize=(W, 3.9), constrained_layout=True)
    forest(ax, rows, (-0.2, 0.15))
    ax.set_xlabel("Difference in Score (95% CI)")
    finish(fig, "valence_ablation_results", "Valence Ablation Results")


# ===================================================================== per_class_performance
def per_class_table():
    rows = []
    for task, (size, seed) in RETAINED.items():
        y = val_labels(task).y.values
        pr, rc, f, sup = precision_recall_fscore_support(y, best_probs(task, size, seed).argmax(1), labels=[0, 1, 2], zero_division=0)
        for i, c in enumerate(CLASSES[task]):
            rows.append(dict(task=task, cls=c, support=int(sup[i]), precision=pr[i], recall=rc[i], f1=f[i]))
    return pd.DataFrame(rows)


def per_class_performance():
    T = per_class_table(); T.to_csv(f"{DATA}/per_class_performance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True, sharey=True)
    cols = [("precision", C["sky"], "Precision"), ("recall", C["orange"], "Recall"), ("f1", C["green"], "F1")]
    for ax, task in zip(axes, ("valence", "arousal")):
        t = T[T.task == task]
        x = np.arange(3)
        for k, (m, col, lab) in enumerate(cols):
            ax.bar(x + (k - 1) * 0.27, t[m].values, 0.26, color=col, zorder=2)
            for xi, v in zip(x + (k - 1) * 0.27, t[m].values):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES[task]); ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(TITLE[task]); hgrid(ax)
    axes[0].set_ylabel("Score")
    finish(fig, "per_class_performance", "Per-Class Performance", [Line2D([], [], marker="s", ls="", color=c, ms=8, label=l) for _, c, l in cols])


# ===================================================================== training_label_distribution
def training_label_distribution():
    S = pd.read_csv(f"{L3}/scaling_train_sets.csv")
    shades = {1783: "#9ECAE1", 3000: "#4292C6", 5000: "#08519C"}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True)
    for ax, task in zip(axes, ("valence", "arousal")):
        for k, size in enumerate(SIZES):
            x = S[S[f"in_{size}"]][f"{task}_maj"]
            n, cnt = int(x.notna().sum()), x.value_counts()
            for i, c in enumerate(CLASSES[task]):
                v = int(cnt.get(c.lower(), 0))
                rows.append(dict(task=task, size=size, cls=c, count=v, labeled_rows=n, rows_without_majority=size - n, proportion=v / n))
                ax.bar(i + (k - 1) * 0.27, v, 0.26, color=shades[size], zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES[task]); ax.set_ylabel("Number of Examples"); ax.set_title(TITLE[task])
        thousands(ax); hgrid(ax)
    h = [Line2D([], [], marker="s", ls="", color=shades[s], ms=8, label=f"{s:,}") for s in SIZES]
    pd.DataFrame(rows).to_csv(f"{DATA}/training_label_distribution.csv", index=False)
    finish(fig, "training_label_distribution", "Training Label Distribution", h, legend_title="Training Set Size")


# ===================================================================== single_model_vs_ensemble
def single_model_vs_ensemble():
    comp = pd.read_csv(f"{EN}/comparison.csv")
    names = [("selected single", "Single\nModel", C["grey"]), ("mean single (avg of seeds)", "Seed\nAverage", C["sky"]), ("ensemble", "Ensemble", C["pink"])]
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.2), constrained_layout=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
    rows, brow = [], []
    for j, task in enumerate(("valence", "arousal")):
        ax = axes[0, j]
        for k, (key, lab, col) in enumerate(names):
            v = comp[(comp.task == task) & (comp.model == key)].macro_f1.iloc[0]
            rows.append(dict(task=task, model=key, macro_f1=v))
            ax.bar(k, v, 0.62, color=col, zorder=2)
            ax.text(k, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in names]); ax.set_ylim(0, 1.0); ax.set_title(TITLE[task]); hgrid(ax)
        ax.set_ylabel("Macro-F1" if j == 0 else "")
        if j == 1:
            ax.tick_params(labelleft=False)
        ax = axes[1, j]
        B = pd.read_csv(f"{EN}/bootstrap_{task}.csv")
        ypos = [1, 0]
        for y, ref in zip(ypos, ("selected single", "mean single")):
            r = B[(B.ensemble_minus == ref) & (B.metric == "macro_f1")].iloc[0]
            brow.append(r.to_dict())
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", color=TASK_COL[task], ms=5.5, capsize=3, lw=1.4)
        ax.axvline(0, color="black", lw=0.9, ls="--"); ax.set_xlim(-0.065, 0.065); ax.set_ylim(-0.6, 1.6)
        ax.set_xticks([-0.05, 0, 0.05]); ax.set_xticklabels(["−0.05", "0.00", "0.05"])
        ax.set_yticks(ypos); ax.set_yticklabels(["Ensemble − Single Model", "Ensemble − Seed Average"] if j == 0 else ["", ""])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E3E3E3", lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("Macro-F1 Difference (95% CI)")
    pd.DataFrame(rows).to_csv(f"{DATA}/single_model_vs_ensemble_macro_f1.csv", index=False)
    pd.DataFrame(brow).to_csv(f"{DATA}/single_model_vs_ensemble_differences.csv", index=False)
    finish(fig, "single_model_vs_ensemble", "Single Model vs. Ensemble")


# ===================================================================== performance_by_gender_dyad
def dyad_table():
    """Macro-F1 over ALL THREE classes per dyad. Defined only if every class has >= 1 true turn; flagged unreliable if any class < MIN_SUPPORT."""
    rows = []
    for task, (size, seed) in RETAINED.items():
        lab = val_labels(task)
        p = best_probs(task, size, seed).argmax(1)
        old = pd.read_csv(f"{EN}/by_dyad_{task}.csv").set_index("dyad")
        for d in DYADS:
            m = (lab.dyad == d).values
            y, pr = lab.y.values[m], p[m]
            cnt = np.bincount(y, minlength=3)
            if (cnt == 0).any():
                f, status = np.nan, "undefined (a class has no true turns)"
            else:
                f = f1_score(y, pr, average="macro", labels=[0, 1, 2], zero_division=0)
                status = "unreliable (a class has < 5 true turns)" if (cnt < MIN_SUPPORT).any() else "reliable"
            rows.append(dict(task=task, dyad=d, n=int(m.sum()), class_counts=str(cnt.tolist()), macro_f1_3class=f, status=status,
                             earlier_present_class_macro_f1=float(old.loc[d, "single"])))
    return pd.DataFrame(rows)


def performance_by_gender_dyad():
    T = dyad_table(); T.to_csv(f"{DATA}/performance_by_gender_dyad.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True, sharey=True)
    for ax, task in zip(axes, ("valence", "arousal")):
        t = T[T.task == task].set_index("dyad").loc[DYADS].reset_index()
        for i, r in t.iterrows():
            if r.status.startswith("undefined"):
                ax.text(i, 0.5, "Undefined", ha="center", va="center", fontsize=8, color=C["grey"], style="italic")
            else:
                hollow = r.status.startswith("unreliable")
                ax.scatter(i, r.macro_f1_3class, s=48, facecolors="white" if hollow else TASK_COL[task], edgecolors=TASK_COL[task], linewidths=1.5, zorder=3)
                ax.text(i, r.macro_f1_3class + 0.045, f"{r.macro_f1_3class:.2f}", ha="center", fontsize=8)
            ax.text(i, 0.025, f"n = {r.n}", ha="center", fontsize=7.5, color=C["grey"])
        ax.set_xticks(range(4)); ax.set_xticklabels([DYAD_LABEL[d].replace(" → ", "\n→ ") for d in DYADS])
        ax.set_ylim(0, 1); ax.set_xlim(-0.5, 3.5); ax.set_title(TITLE[task]); hgrid(ax)
    axes[0].set_ylabel("Macro-F1")
    h = [Line2D([], [], marker="o", ls="", mfc=C["grey"], mec=C["grey"], ms=7, label="Reliable"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=C["grey"], mew=1.5, ms=7, label="Unreliable (a class has fewer than 5 turns)")]
    finish(fig, "performance_by_gender_dyad", "Performance by Gender Dyad", h)


# ===================================================================== per_episode_accuracy_change
def per_episode_accuracy_change():
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True, sharey=True)
    rows = []
    for ax, task in zip(axes, ("valence", "arousal")):
        d = pd.read_csv(f"{SC}/per_episode_accuracy_{task}.csv").sort_values("delta").reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        cols = np.where(d.delta > 1e-9, C["green"], np.where(d.delta < -1e-9, C["verm"], C["lgrey"]))
        ax.scatter(d["rank"], d.delta, s=32, c=cols, edgecolors="black", linewidths=0.5, zorder=3)
        ax.axhline(0, color="black", lw=0.9, ls="--", zorder=1)
        ax.set_xticks(range(2, 19, 4)); ax.set_xlim(0.2, 18.8)
        ax.set_xlabel("Episode (Ranked by Change)"); ax.set_title(TITLE[task]); hgrid(ax)
        for _, r in d.iterrows():
            rows.append(dict(task=task, rank=int(r["rank"]), episode_id=r.episode_id, n_turns=int(r.n), acc_1783=r.acc_1783, acc_5000=r.acc_5000, delta=r.delta))
    axes[0].set_ylabel("Accuracy Change")
    h = [Line2D([], [], marker="o", ls="", mfc=c, mec="black", mew=0.5, ms=6, label=l) for c, l in ((C["green"], "Improved"), (C["verm"], "Worse"), (C["lgrey"], "No change"))]
    pd.DataFrame(rows).to_csv(f"{DATA}/per_episode_accuracy_change.csv", index=False)
    finish(fig, "per_episode_accuracy_change", "Per-Episode Accuracy Change", h)


# ===================================================================== earlier_models_test_vs_human_gold
def earlier_models_test_vs_human_gold():
    M = pd.read_csv(f"{AU}/metrics_full.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), constrained_layout=True, sharey=True)
    rows = []
    for ax, task in zip(axes, ("valence", "arousal")):
        for x, (es, lab, mk) in enumerate((("test", "LLM Test", "o"), ("gold", "Human Gold", "s"))):
            r = M[(M.task == task) & (M.eval_set == es)].iloc[0]
            ax.errorbar(x, r.macro_f1, yerr=[[r.macro_f1 - r.macro_f1_lo], [r.macro_f1_hi - r.macro_f1]], fmt=mk, color=TASK_COL[task], ms=6.5, capsize=3.5, lw=1.4, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [r.base_majority_macro_f1] * 2, color="black", lw=1.3, ls=(0, (3, 2)), zorder=2)
            ax.text(x + 0.12, r.macro_f1_hi + 0.03, f"{r.macro_f1:.2f}", ha="left", fontsize=8.5)
            rows.append(dict(task=task, eval_set=es, n_turns=int(r.n), n_episodes=int(r.n_episodes), macro_f1=r.macro_f1, ci_lo=r.macro_f1_lo, ci_hi=r.macro_f1_hi,
                             majority_class_macro_f1=r.base_majority_macro_f1, model_variant=r.variant))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["LLM Test", "Human Gold"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.0); ax.set_title(TITLE[task]); hgrid(ax)
    axes[0].set_ylabel("Macro-F1")
    h = [Line2D([], [], color="black", lw=1.3, ls=(0, (3, 2)), label="Majority-class baseline")]
    pd.DataFrame(rows).to_csv(f"{DATA}/earlier_models_test_vs_human_gold.csv", index=False)
    finish(fig, "earlier_models_test_vs_human_gold", "Earlier Models: LLM Test vs. Human Gold", h)


if __name__ == "__main__":
    for fn in (performance_by_training_size, lambda: confusion_matrix_figure("valence"), lambda: confusion_matrix_figure("arousal"), valence_data_comparison, valence_ablation_results, per_class_performance, training_label_distribution,
               single_model_vs_ensemble, performance_by_gender_dyad, per_episode_accuracy_change, earlier_models_test_vs_human_gold):
        fn(); print("done", getattr(fn, "__name__", "confusion_matrix"))
