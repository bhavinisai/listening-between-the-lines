#!/usr/bin/env python3
"""
Report 2 figures for the question-type classifier (CPU only, no training).

Reads only saved outputs:
  results/question_type_v2/experiments/E{1,2,3,6}_*/oof_predictions.csv, metrics.json
  results/question_type_v2/experiments/A_v1_baseline/metrics.json      (historical v1)
  results/question_type_v2/v3_gold_check.csv                           (exploratory)
  results/question_type_v2/v3/train_dev_v3.csv                         (exploratory)
  results/question_type_v2/train_dev.csv                               (LLM pass agreement)

Ensembles (E1+E3, E1+E2+E3) are equal-weight averages of the saved out-of-fold
probabilities, i.e. the same post hoc combination as before; nothing is refit.

Writes PNGs (300 dpi) and the numbers behind them to --out_dir. Refuses to
overwrite existing files unless --overwrite is passed.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

T = ["closed", "open", "leading", "personal", "professional", "challenge"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal: none -> sqrt -> full weight
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})

EXP = "results/question_type_v2/experiments"
RUNS = {  # key: (folder, short label, long description)
    "E1": ("E1_base_nocontext_cw", "E1", "no context, full class weights"),
    "E2": ("E2_base_nocontext_cwsqrt", "E2", "no context, square-root class weights"),
    "E3": ("E3_base_context_cw", "E3", "with context, full class weights"),
    "E6": ("E6_base_nocontext_unweighted", "E6", "no context, unweighted"),
}
ENS = {"E1+E3": ["E1", "E3"], "E1+E2+E3": ["E1", "E2", "E3"]}
ORDER = ["E6", "E2", "E1", "E3", "E1+E3", "E1+E2+E3"]


def save(fig, d, name, overwrite):
    path = os.path.join(d, name)
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ---------------------------------------------------------------- data
def load_cv():
    oof = {k: pd.read_csv(f"{EXP}/{f}/oof_predictions.csv").set_index("qid") for k, (f, _, _) in RUNS.items()}
    ref = oof["E1"]
    for k, x in oof.items():
        x = x.loc[ref.index]
        assert (x.label == ref.label).all() and (x.fold == ref.fold).all(), k
        oof[k] = x
    probs = {k: x[[f"p_{c}" for c in T]].values for k, x in oof.items()}
    for k, ms in ENS.items():
        probs[k] = sum(probs[m] for m in ms) / len(ms)
    y = ref.label.values
    pred = {k: np.array(T)[p.argmax(1)] for k, p in probs.items()}
    # sanity: recomputed single-model scores must match the saved metrics.json
    for k, (f, _, _) in RUNS.items():
        m = json.load(open(f"{EXP}/{f}/metrics.json"))
        assert abs(f1_score(y, pred[k], average="macro") - m["macro_f1"]) < 1e-9, k
        assert (confusion_matrix(y, pred[k], labels=T) == np.array(m["confusion_matrix"])).all(), k
    return y, ref.fold.values, ref.host_id.values, pred


def summary_table(y, folds, pred):
    rows = []
    for k in ORDER:
        p = pred[k]
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=T, zero_division=0)
        r = {"model": k, "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}
        for i, c in enumerate(T):
            r[f"f1_{c}"], r[f"precision_{c}"], r[f"recall_{c}"] = f1[i], pr[i], rc[i]
            r[f"n_pred_{c}"] = int((p == c).sum())
        for fo in sorted(set(folds)):
            m = folds == fo
            r[f"macro_f1_fold{fo}"] = f1_score(y[m], p[m], labels=T, average="macro", zero_division=0)
            r[f"acc_fold{fo}"] = accuracy_score(y[m], p[m])
        rows.append(r)
    return pd.DataFrame(rows)


def xlabel(k):
    if k in RUNS:
        parts = [x.replace(" class weights", "\nclass weights") for x in RUNS[k][2].split(", ")]
        return "\n".join([k] + parts)
    return f"{k}\nensemble"


# ---------------------------------------------------------------- figures
def fig01_model_comparison(S, d, ow):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.array([0, 1, 2, 3, 4.5, 5.5]); w = 0.36
    for off, col, colr, lab in [(-w / 2, "accuracy", BLUE, "Accuracy"), (w / 2, "macro_f1", ORANGE, "Macro-F1")]:
        v = S.set_index("model").loc[ORDER, col].values
        b = ax.bar(x + off, v, w - 0.03, color=colr, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.01, f"{vi:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [xlabel(k) for k in ORDER], fontsize=8.5)
    ax.set_ylim(0, 0.8); ax.set_ylabel("Score")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("Accuracy and macro-F1 by model")
    save(fig, d, "fig01_cv_model_comparison.png", ow)


def fig02_fold_stability(S, folds, hosts, d, ow):
    fos = sorted(set(folds))
    names = []
    for fo in fos:
        h = sorted(set(hosts[folds == fo]))
        named = [x for x in h if not x.startswith("unknown")]
        lab = named[0] if len(h) == 1 else f"{len(h)} hosts"
        names.append(f"Fold {fo}\n{lab}\nn = {(folds == fo).sum():,}")
    M = S.set_index("model").loc[ORDER, [f"macro_f1_fold{f}" for f in fos]].values
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    im = ax.imshow(M, cmap=SEQ, vmin=0.3, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(fos)), names, fontsize=8.5)
    ax.set_yticks(range(len(ORDER)), [k if k in RUNS else f"{k} ensemble" for k in ORDER])
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Macro-F1")
    ax.set_title("Macro-F1 by held-out host fold")
    save(fig, d, "fig02_cv_macro_f1_by_fold.png", ow)


def fig03_per_class_f1(S, y, d, ow):
    M = S.set_index("model").loc[ORDER, [f"f1_{c}" for c in T]].values
    sup = [int((y == c).sum()) for c in T]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(T)), [f"{c}\n(n = {n})" for c, n in zip(T, sup)])
    ax.set_yticks(range(len(ORDER)), [k if k in RUNS else f"{k} ensemble" for k in ORDER])
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("F1")
    ax.set_title("F1 score by question type and model")
    save(fig, d, "fig03_cv_per_class_f1.png", ow)


def draw_cm(ax, y, p, title, small=False):
    cm = confusion_matrix(y, p, labels=T)
    R = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(R, cmap=SEQ, vmin=0, vmax=1)
    fs = 7.5 if small else 9.5
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{R[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=fs,
                    color="white" if R[i, j] > 0.5 else INK, linespacing=1.1)
    ax.set_xticks(range(6), T, rotation=35, ha="right", fontsize=fs)
    ax.set_yticks(range(6), T, fontsize=fs)
    ax.set_xlabel("Predicted label", fontsize=fs + 0.5); ax.set_ylabel("True label", fontsize=fs + 0.5)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title, fontsize=10 if small else 11)
    return im


def fig04_confusions(y, pred, d, ow):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.6))
    for ax, k in zip(axes.flat, ORDER):
        t = f"{k} ({RUNS[k][2]})" if k in RUNS else f"{k} ensemble"
        im = draw_cm(ax, y, pred[k], t, small=True)
    fig.suptitle("Confusion matrices for all models", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.97), h_pad=2.0)
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax).set_label("Share of true class (recall)")
    save(fig, d, "fig04_cv_confusion_all_models.png", ow)

    for k in ["E1", "E1+E3"]:
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        t = f"{k} ({RUNS[k][2]})" if k in RUNS else f"the {k} ensemble"
        im = draw_cm(ax, y, pred[k], f"Confusion matrix for {t}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Share of true class (recall)")
        fig.tight_layout()
        save(fig, d, {"E1": "fig05_cv_confusion_e1.png", "E1+E3": "fig06_cv_confusion_e1_e3.png"}[k], ow)


def fig06_weighting(S, y, d, ow):
    ks = ["E6", "E2", "E1"]
    labs = ["E6: unweighted", "E2: square-root class weights", "E1: full class weights"]
    St = S.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(6); w = 0.27
    true_share = np.array([(y == c).mean() for c in T])
    panels = [("recall", "Recall", "(a) Recall"),
              ("f1", "F1", "(b) F1 score"),
              ("n_pred", "Share of predictions", "(c) Predicted vs. true class share")]
    for ax, (key, ylab, title) in zip(axes, panels):
        for i, (k, lab, col) in enumerate(zip(ks, labs, RAMP3)):
            v = St.loc[k, [f"{key}_{c}" for c in T]].values.astype(float)
            if key == "n_pred":
                v = v / len(y)
            ax.bar(x + (i - 1) * w, v, w - 0.02, color=col, label=lab)
        if key == "n_pred":
            ax.scatter(x, true_share, marker="_", s=420, color=ORANGE, lw=2.2, zorder=3, label="True class share")
            ax.set_ylim(0, 0.42)
        else:
            ax.set_ylim(0, 1)
        ax.set_xticks(x, T, rotation=25, ha="right"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Effect of class weighting on per-class performance", fontweight="bold", color=INK, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig07_cv_class_weighting.png", ow)


def fig07_labeler_vs_gold(d, ow):
    g = pd.read_csv("results/question_type_v2/v3_gold_check.csv")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 4]})
    rows = []
    for ax, groups in [(a0, [("all questions", g)]), (a1, [(c, g[g.human == c]) for c in T])]:
        x = np.arange(len(groups)); w = 0.36
        for i, (col, colr, lab) in enumerate([("v2", BLUE, "Original guidelines"), ("v3", ORANGE, "Revised guidelines")]):
            vals, lo, hi = [], [], []
            for name, s in groups:
                k, n = int((s[col] == s.human).sum()), len(s)
                l_, h_ = wilson(k, n); vals.append(k / n); lo.append(k / n - l_); hi.append(h_ - k / n)
                rows.append({"scheme": col, "gold_class": name, "agree": k, "n": n, "rate": k / n, "ci_lo": l_, "ci_hi": h_})
            xs = x + (i - 0.5) * w
            ax.bar(xs, vals, w - 0.03, color=colr, label=lab)
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9, capsize=3)
            for xi, v, h_ in zip(xs, vals, hi):
                ax.text(xi, v + h_ + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, [f"{nm}\n(n = {len(s)})" for nm, s in groups])
        ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.grid(axis="x", visible=False)
    a0.set_ylabel("Agreement with gold label"); a0.set_title("(a) Overall", loc="left")
    a1.set_title("(b) By question type", loc="left")
    a1.legend(loc="upper right", bbox_to_anchor=(1, 1.13), ncol=2)
    fig.suptitle("Agreement between LLM annotations and human gold labels", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, d, "fig08_exploratory_labels_vs_gold.png", ow)
    return pd.DataFrame(rows)


def fig08_v2_v3_transitions(d, ow):
    t = pd.read_csv("results/question_type_v2/v3/train_dev_v3.csv")
    t = t[t.label.notna()]
    cm = pd.crosstab(t.label_v2, t.label).reindex(index=T, columns=T, fill_value=0).values
    changed = cm.sum() - np.trace(cm)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    off = cm.astype(float).copy(); np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap=SEQ, vmin=0, vmax=max(30, np.nanmax(off)))
    for i in range(6):
        for j in range(6):
            diag = i == j
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9.5,
                    color=MUTED if diag else ("white" if cm[i, j] > 18 else INK), style="italic" if diag else "normal")
    ax.set_xticks(range(6), T, rotation=35, ha="right"); ax.set_yticks(range(6), T)
    ax.set_xlabel("Revised label"); ax.set_ylabel("Original label")
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Label changes after annotation guideline revision")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Number of changed labels")
    fig.tight_layout()
    save(fig, d, "fig09_exploratory_v2_v3_label_changes.png", ow)


def fig09_v1_historical(d, ow):
    m = json.load(open(f"{EXP}/A_v1_baseline/metrics.json"))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    groups = ["Accuracy", "Macro-F1"] + T
    x = np.r_[0, 1, np.arange(len(T)) + 2.5]; w = 0.36
    for i, (key, colr, lab) in enumerate([("test_llm", BLUE, "LLM-labeled test set"), ("gold", ORANGE, "Human gold set")]):
        r = m[key]
        v = [r["accuracy"], r["macro_f1"]] + [r["per_class"][c]["f1"] for c in T]
        xs = x + (i - 0.5) * w
        ax.bar(xs, v, w - 0.03, color=colr, label=f"{lab} (n = {r['n']})")
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [g if i < 2 else f"F1\n{g}" for i, g in enumerate(groups)])
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=1)
    ax.set_title("Baseline classifier performance on held-out data")
    save(fig, d, "fig10_historical_v1_baseline.png", ow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/question_type_v2/report2_figures")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    y, folds, hosts, pred = load_cv()
    S = summary_table(y, folds, pred)
    p = os.path.join(a.out_dir, "cv_model_summary.csv")
    if os.path.exists(p) and not a.overwrite:
        raise SystemExit(f"refusing to overwrite {p}")
    S.to_csv(p, index=False)
    fig01_model_comparison(S, a.out_dir, a.overwrite)
    fig02_fold_stability(S, folds, hosts, a.out_dir, a.overwrite)
    fig03_per_class_f1(S, y, a.out_dir, a.overwrite)
    fig04_confusions(y, pred, a.out_dir, a.overwrite)
    fig06_weighting(S, y, a.out_dir, a.overwrite)
    fig07_labeler_vs_gold(a.out_dir, a.overwrite).to_csv(os.path.join(a.out_dir, "exploratory_labels_vs_gold.csv"), index=False)
    fig08_v2_v3_transitions(a.out_dir, a.overwrite)
    fig09_v1_historical(a.out_dir, a.overwrite)


if __name__ == "__main__":
    main()
