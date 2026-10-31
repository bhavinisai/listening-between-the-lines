#!/usr/bin/env python3
"""
Evaluation figures for the LLM-label models (llm_pipeline.py output).

    eval1  accuracy / macro-F1 / kappa on the held-out LLM-majority test set vs. the human gold sheet
    eval2  confusion matrices (valence): held-out test and human gold
    eval3  confusion matrices (arousal): held-out test and human gold
    eval4  3-annotator LLM agreement (pairwise kappa) per task

Usage:
    python src/sentiment/make_llm_eval_figures.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3dc"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})
R, OUT = "results/sentiment/llm_ft2", "results/figures/sentiment_llm2"
CLS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
os.makedirs(OUT, exist_ok=True)
M = json.load(open(f"{R}/metrics.json"))


def eval1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, task in zip(axes, CLS):
        r = M[task]["variants"][M[task]["chosen"]]
        rows = [("LLM test\n(majority)", r["test"]), ("LLM test\n(3/3 agree)", r["test_all3"]), ("human\ngold", r["human_gold"])]
        x = np.arange(len(rows)); w = .26
        for k, (m, c) in enumerate([("acc", BLUE), ("macro_f1", ORANGE), ("kappa", GREEN)]):
            ax.bar(x + (k - 1) * w, [v[m] for _, v in rows], w, color=c, label={"acc": "accuracy", "macro_f1": "macro-F1", "kappa": "κ"}[m])
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\nn={v['n']}" for n, v in rows], fontsize=9)
        ax.set_title(f"{task} ({M[task]['chosen'].replace('_', ' ')})"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("score"); axes[0].legend(loc="upper right", fontsize=9)
    fig.savefig(f"{OUT}/eval1_test_vs_human.png"); plt.close(fig)


def cms(task, name):
    ch = M[task]["chosen"]
    cls = CLS[task]
    t = pd.read_csv(f"{R}/test_preds_{task}_{ch}.csv"); g = pd.read_csv(f"{R}/gold_preds_{task}_{ch}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (d, title) in zip(axes, [(t, "held-out LLM-majority test"), (g, "human gold")]):
        y = d.y.values; p = d[["p0", "p1", "p2"]].values.argmax(1)
        cm = confusion_matrix(y, p, labels=[0, 1, 2]); cn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.grid(False); ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cn[i, j]:.0%}", ha="center", va="center", color="white" if cn[i, j] > .55 else INK, fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(cls); ax.set_yticks(range(3)); ax.set_yticklabels(cls)
        ax.set_xlabel("predicted"); ax.set_title(title, fontsize=11)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].set_ylabel("label")
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)


def eval4():
    d = pd.read_csv(f"{R}/labels_majority_split.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4)); w = .25
    for k, task in enumerate(CLS):
        ks = [cohen_kappa_score(d[f"{task}_{a}"], d[f"{task}_{b}"]) for a, b in [("A", "B"), ("A", "C"), ("B", "C")]]
        ax.bar(np.arange(3) + (k - .5) * w * 1.6, ks, w * 1.5, color=[BLUE, ORANGE][k], label=task)
    ax.set_xticks(range(3)); ax.set_xticklabels(["A–B", "A–C", "B–C"]); ax.set_ylim(0, 1)
    ax.set_ylabel("Cohen's κ between LLM passes"); ax.legend()
    fig.savefig(f"{OUT}/eval4_llm_annotator_agreement.png"); plt.close(fig)


eval1(); cms("valence", "eval2_confusion_valence"); cms("arousal", "eval3_confusion_arousal"); eval4()
print("figures ->", OUT)
