#!/usr/bin/env python3
"""
Evaluate valence scoring against the hand-labeled gold sheet.

For every pretrained model: raw zero-shot accuracy / macro-F1, and the same
model after a small logistic-regression recalibration of its class
probabilities (fit with episode-grouped cross-validation, so no episode is in
both train and test). Also a stacked model over all models' probabilities.
Gold is ~72% neutral, so the majority-class baseline and macro-F1 are reported
next to accuracy.

Usage:
    python src/sentiment/evaluate_valence.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LAB = ["negative", "neutral", "positive"]
SENT = ["cardiff_xlmr", "cardiff_xlmr_multi", "cardiff_roberta_en", "tabularis_multi", "lxyuan_distil_multi"]


def wilson(k, n, a=0.05):
    from statsmodels.stats.proportion import proportion_confint
    return proportion_confint(k, n, a, method="wilson")


def cv_predict(X, y, groups, C=1.0, reps=5):
    """Episode-grouped CV predictions averaged over repeated shuffled folds (majority vote of probs)."""
    P = np.zeros((len(y), 3))
    for r in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
            m.fit(X[tr], y[tr])
            P[te] += m.predict_proba(X[te])
    return P / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="results/sentiment/gold_model_probs.csv")
    ap.add_argument("--out_dir", default="results/sentiment")
    args = ap.parse_args()

    df = pd.read_csv(args.probs)
    df = df[df.valence.isin(LAB)].reset_index(drop=True)
    y = df.valence.map({l: i for i, l in enumerate(LAB)}).values
    g = df.episode_id.values
    print(f"gold n={len(df)}, episodes={len(set(g))}; majority baseline "
          f"{np.bincount(y).max() / len(y):.3f}")

    def feats(m):
        return df[[f"{m}__neg", f"{m}__neu", f"{m}__pos"]].values

    rows, preds = [], {}
    def add(name, kind, pred):
        k = int((pred == y).sum())
        lo, hi = wilson(k, len(y))
        rows.append(dict(model=name, kind=kind, accuracy=k / len(y), ci_lo=lo, ci_hi=hi,
                         macro_f1=f1_score(y, pred, average="macro"),
                         kappa=cohen_kappa_score(y, pred),
                         recall_neg=(pred[y == 0] == 0).mean(), recall_neu=(pred[y == 1] == 1).mean(),
                         recall_pos=(pred[y == 2] == 2).mean()))
        preds[(name, kind)] = pred

    add("majority baseline", "baseline", np.full(len(y), 1))
    for m in SENT:
        add(m, "zero-shot", feats(m).argmax(1))
        add(m, "recalibrated (grouped CV)", cv_predict(feats(m), y, g).argmax(1))
    emo = df[[c for c in df.columns if c.startswith("emotion__")]].values
    allX = np.hstack([feats(m) for m in SENT] + [emo])
    for C in (0.05, 0.3, 1.0):
        add(f"stack all models (C={C})", "stacked (grouped CV)", cv_predict(allX, y, g, C=C).argmax(1))

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(f"{args.out_dir}/valence_eval_models.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    best = res[res.kind != "baseline"].iloc[0]
    bp = preds[(best.model, best.kind)]
    cm = pd.DataFrame(confusion_matrix(y, bp), index=LAB, columns=LAB)
    cm.to_csv(f"{args.out_dir}/valence_eval_best_confusion.csv")
    print("\nbest by macro-F1:", best.model, best.kind, "\n", cm)
    df.assign(pred=[LAB[i] for i in bp]).to_csv(f"{args.out_dir}/valence_eval_best_preds.csv", index=False)


if __name__ == "__main__":
    main()
