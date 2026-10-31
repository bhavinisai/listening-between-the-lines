#!/usr/bin/env python3
"""
Check the emotion-weighted arousal score against the hand-labeled gold arousal.

The gold sheet is ~98% 'medium', so only ranking agreement can be reported:
Spearman between the score and low<medium<high, and AUC for the few gold
high / low turns vs. the rest. Treat this as a sanity check, not validation.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from score_models import EMO
from score_corpus import AROUSAL_W

df = pd.read_csv("results/sentiment/gold_model_probs.csv")
df = df[df.arousal.isin(["low", "medium", "high"])]
df["score"] = df[[f"emotion__{e}" for e in EMO]].values @ np.array([AROUSAL_W[e] for e in EMO])
lev = df.arousal.map({"low": 0, "medium": 1, "high": 2})
lines = [f"gold n={len(df)}: {df.arousal.value_counts().to_dict()}",
         f"Spearman(score, low<medium<high) = {spearmanr(df.score, lev).statistic:.3f} (p={spearmanr(df.score, lev).pvalue:.3f})"]
for k in ["high", "low"]:
    y = (df.arousal == k).astype(int)
    lines.append(f"AUC for gold '{k}' (n={y.sum()}) vs rest = {roc_auc_score(y, df.score if k == 'high' else -df.score):.3f}")
print("\n".join(lines))
open("results/sentiment/arousal_gold_check.txt", "w").write("\n".join(lines) + "\n")
