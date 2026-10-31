#!/usr/bin/env python3
"""
Score every turn in sentiment_turns.csv.

Columns written (in addition to the turn table):
    ft_neg / ft_neu / ft_pos    fine-tuned valence model class probabilities
    valence                     ft_pos - ft_neg, in [-1, 1]
    valence_label               argmax class
    zs_valence                  same score from zero-shot cardiff_xlmr_multi (robustness check)
    emo_*                       7-class emotion probabilities
    arousal                     emotion-weighted activation in [0, 1] (see AROUSAL_W)

Usage:
    python src/sentiment/score_corpus.py --valence_model models/valence_ft_final
"""

import argparse

import numpy as np
import pandas as pd
import torch

from score_models import score

LAB = ["negative", "neutral", "positive"]
# Activation level of each emotion (Russell circumplex-style): angry / afraid /
# excited / surprised states are high-arousal, sadness is low, neutral is none.
AROUSAL_W = {"anger": 1.0, "fear": 0.9, "surprise": 0.8, "joy": 0.7, "disgust": 0.6, "sadness": 0.25, "neutral": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="results/sentiment/sentiment_turns.csv")
    ap.add_argument("--valence_model", required=True)
    ap.add_argument("--arousal_model", help="classifier (low/medium/high in the 3 output slots); replaces the emotion-based arousal")
    ap.add_argument("--out", default="results/sentiment/turn_scores.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.turns)
    texts = df.text.astype(str).tolist()
    p, _ = score("cardiff_xlmr", texts, path=args.valence_model)
    df[["ft_neg", "ft_neu", "ft_pos"]] = p
    df["valence"] = p[:, 2] - p[:, 0]
    df["valence_label"] = [LAB[i] for i in p.argmax(1)]
    print("valence done", flush=True)

    z, _ = score("cardiff_xlmr_multi", texts)
    df["zs_valence"] = z[:, 2] - z[:, 0]
    print("zero-shot done", flush=True)

    e, cols = score("emotion", texts)
    for j, c in enumerate(cols):
        df[f"emo_{c}"] = e[:, j]
    df["arousal"] = e @ np.array([AROUSAL_W[c] for c in cols])
    if args.arousal_model:
        df["arousal_emo"] = df["arousal"]
        a, _ = score("cardiff_xlmr", texts, path=args.arousal_model)  # columns = low, medium, high
        df[["ar_low", "ar_med", "ar_high"]] = a
        df["arousal"] = 0.5 * a[:, 1] + a[:, 2]  # expected arousal in [0, 1]
    df.to_csv(args.out, index=False)
    print(f"{len(df)} turns -> {args.out}")


if __name__ == "__main__":
    main()
