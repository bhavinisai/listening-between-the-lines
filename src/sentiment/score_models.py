#!/usr/bin/env python3
"""
Score turns with several pretrained sentiment / emotion models and write the
class probabilities, harmonised to valence (neg / neu / pos) for the sentiment
models and 7 emotions for the emotion model.

Usage:
    python src/sentiment/score_models.py --input results/sentiment/gold_sheet_labeled.csv \
        --out results/sentiment/gold_model_probs.csv
    python src/sentiment/score_models.py --input results/sentiment/sentiment_turns.csv \
        --out results/sentiment/turn_model_probs.csv --models cardiff_xlmr emotion
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "cardiff_xlmr": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "cardiff_xlmr_multi": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    "cardiff_roberta_en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularis_multi": "tabularisai/multilingual-sentiment-analysis",
    "lxyuan_distil_multi": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "emotion": "j-hartmann/emotion-english-distilroberta-base",
}
VAL = ["neg", "neu", "pos"]
EMO = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def harmonise(name, id2label, probs):
    """Return (n, k) array with columns in canonical order."""
    labels = [id2label[i].lower() for i in range(len(id2label))]
    if name == "emotion":
        return probs[:, [labels.index(e) for e in EMO]], EMO
    if name == "tabularis_multi":
        g = lambda k: probs[:, labels.index(k)]
        return np.stack([g("very negative") + g("negative"), g("neutral"),
                         g("positive") + g("very positive")], 1), VAL
    m = {"negative": "neg", "neutral": "neu", "positive": "pos"}
    return probs[:, [labels.index(k) for k, v in m.items()]], VAL


@torch.no_grad()
def score(name, texts, bs=64, max_len=256, device="cuda", path=None):
    path = path or f"models/hf_converted/{name}"  # safetensors copies, see convert_to_safetensors.py
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    if device == "cuda":
        mdl = mdl.half()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), mdl.config.num_labels), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        out[idx] = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
    return harmonise(name, mdl.config.id2label, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    texts = df.text.astype(str).tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = df.copy()
    for name in args.models:
        p, cols = score(name, texts, device=dev)
        for j, c in enumerate(cols):
            res[f"{name}__{c}"] = p[:, j]
        print("scored", name, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
