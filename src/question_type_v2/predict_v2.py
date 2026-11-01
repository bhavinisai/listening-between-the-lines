#!/usr/bin/env python3
"""
Classify host questions with one or more trained v2/v3 runs (train_v2.py
--mode final --save_model). With several run dirs the class probabilities are
averaged (probability ensemble). Context/max_length are read from each run's
config.json so inference matches training.

Input : CSV with qid,text,prev_sentence,prev_turn_tail (e.g. pool_host_questions.csv)
Output: CSV with qid, p_<class> (ensemble), pred, confidence (max prob), plus
        per-model argmax columns pred_<k>.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import DS, LABELS, collate, encode, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.input)
    df["text"] = df.text.astype(str)
    all_p = []
    out = df[["qid"]].copy()
    for i, rd in enumerate(args.run_dirs):
        cfg = json.load(open(os.path.join(rd, "config.json")))
        mdir = os.path.join(rd, "model")
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(device)
        dummy = df.assign(label="closed")
        dl = DataLoader(DS(encode(tok, dummy, cfg["context"], cfg["max_length"]), np.zeros(len(df), dtype=int)),
                        batch_size=args.batch_size, collate_fn=collate(tok))
        p = predict(model, dl, device)
        all_p.append(p)
        out[f"pred_m{i}"] = [LABELS[j] for j in p.argmax(1)]
        print(f"[OK] {rd}: context={cfg['context']} max_length={cfg['max_length']}", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(all_p, axis=0)
    for j, l in enumerate(LABELS):
        out[f"p_{l}"] = P[:, j]
    out["pred"] = [LABELS[j] for j in P.argmax(1)]
    out["confidence"] = P.max(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] {len(out)} questions -> {args.out}; predicted distribution:")
    print(out.pred.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
