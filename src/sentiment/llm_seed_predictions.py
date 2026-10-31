#!/usr/bin/env python3
"""
Re-train the train-only LLM-label models for several seeds and save validation, test and
human-gold probabilities (llm_pipeline.py saved only test/gold from seed 0). Same split,
labels and hyperparameters as llm_pipeline.py; writes to a new directory and never touches
llm_ft2 or models/. Used for validation-only calibration / ensembling in tune_macro_f1.py.

Usage (GPU):
    python src/sentiment/llm_seed_predictions.py --seeds 0 1 2
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_valence import train  # noqa: E402
from llm_pipeline import CLASSES, load_labels, predict_sorted, split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="results/sentiment/llm_labels_passes")
    ap.add_argument("--out_dir", default="results/sentiment/llm_ft2_seeds")
    ap.add_argument("--base", default="cardiff_xlmr_multi")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = load_labels(a.scratch)
    d["split"] = split(d)
    ref = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
    assert (ref.set_index("turn_id").split.reindex(d.turn_id).values == d.split.values).all(), "split differs from llm_ft2"
    gold = pd.read_csv("results/sentiment/gold_sheet_labeled.csv")

    for task, cls in CLASSES.items():
        idx = {c: i for i, c in enumerate(cls)}
        lab = d[d[f"{task}_maj"].notna()].copy()
        lab["y"] = lab[f"{task}_maj"].map(idx)
        parts = {s: lab[lab.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
        parts["gold"] = gold[gold[task].isin(cls)].reset_index(drop=True)
        parts["gold"]["y"] = parts["gold"][task].map(idx)
        for ctx in (False, True):
            name = "context" if ctx else "no_context"
            for seed in a.seeds:
                t0 = time.time()
                args = SimpleNamespace(lr=a.lr, epochs=a.epochs, bs=16, max_len=256, weight_pow=0.5, context=ctx)
                tok, mdl = train(a.base, parts["train"], args, seed)
                for s in ("val", "test", "gold"):
                    p = predict_sorted(tok, mdl, parts[s], ctx)
                    keep = [c for c in ["turn_id", "gold_id", "episode_id", "y"] if c in parts[s]]
                    parts[s][keep].assign(p0=p[:, 0], p1=p[:, 1], p2=p[:, 2]).to_csv(
                        f"{a.out_dir}/{task}_{name}_seed{seed}_{s}.csv", index=False)
                del mdl; torch.cuda.empty_cache()
                print(task, name, "seed", seed, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
