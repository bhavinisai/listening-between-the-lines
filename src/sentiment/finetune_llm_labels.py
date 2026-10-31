#!/usr/bin/env python3
"""
Fine-tune a classifier on the LLM-labeled turns (rows where both independent
LLM passes agree) for one task (valence or arousal).

Reports, per run:
  * episode-grouped 5-fold CV against the LLM consensus labels
  * a final model trained on all consensus rows, scored on the 400 human-labeled
    gold turns (episodes disjoint from the LLM-labeled turns)
Writes one JSON line of metrics to --metrics and optionally saves the final model.

Usage:
    python src/sentiment/finetune_llm_labels.py --task valence --base cardiff_xlmr_multi \
        --lr 2e-5 --epochs 4 --tag v_cardiff --metrics results/sentiment/llm_ft/metrics.jsonl \
        --final_dir models/llm_valence_v_cardiff
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_valence import predict, train  # noqa: E402

CLASSES = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}


def metrics(y, pred):
    return dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                kappa=cohen_kappa_score(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(CLASSES), required=True)
    ap.add_argument("--labels", default="results/sentiment/llm_labels/llm_labels_AB.csv")
    ap.add_argument("--gold", default="results/sentiment/gold_sheet_labeled.csv")
    ap.add_argument("--base", default="cardiff_xlmr_multi")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_pow", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--final_dir")
    args = ap.parse_args()
    args.context = False
    cls = CLASSES[args.task]
    idx = {c: i for i, c in enumerate(cls)}

    d = pd.read_csv(args.labels)
    a, b = d[f"{args.task}_A"], d[f"{args.task}_B"]
    d = d[a == b].copy()
    d["y"] = a[a == b].map(idx)
    d = d.reset_index(drop=True)
    print(f"{args.task}: {len(d)} consensus rows; class counts {np.bincount(d.y, minlength=3).tolist()}", flush=True)

    P = np.zeros((len(d), 3))
    for tr, te in StratifiedGroupKFold(args.folds, shuffle=True, random_state=0).split(d, d.y, d.episode_id):
        tok, mdl = train(args.base, d.iloc[tr], args, 0)
        P[te] = predict(tok, mdl, d.iloc[te], args)
        del mdl; torch.cuda.empty_cache()
    cv = metrics(d.y, P.argmax(1))
    print("CV vs LLM consensus:", cv, flush=True)

    tok, mdl = train(args.base, d, args, 0)
    g = pd.read_csv(args.gold)
    g = g[g[args.task].isin(cls)].reset_index(drop=True)
    gp = predict(tok, mdl, g, args)
    gm = metrics(g[args.task].map(idx), gp.argmax(1))
    print("human gold:", gm, flush=True)
    pd.DataFrame(gp, columns=["p0", "p1", "p2"]).assign(gold_id=g.gold_id, gold=g[args.task]).to_csv(
        os.path.join(os.path.dirname(args.metrics), f"gold_preds_{args.task}_{args.tag}.csv"), index=False)

    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "a") as f:
        f.write(json.dumps(dict(task=args.task, tag=args.tag, base=args.base, lr=args.lr, epochs=args.epochs,
                                n_train=len(d), cv=cv, gold=gm)) + "\n")
    if args.final_dir:
        # the scorers read class order from id2label; keep the 3 slots named neg/neu/pos for both tasks
        mdl.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        mdl.config.label2id = {v: k for k, v in mdl.config.id2label.items()}
        mdl.save_pretrained(args.final_dir, safe_serialization=True); tok.save_pretrained(args.final_dir)


if __name__ == "__main__":
    main()
