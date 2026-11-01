#!/usr/bin/env python3
"""
v2 controlled experiments for the question-type classifier.

Two modes (never mixes them, so the sealed test set is not tuned against):

    --mode cv     host-grouped K-fold CV on the DEV labeled set. Used for model
                  selection. Writes oof_predictions.csv + metrics.json.
    --mode final  train on ALL dev labels (optionally with a held-out-by-host
                  early-stopping fold), then score every --eval_csv given
                  (gold, sealed test). Writes metrics_<name>.json per eval set.

Inputs (CSV, one row per question):
    train_csv  columns: qid, text, label, host_id, prev_sentence, prev_turn_tail
    eval_csv   same columns (gold / test)

Options that define an experiment:
    --model xlm-roberta-base | xlm-roberta-large | ...
    --context none | prev     "prev" encodes (question, prev_sentence + guest tail)
    --max_length, --lr, --epochs, --batch_size, --class_weighted, --seed

Metrics reported: accuracy, macro-F1, per-class precision/recall/F1/support,
confusion matrix (rows = reference label, cols = prediction).
Nothing here touches v1 files.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]
L2I = {l: i for i, l in enumerate(LABELS)}


def host_folds(df, k, seed):
    """Greedy balanced assignment of whole hosts to k folds (no host in two folds)."""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("host_id").size().sample(frac=1, random_state=seed).sort_values(ascending=False, kind="stable")
    load = np.zeros(k)
    assign = {}
    for h, n in sizes.items():
        f = int(np.argmin(load + rng.random(k) * 1e-6))
        assign[h] = f
        load[f] += n
    return df.host_id.map(assign).values


def encode(tok, df, context, max_length):
    a = df.text.astype(str).tolist()
    if context == "prev":
        b = (df.prev_sentence.fillna("").astype(str) + " | " + df.prev_turn_tail.fillna("").astype(str)).tolist()
        return tok(a, b, truncation="only_second", max_length=max_length)
    return tok(a, truncation=True, max_length=max_length)


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = int(self.labels[i])
        return d


def collate(tok):
    def f(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return f


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**b).logits
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def fit(args, tok, tr, device, val=None):
    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(device)
    y = tr.label.map(L2I).values
    ds = DS(encode(tok, tr, args.context, args.max_length), y)
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate(tok), generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    w = None
    if args.class_weighted:
        cnt = np.bincount(y, minlength=len(LABELS)).clip(min=1)
        w = torch.tensor(len(y) / (len(LABELS) * cnt), dtype=torch.float).to(device)
        w = w ** args.weight_power
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(args.epochs):
        model.train()
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            lab = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**b).logits
            loss = lossf(logits.float(), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    return model


def score(df_true, probs):
    y = df_true.label.map(L2I).values
    p = probs.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(LABELS)), zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1.mean()),
        "per_class": {l: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                      for i, l in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(LABELS))).tolist(),
        "labels_order": LABELS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "final"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--eval_csv", nargs="*", default=[], help="name=path pairs, final mode only")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--context", choices=["none", "prev"], default="none")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--class_weighted", action="store_true")
    ap.add_argument("--weight_power", type=float, default=1.0, help="exponent on inverse-frequency weights (0.5 = sqrt)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    tr = pd.read_csv(args.train_csv)
    tr = tr[tr.label.notna()].reset_index(drop=True)   # v3 files carry unresolved rows with no label
    assert tr.label.isin(LABELS).all()
    tok = AutoTokenizer.from_pretrained(args.model)
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    if args.mode == "cv":
        tr["fold"] = host_folds(tr, args.folds, args.seed)
        oof = np.zeros((len(tr), len(LABELS)))
        for f in range(args.folds):
            m = tr.fold == f
            model = fit(args, tok, tr[~m], device)
            dl = DataLoader(DS(encode(tok, tr[m], args.context, args.max_length), tr[m].label.map(L2I).values),
                            batch_size=64, collate_fn=collate(tok))
            oof[m.values] = predict(model, dl, device)
            fm = score(tr[m], oof[m.values])
            print(f"[fold {f}] n={m.sum()} acc={fm['accuracy']:.3f} macroF1={fm['macro_f1']:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
        res = score(tr, oof)
        res["fold_sizes"] = tr.fold.value_counts().sort_index().tolist()
        json.dump(res, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
        out = tr[["qid", "host_id", "label", "fold"]].copy()
        for i, l in enumerate(LABELS):
            out[f"p_{l}"] = oof[:, i]
        out.to_csv(os.path.join(args.out_dir, "oof_predictions.csv"), index=False)
        print(f"[CV] n={res['n']} acc={res['accuracy']:.3f} macroF1={res['macro_f1']:.3f}")
        for l in LABELS:
            c = res["per_class"][l]
            print(f"   {l:12s} P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f} n={c['support']}")
    else:
        model = fit(args, tok, tr, device)
        for spec in args.eval_csv:
            name, path = spec.split("=", 1)
            ev = pd.read_csv(path)
            dl = DataLoader(DS(encode(tok, ev, args.context, args.max_length), ev.label.map(L2I).values),
                            batch_size=64, collate_fn=collate(tok))
            pr = predict(model, dl, device)
            res = score(ev, pr)
            json.dump(res, open(os.path.join(args.out_dir, f"metrics_{name}.json"), "w"), indent=2)
            o = ev[["qid", "label"]].copy()
            for i, l in enumerate(LABELS):
                o[f"p_{l}"] = pr[:, i]
            o.to_csv(os.path.join(args.out_dir, f"predictions_{name}.csv"), index=False)
            print(f"[{name}] n={res['n']} acc={res['accuracy']:.3f} macroF1={res['macro_f1']:.3f}")
        if args.save_model:
            model.save_pretrained(os.path.join(args.out_dir, "model"))
            tok.save_pretrained(os.path.join(args.out_dir, "model"))


if __name__ == "__main__":
    main()
