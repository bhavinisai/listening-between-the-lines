#!/usr/bin/env python3
"""
Fine-tune a pretrained 3-class sentiment model (default: cardiff_xlmr) on the
hand-labeled gold turns, evaluated with episode-grouped 5-fold CV (no episode
in both train and test). Optionally trains a final model on all gold rows and
saves it for scoring the full corpus.

Usage:
    python src/sentiment/finetune_valence.py --base cardiff_xlmr --cv_out results/sentiment/valence_ft_cv_cardiff_xlmr.csv
    python src/sentiment/finetune_valence.py --base cardiff_xlmr --final_dir models/valence_ft_cardiff_xlmr
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAB = ["negative", "neutral", "positive"]  # matches cardiff head order (0 neg, 1 neu, 2 pos)


def encode(tok, df, ctx, max_len):
    if ctx:
        return tok(df.text.astype(str).tolist(), df.prev_text.fillna("").astype(str).tolist(),
                   truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt")
    return tok(df.text.astype(str).tolist(), truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def train(base, tr, args, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(f"models/hf_converted/{base}")
    mdl = AutoModelForSequenceClassification.from_pretrained(f"models/hf_converted/{base}").to(DEV)
    y = torch.tensor(tr.y.values).to(DEV)
    enc = {k: v.to(DEV) for k, v in encode(tok, tr, args.context, args.max_len).items()}
    cnt = np.bincount(tr.y.values, minlength=3)
    w = torch.tensor((len(tr) / (3 * np.maximum(cnt, 1))) ** args.weight_pow, dtype=torch.float).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(tr) / args.bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    mdl.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(tr)).to(DEV)
        for i in range(0, len(tr), args.bs):
            idx = perm[i:i + args.bs]
            with torch.autocast(DEV, dtype=torch.bfloat16):
                logits = mdl(**{k: v[idx] for k, v in enc.items()}).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y[idx], weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    return tok, mdl.eval()


@torch.no_grad()
def predict(tok, mdl, df, args):
    out = []
    for i in range(0, len(df), 64):
        enc = {k: v.to(DEV) for k, v in encode(tok, df.iloc[i:i + 64], args.context, args.max_len).items()}
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out.append(torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="results/sentiment/gold_sheet_labeled.csv")
    ap.add_argument("--base", default="cardiff_xlmr")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_pow", type=float, default=1.0, help="class-weight strength (0 = unweighted)")
    ap.add_argument("--context", action="store_true", help="feed the previous turn as a second segment")
    ap.add_argument("--limit", type=int, help="debug: use only the first N gold rows")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cv_out")
    ap.add_argument("--final_dir")
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    df = df[df.valence.isin(LAB)].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    df["y"] = df.valence.map({l: i for i, l in enumerate(LAB)})

    if args.cv_out:
        P = np.zeros((len(df), 3))
        for s in range(args.seeds):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(df, df.y, df.episode_id):
                tok, mdl = train(args.base, df.iloc[tr], args, s)
                P[te] += predict(tok, mdl, df.iloc[te], args)
                del mdl; torch.cuda.empty_cache()
            pred = P.argmax(1)
            print(f"seed-ensemble {s + 1}: acc={accuracy_score(df.y, pred):.3f} macroF1={f1_score(df.y, pred, average='macro'):.3f}", flush=True)
        out = df[["gold_id", "episode_id", "valence"]].copy()
        out[["p_neg", "p_neu", "p_pos"]] = P / args.seeds
        out.to_csv(args.cv_out, index=False)

    if args.final_dir:
        tok, mdl = train(args.base, df, args, 0)
        mdl.save_pretrained(args.final_dir, safe_serialization=True); tok.save_pretrained(args.final_dir)
        print("saved", args.final_dir)


if __name__ == "__main__":
    main()
