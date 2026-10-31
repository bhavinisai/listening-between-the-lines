#!/usr/bin/env python3
"""
LLM-label pipeline (final version).

1. Build majority-vote labels (>= 2 of 3 independent LLM passes agree) for valence and
   arousal from the two LLM-labeled draws (results/sentiment/llm_labels{,2}).
2. Split by episode: ~70% train / ~10% validation / ~20% test (no episode in two splits).
3. For each task, train the base model with and without the previous turn as context;
   pick the variant by validation macro-F1; report test macro-F1 / accuracy / kappa
   (vs. the LLM majority labels) and agreement with the human gold sheet.
4. Retrain the chosen variant on train+validation and score every turn in
   sentiment_turns.csv -> results/sentiment/turn_scores_llm2.csv.

Usage (GPU):
    python src/sentiment/llm_pipeline.py --scratch <dir holding the pass files> --out_dir results/sentiment/llm_ft2
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_valence import encode, train  # noqa: E402

CLASSES = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}


def majority(row, cols):
    v = [row[c] for c in cols if isinstance(row[c], str)]
    if len(v) < 2:
        return np.nan
    top = max(set(v), key=v.count)
    return top if v.count(top) >= 2 else np.nan


def load_labels(scratch):
    parts = []
    for meta_csv, folder, n in [("results/sentiment/llm_labels/sample_meta.csv", "llm_labels", 20),
                                ("results/sentiment/llm_labels2/sample_meta.csv", "llm_labels2", 9)]:
        meta = pd.read_csv(meta_csv).set_index("turn_id")
        for p in "ABC":
            lab = pd.concat([pd.read_csv(f"{scratch}/{folder}/{p}_{i:02d}.csv") for i in range(n)])
            lab = lab.drop_duplicates("turn_id").set_index("turn_id")
            meta[f"valence_{p}"], meta[f"arousal_{p}"] = lab.valence, lab.arousal
        parts.append(meta.reset_index().assign(draw=folder))
    d = pd.concat(parts, ignore_index=True)
    for t in CLASSES:
        d[f"{t}_maj"] = d.apply(lambda r: majority(r, [f"{t}_{p}" for p in "ABC"]), axis=1)
        d[f"{t}_all3"] = d[[f"{t}_{p}" for p in "ABC"]].nunique(axis=1).eq(1)
    return d


def split(d, seed=0):
    eps = d.groupby("episode_id").dyad.first().reset_index()
    g = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    rest, test = next(g.split(eps, groups=eps.episode_id))
    rest_eps = eps.iloc[rest]
    tr, va = next(GroupShuffleSplit(1, test_size=0.125, random_state=seed).split(rest_eps, groups=rest_eps.episode_id))
    m = {e: "train" for e in rest_eps.iloc[tr].episode_id}
    m.update({e: "val" for e in rest_eps.iloc[va].episode_id})
    m.update({e: "test" for e in eps.iloc[test].episode_id})
    return d.episode_id.map(m)


@torch.no_grad()
def predict_sorted(tok, mdl, df, ctx, max_len=256, bs=64, dev="cuda"):
    order = np.argsort(df.text.str.len().values)
    out = np.zeros((len(df), 3), dtype=np.float32)
    for i in range(0, len(df), bs):
        idx = order[i:i + bs]
        enc = {k: v.to(dev) for k, v in encode(tok, df.iloc[idx], ctx, max_len).items()}
        with torch.autocast(dev, dtype=torch.bfloat16):
            out[idx] = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
    return out


def metrics(y, p):
    return dict(n=int(len(y)), acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")),
                kappa=float(cohen_kappa_score(y, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out_dir", default="results/sentiment/llm_ft2")
    ap.add_argument("--base", default="cardiff_xlmr_multi")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--turns", default="results/sentiment/sentiment_turns.csv")
    ap.add_argument("--old_scores", default="results/sentiment/turn_scores.csv")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_labels(a.scratch)
    d["split"] = split(d)
    d.to_csv(f"{a.out_dir}/labels_majority_split.csv", index=False)
    for t in CLASSES:
        print(t, "majority rows", d[f"{t}_maj"].notna().sum(), "all-3-agree", int(d[f"{t}_all3"].sum()),
              d[f"{t}_maj"].value_counts().to_dict(), flush=True)
    print(d.groupby("split").agg(turns=("turn_id", "size"), eps=("episode_id", "nunique")), flush=True)
    print(pd.crosstab(d.split, d.dyad), flush=True)

    gold = pd.read_csv("results/sentiment/gold_sheet_labeled.csv")
    turns = pd.read_csv(a.turns)
    prev = turns[["episode_id", "turn_idx", "text"]].rename(columns={"turn_idx": "prev_turn_idx", "text": "prev_text"})
    turns = turns.merge(prev, on=["episode_id", "prev_turn_idx"], how="left")
    scores = turns[["episode_id", "turn_idx", "speaker", "role", "gender", "start", "end", "word_count", "dyad", "text"]].copy()
    results, final = {}, {}
    for task, cls in CLASSES.items():
        idx = {c: i for i, c in enumerate(cls)}
        lab = d[d[f"{task}_maj"].notna()].copy()
        lab["y"] = lab[f"{task}_maj"].map(idx)
        parts = {s: lab[lab.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
        g = gold[gold[task].isin(cls)].reset_index(drop=True)
        g["y"] = g[task].map(idx)
        res = {}
        for ctx in (False, True):
            name = "context" if ctx else "no_context"
            args = SimpleNamespace(lr=a.lr, epochs=a.epochs, bs=16, max_len=256, weight_pow=0.5, context=ctx)
            tok, mdl = train(a.base, parts["train"], args, 0)
            r = {}
            for s in ("val", "test"):
                p = predict_sorted(tok, mdl, parts[s], ctx)
                r[s] = metrics(parts[s].y, p.argmax(1))
                if s == "test":
                    t3 = parts[s][parts[s][f"{task}_all3"]]
                    r["test_all3"] = metrics(t3.y, p[parts[s][f"{task}_all3"].values].argmax(1))
                    parts[s].assign(**{f"p{i}": p[:, i] for i in range(3)}).to_csv(
                        f"{a.out_dir}/test_preds_{task}_{name}.csv", index=False)
            gp = predict_sorted(tok, mdl, g, ctx)
            r["human_gold"] = metrics(g.y, gp.argmax(1))
            g.assign(**{f"p{i}": gp[:, i] for i in range(3)}).to_csv(f"{a.out_dir}/gold_preds_{task}_{name}.csv", index=False)
            res[name] = r
            print(task, name, json.dumps(r), flush=True)
            del mdl; torch.cuda.empty_cache()
        best = max(res, key=lambda k: res[k]["val"]["macro_f1"])
        results[task] = dict(variants=res, chosen=best, n_train=len(parts["train"]), n_val=len(parts["val"]),
                             n_test=len(parts["test"]))
        print(task, "chosen", best, flush=True)

        # final model on train + val, then score the corpus
        ctx = best == "context"
        args = SimpleNamespace(lr=a.lr, epochs=a.epochs, bs=16, max_len=256, weight_pow=0.5, context=ctx)
        tok, mdl = train(a.base, pd.concat([parts["train"], parts["val"]], ignore_index=True), args, 0)
        P = predict_sorted(tok, mdl, turns, ctx)
        final[task] = P
        mdl.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        mdl.save_pretrained(f"models/llm2_{task}", safe_serialization=True); tok.save_pretrained(f"models/llm2_{task}")
        del mdl; torch.cuda.empty_cache()
        print(task, "corpus scored", flush=True)

    pv, pa = final["valence"], final["arousal"]
    scores[["ft_neg", "ft_neu", "ft_pos"]] = pv
    scores["valence"] = pv[:, 2] - pv[:, 0]
    scores["valence_label"] = [CLASSES["valence"][i] for i in pv.argmax(1)]
    scores[["ar_low", "ar_med", "ar_high"]] = pa
    scores["arousal"] = 0.5 * pa[:, 1] + pa[:, 2]
    old = pd.read_csv(a.old_scores)
    keep = ["episode_id", "turn_idx", "zs_valence"] + [c for c in old.columns if c.startswith("emo_")]
    scores = scores.merge(old[keep].assign(arousal_emo=old.arousal), on=["episode_id", "turn_idx"], how="left")
    scores.to_csv("results/sentiment/turn_scores_llm2.csv", index=False)
    json.dump(results, open(f"{a.out_dir}/metrics.json", "w"), indent=1)
    print("wrote turn_scores_llm2.csv", flush=True)


if __name__ == "__main__":
    main()
