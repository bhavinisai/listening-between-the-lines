#!/usr/bin/env python3
"""
Controlled data-scaling experiment: retrain the valence and arousal models on nested LLM-label
training sets (1,783 ⊂ 3,000 ⊂ 5,000) and evaluate on the fixed validation episodes ONLY.
Held-out test and human-gold files are never read for evaluation.

Recipe (identical to llm_pipeline.py / finetune_valence.train): cardiff_xlmr_multi, lr 2e-5, 4 epochs,
batch 16, max_len 256, class weights (n/(3*count))**0.5, AdamW wd 0.01, 10% warmup, bf16 autocast.
Variant per task is the one llm_pipeline chose on validation: valence = no previous-turn context,
arousal = context. No dyad weights, thresholds, logit biases or ensembling.

Checkpoint selection (same rule for every run): after each epoch score validation, keep the epoch with the
best validation macro-F1 (ties -> earlier epoch). Per-epoch validation probabilities are saved so the
recipe's fixed final epoch can be reported as well.

previous-turn text is capped to its last 100 words for ALL rows (train and validation), because the new
labeling set capped it that way; this keeps the context input identical across sizes.

Usage (GPU):
    python src/sentiment/scaling_retrain.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_valence import encode  # noqa: E402
from llm_pipeline import metrics, predict_sorted  # noqa: E402

BASE = "results/sentiment/llm_labels3"
CLASSES = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
VARIANT_CONTEXT = {"valence": False, "arousal": True}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cap_prev(s, n=100):
    return " ".join(str(s).split()[-n:]) if isinstance(s, str) else ""


def load_data(out_dir):
    S = pd.read_csv(f"{BASE}/scaling_train_sets.csv")
    O = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
    G = pd.read_csv("results/sentiment/gold_sheet_labeled.csv", usecols=["episode_id", "text"])
    V = O[O.split == "val"].copy()
    S["prev_text"], V["prev_text"] = S.prev_text.map(cap_prev), V.prev_text.map(cap_prev)

    # ---- verification (test/gold only used here for episode / text exclusion checks)
    ep_tr, ep_va = set(S.episode_id), set(V.episode_id)
    ep_te, ep_gold = set(O[O.split == "test"].episode_id), set(G.episode_id)
    chk = dict(train_episodes=len(ep_tr), val_episodes=len(ep_va),
               train_val_overlap=len(ep_tr & ep_va), train_test_overlap=len(ep_tr & ep_te), train_gold_overlap=len(ep_tr & ep_gold),
               train_text_in_val=int(S.text.isin(V.text).sum()), train_text_in_gold=int(S.text.isin(G.text).sum()),
               nested_1783_in_3000=bool((S.in_1783 <= S.in_3000).all()), nested_3000_in_5000=bool((S.in_3000 <= S.in_5000).all()),
               duplicate_turn_ids=int(S.turn_id.duplicated().sum()))
    assert chk["train_val_overlap"] == chk["train_test_overlap"] == chk["train_gold_overlap"] == 0, chk
    assert chk["nested_1783_in_3000"] and chk["nested_3000_in_5000"]
    sets = {}
    for size in (1783, 3000, 5000):
        x = S[S[f"in_{size}"]]
        sets[size] = dict(rows=len(x), valence_rows=int(x.valence_maj.notna().sum()), arousal_rows=int(x.arousal_maj.notna().sum()),
                          valence=x.valence_maj.value_counts().to_dict(), arousal=x.arousal_maj.value_counts().to_dict(),
                          dyad=x.dyad.value_counts().to_dict(), dyad_share=x.dyad.value_counts(normalize=True).round(3).to_dict(),
                          prev_text_words_p99=float(x.prev_text.str.split().str.len().quantile(.99)))
    chk["sets"] = sets
    chk["val"] = dict(rows=len(V), valence=V.valence_maj.value_counts().to_dict(), arousal=V.arousal_maj.value_counts().to_dict(),
                      dyad=V.dyad.value_counts().to_dict())
    json.dump(chk, open(f"{out_dir}/data_verification.json", "w"), indent=1, default=str)
    return S, V


def train_eval(base, tr, va, ctx, seed, epochs=4, lr=2e-5, bs=16, max_len=256, weight_pow=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    path = f"models/hf_converted/{base}"
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).to(DEV)
    y = torch.tensor(tr.y.values).to(DEV)
    enc = {k: v.to(DEV) for k, v in encode(tok, tr, ctx, max_len).items()}
    cnt = np.bincount(tr.y.values, minlength=3)
    w = torch.tensor((len(tr) / (3 * np.maximum(cnt, 1))) ** weight_pow, dtype=torch.float).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    probs, best, best_state = [], (-1, 0), None
    for ep in range(epochs):
        mdl.train()
        perm = torch.randperm(len(tr)).to(DEV)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            with torch.autocast(DEV, dtype=torch.bfloat16):
                logits = mdl(**{k: v[idx] for k, v in enc.items()}).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y[idx], weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        mdl.eval()
        p = predict_sorted(tok, mdl, va, ctx, max_len=max_len, dev=DEV)
        probs.append(p)
        f1 = f1_score(va.y, p.argmax(1), average="macro", labels=[0, 1, 2])
        if f1 > best[0] + 1e-12:
            best = (f1, ep)
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_state)
    return tok, mdl, np.stack(probs), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/sentiment/llm_scaling_exp")
    ap.add_argument("--ckpt_dir", default="models/llm_scaling")
    ap.add_argument("--base", default="cardiff_xlmr_multi")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1783, 3000, 5000])
    ap.add_argument("--no_ckpt", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.ckpt_dir, exist_ok=True)
    S, V = load_data(a.out_dir)
    log = []
    for task, cls in CLASSES.items():
        idx = {c: i for i, c in enumerate(cls)}
        va = V[V[f"{task}_maj"].notna()].reset_index(drop=True)
        va["y"] = va[f"{task}_maj"].map(idx)
        va[["turn_id", "episode_id", "dyad", "y"]].to_csv(f"{a.out_dir}/val_labels_{task}.csv", index=False)
        for size in a.sizes:
            tr = S[S[f"in_{size}"] & S[f"{task}_maj"].notna()].reset_index(drop=True)
            tr["y"] = tr[f"{task}_maj"].map(idx)
            for seed in a.seeds:
                t0 = time.time()
                tok, mdl, probs, best_ep = train_eval(a.base, tr, va, VARIANT_CONTEXT[task], seed)
                name = f"{task}_{size}_seed{seed}"
                np.save(f"{a.out_dir}/valprobs_{name}.npy", probs)                        # [epoch, n, 3]
                ep_metrics = [metrics(va.y, p.argmax(1)) for p in probs]
                if not a.no_ckpt:
                    mdl.save_pretrained(f"{a.ckpt_dir}/{name}", safe_serialization=True); tok.save_pretrained(f"{a.ckpt_dir}/{name}")
                rec = dict(task=task, size=size, seed=seed, n_train=len(tr), best_epoch=best_ep + 1, val_best=ep_metrics[best_ep],
                           val_final_epoch=ep_metrics[-1], per_epoch_macro_f1=[m["macro_f1"] for m in ep_metrics],
                           seconds=round(time.time() - t0, 1))
                log.append(rec)
                print(json.dumps(rec), flush=True)
                json.dump(log, open(f"{a.out_dir}/run_log.json", "w"), indent=1)
                del mdl; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
