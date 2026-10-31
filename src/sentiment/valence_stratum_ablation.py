#!/usr/bin/env python3
"""
Valence stratum ablation, step 2 (GPU): train baseline / A (targeted) / B (random) valence models and score the
fixed validation episodes ONLY. Same code path, hyperparameters and checkpoint rule (best validation macro-F1
epoch) as scaling_retrain.py. The baseline seeds 0-2 already exist in llm_scaling_exp (identical code), so only
the missing baseline seeds (3,4) are trained here; conditions A and B get seeds 0-4. Test and gold are not read.

Usage (GPU):
    python src/sentiment/valence_stratum_ablation.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_pipeline import metrics  # noqa: E402
from scaling_retrain import cap_prev, train_eval  # noqa: E402

OUT, CK = "results/sentiment/llm_scaling_stratum", "models/llm_scaling_stratum"
CLS = ["negative", "neutral", "positive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True)
    sub = pd.read_csv(f"{OUT}/subsets.csv")
    O = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
    va = O[(O.split == "val") & O.valence_maj.notna()].reset_index(drop=True)
    va["prev_text"] = va.prev_text.map(cap_prev)
    va["y"] = va.valence_maj.map({c: i for i, c in enumerate(CLS)})
    va[["turn_id", "episode_id", "dyad", "y"]].to_csv(f"{OUT}/val_labels_valence.csv", index=False)
    base = sub[sub.condition == "base"]
    conds = {"base": base[base.valence_maj.notna()],
             "A_targeted": sub[sub.condition.isin(["base", "A_targeted"]) & sub.valence_maj.notna()],
             "B_random": sub[sub.condition.isin(["base", "B_random"]) & sub.valence_maj.notna()]}
    log = []
    for cond, tr in conds.items():
        tr = tr.reset_index(drop=True)
        tr["y"] = tr.valence_maj.map({c: i for i, c in enumerate(CLS)})
        for seed in a.seeds:
            if cond == "base" and seed < 3:
                continue                                    # seeds 0-2 already produced by scaling_retrain.py (same code)
            t0 = time.time()
            tok, mdl, probs, best_ep = train_eval("cardiff_xlmr_multi", tr, va, False, seed)
            np.save(f"{OUT}/valprobs_{cond}_seed{seed}.npy", probs)
            mdl.save_pretrained(f"{CK}/{cond}_seed{seed}", safe_serialization=True); tok.save_pretrained(f"{CK}/{cond}_seed{seed}")
            m = [metrics(va.y, p.argmax(1)) for p in probs]
            rec = dict(cond=cond, seed=seed, n_train=len(tr), best_epoch=best_ep + 1, val_best=m[best_ep], val_final_epoch=m[-1],
                       per_epoch_macro_f1=[x["macro_f1"] for x in m], seconds=round(time.time() - t0, 1))
            log.append(rec); print(json.dumps(rec), flush=True)
            json.dump(log, open(f"{OUT}/run_log.json", "w"), indent=1)
            del mdl; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
