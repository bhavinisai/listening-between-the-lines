#!/usr/bin/env python3
"""
Valence stratum ablation, step 3 (CPU): validation-only comparison of baseline (1,783) vs A (+targeted) vs B (+random).
Baseline seeds 0-2 come from llm_scaling_exp (identical code), seeds 3-4 from this run. Reference row: the full
5,000-example set (seeds 0-2). Episode-cluster bootstrap on the mean-over-seeds metric; no ensembling.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

OUT, SC = "results/sentiment/llm_scaling_stratum", "results/sentiment/llm_scaling_exp"
SEEDS = [0, 1, 2, 3, 4]
rng = np.random.default_rng(0)
lab = pd.read_csv(f"{OUT}/val_labels_valence.csv")
assert (lab.turn_id == pd.read_csv(f"{SC}/val_labels_valence.csv").turn_id).all()
y, ep = lab.y.values, lab.episode_id.values
log_sc = {(r["size"], r["seed"]): r for r in json.load(open(f"{SC}/run_log.json")) if r["task"] == "valence"}
log_st = {(r["cond"], r["seed"]): r for r in json.load(open(f"{OUT}/run_log.json"))}


def get(cond, seed):
    if cond == "base" and seed < 3:
        r, f = log_sc[(1783, seed)], f"{SC}/valprobs_valence_1783_seed{seed}.npy"
    elif cond == "full5000":
        r, f = log_sc[(5000, seed)], f"{SC}/valprobs_valence_5000_seed{seed}.npy"
    else:
        r, f = log_st[(cond, seed)], f"{OUT}/valprobs_{cond}_seed{seed}.npy"
    pr = np.load(f)
    return pr[r["best_epoch"] - 1].argmax(1), pr[-1].argmax(1), r["best_epoch"]


CONDS = {"base": SEEDS, "A_targeted": SEEDS, "B_random": SEEDS, "full5000": [0, 1, 2]}
P, PF, BE = {}, {}, {}
for c, ss in CONDS.items():
    for s in ss:
        P[(c, s)], PF[(c, s)], BE[(c, s)] = get(c, s)


def stats(p, yy=y):
    pr, rc, f, sup = precision_recall_fscore_support(yy, p, labels=[0, 1, 2], zero_division=0)
    return dict(macro_f1=f1_score(yy, p, average="macro", labels=[0, 1, 2]), acc=accuracy_score(yy, p),
                weighted_f1=f1_score(yy, p, average="weighted"), neg_p=pr[0], neg_r=rc[0], neg_f1=f[0],
                neu_f1=f[1], pos_p=pr[2], pos_r=rc[2], pos_f1=f[2])


rows = []
for c, ss in CONDS.items():
    d = pd.DataFrame([stats(P[(c, s)]) | dict(seed=s, final_epoch_macro_f1=f1_score(y, PF[(c, s)], average="macro")) | dict(best_epoch=BE[(c, s)]) for s in ss])
    r = dict(condition=c, n_seeds=len(ss))
    for m in ["macro_f1", "acc", "weighted_f1", "neg_p", "neg_r", "neg_f1", "neu_f1", "pos_p", "pos_r", "pos_f1", "final_epoch_macro_f1"]:
        r[f"{m}_mean"], r[f"{m}_sd"] = d[m].mean(), d[m].std(ddof=1)
    r["per_seed_macro_f1"] = [round(v, 4) for v in d.macro_f1]; r["per_seed_neg_recall"] = [round(v, 3) for v in d.neg_r]
    r["best_epochs"] = list(d.best_epoch)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/summary.csv", index=False)

# episode bootstrap on mean-over-seeds metrics
members = {e: np.where(ep == e)[0] for e in np.unique(ep)}
def metric_mean(cond, ix, key):
    return np.mean([stats(P[(cond, s)][ix], y[ix])[key] for s in CONDS[cond]])
B, keys = 2000, ["macro_f1", "neg_r", "neg_f1", "neg_p"]
store = {(c, k): [] for c in ("base", "A_targeted", "B_random") for k in keys}
for _ in range(B):
    ix = np.concatenate([members[e] for e in rng.choice(list(members), len(members))])
    for c in ("base", "A_targeted", "B_random"):
        st = [stats(P[(c, s)][ix], y[ix]) for s in SEEDS]
        for k in keys:
            store[(c, k)].append(np.mean([x[k] for x in st]))
brow = []
for a, b in [("A_targeted", "base"), ("B_random", "base"), ("A_targeted", "B_random")]:
    for k in keys:
        dd = np.array(store[(a, k)]) - np.array(store[(b, k)])
        brow.append(dict(comparison=f"{a} - {b}", metric=k, diff=dd.mean(), ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5), p_gt0=(dd > 0).mean()))
BT = pd.DataFrame(brow); BT.to_csv(f"{OUT}/bootstrap_differences.csv", index=False)

# per-episode variability (accuracy and negative recall) and leave-one-episode-out
pe = []
for e, ix in members.items():
    r = dict(episode_id=e, n=len(ix), neg_support=int((y[ix] == 0).sum()))
    for c in ("base", "A_targeted", "B_random"):
        r[f"acc_{c}"] = np.mean([np.mean(P[(c, s)][ix] == y[ix]) for s in SEEDS])
        r[f"negrecall_{c}"] = np.mean([np.mean(P[(c, s)][ix][y[ix] == 0] == 0) for s in SEEDS]) if (y[ix] == 0).any() else np.nan
    pe.append(r)
pe = pd.DataFrame(pe); pe.to_csv(f"{OUT}/per_episode.csv", index=False)
var = {}
for c in ("A_targeted", "B_random"):
    d = pe[f"acc_{c}"] - pe["acc_base"]
    loo = []
    for e in members:
        ix = np.concatenate([v for k, v in members.items() if k != e])
        loo.append(metric_mean(c, ix, "macro_f1") - metric_mean("base", ix, "macro_f1"))
    var[c] = dict(episodes_better=int((d > 1e-9).sum()), episodes_worse=int((d < -1e-9).sum()), episodes_same=int((d.abs() <= 1e-9).sum()),
                  per_episode_acc_delta_sd=float(d.std(ddof=1)), loo_macro_f1_diff_min=float(min(loo)), loo_macro_f1_diff_max=float(max(loo)))
json.dump(var, open(f"{OUT}/episode_variability.json", "w"), indent=1)

# confusion (summed over seeds)
from sklearn.metrics import confusion_matrix
for c in ("base", "A_targeted", "B_random"):
    cm = sum(confusion_matrix(y, P[(c, s)], labels=[0, 1, 2]) for s in SEEDS)
    pd.DataFrame(cm, index=["true_neg", "true_neu", "true_pos"], columns=["pred_neg", "pred_neu", "pred_pos"]).to_csv(f"{OUT}/confusion_{c}_summed_over_seeds.csv")

pd.set_option("display.width", 250, "display.max_columns", 40)
print(T[["condition", "n_seeds", "macro_f1_mean", "macro_f1_sd", "acc_mean", "neg_p_mean", "neg_r_mean", "neg_r_sd", "neg_f1_mean", "neg_f1_sd", "pos_f1_mean", "final_epoch_macro_f1_mean", "per_seed_macro_f1", "per_seed_neg_recall", "best_epochs"]].round(4).to_string(index=False))
print(BT.round(4).to_string(index=False)); print(json.dumps(var, indent=1))
