#!/usr/bin/env python3
"""
Final low-cost check: does a fixed equal-probability seed ensemble beat the selected single checkpoint?
CPU only; uses saved validation probabilities (no retraining, no tuning, no test / gold data).

    valence : A = models/llm_scaling/valence_1783_seed2   B = mean prob of the 1,783-example seeds 0-4
    arousal : A = models/llm_scaling/arousal_5000_seed2   B = mean prob of the 5,000-example seeds 0-2

Pre-specified decision rule (written before running): adopt the ensemble only if
    (1) macro-F1 gain over the selected single >= +0.01,
    (2) the episode-bootstrap 95% CI of that gain has lower bound > 0, and
    (3) it improves at least twice as many episodes as it worsens (ties excluded; per-episode accuracy)
        and does not lower rare-class F1.
Otherwise keep the single checkpoint. Reference row "mean single" = average of the individual seed models.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

OUT = "results/sentiment/llm_ensemble_check"
SC, ST = "results/sentiment/llm_scaling_exp", "results/sentiment/llm_scaling_stratum"
CLS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
RARE = {"valence": [0], "arousal": [0, 2]}
SPEC = {"valence": dict(seeds=[0, 1, 2, 3, 4], single_seed=2, size=1783), "arousal": dict(seeds=[0, 1, 2], single_seed=2, size=5000)}
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
log_sc = {(r["task"], r["size"], r["seed"]): r for r in json.load(open(f"{SC}/run_log.json"))}
log_st = {(r["cond"], r["seed"]): r for r in json.load(open(f"{ST}/run_log.json"))}


def labels(task):
    L = pd.read_csv(f"{SC}/val_labels_{task}.csv")
    if task == "valence":
        L2 = pd.read_csv(f"{ST}/val_labels_valence.csv")
        assert L.equals(L2), "validation label files differ between experiments"
    return L


def probs(task, seed):
    size = SPEC[task]["size"]
    if task == "valence" and seed >= 3:
        r, f, ck = log_st[("base", seed)], f"{ST}/valprobs_base_seed{seed}.npy", f"models/llm_scaling_stratum/base_seed{seed}"
    else:
        r, f, ck = log_sc[(task, size, seed)], f"{SC}/valprobs_{task}_{size}_seed{seed}.npy", f"models/llm_scaling/{task}_{size}_seed{seed}"
    assert os.path.exists(f"{ck}/model.safetensors"), ck
    return np.load(f)[r["best_epoch"] - 1], ck


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)


def rare(y, p, task):
    return float(np.mean(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[RARE[task]]))


def full(y, p, task):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = dict(macro_f1=mf1(y, p), acc=accuracy_score(y, p), weighted_f1=f1_score(y, p, average="weighted"), rare_f1=rare(y, p, task))
    for i, c in enumerate(CLS[task]):
        d.update({f"{c}_P": pr[i], f"{c}_R": rc[i], f"{c}_F1": f[i], f"{c}_n": int(sup[i])})
    return d


report, decisions, verif = [], {}, {}
for task in CLS:
    lab = labels(task)
    y, ep, dy = lab.y.values, lab.episode_id.values, lab.dyad.values
    sd = SPEC[task]
    P = {}
    for s in sd["seeds"]:
        P[s], ck = probs(task, s)
    stack = np.stack([P[s] for s in sd["seeds"]])
    # ---- step 1 verification
    verif[task] = dict(n_val=len(y), classes=CLS[task], n_models=len(P), shapes_ok=all(v.shape == (len(y), 3) for v in P.values()),
                       no_nan=bool(np.isfinite(stack).all()), rows_sum_to_1=bool(np.allclose(stack.sum(2), 1, atol=1e-3)),
                       val_labels_identical_across_experiments=True, missing_val_rows=0, n_episodes=int(len(np.unique(ep))),
                       same_val_order="val_labels file shared by all runs (turn_id order identical, asserted)",
                       preprocessing=("no-context text only, max_len 256" if task == "valence" else "text + previous turn (last 100 words), max_len 256"),
                       training="same code path / hyperparameters (scaling_retrain.train_eval); checkpoint = best-validation epoch",
                       test_or_gold_used=False)
    ens = stack.mean(0)
    single = P[sd["single_seed"]]
    preds = {"single": single.argmax(1), "ensemble": ens.argmax(1)}
    for s in sd["seeds"]:
        preds[f"seed{s}"] = P[s].argmax(1)
    pd.DataFrame({"turn_id": lab.turn_id, "episode_id": ep, "dyad": dy, "y": y, **{f"ens_p{i}": ens[:, i] for i in range(3)},
                  **{f"single_p{i}": single[:, i] for i in range(3)}}).to_csv(f"{OUT}/predictions_{task}.csv", index=False)
    # ---- step 3 metrics
    seed_metrics = pd.DataFrame([full(y, preds[f"seed{s}"], task) for s in sd["seeds"]])
    rows = [dict(task=task, model="selected single", **full(y, preds["single"], task)),
            dict(task=task, model="ensemble", **full(y, preds["ensemble"], task)),
            dict(task=task, model="mean single (avg of seeds)", **seed_metrics.mean().to_dict())]
    report += rows
    for nm in ("single", "ensemble"):
        pd.DataFrame(confusion_matrix(y, preds[nm], labels=[0, 1, 2]), index=[f"true_{c}" for c in CLS[task]],
                     columns=[f"pred_{c}" for c in CLS[task]]).to_csv(f"{OUT}/confusion_{task}_{nm}.csv")
    # bootstrap (episode clusters)
    members = {e: np.where(ep == e)[0] for e in np.unique(ep)}
    keys = ["macro_f1", "acc", "rare_f1"] + [f"{c}_F1" for c in CLS[task]] + [f"{c}_R" for c in CLS[task]]
    diff_s, diff_m = {k: [] for k in keys}, {k: [] for k in keys}
    for _ in range(2000):
        ix = np.concatenate([members[e] for e in rng.choice(list(members), len(members))])
        e_m, s_m = full(y[ix], preds["ensemble"][ix], task), full(y[ix], preds["single"][ix], task)
        m_m = pd.DataFrame([full(y[ix], preds[f"seed{s}"][ix], task) for s in sd["seeds"]]).mean()
        for k in keys:
            diff_s[k].append(e_m[k] - s_m[k]); diff_m[k].append(e_m[k] - m_m[k])
    bt = []
    for ref, dd in (("selected single", diff_s), ("mean single", diff_m)):
        for k in keys:
            a = np.array(dd[k]); bt.append(dict(task=task, ensemble_minus=ref, metric=k, diff=a.mean(), ci_lo=np.percentile(a, 2.5), ci_hi=np.percentile(a, 97.5), p_gt0=(a > 0).mean()))
    BT = pd.DataFrame(bt)
    BT.to_csv(f"{OUT}/bootstrap_{task}.csv", index=False)
    # episode consistency
    pe = []
    for e, ix in members.items():
        r = dict(episode_id=e, n=len(ix), acc_single=np.mean(preds["single"][ix] == y[ix]), acc_ens=np.mean(preds["ensemble"][ix] == y[ix]))
        for ci in RARE[task]:
            m = ix[y[ix] == ci]
            r[f"recall_{CLS[task][ci]}_single"] = np.mean(preds["single"][m] == ci) if len(m) else np.nan
            r[f"recall_{CLS[task][ci]}_ens"] = np.mean(preds["ensemble"][m] == ci) if len(m) else np.nan
        pe.append(r)
    pe = pd.DataFrame(pe); pe.to_csv(f"{OUT}/per_episode_{task}.csv", index=False)
    d = pe.acc_ens - pe.acc_single
    cons = dict(episodes_improved=int((d > 1e-9).sum()), episodes_worse=int((d < -1e-9).sum()), episodes_same=int((d.abs() <= 1e-9).sum()))
    for ci in RARE[task]:
        c = CLS[task][ci]; dd_ = (pe[f"recall_{c}_ens"] - pe[f"recall_{c}_single"]).dropna()
        cons[f"{c}_recall_episodes_better"], cons[f"{c}_recall_episodes_worse"], cons[f"{c}_recall_episodes_same"] = int((dd_ > 1e-9).sum()), int((dd_ < -1e-9).sum()), int((dd_.abs() <= 1e-9).sum())
    loo = []
    for e in members:
        ix = np.concatenate([v for k, v in members.items() if k != e])
        loo.append(mf1(y[ix], preds["ensemble"][ix]) - mf1(y[ix], preds["single"][ix]))
    cons["loo_macro_f1_diff_min"], cons["loo_macro_f1_diff_max"] = float(min(loo)), float(max(loo))
    # dyads
    dyrows = []
    for dname in sorted(set(dy)):
        m = dy == dname
        present = sorted(set(y[m]))
        f = lambda p: f1_score(y[m], p[m], average="macro", labels=present, zero_division=0)
        dyrows.append(dict(task=task, dyad=dname, n=int(m.sum()), class_counts=str(np.bincount(y[m], minlength=3).tolist()),
                           single=f(preds["single"]), ensemble=f(preds["ensemble"]), mean_single=float(np.mean([f(preds[f"seed{s}"]) for s in sd["seeds"]]))))
    pd.DataFrame(dyrows).to_csv(f"{OUT}/by_dyad_{task}.csv", index=False)
    # ---- step 4 rule
    g = BT[(BT.ensemble_minus == "selected single") & (BT.metric == "macro_f1")].iloc[0]
    rg = BT[(BT.ensemble_minus == "selected single") & (BT.metric == "rare_f1")].iloc[0]
    c1, c2 = g["diff"] >= 0.01, g.ci_lo > 0
    c3 = cons["episodes_improved"] >= 2 * max(cons["episodes_worse"], 1e-9) and rg["diff"] >= 0 and cons["episodes_improved"] > 0
    decisions[task] = dict(macro_f1_gain=float(g["diff"]), ci=[float(g.ci_lo), float(g.ci_hi)], rare_f1_gain=float(rg["diff"]),
                           cond1_gain_ge_0p01=bool(c1), cond2_ci_lower_gt0=bool(c2), cond3_consistent=bool(c3), episode_consistency=cons,
                           adopt_ensemble=bool(c1 and c2 and c3),
                           decision="adopt ensemble" if (c1 and c2 and c3) else "keep selected single checkpoint")

R = pd.DataFrame(report); R.to_csv(f"{OUT}/comparison.csv", index=False)
json.dump(verif, open(f"{OUT}/verification.json", "w"), indent=1); json.dump(decisions, open(f"{OUT}/decision.json", "w"), indent=1)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(json.dumps(verif, indent=1)); print(R.round(4).T.to_string()); print(json.dumps(decisions, indent=1, default=str))
for t in CLS:
    print(pd.read_csv(f"{OUT}/bootstrap_{t}.csv").round(4).to_string(index=False)); print(pd.read_csv(f"{OUT}/by_dyad_{t}.csv").round(3).to_string(index=False))
