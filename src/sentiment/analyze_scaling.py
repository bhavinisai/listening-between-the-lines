#!/usr/bin/env python3
"""
Validation-only analysis of the data-scaling experiment (scaling_retrain.py output).

Per task and training size (mean / SD across seeds): macro-F1, accuracy, weighted-F1, kappa, per-class
P/R/F1/support, rare-class F1, confusion matrices, fixed-final-epoch macro-F1; episode-cluster bootstrap CIs
for size differences (mean-over-seeds metric, no prediction ensembling); leave-one-episode-out stability;
per-episode consistency; per-dyad results; seed sensitivity; learning-curve figure.

Usage:
    python src/sentiment/analyze_scaling.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

D = "results/sentiment/llm_scaling_exp"
CLASSES = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
RARE = {"valence": [0], "arousal": [0, 2]}                    # negative; low + high
SIZES, SEEDS = [1783, 3000, 5000], [0, 1, 2]
rng = np.random.default_rng(0)
log = {(r["task"], r["size"], r["seed"]): r for r in json.load(open(f"{D}/run_log.json"))}


def mf1(y, p, labels=(0, 1, 2)):
    return f1_score(y, p, average="macro", labels=list(labels), zero_division=0)


def rare_f1(y, p, task):
    f = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    return float(np.mean(f[RARE[task]]))


def load(task):
    lab = pd.read_csv(f"{D}/val_labels_{task}.csv")
    P = {}
    for size in SIZES:
        for seed in SEEDS:
            probs = np.load(f"{D}/valprobs_{task}_{size}_seed{seed}.npy")
            P[(size, seed)] = probs[log[(task, size, seed)]["best_epoch"] - 1]
    return lab, P


def table(task, lab, P):
    cls, y = CLASSES[task], lab.y.values
    rows, perclass, cms = [], [], {}
    for size in SIZES:
        per = []
        for seed in SEEDS:
            p = P[(size, seed)].argmax(1)
            pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
            per.append(dict(seed=seed, macro_f1=mf1(y, p), acc=accuracy_score(y, p), weighted_f1=f1_score(y, p, average="weighted"),
                            kappa=cohen_kappa_score(y, p), rare_f1=rare_f1(y, p, task), best_epoch=log[(task, size, seed)]["best_epoch"],
                            final_epoch_macro_f1=log[(task, size, seed)]["val_final_epoch"]["macro_f1"], pr=pr, rc=rc, f=f, sup=sup))
            cms.setdefault(size, np.zeros((3, 3), int))
            cms[size] += confusion_matrix(y, p, labels=[0, 1, 2])
        df = pd.DataFrame(per)
        r = dict(task=task, size=size, n_train=log[(task, size, 0)]["n_train"])
        for m in ["macro_f1", "acc", "weighted_f1", "kappa", "rare_f1", "final_epoch_macro_f1"]:
            r[f"{m}_mean"], r[f"{m}_sd"] = df[m].mean(), df[m].std(ddof=1)
        r["per_seed_macro_f1"] = [round(v, 4) for v in df.macro_f1]
        r["best_epochs"] = list(df.best_epoch)
        rows.append(r)
        for i, c in enumerate(cls):
            perclass.append(dict(task=task, size=size, cls=c, support=int(df.sup.iloc[0][i]),
                                 precision_mean=np.mean([x[i] for x in df.pr]), precision_sd=np.std([x[i] for x in df.pr], ddof=1),
                                 recall_mean=np.mean([x[i] for x in df.rc]), recall_sd=np.std([x[i] for x in df.rc], ddof=1),
                                 f1_mean=np.mean([x[i] for x in df.f]), f1_sd=np.std([x[i] for x in df.f], ddof=1)))
        pd.DataFrame(cms[size], index=[f"true_{c}" for c in cls], columns=[f"pred_{c}" for c in cls]).to_csv(
            f"{D}/confusion_{task}_{size}_summed_over_seeds.csv")
    return pd.DataFrame(rows), pd.DataFrame(perclass)


def boot(task, lab, P, B=2000):
    """Episode-cluster bootstrap: metric = mean over seeds, difference between sizes."""
    y, ep = lab.y.values, lab.episode_id.values
    eps = np.unique(ep)
    members = {e: np.where(ep == e)[0] for e in eps}
    preds = {k: v.argmax(1) for k, v in P.items()}
    out = {(s, m): [] for s in SIZES for m in ("macro_f1", "rare_f1")}
    for _ in range(B):
        ix = np.concatenate([members[e] for e in rng.choice(eps, len(eps))])
        for s in SIZES:
            out[(s, "macro_f1")].append(np.mean([mf1(y[ix], preds[(s, sd)][ix]) for sd in SEEDS]))
            out[(s, "rare_f1")].append(np.mean([rare_f1(y[ix], preds[(s, sd)][ix], task) for sd in SEEDS]))
    rows = []
    for a, b in [(1783, 3000), (1783, 5000), (3000, 5000)]:
        for m in ("macro_f1", "rare_f1"):
            d = np.array(out[(b, m)]) - np.array(out[(a, m)])
            rows.append(dict(task=task, metric=m, comparison=f"{b} - {a}", diff_mean=d.mean(), ci_lo=np.percentile(d, 2.5),
                             ci_hi=np.percentile(d, 97.5), p_gt0=float((d > 0).mean())))
    return pd.DataFrame(rows)


def stability(task, lab, P):
    """Leave-one-episode-out change in the (5000 - 1783) mean macro-F1 difference; per-episode accuracy changes."""
    y, ep = lab.y.values, lab.episode_id.values
    preds = {k: v.argmax(1) for k, v in P.items()}
    def d(mask):
        return np.mean([mf1(y[mask], preds[(5000, s)][mask]) for s in SEEDS]) - np.mean([mf1(y[mask], preds[(1783, s)][mask]) for s in SEEDS])
    full = d(np.ones(len(y), bool))
    loo = [d(ep != e) for e in np.unique(ep)]
    per_ep = []
    for e in np.unique(ep):
        m = ep == e
        a5 = np.mean([np.mean(preds[(5000, s)][m] == y[m]) for s in SEEDS]); a1 = np.mean([np.mean(preds[(1783, s)][m] == y[m]) for s in SEEDS])
        per_ep.append(dict(episode_id=e, n=int(m.sum()), acc_1783=a1, acc_5000=a5, delta=a5 - a1))
    pe = pd.DataFrame(per_ep)
    pe.to_csv(f"{D}/per_episode_accuracy_{task}.csv", index=False)
    return dict(task=task, full_diff_5000_vs_1783=full, loo_min=min(loo), loo_max=max(loo),
                episodes_improved=int((pe.delta > 1e-9).sum()), episodes_worse=int((pe.delta < -1e-9).sum()),
                episodes_same=int((pe.delta.abs() <= 1e-9).sum()), n_episodes=len(pe))


def by_dyad(task, lab, P):
    y, dy = lab.y.values, lab.dyad.values
    rows = []
    for size in SIZES:
        for d in sorted(set(dy)):
            m = dy == d
            f = [mf1(y[m], P[(size, s)].argmax(1)[m], labels=sorted(set(y[m]))) for s in SEEDS]
            a = [accuracy_score(y[m], P[(size, s)].argmax(1)[m]) for s in SEEDS]
            rows.append(dict(task=task, size=size, dyad=d, n=int(m.sum()), classes_present=str(np.bincount(y[m], minlength=3).tolist()),
                             macro_f1_mean=np.mean(f), macro_f1_sd=np.std(f, ddof=1), acc_mean=np.mean(a)))
    return pd.DataFrame(rows)


def figure(T):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(axes, [("macro_f1", "Validation macro-F1"), ("rare_f1", "Rare-class F1")]):
        for task, col in (("valence", "#2a78d6"), ("arousal", "#eb6834")):
            t = T[T.task == task].sort_values("size")
            ax.errorbar(t["size"], t[f"{key}_mean"], yerr=t[f"{key}_sd"], marker="o", capsize=4, color=col, label=task)
        ax.set_xlabel("training examples"); ax.set_title(title); ax.set_xticks(SIZES); ax.grid(alpha=.3)
    axes[0].set_ylabel("mean ± SD over 3 seeds"); axes[0].legend()
    fig.suptitle("Learning curve (validation episodes only; rare = negative | mean(low,high))", fontsize=10)
    fig.savefig(f"{D}/learning_curve.png", dpi=200, bbox_inches="tight")


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40)
    Ts, PCs, Bs, Ss, DYs = [], [], [], [], []
    for task in CLASSES:
        lab, P = load(task)
        t, pc = table(task, lab, P)
        Ts.append(t); PCs.append(pc); Bs.append(boot(task, lab, P)); Ss.append(stability(task, lab, P)); DYs.append(by_dyad(task, lab, P))
    T, PC, B, DY = pd.concat(Ts), pd.concat(PCs), pd.concat(Bs), pd.concat(DYs)
    T.to_csv(f"{D}/summary_by_size.csv", index=False); PC.to_csv(f"{D}/per_class_by_size.csv", index=False)
    B.to_csv(f"{D}/bootstrap_size_differences.csv", index=False); DY.to_csv(f"{D}/by_dyad.csv", index=False)
    json.dump(Ss, open(f"{D}/stability.json", "w"), indent=1)
    figure(T)
    # development-selected model: highest mean validation macro-F1 size, then its best seed
    sel = {}
    for task in CLASSES:
        t = T[T.task == task]
        s = int(t.loc[t.macro_f1_mean.idxmax(), "size"])
        seeds = {sd: log[(task, s, sd)]["val_best"]["macro_f1"] for sd in SEEDS}
        sd = max(seeds, key=seeds.get)
        sel[task] = dict(size=s, seed=sd, checkpoint=f"models/llm_scaling/{task}_{s}_seed{sd}", val_macro_f1=seeds[sd],
                         mean_val_macro_f1_for_size=float(t.loc[t["size"] == s, "macro_f1_mean"].iloc[0]))
    json.dump(sel, open(f"{D}/development_selected.json", "w"), indent=1)
    print(T[["task", "size", "n_train", "macro_f1_mean", "macro_f1_sd", "acc_mean", "weighted_f1_mean", "kappa_mean", "rare_f1_mean", "rare_f1_sd",
             "final_epoch_macro_f1_mean", "per_seed_macro_f1", "best_epochs"]].round(4).to_string(index=False))
    print(PC.round(3).to_string(index=False)); print(B.round(4).to_string(index=False)); print(json.dumps(Ss, indent=1)); print(DY.round(3).to_string(index=False)); print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
