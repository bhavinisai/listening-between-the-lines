#!/usr/bin/env python3
"""
Read-only audit of the LLM-label valence / arousal models (llm_ft2). Recomputes every metric
from the saved prediction files and checks splits and leakage. Writes only to
results/sentiment/audit/; never modifies llm_ft2.

    full metrics (acc, macro/weighted F1, per-class P/R/F1/support, kappa), confusion matrices,
    majority + prior baselines, episode-bootstrap CIs, split / duplicate / gold-overlap checks,
    class distributions, LLM-vs-human disagreement.

Usage:
    python src/sentiment/audit_eval.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

R, OUT = "results/sentiment/llm_ft2", "results/sentiment/audit"
CLS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
os.makedirs(OUT, exist_ok=True)
M = json.load(open(f"{R}/metrics.json"))
rng = np.random.default_rng(0)


def full(y, p, cls):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    row = dict(n=len(y), accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro", labels=[0, 1, 2]),
               weighted_f1=f1_score(y, p, average="weighted"), kappa=cohen_kappa_score(y, p))
    for i, c in enumerate(cls):
        row.update({f"{c}_P": pr[i], f"{c}_R": rc[i], f"{c}_F1": f[i], f"{c}_n": int(sup[i])})
    return row


def boot_ci(df, B=2000):
    """Episode-cluster bootstrap 95% CI for macro-F1 and accuracy."""
    g = {e: x for e, x in df.groupby("episode_id")}
    eps = np.array(list(g))
    f1s, accs = [], []
    for _ in range(B):
        s = pd.concat([g[e] for e in rng.choice(eps, len(eps))])
        p = s[["p0", "p1", "p2"]].values.argmax(1)
        f1s.append(f1_score(s.y, p, average="macro", labels=[0, 1, 2])); accs.append(accuracy_score(s.y, p))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])


rows, cms = [], {}
for task, cls in CLS.items():
    ch = M[task]["chosen"]
    for split in ("test", "gold"):
        d = pd.read_csv(f"{R}/{split}_preds_{task}_{ch}.csv")
        p = d[["p0", "p1", "p2"]].values.argmax(1)
        r = dict(task=task, variant=ch, eval_set=split, **full(d.y, p, cls))
        (f_lo, f_hi), (a_lo, a_hi) = boot_ci(d)
        r.update(macro_f1_lo=f_lo, macro_f1_hi=f_hi, acc_lo=a_lo, acc_hi=a_hi, n_episodes=d.episode_id.nunique())
        # baselines
        maj = np.bincount(d.y, minlength=3).argmax()
        r.update(base_majority_acc=float(np.mean(d.y == maj)),
                 base_majority_macro_f1=f1_score(d.y, np.full(len(d), maj), average="macro", labels=[0, 1, 2]),
                 base_majority_class=cls[maj])
        rows.append(r)
        cm = confusion_matrix(d.y, p, labels=[0, 1, 2])
        pd.DataFrame(cm, index=[f"true_{c}" for c in cls], columns=[f"pred_{c}" for c in cls]).to_csv(
            f"{OUT}/confusion_{task}_{split}.csv")
pd.DataFrame(rows).to_csv(f"{OUT}/metrics_full.csv", index=False)

# --- splits, duplicates, gold overlap
d = pd.read_csv(f"{R}/labels_majority_split.csv")
gold = pd.read_csv("results/sentiment/gold_sheet_labeled.csv")
S = {s: set(d[d.split == s].episode_id) for s in ("train", "val", "test")}
chk = dict(episodes=d.episode_id.nunique(), turns=len(d),
           train_test_overlap=len(S["train"] & S["test"]), train_val_overlap=len(S["train"] & S["val"]),
           val_test_overlap=len(S["val"] & S["test"]), gold_llm_episode_overlap=len(set(gold.episode_id) & set(d.episode_id)),
           duplicate_turn_ids=int(d.turn_id.duplicated().sum()),
           texts_in_multiple_splits=int((d.groupby("text").split.nunique() > 1).sum()),
           gold_texts_in_llm_set=int(gold.text.isin(d.text).sum()))
for s in S:
    chk[f"{s}_episodes"] = len(S[s]); chk[f"{s}_turns"] = int((d.split == s).sum())
json.dump(chk, open(f"{OUT}/leakage_checks.json", "w"), indent=1)

# --- class distributions
dist = []
for task, cls in CLS.items():
    for s in ("train", "val", "test"):
        x = d[d.split == s][f"{task}_maj"].value_counts()
        dist.append(dict(task=task, set=s, **{c: int(x.get(c, 0)) for c in cls}))
    g = gold[task].value_counts()
    dist.append(dict(task=task, set="human_gold", **{c: int(g.get(c, 0)) for c in cls}))
pd.DataFrame(dist).to_csv(f"{OUT}/class_distributions.csv", index=False)

# --- LLM vs human disagreement on the gold turns
h = pd.read_csv("results/sentiment/llm_vs_human_gold.csv")
dis = []
for task in CLS:
    ct = pd.crosstab(h[f"{task}_h"], h[f"{task}_l"])
    ct.to_csv(f"{OUT}/llm_vs_human_{task}_crosstab.csv")
    dis.append(dict(task=task, n=len(h), agree=float((h[f"{task}_h"] == h[f"{task}_l"]).mean()),
                    kappa=cohen_kappa_score(h[f"{task}_h"], h[f"{task}_l"])))
pd.DataFrame(dis).to_csv(f"{OUT}/llm_vs_human_agreement.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 40)
print(pd.DataFrame(rows).T.to_string()); print(json.dumps(chk)); print(pd.DataFrame(dist)); print(pd.DataFrame(dis))
