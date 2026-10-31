#!/usr/bin/env python3
"""
v2: validate + merge pass-1 LLM labels, then build the pass-2 (independent
relabel) chunks.

Pass 2 covers:
    * all sealed-test questions (so test labels are double-labeled)
    * the 120 human gold questions, as hidden rows (measures LLM-labeler vs
      human agreement under the v2 rules; gold is never used for training)
    * dev rows with confidence 1, or whose label/alt involves challenge or
      leading (the hardest boundary)
    * a random 300 of the remaining dev rows (estimates consistency on
      "confident" items)
Labelers see only qid/text/context, and the chunks are shuffled.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

D = "results/question_type_v2"
LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]

man = pd.read_csv(f"{D}/labeling_manifest.csv")
p1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/label_chunks/pass1_labels_*.csv"))])
assert p1.qid.is_unique and set(p1.qid) == set(man.qid), "pass-1 qid mismatch"
assert p1.label.isin(LABELS).all() and p1.confidence.isin([1, 2, 3]).all()
p1 = p1.rename(columns={"label": "label1", "confidence": "conf1", "alt": "alt1"})
m = man.merge(p1, on="qid", validate="one_to_one")
m.to_csv(f"{D}/pass1_merged.csv", index=False)

print("pass-1 label distribution by split:")
print(pd.crosstab(m.split, m.label1))
print("\nconfidence:", m.conf1.value_counts(normalize=True).round(3).sort_index().to_dict())
print("\nlabel1 vs v1 prediction (dev+test, v1 = old model, for information only):")
print(pd.crosstab(m.v1_pred, m.label1))
print("\nagreement with v1 model:", (m.v1_pred == m.label1).mean().round(3))

# ---- pass 2
rng = np.random.default_rng(42)
test = m[m.split == "test"]
dev = m[m.split == "dev"]
hard = dev[(dev.conf1 == 1) | dev.label1.isin(["challenge", "leading"]) | dev.alt1.isin(["challenge", "leading"])]
rest = dev[~dev.qid.isin(hard.qid)]
rand = rest.sample(n=min(300, len(rest)), random_state=42)
sel = pd.concat([test, hard, rand])[["qid", "text", "prev_sentence", "prev_turn_tail"]]

# gold rows with context from the pool where possible
gold = pd.read_csv("gold_work/gold_full.csv")
pool = pd.read_csv(f"{D}/pool_host_questions.csv", usecols=["episode_id", "text", "prev_sentence", "prev_turn_tail"]
                   ).drop_duplicates(["episode_id", "text"])
g = gold.merge(pool, on=["episode_id", "text"], how="left")
g["qid"] = g.gold_id
g = g[["qid", "text", "prev_sentence", "prev_turn_tail"]]
print(f"\ngold rows with context: {g.prev_turn_tail.notna().sum()}/{len(g)}")
sel = pd.concat([sel, g]).drop_duplicates("qid")
sel["prev_turn_tail"] = sel.prev_turn_tail.fillna("").map(lambda s: " ".join(str(s).split()[-30:]))
sel = sel.fillna("").sample(frac=1, random_state=7).reset_index(drop=True)
nch = int(np.ceil(len(sel) / 600))
for c in range(nch):
    sel.iloc[c * 600:(c + 1) * 600].to_csv(f"{D}/label_chunks/pass2_chunk_{c}.csv", index=False)
print(f"pass 2: {len(sel)} rows (test {len(test)}, hard dev {len(hard)}, random dev {len(rand)}, gold {len(g)}) in {nch} chunks")
