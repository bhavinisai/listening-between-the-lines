#!/usr/bin/env python3
"""
v2: resolve LLM labels and write the train / test / gold CSVs used by train_v2.py.

Label resolution (column `label_source` records which rule applied):
    two_pass_agree    pass 1 == pass 2
    adjudicated       pass 1 != pass 2, pass 3 agrees with one of them
    unresolved        three different labels -> excluded from training, kept in
                      the test file with `label_source = unresolved` and
                      dropped from headline test metrics (reported separately)
    single_pass       dev row labeled only in pass 1 (pass-1 confidence >= 2,
                      outside the hard/random pass-2 subset)

NONE of these labels is human-verified. The only human labels are the 120 gold
questions (gold.csv), which are used for evaluation only.

Outputs (results/question_type_v2/):
    train_dev.csv   dev-host rows with resolved labels (training + host-grouped CV)
    test_llm.csv    sealed host-held-out test rows (LLM consensus labels)
    gold.csv        120 human-labeled questions with context
"""

import glob

import pandas as pd

D = "results/question_type_v2"

m = pd.read_csv(f"{D}/pass1_merged.csv")
p2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/label_chunks/pass2_labels_*.csv"))])
p2 = p2[~p2.qid.str.startswith("g")].rename(columns={"label": "label2", "confidence": "conf2", "alt": "alt2"})
p3 = pd.read_csv(f"{D}/label_chunks/pass3_labels_0.csv").rename(columns={"label": "label3", "confidence": "conf3"})
m = m.merge(p2, on="qid", how="left").merge(p3, on="qid", how="left")


def resolve(r):
    if pd.isna(r.label2):
        return r.label1, "single_pass"
    if r.label1 == r.label2:
        return r.label1, "two_pass_agree"
    if r.label3 in (r.label1, r.label2):
        return r.label3, "adjudicated"
    return None, "unresolved"


res = m.apply(resolve, axis=1, result_type="expand")
m["label"], m["label_source"] = res[0], res[1]
cols = ["qid", "text", "label", "label_source", "host_id", "host_gender", "guest_gender", "episode_id",
        "split", "source", "prev_sentence", "prev_turn_tail", "label1", "conf1", "label2", "label3"]
dev = m[(m.split == "dev") & m.label.notna()][cols]
test = m[m.split == "test"][cols]
dev.to_csv(f"{D}/train_dev.csv", index=False)
test.to_csv(f"{D}/test_llm.csv", index=False)

gold = pd.read_csv("gold_work/gold_full.csv")
hum = pd.read_csv("gold_work/gold_sheet_labeled.csv")[["gold_id", "label"]]
pool = pd.read_csv(f"{D}/pool_host_questions.csv", usecols=["episode_id", "text", "host_id", "prev_sentence", "prev_turn_tail"]
                   ).drop_duplicates(["episode_id", "text"])
g = gold.merge(hum, on="gold_id").merge(pool, on=["episode_id", "text"], how="left")
g = g.rename(columns={"gold_id": "qid"})[["qid", "text", "label", "host_id", "prev_sentence", "prev_turn_tail"]]
g["label"] = g.label.str.strip().str.lower()
g.to_csv(f"{D}/gold.csv", index=False)

print("label_source:\n", m.groupby(["split", "label_source"]).size().to_string())
print("\ntrain_dev labels:\n", dev.label.value_counts().to_string(), f"\n n={len(dev)}  hosts={dev.host_id.nunique()}")
print("\nhost counts (train_dev):\n", dev.host_id.value_counts().head(6).to_string())
t = test[test.label.notna()]
print("\ntest labels (resolved):\n", t.label.value_counts().to_string(), f"\n n={len(t)} of {len(test)}; hosts={t.host_id.nunique()}")
print("gold:", len(g), g.label.value_counts().to_dict())
assert not (set(dev.text) & set(test.text)), "text overlap dev/test"
assert not (set(dev.host_id) & set(test.host_id)), "host overlap dev/test"
