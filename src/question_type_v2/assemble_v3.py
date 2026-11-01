#!/usr/bin/env python3
"""
v3 relabeling, steps 2-3.

    --stage adjudicate   merge v3 passes A and B, write the adjudication chunk
                         for rows where A != B
    --stage final        resolve labels, write v3 train/test files, print the
                         v2 -> v3 change summary

Label resolution for relabeled rows (selected + control):
    v3_two_pass_agree   A == B
    v3_adjudicated      A != B and pass C equals one of them
    v3_unresolved       otherwise (dropped from training; excluded from headline
                        test metrics, reported separately)
Rows that were not relabeled keep their v2 label (`v2_carryover`).

Never overwrites v2 files. Writes:
    v3/train_dev_v3.csv, v3/test_llm_v3.csv, v3/v3_pass_merged.csv
Each file keeps label_v2, v3 pass labels and label_source for provenance.
"""

import argparse
import glob

import numpy as np
import pandas as pd

D = "results/question_type_v2"
O = f"{D}/v3"
LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]


def load(tag):
    f = sorted(glob.glob(f"{O}/label_chunks/v3{tag}_labels_*.csv"))
    d = pd.concat([pd.read_csv(x) for x in f])
    assert d.qid.is_unique and d.label.isin(LABELS).all(), tag
    return d.rename(columns={"label": f"l{tag}", "confidence": f"c{tag}", "alt": f"alt{tag}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["adjudicate", "adjudicate2", "final"], required=True)
    args = ap.parse_args()

    man = pd.read_csv(f"{O}/relabel_manifest.csv")
    m = man.merge(load("A"), on="qid", validate="one_to_one").merge(load("B"), on="qid", validate="one_to_one")
    assert len(m) == len(man), "missing labels"
    m.to_csv(f"{O}/v3_pass_merged.csv", index=False)

    if args.stage == "adjudicate2":
        c = pd.read_csv(f"{O}/label_chunks/v3C_chunk_0.csv")
        c = c.rename(columns={"cand_a": "cand_b", "cand_b": "cand_a"})[
            ["qid", "text", "prev_sentence", "prev_turn_tail", "cand_a", "cand_b"]]
        c.to_csv(f"{O}/label_chunks/v3D_chunk_0.csv", index=False)
        print(f"wrote swapped-order adjudication chunk: {len(c)} rows")
        return

    if args.stage == "adjudicate":
        d = m[m.lA != m.lB].copy()
        rng = np.random.default_rng(5)
        sw = rng.random(len(d)) < 0.5
        d["cand_a"] = np.where(sw, d.lB, d.lA)
        d["cand_b"] = np.where(sw, d.lA, d.lB)
        d["prev_turn_tail"] = d.prev_turn_tail.fillna("").map(lambda t: " ".join(str(t).split()[-30:]))
        d[["qid", "text", "prev_sentence", "prev_turn_tail", "cand_a", "cand_b"]].fillna("").to_csv(
            f"{O}/label_chunks/v3C_chunk_0.csv", index=False)
        print(f"A vs B agreement {(m.lA == m.lB).mean():.3f} on {len(m)} rows; {len(d)} to adjudicate")
        print(m.groupby("split").apply(lambda x: (x.lA == x.lB).mean()).round(3).to_dict())
        return

    c = pd.read_csv(f"{O}/label_chunks/v3C_labels_0.csv").rename(columns={"label": "lC", "confidence": "cC"})
    dd = pd.read_csv(f"{O}/label_chunks/v3D_labels_0.csv").rename(columns={"label": "lD", "confidence": "cD"})
    m = m.merge(c, on="qid", how="left").merge(dd, on="qid", how="left")

    def res(r):
        # A != B: adjudicated twice with swapped candidate order (the first
        # adjudicator chose the first-listed candidate 74% of the time).
        # Accept a label only with >= 3 of 4 votes (A, B, C, D).
        if r.lA == r.lB:
            return r.lA, "v3_two_pass_agree"
        votes = pd.Series([r.lA, r.lB, r.lC, r.lD]).value_counts()
        if votes.iloc[0] >= 3:
            return votes.index[0], "v3_adjudicated"
        return None, "v3_unresolved"

    rr = m.apply(res, axis=1, result_type="expand")
    m["label_v3"], m["label_source_v3"] = rr[0], rr[1]
    m = m[["qid", "label_v3", "label_source_v3", "lA", "lB", "lC", "group", "reasons"]]

    for name, src, out in [("dev", f"{D}/train_dev.csv", "train_dev_v3.csv"), ("test", f"{D}/test_llm.csv", "test_llm_v3.csv")]:
        d = pd.read_csv(src).rename(columns={"label": "label_v2", "label_source": "label_source_v2"})
        d = d.merge(m, on="qid", how="left")
        carry = d.label_v3.isna() & d.label_source_v3.isna()
        d.loc[carry, "label_v3"] = d.loc[carry, "label_v2"]
        d.loc[carry, "label_source_v3"] = "v2_carryover"
        d["label"] = d["label_v3"]
        d.to_csv(f"{O}/{out}", index=False)
        ok = d.label.notna()
        print(f"\n== {name}: {len(d)} rows, {int(ok.sum())} with a v3 label")
        print(d.label_source_v3.value_counts().to_string())
        rel = d[d.label_source_v3.str.startswith("v3_") & d.label_v3.notna()]
        ch = rel[rel.label_v2 != rel.label_v3]
        print(f"relabeled rows {len(rel)}; label changed v2->v3: {len(ch)} ({len(ch)/max(len(rel),1):.1%})")
        for g, x in rel.groupby("group"):
            print(f"   {g}: {len(x)} rows, changed {(x.label_v2 != x.label_v3).sum()} ({(x.label_v2 != x.label_v3).mean():.1%})")
        print(pd.crosstab(rel.label_v2.fillna("unresolved_v2"), rel.label_v3).to_string())
        print("v3 label distribution:", d.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
