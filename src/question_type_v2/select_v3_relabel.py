#!/usr/bin/env python3
"""
v3 relabeling, step 1: MODEL-BLIND selection of questions to relabel.

Selection uses only v2 LLM-labeling metadata (labels from passes 1-3, the
labelers' `alt` second choice, confidence, adjudication status). It does NOT
use any classifier prediction, CV/test metric, or gold-set error. The same
rule is applied to dev and test rows.

Criteria (a row is selected if any holds; `reasons` records which):
    form_vs_content   final label open/closed, but personal/professional appears
                      as a label or alt in any pass
    pp_conflict       final label personal/professional and the other one
                      appears as a label or alt
    pp_vs_form        final label personal/professional but open/closed
                      appears as a label or alt (v3's clarity test may demote it)
    lead_vs_chal      leading and challenge both appear among labels/alts
    low_conf          adjudicated, unresolved, or pass-1 confidence 1
Control: a random sample of NOT-selected rows (150 dev, 100 test) is also
relabeled, to estimate how many unselected rows v3 would have changed. Rows
that are neither selected nor in the control keep their v2 label (documented as
"carried over from v2" in the v3 files).

Outputs (results/question_type_v2/v3/):
    relabel_manifest.csv
    label_chunks/v3A_chunk_k.csv, v3B_chunk_k.csv   (two independent orderings)
"""

import os

import numpy as np
import pandas as pd

D = "results/question_type_v2"
O = f"{D}/v3"
FORM = {"open", "closed"}
PP = {"personal", "professional"}
N_CTRL = {"dev": 150, "test": 100}
CHUNK = 500


def lab_set(r):
    s = {r.label1, r.label2, r.label3, r.alt1, r.alt2}
    return {x for x in s if isinstance(x, str) and x != "none"}


def main():
    os.makedirs(f"{O}/label_chunks", exist_ok=True)
    m = pd.read_csv(f"{D}/pass12_merged.csv")[["qid", "alt1", "alt2", "conf1"]]
    parts = []
    for name, path in [("dev", f"{D}/train_dev.csv"), ("test", f"{D}/test_llm.csv")]:
        d = pd.read_csv(path).merge(m, on="qid", how="left", suffixes=("", "_m"))
        d["split"] = name
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["conf1"] = d["conf1_m"].fillna(d["conf1"]) if "conf1_m" in d else d["conf1"]

    reasons = []
    for r in d.itertuples():
        S, lab = lab_set(r), r.label
        rs = []
        if lab in FORM and S & PP:
            rs.append("form_vs_content")
        if lab in PP and (PP - {lab}) & S:
            rs.append("pp_conflict")
        if lab in PP and S & FORM:
            rs.append("pp_vs_form")
        if {"leading", "challenge"} <= S:
            rs.append("lead_vs_chal")
        if r.label_source in ("adjudicated", "unresolved") or r.conf1 == 1:
            rs.append("low_conf")
        reasons.append(";".join(rs))
    d["reasons"] = reasons
    d["selected"] = d.reasons != ""
    # unresolved test rows (label NaN) cannot be carried over; always relabel
    d.loc[d.label.isna(), "selected"] = True
    d.loc[d.label.isna() & (d.reasons == ""), "reasons"] = "low_conf"

    rng = np.random.default_rng(123)
    d["control"] = False
    for sp, n in N_CTRL.items():
        pool = d[(d.split == sp) & ~d.selected].index
        d.loc[rng.choice(pool, size=min(n, len(pool)), replace=False), "control"] = True
    sel = d[d.selected | d.control].copy()
    sel["group"] = np.where(sel.selected, "selected", "control")
    cols = ["qid", "split", "group", "reasons", "host_id", "label", "label_source", "text",
            "prev_sentence", "prev_turn_tail"]
    sel[cols].rename(columns={"label": "label_v2", "label_source": "label_source_v2"}).to_csv(
        f"{O}/relabel_manifest.csv", index=False)

    for tag, seed in [("A", 1), ("B", 2)]:
        s = sel.sample(frac=1, random_state=seed).reset_index(drop=True)
        s["prev_turn_tail"] = s.prev_turn_tail.fillna("").map(lambda t: " ".join(str(t).split()[-30:]))
        for c in range(int(np.ceil(len(s) / CHUNK))):
            s.iloc[c * CHUNK:(c + 1) * CHUNK][["qid", "text", "prev_sentence", "prev_turn_tail"]].fillna("").to_csv(
                f"{O}/label_chunks/v3{tag}_chunk_{c}.csv", index=False)

    print(f"rows: {len(d)}; selected {int(d.selected.sum())}, control {int(d.control.sum())}, total to relabel {len(sel)}")
    print(sel.groupby(["split", "group"]).size().to_string())
    for k in ["form_vs_content", "pp_conflict", "pp_vs_form", "lead_vs_chal", "low_conf"]:
        print(f"  {k:16s}", int(sel.reasons.str.contains(k).sum()))
    print("chunks per pass:", int(np.ceil(len(sel) / CHUNK)))


if __name__ == "__main__":
    main()
