#!/usr/bin/env python3
"""
Step 5 (run AFTER both annotators finish): inter-annotator agreement.

Reports raw agreement and Cohen's kappa for Q1 (new topic) and Q2
(engagement), on the full category set and on the binary codes used for
validation, with stratified-bootstrap 95% CIs. Agreement is reported
unweighted: it describes how consistently two people apply the guide on
these items, not a corpus prevalence. Also writes the list of disagreements
for adjudication.

Usage:  python src/topic_control_revision/05_validation_agreement.py [--dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

from validation_common import Q1_CODES, Q2_CODES, load_annotations, strat_boot, vdir


def kappa(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    cats = sorted(set(a) | set(b))
    if len(a) == 0:
        return np.nan
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = vdir(args.dir)
    df = load_annotations(args.dir)
    rows = []
    for q, codes in [("Q1", Q1_CODES), ("Q2", Q2_CODES)]:
        for scheme in ["all categories", "binary (excl. UNCLEAR/NA)"]:
            sub = df.copy()
            a, b = f"a1_{q}", f"a2_{q}"
            if scheme.startswith("binary"):
                sub = sub[sub[a].isin(codes) & sub[b].isin(codes)]
                sub[a], sub[b] = sub[a].map(codes), sub[b].map(codes)
            k = kappa(sub[a], sub[b])
            lo, hi = strat_boot(sub, lambda x: kappa(x[a], x[b]))
            rows.append({"question": q, "scheme": scheme, "n_items": len(sub),
                         "raw_agreement": round((sub[a] == sub[b]).mean(), 3),
                         "cohens_kappa": round(k, 3), "kappa_lo": round(lo, 3), "kappa_hi": round(hi, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "results_agreement.csv"), index=False)
    for q in ["Q1", "Q2"]:
        print(f"\n{q} confusion (annotator 1 rows x annotator 2 cols):")
        print(pd.crosstab(df[f"a1_{q}"], df[f"a2_{q}"]))
    dis = df[(df.a1_Q1 != df.a2_Q1) | (df.a1_Q2 != df.a2_Q2)]
    adj = pd.DataFrame({"item_id": dis.item_id, "annotator_1_Q1": dis.a1_Q1, "annotator_2_Q1": dis.a2_Q1,
                        "annotator_1_Q2": dis.a1_Q2, "annotator_2_Q2": dis.a2_Q2,
                        "final_Q1": "", "final_Q2": "", "adjudicated_by": "", "rationale": ""})
    p = os.path.join(d, "adjudication_to_fill.csv")
    adj.to_csv(p, index=False)
    print("\n", out.to_string(index=False))
    print(f"\n{len(adj)} items need adjudication -> {p}\n"
          "Fill final_Q1/final_Q2 (only where the annotators disagree on that question), "
          "then save the file as adjudication_completed.csv in the same folder.")


if __name__ == "__main__":
    main()
