#!/usr/bin/env python3
"""
Step 6 (run AFTER agreement + adjudication): how well do the automated
labels match the human (final) labels?

Because the sample is deliberately stratified (equal n per stratum, very
unequal stratum sizes), every corpus-level metric is a design-weighted
(Horvitz-Thompson ratio) estimate using design_weight = N_h / n_h. Unweighted
figures are also written but labeled "sample only" -- they describe the 176
items, not the corpus. CIs: 2,000 bootstrap resamples within strata.

Automated labels evaluated:
  * initiation: every sampled pair is an automated initiation, so only the
    share that humans also judge to introduce a new topic/sub-topic can be
    estimated (precision of initiation). Recall cannot be estimated from this
    sample, because non-initiation turns were not sampled.
  * follow-up: automated followed_up vs human Q2 (ENGAGE vs MINIMAL/NONE).

Usage:  python src/topic_control_revision/06_validation_accuracy.py [--dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

from validation_common import final_labels, load_annotations, strat_boot, vdir, wmean


def metrics(df, weighted=True):
    w = df["design_weight"].values if weighted else np.ones(len(df))
    h, a = df["human_engaged"].values, df["followed_up"].values.astype(float)
    ok = ~np.isnan(h)
    h, a, w = h[ok], a[ok], w[ok]

    def r(num, den):
        return num / den if den > 0 else np.nan
    tp, fp = np.sum(w * (a == 1) * (h == 1)), np.sum(w * (a == 1) * (h == 0))
    fn, tn = np.sum(w * (a == 0) * (h == 1)), np.sum(w * (a == 0) * (h == 0))
    tot = tp + fp + fn + tn
    po = r(tp + tn, tot)
    pa, ph = r(tp + fp, tot), r(tp + fn, tot)
    pe = pa * ph + (1 - pa) * (1 - ph)
    return {"accuracy": po, "sensitivity_recall": r(tp, tp + fn), "specificity": r(tn, tn + fp),
            "precision_ppv": r(tp, tp + fp), "npv": r(tn, tn + fn),
            "human_engagement_rate": ph, "automated_followup_rate": pa,
            "kappa_auto_vs_human": r(po - pe, 1 - pe)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = vdir(args.dir)
    df = final_labels(load_annotations(args.dir), args.dir)
    unresolved = ((df.final_Q1 == "") | (df.final_Q2 == "")).sum()
    if unresolved:
        print(f"WARNING: {unresolved} items have unresolved disagreements; run adjudication first. "
              "They are excluded below.")
    rows = []
    for weighted in [True, False]:
        label = "corpus estimate (design-weighted)" if weighted else "sample only (unweighted)"
        pt = metrics(df, weighted)
        for k, v in pt.items():
            lo, hi = strat_boot(df, lambda x, k=k: metrics(x, weighted)[k])
            rows.append({"target": "follow-up", "metric": k, "estimate_type": label,
                         "estimate": round(v, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
        for col, name in [("human_new_topic", "initiation precision (new topic or sub-topic)"),
                          ("human_new_topic_strict", "initiation precision (new topic only)")]:
            f = (lambda x, c=col: wmean(x[c], x["design_weight"])) if weighted else \
                (lambda x, c=col: x[c].mean())
            lo, hi = strat_boot(df, f)
            rows.append({"target": "initiation", "metric": name, "estimate_type": label,
                         "estimate": round(f(df), 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    excl = pd.DataFrame([{
        "items_labeled": len(df),
        "Q1_unclear_or_NA": int(df.human_new_topic.isna().sum()),
        "Q2_unclear_or_NA": int(df.human_engaged.isna().sum()),
        "weighted_share_Q2_excluded": round(wmean(df.human_engaged.isna().astype(float), df.design_weight), 3)}])
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "results_accuracy.csv"), index=False)
    excl.to_csv(os.path.join(d, "results_exclusions.csv"), index=False)
    by_stratum = (df.groupby("stratum").agg(n=("item_id", "size"), N_h=("N_h", "first"),
                                            human_engaged=("human_engaged", "mean"),
                                            human_new_topic=("human_new_topic", "mean"),
                                            automated_followup=("followed_up", "mean")).reset_index())
    by_stratum.to_csv(os.path.join(d, "results_by_stratum.csv"), index=False)
    print(out.to_string(index=False))
    print(excl.to_string(index=False))


if __name__ == "__main__":
    main()
