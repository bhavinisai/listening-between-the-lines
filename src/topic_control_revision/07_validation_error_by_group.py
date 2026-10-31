#!/usr/bin/env python3
"""
Step 7 (run AFTER 06): do automated errors differ by initiator role, gender
or dyad? Differential error is what would bias the gender/role comparisons.

For each grouping, design-weighted error rates of the automated follow-up
label (overall disagreement, false-positive rate among human non-engagement,
false-negative rate among human engagement) and of the initiation label
(share of automated initiations that humans judge as SAME topic), with
stratified-bootstrap CIs, plus the between-group difference.

Gender and role are not separable in mixed-gender episodes (gender is fixed
by dyad x role), so the four dyad x role cells are also reported.
With ~44 items per role-by-dyad cell, only large differences are detectable;
treat the output as descriptive.

Usage:  python src/topic_control_revision/07_validation_error_by_group.py [--dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

from validation_common import final_labels, load_annotations, strat_boot, vdir, wmean


def err_stats(x):
    w = x["design_weight"]
    ok = x["human_engaged"].notna()
    e = (x["followed_up"] != x["human_engaged"]).astype(float).where(ok)
    fp = ((x["followed_up"] == 1) & (x["human_engaged"] == 0)).astype(float).where(x["human_engaged"] == 0)
    fn = ((x["followed_up"] == 0) & (x["human_engaged"] == 1)).astype(float).where(x["human_engaged"] == 1)
    init_fp = (x["human_new_topic"] == 0).astype(float).where(x["human_new_topic"].notna())
    return {"followup_error": wmean(e, w), "followup_false_pos_rate": wmean(fp, w),
            "followup_false_neg_rate": wmean(fn, w), "initiation_not_new_topic": wmean(init_fp, w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = vdir(args.dir)
    df = final_labels(load_annotations(args.dir), args.dir)
    df["role_x_dyad"] = df["initiator_role"].str.lower() + " | " + df["dyad"]
    rows = []
    for grouping in ["initiator_role", "initiator_gender", "dyad", "role_x_dyad"]:
        levels = sorted(df[grouping].unique())
        for lev in levels:
            sub = df[df[grouping] == lev]
            pt = err_stats(sub)
            for k, v in pt.items():
                lo, hi = strat_boot(sub, lambda x, k=k: err_stats(x)[k], n=1000)
                rows.append({"grouping": grouping, "level": lev, "n_items": len(sub), "metric": k,
                             "estimate": round(v, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
        if len(levels) == 2:
            a, b = levels
            for k in err_stats(df):
                def diff(x, k=k):
                    xa, xb = x[x[grouping] == a], x[x[grouping] == b]
                    if len(xa) == 0 or len(xb) == 0:
                        return np.nan
                    return err_stats(xa)[k] - err_stats(xb)[k]
                lo, hi = strat_boot(df, diff, n=1000)
                rows.append({"grouping": grouping, "level": f"{a} minus {b}", "n_items": len(df),
                             "metric": k, "estimate": round(diff(df), 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "results_error_by_group.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
