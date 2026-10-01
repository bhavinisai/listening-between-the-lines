#!/usr/bin/env python3
"""
style_dyad_tests_v2.py

Per-dyad analysis of the v2 conversational style features for the results
report, over all 418 episodes (results/dyads/dyad_analysis_v2.csv), with the
same methods as speaking_time_dyad_tests_v2.py:

  - summary per dyad and feature: median, IQR, 95% bootstrap CI of the median
    (percentile bootstrap over episodes), mean, and the share of episodes in
    which the feature occurs at all
  - Kruskal-Wallis test across the 4 dyads per feature, Holm-corrected
    across the features
  - pairwise two-sided Mann-Whitney U tests (6 dyad pairs) per feature,
    Holm-corrected within the feature, with the rank-biserial correlation
    (positive = first dyad higher)

Features (per 1,000 words): hedges, boosters, factuality, deference for host
and guest; direct questions for the host only (see dyad_analysis_v2.py).
The raw statistics equal those of kruskal_wallis_test_v2.py and
Mann_Whitney_test_v2.py; only the multiple-comparison correction differs
(Holm here, Benjamini-Hochberg there).

Usage:
  python src/stat_analysis/style_dyad_tests_v2.py
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from dyad_analysis_v2 import DYAD_ORDER, metric_list  # noqa: E402

AB = {"MALE->MALE": "MM", "MALE->FEMALE": "MF", "FEMALE->MALE": "FM", "FEMALE->FEMALE": "FF"}
STYLE_METRICS = [m for m in metric_list() if m.endswith("_per_1k")]


def main():
    ap = argparse.ArgumentParser(description="Per-dyad tests of v2 style features")
    ap.add_argument("--input", default="results/dyads/dyad_analysis_v2.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/conversational_style/style_dyad_tests_v2")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = pd.read_csv(args.input)
    d = d[d.dyad.isin(DYAD_ORDER)]
    rng = np.random.default_rng(args.seed)

    rows = []
    for m in STYLE_METRICS:
        for dy in DYAD_ORDER:
            x = d.loc[d.dyad == dy, m].to_numpy()
            meds = np.median(rng.choice(x, size=(args.n_boot, len(x)), replace=True), axis=1)
            lo, hi = np.percentile(meds, [2.5, 97.5])
            rows.append({"metric": m, "dyad": dy, "episodes": len(x), "median": np.median(x),
                         "q1": np.quantile(x, 0.25), "q3": np.quantile(x, 0.75),
                         "median_ci_low": lo, "median_ci_high": hi, "mean": x.mean(),
                         "share_nonzero": (x > 0).mean()})
    summary = pd.DataFrame(rows)

    kw = []
    for m in STYLE_METRICS:
        h, p = kruskal(*[d.loc[d.dyad == dy, m] for dy in DYAD_ORDER])
        kw.append({"metric": m, "H_statistic": h, "df": len(DYAD_ORDER) - 1, "p_value": p})
    kw = pd.DataFrame(kw)
    kw["p_holm"] = multipletests(kw.p_value, method="holm")[1]

    pw = []
    for m in STYLE_METRICS:
        block = []
        for a, b in combinations(DYAD_ORDER, 2):
            x, y = d.loc[d.dyad == a, m], d.loc[d.dyad == b, m]
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            block.append({"metric": m, "contrast": f"{AB[a]} vs {AB[b]}", "group_a": a, "group_b": b,
                          "median_a": x.median(), "median_b": y.median(), "U_statistic": u, "p": p,
                          "rank_biserial": 2 * u / (len(x) * len(y)) - 1})
        for r, q in zip(block, multipletests([r["p"] for r in block], method="holm")[1]):
            r["p_holm"] = q
        pw += block
    pw = pd.DataFrame(pw)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    kw.to_csv(f"{args.out_prefix}_kruskal.csv", index=False)
    pw.to_csv(f"{args.out_prefix}_pairwise.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(d)} episodes\n")
    print(summary.pivot(index="metric", columns="dyad", values="median")[DYAD_ORDER].round(2).to_string())
    print("\nKruskal-Wallis (Holm across features):")
    print(kw.round(4).to_string(index=False))
    print("\nPairwise rank-biserial (* = Holm p < 0.05):")
    pw["cell"] = pw.apply(lambda r: f"{r.rank_biserial:+.2f}{'*' if r.p_holm < 0.05 else ''}", axis=1)
    print(pw.pivot(index="metric", columns="contrast", values="cell").to_string())


if __name__ == "__main__":
    main()
