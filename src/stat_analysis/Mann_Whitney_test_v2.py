#!/usr/bin/env python3
"""
Mann_Whitney_test_v2.py

Pairwise follow-up to kruskal_wallis_test_v2.py: for each metric that was
significant there, a two-sided Mann-Whitney U test on all 6 pairs of
host->guest gender dyads, Benjamini-Hochberg FDR-corrected within each
metric. Effect size is the rank-biserial correlation, 2U/(n_a*n_b) - 1,
so positive means group_a is higher (the opposite sign of
Mann_Whitney_test_dialogue_acts.py).

Usage:
  python src/stat_analysis/Mann_Whitney_test_v2.py
"""

import argparse
import os
import sys
from itertools import combinations

import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dyad_analysis_v2 import DYAD_ORDER  # noqa: E402
from fdr import bh_fdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pairwise Mann-Whitney between dyads for v2 metrics")
    ap.add_argument("--input", default="results/dyads/dyad_analysis_v2.csv")
    ap.add_argument("--kruskal", default="results/stat_analysis/kruskal_wallis_v2_results.csv",
                    help="Output of kruskal_wallis_test_v2.py; only its significant metrics are tested")
    ap.add_argument("--out", default="results/stat_analysis/pairwise_dyad_tests_v2.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    kw = pd.read_csv(args.kruskal)
    metrics = kw.loc[kw.significant, "metric"].tolist()

    rows = []
    for m in metrics:
        block = []
        for a, b in combinations(DYAD_ORDER, 2):
            x, y = df.loc[df.dyad == a, m].dropna(), df.loc[df.dyad == b, m].dropna()
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            block.append({"metric": m, "group_a": a, "group_b": b,
                          "median_a": x.median(), "median_b": y.median(),
                          "U_statistic": u, "raw_p": p,
                          "rank_biserial": 2 * u / (len(x) * len(y)) - 1})
        for r, q in zip(block, bh_fdr([r["raw_p"] for r in block])):
            r["p_fdr"] = q
            r["significant"] = q < args.alpha
        rows += block
    pw = pd.DataFrame(rows, columns=["metric", "group_a", "group_b", "median_a", "median_b",
                                     "U_statistic", "raw_p", "rank_biserial", "p_fdr", "significant"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pw.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(f"{len(metrics)} metrics x 6 pairs; {int(pw.significant.sum())} of {len(pw)} "
          "significant after BH-FDR within metric:")
    print(pw[pw.significant].round(4).to_string(index=False) if len(pw) else "(none)")


if __name__ == "__main__":
    main()
