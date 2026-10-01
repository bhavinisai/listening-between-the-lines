#!/usr/bin/env python3
"""
kruskal_wallis_test_v2.py

Kruskal-Wallis H test across the 4 host->guest gender dyads for each v2
dyad metric (results/dyads/dyad_analysis_v2.csv, from
src/conversational_style/dyad_analysis_v2.py). P-values are Benjamini-Hochberg FDR-corrected
across the metrics. Metrics significant here are followed up pairwise by
Mann_Whitney_test_v2.py.

Usage:
  python src/stat_analysis/kruskal_wallis_test_v2.py
"""

import argparse
import os
import sys

import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from dyad_analysis_v2 import DYAD_ORDER, metric_list  # noqa: E402
from fdr import bh_fdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Kruskal-Wallis across dyads for v2 metrics")
    ap.add_argument("--input", default="results/dyads/dyad_analysis_v2.csv")
    ap.add_argument("--out", default="results/stat_analysis/conversational_style/kruskal_wallis_v2_results.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for m in metric_list():
        groups = [df.loc[df.dyad == d, m].dropna() for d in DYAD_ORDER]
        h, p = kruskal(*groups)
        rows.append({"metric": m, "H_statistic": h, "p_value": p,
                     **{f"median_{d}": g.median() for d, g in zip(DYAD_ORDER, groups)}})
    kw = pd.DataFrame(rows)
    kw["p_fdr"] = bh_fdr(kw["p_value"])
    kw["significant"] = kw["p_fdr"] < args.alpha

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kw.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print("Kruskal-Wallis across dyads (medians per dyad; BH-FDR across metrics):")
    print(kw.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
