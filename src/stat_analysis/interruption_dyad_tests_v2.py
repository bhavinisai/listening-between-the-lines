#!/usr/bin/env python3
"""
interruption_dyad_tests_v2.py

Per-dyad tests of interruption rates
(results/interruptions/interruption_rates_by_episode.csv, from
src/interruptions/detect_interruptions.py). Rates are events per 10 minutes of
the overlapped speaker's speaking time.

  - summary per dyad: median and IQR of each host and guest rate
  - Kruskal-Wallis test across the 4 dyads for each rate, BH-FDR corrected
    across rates
  - pairwise two-sided Mann-Whitney U tests (6 dyad pairs) for each rate,
    BH-FDR corrected within rate, with the rank-biserial correlation
    (positive = first dyad higher)
  - within-episode Wilcoxon signed-rank test of host vs guest rate, per dyad
    and overall, BH-FDR corrected across rates

Usage:
  python src/stat_analysis/interruption_dyad_tests_v2.py
"""

import argparse
import os
import sys
from itertools import combinations

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from dyad_analysis_v2 import DYAD_ORDER  # noqa: E402
from fdr import bh_fdr  # noqa: E402

AB = {"MALE->MALE": "MM", "MALE->FEMALE": "MF", "FEMALE->MALE": "FM", "FEMALE->FEMALE": "FF"}
KINDS = ["successful_interruption", "unsuccessful_interruption", "backchannel"]


def main():
    ap = argparse.ArgumentParser(description="Per-dyad tests of interruption rates")
    ap.add_argument("--input", default="results/interruptions/interruption_rates_by_episode.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/interruptions/interruption_dyad_tests_v2")
    args = ap.parse_args()

    d = pd.read_csv(args.input)
    d = d[d.dyad.isin(DYAD_ORDER)]
    metrics = [f"{who}_{k}_per10" for k in KINDS for who in ["host", "guest"]]

    rows = []
    for dy in DYAD_ORDER:
        g = d[d.dyad == dy]
        row = {"dyad": dy, "episodes": len(g)}
        for m in metrics:
            row[f"{m}_median"] = g[m].median()
            row[f"{m}_q1"] = g[m].quantile(0.25)
            row[f"{m}_q3"] = g[m].quantile(0.75)
        rows.append(row)
    summary = pd.DataFrame(rows)

    kw = []
    for m in metrics:
        h, p = kruskal(*[d.loc[d.dyad == dy, m].dropna() for dy in DYAD_ORDER])
        kw.append({"metric": m, "H_statistic": h, "df": len(DYAD_ORDER) - 1, "p_value": p})
    kw = pd.DataFrame(kw)
    kw["p_fdr"] = bh_fdr(kw.p_value)

    pw = []
    for m in metrics:
        block = []
        for a, b in combinations(DYAD_ORDER, 2):
            x, y = d.loc[d.dyad == a, m].dropna(), d.loc[d.dyad == b, m].dropna()
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            block.append({"metric": m, "contrast": f"{AB[a]} vs {AB[b]}", "group_a": a, "group_b": b,
                          "median_a": x.median(), "median_b": y.median(), "U_statistic": u, "p": p,
                          "rank_biserial": 2 * u / (len(x) * len(y)) - 1})
        for r, q in zip(block, bh_fdr([r["p"] for r in block])):
            r["p_fdr"] = q
        pw += block
    pw = pd.DataFrame(pw)

    hg = []
    for k in KINDS:
        for dy in DYAD_ORDER + ["ALL"]:
            g = d if dy == "ALL" else d[d.dyad == dy]
            g = g[[f"host_{k}_per10", f"guest_{k}_per10"]].dropna()
            diff = g.iloc[:, 0] - g.iloc[:, 1]
            stat, p = wilcoxon(g.iloc[:, 0], g.iloc[:, 1]) if (diff != 0).any() else (float("nan"), 1.0)
            hg.append({"kind": k, "dyad": dy, "episodes": len(g),
                       "host_median": g.iloc[:, 0].median(), "guest_median": g.iloc[:, 1].median(),
                       "median_host_minus_guest": diff.median(), "W_statistic": stat, "p": p})
    hg = pd.DataFrame(hg)
    hg["p_fdr"] = bh_fdr(hg.p)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    kw.to_csv(f"{args.out_prefix}_kruskal.csv", index=False)
    pw.to_csv(f"{args.out_prefix}_pairwise.csv", index=False)
    hg.to_csv(f"{args.out_prefix}_host_vs_guest.csv", index=False)

    pd.set_option("display.width", 200)
    print(kw.round(4).to_string(index=False))
    print()
    print(hg.round(4).to_string(index=False))
    print(f"\nWrote {args.out_prefix}_{{summary,kruskal,pairwise,host_vs_guest}}.csv")


if __name__ == "__main__":
    main()
