#!/usr/bin/env python3
"""
speaking_time_dyad_tests_v2.py

Per-dyad analysis of host and guest speaking time over all 418 episodes
(results/dyads/speaking_time_dyads_v2.csv, from src/speaking_time/speaking_time_v2.py):

  - summary per dyad: median host / guest / total minutes, and the host share
    of host+guest speaking time as median, IQR, and a 95% bootstrap CI of the
    median (percentile bootstrap, resampling episodes within the dyad)
  - Kruskal-Wallis test across the 4 dyads for host share, host minutes and
    guest minutes
  - pairwise two-sided Mann-Whitney U tests for host share (6 dyad pairs),
    Holm-corrected, with the rank-biserial correlation (positive = first
    dyad higher)

Usage:
  python src/stat_analysis/speaking_time_dyad_tests_v2.py
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
from dyad_analysis_v2 import DYAD_ORDER  # noqa: E402

AB = {"MALE->MALE": "MM", "MALE->FEMALE": "MF", "FEMALE->MALE": "FM", "FEMALE->FEMALE": "FF"}


def boot_median_ci(x, n_boot, rng):
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description="Per-dyad tests of host/guest speaking time")
    ap.add_argument("--input", default="results/dyads/speaking_time_dyads_v2.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/speaking_time/speaking_time_dyad_tests_v2")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = pd.read_csv(args.input)
    d = d[d.dyad.isin(DYAD_ORDER)]
    rng = np.random.default_rng(args.seed)

    rows = []
    for dy in DYAD_ORDER:
        g = d[d.dyad == dy]
        lo, hi = boot_median_ci(g.host_time_share, args.n_boot, rng)
        rows.append({"dyad": dy, "episodes": len(g),
                     "host_min_median": g.host_min.median(), "guest_min_median": g.guest_min.median(),
                     "host_guest_min_median": g.host_guest_min.median(),
                     "host_share_median": g.host_time_share.median(),
                     "host_share_q1": g.host_time_share.quantile(0.25),
                     "host_share_q3": g.host_time_share.quantile(0.75),
                     "host_share_median_ci_low": lo, "host_share_median_ci_high": hi,
                     "host_share_mean": g.host_time_share.mean()})
    summary = pd.DataFrame(rows)

    kw = []
    for m in ["host_time_share", "host_min", "guest_min"]:
        h, p = kruskal(*[d.loc[d.dyad == dy, m] for dy in DYAD_ORDER])
        kw.append({"metric": m, "H_statistic": h, "df": len(DYAD_ORDER) - 1, "p_value": p})
    kw = pd.DataFrame(kw)

    pw = []
    for a, b in combinations(DYAD_ORDER, 2):
        x, y = d.loc[d.dyad == a, "host_time_share"], d.loc[d.dyad == b, "host_time_share"]
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        pw.append({"contrast": f"{AB[a]} vs {AB[b]}", "group_a": a, "group_b": b,
                   "median_a": x.median(), "median_b": y.median(), "U_statistic": u, "p": p,
                   "rank_biserial": 2 * u / (len(x) * len(y)) - 1})
    pw = pd.DataFrame(pw)
    pw["p_holm"] = multipletests(pw.p, method="holm")[1]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    kw.to_csv(f"{args.out_prefix}_kruskal.csv", index=False)
    pw.to_csv(f"{args.out_prefix}_pairwise.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(d)} episodes\n")
    print(summary.round(4).to_string(index=False))
    print("\nKruskal-Wallis across dyads:")
    print(kw.round(4).to_string(index=False))
    print("\nPairwise Mann-Whitney on host share (Holm-corrected):")
    print(pw.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
