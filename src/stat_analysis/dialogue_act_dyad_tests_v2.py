#!/usr/bin/env python3
"""
dialogue_act_dyad_tests_v2.py

Per-dyad analysis of dialogue acts for the results report, over all 418
episodes (results/dialogue_acts/all_418/, from dialogue_act_classification.py
combined by combine_dialogue_acts.py), with the same methods as
speaking_time_dyad_tests_v2.py and style_dyad_tests_v2.py:

  - overall label distribution for host and guest segments
  - per dyad, role and act: median, IQR and 95% bootstrap CI of the median of
    the share of the role's segments with that act
  - Kruskal-Wallis across the 4 dyads per (role, act), Holm-corrected across
    the tests
  - pairwise two-sided Mann-Whitney U tests (6 dyad pairs), Holm-corrected
    within each (role, act), with the rank-biserial correlation
    (positive = first dyad higher)

Acts: the six that make up at least 1% of segments overall (ask, answer,
say, acknowledge, intent, reply_yes); the other five are too rare to compare
per episode.

Usage:
  python src/stat_analysis/dialogue_act_dyad_tests_v2.py
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
ACTS = ["ask", "answer", "say", "acknowledge", "intent", "reply_yes"]


def main():
    ap = argparse.ArgumentParser(description="Per-dyad tests of dialogue acts")
    ap.add_argument("--labels", default="results/dialogue_acts/all_418/dialogue_act_labels.csv")
    ap.add_argument("--rates", default="results/dialogue_acts/all_418/dialogue_act_rates_by_episode_role.csv")
    ap.add_argument("--dyads", default="results/features/episode_dyads.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/dialogue_act/dialogue_act_dyad_tests_v2")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, usecols=["role", "dialogue_act"])
    lab = lab[lab.role.isin(["HOST", "GUEST"])]
    dist = (lab.groupby("role").dialogue_act.value_counts(normalize=True).unstack(0)
            .reindex(columns=["HOST", "GUEST"]).fillna(0))
    dist["ALL"] = lab.dialogue_act.value_counts(normalize=True)
    dist = dist.sort_values("ALL", ascending=False)
    counts = lab.role.value_counts()

    rates = pd.read_csv(args.rates)
    dy = pd.read_csv(args.dyads)[["episode_id", "dyad"]]
    d = rates[rates.role.isin(["HOST", "GUEST"])].merge(dy, on="episode_id")
    d = d[d.dyad.isin(DYAD_ORDER)]
    rng = np.random.default_rng(args.seed)

    rows, kw, pw = [], [], []
    for role in ["HOST", "GUEST"]:
        r = d[d.role == role]
        for act in ACTS:
            for dyd in DYAD_ORDER:
                x = r.loc[r.dyad == dyd, act].to_numpy()
                meds = np.median(rng.choice(x, size=(args.n_boot, len(x)), replace=True), axis=1)
                lo, hi = np.percentile(meds, [2.5, 97.5])
                rows.append({"role": role, "act": act, "dyad": dyd, "episodes": len(x), "median": np.median(x),
                             "q1": np.quantile(x, 0.25), "q3": np.quantile(x, 0.75),
                             "median_ci_low": lo, "median_ci_high": hi, "mean": x.mean()})
            h, p = kruskal(*[r.loc[r.dyad == dyd, act] for dyd in DYAD_ORDER])
            kw.append({"role": role, "act": act, "H_statistic": h, "df": len(DYAD_ORDER) - 1, "p_value": p})
            block = []
            for a, b in combinations(DYAD_ORDER, 2):
                x, y = r.loc[r.dyad == a, act], r.loc[r.dyad == b, act]
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                block.append({"role": role, "act": act, "contrast": f"{AB[a]} vs {AB[b]}", "group_a": a,
                              "group_b": b, "median_a": x.median(), "median_b": y.median(), "U_statistic": u,
                              "p": p, "rank_biserial": 2 * u / (len(x) * len(y)) - 1})
            for row, q in zip(block, multipletests([b["p"] for b in block], method="holm")[1]):
                row["p_holm"] = q
            pw += block
    summary, kw, pw = pd.DataFrame(rows), pd.DataFrame(kw), pd.DataFrame(pw)
    kw["p_holm"] = multipletests(kw.p_value, method="holm")[1]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    dist.to_csv(f"{args.out_prefix}_distribution.csv")
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    kw.to_csv(f"{args.out_prefix}_kruskal.csv", index=False)
    pw.to_csv(f"{args.out_prefix}_pairwise.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{d.episode_id.nunique()} episodes; segments: host {counts.get('HOST', 0):,}, guest {counts.get('GUEST', 0):,}\n")
    print((100 * dist).round(1).to_string())
    print("\nMedian share of segments by dyad:")
    print(summary.pivot_table(index=["role", "act"], columns="dyad", values="median")[DYAD_ORDER].round(3).to_string())
    print("\nKruskal-Wallis (Holm across tests):")
    print(kw.round(4).to_string(index=False))
    pw["cell"] = pw.apply(lambda x: f"{x.rank_biserial:+.2f}{'*' if x.p_holm < 0.05 else ''}", axis=1)
    print("\nPairwise rank-biserial (* = Holm p < 0.05):")
    print(pw.pivot_table(index=["role", "act"], columns="contrast", values="cell", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
