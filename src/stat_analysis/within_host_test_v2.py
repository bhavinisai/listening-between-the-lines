#!/usr/bin/env python3
"""
within_host_test_v2.py

Guest-gender effect on the v2 dyad metrics with the host held fixed.

Host gender in this corpus is tied to 9 individual hosts, so the pooled
dyad tests (kruskal_wallis_test_v2.py, Mann_Whitney_test_v2.py) cannot
separate host gender from host identity. Guest gender varies within each
host's show, so it can be tested within host:

  - Each metric is converted to a percentile rank within the host's episodes.
  - Per host: mean rank of female-guest episodes minus mean rank of
    male-guest episodes; averaged over hosts with both guest genders,
    weighting each host by n_f*n_m/n.
  - p from a permutation test shuffling guest gender within host;
    BH-FDR across metrics. Episodes whose host did not match the speaker
    library are excluded.
  - Exploratory: the same effect computed separately for male and female
    hosts, and their difference (does the guest-gender effect depend on host
    gender?), with a permutation p from the same shuffles.

Usage:
  python src/stat_analysis/within_host_test_v2.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from dyad_analysis_v2 import metric_list  # noqa: E402
from fdr import bh_fdr  # noqa: E402


def within_host_guest_gender(df, metrics, n_perm, seed):
    """Female-minus-male-guest difference in within-host percentile rank,
    weighted over hosts, with a within-host permutation p-value."""
    rng = np.random.default_rng(seed)
    d = df[df.host_name.notna()]
    hosts = [(h, g) for h, g in d.groupby("host_name")
             if g.guest_gender.nunique() == 2]
    fem = [(g.guest_gender == "female").to_numpy() for _, g in hosts]
    w = np.array([f.sum() * (~f).sum() / len(f) for f in fem])
    fhost = np.array([g.host_gender.iloc[0] == "female" for _, g in hosts])
    rows = []
    for m in metrics:
        ranks = [g[m].rank(pct=True).to_numpy() for _, g in hosts]

        def stats(masks):
            diffs = np.array([r[f].mean() - r[~f].mean() for r, f in zip(ranks, masks)])
            male_h = np.average(diffs[~fhost], weights=w[~fhost])
            female_h = np.average(diffs[fhost], weights=w[fhost])
            return np.average(diffs, weights=w), male_h, female_h, male_h - female_h

        obs = stats(fem)
        null = np.array([stats([rng.permutation(f) for f in fem]) for _ in range(n_perm)])
        pval = lambda k: (1 + np.sum(np.abs(null[:, k]) >= abs(obs[k]))) / (n_perm + 1)
        per_host = {f"diff_{h}": r[f].mean() - r[~f].mean() for (h, _), r, f in zip(hosts, ranks, fem)}
        rows.append({"metric": m, "rank_diff_female_minus_male": obs[0], "p_perm": pval(0),
                     "rank_diff_male_hosts": obs[1], "rank_diff_female_hosts": obs[2],
                     "host_gender_interaction": obs[3], "p_interaction": pval(3),
                     "n_hosts": len(hosts), "n_episodes": int(sum(len(f) for f in fem)), **per_host})
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_perm"])
    out["p_interaction_fdr"] = bh_fdr(out["p_interaction"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Guest-gender effect within host for v2 metrics")
    ap.add_argument("--input", default="results/dyads/dyad_analysis_v2.csv")
    ap.add_argument("--out", default="results/stat_analysis/conversational_style/within_host_guest_gender_v2.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # dominance_ratio has the same within-host ranks as host_speaking_share
    metrics = [m for m in metric_list() if m != "dominance_ratio"]
    wh = within_host_guest_gender(df, metrics, args.n_perm, args.seed)
    wh["significant"] = wh["p_fdr"] < args.alpha

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wh.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(f"Guest gender within host ({wh.n_hosts.iloc[0]} hosts, {wh.n_episodes.iloc[0]} episodes; "
          "percentile-rank difference, female minus male guests):")
    print(wh[["metric", "rank_diff_female_minus_male", "p_perm", "p_fdr", "significant"]].round(4).to_string(index=False))
    print("\nExploratory: guest-gender effect by host gender (male-host effect minus female-host effect):")
    print(wh[["metric", "rank_diff_male_hosts", "rank_diff_female_hosts", "host_gender_interaction",
              "p_interaction", "p_interaction_fdr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
