#!/usr/bin/env python3
"""
ask_within_host_check_v2.py

Checks the exploratory direct-question result of within_host_test_v2.py
(male hosts ask female guests more direct questions than male guests; female
hosts ask female guests fewer) with an independent measure of questions:
the dialogue-act classifier's "ask" label (dialogue_act_classification.py,
418 episodes combined by combine_dialogue_acts.py).

Criterion, fixed before looking at the results: the pattern is confirmed if
the host-gender x guest-gender interaction of the within-host test on the
host "ask" rate has the same sign as for ck_direct_question (guest-gender
effect positive for male hosts, negative for female hosts) and p < 0.05.

Host "ask" measures per episode (all segments labeled HOST):
  host_ask_rate     share of host segments labeled ask
  host_ask_per_1k   host segments labeled ask per 1,000 host words
The same episodes as the direct-question result are used, so this is an
independent measurement of questions, not independent data.

Also reported: agreement between host_ck_direct_question_per_1k and the ask
measures (Spearman correlation across episodes and within host), the
per-host guest-gender differences, the guest-gender effect tested separately
within male hosts and within female hosts, and the interaction re-run
leaving out one host at a time.

Usage:
  python src/stat_analysis/ask_within_host_check_v2.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from within_host_test_v2 import within_host_guest_gender  # noqa: E402

DQ = "host_ck_direct_question_per_1k"
ASK = ["host_ask_rate", "host_ask_per_1k"]


def host_ask(labels_csv):
    lab = pd.read_csv(labels_csv, usecols=["episode_id", "role", "text", "dialogue_act"])
    h = lab[lab.role == "HOST"].copy()
    h["words"] = h.text.fillna("").str.split().str.len()
    h["is_ask"] = (h.dialogue_act == "ask").astype(int)
    g = h.groupby("episode_id").agg(host_segments=("is_ask", "size"), host_ask=("is_ask", "sum"),
                                    host_words_da=("words", "sum"))
    g["host_ask_rate"] = g.host_ask / g.host_segments
    g["host_ask_per_1k"] = 1000 * g.host_ask / g.host_words_da
    return g.reset_index()


def guest_effect(d, metrics, n_perm, seed):
    """Weighted within-host guest-gender effect (female minus male guests, in
    within-host percentile rank) for one set of hosts, with a within-host
    permutation p. Same statistic as within_host_test_v2.py, without the
    host-gender split."""
    rng = np.random.default_rng(seed)
    hosts = [g for _, g in d[d.host_name.notna()].groupby("host_name") if g.guest_gender.nunique() == 2]
    fem = [(g.guest_gender == "female").to_numpy() for g in hosts]
    w = np.array([f.sum() * (~f).sum() / len(f) for f in fem])
    rows = []
    for m in metrics:
        ranks = [g[m].rank(pct=True).to_numpy() for g in hosts]
        stat = lambda masks: np.average([r[f].mean() - r[~f].mean() for r, f in zip(ranks, masks)], weights=w)
        obs = stat(fem)
        null = np.array([stat([rng.permutation(f) for f in fem]) for _ in range(n_perm)])
        rows.append({"metric": m, "effect": obs, "p_perm": (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1),
                     "n_hosts": len(hosts), "n_episodes": int(sum(len(f) for f in fem))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Check the direct-question result with the dialogue-act ask label")
    ap.add_argument("--labels", default="results/dialogue_acts/all_418/dialogue_act_labels.csv")
    ap.add_argument("--dyads", default="results/dyads/dyad_analysis_v2.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/ask_within_host_check_v2")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = pd.read_csv(args.dyads).merge(host_ask(args.labels), on="episode_id", how="left")
    missing = d[ASK[0]].isna().sum()
    if missing:
        print(f"WARNING: {missing} episodes have no host dialogue-act labels")

    # 1. Agreement between the two measures of host questions
    agree = []
    for m in ASK:
        rho, p = spearmanr(d[DQ], d[m], nan_policy="omit")
        wd = d[d.host_name.notna()].copy()
        rw, pw = spearmanr(wd.groupby("host_name")[DQ].rank(pct=True), wd.groupby("host_name")[m].rank(pct=True))
        agree.append({"measure": m, "spearman_all": rho, "p_all": p, "spearman_within_host": rw, "p_within_host": pw})
    agree = pd.DataFrame(agree)

    # 2. Within-host guest-gender test and host-gender interaction
    wh = within_host_guest_gender(d, [DQ] + ASK, args.n_perm, args.seed)
    cols = ["metric", "rank_diff_female_minus_male", "p_perm", "rank_diff_male_hosts", "rank_diff_female_hosts",
            "host_gender_interaction", "p_interaction", "n_hosts", "n_episodes"]
    per_host = wh[["metric"] + [c for c in wh.columns if c.startswith("diff_")]]
    wh = wh[cols]
    ref = wh.set_index("metric").loc[DQ, "host_gender_interaction"]
    wh["same_sign_as_direct_question"] = (wh.host_gender_interaction * ref > 0)
    wh["confirms"] = wh.same_sign_as_direct_question & (wh.p_interaction < 0.05) & \
        (wh.rank_diff_male_hosts > 0) & (wh.rank_diff_female_hosts < 0)

    # 3. Guest-gender effect separately within male and within female hosts
    split = pd.concat([guest_effect(d[d.host_gender == hg], [DQ] + ASK, args.n_perm, args.seed).assign(host_gender=hg)
                       for hg in ["male", "female"]])

    # 4. Interaction leaving out one host at a time
    loho = []
    for h in sorted(d.host_name.dropna().unique()):
        w = within_host_guest_gender(d[d.host_name != h], [DQ] + ASK, 3000, args.seed).set_index("metric")
        for m in [DQ] + ASK:
            loho.append({"dropped_host": h, "metric": m, "host_gender_interaction": w.loc[m, "host_gender_interaction"],
                         "p_interaction": w.loc[m, "p_interaction"]})
    loho = pd.DataFrame(loho)

    # 5. Medians by dyad, for description
    by_dyad = d.groupby("dyad")[[DQ] + ASK].median()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    agree.to_csv(f"{args.out_prefix}_agreement.csv", index=False)
    wh.to_csv(f"{args.out_prefix}_within_host.csv", index=False)
    per_host.to_csv(f"{args.out_prefix}_per_host.csv", index=False)
    by_dyad.to_csv(f"{args.out_prefix}_by_dyad.csv")
    split.to_csv(f"{args.out_prefix}_by_host_gender.csv", index=False)
    loho.to_csv(f"{args.out_prefix}_leave_one_host_out.csv", index=False)
    d[["episode_id", "host_name", "dyad", DQ, "host_segments", "host_ask", "host_words_da"] + ASK].to_csv(
        f"{args.out_prefix}_episodes.csv", index=False)

    pd.set_option("display.width", 220)
    print("Agreement between direct-question count and ask label:")
    print(agree.round(3).to_string(index=False))
    print("\nWithin-host guest-gender effect (percentile rank, female minus male guests):")
    print(wh.round(4).to_string(index=False))
    print("\nPer-host effect:")
    print(per_host.set_index("metric").T.round(3).to_string())
    print("\nGuest-gender effect within male hosts and within female hosts:")
    print(split.round(4).to_string(index=False))
    print("\nInteraction, leaving one host out:")
    print(loho.pivot(index="dropped_host", columns="metric", values="p_interaction").round(3).to_string())
    print("\nMedians by dyad:")
    print(by_dyad.round(3).to_string())


if __name__ == "__main__":
    main()
