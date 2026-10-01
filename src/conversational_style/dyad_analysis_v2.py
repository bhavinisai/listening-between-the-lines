#!/usr/bin/env python3
"""
dyad_analysis_v2.py

Episode-level dyad dataset from the v2 style features
(compute_bias_features_v2.py), over all episodes with a valid host->guest
gender dyad (episode_dyads.py). The statistical tests on this dataset are in
src/stat_analysis/: kruskal_wallis_test_v2.py, Mann_Whitney_test_v2.py and
within_host_test_v2.py.

Features (per 1,000 words), chosen from the precision audits in
results/audit/ (audit_v2c/v2d/v2e):
    hy_hedge, hy_booster, ck_factuality, ck_deference  -> host and guest
    ck_direct_question                                 -> host only
Direct_question is host-only because guest hits were mostly rhetorical
(guest precision 0.40 vs host 0.97). Booster and deference precision also
differs by role (host 0.56 vs guest 0.77; 0.92 vs 0.69), so the analysis
compares each role across dyads (e.g. host hedging with male vs female
guests) instead of host-minus-guest asymmetries, which would mix the two
error rates. Speaking share and dominance ratio are word-count based.

The corpus has only 9 hosts (speaker library), each always in one
host-gender group, so host gender is confounded with host identity and
show. host_name is added so the tests can hold the host fixed.

Outputs:
  results/dyads/dyad_analysis_v2.csv            one row per episode
  results/dyads/dyad_median_mad_stats_v2.csv    median / MAD / n per dyad
  results/dyads/host_level_medians_v2.csv       median of each metric per host

Usage:
  python src/conversational_style/dyad_analysis_v2.py
"""

import argparse
import glob
import json
import os

import pandas as pd
from scipy.stats import median_abs_deviation

DYAD_ORDER = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
BOTH_ROLES = ["hy_hedge", "hy_booster", "ck_factuality", "ck_deference"]
HOST_ONLY = ["ck_direct_question"]


def metric_list():
    """Metrics tested across dyads, in reporting order."""
    m = ["host_speaking_share", "dominance_ratio"]
    for f in BOTH_ROLES:
        m += [f"host_{f}_per_1k", f"guest_{f}_per_1k"]
    m += [f"host_{f}_per_1k" for f in HOST_ONLY]
    return m


def host_names(transcript_glob):
    """episode_id -> host name from the speaker-library match (None if unmatched)."""
    out = {}
    for path in glob.glob(transcript_glob):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        host = obj.get("host_speaker_raw")
        info = obj.get("speaker_gender_mapping", {}).get(host, {}) if host else {}
        out[os.path.splitext(os.path.basename(path))[0]] = (
            info.get("name") if info.get("source") == "library" else None)
    return out


def build_wide(features_csv, dyads_csv):
    sf = pd.read_csv(features_csv)
    dy = pd.read_csv(dyads_csv)[["episode_id", "dyad", "host_gender", "guest_gender"]]

    # One HOST and one GUEST row per episode (the highest-word-count one, as in episode_dyads.py)
    sf = (sf[sf["role"].isin(["HOST", "GUEST"])]
          .sort_values(["episode_id", "role", "word_count"], ascending=[True, True, False])
          .drop_duplicates(["episode_id", "role"]))

    cols = ["word_count"] + [f"{f}_per_1k" for f in BOTH_ROLES + HOST_ONLY]
    host = sf[sf.role == "HOST"].set_index("episode_id")[cols].add_prefix("host_")
    guest = sf[sf.role == "GUEST"].set_index("episode_id")[cols].add_prefix("guest_")
    guest = guest.drop(columns=[f"guest_{f}_per_1k" for f in HOST_ONLY])

    df = dy.merge(host, left_on="episode_id", right_index=True).merge(
        guest, left_on="episode_id", right_index=True)
    total = df["host_word_count"] + df["guest_word_count"]
    df["host_speaking_share"] = df["host_word_count"] / total
    df["dominance_ratio"] = df["host_word_count"] / df["guest_word_count"]
    return df


def main():
    ap = argparse.ArgumentParser(description="Episode-level dyad dataset from v2 style features")
    ap.add_argument("--features", default="results/features/conversational_style/speaker_features_v2.csv")
    ap.add_argument("--dyads", default="results/features/episode_dyads.csv")
    ap.add_argument("--dyad_dir", default="results/dyads")
    ap.add_argument("--transcript_glob",
                    default="data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json")
    args = ap.parse_args()

    df = build_wide(args.features, args.dyads)
    df.insert(1, "host_name", df.episode_id.map(host_names(args.transcript_glob)))
    metrics = metric_list()
    print(f"{len(df)} episodes: " + ", ".join(
        f"{d}={n}" for d, n in df.dyad.value_counts().reindex(DYAD_ORDER).items()))
    print(f"{df.host_name.notna().sum()} with a library-matched host ({df.host_name.nunique()} hosts)")

    os.makedirs(args.dyad_dir, exist_ok=True)
    df.to_csv(f"{args.dyad_dir}/dyad_analysis_v2.csv", index=False)

    summary = pd.concat({
        m: df.groupby("dyad")[m].agg(median="median", mad=median_abs_deviation, n="count").reindex(DYAD_ORDER)
        for m in metrics}, axis=1)
    summary.to_csv(f"{args.dyad_dir}/dyad_median_mad_stats_v2.csv")

    matched = df[df.host_name.notna()]
    hl = matched.groupby(["host_name", "host_gender"])[metrics].median()
    hl.insert(0, "n_episodes", matched.groupby(["host_name", "host_gender"]).size())
    hl = hl.sort_index(level="host_gender")
    hl.to_csv(f"{args.dyad_dir}/host_level_medians_v2.csv")

    pd.set_option("display.width", 200)
    print("\nMedian by dyad:")
    print(summary.xs("median", axis=1, level=1).T.round(3).to_string())
    print("\nHost-level medians:")
    print(hl.round(2).to_string())


if __name__ == "__main__":
    main()
