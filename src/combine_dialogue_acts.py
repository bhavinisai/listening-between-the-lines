#!/usr/bin/env python3
"""
combine_dialogue_acts.py

Combines the two dialogue_act_classification.py runs (the balanced 200
episodes in results/dialogue_acts/ and the remaining 218 in
results/dialogue_acts/remaining_218/, same script and settings) into
418-episode tables in results/dialogue_acts/all_418/.

Usage:
  python src/combine_dialogue_acts.py
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Combine dialogue-act runs into one 418-episode table")
    ap.add_argument("--parts", nargs="+", default=["results/dialogue_acts", "results/dialogue_acts/remaining_218"])
    ap.add_argument("--out_dir", default="results/dialogue_acts/all_418")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in ["dialogue_act_labels.csv", "dialogue_act_rates_by_episode_role.csv"]:
        parts = [pd.read_csv(os.path.join(p, name)) for p in args.parts]
        ids = [set(d.episode_id) for d in parts]
        overlap = set.intersection(*ids)
        if overlap:
            raise SystemExit(f"{name}: {len(overlap)} episodes appear in more than one run")
        df = pd.concat(parts, ignore_index=True).fillna(0) if "rates" in name else pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(args.out_dir, name), index=False)
        print(f"{name}: {len(df)} rows, {df.episode_id.nunique()} episodes")


if __name__ == "__main__":
    main()
