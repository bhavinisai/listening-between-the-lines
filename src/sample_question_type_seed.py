#!/usr/bin/env python3
"""
Draw a stratified sample of extracted questions for LLM-assisted seed labeling
(the training data for the fine-tuned XLM-R question-type classifier in
question_type_analysis.py).

Stratifies by (dyad, asker_role) so the seed set represents host- and
guest-asked questions across all four dyads roughly proportionally to their
share of the full extracted set, and dedupes on exact text first so the same
recurring line ("What do you mean by that?") isn't labeled/counted twice.

Input:  results/question_classification.csv (from
        `python src/question_type_analysis.py --extract_only`)
Output: results/question_type_llm_seed_sample.csv
        (question_id, text, dyad, asker_role, asker_gender, episode_id)
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/question_classification.csv")
    ap.add_argument("--out", default="results/question_type_llm_seed_sample.csv")
    ap.add_argument("--n_total", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    strata = df.groupby(["dyad", "asker_role"], group_keys=False)
    sizes = strata.size()
    rng = np.random.default_rng(args.seed)

    sampled_parts = []
    for (dyad, role), group in strata:
        share = len(group) / len(df)
        n_stratum = max(1, round(args.n_total * share))
        n_stratum = min(n_stratum, len(group))
        idx = rng.choice(group.index.values, size=n_stratum, replace=False)
        sampled_parts.append(df.loc[idx])

    sample = pd.concat(sampled_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "question_id", [f"q{i:05d}" for i in range(len(sample))])
    sample = sample[["question_id", "text", "dyad", "asker_role", "asker_gender", "episode_id"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"[OK] Sampled {len(sample)} unique questions -> {args.out}")
    print(sample.groupby(["dyad", "asker_role"]).size())


if __name__ == "__main__":
    main()
