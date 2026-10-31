#!/usr/bin/env python3
"""
Draw a dyad-stratified random sample of turns to hand-label as the sentiment
gold set.

Each row gets blank `valence` (positive / neutral / negative) and `arousal`
(low / medium / high) columns to fill in. The previous turn is included as
context so the labeler can judge tone in the conversation, but labels are for
the `text` column only.

Usage:
    python src/sentiment/make_gold_sheet.py \
        --turns results/sentiment/sentiment_turns.csv \
        --out results/sentiment/gold_sheet.csv --per_dyad 50
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="results/sentiment/sentiment_turns.csv")
    ap.add_argument("--out", default="results/sentiment/gold_sheet.csv")
    ap.add_argument("--per_dyad", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.turns)
    prev = df[["episode_id", "turn_idx", "text"]].rename(
        columns={"turn_idx": "prev_turn_idx", "text": "prev_text"})
    df = df.merge(prev, on=["episode_id", "prev_turn_idx"], how="left")

    # Cap one episode's contribution so no single episode dominates a dyad.
    # MALE->UNKNOWN (2 turns) is excluded: too small to evaluate.
    parts = []
    for dyad, g in df[df.dyad != "MALE->UNKNOWN"].groupby("dyad"):
        g = g.sample(frac=1, random_state=args.seed).groupby("episode_id").head(3)
        parts.append(g.head(args.per_dyad))
    sample = pd.concat(parts)
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "gold_id", [f"s{i:03d}" for i in range(len(sample))])
    sample["valence"] = ""
    sample["arousal"] = ""
    sample[["gold_id", "prev_text", "text", "valence", "arousal",
            "episode_id", "turn_idx", "dyad", "role", "gender"]].to_csv(args.out, index=False)
    print(sample.dyad.value_counts())
    print(f"wrote {len(sample)} rows -> {args.out}")


if __name__ == "__main__":
    main()
