#!/usr/bin/env python3
"""
Merge the per-chunk LLM-produced labels (results/llm_label_chunks/labels_*.csv,
question_id,label) back onto the seed sample (results/question_type_llm_seed_sample.csv)
into the training file finetune_question_type_xlmr.py expects.

Output: results/question_type_seed_labeled.csv (question_id, text, label, dyad,
asker_role, asker_gender, episode_id)
"""

import argparse
import glob
import os

import pandas as pd

CLASS_LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="results/question_type_llm_seed_sample.csv")
    ap.add_argument("--label_glob", default="results/llm_label_chunks/labels_*.csv")
    ap.add_argument("--out", default="results/question_type_seed_labeled.csv")
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)

    label_files = sorted(glob.glob(args.label_glob))
    if not label_files:
        raise SystemExit(f"No label files matched {args.label_glob}")

    labels = pd.concat([pd.read_csv(f) for f in label_files], ignore_index=True)
    labels["label"] = labels["label"].str.strip().str.lower()

    dupes = labels["question_id"].duplicated()
    if dupes.any():
        raise SystemExit(f"{dupes.sum()} duplicate question_id(s) across label chunks")

    bad = ~labels["label"].isin(CLASS_LABELS)
    if bad.any():
        raise SystemExit(f"{bad.sum()} rows have a label outside {CLASS_LABELS}: "
                          f"{sorted(labels.loc[bad, 'label'].unique().tolist())}")

    merged = sample.merge(labels, on="question_id", how="left", validate="one_to_one")
    missing = merged["label"].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum()} sampled question_id(s) have no label "
                          f"(missing from chunk output): "
                          f"{merged.loc[missing, 'question_id'].tolist()[:10]}...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[OK] Wrote {len(merged)} labeled rows -> {args.out}")
    print(merged["label"].value_counts())
    print(pd.crosstab(merged["dyad"], merged["label"]))


if __name__ == "__main__":
    main()
