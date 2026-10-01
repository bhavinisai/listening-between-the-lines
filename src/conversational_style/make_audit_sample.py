#!/usr/bin/env python3
"""
make_audit_sample.py

Builds the manual precision-audit file for the style features chosen for the
dyad analysis, from the random hits sampled by compute_bias_features_v2.py
(--audit_csv). Each hit's context is rebuilt from the full turn text, with the
matched words wrapped in [[ ]], and each row carries the yes/no question to
answer for its feature.

Label is_true_positive with 1 (yes) or 0 (no). Leave notes in notes.

Usage:
  python src/conversational_style/make_audit_sample.py \
      --audit_in results/features/conversational_style/audit_sample_v2.csv \
      --out results/audit/audit_v2_to_label.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_bias_features_v2 import audit_snippet, load_turns  # noqa: E402

QUESTIONS = {
    "hy_hedge": (
        "Does [[ ]] express uncertainty, tentativeness, approximation, or reduced "
        "commitment to the claim? Yes: 'maybe it was 2015', 'I think he left', "
        "'about 50 people'. No: literal uses, e.g. 'it's about passion', "
        "'we would go every Sunday' (habitual), 'I feel sick'."
    ),
    "hy_booster": (
        "Does [[ ]] express certainty, emphasis, or strong commitment to the claim? "
        "Yes: 'definitely', 'I know for a fact', 'it never works'. No: literal uses, "
        "e.g. 'I know him', 'we found a flat', 'the show'."
    ),
    "ck_gratitude": (
        "Is the speaker thanking someone? No: e.g. reported speech about "
        "someone else thanking, or 'thanks to X' meaning 'because of X'."
    ),
    "ck_apologizing": (
        "Is the speaker apologizing or politely asking pardon (incl. 'sorry?' to ask "
        "for a repeat)? No: 'I feel sorry for them', reported apologies, song/film titles."
    ),
    "ck_deference": (
        "Is [[ ]] used to praise or defer to the other speaker or their contribution "
        "(e.g. 'Great question', 'Interesting, tell me more')? No: describing something "
        "else, e.g. 'good food', 'a nice house'."
    ),
    "ck_factuality": (
        "Does [[ ]] present the statement as a matter of fact or as contrasting with "
        "expectation ('actually, it was cheaper', 'in fact', 'the truth is')? "
        "No: plain intensifier ('really good'), 'really?' as a reaction."
    ),
    "ck_direct_question": (
        "Is this a direct wh-question (what/why/how/who) actually asked of someone? "
        "No: embedded or relative clauses, e.g. 'what I did was...', "
        "'I don't know why', 'how we met is a long story'."
    ),
}


def main():
    ap = argparse.ArgumentParser(description="Build the manual audit file for selected style features")
    ap.add_argument("--audit_in", default="results/features/conversational_style/audit_sample_v2.csv")
    ap.add_argument("--out", default="results/audit/audit_v2_to_label.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--features", default=",".join(QUESTIONS),
                    help="Comma-separated features to include")
    ap.add_argument("--exclude_edges_sec", type=float, default=60.0,
                    help="Must match the value used by compute_bias_features_v2.py")
    ap.add_argument("--context_chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = [f for f in features if f not in QUESTIONS]
    if unknown:
        sys.exit(f"No audit question defined for: {unknown}")

    df = pd.read_csv(args.audit_in)
    df = df[df["feature"].isin(features)].copy()

    turn_cache = {}
    contexts, missing = [], 0
    for r in df.itertuples():
        if r.episode_id not in turn_cache:
            path = os.path.join(args.transcript_dir, f"{r.episode_id}.json")
            turn_cache[r.episode_id] = {
                (t["speaker"], round(t["start"], 1)): t["text"]
                for t in load_turns(path, args.exclude_edges_sec)
            }
        text = turn_cache[r.episode_id].get((r.speaker_id, round(r.turn_start_sec, 1)))
        if text is None:
            missing += 1
            contexts.append(r.context)
        else:
            contexts.append(audit_snippet(text, str(r.term), args.context_chars))
    df["context"] = contexts
    if missing:
        print(f"WARNING: {missing} hits could not be matched to a turn; kept their old context",
              file=sys.stderr)

    df["question"] = df["feature"].map(QUESTIONS)
    df["is_true_positive"] = ""
    df["notes"] = ""
    # Shuffle within feature so episodes/speakers are interleaved, keep features grouped
    order = {f: i for i, f in enumerate(features)}
    df = df.sample(frac=1, random_state=args.seed)
    df = df.sort_values("feature", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    df.insert(0, "audit_id", range(1, len(df) + 1))
    cols = ["audit_id", "feature", "question", "term", "context", "is_true_positive", "notes",
            "episode_id", "speaker_id", "role", "turn_start_sec"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[cols].to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out}: {len(df)} hits "
          + ", ".join(f"{f}={n}" for f, n in df["feature"].value_counts().sort_index().items()))


if __name__ == "__main__":
    main()
