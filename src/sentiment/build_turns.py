#!/usr/bin/env python3
"""
Build the turn table for the sentiment analysis.

Reuses the same turn construction as topic_control.py: parse the diarized
host/guest transcripts, merge consecutive lines from the same speaker, and
drop pure backchannels. On top of that, very short turns (< --min_words
words) are dropped so the sentiment model only scores substantive turns.

Each kept turn also records the index of the turn before it in the filtered
sequence (prev_turn_idx), which the mirroring / escalation / recovery
measures need later.

Usage:
    python src/sentiment/build_turns.py \
        --episodes results/features/all_episodes.csv \
        --transcript_dir data/outputs/whisperx \
        --out results/sentiment/sentiment_turns.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "topic_control"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "others", "v1"))
from topic_control import parse_file, merge_consecutive_same_speaker, raw_word_count
from interruption_matrix import is_backchannel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/features/all_episodes.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--out", default="results/sentiment/sentiment_turns.csv")
    ap.add_argument("--min_words", type=int, default=5)
    args = ap.parse_args()

    episodes = pd.read_csv(args.episodes)
    rows = []
    for _, ep in episodes.iterrows():
        path = os.path.join(args.transcript_dir, f"{ep.episode_id}.txt")
        if not os.path.exists(path):
            continue
        turns = merge_consecutive_same_speaker(parse_file(path, ep.episode_id))
        turns = [t for t in turns if not is_backchannel(t.text)]
        turns = [t for t in turns if raw_word_count(t.text) >= args.min_words]
        for i, t in enumerate(turns):
            rows.append({
                "episode_id": ep.episode_id,
                "turn_idx": i,
                "prev_turn_idx": i - 1 if i > 0 else None,
                "speaker": t.speaker_id,
                "role": t.role,
                "gender": t.gender,
                "start": t.start,
                "end": t.end,
                "word_count": raw_word_count(t.text),
                "dyad": ep.dyad,
                "text": t.text,
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} turns from {df.episode_id.nunique()} episodes -> {args.out}")
    print(df.groupby("dyad").agg(turns=("text", "size"), episodes=("episode_id", "nunique")))


if __name__ == "__main__":
    main()
