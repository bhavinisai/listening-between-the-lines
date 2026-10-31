#!/usr/bin/env python3
"""
v2 Step 0: re-extract every HOST-asked question from all episodes, this time
keeping conversational context, and attach host/episode metadata and a
normalised-text key for deduplication.

Extraction rule is identical to question_type_analysis.py (sentence ending in
"?" within a merged speaker turn), so question counts match the v1 host
subset. New columns:

    qid             stable id: <episode_id>_q<k>
    host_id         named host, or unknown_<episode_id> if the diarizer left
                    SPEAKER_XX (same rule as question_type_guest_gender.py)
    prev_sentence   sentence immediately before the question in the same turn
    prev_turn_tail  last ~60 words of the previous turn (the guest's answer)
    text_norm       lowercased, punctuation-stripped text (dedupe key)

Input:  results/features/all_episodes.csv, data/outputs/whisperx/*.txt
Output: results/question_type_v2/pool_host_questions.csv  (new file; v1
        outputs are not touched)
"""

import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "topic_control"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "question_type_analysis"))
from topic_control import parse_file, merge_consecutive_same_speaker  # noqa: E402
from question_type_analysis import split_sentences  # noqa: E402

TAIL_WORDS = 60


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def tail(text, n=TAIL_WORDS):
    w = text.split()
    return " ".join(w[-n:])


def host_id(speaker, episode_id):
    return f"unknown_{episode_id}" if str(speaker).startswith("SPEAKER_") else str(speaker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/features/all_episodes.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--out", default="results/question_type_v2/pool_host_questions.csv")
    args = ap.parse_args()

    eps = pd.read_csv(args.episodes)
    rows = []
    for _, e in eps.iterrows():
        path = os.path.join(args.transcript_dir, f"{e['episode_id']}.txt")
        if not os.path.exists(path):
            continue
        turns = merge_consecutive_same_speaker(parse_file(path, e["episode_id"]))
        k = 0
        for i, t in enumerate(turns):
            if t.role != "HOST":
                continue
            prev_turn = turns[i - 1].text if i > 0 else ""
            sents = split_sentences(t.text)
            for j, s in enumerate(sents):
                if not s.endswith("?"):
                    continue
                rows.append({
                    "qid": f"{e['episode_id']}_q{k:04d}",
                    "episode_id": e["episode_id"],
                    "host_id": host_id(t.speaker_id, e["episode_id"]),
                    "host_gender": e["host_gender"],
                    "guest_gender": e["guest_gender"],
                    "dyad": e["dyad"],
                    "start": t.start,
                    "text": s,
                    "prev_sentence": sents[j - 1] if j > 0 else "",
                    "prev_turn_tail": tail(prev_turn),
                    "text_norm": norm(s),
                })
                k += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] {len(df)} host questions, {df.episode_id.nunique()} episodes, "
          f"{df.host_id.nunique()} host ids -> {args.out}")
    print(f"     unique normalised texts: {df.text_norm.nunique()}")
    print(df.host_id.value_counts().head(8))


if __name__ == "__main__":
    main()
