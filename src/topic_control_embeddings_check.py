#!/usr/bin/env python3
"""
SUPPLEMENTARY robustness check for topic_control.py -- NOT the required
measure (that one specifies TF-IDF cosine similarity / lexical overlap
explicitly, and topic_control.py implements exactly that).

Question this answers: TF-IDF only catches shared *words*. Two turns about
the same topic phrased with different vocabulary (paraphrase, pronouns,
synonyms) score as unrelated under TF-IDF. Is the null result in
topic_control.py (no significant male/female gap in topic follow-up once
episode clustering is accounted for) an artifact of that crude lexical
measure, or does it hold up under a semantic similarity measure too?

Same initiation/follow-up pipeline as topic_control.py (episode-median
thresholds, backchannel filtering, first-turn-after-initiation pairing) --
only the similarity function changes: sentence-embedding cosine similarity
(all-MiniLM-L6-v2) instead of TF-IDF cosine similarity. Re-testing with the
same suite of models as the primary script would overstate how much this
adds; the episode-level paired test is the clearest, most direct comparison
point against topic_control.py's paired_ttest_episode_* results, so that's
what this script reports.

Usage:
    python src/topic_control_embeddings_check.py \
        --episodes results/features/balanced_200_episodes.csv \
        --transcript_dir data/outputs/whisperx \
        --out_dir results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from interruption_matrix import is_backchannel
from topic_control import parse_file, merge_consecutive_same_speaker, find_next_response

MODEL_NAME = "all-MiniLM-L6-v2"


def process_episode(episode_id, transcript_path, embeddings_by_text):
    turns = parse_file(transcript_path, episode_id=episode_id)
    turns = merge_consecutive_same_speaker(turns)
    turns = [t for t in turns if not is_backchannel(t.text)]
    if len(turns) < 3:
        return []

    vectors = [embeddings_by_text[t.text] for t in turns]

    incoming_sim = [None] * len(turns)
    for i in range(1, len(turns)):
        incoming_sim[i] = float(np.dot(vectors[i - 1], vectors[i]))

    known = [s for s in incoming_sim if s is not None]
    init_cutoff = sorted(known)[len(known) // 2] if known else 0.0
    is_initiation = [True if i == 0 else incoming_sim[i] <= init_cutoff for i in range(len(turns))]

    pairs = []
    for i, t in enumerate(turns):
        if not is_initiation[i]:
            continue
        j, response = find_next_response(turns, i, from_speaker=t.speaker_id)
        if response is None:
            continue
        pairs.append({
            "episode_id": episode_id,
            "initiator_gender": t.gender,
            "initiator_role": t.role,
            "similarity": float(np.dot(vectors[i], vectors[j])),
        })

    if not pairs:
        return pairs
    cutoff = sorted(p["similarity"] for p in pairs)[len(pairs) // 2]
    for p in pairs:
        p["followed_up"] = int(p["similarity"] > cutoff)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/features/balanced_200_episodes.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    episodes = pd.read_csv(args.episodes)

    print(f"Loading sentence-transformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # Parse every episode once up front so we can embed all turn texts in one
    # batch (much faster than re-embedding per episode).
    all_turns_by_episode = {}
    missing = 0
    unique_texts = set()
    for _, row in episodes.iterrows():
        episode_id = row["episode_id"]
        transcript_path = os.path.join(args.transcript_dir, f"{episode_id}.txt")
        if not os.path.exists(transcript_path):
            missing += 1
            continue
        turns = parse_file(transcript_path, episode_id=episode_id)
        turns = merge_consecutive_same_speaker(turns)
        turns = [t for t in turns if not is_backchannel(t.text)]
        all_turns_by_episode[episode_id] = (turns, transcript_path)
        unique_texts.update(t.text for t in turns)

    if missing:
        print(f"WARNING: {missing} episode(s) missing a transcript .txt, skipped")

    unique_texts = list(unique_texts)
    print(f"Embedding {len(unique_texts)} unique turn texts...")
    embeddings = model.encode(unique_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    embeddings_by_text = dict(zip(unique_texts, embeddings))

    all_pairs = []
    for episode_id, (turns, transcript_path) in all_turns_by_episode.items():
        all_pairs.extend(process_episode(episode_id, transcript_path, embeddings_by_text))

    os.makedirs(args.out_dir, exist_ok=True)
    turns_path = os.path.join(args.out_dir, "topic_control_embeddings_turns.csv")
    stats_path = os.path.join(args.out_dir, "topic_control_embeddings_stats.csv")

    if not all_pairs:
        print("WARNING: no initiation -> response pairs found, nothing to write")
        return

    turns_df = pd.DataFrame(all_pairs)
    dyads = episodes.set_index("episode_id")["dyad"]
    turns_df["dyad"] = turns_df["episode_id"].map(dyads)
    turns_df.to_csv(turns_path, index=False)
    print(f"[OK] Wrote {len(turns_df)} initiation -> response pairs to {turns_path}")

    mixed = turns_df[turns_df["dyad"].isin(["MALE->FEMALE", "FEMALE->MALE"])].copy()
    if mixed.empty:
        print("WARNING: no mixed-gender initiation -> response pairs found, nothing to summarize")
        return

    summary = (
        mixed.groupby("initiator_gender")["followed_up"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "followup_rate", "count": "n_initiations"})
        .reset_index()
    )
    print(summary)

    # Same episode-level paired comparison as topic_control.py's
    # paired_ttest_episode_* rows, for a direct apples-to-apples check.
    ep_gender = (
        mixed.groupby(["episode_id", "initiator_gender"])
        .agg(followup_rate=("followed_up", "mean"), mean_similarity=("similarity", "mean"))
        .reset_index()
    )
    rng = np.random.default_rng(42)
    n_perm = 10000
    stats_rows = []
    for value_col, test_stub in [
        ("followup_rate", "episode_followup_rate"),
        ("mean_similarity", "episode_similarity"),
    ]:
        pivoted = ep_gender.pivot(index="episode_id", columns="initiator_gender", values=value_col)
        pivoted = pivoted.dropna(subset=["male", "female"])
        if len(pivoted) > 1:
            t_stat, t_p = stats.ttest_rel(pivoted["male"], pivoted["female"])
            stats_rows.append({
                "test": f"embeddings_paired_ttest_{test_stub}",
                "statistic": t_stat, "p_value": t_p, "dof": len(pivoted) - 1,
                "notes": (
                    f"male mean={pivoted['male'].mean():.4f}, "
                    f"female mean={pivoted['female'].mean():.4f} "
                    f"(n={len(pivoted)} episodes; similarity = {MODEL_NAME} embedding cosine, "
                    "not TF-IDF)"
                ),
            })
            diffs = (pivoted["male"] - pivoted["female"]).values
            observed = diffs.mean()
            signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
            null_means = (diffs[None, :] * signs).mean(axis=1)
            perm_p = (np.sum(np.abs(null_means) >= abs(observed)) + 1) / (n_perm + 1)
            stats_rows.append({
                "test": f"embeddings_permutation_test_{test_stub}",
                "statistic": observed, "p_value": perm_p, "dof": float("nan"),
                "notes": f"two-sided p from {n_perm} sign-flip permutations of {len(diffs)} paired episodes",
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Wrote statistical tests to {stats_path}")
    print(stats_df)


if __name__ == "__main__":
    main()
