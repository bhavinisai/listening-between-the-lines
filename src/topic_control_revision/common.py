"""
Shared helpers for the topic-control revision (Report 2 / final capstone).

Nothing here changes the Report 1 measurement. `rebuild_episode_pairs` re-runs
the exact initiation/follow-up definition from src/topic_control/topic_control.py
(same parser, turn merging, backchannel filter, TF-IDF, episode-median
thresholds), but also keeps the turn indices and texts, which the original
output file does not store. `load_pairs_with_text` checks the rebuilt pairs
against the saved Report 1 file row by row before anything else uses them.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "topic_control"))
sys.path.insert(0, os.path.join(REPO, "src", "others", "v1"))

from interruption_matrix import is_backchannel  # noqa: E402
from topic_control import (  # noqa: E402
    build_tfidf_vectors, cosine_sim, find_next_response,
    merge_consecutive_same_speaker, parse_file, raw_word_count, tokenize,
)

# Inputs (read-only)
EPISODES_CSV = os.path.join(REPO, "results", "features", "all_episodes.csv")
TRANSCRIPT_DIR = os.path.join(REPO, "data", "outputs", "whisperx")
ORIG_DIR = os.path.join(REPO, "results", "topic_control")
ORIG_TURNS = os.path.join(ORIG_DIR, "topic_control_turns.csv")
ORIG_STATS = os.path.join(ORIG_DIR, "topic_control_stats.csv")
ORIG_EMB_TURNS = os.path.join(ORIG_DIR, "topic_control_embeddings_turns.csv")
ORIG_EMB_STATS = os.path.join(ORIG_DIR, "topic_control_embeddings_stats.csv")

# Outputs
OUT = os.path.join(REPO, "results", "topic_control_revision")
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")
DIAG = os.path.join(OUT, "diagnostics")
VERIFY = os.path.join(OUT, "verification")
VALID = os.path.join(OUT, "human_validation")
DATA = os.path.join(OUT, "data")
for d in (OUT, TABLES, FIGURES, DIAG, VERIFY, VALID, DATA):
    os.makedirs(d, exist_ok=True)

MIXED = ["MALE->FEMALE", "FEMALE->MALE"]
DYAD_SHORT = {"MALE->MALE": "M→M", "MALE->FEMALE": "M→F",
              "FEMALE->MALE": "F→M", "FEMALE->FEMALE": "F→F"}
SEED = 20261029


def load_original_mixed():
    """Saved Report 1 turn-pair file, mixed-gender episodes only, with model columns."""
    t = pd.read_csv(ORIG_TURNS)
    m = t[t["dyad"].isin(MIXED)].copy()
    m["is_male_initiator"] = (m["initiator_gender"] == "male").astype(float)
    m["is_host_initiator"] = (m["initiator_role"] == "HOST").astype(float)
    m["wc100"] = m["initiator_word_count"] / 100.0
    m["log_init_wc"] = np.log1p(m["initiator_word_count"])
    m["log_resp_wc"] = np.log1p(m["responder_word_count"])
    host = m[m["initiator_role"] == "HOST"].groupby("episode_id")["initiator_speaker"].first()
    # Hosts identified only by diarization label (LLM fallback) cannot be linked
    # across episodes, so each such episode is treated as its own host.
    host = host.where(~host.str.startswith("SPEAKER_"), "unnamed:" + host.index.to_series())
    m["host_id"] = m["episode_id"].map(host)
    return m


def rebuild_episode_pairs(episode_id, path):
    """Original Report 1 definition (episode-median thresholds), plus turn
    indices, texts and the previous/next turn for annotation context."""
    turns = parse_file(path, episode_id=episode_id)
    turns = merge_consecutive_same_speaker(turns)
    turns = [t for t in turns if not is_backchannel(t.text)]
    if len(turns) < 3:
        return [], 0, None
    vectors = build_tfidf_vectors([t.text for t in turns])
    incoming = [None] + [cosine_sim(vectors[i - 1], vectors[i]) for i in range(1, len(turns))]
    known = [s for s in incoming if s is not None]
    init_cutoff = sorted(known)[len(known) // 2]
    is_init = [True if i == 0 else incoming[i] <= init_cutoff for i in range(len(turns))]
    pairs = []
    for i, t in enumerate(turns):
        if not is_init[i]:
            continue
        j, resp = find_next_response(turns, i, from_speaker=t.speaker_id)
        if resp is None:
            continue
        prev = turns[i - 1] if i > 0 else None
        pairs.append({
            "episode_id": episode_id,
            "init_idx": i, "resp_idx": j,
            "initiator_speaker": t.speaker_id, "initiator_role": t.role,
            "initiator_gender": t.gender,
            "initiator_word_count": raw_word_count(t.text),
            "responder_word_count": raw_word_count(resp.text),
            "initiator_content_tokens": len(tokenize(t.text)),
            "responder_content_tokens": len(tokenize(resp.text)),
            "incoming_similarity": incoming[i] if i > 0 else np.nan,
            "init_cutoff": init_cutoff,
            "similarity": cosine_sim(vectors[i], vectors[j]),
            "initiator_start_sec": t.start,
            "prev_text": prev.text if prev else "",
            "prev_role": prev.role if prev else "",
            "init_text": t.text, "resp_text": resp.text,
        })
    if pairs:
        cutoff = sorted(p["similarity"] for p in pairs)[len(pairs) // 2]
        for p in pairs:
            p["followup_threshold"] = cutoff
            p["followed_up"] = int(p["similarity"] > cutoff)
    return pairs, len(turns), init_cutoff


def build_pairs_with_text(dyads=MIXED, cache=True):
    """Rebuild pairs for the given dyads and verify against the saved file."""
    cache_path = os.path.join(DATA, "mixed_pairs_with_text.csv")
    if cache and dyads == MIXED and os.path.exists(cache_path):
        return pd.read_csv(cache_path, keep_default_na=False, na_values=[""])
    eps = pd.read_csv(EPISODES_CSV)
    eps = eps[eps["dyad"].isin(dyads)]
    rows, ep_rows = [], []
    for _, r in eps.iterrows():
        path = os.path.join(TRANSCRIPT_DIR, f"{r['episode_id']}.txt")
        if not os.path.exists(path):
            continue
        pairs, n_turns, init_cut = rebuild_episode_pairs(r["episode_id"], path)
        for p in pairs:
            p["dyad"] = r["dyad"]
        rows.extend(pairs)
        ep_rows.append({"episode_id": r["episode_id"], "dyad": r["dyad"],
                        "n_turns_after_filtering": n_turns, "init_cutoff": init_cut})
    df = pd.DataFrame(rows)
    verify_against_original(df, dyads)
    if cache and dyads == MIXED:
        df.to_csv(cache_path, index=False)
        pd.DataFrame(ep_rows).to_csv(os.path.join(DATA, "mixed_episode_turn_counts.csv"), index=False)
    return df


def verify_against_original(rebuilt, dyads):
    orig = pd.read_csv(ORIG_TURNS)
    orig = orig[orig["dyad"].isin(dyads)].reset_index(drop=True)
    rb = rebuilt.reset_index(drop=True)
    assert len(orig) == len(rb), f"row count differs: {len(orig)} vs {len(rb)}"
    for col in ["episode_id", "initiator_role", "initiator_gender",
                "initiator_word_count", "responder_word_count", "followed_up"]:
        assert (orig[col].values == rb[col].values).all(), f"column {col} differs"
    assert np.allclose(orig["similarity"].values, rb["similarity"].values, atol=1e-12)
    print(f"[verify] rebuilt pairs match the saved Report 1 file ({len(rb)} rows)")
