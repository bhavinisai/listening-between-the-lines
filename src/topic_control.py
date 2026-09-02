#!/usr/bin/env python3
"""
Measure topic control: does a speaker's turn get "followed up" by the other
speaker's next substantive turn? Uses TF-IDF cosine similarity between
consecutive turns as a proxy for topical overlap.

A turn counts as a "topic initiation" only if it diverges from what was just
said (low similarity to the immediately preceding turn) -- not every turn,
as in the earlier version of this script. The strength of the follow-up is
the (continuous) similarity between that initiation and the next turn from
the other speaker.

We then test whether male speakers' topic introductions are followed up more
than female speakers' in mixed-gender dyads (MALE->FEMALE / FEMALE->MALE),
using:
  1. A chi-square test on the binarized follow-up outcome.
  2. A Welch's t-test on the raw (non-binarized) similarity scores.
  3. An OLS regression of similarity on initiator gender, controlling for
     initiator turn length and host/guest role, so a raw gender gap can't
     be explained away by men simply talking longer or being the host.
  4. A robustness check re-running the follow-up rate at a few fixed
     thresholds, to confirm the result isn't an artifact of one cutoff.

Input:
    - results/balanced_200_episodes.csv   (episode_id, host_gender, guest_gender, dyad)
    - data/outputs/whisperx/{episode_id}.txt
      (diarized transcript, same "[start - end] SPEAKER (gender, ROLE): text"
      format parsed by interruption_matrix.py)

Output:
    - results/topic_control_turns.csv    (one row per initiation -> response pair)
    - results/topic_control_summary.csv  (follow-up rate by initiator gender,
                                           mixed-gender dyads only)
    - results/topic_control_stats.csv    (chi-square, t-test, regression,
                                           and robustness-check results)

Thresholds:
    --init_threshold  cutoff on incoming similarity below which a turn counts
                       as a new-topic initiation. Default: 'episode_median'
                       (that episode's own median incoming-similarity value).
    --threshold        cutoff on outgoing similarity above which a response
                       counts as "followed up". Default: 'episode_median'.
    Both accept a fixed float instead, e.g. --threshold 0.15.

Backchannel handling: pure backchannel turns (from interruption_matrix.py's
BACKCHANNEL_RE, e.g. "yeah", "right", "mhm") are dropped from the turn
sequence before pairing, so a filler reply never counts as either a topic
initiation or a follow-up response.

Usage:
    python src/topic_control.py \
        --episodes results/balanced_200_episodes.csv \
        --transcript_dir data/outputs/whisperx \
        --out_dir results
"""

import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from interruption_matrix import Turn, is_backchannel, ts_to_seconds

WORD_RE = re.compile(r"[A-Za-z']+")

# interruption_matrix.py's LINE_RE requires speaker == "SPEAKER_\d+", but in
# these *.host_guest.txt transcripts the host has already been relabeled with
# their real name (e.g. "Raj Shamani") while the guest stays "SPEAKER_01" -
# so that regex silently drops every host line. Use a looser speaker field
# here instead of touching the shared parser.
LINE_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s+"
    r"(?P<speaker>.+?)\s+\((?P<gender>[^,]+),\s*(?P<role>[^)]+)\):\s*(?P<text>.*)$"
)

# Small custom stopword list (no sklearn/nltk dependency in this repo).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "because", "as", "of",
    "to", "in", "on", "at", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "i",
    "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "am", "not", "no", "yes", "just", "like", "really",
    "actually", "um", "uh", "yeah", "okay", "ok", "well", "very", "much",
    "also",
}


def parse_file(path, episode_id):
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            start = ts_to_seconds(m.group("start"))
            end = ts_to_seconds(m.group("end"))
            if end < start:
                start, end = end, start
            turns.append(Turn(
                episode_id=episode_id,
                speaker_id=m.group("speaker").strip(),
                gender=m.group("gender").strip().lower(),
                role=m.group("role").strip().upper(),
                start=float(start),
                end=float(end),
                text=m.group("text").strip(),
            ))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def raw_word_count(text):
    return len(WORD_RE.findall(text))


def tokenize(text, ngram_max=2):
    """Unigrams (minus stopwords) plus bigrams, so phrases like
    'climate change' contribute a matchable token, not just two loose words."""
    words = [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]
    tokens = list(words)
    if ngram_max >= 2:
        tokens += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return tokens


def merge_consecutive_same_speaker(turns):
    """Collapse consecutive lines from the same speaker (diarization
    fragments a single turn into several lines) into one turn."""
    merged = []
    for t in turns:
        if merged and merged[-1].speaker_id == t.speaker_id:
            prev = merged[-1]
            merged[-1] = Turn(
                episode_id=prev.episode_id,
                speaker_id=prev.speaker_id,
                gender=prev.gender,
                role=prev.role,
                start=prev.start,
                end=t.end,
                text=(prev.text + " " + t.text).strip(),
            )
        else:
            merged.append(t)
    return merged


def build_tfidf_vectors(docs):
    """Sklearn-style smooth-idf TF-IDF + L2 normalization, implemented
    directly (no scikit-learn dependency in this repo's requirements.txt)."""
    n_docs = len(docs)
    tokenized = [tokenize(d) for d in docs]

    df = defaultdict(int)
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    idf = {term: math.log((1 + n_docs) / (1 + d)) + 1 for term, d in df.items()}

    vectors = []
    for tokens in tokenized:
        tf = defaultdict(int)
        for term in tokens:
            tf[term] += 1
        vec = {term: count * idf[term] for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {term: v / norm for term, v in vec.items()}
        vectors.append(vec)
    return vectors


def cosine_sim(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(v * vec_b.get(term, 0.0) for term, v in vec_a.items())


def find_next_response(turns, start_idx, from_speaker):
    """First turn after start_idx spoken by someone other than from_speaker."""
    for j in range(start_idx + 1, len(turns)):
        if turns[j].speaker_id != from_speaker:
            return j, turns[j]
    return None, None


def process_episode(episode_id, transcript_path, followup_threshold_mode, init_threshold_mode):
    turns = parse_file(transcript_path, episode_id=episode_id)
    turns = merge_consecutive_same_speaker(turns)
    turns = [t for t in turns if not is_backchannel(t.text)]

    if len(turns) < 3:
        return []

    vectors = build_tfidf_vectors([t.text for t in turns])
    word_counts = [raw_word_count(t.text) for t in turns]

    # Incoming similarity: how similar is this turn to the turn right before
    # it? Low incoming similarity = the speaker likely shifted to something
    # new, i.e. an actual topic initiation (rather than treating every turn
    # as an "initiation," which the earlier version of this script did).
    incoming_sim = [None] * len(turns)
    for i in range(1, len(turns)):
        incoming_sim[i] = cosine_sim(vectors[i - 1], vectors[i])

    if init_threshold_mode == "episode_median":
        known = [s for s in incoming_sim if s is not None]
        init_cutoff = sorted(known)[len(known) // 2] if known else 0.0
    else:
        init_cutoff = init_threshold_mode

    is_initiation = [
        True if i == 0 else incoming_sim[i] <= init_cutoff
        for i in range(len(turns))
    ]

    pairs = []
    for i, t in enumerate(turns):
        if not is_initiation[i]:
            continue
        j, response = find_next_response(turns, i, from_speaker=t.speaker_id)
        if response is None:
            continue
        pairs.append({
            "episode_id": episode_id,
            "initiator_speaker": t.speaker_id,
            "initiator_role": t.role,
            "initiator_gender": t.gender,
            "initiator_word_count": word_counts[i],
            "responder_speaker": response.speaker_id,
            "responder_role": response.role,
            "responder_gender": response.gender,
            "responder_word_count": word_counts[j],
            "incoming_similarity": incoming_sim[i] if i > 0 else float("nan"),
            "similarity": cosine_sim(vectors[i], vectors[j]),  # outgoing = follow-up strength
        })

    if not pairs:
        return pairs

    if followup_threshold_mode == "episode_median":
        sims = sorted(p["similarity"] for p in pairs)
        cutoff = sims[len(sims) // 2]
    else:
        cutoff = followup_threshold_mode

    for p in pairs:
        p["followup_threshold"] = cutoff
        p["followed_up"] = int(p["similarity"] > cutoff)

    return pairs


def ols_with_stats(X, y):
    """Plain OLS via the normal equations, with classical SEs, t-stats and
    two-tailed p-values. No statsmodels/sklearn dependency in this repo."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))
    return beta, se, tvals, pvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/balanced_200_episodes.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument(
        "--threshold", default="episode_median",
        help="Follow-up cutoff on outgoing similarity: 'episode_median' (default) "
             "or a fixed float, e.g. 0.15.",
    )
    ap.add_argument(
        "--init_threshold", default="episode_median",
        help="Topic-initiation cutoff on incoming similarity: 'episode_median' "
             "(default) or a fixed float. A turn at/below this cutoff counts as "
             "a new-topic initiation.",
    )
    ap.add_argument(
        "--extra_thresholds", default="0.05,0.10,0.20",
        help="Comma-separated fixed follow-up thresholds to re-check the gender "
             "gap against, as a robustness check on top of --threshold.",
    )
    args = ap.parse_args()

    def parse_threshold(val):
        try:
            return float(val)
        except ValueError:
            return val

    followup_threshold_mode = parse_threshold(args.threshold)
    init_threshold_mode = parse_threshold(args.init_threshold)
    extra_thresholds = [float(x) for x in args.extra_thresholds.split(",") if x.strip()]

    episodes = pd.read_csv(args.episodes)

    all_pairs = []
    missing = 0
    for _, row in episodes.iterrows():
        episode_id = row["episode_id"]
        transcript_path = os.path.join(args.transcript_dir, f"{episode_id}.txt")
        if not os.path.exists(transcript_path):
            missing += 1
            continue
        all_pairs.extend(
            process_episode(episode_id, transcript_path, followup_threshold_mode, init_threshold_mode)
        )

    if missing:
        print(f"WARNING: {missing} episode(s) missing a transcript .txt, skipped")

    os.makedirs(args.out_dir, exist_ok=True)
    turns_path = os.path.join(args.out_dir, "topic_control_turns.csv")
    summary_path = os.path.join(args.out_dir, "topic_control_summary.csv")
    stats_path = os.path.join(args.out_dir, "topic_control_stats.csv")

    if not all_pairs:
        print("WARNING: no initiation -> response pairs found, nothing to write")
        return

    turns_df = pd.DataFrame(all_pairs)
    turns_df.to_csv(turns_path, index=False)
    print(f"[OK] Wrote {len(turns_df)} initiation -> response pairs to {turns_path}")

    dyads = episodes.set_index("episode_id")["dyad"]
    turns_df["dyad"] = turns_df["episode_id"].map(dyads)
    mixed = turns_df[turns_df["dyad"].isin(["MALE->FEMALE", "FEMALE->MALE"])].copy()

    if mixed.empty:
        print("WARNING: no mixed-gender initiation -> response pairs found, nothing to summarize")
        return

    # --- Descriptive summary: follow-up rate by initiator gender ---
    summary = (
        mixed.groupby("initiator_gender")["followed_up"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "followup_rate", "count": "n_initiations"})
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote summary to {summary_path}")
    print(summary)

    # --- Statistical tests ---
    stats_rows = []
    male = mixed[mixed["initiator_gender"] == "male"]
    female = mixed[mixed["initiator_gender"] == "female"]

    # 1. Chi-square: is the binary follow-up outcome independent of initiator gender?
    contingency = pd.crosstab(mixed["initiator_gender"], mixed["followed_up"])
    if contingency.shape[0] == 2 and contingency.shape[1] == 2:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        stats_rows.append({
            "test": "chi_square_followup_by_gender",
            "statistic": chi2, "p_value": chi2_p, "dof": dof,
            "notes": "H0: follow-up (binary) is independent of initiator gender",
        })

    # 2. Welch's t-test on the raw continuous similarity score (no binarizing).
    if len(male) > 1 and len(female) > 1:
        t_stat, t_p = stats.ttest_ind(male["similarity"], female["similarity"], equal_var=False)
        stats_rows.append({
            "test": "welch_ttest_similarity_by_gender",
            "statistic": t_stat, "p_value": t_p, "dof": float("nan"),
            "notes": (
                f"male mean sim={male['similarity'].mean():.4f} (n={len(male)}), "
                f"female mean sim={female['similarity'].mean():.4f} (n={len(female)})"
            ),
        })

    # 3. OLS regression, controlling for initiator turn length and host/guest
    #    role, so a raw gender gap isn't just men talking longer or being host.
    reg_df = mixed.dropna(subset=["similarity", "initiator_word_count"]).copy()
    reg_df["is_male_initiator"] = (reg_df["initiator_gender"] == "male").astype(float)
    reg_df["is_host_initiator"] = (reg_df["initiator_role"] == "HOST").astype(float)
    if len(reg_df) > 4:
        X = np.column_stack([
            np.ones(len(reg_df)),
            reg_df["is_male_initiator"].values,
            reg_df["initiator_word_count"].values.astype(float),
            reg_df["is_host_initiator"].values,
        ])
        y = reg_df["similarity"].values.astype(float)
        coef, se, tvals, pvals = ols_with_stats(X, y)
        labels = ["intercept", "is_male_initiator", "initiator_word_count", "is_host_initiator"]
        for label, c, s, tv, pv in zip(labels, coef, se, tvals, pvals):
            stats_rows.append({
                "test": f"ols_similarity_{label}",
                "statistic": tv, "p_value": pv, "dof": len(reg_df) - X.shape[1],
                "notes": f"coef={c:.4f}, se={s:.4f} (outcome: outgoing similarity)",
            })

    # 4. Robustness check: does the gender gap hold across other fixed thresholds?
    for thr in extra_thresholds:
        followed = (mixed["similarity"] > thr).astype(int)
        rate_by_gender = followed.groupby(mixed["initiator_gender"]).mean()
        note = ", ".join(f"{g}={r:.3f}" for g, r in rate_by_gender.items())
        stats_rows.append({
            "test": f"robustness_threshold_{thr}",
            "statistic": float("nan"), "p_value": float("nan"), "dof": float("nan"),
            "notes": f"follow-up rate by gender @ threshold {thr}: {note}",
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Wrote statistical tests to {stats_path}")
    print(stats_df)


if __name__ == "__main__":
    main()
