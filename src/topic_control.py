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
than female speakers' in mixed-gender dyads (MALE->FEMALE / FEMALE->MALE).

IMPORTANT: turn-pairs within the same episode are not independent observations
-- one chatty episode can contribute hundreds of pairs out of the ~100
episodes per gender in this dataset. Naive tests that ignore this (chi-square,
Welch's t-test) badly overstate significance, so this script reports those
only as descriptive references and uses episode-aware tests for inference:
  1. Chi-square / Welch's t-test on the raw/binarized outcome (NAIVE, kept
     for comparison only -- see their "notes" column).
  2. A mixed-effects (MixedLM) regression of similarity on initiator gender,
     controlling for initiator turn length and host/guest role, with random
     intercepts per episode and per speaker nested in episode.
  3. GEE logistic regression of the binarized follow-up outcome, clustered by
     episode (exchangeable correlation, robust SEs) -- the primary,
     confound-controlled significance test for the binary outcome.
  4. A variational-Bayes mixed logistic regression (BinomialBayesMixedGLM),
     the literal random-intercept analogue of #3, kept as a secondary
     cross-check -- its posterior SDs are known to run anti-conservative, so
     it is not the test of record.
  5. Episode-level paired t-test, Wilcoxon signed-rank, and a sign-flip
     permutation test, all model-free confirmations comparing male vs. female
     follow-up within the same ~100 episodes.
  6. A robustness check re-running the follow-up rate at a few fixed
     thresholds, to confirm the result isn't an artifact of one cutoff.

Input:
    - results/features/balanced_200_episodes.csv (episode_id, host_gender, guest_gender, dyad)
    - data/outputs/whisperx/{episode_id}.txt
      (diarized transcript, same "[start - end] SPEAKER (gender, ROLE): text"
      format parsed by interruption_matrix.py)

Output:
    - results/topic_control_turns.csv    (one row per initiation -> response pair)
    - results/topic_control_summary.csv  (follow-up rate by initiator gender,
                                           mixed-gender dyads only)
    - results/topic_control_stats.csv    (naive reference tests, mixed-effects/
                                           GEE/GLMM regressions, paired episode-
                                           level tests, and robustness checks)

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
        --episodes results/features/balanced_200_episodes.csv \
        --transcript_dir data/outputs/whisperx \
        --out_dir results
"""

import argparse
import math
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.regression.mixed_linear_model import MixedLM

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/features/balanced_200_episodes.csv")
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
    #    NAIVE reference only -- treats every turn-pair as an independent draw,
    #    when really they're clustered within ~100 episodes per gender (one
    #    episode can contribute hundreds of pairs). This overstates significance;
    #    see lpm_followedup_is_male_initiator below for the clustered equivalent.
    contingency = pd.crosstab(mixed["initiator_gender"], mixed["followed_up"])
    if contingency.shape[0] == 2 and contingency.shape[1] == 2:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        stats_rows.append({
            "test": "chi_square_followup_by_gender",
            "statistic": chi2, "p_value": chi2_p, "dof": dof,
            "notes": (
                "H0: follow-up (binary) is independent of initiator gender. "
                "NAIVE: does not account for episode clustering -- treat as "
                "descriptive only, not the significance test of record."
            ),
        })

    # 2. Welch's t-test on the raw continuous similarity score (no binarizing).
    #    Same clustering caveat as the chi-square above -- descriptive only.
    if len(male) > 1 and len(female) > 1:
        t_stat, t_p = stats.ttest_ind(male["similarity"], female["similarity"], equal_var=False)
        stats_rows.append({
            "test": "welch_ttest_similarity_by_gender",
            "statistic": t_stat, "p_value": t_p, "dof": float("nan"),
            "notes": (
                f"male mean sim={male['similarity'].mean():.4f} (n={len(male)}), "
                f"female mean sim={female['similarity'].mean():.4f} (n={len(female)}). "
                "NAIVE: does not account for episode clustering, see paired "
                "episode-level tests below."
            ),
        })

    # 3. Mixed-effects regression on the continuous similarity outcome,
    #    controlling for initiator turn length and host/guest role, with a
    #    random intercept per episode AND per speaker nested within episode.
    #    Turn-pairs within an episode aren't independent draws -- one chatty
    #    episode can contribute hundreds of pairs -- so this models that
    #    structure directly (more efficient than just correcting the SEs
    #    for it, which is what the now-removed cluster-robust OLS did).
    reg_df = mixed.dropna(subset=["similarity", "initiator_word_count"]).copy()
    reg_df["is_male_initiator"] = (reg_df["initiator_gender"] == "male").astype(float)
    reg_df["is_host_initiator"] = (reg_df["initiator_role"] == "HOST").astype(float)
    reg_df["speaker_key"] = reg_df["episode_id"].astype(str) + "::" + reg_df["initiator_speaker"].astype(str)
    labels = ["intercept", "is_male_initiator", "initiator_word_count", "is_host_initiator"]
    n_episodes = reg_df["episode_id"].nunique()
    n_speakers = reg_df["speaker_key"].nunique()

    if len(reg_df) > 4 and n_episodes > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixedlm_result = MixedLM.from_formula(
                "similarity ~ is_male_initiator + initiator_word_count + is_host_initiator",
                groups="episode_id",
                vc_formula={"speaker": "0 + C(speaker_key)"},
                data=reg_df,
            ).fit(reml=True)
        for label, name in zip(labels, mixedlm_result.fe_params.index):
            c, s = mixedlm_result.fe_params[name], mixedlm_result.bse_fe[name]
            tv, pv = mixedlm_result.tvalues[name], mixedlm_result.pvalues[name]
            stats_rows.append({
                "test": f"mixedlm_similarity_{label}",
                "statistic": tv, "p_value": pv, "dof": float("nan"),
                "notes": (
                    f"coef={c:.4f}, se={s:.4f} (outcome: outgoing similarity; "
                    f"mixed-effects model, random intercept per episode + per "
                    f"speaker nested in episode; {n_episodes} episodes, {n_speakers} speakers)"
                ),
            })

        # 3b. GEE logistic regression on the binarized follow-up outcome,
        #     clustered by episode (exchangeable working correlation, robust
        #     sandwich SEs) -- the clustering-aware, confound-controlled
        #     replacement for the naive chi-square test above, and the
        #     primary significance test for the binary outcome.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee_result = smf.gee(
                "followed_up ~ is_male_initiator + initiator_word_count + is_host_initiator",
                groups="episode_id",
                data=reg_df,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        for label, name in zip(labels, gee_result.params.index):
            c, s = gee_result.params[name], gee_result.bse[name]
            tv, pv = gee_result.tvalues[name], gee_result.pvalues[name]
            stats_rows.append({
                "test": f"gee_followedup_{label}",
                "statistic": tv, "p_value": pv, "dof": float("nan"),
                "notes": (
                    f"coef={c:.4f}, se={s:.4f} (outcome: followed_up 0/1; "
                    f"GEE, exchangeable correlation clustered by episode_id, "
                    f"{n_episodes} clusters)"
                ),
            })

        # 3c. Multilevel (variational Bayes) mixed logistic regression -- the
        #     literal random-intercept-per-episode-and-per-speaker analogue of
        #     the GEE model above, on a standardized word count for numerical
        #     stability. CAVEAT: on this data its posterior SDs came out ~2.4x
        #     tighter than GEE's robust SEs for the same coefficient, which
        #     matches a known failure mode of mean-field variational GLMM
        #     fitting (understates uncertainty). Treat gee_followedup_* above
        #     as the significance test of record; use this only to cross-check
        #     direction and rough magnitude.
        wc = reg_df["initiator_word_count"].astype(float)
        reg_df["word_count_z"] = (wc - wc.mean()) / wc.std()
        glmm_labels = ["intercept", "is_male_initiator", "word_count_z", "is_host_initiator"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glmm_result = BinomialBayesMixedGLM.from_formula(
                "followed_up ~ is_male_initiator + word_count_z + is_host_initiator",
                vc_formulas={"episode": "0 + C(episode_id)", "speaker": "0 + C(speaker_key)"},
                data=reg_df,
            ).fit_vb()
        for label, c, s in zip(glmm_labels, glmm_result.fe_mean, glmm_result.fe_sd):
            z = c / s
            pv = 2 * (1 - stats.norm.cdf(abs(z)))
            stats_rows.append({
                "test": f"glmm_followedup_{label}",
                "statistic": z, "p_value": pv, "dof": float("nan"),
                "notes": (
                    f"coef={c:.4f} (log-odds), sd={s:.4f} (outcome: followed_up 0/1; "
                    "variational Bayes GLMM, random intercept per episode + per "
                    "speaker nested in episode -- SDs likely anti-conservative, "
                    "see caveat above; not the test of record)"
                ),
            })

    # 3d. Episode-level paired tests: collapse to one follow-up rate and one
    #     mean similarity per (episode, initiator_gender), then compare male
    #     vs. female within each episode. Model-free confirmation that the
    #     gap isn't an artifact of any of the models above -- n here is
    #     genuinely ~100 episodes, not ~6,000 turn-pairs. The permutation test
    #     adds a second, assumption-free check: it randomly relabels which
    #     side of each episode's pair counts as "male" vs. "female" thousands
    #     of times to build a null distribution for the paired mean difference,
    #     then reports how extreme the real gap is against pure chance.
    ep_gender = (
        mixed.groupby(["episode_id", "initiator_gender"])
        .agg(followup_rate=("followed_up", "mean"), mean_similarity=("similarity", "mean"))
        .reset_index()
    )
    rng = np.random.default_rng(42)
    n_perm = 10000
    for value_col, test_stub in [
        ("followup_rate", "episode_followup_rate"),
        ("mean_similarity", "episode_similarity"),
    ]:
        pivoted = ep_gender.pivot(index="episode_id", columns="initiator_gender", values=value_col)
        pivoted = pivoted.dropna(subset=["male", "female"])
        if len(pivoted) > 1:
            t_stat, t_p = stats.ttest_rel(pivoted["male"], pivoted["female"])
            stats_rows.append({
                "test": f"paired_ttest_{test_stub}",
                "statistic": t_stat, "p_value": t_p, "dof": len(pivoted) - 1,
                "notes": (
                    f"male mean={pivoted['male'].mean():.4f}, "
                    f"female mean={pivoted['female'].mean():.4f} "
                    f"(n={len(pivoted)} episodes with both initiator genders)"
                ),
            })
            w_stat, w_p = stats.wilcoxon(pivoted["male"], pivoted["female"])
            stats_rows.append({
                "test": f"paired_wilcoxon_{test_stub}",
                "statistic": w_stat, "p_value": w_p, "dof": float("nan"),
                "notes": f"non-parametric confirmation of paired_ttest_{test_stub}",
            })

            diffs = (pivoted["male"] - pivoted["female"]).values
            observed = diffs.mean()
            signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
            null_means = (diffs[None, :] * signs).mean(axis=1)
            perm_p = (np.sum(np.abs(null_means) >= abs(observed)) + 1) / (n_perm + 1)
            stats_rows.append({
                "test": f"permutation_test_{test_stub}",
                "statistic": observed, "p_value": perm_p, "dof": float("nan"),
                "notes": (
                    f"observed male-female diff={observed:.4f}; two-sided p from "
                    f"{n_perm} sign-flip permutations of the {len(diffs)} paired "
                    "episodes (H0: which side of each episode's pair is labeled "
                    "\"male\" vs. \"female\" is arbitrary)"
                ),
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
