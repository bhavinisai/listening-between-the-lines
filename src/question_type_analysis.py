#!/usr/bin/env python3
"""
Extract every question asked in a transcript, classify each one into a
question-type taxonomy with a fine-tuned XLM-R sequence classifier, and test
whether the distribution of question types the HOST asks differs across the
four host-gender x guest-gender dyads: MALE->MALE, MALE->FEMALE,
FEMALE->MALE, FEMALE->FEMALE (e.g. do male hosts ask female guests more
closed/personal questions and fewer open/professional/challenge questions
than they ask male guests -- and does that same pattern hold for female
hosts, or is it host-gender-specific?).

Taxonomy (6 classes, must match the fine-tuned classifier's label set):
    closed        - answerable with yes/no or a short factual reply
    open          - invites an elaborated, exploratory answer
    leading       - presupposes/steers toward a particular answer
    personal      - about the guest's private life, feelings, relationships
    professional  - about the guest's work, career, expertise
    challenge     - pushes back, asks the guest to defend/justify something

Question extraction:
    Turns are parsed with topic_control.py's transcript parser (handles the
    ".host_guest.txt" format where the host's SPEAKER_XX id has been
    replaced with their real name). Consecutive same-speaker lines are
    merged first so a question split across two diarization fragments
    ("What's your favorite" / "sport?") isn't lost. A turn is then split
    into sentences on [.?!] boundaries, and any sentence ending in "?" is
    kept as a question. This is a simple, dependency-free heuristic (no
    nltk/spacy) -- it depends on WhisperX having produced a "?" for the
    question, so ASR punctuation errors will cause some misses.

Both HOST- and GUEST-asked questions are extracted and labeled with
asker_role/asker_gender, but the dyad comparison (the point of this script)
only makes sense for questions the HOST asks -- so the statistical tests
below filter to asker_role == "HOST", then split by that episode's dyad
(host_gender -> guest_gender, from balanced_200_episodes.csv's "dyad" column).

Classifier:
    Loads a fine-tuned XLM-R (xlm-roberta) sequence-classification checkpoint
    via `transformers.AutoModelForSequenceClassification` /
    `AutoTokenizer` from --model_path. THIS REPO DOES NOT CURRENTLY CONTAIN
    A FINE-TUNED CHECKPOINT -- you must point --model_path at one you've
    trained (a local directory or a HF Hub repo id) before this script can
    run past --extract_only. See the --model_path help below.

    Requires `torch`, `transformers`, and `sentencepiece` (XLM-R's tokenizer
    needs sentencepiece). None of these are in requirements.txt yet --
    `pip install torch transformers sentencepiece` before running.

Input:
    - results/balanced_200_episodes.csv (episode_id, host_gender, guest_gender, dyad)
    - data/outputs/whisperx/{episode_id}.txt (diarized transcript)

Output:
    - results/question_classification.csv  (one row per extracted question, all
                                              asker roles; includes the "dyad" column)
    - results/question_type_by_dyad.csv     (type x dyad counts/proportions across
                                              all 4 dyads, HOST-asked questions only)
    - results/question_type_stats.csv       (omnibus chi-square on type x dyad,
                                              per-type chi-square vs dyad with
                                              Bonferroni correction, and a
                                              linear-probability regression per
                                              type on is_male_host x is_female_guest
                                              -- the interaction term tests whether
                                              the guest-gender effect on question
                                              type depends on the host's own gender)

Usage:
    # 1. Sanity-check extraction without needing a model/GPU:
    python src/question_type_analysis.py --extract_only

    # 2. Full run once you have a fine-tuned checkpoint:
    python src/question_type_analysis.py \
        --episodes results/balanced_200_episodes.csv \
        --transcript_dir data/outputs/whisperx \
        --model_path /path/to/finetuned-xlmr-question-type \
        --out_dir results
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from topic_control import parse_file, merge_consecutive_same_speaker

CLASS_LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]

# Split turn text into sentences on [.?!] boundaries. Keeps the punctuation
# on the preceding sentence (so we can check `.endswith("?")`).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def extract_questions_from_turns(turns):
    """turns: list of topic_control.Turn-like objects (has .role, .gender,
    .speaker_id, .start, .end, .text). Returns one dict per question."""
    questions = []
    for t in turns:
        for sent in split_sentences(t.text):
            if not sent.endswith("?"):
                continue
            questions.append({
                "asker_speaker": t.speaker_id,
                "asker_role": t.role,
                "asker_gender": t.gender,
                "start": t.start,
                "end": t.end,
                "text": sent,
            })
    return questions


def process_episode(episode_id, transcript_path):
    turns = parse_file(transcript_path, episode_id=episode_id)
    turns = merge_consecutive_same_speaker(turns)
    qs = extract_questions_from_turns(turns)
    for q in qs:
        q["episode_id"] = episode_id
    return qs


def load_classifier(model_path, labels, device):
    """Loads a fine-tuned XLM-R sequence-classification checkpoint. Returns
    (tokenizer, model, id2label) where id2label[i] is one of CLASS_LABELS."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: {}\n"
            "This script needs torch + transformers + sentencepiece for XLM-R "
            "inference. Install with:\n"
            "    pip install torch transformers sentencepiece\n"
            "(none of these are in requirements.txt yet).".format(e)
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    config_id2label = [
        str(model.config.id2label.get(i, model.config.id2label.get(str(i), ""))).strip().lower()
        for i in range(num_labels)
    ]

    if sorted(config_id2label) == sorted(l.lower() for l in labels):
        id2label = config_id2label
    else:
        # Checkpoint has no informative id2label (e.g. default "LABEL_0",
        # "LABEL_1", ...) -- fall back to the --labels order and trust that
        # it matches how the classifier head was fine-tuned.
        print(
            f"WARNING: model.config.id2label ({config_id2label}) doesn't match "
            f"the expected taxonomy {labels}. Falling back to positional "
            f"--labels order -- verify this matches the order the classifier "
            f"was fine-tuned with, or predictions will be mislabeled."
        )
        if num_labels != len(labels):
            raise SystemExit(
                f"Checkpoint has {num_labels} output classes but {len(labels)} "
                f"--labels were given; can't align them positionally."
            )
        id2label = [l.lower() for l in labels]

    return tokenizer, model, id2label


def classify_questions(texts, tokenizer, model, id2label, device, batch_size=32, max_length=128):
    import torch

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            preds.extend(id2label[i] for i in top_idx.cpu().tolist())
            confs.extend(top_prob.cpu().tolist())
    return preds, confs


def ols_with_stats(X, y):
    """Plain OLS via normal equations (linear probability model here, since
    the outcome is a 0/1 type indicator) -- classical SEs/t/p. No
    statsmodels/sklearn dependency, matching topic_control.py's helper."""
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
        "--model_path", default=None,
        help="Path or HF Hub id of the fine-tuned XLM-R sequence-classification "
             "checkpoint. Required unless --extract_only.",
    )
    ap.add_argument(
        "--labels", default=",".join(CLASS_LABELS),
        help="Comma-separated class order to fall back to if the checkpoint's "
             f"id2label doesn't already spell out {CLASS_LABELS}.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    ap.add_argument(
        "--extract_only", action="store_true",
        help="Only run question extraction (no classifier needed) and write "
             "results/question_classification.csv with predicted_type left blank. "
             "Useful for sanity-checking extraction before a checkpoint is ready.",
    )
    args = ap.parse_args()

    labels = [l.strip().lower() for l in args.labels.split(",") if l.strip()]
    if not args.extract_only and not args.model_path:
        raise SystemExit("--model_path is required (or pass --extract_only).")

    episodes = pd.read_csv(args.episodes)

    all_questions = []
    missing = 0
    for _, row in episodes.iterrows():
        episode_id = row["episode_id"]
        transcript_path = os.path.join(args.transcript_dir, f"{episode_id}.txt")
        if not os.path.exists(transcript_path):
            missing += 1
            continue
        qs = process_episode(episode_id, transcript_path)
        for q in qs:
            q["dyad"] = row["dyad"]
            q["host_gender"] = row["host_gender"]
            q["guest_gender"] = row["guest_gender"]
        all_questions.extend(qs)

    if missing:
        print(f"WARNING: {missing} episode(s) missing a transcript .txt, skipped")

    if not all_questions:
        print("WARNING: no questions extracted, nothing to write")
        return

    df = pd.DataFrame(all_questions)
    print(f"[OK] Extracted {len(df)} questions from {df['episode_id'].nunique()} episodes "
          f"({(df['asker_role'] == 'HOST').sum()} host-asked, "
          f"{(df['asker_role'] == 'GUEST').sum()} guest-asked)")

    os.makedirs(args.out_dir, exist_ok=True)
    class_path = os.path.join(args.out_dir, "question_classification.csv")

    if args.extract_only:
        df["predicted_type"] = pd.NA
        df["confidence"] = pd.NA
        df.to_csv(class_path, index=False)
        print(f"[OK] Wrote {class_path} (extraction only, no classification)")
        return

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[OK] Using device: {device}")

    tokenizer, model, id2label = load_classifier(args.model_path, labels, device)
    preds, confs = classify_questions(
        df["text"].tolist(), tokenizer, model, id2label, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    df["predicted_type"] = preds
    df["confidence"] = confs
    df.to_csv(class_path, index=False)
    print(f"[OK] Wrote {len(df)} classified questions to {class_path}")

    # --- Restrict to HOST-asked questions, broken out by the full 2x2 dyad
    # (MALE->MALE, MALE->FEMALE, FEMALE->MALE, FEMALE->FEMALE), not just
    # guest gender collapsed across host gender -- the question is whether
    # host gender and guest gender jointly shape what gets asked. ---
    host_q = df[df["asker_role"] == "HOST"].copy()
    if host_q.empty:
        print("WARNING: no host-asked questions found, skipping dyad tests")
        return

    dyad_order = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
    present_dyads = [d for d in dyad_order if d in host_q["dyad"].unique()]

    summary_path = os.path.join(args.out_dir, "question_type_by_dyad.csv")
    counts = pd.crosstab(host_q["predicted_type"], host_q["dyad"])[present_dyads]
    props = pd.crosstab(host_q["predicted_type"], host_q["dyad"], normalize="columns")[present_dyads]
    summary = counts.add_suffix("_n").join(props.add_suffix("_share")).reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote {summary_path}")
    print(summary)

    # --- Statistical tests ---
    stats_rows = []

    # 1. Omnibus chi-square: is question_type independent of dyad (all 4)?
    contingency = pd.crosstab(host_q["predicted_type"], host_q["dyad"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        stats_rows.append({
            "test": "chi_square_omnibus_type_by_dyad",
            "statistic": chi2, "p_value": chi2_p, "dof": dof,
            "notes": f"H0: question type independent of dyad (MM/MF/FM/FF); n={len(host_q)}",
        })

    # 2. Per-type test (this type vs all others) x dyad (4 categories), with
    #    Bonferroni-corrected significance across the 6 types.
    n_types = host_q["predicted_type"].nunique()
    for qtype in sorted(host_q["predicted_type"].unique()):
        is_type = (host_q["predicted_type"] == qtype)
        two_by_four = pd.crosstab(is_type, host_q["dyad"])
        if two_by_four.shape[0] == 2 and two_by_four.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(two_by_four)
            shares = props.loc[qtype] if qtype in props.index else None
            stats_rows.append({
                "test": f"chi_square_{qtype}_vs_rest_by_dyad",
                "statistic": chi2, "p_value": p, "dof": dof,
                "notes": (
                    f"bonferroni_alpha={0.05 / n_types:.4f} (n_types={n_types}); "
                    f"share of each dyad's questions that are '{qtype}': "
                    f"{shares.round(4).to_dict() if shares is not None else 'n/a'}"
                ),
            })

    # 3. Linear-probability regression per type: type_is_X ~ is_male_host +
    #    is_female_guest + their interaction. The interaction term is the
    #    key test here -- it asks whether the guest-gender effect on
    #    question type differs depending on the host's own gender (i.e.
    #    whether the 4 dyads collapse to two additive main effects, or
    #    whether e.g. male hosts and female hosts treat guest gender
    #    differently). Baseline (all dummies 0) is a female host with a
    #    male guest (FEMALE->MALE).
    host_q["is_male_host"] = (host_q["host_gender"] == "male").astype(float)
    host_q["is_female_guest"] = (host_q["guest_gender"] == "female").astype(float)
    host_q["host_x_guest"] = host_q["is_male_host"] * host_q["is_female_guest"]
    if len(host_q) > 6 and len(present_dyads) == 4:
        X = np.column_stack([
            np.ones(len(host_q)),
            host_q["is_male_host"].values,
            host_q["is_female_guest"].values,
            host_q["host_x_guest"].values,
        ])
        labels_out = ["intercept", "is_male_host", "is_female_guest", "is_male_host_x_is_female_guest"]
        for qtype in sorted(host_q["predicted_type"].unique()):
            y = (host_q["predicted_type"] == qtype).astype(float).values
            if y.std() == 0:
                continue
            coef, se, tvals, pvals = ols_with_stats(X, y)
            for label, c, s, tv, pv in zip(labels_out, coef, se, tvals, pvals):
                stats_rows.append({
                    "test": f"lpm_{qtype}_{label}",
                    "statistic": tv, "p_value": pv, "dof": len(host_q) - X.shape[1],
                    "notes": f"coef={c:.4f}, se={s:.4f} (outcome: P(type == '{qtype}'); baseline dyad=FEMALE->MALE)",
                })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(args.out_dir, "question_type_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Wrote statistical tests to {stats_path}")
    print(stats_df)


if __name__ == "__main__":
    main()
