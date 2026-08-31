"""
dialogue_act_classification.py

Classifies every turn in each episode's diarized transcript into one of 11
dialogue act categories, using the pretrained diwank/silicone-deberta-pair
model (DeBERTa fine-tuned on the SILICONE dataset).

This is an alternative/complement to the lexicon-based hedge/booster/polite/
directive counts in compute_bias_features.py — it uses full-sentence context
(previous turn + current turn) rather than keyword matching, which avoids
false positives like "actually" or "listen" used non-directively.

Labels produced (from the model card):
    acknowledge, answer, backchannel, reply_yes, exclaim,
    say, reply_no, hold, ask, intent, ask_yes_no

Install:
    pip install simpletransformers --break-system-packages

Usage:
    python dialogue_act_classification.py \
        --input_glob "data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json" \
        --balanced_csv results/features/balanced_200_episodes.csv \
        --out_dir results/dialogue_acts/ \
        --max_episodes 200
"""

import argparse
import glob
import json
import os

import pandas as pd

LABELS = [
    "acknowledge", "answer", "backchannel", "reply_yes", "exclaim",
    "say", "reply_no", "hold", "ask", "intent", "ask_yes_no",
]


def load_model(use_cuda: bool = False):
    """Lazy import + load so the script can be imported without requiring
    simpletransformers/torch until actually needed."""
    from simpletransformers.classification import ClassificationModel

    model = ClassificationModel(
        "deberta",
        "diwank/silicone-deberta-pair",
        use_cuda=use_cuda,
        # fp16 must be off: DeBERTa's disentangled-attention masking overflows
        # half precision ("value cannot be converted to type at::Half without overflow").
        args={"silent": True, "fp16": False},
    )
    return model


def load_episode_segments(json_path: str):
    """Return list of segment dicts with text/speaker/role/gender, in order."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "text": text,
            "speaker": seg.get("speaker", "UNKNOWN"),
            "role": (seg.get("speaker_role") or "UNKNOWN").upper(),
            "gender": (seg.get("gender") or "unknown").lower(),
        })
    return segments


def build_pairs(segments):
    """
    Build (previous_text, current_text) pairs as required by the model.
    Previous text is empty string for the very first turn.
    """
    pairs = []
    for i, seg in enumerate(segments):
        prev_text = segments[i - 1]["text"] if i > 0 else ""
        pairs.append([prev_text, seg["text"]])
    return pairs


def classify_episode(model, json_path: str, episode_id: str, batch_size: int = 32):
    """Run dialogue act classification over every turn in one episode."""
    segments = load_episode_segments(json_path)
    if not segments:
        return []

    pairs = build_pairs(segments)

    all_preds = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        preds, _ = model.predict(batch)
        all_preds.extend(preds)

    rows = []
    for seg, pred_idx in zip(segments, all_preds):
        rows.append({
            "episode_id": episode_id,
            "speaker": seg["speaker"],
            "role": seg["role"],
            "gender": seg["gender"],
            "text": seg["text"],
            "dialogue_act": LABELS[pred_idx],
        })
    return rows


def episode_id_from_path(path: str) -> str:
    """Matches the episode_id convention used in compute_bias_features.py and
    balanced_200_episodes.csv: full filename minus the .json extension."""
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def main(input_glob: str, balanced_csv: str, out_dir: str, max_episodes: int, use_cuda: bool, batch_size: int):
    os.makedirs(out_dir, exist_ok=True)

    bal = pd.read_csv(balanced_csv)
    balanced_ids = set(bal["episode_id"].astype(str))

    all_paths = sorted(glob.glob(input_glob))
    matched_paths = []
    for p in all_paths:
        eid = episode_id_from_path(p)
        if eid in balanced_ids:
            matched_paths.append((eid, p))
    matched_paths = matched_paths[:max_episodes]

    print(f"Found {len(all_paths)} total transcript files.")
    print(f"Matched {len(matched_paths)} against balanced_200_episodes.csv, "
          f"processing first {len(matched_paths)} (max_episodes={max_episodes}).\n")

    print("Loading model (diwank/silicone-deberta-pair)...")
    model = load_model(use_cuda=use_cuda)
    print("Model loaded.\n")

    all_rows = []
    for i, (episode_id, path) in enumerate(matched_paths, 1):
        try:
            rows = classify_episode(model, path, episode_id, batch_size=batch_size)
            all_rows.extend(rows)
            print(f"[{i}/{len(matched_paths)}] {episode_id}: {len(rows)} turns classified")
        except Exception as e:
            print(f"[{i}/{len(matched_paths)}] {episode_id}: FAILED ({e})")

    if not all_rows:
        print("No turns were classified. Check input paths and JSON structure.")
        return

    df = pd.DataFrame(all_rows)
    turns_out = os.path.join(out_dir, "dialogue_act_labels.csv")
    df.to_csv(turns_out, index=False)
    print(f"\nSaved per-turn dialogue act labels to {turns_out} ({len(df)} rows)")

    # -------------------------
    # Aggregate: per-episode, per-role dialogue act rate table
    # -------------------------
    act_rates = (
        df.groupby(["episode_id", "role"])["dialogue_act"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    rates_out = os.path.join(out_dir, "dialogue_act_rates_by_episode_role.csv")
    act_rates.to_csv(rates_out, index=False)
    print(f"Saved per-episode/role dialogue act rates to {rates_out}")

    # -------------------------
    # Aggregate: overall rate by gender (quick sanity-check summary)
    # -------------------------
    gender_summary = (
        df.groupby("gender")["dialogue_act"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print("\nDialogue act rate by gender (overall, sanity check):")
    print(gender_summary.round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="Glob for diarized WhisperX JSON files")
    ap.add_argument("--balanced_csv", required=True, help="Path to balanced_200_episodes.csv")
    ap.add_argument("--out_dir", default=".", help="Directory to write output CSVs to")
    ap.add_argument("--max_episodes", type=int, default=200, help="Max number of episodes to process")
    ap.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for model.predict()")
    args = ap.parse_args()
    main(args.input_glob, args.balanced_csv, args.out_dir, args.max_episodes, args.use_cuda, args.batch_size)