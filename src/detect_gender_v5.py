#!/usr/bin/env python3
"""
detect_gender_v4.py

Detect speaker gender using a hybrid approach:
  1. Match each diarized speaker against a known host library (voice embeddings)
  2. If no match found, fall back to inaSpeechSegmenter acoustic detection

Outputs:
1) updated JSON with gender fields
2) clean TXT transcript regenerated from JSON

Usage:
    python src/detect_gender_v5.py \
        --audio data/raw_audio/ep_093.wav \
        --input_json data/outputs/whisperx/ep_093_whisperx_diarized.json \
        --output_json data/outputs/whisperx/ep_093_whisperx_diarized.gender.json \
        --output_txt data/outputs/whisperx/ep_093_whisperx_diarized.gender.txt \
        --speaker_library data/speaker_library.json
        --match_threshold 0.70

Optional:
    --match_threshold 0.80   Cosine similarity threshold for a confident match (default: 0.80)
    --min_confidence  0.60   inaSpeechSegmenter confidence threshold (default: 0.60)
    --speaker_key     speaker_raw  Which key to read speaker ID from in each segment
"""

from __future__ import annotations

import argparse
import json
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ts(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_txt(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def get_speaker_id(seg: Dict[str, Any], preferred_key: Optional[str] = None) -> str:
    if preferred_key:
        val = seg.get(preferred_key)
        if val is not None:
            return str(val)
    val = seg.get("speaker_raw")
    if val is not None:
        return str(val)
    val = seg.get("speaker")
    if val is not None:
        return str(val)
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Step 1: Extract a combined audio clip per speaker using diarized timestamps
# ---------------------------------------------------------------------------

def extract_speaker_audio(
    audio_path: Path,
    segments: List[Dict[str, Any]],
    speaker_id: str,
    speaker_key: Optional[str],
    max_duration: float = 120.0,
) -> Optional[Path]:
    """
    Concatenate up to max_duration seconds of a speaker's segments into a
    temporary wav file for embedding computation.

    Returns path to the temp file, or None if no segments found.
    """
    speaker_segments = []
    total = 0.0

    for seg in segments:
        if get_speaker_id(seg, speaker_key) != speaker_id:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = end - start
        if dur <= 0.5:  # skip very short segments
            continue
        speaker_segments.append((start, end))
        total += dur
        if total >= max_duration:
            break

    if not speaker_segments:
        return None

    # Use ffmpeg to extract and concatenate segments into one temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    # Build ffmpeg filter for concatenation
    inputs = []
    filter_parts = []
    for i, (start, end) in enumerate(speaker_segments):
        inputs += ["-ss", str(start), "-to", str(end), "-i", str(audio_path)]
        filter_parts.append(f"[{i}:a]")

    n = len(speaker_segments)
    filter_complex = "".join(filter_parts) + f"concat=n={n}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(tmp_path)
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg failed for {speaker_id}")
        return None

    return tmp_path


# ---------------------------------------------------------------------------
# Step 2: Compute embedding for a wav file
# ---------------------------------------------------------------------------

def load_embedding_model():
    try:
        from pyannote.audio import Model, Inference
    except ImportError:
        raise RuntimeError(
            "pyannote.audio is not installed.\n"
            "Install with: pip install pyannote.audio torch torchaudio"
        )
    import os
    token = os.environ.get("HF_TOKEN")
    print("Loading speaker embedding model...")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    inference = Inference(model, window="whole")
    return inference


def compute_embedding(inference, audio_path: Path) -> Optional[np.ndarray]:
    try:
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)

        audio_input = {"waveform": waveform, "sample_rate": 16000}
        embedding = inference(audio_input)
        vec = np.array(embedding).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        print(f"  ERROR computing embedding: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 3: Match embedding against speaker library
# ---------------------------------------------------------------------------

def load_speaker_library(library_path: Path) -> List[Dict[str, Any]]:
    data = load_json(library_path)
    speakers = data.get("speakers", [])
    # Convert embedding lists back to numpy arrays
    for entry in speakers:
        entry["embedding"] = np.array(entry["embedding"])
    return speakers


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are already L2 normalised


def match_speaker(
    embedding: np.ndarray,
    library: List[Dict[str, Any]],
    threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Compare embedding against all library entries.
    Returns the best match if above threshold, else None.
    """
    scores = []
    for entry in library:
        score = cosine_similarity(embedding, entry["embedding"])
        scores.append((score, entry))

    # Sort by score descending and print all
    scores.sort(key=lambda x: x[0], reverse=True)
    print(f"  Similarity scores against library:")
    for score, entry in scores:
        flag = " ← best" if entry == scores[0][1] else ""
        print(f"    {entry['name']:<20} {score:.4f}{flag}")

    best_score, best_entry = scores[0]

    if best_score >= threshold:
        return {
            "name": best_entry["name"],
            "gender": best_entry["gender"],
            "similarity": round(best_score, 4),
            "source": "library",
            "confidence": 1.0,
            "needs_review": False,
        }

    print(f"  No library match (best similarity: {best_score:.4f} < threshold {threshold})")
    return None


# ---------------------------------------------------------------------------
# Step 4: Acoustic fallback via inaSpeechSegmenter
# ---------------------------------------------------------------------------

def detect_gender_inaspeech(audio_path: Path) -> List[Dict[str, Any]]:
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError:
        raise RuntimeError(
            "inaSpeechSegmenter is not installed.\n"
            "Install with: pip install inaSpeechSegmenter tensorflow"
        )
    print("  Loading inaSpeechSegmenter...")
    seg = Segmenter()
    raw_segments = seg(str(audio_path))
    return [
        {"start": float(s), "end": float(e), "gender": lbl}
        for lbl, s, e in raw_segments
        if lbl in {"male", "female"}
    ]


def overlap_duration(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def acoustic_gender(
    speaker_audio_path: Path,
    min_confidence: float,
) -> Dict[str, Any]:
    """
    Run inaSpeechSegmenter on an already-extracted speaker audio clip
    and return gender info.
    """
    gender_segments = detect_gender_inaspeech(speaker_audio_path)

    male_time = sum(g["end"] - g["start"] for g in gender_segments if g["gender"] == "male")
    female_time = sum(g["end"] - g["start"] for g in gender_segments if g["gender"] == "female")
    total = male_time + female_time

    if total <= 0:
        return {
            "gender": "unknown",
            "source": "acoustic",
            "confidence": 0.0,
            "male_time": 0.0,
            "female_time": 0.0,
            "needs_review": True,
        }

    if male_time >= female_time:
        confidence = male_time / total
        label = "male" if confidence >= min_confidence else "unknown"
    else:
        confidence = female_time / total
        label = "female" if confidence >= min_confidence else "unknown"

    return {
        "gender": label,
        "source": "acoustic",
        "confidence": round(confidence, 4),
        "male_time": round(male_time, 3),
        "female_time": round(female_time, 3),
        "needs_review": label == "unknown",
    }


# ---------------------------------------------------------------------------
# Step 5: Apply results to JSON and build TXT
# ---------------------------------------------------------------------------

def apply_gender_to_json(
    data: Dict[str, Any],
    speaker_gender: Dict[str, Dict[str, Any]],
    speaker_key: Optional[str],
) -> Dict[str, Any]:
    for seg in data.get("segments", []):
        speaker = get_speaker_id(seg, speaker_key)
        info = speaker_gender.get(speaker, {
            "gender": "unknown",
            "source": "none",
            "confidence": 0.0,
            "needs_review": True,
        })
        seg["gender"] = info["gender"]
        seg["gender_source"] = info.get("source", "none")
        seg["gender_confidence"] = info.get("confidence", 0.0)
        if info.get("name"):
            seg["speaker_name"] = info["name"]

    data["speaker_gender_mapping"] = speaker_gender
    return data


def build_txt_lines(
    data: Dict[str, Any],
    speaker_key: Optional[str],
) -> List[str]:
    lines = []
    for seg in data.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = get_speaker_id(seg, speaker_key)
        name = seg.get("speaker_name", speaker)
        gender = seg.get("gender", "unknown")
        source = seg.get("gender_source", "none")
        tag = gender

        lines.append(f"[{ts(start)} - {ts(end)}] {name} ({tag}): {text}")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid gender detection: speaker library match + acoustic fallback"
    )
    ap.add_argument("--audio", required=True, help="Path to episode audio (.wav)")
    ap.add_argument("--input_json", required=True, help="WhisperX diarized JSON")
    ap.add_argument("--output_json", required=True, help="Output JSON with gender labels")
    ap.add_argument("--output_txt", required=True, help="Output TXT transcript")
    ap.add_argument("--speaker_library", required=True, help="Speaker library JSON from build_speaker_library.py")
    ap.add_argument("--match_threshold", type=float, default=0.70,
                    help="Cosine similarity threshold for a library match (default: 0.70)")
    ap.add_argument("--min_confidence", type=float, default=0.60,
                    help="inaSpeechSegmenter confidence threshold (default: 0.60)")
    ap.add_argument("--speaker_key", default=None,
                    help="Preferred speaker key in each segment (e.g. speaker_raw)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    output_txt_path = Path(args.output_txt)
    library_path = Path(args.speaker_library)

    for p in [audio_path, input_json_path, library_path]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    print(f"Loading WhisperX JSON: {input_json_path}")
    data = load_json(input_json_path)
    segments = data.get("segments", [])
    if not segments:
        raise ValueError("Input JSON has no segments.")

    # Get unique speakers
    speaker_ids = sorted(set(
        get_speaker_id(seg, args.speaker_key)
        for seg in segments
        if get_speaker_id(seg, args.speaker_key) != "UNKNOWN"
    ))
    print(f"Speakers found: {speaker_ids}")

    # Load library and embedding model
    library = load_speaker_library(library_path)
    print(f"Library has {len(library)} enrolled speakers: {[e['name'] for e in library]}")
    inference = load_embedding_model()

    speaker_gender: Dict[str, Dict[str, Any]] = {}
    temp_files = []

    for speaker_id in speaker_ids:
        print(f"\nProcessing {speaker_id}...")

        # Extract audio for this speaker
        tmp_audio = extract_speaker_audio(
            audio_path, segments, speaker_id, args.speaker_key
        )
        if tmp_audio is None:
            print(f"  No audio extracted for {speaker_id} — marking unknown")
            speaker_gender[speaker_id] = {
                "gender": "unknown", "source": "none",
                "confidence": 0.0, "needs_review": True,
            }
            continue
        temp_files.append(tmp_audio)

        # Compute embedding
        print(f"  Computing embedding...")
        embedding = compute_embedding(inference, tmp_audio)

        if embedding is not None:
            # Try library match first
            match = match_speaker(embedding, library, args.match_threshold)
            if match:
                print(f"  Matched: {match['name']} ({match['gender']}) — similarity: {match['similarity']}")
                speaker_gender[speaker_id] = match
            else:
                # Fall back to acoustic
                print(f"  Falling back to acoustic detection...")
                result = acoustic_gender(tmp_audio, args.min_confidence)
                print(f"  Acoustic result: {result['gender']} (conf={result['confidence']})")
                speaker_gender[speaker_id] = result
        else:
            # Embedding failed — go straight to acoustic
            print(f"  Embedding failed, falling back to acoustic detection...")
            result = acoustic_gender(tmp_audio, args.min_confidence)
            print(f"  Acoustic result: {result['gender']} (conf={result['confidence']})")
            speaker_gender[speaker_id] = result

    # Apply to JSON and save
    updated = apply_gender_to_json(data, speaker_gender, args.speaker_key)
    txt_lines = build_txt_lines(updated, args.speaker_key)
    save_json(output_json_path, updated)
    save_txt(output_txt_path, txt_lines)

    # Clean up temp files
    for f in temp_files:
        try:
            f.unlink()
        except Exception:
            pass

    # Summary
    print(f"\n{'='*50}")
    print("Speaker gender summary:")
    for speaker, info in sorted(speaker_gender.items()):
        name = info.get("name", speaker)
        flag = " ⚠ needs_review" if info.get("needs_review") else ""
        print(f"  {speaker} → {name} | {info['gender']} | source={info['source']} | conf={info.get('confidence', 0)}{flag}")

    needs_review = [s for s, i in speaker_gender.items() if i.get("needs_review")]
    if needs_review:
        print(f"\n⚠  These speakers need manual review: {needs_review}")

    print(f"\nDone.")
    print(f"JSON: {output_json_path}")
    print(f"TXT:  {output_txt_path}")


if __name__ == "__main__":
    main()
