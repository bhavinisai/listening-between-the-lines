#!/usr/bin/env python3
"""
Detect speaker gender using voice analysis and update transcript JSON/TXT files.

Uses inaSpeechSegmenter for gender detection from audio.

Install:
  pip install inaSpeechSegmenter tensorflow

Usage:
  python src/detect_gender_v3.py --audio data/raw_audio/ep_002.wav --input_json data/outputs/whisperx/ep_002_whisperx_diarized.json --output_json data/outputs/whisperx/ep_002_whisperx_diarized.gender.json --output_txt data/outputs/whisperx/ep_002_whisperx_diarized.gender.txt

"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(r'^\[(?P<ts>[^\]]+)\]\s+(?P<label>[^:]+):\s*(?P<text>.*)$')
SPEAKER_ID_RE = re.compile(r'(SPEAKER_\d+)')


def detect_gender_inaspeech(audio_path: Path):
    """
    Use inaSpeechSegmenter to detect gender from audio.
    Returns list of dicts: [{'start': float, 'end': float, 'gender': 'male'|'female'}]
    """
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError:
        print("Error: inaSpeechSegmenter not installed")
        print("Install with: pip install inaSpeechSegmenter tensorflow")
        return None

    print("Loading gender detection model...")
    seg = Segmenter()

    print(f"Analyzing audio: {audio_path}")
    segmentation = seg(str(audio_path))

    gender_segments = []
    for label, start, end in segmentation:
        if label in {"male", "female"}:
            gender_segments.append({
                "start": float(start),
                "end": float(end),
                "gender": label
            })

    return gender_segments


def overlap_duration(a_start, a_end, b_start, b_end):
    """Calculate overlap duration between two time segments."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_gender_to_speakers(segments, gender_segments, threshold=0.60):
    """
    Assign gender to each speaker based on overlap with gender segments.
    Returns dict: {speaker_raw: gender}
    """
    all_speakers = set()
    speaker_gender_time = defaultdict(lambda: {"male": 0.0, "female": 0.0})

    for seg in segments:
        speaker = seg.get("speaker_raw", "Unknown")
        if speaker != "Unknown":
            all_speakers.add(speaker)

        if speaker == "Unknown":
            continue

        d_start = float(seg.get("start", 0.0))
        d_end = float(seg.get("end", 0.0))

        for gseg in gender_segments:
            overlap = overlap_duration(d_start, d_end, gseg["start"], gseg["end"])
            if overlap > 0:
                speaker_gender_time[speaker][gseg["gender"]] += overlap

    speaker_gender = {}
    for speaker in sorted(all_speakers):
        male_time = speaker_gender_time[speaker]["male"]
        female_time = speaker_gender_time[speaker]["female"]
        total = male_time + female_time

        if total == 0:
            speaker_gender[speaker] = "unknown"
        elif male_time > female_time:
            confidence = male_time / total
            speaker_gender[speaker] = "male" if confidence >= threshold else "unknown"
        elif female_time > male_time:
            confidence = female_time / total
            speaker_gender[speaker] = "female" if confidence >= threshold else "unknown"
        else:
            speaker_gender[speaker] = "unknown"

    return speaker_gender


def update_json_with_gender(json_path: Path, speaker_gender: dict, output_path: Path):
    """Update JSON file with gender information."""
    print(f"Loading JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for seg in data.get("segments", []):
        speaker = seg.get("speaker_raw", "Unknown")
        seg["gender"] = speaker_gender.get(speaker, "unknown")

    data["speaker_gender"] = speaker_gender

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated JSON: {output_path}")


def extract_speaker_id(label: str):
    """Extract SPEAKER_XX from a label if present."""
    m = SPEAKER_ID_RE.search(label)
    return m.group(1) if m else None


def build_updated_label(original_label: str, speaker_raw: str, detected_gender: str, speaker_role: str | None):
    """
    Build a label preserving SPEAKER_XX and appending gender + role.

    Examples:
      original 'SPEAKER_00 (male)' -> 'SPEAKER_00 (male, HOST)'
      original 'HOST'              -> 'SPEAKER_00 (male, HOST)'
      original 'GUEST'             -> 'SPEAKER_01 (female, GUEST)'
    """
    existing_speaker_id = extract_speaker_id(original_label)
    speaker_label = existing_speaker_id or (speaker_raw if speaker_raw != "Unknown" else original_label)

    meta_parts = [detected_gender]
    if speaker_role:
        meta_parts.append(speaker_role)

    return f"{speaker_label} ({', '.join(meta_parts)})"


def update_txt_with_gender(txt_path: Path, json_path: Path, speaker_gender: dict, output_path: Path):
    """
    Update TXT file with gender info while preserving:
    - original timestamps
    - original text
    - speaker IDs like SPEAKER_00 / SPEAKER_01

    If the original line already has SPEAKER_XX, it is preserved.
    If the original line uses HOST/GUEST, speaker_raw from JSON is used.
    """
    print(f"Loading TXT: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    segments = json_data.get("segments", [])

    speaker_role_map = {}
    for seg in segments:
        speaker = seg.get("speaker_raw", "Unknown")
        role = seg.get("speaker_role")
        if speaker != "Unknown" and role:
            speaker_role_map[speaker] = role

    updated_lines = []
    seg_idx = 0

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            updated_lines.append(line)
            continue

        m = LINE_RE.match(stripped)
        if not m:
            updated_lines.append(line)
            continue

        ts = m.group("ts")
        original_label = m.group("label").strip()
        text = m.group("text")

        if seg_idx < len(segments):
            seg = segments[seg_idx]
            seg_idx += 1
        else:
            updated_lines.append(line)
            continue

        speaker_raw = seg.get("speaker_raw", "Unknown")
        detected_gender = speaker_gender.get(speaker_raw, "unknown")
        speaker_role = seg.get("speaker_role")

        new_label = build_updated_label(
            original_label=original_label,
            speaker_raw=speaker_raw,
            detected_gender=detected_gender,
            speaker_role=speaker_role
        )

        updated_lines.append(f"[{ts}] {new_label}: {text}")

    updated_lines.append("")
    updated_lines.append("# Speaker Gender Detection Summary:")
    for speaker in sorted(speaker_gender.keys()):
        role = speaker_role_map.get(speaker, "Unknown")
        gender = speaker_gender[speaker]
        updated_lines.append(f"# {speaker}: {gender} | role={role}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines))

    print(f"Updated TXT: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Detect speaker gender from audio and update transcript JSON/TXT files"
    )
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--input_json", required=True, help="Input JSON file path")
    ap.add_argument("--output_json", required=True, help="Output JSON file path with gender info")
    ap.add_argument("--output_txt", required=True, help="Output TXT file path with gender info")
    ap.add_argument("--input_txt", default=None, help="Optional input TXT file path")
    ap.add_argument(
        "--method",
        default="inaspeech",
        choices=["inaspeech", "pitch"],
        help="Gender detection method"
    )
    args = ap.parse_args()

    audio_path = Path(args.audio)
    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    output_txt_path = Path(args.output_txt)

    if args.input_txt:
        input_txt_path = Path(args.input_txt)
    else:
        input_txt_path = input_json_path.parent / f"{input_json_path.stem}.txt"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    if not input_json_path.exists():
        print(f"Error: Input JSON file not found: {input_json_path}")
        return 1

    print(f"Loading transcript JSON: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: No segments found in input JSON")
        return 1

    if args.method == "inaspeech":
        gender_segments = detect_gender_inaspeech(audio_path)
        if gender_segments is None:
            return 1
    else:
        print(f"Method {args.method} not implemented yet")
        return 1

    print(f"\nFound {len(gender_segments)} gender segments")

    print("\nAssigning gender to speakers...")
    speaker_gender = assign_gender_to_speakers(segments, gender_segments)

    print("\nResults:")
    for speaker, gender in sorted(speaker_gender.items()):
        print(f"  {speaker}: {gender}")

    update_json_with_gender(input_json_path, speaker_gender, output_json_path)

    if input_txt_path.exists():
        update_txt_with_gender(input_txt_path, input_json_path, speaker_gender, output_txt_path)
    else:
        print(f"Warning: TXT file not found: {input_txt_path}")
        print("Skipping TXT update.")

    print("\nGender detection complete!")
    print("Updated files:")
    print(f"  JSON: {output_json_path}")
    if input_txt_path.exists():
        print(f"  TXT:  {output_txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())