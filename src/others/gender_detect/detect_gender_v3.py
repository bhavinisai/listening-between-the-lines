#!/usr/bin/env python3
"""
add_gender_to_whisperx.py

Detect perceived voice gender from audio using inaSpeechSegmenter and
add it to a WhisperX diarized transcript JSON.

Outputs:
1) updated JSON with gender fields
2) clean TXT transcript regenerated from JSON

Expected WhisperX JSON:
- top-level: {"segments": [...]}
- each segment should have:
    - start
    - end
    - text
    - speaker OR speaker_raw

Usage:
python add_gender_to_whisperx.py \
  --audio data/raw_audio/ep_001.wav \
  --input_json data/outputs/ep_001_whisperx_diarized.json \
  --output_json data/outputs/ep_001_whisperx_diarized.gender.json \
  --output_txt data/outputs/ep_001_whisperx_diarized.gender.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


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


def get_speaker_id(seg: Dict[str, Any], preferred_key: str | None = None) -> str:
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


def detect_gender_inaspeech(audio_path: Path):
    """
    Returns list of segments like:
    [{"start": ..., "end": ..., "gender": "male"|"female"}]
    """
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError as e:
        raise RuntimeError(
            "inaSpeechSegmenter is not installed. Install with:\n"
            "pip install inaSpeechSegmenter tensorflow"
        ) from e

    print("Loading inaSpeechSegmenter model...")
    seg = Segmenter()

    print(f"Analyzing audio: {audio_path}")
    raw_segments = seg(str(audio_path))

    gender_segments = []
    for label, start, end in raw_segments:
        if label in {"male", "female"}:
            gender_segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "gender": label,
                }
            )

    return gender_segments


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    Correct overlap formula.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_gender_to_speakers(
    whisperx_segments: List[Dict[str, Any]],
    gender_segments: List[Dict[str, Any]],
    speaker_key: str | None = None,
    min_confidence: float = 0.60,
) -> Dict[str, Dict[str, Any]]:
    """
    Accumulate overlap time between diarized speaker turns and inaSpeech gender regions.

    Returns:
    {
      "SPEAKER_00": {
        "gender": "male",
        "male_time": 12.3,
        "female_time": 1.2,
        "total_gendered_time": 13.5,
        "confidence": 0.91
      },
      ...
    }
    """
    speaker_gender_time = defaultdict(lambda: {"male": 0.0, "female": 0.0})

    for seg in whisperx_segments:
        speaker = get_speaker_id(seg, speaker_key)
        if speaker == "UNKNOWN":
            continue

        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", 0.0))
        if s_end <= s_start:
            continue

        for gseg in gender_segments:
            ov = overlap_duration(
                s_start,
                s_end,
                float(gseg["start"]),
                float(gseg["end"]),
            )
            if ov > 0:
                speaker_gender_time[speaker][gseg["gender"]] += ov

    speaker_gender: Dict[str, Dict[str, Any]] = {}

    for speaker, times in speaker_gender_time.items():
        male_time = times["male"]
        female_time = times["female"]
        total = male_time + female_time

        if total <= 0:
            speaker_gender[speaker] = {
                "gender": "unknown",
                "male_time": 0.0,
                "female_time": 0.0,
                "total_gendered_time": 0.0,
                "confidence": 0.0,
            }
            continue

        if male_time >= female_time:
            confidence = male_time / total
            label = "male" if confidence >= min_confidence else "unknown"
        else:
            confidence = female_time / total
            label = "female" if confidence >= min_confidence else "unknown"

        speaker_gender[speaker] = {
            "gender": label,
            "male_time": round(male_time, 3),
            "female_time": round(female_time, 3),
            "total_gendered_time": round(total, 3),
            "confidence": round(confidence, 4),
        }

    return speaker_gender


def apply_gender_to_json(
    data: Dict[str, Any],
    speaker_gender: Dict[str, Dict[str, Any]],
    speaker_key: str | None = None,
) -> Dict[str, Any]:
    for seg in data.get("segments", []):
        speaker = get_speaker_id(seg, speaker_key)
        info = speaker_gender.get(
            speaker,
            {
                "gender": "unknown",
                "male_time": 0.0,
                "female_time": 0.0,
                "total_gendered_time": 0.0,
                "confidence": 0.0,
            },
        )

        seg["gender"] = info["gender"]
        seg["gender_confidence"] = info["confidence"]

    data["speaker_gender_mapping"] = speaker_gender
    return data


def build_txt_lines(
    data: Dict[str, Any],
    speaker_key: str | None = None,
    include_role_if_present: bool = True,
) -> List[str]:
    lines = []

    for seg in data.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = get_speaker_id(seg, speaker_key)
        gender = seg.get("gender", "unknown")

        if include_role_if_present and seg.get("speaker_role"):
            label = f'{seg["speaker_role"]} ({gender})'
        else:
            label = f"{speaker} ({gender})"

        lines.append(f"[{ts(start)} - {ts(end)}] {label}: {text}")

    return lines


def main():
    ap = argparse.ArgumentParser(description="Add gender labels to WhisperX diarized transcript")
    ap.add_argument("--audio", required=True, help="Path to source audio")
    ap.add_argument("--input_json", required=True, help="WhisperX diarized JSON")
    ap.add_argument("--output_json", required=True, help="Output JSON with gender labels")
    ap.add_argument("--output_txt", required=True, help="Output TXT transcript with gender labels")
    ap.add_argument(
        "--speaker_key",
        default=None,
        help="Preferred speaker key to read from each segment (e.g. speaker_raw or speaker)",
    )
    ap.add_argument(
        "--min_confidence",
        type=float,
        default=0.60,
        help="Minimum male/female overlap ratio to assign a label, else unknown",
    )
    args = ap.parse_args()

    audio_path = Path(args.audio)
    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    output_txt_path = Path(args.output_txt)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    print(f"Loading WhisperX JSON: {input_json_path}")
    data = load_json(input_json_path)

    segments = data.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("Input JSON is missing segments[] or it is empty.")

    gender_segments = detect_gender_inaspeech(audio_path)
    print(f"Found {len(gender_segments)} gender-labeled audio regions")

    speaker_gender = assign_gender_to_speakers(
        whisperx_segments=segments,
        gender_segments=gender_segments,
        speaker_key=args.speaker_key,
        min_confidence=args.min_confidence,
    )

    print("\nSpeaker gender results:")
    for speaker, info in sorted(speaker_gender.items()):
        print(
            f"  {speaker}: {info['gender']} "
            f"(male_time={info['male_time']}, female_time={info['female_time']}, conf={info['confidence']})"
        )

    updated = apply_gender_to_json(
        data=data,
        speaker_gender=speaker_gender,
        speaker_key=args.speaker_key,
    )

    txt_lines = build_txt_lines(
        updated,
        speaker_key=args.speaker_key,
        include_role_if_present=True,
    )

    save_json(output_json_path, updated)
    save_txt(output_txt_path, txt_lines)

    print("\nDone.")
    print("JSON:", output_json_path)
    print("TXT :", output_txt_path)


if __name__ == "__main__":
    main()
