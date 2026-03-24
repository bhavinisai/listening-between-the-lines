#!/usr/bin/env python3
"""
add_gender_to_host_guest.py

Adds gender labels to a Deepgram diarized + host/guest labeled transcript JSON using the WAV audio.

Expected JSON format (like your sample):
{
  "segments": [
     {"speaker_id": 0, "start": 12.3, "end": 14.0, "transcript": "...", ...},
     ...
  ],
  ...
}

Output:
- Writes output JSON with:
  - segment["gender"] added for every segment
  - top-level "speaker_genders" summary map
- Optionally writes a TXT transcript with gender inline.

Install:
  pip install numpy librosa soundfile

Usage:
  python add_gender_to_host_guest.py \
    --audio path/to/episode.wav \
    --input-json path/to/episode.host_guest.json \
    --output-json path/to/episode.host_guest.gender.json \
    --output-txt path/to/episode.host_guest.gender.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


# ----------------------------
# Audio helpers
# ----------------------------

def _safe_import_soundfile():
    try:
        import soundfile as sf
        return sf
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'soundfile'. Install with: pip install soundfile"
        ) from e


def read_wav_chunk(audio_path: Path, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """
    Efficiently read a chunk [start_s, end_s] from WAV using soundfile.
    Returns mono float32 audio and the original sample rate.
    """
    sf = _safe_import_soundfile()
    if end_s <= start_s:
        return np.zeros((0,), dtype=np.float32), 0

    with sf.SoundFile(str(audio_path), "r") as f:
        sr = f.samplerate
        start_frame = max(0, int(start_s * sr))
        end_frame = max(start_frame, int(end_s * sr))
        f.seek(start_frame)
        frames_to_read = end_frame - start_frame
        if frames_to_read <= 0:
            return np.zeros((0,), dtype=np.float32), sr

        audio = f.read(frames_to_read, dtype="float32", always_2d=True)  # (n, ch)
        # mono
        y = audio.mean(axis=1)
        return y, sr


def resample_if_needed(y: np.ndarray, sr_in: int, sr_target: int) -> tuple[np.ndarray, int]:
    if sr_in == 0 or y.size == 0:
        return y, sr_in
    if sr_in == sr_target:
        return y, sr_in

    import librosa
    y_rs = librosa.resample(y, orig_sr=sr_in, target_sr=sr_target)
    return y_rs.astype(np.float32), sr_target


# ----------------------------
# Pitch / gender inference
# ----------------------------

def estimate_f0_stats(y: np.ndarray, sr: int) -> tuple[float | None, float]:
    """
    Returns (median_f0_hz, voiced_ratio) or (None, 0.0) if can't estimate.
    Uses librosa.pyin if available; falls back to librosa.yin + energy gating.
    """
    if y.size < int(0.4 * sr):  # too short
        return None, 0.0

    import librosa

    # light normalization
    y = y - np.mean(y)
    peak = np.max(np.abs(y)) + 1e-8
    y = (y / peak).astype(np.float32)

    fmin, fmax = 50.0, 400.0
    frame_length = 2048
    hop_length = 256

    # Try pyin first (gives voiced flags)
    try:
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        if f0 is None:
            return None, 0.0
        voiced = f0[voiced_flag & np.isfinite(f0)]
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        if voiced.size == 0:
            return None, 0.0
        return float(np.median(voiced)), voiced_ratio
    except Exception:
        pass

    # Fallback: yin + RMS energy gating
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # gate out quiet frames
    thr = np.percentile(rms, 40)  # keep upper 60% energy frames
    active = rms > max(thr, 1e-6)

    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0 = np.asarray(f0)
    valid = np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax) & active
    voiced = f0[valid]
    voiced_ratio = float(np.mean(valid)) if valid.size else 0.0
    if voiced.size == 0:
        return None, 0.0
    return float(np.median(voiced)), voiced_ratio


def classify_gender_from_f0(median_f0_hz: float | None, voiced_ratio: float) -> tuple[str, float]:
    """
    Heuristic label:
      - male if median_f0 < 165
      - female if median_f0 > 180
      - unknown otherwise
    Returns (gender_label, confidence_0to1)
    """
    if median_f0_hz is None or voiced_ratio < 0.15:
        return "unknown", 0.0

    # soft confidence: more voiced frames => more confidence
    conf = min(1.0, voiced_ratio / 0.55)

    if median_f0_hz < 165.0:
        # farther below threshold => higher confidence
        conf = min(1.0, conf * (1.0 + (165.0 - median_f0_hz) / 80.0))
        return "male", float(conf)
    if median_f0_hz > 180.0:
        conf = min(1.0, conf * (1.0 + (median_f0_hz - 180.0) / 80.0))
        return "female", float(conf)

    return "unknown", float(conf * 0.5)


# ----------------------------
# Transcript IO helpers
# ----------------------------

def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def pick_role_label(seg: dict) -> str:
    """
    Try common fields; fall back to speaker_label or Speaker <id>.
    """
    for k in ["role", "host_guest", "speaker_role", "label", "who"]:
        v = seg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(seg.get("speaker_label"), str) and seg["speaker_label"].strip():
        return seg["speaker_label"].strip()
    sid = seg.get("speaker_id")
    return f"Speaker {sid}" if sid is not None else "Unknown"


# ----------------------------
# Main pipeline
# ----------------------------

def build_speaker_gender_map(
    audio_path: Path,
    segments: list[dict],
    target_sr: int = 16000,
    max_seconds_per_speaker: float = 75.0,
    min_segment_sec: float = 0.6,
    max_chunk_sec: float = 6.0,
) -> dict[int, dict]:
    """
    For each speaker_id, sample up to `max_seconds_per_speaker` seconds from their speech segments,
    estimate median F0 and classify gender.
    """
    speaker_to_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for seg in segments:
        sid = seg.get("speaker_id")
        if sid is None:
            continue
        try:
            sid_int = int(sid)
        except Exception:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = end - start
        if dur >= min_segment_sec:
            speaker_to_intervals[sid_int].append((start, end))

    speaker_genders: dict[int, dict] = {}

    for sid, intervals in speaker_to_intervals.items():
        # sort by duration (use longer segments first for better pitch estimation)
        intervals = sorted(intervals, key=lambda t: (t[1] - t[0]), reverse=True)

        collected = 0.0
        f0_values = []
        voiced_ratios = []

        for (start, end) in intervals:
            if collected >= max_seconds_per_speaker:
                break
            # cap each chunk length to keep computation reasonable
            chunk_end = min(end, start + max_chunk_sec)
            chunk_len = max(0.0, chunk_end - start)
            if chunk_len < min_segment_sec:
                continue

            y, sr_in = read_wav_chunk(audio_path, start, chunk_end)
            if sr_in == 0 or y.size == 0:
                continue
            y, sr = resample_if_needed(y, sr_in, target_sr)

            median_f0, voiced_ratio = estimate_f0_stats(y, sr)
            if median_f0 is not None:
                f0_values.append(median_f0)
                voiced_ratios.append(voiced_ratio)

            collected += chunk_len

        if len(f0_values) == 0:
            speaker_genders[sid] = {
                "gender": "unknown",
                "confidence": 0.0,
                "f0_median_hz": None,
                "voiced_ratio": 0.0,
                "sampled_seconds": round(collected, 3),
            }
            continue

        # robust aggregate across chunks: median of chunk medians
        agg_f0 = float(np.median(np.array(f0_values, dtype=np.float32)))
        agg_voiced = float(np.median(np.array(voiced_ratios, dtype=np.float32))) if voiced_ratios else 0.0
        gender, conf = classify_gender_from_f0(agg_f0, agg_voiced)

        speaker_genders[sid] = {
            "gender": gender,
            "confidence": round(conf, 3),
            "f0_median_hz": round(agg_f0, 2),
            "voiced_ratio": round(agg_voiced, 3),
            "sampled_seconds": round(collected, 3),
        }

    return speaker_genders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True, help="Path to episode WAV file")
    ap.add_argument("--input-json", type=str, required=True, help="Host/guest labeled transcript JSON")
    ap.add_argument("--output-json", type=str, required=True, help="Output JSON with gender labels")
    ap.add_argument("--output-txt", type=str, default="", help="Optional output TXT with gender labels")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for pitch extraction")
    ap.add_argument("--max-seconds-per-speaker", type=float, default=75.0, help="Audio to sample per speaker (sec)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    in_json = Path(args.input_json)
    out_json = Path(args.output_json)
    out_txt = Path(args.output_txt) if args.output_txt else None

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not in_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_json}")

    data = json.loads(in_json.read_text(encoding="utf-8"))
    if "segments" not in data or not isinstance(data["segments"], list):
        raise ValueError("Input JSON must have a top-level key 'segments' that is a list.")

    segments: list[dict] = data["segments"]

    # 1) infer per-speaker genders from audio
    speaker_gender_map = build_speaker_gender_map(
        audio_path=audio_path,
        segments=segments,
        target_sr=args.sr,
        max_seconds_per_speaker=args.max_seconds_per_speaker,
    )

    # 2) apply to every segment
    for seg in segments:
        sid = seg.get("speaker_id")
        try:
            sid_int = int(sid)
        except Exception:
            seg["gender"] = "unknown"
            continue
        seg["gender"] = speaker_gender_map.get(sid_int, {}).get("gender", "unknown")

    # 3) store summary at top-level (handy for debugging)
    data["speaker_genders"] = {str(k): v for k, v in speaker_gender_map.items()}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote JSON: {out_json}")

    # 4) optional TXT
    if out_txt is not None:
        lines = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            role = pick_role_label(seg)
            gender = seg.get("gender", "unknown")
            text = (seg.get("transcript") or "").strip()
            ts = format_timestamp(start)
            lines.append(f"[{ts}] {role} ({gender}): {text}")

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[OK] Wrote TXT:  {out_txt}")


if __name__ == "__main__":
    main()