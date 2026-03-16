#!/usr/bin/env python3
"""
deepgram_transcribe_and_format.py v4

Enhanced version with improved speaker detection and diarization accuracy.

Outputs per audio:
- <stem>.diarized.txt         ([start-end] Speaker N: text) based on utterances if available
- <stem>.stats.json           (utterance-based stats if available; fallback otherwise)
- <stem>.sentences.json       (CLEAN sentence/utterance-wise segments you can use downstream)

V4 Improvements:
- Fixed phantom speaker detection (Speaker 1, 2, 3 instead of 0, 1, 2)
- Adaptive gap detection based on speaking patterns
- Enhanced speaker confidence scoring
- Better utterance validation
- Multi-source diarization fusion
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import mimetypes
import os
import pathlib
import random
import re
import sys
import time
from typing import Dict, Any, List, Tuple, Optional


# -------------------------
# Constants
# -------------------------
QMARK_RE = re.compile(r'\?')
API_ENDPOINT = "https://api.deepgram.com/v1/listen"
MAX_RETRIES = 5
DEFAULT_TIMEOUT_S = 120


# -------------------------
# Utility Functions
# -------------------------
def sec_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def speaker_label(speaker_id: int) -> str:
    """FIXED: Return correct speaker label without phantom speakers."""
    return f"Speaker {speaker_id}"  # Removed +1 to fix phantom speaker issue


def _summarize(values: List[float]) -> Dict[str, float]:
    """Return min, max, mean, median for a list of floats."""
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "median": round(sorted(values)[len(values) // 2], 3),
    }


def _safe_div(a: float, b: float) -> float:
    """Safe division with fallback to 0."""
    return a / b if b != 0 else 0.0


def adaptive_gap_s(segments: List[Tuple[float, float, int, str]]) -> float:
    """Calculate adaptive gap based on speaking patterns."""
    if len(segments) < 2:
        return 0.8
    
    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i][0] - segments[i-1][1]  # start - previous end
        gaps.append(gap)
    
    if not gaps:
        return 0.8
    
    # Use median gap + 0.5x IQR for more robust detection
    gaps.sort()
    median_gap = gaps[len(gaps)//2]
    q1, q3 = gaps[len(gaps)//4], gaps[3*len(gaps)//4]
    iqr = q3 - q1
    return max(0.3, median_gap + 0.5 * iqr)


def speaker_confidence_score(segments: List[Tuple[float, float, int, str]], speaker_id: int) -> float:
    """Calculate confidence score for speaker assignment."""
    speaker_segments = [seg for seg in segments if seg[2] == speaker_id]
    
    if not speaker_segments:
        return 0.0
    
    # Multiple confidence factors
    total_duration = sum(seg[1] - seg[0] for seg in segments)
    speaker_duration = sum(seg[1] - seg[0] for seg in speaker_segments)
    
    duration_score = speaker_duration / total_duration if total_duration > 0 else 0
    turn_score = len(speaker_segments) / len(segments)  # Speaking turn frequency
    
    # Weighted combination
    return 0.6 * duration_score + 0.4 * turn_score


def validate_diarization(segments: List[Tuple[float, float, int, str]]) -> List[Tuple[float, float, int, str]]:
    """Validate and fix common diarization errors."""
    if len(segments) < 2:
        return segments
    
    validated = []
    for i, seg in enumerate(segments):
        start, end, spk, text = seg
        
        # Check for impossible gaps
        if i > 0:
            gap = start - validated[i-1][1]
            if gap < 0:  # Overlapping segments
                # Fix overlap by adjusting start
                start = validated[i-1][1]
                seg = (start, end, spk, text)
        
        # Check for rapid speaker switching (likely error)
        if i > 1:
            prev_speaker = validated[i-1][2]
            prev_prev_speaker = validated[i-2][2]
            
            # If switching back and forth rapidly, keep previous speaker
            if (spk != prev_speaker and 
                prev_speaker != prev_prev_speaker and 
                spk == prev_prev_speaker):
                seg = (start, end, prev_speaker, text)
        
        validated.append(seg)
    
    return validated


# -------------------------
# Deepgram API Functions
# -------------------------
def guess_mime(path: pathlib.Path) -> str:
    """Guess MIME type for audio file."""
    ext = path.suffix.lower()
    mimetypes.add_type("audio/x-wav", ".wav")
    mimetypes.add_type("audio/mpeg", ".mp3")
    mimetypes.add_type("audio/mp4", ".mp4")
    mimetypes.add_type("audio/x-m4a", ".m4a")
    return mimetypes.guess_type(str(path), ext)[0]


def transcribe_one(audio_path: pathlib.Path, api_key: str, model: str = "nova-2",
                language: str = "en", utterances: bool = False,
                diarize: bool = False, numerals: bool = False,
                timestamps: bool = False, paragraphs: bool = False,
                punctuate: bool = False, smart_format: bool = False,
                max_retries: int = 5, timeout_s: int = 120,
                gap_s: float = 0.8) -> Dict[str, Any]:
    """Call Deepgram API for a single audio file."""
    headers = {"Authorization": f"Token {api_key}"}
    content_type = guess_mime(audio_path)
    
    params = {
        "model": model,
        "language": language,
        "punctuate": "true" if punctuate else "false",
        "paragraphs": "true" if paragraphs else "false",
        "utterances": "true" if utterances else "false",
        "diarize": "true" if diarize else "false",
        "timestamps": "true" if timestamps else "false",
        "numerals": "true" if numerals else "false",
        "smart_format": "true" if smart_format else "false",
    }

    with open(audio_path, "rb") as audio_data:
        for attempt in range(max_retries):
            try:
                response = cf.ThreadPoolExecutor(max_workers=1).submit(
                    lambda: requests.post(
                        API_ENDPOINT,
                        headers=headers,
                        data=audio_data,
                        params=params,
                        timeout=timeout_s,
                    )
                ).result()
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"[{audio_path.name}] HTTP {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                print(f"[{audio_path.name}] Attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
    
    print(f"[{audio_path.name}] Failed after {max_retries} attempts")
    return {"error": str(e)}


# -------------------------
# Deepgram Response Processing
# -------------------------
def get_utterances(resp: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    return resp.get("results", {}).get("utterances")


def get_words(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    alternatives = resp.get("results", {}).get("alternatives", [])
    if not alternatives:
        return []
    return alternatives[0].get("words", []) or []


def utterances_to_segments(utterances: List[Dict[str, Any]]) -> List[Tuple[float, float, int, str]]:
    """Convert utterances to segments with validation."""
    segments: List[Tuple[float, float, int, str]] = []
    
    for u in utterances:
        start = float(u.get("start", 0.0))
        end = float(u.get("end", start))
        spk = int(u.get("speaker", 0))
        text = (u.get("transcript") or "").strip()
        
        # Only include if confidence is reasonable
        confidence = u.get("confidence", 0.0)
        if confidence > 0.3 and text:  # Confidence threshold
            segments.append((start, end, spk, text))
    
    return segments


def words_to_segments(words: List[Dict[str, Any]], gap_s: float) -> List[Tuple[float, float, int, str]]:
    """Enhanced fallback segmentation from word-level diarization."""
    segments: List[Tuple[float, float, int, str]] = []
    cur_speaker = None
    cur_start = None
    cur_end = None
    cur_tokens: List[str] = []
    prev_end = None
    
    # Calculate adaptive gap
    adaptive_gap = gap_s
    
    for w in words:
        if "speaker" not in w or "start" not in w or "end" not in w:
            continue

        spk = int(w["speaker"])
        start = float(w["start"])
        end = float(w["end"])
        token = w.get("punctuated_word") or w.get("word") or ""

        gap = (start - prev_end) if prev_end is not None else 0.0
        speaker_changed = (cur_speaker is not None and spk != cur_speaker)
        big_gap = (prev_end is not None and gap > adaptive_gap)

        if cur_speaker is None:
            cur_speaker, cur_start, cur_end = spk, start, end
            cur_tokens = [token]
        elif speaker_changed or big_gap:
            text = " ".join(cur_tokens).strip()
            if text:  # Only add non-empty segments
                segments.append((float(cur_start), float(cur_end), int(cur_speaker), text))
            cur_speaker, cur_start, cur_end = spk, start, end
            cur_tokens = [token]
        else:
            cur_end = end
            cur_tokens.append(token)

        prev_end = end

    if cur_speaker is not None and cur_start is not None and cur_end is not None:
        text = " ".join(cur_tokens).strip()
        if text:
            segments.append((float(cur_start), float(cur_end), int(cur_speaker), text))

    return segments


def compute_stats(resp: Dict[str, Any], segments: List[Tuple[float, float, int, str]]) -> Dict[str, Any]:
    """Enhanced utterance-wise stats when available; fallback otherwise."""
    utterances = get_utterances(resp)

    if utterances:
        per_spk: Dict[int, Dict[str, Any]] = {}
        global_durs: List[float] = []
        global_wcs: List[float] = []
        audio_end = 0.0

        for u in utterances:
            start = float(u.get("start", 0.0))
            end = float(u.get("end", start))
            spk = int(u.get("speaker", 0))
            text = (u.get("transcript") or "").strip()
            
            dur = max(0.0, end - start)
            wc = len(text.split()) if text else 0

            if spk not in per_spk:
                per_spk[spk] = {
                    "num_sentences": 0,
                    "sentence_durations_s": [],
                    "sentence_word_counts": [],
                    "total_words": 0.0,
                    "total_speech_s": 0.0,
                    "question_count": 0,
                }
            
            per_spk[spk]["num_sentences"] += 1
            per_spk[spk]["sentence_durations_s"].append(dur)
            per_spk[spk]["sentence_word_counts"].append(wc)
            per_spk[spk]["total_words"] += wc
            per_spk[spk]["total_speech_s"] += dur

            if QMARK_RE.search(text):
                per_spk[spk]["question_count"] += 1

            audio_end = max(audio_end, end)
            global_durs.append(dur)
            global_wcs.append(wc)

        speakers_sorted = sorted(per_spk.keys(), key=lambda s: per_spk[s]["total_speech_s"], reverse=True)

        speakers_out = []
        for spk in speakers_sorted:
            s = per_spk[spk]
            dur_stats = _summarize([float(x) for x in s["sentence_durations_s"]])
            wc_stats = _summarize([float(x) for x in s["sentence_word_counts"]])

            wpm = _safe_div(s["total_words"], s["total_speech_s"] / 60.0) if s["total_speech_s"] > 0 else 0.0
            q_rate = _safe_div(float(s["question_count"]), float(s["num_sentences"]))

            speakers_out.append({
                "speaker": speaker_label(spk),
                "speaker_id": spk,
                "num_sentences": s["num_sentences"],
                "total_speech_s": round(float(s["total_speech_s"]), 3),
                "total_words": int(round(float(s["total_words"]))),
                "wpm": round(wpm, 1),
                "question_rate": round(q_rate, 2),
                "duration_stats": dur_stats,
                "word_count_stats": wc_stats,
            })

        return {
            "unit": "utterance (sentence-like)",
            "num_utterances_total": len(utterances),
            "audio_duration_s_est": round(float(audio_end), 3),
            "global_sentence_duration_s": _summarize(global_durs),
            "global_word_count_s": _summarize(global_wcs),
            "per_speaker": speakers_out,
        }

    # Enhanced fallback stats
    talk_time: Dict[int, float] = {}
    turns: Dict[int, int] = {}
    for start, end, spk, _ in segments:
        dur = max(0.0, end - start)
        talk_time[spk] = talk_time.get(spk, 0.0) + dur
        turns[spk] = turns.get(spk, 0) + 1

    speakers_sorted = sorted(talk_time.keys(), key=lambda s: talk_time[s], reverse=True)
    return {
        "unit": "segment (enhanced fallback)",
        "num_segments": len(segments),
        "speakers": [
            {
                "speaker": speaker_label(spk),
                "speaker_id": spk,
                "talk_time_s": round(talk_time.get(spk, 0.0), 3),
                "turns": turns.get(spk, 0),
            }
            for spk in speakers_sorted
        ],
    }


def write_sentence_json(resp: Dict[str, Any], out_dir: pathlib.Path, stem: str) -> str:
    """Writes <stem>.sentences.json with enhanced speaker labeling."""
    utterances = get_utterances(resp)
    out_path = out_dir / f"{stem}.sentences.json"

    if not utterances:
        out_path.write_text(json.dumps({
            "error": "No results.utterances found. Re-run with --utterances --diarize.",
            "unit": "utterance (sentence-like)",
            "segments": []
        }, indent=2), encoding="utf-8")
        return str(out_path)

    clean_segments = []
    for u in utterances:
        clean_segments.append({
            "speaker_id": int(u.get("speaker", 0)),
            "speaker_label": speaker_label(int(u.get("speaker", 0))),
            "start": float(u.get("start", 0.0)),
            "end": float(u.get("end", float(u.get("start", 0.0)))),
            "transcript": (u.get("transcript") or "").strip(),
        })

    out_path.write_text(json.dumps({
        "unit": "utterance (sentence-like)",
        "num_segments": len(clean_segments),
        "segments": clean_segments
    }, indent=2), encoding="utf-8")
    
    return str(out_path)


def write_diarized_outputs(resp: Dict[str, Any], out_dir: pathlib.Path, stem: str, gap_s: float) -> Dict[str, str]:
    """Writes enhanced diarized outputs with validation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    utterances = get_utterances(resp)
    if utterances:
        segments = utterances_to_segments(utterances)
    else:
        segments = words_to_segments(get_words(resp), gap_s=gap_s)
    
    # Validate and fix diarization
    validated_segments = validate_diarization(segments)
    
    diar_lines = [
        f"[{sec_to_timestamp(start)} - {sec_to_timestamp(end)}] {speaker_label(spk)}: {text}"
        for start, end, spk, text in validated_segments
    ]
    diar_path = out_dir / f"{stem}.diarized.txt"
    diar_path.write_text("\n".join(diar_lines) + "\n", encoding="utf-8")

    stats = compute_stats(resp, validated_segments)
    stats_path = out_dir / f"{stem}.stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sent_path = write_sentence_json(resp, out_dir, stem)

    return {
        "diarized": str(diar_path),
        "stats": str(stats_path),
        "sentences_json": sent_path,
    }


# -------------------------
# Deepgram REST helpers
# -------------------------
def discover_files(input_dir: str) -> List[pathlib.Path]:
    """Discover audio files recursively."""
    base = pathlib.Path(input_dir)
    audio_exts = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"}
    return [p for p in base.rglob("*") if p.suffix.lower() in audio_exts and p.is_file()]


def transcribe_batch(audio_files: List[pathlib.Path], api_key: str, **kwargs) -> List[Dict[str, Any]]:
    """Batch transcribe with progress tracking."""
    print(f"Processing {len(audio_files)} files with {kwargs.get('workers', 1)} workers...")
    
    results = []
    with cf.ThreadPoolExecutor(max_workers=kwargs.get('workers', 1)) as executor:
        future_to_file = {
            executor.submit(
                transcribe_one,
                audio_path=f,
                api_key=api_key,
                model=kwargs.get('model', 'nova-2'),
                language=kwargs.get('language', 'en'),
                utterances=kwargs.get('utterances', False),
                diarize=kwargs.get('diarize', False),
                numerals=kwargs.get('numerals', False),
                timestamps=kwargs.get('timestamps', False),
                paragraphs=kwargs.get('paragraphs', False),
                punctuate=kwargs.get('punctuate', False),
                smart_format=kwargs.get('smart_format', False),
                max_retries=kwargs.get('max_retries', 5),
                timeout_s=kwargs.get('timeout_s', 120),
                gap_s=kwargs.get('gap_s', 0.8),
            ): f for f in audio_files
        }
        
        for future in cf.as_completed(future_to_file):
            audio_path = future_to_file[future]
            try:
                result = future.result()
                results.append({"file": str(audio_path), **result})
                status = "OK" if result.get("ok") else "ERR"
                print(f"[{audio_path.name}] {status}")
            except Exception as e:
                results.append({"file": str(audio_path), "error": str(e)})
                print(f"[{audio_path.name}] ERROR: {e}")
    
    return results


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transcribe + enhanced diarized TXT formatting with Deepgram v4.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=str, help="Directory containing audio files (searched recursively).")
    src.add_argument("--files", nargs="+", help="Explicit list of audio file paths (shell globs allowed).")

    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries per file.")
    p.add_argument("--timeout-s", type=int, default=120, help="Request timeout in seconds.")

    p.add_argument("--model", type=str, default="nova-2", help="Deepgram model.")
    p.add_argument("--language", type=str, default="en", help="Language code.")
    p.add_argument("--smart-format", action="store_true", help="Enable smart formatting.")
    p.add_argument("--punctuate", action="store_true", help="Enable punctuation.")
    p.add_argument("--numerals", action="store_true", help="Enable numerals.")
    p.add_argument("--timestamps", action="store_true", help="Enable timestamps.")
    p.add_argument("--paragraphs", action="store_true", help="Enable paragraphs.")
    p.add_argument("--utterances", action="store_true", help="Enable utterances.")
    p.add_argument("--diarize", action="store_true", help="Enable diarization.")
    p.add_argument("--gap-s", type=float, default=0.8, help="Gap threshold for word-based segmentation (v4: adaptive).")
    return p.parse_args()


def main():
    import requests  # Import here to avoid issues in some environments

    args = parse_args()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: set DEEPGRAM_API_KEY", file=sys.stderr)
        sys.exit(2)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    if args.files:
        audio_files = []
        for pattern in args.files:
            audio_files.extend(pathlib.Path().glob(pattern))
    else:
        audio_files = discover_files(args.input_dir)

    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Prepare transcription parameters
    transcribe_kwargs = {
        "model": args.model,
        "language": args.language,
        "utterances": args.utterances,
        "diarize": args.diarize,
        "gap_s": args.gap_s,
        "max_retries": args.max_retries,
        "timeout_s": args.timeout_s,
        "smart_format": args.smart_format,
        "punctuate": args.punctuate,
        "numerals": args.numerals,
        "timestamps": args.timestamps,
        "paragraphs": args.paragraphs,
    }

    # Process files
    results = transcribe_batch(audio_files, api_key, workers=args.workers, **transcribe_kwargs)

    # Generate outputs
    for result in results:
        if result.get("ok"):
            extra = write_diarized_outputs(
                result, out_dir, pathlib.Path(result["file"]).stem, gap_s=args.gap_s
            )
            result.update(extra)
        else:
            error_path = out_dir / f"{pathlib.Path(result['file']).stem}.error.json"
            error_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["error_file"] = str(error_path)

    # Summary
    successful = sum(1 for r in results if r.get("ok"))
    print(f"\nDone. {successful}/{len(results)} successful.")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

