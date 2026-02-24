#!/usr/bin/env python3
"""
deepgram_transcribe_and_format.py

Outputs per audio:
- <stem>.deepgram.json        (RAW Deepgram response; often includes word-level 'words' even if utterances exist)
- <stem>.txt                 (plain transcript)
- <stem>.diarized.txt         ([start-end] Speaker N: text) based on utterances if available
- <stem>.stats.json           (utterance-based stats if available; fallback otherwise)
- <stem>.sentences.json       (CLEAN sentence/utterance-wise segments you can use downstream)
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
from typing import Any, Dict, List, Optional, Tuple

import requests

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".webm", ".mp4"
}

QMARK_RE = re.compile(r"\?\s*$")


# -------------------------
# Helpers
# -------------------------

def sec_to_timestamp(s: float) -> str:
    if s < 0:
        s = 0.0
    ms = int(round((s - int(s)) * 1000))
    total = int(s)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def speaker_label(speaker_id: int) -> str:
    return f"Speaker {speaker_id + 1}"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def extract_best_transcript(resp_json: Dict[str, Any]) -> str:
    try:
        channels = resp_json.get("results", {}).get("channels", [])
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""
        return alternatives[0].get("transcript", "") or ""
    except Exception:
        return ""


def get_utterances(resp: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    return resp.get("results", {}).get("utterances")


def get_words(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = resp.get("results", {}).get("channels", [])
    if not ch:
        return []
    alts = ch[0].get("alternatives", [])
    if not alts:
        return []
    return alts[0].get("words", []) or []


def utterances_to_segments(utterances: List[Dict[str, Any]]) -> List[Tuple[float, float, int, str]]:
    segments: List[Tuple[float, float, int, str]] = []
    for u in utterances:
        start = float(u.get("start", 0.0))
        end = float(u.get("end", start))
        spk = int(u.get("speaker", 0))
        text = (u.get("transcript") or "").strip()
        if text:
            segments.append((start, end, spk, text))
    return segments


def words_to_segments(words: List[Dict[str, Any]], gap_s: float = 0.8) -> List[Tuple[float, float, int, str]]:
    """Fallback segmentation from word-level diarization."""
    segments: List[Tuple[float, float, int, str]] = []
    cur_speaker = None
    cur_start = None
    cur_end = None
    cur_tokens: List[str] = []
    prev_end = None

    for w in words:
        if "speaker" not in w or "start" not in w or "end" not in w:
            continue

        spk = int(w["speaker"])
        start = float(w["start"])
        end = float(w["end"])
        token = w.get("punctuated_word") or w.get("word") or ""

        gap = (start - prev_end) if prev_end is not None else 0.0
        speaker_changed = (cur_speaker is not None and spk != cur_speaker)
        big_gap = (prev_end is not None and gap > gap_s)

        if cur_speaker is None:
            cur_speaker, cur_start, cur_end = spk, start, end
            cur_tokens = [token]
        elif speaker_changed or big_gap:
            text = " ".join(cur_tokens).strip()
            segments.append((float(cur_start), float(cur_end), int(cur_speaker), text))

            cur_speaker, cur_start, cur_end = spk, start, end
            cur_tokens = [token]
        else:
            cur_end = end
            cur_tokens.append(token)

        prev_end = end

    if cur_speaker is not None and cur_start is not None and cur_end is not None:
        text = " ".join(cur_tokens).strip()
        segments.append((float(cur_start), float(cur_end), int(cur_speaker), text))

    return segments


def compute_stats(resp: Dict[str, Any], segments: List[Tuple[float, float, int, str]]) -> Dict[str, Any]:
    """Utterance-wise stats when available; fallback otherwise."""
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
            wc = float(len([w for w in text.split() if w]))

            audio_end = max(audio_end, end)
            global_durs.append(dur)
            global_wcs.append(wc)

            if spk not in per_spk:
                per_spk[spk] = {
                    "num_sentences": 0,
                    "sentence_durations_s": [],
                    "sentence_word_counts": [],
                    "question_count": 0,
                    "total_speech_s": 0.0,
                    "total_words": 0.0,
                }

            per_spk[spk]["num_sentences"] += 1
            per_spk[spk]["sentence_durations_s"].append(dur)
            per_spk[spk]["sentence_word_counts"].append(wc)
            per_spk[spk]["total_speech_s"] += dur
            per_spk[spk]["total_words"] += wc

            if QMARK_RE.search(text):
                per_spk[spk]["question_count"] += 1

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
                "wpm_est": round(float(wpm), 2),
                "question_count": int(s["question_count"]),
                "question_rate_per_sentence": round(float(q_rate), 4),
                "sentence_duration_s": {
                    "count": int(dur_stats["count"]),
                    "mean": round(float(dur_stats["mean"]), 3),
                    "min": round(float(dur_stats["min"]), 3),
                    "max": round(float(dur_stats["max"]), 3),
                },
                "sentence_word_count": {
                    "count": int(wc_stats["count"]),
                    "mean": round(float(wc_stats["mean"]), 3),
                    "min": round(float(wc_stats["min"]), 3),
                    "max": round(float(wc_stats["max"]), 3),
                },
            })

        global_dur_stats = _summarize(global_durs)
        global_wc_stats = _summarize(global_wcs)

        return {
            "unit": "utterance (sentence-like)",
            "num_utterances_total": len(utterances),
            "audio_duration_s_est": round(float(audio_end), 3),
            "global_sentence_duration_s": {
                "count": int(global_dur_stats["count"]),
                "mean": round(float(global_dur_stats["mean"]), 3),
                "min": round(float(global_dur_stats["min"]), 3),
                "max": round(float(global_dur_stats["max"]), 3),
            },
            "global_sentence_word_count": {
                "count": int(global_wc_stats["count"]),
                "mean": round(float(global_wc_stats["mean"]), 3),
                "min": round(float(global_wc_stats["min"]), 3),
                "max": round(float(global_wc_stats["max"]), 3),
            },
            "per_speaker": speakers_out,
        }

    # Fallback
    talk_time: Dict[int, float] = {}
    turns: Dict[int, int] = {}
    for start, end, spk, _ in segments:
        dur = max(0.0, end - start)
        talk_time[spk] = talk_time.get(spk, 0.0) + dur
        turns[spk] = turns.get(spk, 0) + 1

    speakers_sorted = sorted(talk_time.keys(), key=lambda s: talk_time[s], reverse=True)
    return {
        "unit": "segment (fallback)",
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
    """
    Writes <stem>.sentences.json:
    A clean list of sentence/utterance segments (speaker, start, end, transcript).
    """
    utterances = get_utterances(resp) or []
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
    """
    Writes:
      - <stem>.txt
      - <stem>.diarized.txt
      - <stem>.stats.json
      - <stem>.sentences.json  (clean utterance-wise segments)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plain_txt = extract_best_transcript(resp)
    txt_path = out_dir / f"{stem}.txt"
    txt_path.write_text(plain_txt + "\n", encoding="utf-8")

    utterances = get_utterances(resp)
    if utterances:
        segments = utterances_to_segments(utterances)
    else:
        segments = words_to_segments(get_words(resp), gap_s=gap_s)

    diar_lines = [
        f"[{sec_to_timestamp(start)} - {sec_to_timestamp(end)}] {speaker_label(spk)}: {text}"
        for start, end, spk, text in segments
    ]
    diar_path = out_dir / f"{stem}.diarized.txt"
    diar_path.write_text("\n".join(diar_lines) + "\n", encoding="utf-8")

    stats = compute_stats(resp, segments)
    stats_path = out_dir / f"{stem}.stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sent_path = write_sentence_json(resp, out_dir, stem)

    return {
        "txt": str(txt_path),
        "diarized": str(diar_path),
        "stats": str(stats_path),
        "sentences_json": sent_path,
    }


# -------------------------
# Deepgram REST helpers
# -------------------------

def discover_files(input_dir: str) -> List[pathlib.Path]:
    base = pathlib.Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files: List[pathlib.Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def guess_mime(path: pathlib.Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def build_query_params(args: argparse.Namespace) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if args.model:
        params["model"] = args.model
    if args.language:
        params["language"] = args.language

    if args.smart_format:
        params["smart_format"] = "true"
    if args.punctuate:
        params["punctuate"] = "true"
    if args.paragraphs:
        params["paragraphs"] = "true"
    if args.utterances:
        params["utterances"] = "true"
    if args.diarize:
        params["diarize"] = "true"
    if args.timestamps:
        params["timestamps"] = "true"
    if args.numerals:
        params["numerals"] = "true"
    return params


def post_with_retries(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, str],
    data_bytes: bytes,
    content_type: str,
    max_retries: int,
    timeout_s: int,
) -> Tuple[int, Dict[str, Any]]:
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                url,
                headers={**headers, "Content-Type": content_type},
                params=params,
                data=data_bytes,
                timeout=timeout_s,
            )

            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue

            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {"error": "Non-JSON response", "text": r.text}

        except requests.RequestException as e:
            if attempt < max_retries:
                sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            return 0, {"error": "Request failed", "exception": str(e)}

    return 0, {"error": "Unexpected retry loop exit"}


def transcribe_and_format_one(
    audio_path: pathlib.Path,
    out_dir: pathlib.Path,
    api_key: str,
    params: Dict[str, str],
    overwrite: bool,
    max_retries: int,
    timeout_s: int,
    gap_s: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    json_out = out_dir / f"{stem}.deepgram.json"
    err_out = out_dir / f"{stem}.error.json"

    if not overwrite and json_out.exists():
        resp = json.loads(json_out.read_text(encoding="utf-8"))
        extra = write_diarized_outputs(resp, out_dir, stem, gap_s=gap_s)
        return {"file": str(audio_path), "skipped_transcribe": True, "json": str(json_out), **extra}

    headers = {"Authorization": f"Token {api_key}"}
    content_type = guess_mime(audio_path)
    data_bytes = audio_path.read_bytes()

    status, resp = post_with_retries(
        DEEPGRAM_LISTEN_URL,
        headers=headers,
        params=params,
        data_bytes=data_bytes,
        content_type=content_type,
        max_retries=max_retries,
        timeout_s=timeout_s,
    )

    if 200 <= status < 300:
        json_out.write_text(json.dumps(resp, indent=2), encoding="utf-8")
        extra = write_diarized_outputs(resp, out_dir, stem, gap_s=gap_s)
        return {"file": str(audio_path), "ok": True, "status": status, "json": str(json_out), **extra}

    err_out.write_text(json.dumps({"status": status, "response": resp}, indent=2), encoding="utf-8")
    return {"file": str(audio_path), "ok": False, "status": status, "error_json": str(err_out)}


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transcribe + diarized TXT formatting with Deepgram.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=str, help="Directory containing audio files (searched recursively).")
    src.add_argument("--files", nargs="+", help="Explicit list of audio file paths (shell globs allowed).")

    p.add_argument("--out-dir", type=str, required=True, help="Output directory for JSON/TXT.")
    p.add_argument("--workers", type=int, default=3, help="Concurrent requests.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--max-retries", type=int, default=4, help="Max retries for transient errors.")
    p.add_argument("--timeout-s", type=int, default=300, help="Request timeout per file (seconds).")

    # Deepgram options
    p.add_argument("--model", type=str, default="nova-2", help="Model (e.g., nova-2).")
    p.add_argument("--language", type=str, default="en", help="Language tag (e.g., en, en-US).")
    p.add_argument("--smart-format", action="store_true")
    p.add_argument("--punctuate", action="store_true")
    p.add_argument("--numerals", action="store_true")
    p.add_argument("--timestamps", action="store_true")
    p.add_argument("--paragraphs", action="store_true")
    p.add_argument("--utterances", action="store_true")
    p.add_argument("--diarize", action="store_true")

    p.add_argument("--gap-s", type=float, default=0.8, help="Gap threshold for word-based segmentation fallback.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: Please set environment variable DEEPGRAM_API_KEY.", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    params = build_query_params(args)

    if args.input_dir:
        audio_files = discover_files(args.input_dir)
    else:
        audio_files = [pathlib.Path(f) for f in args.files]

    audio_files = [p for p in audio_files if p.exists() and p.is_file()]
    if not audio_files:
        print("ERROR: No valid audio files found.", file=sys.stderr)
        return 2

    print(f"Found {len(audio_files)} file(s). Output -> {out_dir}")
    print(f"Deepgram params: {params}")

    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                transcribe_and_format_one,
                audio_path=p,
                out_dir=out_dir,
                api_key=api_key,
                params=params,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
                timeout_s=args.timeout_s,
                gap_s=args.gap_s,
            )
            for p in audio_files
        ]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            status = "OK" if res.get("ok") else ("SKIP" if res.get("skipped_transcribe") else "ERR")
            print(f"[{status}] {res['file']}")

    summary_path = out_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved batch summary: {summary_path}")

    any_err = any((r.get("ok") is False) for r in results)
    return 1 if any_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

