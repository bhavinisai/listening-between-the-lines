#!/usr/bin/env python3
"""
deepgram_batch_transcribe.py

Batch-transcribe multiple audio files with Deepgram via REST API.

Features:
- Recursively finds audio files in a directory (or accepts explicit file paths)
- Uploads each file to Deepgram /v1/listen (pre-recorded transcription)
- Saves per-file JSON response + plain-text transcript
- Optional diarization, utterances, paragraphs, timestamps, etc.
- Simple retry/backoff and concurrency

Usage examples:
  export DEEPGRAM_API_KEY="YOUR_KEY"

  # Transcribe all audio files in a folder (recursive)
  python deepgram_batch_transcribe.py --input-dir ./audio --out-dir ./dg_out --diarize --utterances

  # Transcribe specific files
  python deepgram_batch_transcribe.py --files a.wav b.mp3 --out-dir ./dg_out --diarize

  # Choose model + language and enable smart formatting
  python deepgram_batch_transcribe.py --input-dir ./audio --model nova-2 --language en --smart-format

Notes:
- Deepgram supports many formats (wav/mp3/m4a/flac/ogg, etc.)
- If you hit large file sizes, consider Deepgram's "transcription from URL" or their async endpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import mimetypes
import os
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests


DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".webm", ".mp4"
}


def discover_files(input_dir: str) -> List[pathlib.Path]:
    base = pathlib.Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def guess_mime(path: pathlib.Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    # Deepgram accepts application/octet-stream too; pick something reasonable.
    return mt or "application/octet-stream"


def build_query_params(args: argparse.Namespace) -> Dict[str, str]:
    """
    Map CLI flags to Deepgram query parameters.
    Adjust/add parameters as needed.
    """
    params: Dict[str, str] = {}

    if args.model:
        params["model"] = args.model
    if args.language:
        params["language"] = args.language

    # Formatting / structure
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

    # If you want keyword boosting, profanity_filter, redaction, etc., add them here.
    return params


def extract_best_transcript(resp_json: Dict) -> str:
    """
    Extract the top transcript string (best effort) from Deepgram response.
    """
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


def post_with_retries(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, str],
    data_bytes: bytes,
    content_type: str,
    max_retries: int = 4,
    timeout_s: int = 300,
) -> Tuple[int, Dict]:
    """
    POST with basic exponential backoff retry on transient errors.
    Returns (status_code, json_or_error_dict).
    """
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                url,
                headers={**headers, "Content-Type": content_type},
                params=params,
                data=data_bytes,
                timeout=timeout_s,
            )
            # Retry on rate limits / server errors
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue

            # Try to parse JSON either way
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


def transcribe_one(
    audio_path: pathlib.Path,
    out_dir: pathlib.Path,
    api_key: str,
    params: Dict[str, str],
    overwrite: bool = False,
    max_retries: int = 4,
    timeout_s: int = 300,
) -> Dict:
    """
    Transcribe a single file and write outputs.
    Returns a summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    json_out = out_dir / f"{stem}.deepgram.json"
    txt_out = out_dir / f"{stem}.txt"
    err_out = out_dir / f"{stem}.error.json"

    if not overwrite and json_out.exists():
        # Skip if already done
        return {"file": str(audio_path), "skipped": True, "json": str(json_out)}

    content_type = guess_mime(audio_path)
    headers = {"Authorization": f"Token {api_key}"}

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

    if status >= 200 and status < 300:
        json_out.write_text(json.dumps(resp, indent=2), encoding="utf-8")
        transcript = extract_best_transcript(resp)
        txt_out.write_text(transcript, encoding="utf-8")
        return {
            "file": str(audio_path),
            "ok": True,
            "status": status,
            "json": str(json_out),
            "txt": str(txt_out),
            "chars": len(transcript),
        }
    else:
        err_out.write_text(json.dumps({"status": status, "response": resp}, indent=2), encoding="utf-8")
        return {
            "file": str(audio_path),
            "ok": False,
            "status": status,
            "error_json": str(err_out),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transcribe audio files with Deepgram.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=str, help="Directory containing audio files (searched recursively).")
    src.add_argument("--files", nargs="+", help="Explicit list of audio file paths.")

    p.add_argument("--out-dir", type=str, required=True, help="Output directory for transcripts and JSON.")
    p.add_argument("--workers", type=int, default=3, help="Number of concurrent uploads.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    # Deepgram options
    p.add_argument("--model", type=str, default="nova-2", help="Deepgram model name (e.g., nova-2).")
    p.add_argument("--language", type=str, default="en", help="BCP-47 language tag (e.g., en, en-US).")

    p.add_argument("--smart-format", action="store_true", help="Enable smart formatting.")
    p.add_argument("--punctuate", action="store_true", help="Enable punctuation.")
    p.add_argument("--numerals", action="store_true", help="Convert numbers to numerals.")
    p.add_argument("--timestamps", action="store_true", help="Include timestamps.")
    p.add_argument("--paragraphs", action="store_true", help="Include paragraphs.")
    p.add_argument("--utterances", action="store_true", help="Include utterances.")
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")

    p.add_argument("--max-retries", type=int, default=4, help="Max retries for transient errors.")
    p.add_argument("--timeout-s", type=int, default=300, help="Request timeout per file in seconds.")

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

    results: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                transcribe_one,
                audio_path=p,
                out_dir=out_dir,
                api_key=api_key,
                params=params,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
                timeout_s=args.timeout_s,
            )
            for p in audio_files
        ]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            status = "SKIP" if res.get("skipped") else ("OK" if res.get("ok") else "ERR")
            print(f"[{status}] {res['file']}")

    # Write batch summary
    summary_path = out_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved batch summary: {summary_path}")

    # Exit code: 0 if all ok/skip, else 1
    any_err = any((r.get("ok") is False) for r in results)
    return 1 if any_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

