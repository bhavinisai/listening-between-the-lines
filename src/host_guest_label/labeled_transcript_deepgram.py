#!/usr/bin/env python3
"""
labeled_transcript_deepgram.py

- Loads Deepgram transcript JSON (both .sentences.json and .deepgram.json formats)
- Uses OpenRouter to pick HOST speaker (exactly one) and labels others as GUEST
- Writes:
    1) <stem>.host_guest.txt
    2) <stem>.host_guest.json (adds speaker_role per segment, plus speaker_role_mapping)

Supports Deepgram formats:
- .sentences.json: segments[] with speaker_id, speaker_label, transcript
- .deepgram.json: results.channels[].alternatives[].words[] with speaker field

Usage:
  pip install requests
  export OPENROUTER_API_KEY="..."

python src/host_guest_label/labeled_transcript_deepgram.py \
  --input_dir outputs/unit_testing \
  --pattern "*.sentences.json" \
  --out_dir outputs/deepgram_output \
  --model openrouter/free

  python src/host_guest_label/labeled_transcript_deepgram.py \
    --input outputs/deepgram_output/episode_001.sentences.json \
    --out_dir outputs/deepgram_output \
    --model openrouter/free \
    --debug_dump_raw
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Utilities
# -----------------------------
def hhmmss(seconds: float) -> str:
    if seconds is None:
        return "00:00:00"
    td = timedelta(seconds=float(seconds))
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_txt(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def safe_parse_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    # Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract first JSON object
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    preview = raw[:1500].replace("\n", "\\n")
    raise ValueError(
        "Could not find JSON object in model output. "
        f"Model output preview (first 1500 chars): {preview}"
    )


# -----------------------------
# Deepgram format handling
# -----------------------------
def load_deepgram_sentences_format(path: Path) -> List[Dict[str, Any]]:
    """Load .sentences.json format and convert to standard format"""
    data = load_json(path)
    segments = data.get("segments", [])
    
    converted = []
    for seg in segments:
        converted.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("transcript", ""),
            "speaker_raw": f"SPEAKER_{seg.get('speaker_id', 0):02d}",
            "speaker": seg.get("speaker_label", f"Speaker {seg.get('speaker_id', 0) + 1}")
        })
    return converted


def load_deepgram_raw_format(path: Path) -> List[Dict[str, Any]]:
    """Load .deepgram.json format and convert to standard format"""
    data = load_json(path)
    
    # Extract words from the first channel and alternative
    try:
        words = data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError):
        raise ValueError(f"Invalid Deepgram format in {path}")
    
    # Group words by speaker to create segments
    speaker_segments = defaultdict(list)
    for word in words:
        speaker_id = word.get("speaker", 0)
        speaker_segments[speaker_id].append(word)
    
    # Create segments from consecutive words by same speaker
    converted = []
    for speaker_id, speaker_words in speaker_segments.items():
        # Sort by start time
        speaker_words.sort(key=lambda w: w.get("start", 0))
        
        # Group consecutive words (simple approach: group every 10-15 words)
        segment_size = 12  # words per segment
        for i in range(0, len(speaker_words), segment_size):
            segment_words = speaker_words[i:i + segment_size]
            if not segment_words:
                continue
                
            start = segment_words[0].get("start", 0.0)
            end = segment_words[-1].get("end", 0.0)
            text = " ".join(w.get("punctuated_word", w.get("word", "")) for w in segment_words)
            
            converted.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker_raw": f"SPEAKER_{speaker_id:02d}",
                "speaker": f"Speaker {speaker_id + 1}"
            })
    
    # Sort all segments by start time
    converted.sort(key=lambda s: s.get("start", 0.0))
    return converted


def load_deepgram_transcript(path: Path) -> List[Dict[str, Any]]:
    """Auto-detect Deepgram format and load as standard segments"""
    if path.name.endswith(".sentences.json"):
        return load_deepgram_sentences_format(path)
    elif path.name.endswith(".deepgram.json"):
        return load_deepgram_raw_format(path)
    else:
        # Try to detect format by structure
        data = load_json(path)
        if "segments" in data and "transcript" in str(data):
            return load_deepgram_sentences_format(path)
        elif "results" in data and "channels" in str(data):
            return load_deepgram_raw_format(path)
        else:
            raise ValueError(f"Unknown Deepgram format for {path}")


# -----------------------------
# Sampling & prompt
# -----------------------------
def extract_speaker_samples(
    segments: List[Dict[str, Any]],
    speaker_key: str,
    max_segments_per_speaker: int = 14,
    max_chars_per_segment: int = 220,
    take_from_start: int = 250,
) -> Dict[str, List[Tuple[float, float, str]]]:
    by_spk: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    subset = segments[:take_from_start] if take_from_start > 0 else segments

    for seg in subset:
        spk = str(seg.get(speaker_key) or seg.get("speaker") or "UNKNOWN")
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) > max_chars_per_segment:
            txt = txt[: max_chars_per_segment - 1] + "…"
        by_spk[spk].append((float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), txt))

    capped: Dict[str, List[Tuple[float, float, str]]] = {}
    for spk, items in by_spk.items():
        capped[spk] = items[:max_segments_per_speaker]
    return capped


def build_prompt(samples: Dict[str, List[Tuple[float, float, str]]], filename_hint: str) -> List[Dict[str, str]]:
    speaker_list = sorted(samples.keys())

    blocks = []
    for spk in speaker_list:
        ex_lines = [f"- [{hhmmss(st)}–{hhmmss(en)}] {t}" for (st, en, t) in samples[spk]]
        blocks.append(f"{spk} examples:\n" + "\n".join(ex_lines))

    snippet_text = "\n\n".join(blocks)

    user_msg = f"""
You are labeling speakers in a podcast transcript.

Task:
- Decide which diarization speaker is the HOST (the interviewer / show runner).
- All other diarization speakers should be GUEST.
- There must be exactly ONE HOST.

Input transcript file hint: {filename_hint}

Return ONLY valid JSON in this schema:
{{
  "host_speaker_raw": "<one of the speaker ids>",
  "mapping": {{
     "<speaker id>": "HOST" | "GUEST"
  }},
  "reasoning_brief": {{
     "<speaker id>": "<1 sentence reason>"
  }},
  "confidence": {{
     "<speaker id>": 0.0-1.0
  }}
}}

Rules:
- Do NOT guess real names; only assign roles to the speaker IDs shown.
- Output JSON only.

Transcript snippets:
{snippet_text}
""".strip()

    return [
        {"role": "system", "content": "Output ONLY JSON. No markdown. No extra text."},
        {"role": "user", "content": user_msg},
    ]


# -----------------------------
# OpenRouter client (robust)
# -----------------------------
def _dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    app_title: str,
    http_referer: str,
    want_json_object: bool,
    debug_dump_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (assistant_content, full_response_json).
    Handles:
      - 404 no endpoints -> fallback to openrouter/free
      - empty content -> returns "" but still gives response JSON for debugging
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_title,
        "HTTP-Referer": http_referer,
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if want_json_object:
        payload["response_format"] = {"type": "json_object"}

    last_err: Optional[Exception] = None
    last_json: Dict[str, Any] = {}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)

            if resp.status_code == 404 and "No endpoints found" in resp.text:
                payload["model"] = "openrouter/free"
                resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)

            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1200]}")

            last_json = resp.json()

            # dump full response if requested
            if debug_dump_path is not None:
                _dump(debug_dump_path, json.dumps(last_json, ensure_ascii=False, indent=2))

            # Extract message content safely
            content = ""
            try:
                choice0 = last_json.get("choices", [{}])[0]
                msg = choice0.get("message", {}) or {}
                content = msg.get("content") or ""
                # Some providers might return `text` instead of `content`
                if not content and "text" in choice0:
                    content = choice0.get("text") or ""
            except Exception:
                content = ""

            return content, last_json

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.2 * attempt)
                continue
            raise RuntimeError(f"OpenRouter request failed after {retries} attempts: {last_err}") from last_err

    return "", last_json


def repair_to_json(
    api_key: str,
    model: str,
    bad_text: str,
    http_referer: str,
    app_title: str,
    debug_dump_path: Optional[Path] = None,
) -> str:
    repair_messages = [
        {"role": "system", "content": "Output ONLY valid JSON. No extra text."},
        {"role": "user", "content": f"""
Convert the following into VALID JSON for this schema:

{{
  "host_speaker_raw": "<one speaker id string>",
  "mapping": {{
     "<speaker id>": "HOST" | "GUEST"
  }},
  "reasoning_brief": {{
     "<speaker id>": "<1 sentence reason>"
  }},
  "confidence": {{
     "<speaker id>": 0.0-1.0
  }}
}}

Constraints:
- There must be exactly one HOST.
- mapping keys MUST be speaker IDs like SPEAKER_00, SPEAKER_01, etc.

Text to convert:
{bad_text}
""".strip()},
    ]

    repaired, _ = openrouter_chat(
        api_key=api_key,
        model=model,
        messages=repair_messages,
        temperature=0.0,
        max_tokens=500,
        timeout_s=60,
        retries=2,
        app_title=app_title,
        http_referer=http_referer,
        want_json_object=True,
        debug_dump_path=debug_dump_path,
    )
    return repaired


# -----------------------------
# Heuristic fallback (always works)
# -----------------------------
HOST_CUE_PATTERNS = [
    r"\bepisode\b",
    r"\bpodcast\b",
    r"\bwelcome\b",
    r"\bthank(s)?\b",
    r"\bspecial thanks\b",
    r"\bsubscribe\b",
    r"\bfollow\b",
    r"\bTRS\b",
    r"\bshow\b",
    r"\bwe('ve)?\b",
    r"\?",
]


def heuristic_host(samples: Dict[str, List[Tuple[float, float, str]]]) -> str:
    """
    Score each speaker by:
      - how early they speak
      - how many "host cue" tokens appear in their early snippets
      - total characters (hosts often do intros)
      - question marks (interviewers ask questions)
    """
    cue_re = re.compile("|".join(HOST_CUE_PATTERNS), flags=re.IGNORECASE)
    best_spk = None
    best_score = -1e18

    for spk, items in samples.items():
        if not items:
            continue
        first_t = items[0][0]
        early_bonus = -first_t  # earlier => higher score

        cue_hits = 0
        char_count = 0
        question_marks = 0
        for _, _, t in items[:10]:
            char_count += len(t)
            cue_hits += len(cue_re.findall(t))
            question_marks += t.count('?')

        score = (early_bonus * 0.5) + (cue_hits * 10.0) + (char_count * 0.01) + (question_marks * 5.0)
        if score > best_score:
            best_score = score
            best_spk = spk

    return best_spk or list(samples.keys())[0]


# -----------------------------
# Labeling + output
# -----------------------------
def label_transcript(
    obj: Dict[str, Any],
    mapping: Dict[str, str],
    speaker_key: str,
    add_role_to_words: bool = False,
) -> Dict[str, Any]:
    for seg in obj.get("segments", []):
        spk = str(seg.get(speaker_key) or seg.get("speaker") or "UNKNOWN")
        role = mapping.get(spk, "GUEST")
        seg["speaker_role"] = role
        if add_role_to_words and isinstance(seg.get("words"), list):
            for w in seg["words"]:
                if isinstance(w, dict):
                    w["speaker_role"] = role

    obj["speaker_role_mapping"] = mapping
    return obj


def build_txt_lines(obj: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for seg in obj.get("segments", []):
        start = float(seg.get("start", 0.0))
        role = seg.get("speaker_role", "GUEST")
        txt = (seg.get("text") or "").strip()
        if txt:
            lines.append(f"[{hhmmss(start)}] {role}: {txt}")
    return lines


# -----------------------------
# Single file processing
# -----------------------------
def process_single_file(
    in_path: Path,
    out_dir: Path,
    api_key: str,
    model: str,
    speaker_key: str,
    max_segments_per_speaker: int,
    take_from_start: int,
    temperature: float,
    max_tokens: int,
    add_role_to_words: bool,
    http_referer: str,
    app_title: str,
    debug_dump_raw: bool,
) -> Dict[str, Any]:
    """Process a single Deepgram transcript file and return results."""
    
    # Load Deepgram format and convert to standard segments
    try:
        segments = load_deepgram_transcript(in_path)
    except Exception as e:
        return {"file": str(in_path), "error": f"Failed to load Deepgram transcript: {e}"}

    if not isinstance(segments, list) or not segments:
        return {"file": str(in_path), "error": "No segments found in transcript"}

    # Create standard object format
    obj = {"segments": segments}

    samples = extract_speaker_samples(
        segments=segments,
        speaker_key=speaker_key,
        max_segments_per_speaker=max_segments_per_speaker,
        take_from_start=take_from_start,
    )
    if len(samples) < 2:
        return {"file": str(in_path), "error": "Need at least 2 speakers"}

    # Prepare debug paths
    base = in_path.stem
    debug_primary = out_dir / f"{base}.openrouter_primary.response.json" if debug_dump_raw else None
    debug_repair = out_dir / f"{base}.openrouter_repair.response.json" if debug_dump_raw else None
    debug_text_primary = out_dir / f"{base}.openrouter_primary.content.txt" if debug_dump_raw else None
    debug_text_repair = out_dir / f"{base}.openrouter_repair.content.txt" if debug_dump_raw else None

    # LLM attempt
    host_id: Optional[str] = None
    try:
        messages = build_prompt(samples, filename_hint=in_path.name)
        raw, full_json = openrouter_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=60,
            retries=3,
            app_title=app_title,
            http_referer=http_referer,
            want_json_object=True,
            debug_dump_path=debug_primary,
        )
        if debug_text_primary is not None:
            _dump(debug_text_primary, raw)

        result = safe_parse_json(raw)

    except Exception:
        # Repair attempt if we got SOME text; if empty, skip straight to heuristic
        try:
            if "raw" in locals() and (raw or "").strip():
                repaired = repair_to_json(
                    api_key=api_key,
                    model=model,
                    bad_text=raw,
                    http_referer=http_referer,
                    app_title=app_title,
                    debug_dump_path=debug_repair,
                )
                if debug_text_repair is not None:
                    _dump(debug_text_repair, repaired)
                result = safe_parse_json(repaired)
            else:
                result = {}
        except Exception:
            result = {}

    # Decide host_id (LLM result or heuristic)
    if isinstance(result, dict):
        candidate = result.get("host_speaker_raw")
        if candidate and str(candidate) in samples:
            host_id = str(candidate)

    if not host_id:
        host_id = heuristic_host(samples)

    # Enforce exactly one host
    mapping = {spk: ("HOST" if spk == host_id else "GUEST") for spk in samples.keys()}

    updated = label_transcript(obj, mapping, speaker_key=speaker_key, add_role_to_words=add_role_to_words)

    out_json = out_dir / f"{base}.host_guest.json"
    out_txt = out_dir / f"{base}.host_guest.txt"
    write_json(out_json, updated)
    write_txt(out_txt, build_txt_lines(updated))

    return {
        "file": str(in_path),
        "ok": True,
        "host_speaker": host_id,
        "output_txt": str(out_txt),
        "output_json": str(out_json),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Single input file")
    input_group.add_argument("--input_dir", help="Directory containing Deepgram transcript files")
    
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model", default="openrouter/free")
    ap.add_argument("--speaker_key", default="speaker_raw")
    ap.add_argument("--max_segments_per_speaker", type=int, default=14)
    ap.add_argument("--take_from_start", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=650)
    ap.add_argument("--add_role_to_words", action="store_true")
    ap.add_argument("--http_referer", default="http://localhost")
    ap.add_argument("--app_title", default="podcast-host-guest-labeler")
    ap.add_argument("--debug_dump_raw", action="store_true")
    ap.add_argument("--pattern", default="*.sentences.json", help="File pattern for batch processing")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files to process
    files_to_process = []
    
    if args.input:
        input_path = Path(args.input)
        if input_path.exists() and input_path.is_file():
            files_to_process.append(input_path)
        else:
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(2)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"ERROR: Input directory not found: {args.input_dir}", file=sys.stderr)
            sys.exit(2)
        
        # Find files matching pattern
        files_to_process = list(input_dir.glob(args.pattern))
        if not files_to_process:
            print(f"ERROR: No files found matching pattern '{args.pattern}' in {args.input_dir}", file=sys.stderr)
            sys.exit(2)
    
    print(f"Processing {len(files_to_process)} file(s)...")

    # Process files
    results = []
    for file_path in files_to_process:
        print(f"\nProcessing: {file_path.name}")
        result = process_single_file(
            in_path=file_path,
            out_dir=out_dir,
            api_key=api_key,
            model=args.model,
            speaker_key=args.speaker_key,
            max_segments_per_speaker=args.max_segments_per_speaker,
            take_from_start=args.take_from_start,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            add_role_to_words=args.add_role_to_words,
            http_referer=args.http_referer,
            app_title=args.app_title,
            debug_dump_raw=args.debug_dump_raw,
        )
        results.append(result)
        
        if result.get("ok"):
            print(f"  ✓ Host: {result['host_speaker']}")
            print(f"  ✓ Output: {Path(result['output_txt']).name}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")

    # Summary
    print(f"\n=== Summary ===")
    successful = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for result in failed:
            print(f"  {Path(result['file']).name}: {result.get('error', 'Unknown error')}")
    
    if len(failed) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

