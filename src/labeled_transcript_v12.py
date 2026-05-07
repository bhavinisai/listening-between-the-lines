#!/usr/bin/env python3
"""
labeled_transcript_v12.py

- Loads WhisperX transcript JSON with gender labels (output of detect_gender_v4.py)
- Identifies HOST speaker from the speaker_gender_mapping in the JSON:
    - Speaker matched to the library (source=library) → HOST
    - All others → GUEST
- Falls back to Groq API if no library match is found in the JSON
- Falls back to heuristic if Groq also fails
- Writes:
    1) <stem>.host_guest.txt
    2) <stem>.host_guest.json

Usage:
  export GROQ_API_KEY="..."   # only needed if no library match found

  python src/labeled_transcript_v12.py \
    --input data/outputs/whisperx/ep_032_whisperx_diarized.gender.json \
    --out_dir data/outputs/whisperx/ \
    --speaker_key speaker
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

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


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


def hhmmss_msec(seconds: Optional[float]) -> str:
    if seconds is None:
        return "00:00:00.000"
    total_ms = int(round(float(seconds) * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


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
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    preview = raw[:1500].replace("\n", "\\n")
    raise ValueError(
        "Could not find JSON object in model output. "
        f"Preview: {preview}"
    )


def get_segment_speaker(seg: Dict[str, Any], speaker_key: str) -> str:
    return str(seg.get(speaker_key) or seg.get("speaker") or "UNKNOWN")


def collect_all_speakers(segments: List[Dict[str, Any]], speaker_key: str) -> List[str]:
    speakers = []
    seen = set()
    for seg in segments:
        spk = get_segment_speaker(seg, speaker_key)
        if spk not in seen and spk != "UNKNOWN":
            seen.add(spk)
            speakers.append(spk)
    return speakers


# -----------------------------
# Step 1: Resolve host from speaker library match
# -----------------------------
def resolve_host_from_library(obj: Dict[str, Any]) -> Optional[str]:
    """
    Look at speaker_gender_mapping in the gender JSON.
    The speaker with source=library is the known host.
    If multiple library matches, pick the one with the highest similarity.
    Returns speaker ID (e.g. 'SPEAKER_00') or None if no library match found.
    """
    mapping = obj.get("speaker_gender_mapping", {})
    if not mapping:
        return None

    library_matches = {
        spk: info for spk, info in mapping.items()
        if info.get("source") == "library"
    }

    if not library_matches:
        return None

    # Pick the one with highest similarity score
    host_id = max(
        library_matches,
        key=lambda s: library_matches[s].get("similarity", 0.0)
    )

    name = library_matches[host_id].get("name", host_id)
    similarity = library_matches[host_id].get("similarity", "N/A")
    print(f"INFO: host identified from speaker library: "
          f"{host_id} → {name} (similarity={similarity})")
    return host_id


# -----------------------------
# Sampling & Groq prompt (fallback only)
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
        spk = get_segment_speaker(seg, speaker_key)
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) > max_chars_per_segment:
            txt = txt[: max_chars_per_segment - 1] + "…"
        by_spk[spk].append((
            float(seg.get("start", 0.0)),
            float(seg.get("end", 0.0)),
            txt
        ))

    return {spk: items[:max_segments_per_speaker] for spk, items in by_spk.items()}


def build_messages(
    samples: Dict[str, List[Tuple[float, float, str]]],
    filename_hint: str,
) -> List[Dict[str, str]]:
    speaker_list = sorted(samples.keys())
    blocks = []
    for spk in speaker_list:
        ex_lines = [f"- [{hhmmss(st)}–{hhmmss(en)}] {t}" for (st, en, t) in samples[spk]]
        blocks.append(f"{spk} examples:\n" + "\n".join(ex_lines))

    snippet_text = "\n\n".join(blocks)
    user_msg = f"""You are labeling speakers in a podcast transcript.

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
{snippet_text}""".strip()

    return [
        {"role": "system", "content": "Output ONLY JSON. No markdown. No extra text."},
        {"role": "user", "content": user_msg},
    ]


# -----------------------------
# Groq API client (fallback only)
# -----------------------------
def _dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def groq_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    debug_dump_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_err: Optional[Exception] = None
    last_json: Dict[str, Any] = {}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                GROQ_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )

            if resp.status_code == 429:
                wait = 10.0 * attempt
                retry_ms = resp.headers.get("retry-after-ms")
                retry_s  = resp.headers.get("retry-after")
                if retry_ms:
                    wait = float(retry_ms) / 1000.0 + 1.0
                elif retry_s:
                    wait = float(retry_s) + 1.0
                else:
                    m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", resp.text, re.IGNORECASE)
                    if m:
                        wait = float(m.group(1)) + 1.0
                print(f"INFO: rate limited (429), waiting {wait:.1f}s "
                      f"before retry {attempt}/{retries}...", file=sys.stderr)
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:1200]}")

            last_json = resp.json()
            if debug_dump_path is not None:
                _dump(debug_dump_path, json.dumps(last_json, ensure_ascii=False, indent=2))

            content = ""
            try:
                content = last_json["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                content = ""

            return content, last_json

        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            raise RuntimeError(
                f"Groq request failed after {retries} attempts: {last_err}"
            ) from last_err

    return "", last_json


def repair_to_json(
    api_key: str,
    model: str,
    bad_text: str,
    debug_dump_path: Optional[Path] = None,
) -> str:
    repair_messages = [
        {"role": "system", "content": "Output ONLY valid JSON. No extra text."},
        {
            "role": "user",
            "content": f"""Convert the following into VALID JSON for this schema:

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
{bad_text}""".strip(),
        },
    ]
    repaired, _ = groq_chat(
        api_key=api_key,
        model=model,
        messages=repair_messages,
        temperature=0.0,
        max_tokens=500,
        timeout_s=60,
        retries=2,
        debug_dump_path=debug_dump_path,
    )
    return repaired


# -----------------------------
# Heuristic fallback
# -----------------------------
HOST_CUE_PATTERNS = [
    r"\bepisode\b", r"\bpodcast\b", r"\bwelcome\b", r"\bthank(s)?\b",
    r"\bspecial thanks\b", r"\bsubscribe\b", r"\bfollow\b",
    r"\bshow\b", r"\bwe('ve)?\b",
]


def heuristic_host(samples: Dict[str, List[Tuple[float, float, str]]]) -> str:
    cue_re = re.compile("|".join(HOST_CUE_PATTERNS), flags=re.IGNORECASE)
    best_spk = None
    best_score = -1e18

    for spk, items in samples.items():
        if not items:
            continue
        first_t = items[0][0]
        early_bonus = -first_t
        cue_hits = 0
        char_count = 0
        for _, _, t in items[:10]:
            char_count += len(t)
            cue_hits += len(cue_re.findall(t))
        score = (early_bonus * 0.5) + (cue_hits * 10.0) + (char_count * 0.01)
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
    host_id: str,
    add_role_to_words: bool = False,
) -> Dict[str, Any]:
    for seg in obj.get("segments", []):
        spk = get_segment_speaker(seg, speaker_key)
        role = mapping.get(spk, "GUEST")
        seg["speaker_role"] = role
        if add_role_to_words and isinstance(seg.get("words"), list):
            for w in seg["words"]:
                if isinstance(w, dict):
                    w["speaker_role"] = role

    obj["speaker_role_mapping"] = mapping
    obj["host_speaker_raw"] = host_id
    return obj


def build_txt_speaker_label(seg: Dict[str, Any], speaker_key: str) -> str:
    speaker_raw = get_segment_speaker(seg, speaker_key)
    gender = str(seg.get("gender") or "").strip()
    role = str(seg.get("speaker_role") or "").strip()
    name = str(seg.get("speaker_name") or "").strip()

    # Use real name if available from library match, else use speaker ID
    display = name if name else speaker_raw

    meta: List[str] = []
    if gender and gender.lower() != "unknown":
        meta.append(gender)
    if role:
        meta.append(role)

    if meta:
        return f"{display} ({', '.join(meta)})"
    return display


def build_txt_lines(obj: Dict[str, Any], speaker_key: str) -> List[str]:
    lines: List[str] = []
    for seg in obj.get("segments", []):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue

        start = seg.get("start", 0.0)
        end = seg.get("end", None)
        time_block = (
            f"[{hhmmss_msec(start)} - {hhmmss_msec(end)}]"
            if end is not None
            else f"[{hhmmss_msec(start)}]"
        )

        speaker_label = build_txt_speaker_label(seg, speaker_key)
        lines.append(f"{time_block} {speaker_label}: {txt}")
    return lines


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Label HOST/GUEST using speaker library match, with Groq/heuristic fallback."
    )
    ap.add_argument("--input", required=True,
                    help="Path to gender-labeled WhisperX JSON (output of detect_gender_v4.py)")
    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="llama-3.3-70b-versatile",
                    help="Groq model name (used only as fallback)")
    ap.add_argument("--speaker_key", default="speaker",
                    help="Segment field name for speaker ID")
    ap.add_argument("--max_segments_per_speaker", type=int, default=14)
    ap.add_argument("--take_from_start", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=650)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--add_role_to_words", action="store_true")
    ap.add_argument("--debug_dump_raw", action="store_true",
                    help="Save raw Groq API responses to others/")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    others = out_dir / "others"
    others.mkdir(parents=True, exist_ok=True)

    obj = load_json(in_path)
    segments = obj.get("segments", [])
    if not isinstance(segments, list) or not segments:
        print("ERROR: input JSON missing segments[]", file=sys.stderr)
        sys.exit(2)

    all_speakers = collect_all_speakers(segments, args.speaker_key)
    if len(all_speakers) < 2:
        print("ERROR: need at least 2 distinct speakers", file=sys.stderr)
        sys.exit(2)

    base = in_path.stem
    host_id: Optional[str] = None
    result: Dict[str, Any] = {}

    # --- Priority 1: speaker library match ---
    host_id = resolve_host_from_library(obj)

    # --- Priority 2: Groq fallback ---
    if not host_id:
        print("INFO: no library match found in JSON, falling back to Groq...")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not set, skipping Groq fallback.", file=sys.stderr)
        else:
            samples = extract_speaker_samples(
                segments=segments,
                speaker_key=args.speaker_key,
                max_segments_per_speaker=args.max_segments_per_speaker,
                take_from_start=args.take_from_start,
            )

            debug_primary = others / f"{base}.groq_primary.response.json" if args.debug_dump_raw else None
            debug_repair  = others / f"{base}.groq_repair.response.json"  if args.debug_dump_raw else None

            try:
                messages = build_messages(samples, filename_hint=in_path.name)
                raw, _ = groq_chat(
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout_s=60,
                    retries=args.retries,
                    debug_dump_path=debug_primary,
                )
                result = safe_parse_json(raw)

            except Exception as e:
                print(f"WARNING: Groq call failed: {e}", file=sys.stderr)
                try:
                    if "raw" in locals() and (raw or "").strip():
                        repaired = repair_to_json(api_key, args.model, raw, debug_repair)
                        result = safe_parse_json(repaired)
                except Exception as e2:
                    print(f"WARNING: repair also failed: {e2}", file=sys.stderr)
                    result = {}

            if isinstance(result, dict):
                candidate = result.get("host_speaker_raw")
                if candidate and str(candidate) in all_speakers:
                    host_id = str(candidate)
                    print(f"INFO: host identified by Groq: {host_id}")

    # --- Priority 3: heuristic fallback ---
    if not host_id:
        print("INFO: falling back to heuristic host detection.", file=sys.stderr)
        samples = extract_speaker_samples(
            segments=segments,
            speaker_key=args.speaker_key,
            max_segments_per_speaker=args.max_segments_per_speaker,
            take_from_start=args.take_from_start,
        )
        host_id = heuristic_host(samples)
        print(f"INFO: host identified by heuristic: {host_id}")

    # Build mapping and label
    mapping = {
        spk: ("HOST" if spk == host_id else "GUEST")
        for spk in all_speakers
    }

    updated = label_transcript(
        obj, mapping,
        speaker_key=args.speaker_key,
        host_id=host_id,
        add_role_to_words=args.add_role_to_words,
    )

    out_json = out_dir / f"{base}.host_guest.json"
    out_txt  = out_dir / f"{base}.host_guest.txt"
    write_json(out_json, updated)
    write_txt(out_txt, build_txt_lines(updated, args.speaker_key))

    print("Done.")
    print(f"Host speaker: {host_id}")
    if isinstance(result, dict) and result.get("reasoning_brief"):
        for spk, reason in result["reasoning_brief"].items():
            role = mapping.get(spk, "?")
            print(f"  {spk} ({role}): {reason}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

