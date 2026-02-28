#!/usr/bin/env python3
"""
labeled_transcript_v8_robust.py

- Loads WhisperX transcript JSON (expects `segments[]` with `speaker_raw` + `text`)
- Uses OpenRouter to pick HOST speaker (exactly one) and labels others as GUEST
- Writes:
    1) <stem>.host_guest.txt
    2) <stem>.host_guest.json (adds speaker_role per segment, plus speaker_role_mapping)

Robustness improvements:
- Captures full OpenRouter response JSON when --debug_dump_raw is set
- Handles empty model content (common with router models)
- Uses response_format=json_object when possible
- If parse fails, tries a repair call
- If still fails, falls back to heuristic host detection (so it ALWAYS outputs files)

Usage:
  pip install requests
  export OPENROUTER_API_KEY="..."

  python src/host_guest_label/labeled_transcript_v8_robust.py \
    --input outputs/Episode01_RanveerAllahbadia_DrJaishankar.whisperx.json \
    --out_dir outputs \
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
]


def heuristic_host(samples: Dict[str, List[Tuple[float, float, str]]]) -> str:
    """
    Score each speaker by:
      - how early they speak
      - how many "host cue" tokens appear in their early snippets
      - total characters (hosts often do intros)
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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
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
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = load_json(in_path)
    segments = obj.get("segments", [])
    if not isinstance(segments, list) or not segments:
        print("ERROR: input JSON missing segments[]", file=sys.stderr)
        sys.exit(2)

    samples = extract_speaker_samples(
        segments=segments,
        speaker_key=args.speaker_key,
        max_segments_per_speaker=args.max_segments_per_speaker,
        take_from_start=args.take_from_start,
    )
    if len(samples) < 2:
        print("ERROR: need at least 2 speakers", file=sys.stderr)
        sys.exit(2)

    # Prepare debug paths
    base = in_path.stem
    debug_primary = out_dir / f"{base}.openrouter_primary.response.json" if args.debug_dump_raw else None
    debug_repair = out_dir / f"{base}.openrouter_repair.response.json" if args.debug_dump_raw else None
    debug_text_primary = out_dir / f"{base}.openrouter_primary.content.txt" if args.debug_dump_raw else None
    debug_text_repair = out_dir / f"{base}.openrouter_repair.content.txt" if args.debug_dump_raw else None

    # LLM attempt
    host_id: Optional[str] = None
    try:
        messages = build_prompt(samples, filename_hint=in_path.name)
        raw, full_json = openrouter_chat(
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=60,
            retries=3,
            app_title=args.app_title,
            http_referer=args.http_referer,
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
                    model=args.model,
                    bad_text=raw,
                    http_referer=args.http_referer,
                    app_title=args.app_title,
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

    updated = label_transcript(obj, mapping, speaker_key=args.speaker_key, add_role_to_words=args.add_role_to_words)

    out_json = out_dir / f"{base}.host_guest.json"
    out_txt = out_dir / f"{base}.host_guest.txt"
    write_json(out_json, updated)
    write_txt(out_txt, build_txt_lines(updated))

    print("Done.")
    print("Host diarization speaker:", host_id)
    print("Wrote:", out_txt)
    print("Wrote:", out_json)
    if args.debug_dump_raw:
        print("Debug dumps written alongside outputs (primary/repair response + content).")


if __name__ == "__main__":
    main()