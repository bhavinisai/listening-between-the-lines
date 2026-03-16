#!/usr/bin/env python3
"""
Label podcast HOST/GUEST (and CO-HOST/UNKNOWN) in a timestamped transcript using Gemini API.

Expected input lines:
  [00:00:00.240 - 00:00:03.540] Speaker 1: ...
  [00:00:09.440 - 00:00:13.780] Speaker 2: ...
  [..] Unknown: ...

Output:
  [00:00:00.240 - 00:00:03.540] Speaker 1 (HOST): ...
  [00:00:09.440 - 00:00:13.780] Speaker 2 (GUEST): ...

Install:
  pip install google-genai

Set key:
  export GOOGLE_API_KEY="YOUR_KEY"

Run:
  python label_podcast_roles_gemini.py --in transcript.txt --out transcript_labeled.txt
  # overwrite in place:
  python label_podcast_roles_gemini.py --in transcript.txt
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple

from google import genai

# Timestamp format you showed: [HH:MM:SS.mmm - HH:MM:SS.mmm]
TS_SPK_RE = re.compile(
    r'^\s*'
    r'(\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}\])\s*'
    r'([A-Za-z0-9_ .-]{1,64})\s*:\s*(.*)$'
)

# Detect if speaker already has label: "Speaker 1 (HOST)"
ALREADY_LABELED = re.compile(r'^(.*)\s+\((HOST|CO-HOST|GUEST|UNKNOWN)\)\s*$', re.IGNORECASE)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_speaker_lines(text: str) -> List[Tuple[str, str, str, int]]:
    """
    Return list of (timestamp, speaker_label, utterance, line_idx) for matching lines.
    speaker_label is normalized to remove any existing (HOST)/(GUEST) tag if present.
    """
    lines = text.splitlines()
    out: List[Tuple[str, str, str, int]] = []

    for i, line in enumerate(lines):
        m = TS_SPK_RE.match(line)
        if not m:
            continue
        ts = m.group(1).strip()
        speaker = m.group(2).strip()
        utt = m.group(3).strip()

        if not speaker:
            continue

        # Strip already-added tag in speaker label
        m2 = ALREADY_LABELED.match(speaker)
        if m2:
            speaker = m2.group(1).strip()

        # Allow empty utterances, but they won't help the model much
        out.append((ts, speaker, utt, i))

    return out


def basic_turn_stats(speaker_lines: List[Tuple[str, str, str, int]]) -> Dict[str, Any]:
    """
    Basic stats: turn counts, avg utterance length, first speaker.
    """
    if not speaker_lines:
        return {"speakers": [], "stats": {}}

    speakers = [s for (_, s, _, _) in speaker_lines]
    counts = Counter(speakers)

    avg_chars: Dict[str, float] = {}
    for s in counts:
        utts = [u for (_, sp, u, _) in speaker_lines if sp == s and u]
        avg_chars[s] = (sum(len(u) for u in utts) / len(utts)) if utts else 0.0

    first_speaker = speakers[0]
    return {
        "speakers": sorted(counts.keys()),
        "stats": {
            "turn_counts": dict(counts),
            "avg_utterance_chars": avg_chars,
            "first_speaker": first_speaker,
            "num_turns": len(speaker_lines),
        }
    }


def make_representative_sample(text: str, max_chars: int = 12000) -> str:
    """
    Include beginning+middle+end; role cues often appear in intro/outro.
    """
    lines = text.splitlines()
    n = len(lines)

    head = "\n".join(lines[: min(n, max(80, n // 10))])
    mid_start = max(0, n // 2 - 60)
    mid = "\n".join(lines[mid_start: min(n, mid_start + 120)])
    tail_start = max(0, n - min(n, max(80, n // 10)))
    tail = "\n".join(lines[tail_start:])

    sample = f"=== BEGINNING ===\n{head}\n\n=== MIDDLE ===\n{mid}\n\n=== END ===\n{tail}\n"
    return sample[:max_chars]


def infer_roles_gemini(
    api_key: str,
    model: str,
    sample_text: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    IMPORTANT: Gemini response_schema does NOT support additionalProperties.
    So we request: {"assignments": [{"speaker": "...", "role": "...", "confidence": 0.xx}, ...], "notes": "..."}.

    Returns:
      {
        "speaker_roles": {speaker: role},
        "confidence": {speaker: conf},
        "notes": str
      }
    """
    client = genai.Client(api_key=api_key)

    response_schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["host", "cohost", "guest", "unknown"]
                        },
                        "confidence": {"type": "number"}
                    },
                    "required": ["speaker", "role", "confidence"]
                }
            },
            "notes": {"type": "string"}
        },
        "required": ["assignments"]
    }

    system_instruction = (
        "You are labeling roles in a podcast transcript.\n"
        "Task: for each speaker label (e.g., 'Speaker 1', 'Speaker 2', 'Unknown'), infer role:\n"
        "- host: leads, introduces guest, addresses audience, asks most questions, does outro\n"
        "- cohost: also leads/questions alongside host\n"
        "- guest: primarily answers, is introduced\n"
        "- unknown: unsure\n"
        "Be conservative: if uncertain, return 'unknown'.\n"
        "Speaker names MUST exactly match the labels in the transcript.\n"
        "Return ONLY JSON matching the schema."
    )

    payload = {
        "speaker_labels": stats.get("speakers", []),
        "speaker_stats": stats.get("stats", {}),
        "transcript_sample": sample_text
    }

    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.2,
        },
    )

    data = json.loads(resp.text)

    roles: Dict[str, str] = {}
    conf: Dict[str, float] = {}

    for item in data.get("assignments", []):
        spk = (item.get("speaker") or "").strip()
        role = (item.get("role") or "").strip().lower()
        c = item.get("confidence", 0.0)

        if not spk:
            continue
        if role not in {"host", "cohost", "guest", "unknown"}:
            role = "unknown"

        roles[spk] = role
        conf[spk] = float(c) if isinstance(c, (int, float)) else 0.0

    return {"speaker_roles": roles, "confidence": conf, "notes": data.get("notes", "")}


def role_tag(role: str) -> str:
    role = (role or "unknown").lower()
    if role == "host":
        return "HOST"
    if role == "cohost":
        return "CO-HOST"
    if role == "guest":
        return "GUEST"
    return "UNKNOWN"


def rewrite_transcript(text: str, speaker_roles: Dict[str, str]) -> str:
    """
    Rewrites:
      [ts] Speaker 1: text
    to:
      [ts] Speaker 1 (HOST): text
    """
    out_lines = []
    for line in text.splitlines():
        m = TS_SPK_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        ts = m.group(1).strip()
        speaker = m.group(2).strip()
        utt = m.group(3)

        # Strip any existing label
        m2 = ALREADY_LABELED.match(speaker)
        if m2:
            base_speaker = m2.group(1).strip()
        else:
            base_speaker = speaker

        tag = role_tag(speaker_roles.get(base_speaker, "unknown"))
        out_lines.append(f"{ts} {base_speaker} ({tag}): {utt}")

    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input transcript .txt")
    ap.add_argument("--out", dest="out_path", default=None, help="Output labeled .txt (default overwrites input)")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name (e.g., gemini-2.0-flash)")
    ap.add_argument("--max-sample-chars", type=int, default=12000, help="Max chars sent to Gemini")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY environment variable.")

    text = read_text(args.in_path)
    speaker_lines = extract_speaker_lines(text)

    if not speaker_lines:
        raise SystemExit(
            "No lines matched the expected format.\n"
            "Expected: [00:00:00.240 - 00:00:03.540] Speaker 1: ..."
        )

    stats = basic_turn_stats(speaker_lines)
    sample = make_representative_sample(text, max_chars=args.max_sample_chars)

    result = infer_roles_gemini(api_key=api_key, model=args.model, sample_text=sample, stats=stats)

    labeled = rewrite_transcript(text, result["speaker_roles"])
    out_path = args.out_path or args.in_path
    write_text(out_path, labeled)

    # Print mapping for your logs
    print("Role mapping (speaker -> role):")
    for spk in sorted(stats["speakers"]):
        role = result["speaker_roles"].get(spk, "unknown")
        c = result["confidence"].get(spk, None)
        c_str = f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"
        print(f"  {spk:>16} -> {role:7} (conf={c_str})")

    if result.get("notes"):
        print("\nNotes:", result["notes"])


if __name__ == "__main__":
    main()

