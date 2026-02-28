#!/usr/bin/env python3
"""
Reads a transcript .txt with lines like:
[00:00:00.322 - 00:00:09.053] Speaker 1: text...

Uses Gemini API to infer speaker roles (host/cohost/guest/unknown),
then rewrites the transcript by adding (HOST)/(GUEST) after speaker name.
"""

import os, re, json, argparse
from collections import Counter
from google import genai

# Matches:
# [00:00:00.322 - 00:00:09.053] Speaker 1: ...
LINE_RE = re.compile(
    r'^\s*(\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}\])\s*'
    r'([A-Za-z0-9_ .-]{1,64})\s*:\s*(.*)$'
)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_speaker_lines(text: str):
    """Return list of (timestamp, speaker, utterance, line_index)."""
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = LINE_RE.match(line)
        if not m:
            continue
        ts, speaker, utt = m.group(1), m.group(2).strip(), m.group(3).strip()
        if speaker and utt:
            out.append((ts, speaker, utt, i))
    return out

def basic_turn_stats(speaker_lines):
    if not speaker_lines:
        return {"speakers": [], "stats": {}}
    speakers = [s for (_, s, _, _) in speaker_lines]
    counts = Counter(speakers)
    avg_chars = {
        s: sum(len(u) for (_, sp, u, _) in speaker_lines if sp == s) / counts[s]
        for s in counts
    }
    first_speaker = speaker_lines[0][1]
    return {
        "speakers": sorted(counts.keys()),
        "stats": {
            "turn_counts": dict(counts),
            "avg_utterance_chars": avg_chars,
            "first_speaker": first_speaker,
            "num_turns": len(speaker_lines),
        }
    }

def make_sample(text: str, max_chars=12000) -> str:
    # Include beginning+middle+end for intros/outros
    lines = text.splitlines()
    n = len(lines)
    head = "\n".join(lines[: min(n, max(60, n // 10))])
    mid_start = max(0, n // 2 - 40)
    mid = "\n".join(lines[mid_start: min(n, mid_start + 80)])
    tail_start = max(0, n - min(n, max(60, n // 10)))
    tail = "\n".join(lines[tail_start:])
    sample = f"=== BEGINNING ===\n{head}\n\n=== MIDDLE ===\n{mid}\n\n=== END ===\n{tail}\n"
    return sample[:max_chars]

def infer_roles_gemini(api_key: str, model: str, sample_text: str, stats: dict) -> dict:
    client = genai.Client(api_key=api_key)

    response_schema = {
        "type": "object",
        "properties": {
            "speaker_roles": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": ["host", "cohost", "guest", "unknown"]
                }
            },
            "confidence": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            },
            "notes": {"type": "string"}
        },
        "required": ["speaker_roles", "confidence"]
    }

    system_instruction = (
        "You label podcast roles from a transcript sample.\n"
        "Use cues: intro/outro, addressing audience, asking most questions, sponsor reads, segment steering.\n"
        "Return ONLY JSON matching the schema. If unsure, use 'unknown'."
    )

    payload = {
        "task": "Infer role per speaker label.",
        "speaker_stats": stats,
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
    # normalize
    roles = {}
    for spk, role in data.get("speaker_roles", {}).items():
        r = (role or "").strip().lower()
        roles[spk] = r if r in {"host", "cohost", "guest", "unknown"} else "unknown"
    data["speaker_roles"] = roles
    return data

def role_tag(role: str) -> str:
    role = (role or "unknown").lower()
    return {"host": "HOST", "cohost": "CO-HOST", "guest": "GUEST"}.get(role, "UNKNOWN")

def rewrite_transcript(text: str, speaker_roles: dict) -> str:
    out = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            out.append(line)
            continue

        ts, speaker, utt = m.group(1), m.group(2).strip(), m.group(3)

        # Avoid double labeling if rerun
        if re.search(r"\((HOST|CO-HOST|GUEST|UNKNOWN)\)\s*$", speaker, re.IGNORECASE):
            out.append(line)
            continue

        tag = role_tag(speaker_roles.get(speaker, "unknown"))
        out.append(f"{ts} {speaker} ({tag}): {utt}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default=None)
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--max-sample-chars", type=int, default=12000)
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY env var.")

    text = read_text(args.in_path)
    speaker_lines = extract_speaker_lines(text)
    if not speaker_lines:
        raise SystemExit("No timestamp+speaker lines matched. Check format vs regex.")

    stats = basic_turn_stats(speaker_lines)
    sample = make_sample(text, max_chars=args.max_sample_chars)

    result = infer_roles_gemini(api_key, args.model, sample, stats)
    labeled = rewrite_transcript(text, result["speaker_roles"])

    out_path = args.out_path or args.in_path
    write_text(out_path, labeled)

    print("Role mapping:")
    for spk in sorted(result["speaker_roles"]):
        conf = result.get("confidence", {}).get(spk)
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        print(f"  {spk:>16} -> {result['speaker_roles'][spk]:7} (conf={conf_s})")

    if result.get("notes"):
        print("\nNotes:", result["notes"])

if __name__ == "__main__":
    main()
