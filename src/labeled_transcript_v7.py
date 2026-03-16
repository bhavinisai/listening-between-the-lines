#!/usr/bin/env python3
import os, json, re, argparse
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var


def sec_to_hhmmssmmm(t: float) -> str:
    ms = int(round(float(t) * 1000))
    s, msec = divmod(ms, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{msec:03d}"


def choose_top_speakers_by_talktime(segments, k=2):
    dur = defaultdict(float)
    for seg in segments:
        spk = seg.get("speaker_label") or f"Speaker {seg.get('speaker_id')}"
        dur[spk] += max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
    top = sorted(dur.items(), key=lambda x: x[1], reverse=True)[:k]
    return [s for s, _ in top]


def build_excerpt(segments, speakers, max_segments=240):
    filtered = [s for s in segments if (s.get("speaker_label") in speakers)]
    if len(filtered) > max_segments:
        n1 = max_segments // 2
        n2 = max_segments - n1
        filtered = filtered[:n1] + filtered[-n2:]

    lines = []
    for seg in filtered:
        txt = (seg.get("transcript") or "").strip()
        if not txt:
            continue
        spk = seg.get("speaker_label") or f"Speaker {seg.get('speaker_id')}"
        lines.append(
            f"[{sec_to_hhmmssmmm(seg.get('start',0))} - {sec_to_hhmmssmmm(seg.get('end',0))}] {spk}: {txt}"
        )
    return "\n".join(lines)


def gpt_label_roles(excerpt: str, speakers: list, model: str):
    """
    Compatible with openai-python 2.23.0 and older:
    uses chat.completions (no response_format).
    Returns:
      roles_map: {"Speaker 1":"HOST","Speaker 2":"GUEST"}
      meta: {"host_speaker":..., "guest_speaker":..., "confidence":..., "evidence":[...]}
    """
    import json
    import re

    system = (
        "You label which diarized speaker is the HOST and which is the GUEST in a podcast transcript.\n"
        "HOST cues: introduces guest/show, addresses audience, asks most questions, controls transitions/outro.\n"
        "GUEST cues: answers questions, longer explanations.\n"
        "Return ONLY valid JSON with keys: host_speaker, guest_speaker, confidence, evidence.\n"
        "confidence must be a number in [0,1]. evidence must be an array of 2-6 short strings."
    )

    user = (
        f"Speakers to label: {', '.join(speakers)}\n\n"
        f"Transcript excerpt:\n{excerpt}\n\n"
        "Return JSON only. Example:\n"
        f'{{"host_speaker":"{speakers[0]}","guest_speaker":"{speakers[1]}","confidence":0.9,'
        f'"evidence":["host asks questions","host introduces guest"]}}'
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=250,
    )

    content = (resp.choices[0].message.content or "").strip()

    # Strip ```json fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)

    host = data["host_speaker"]
    guest = data["guest_speaker"]
    if host == guest and len(speakers) >= 2:
        host, guest = speakers[0], speakers[1]
        data["host_speaker"] = host
        data["guest_speaker"] = guest

    return {host: "HOST", guest: "GUEST"}, data


def rewrite_json(obj: dict, roles: dict):
    out = dict(obj)
    out["speaker_roles"] = dict(roles)

    new_segments = []
    for seg in out.get("segments", []):
        s2 = dict(seg)
        spk = s2.get("speaker_label") or f"Speaker {s2.get('speaker_id')}"
        role = roles.get(spk, "UNKNOWN")
        s2["role"] = role
        s2["speaker_role_label"] = f"{role} ({spk})" if role != "UNKNOWN" else spk
        new_segments.append(s2)

    out["segments"] = new_segments
    return out


def rewrite_txt(txt: str, roles: dict) -> str:
    """
    Rewrites lines like:
      [..] Speaker 1: text
    to:
      [..] HOST (Speaker 1): text
    """
    pat = re.compile(r"^(\[.*?\]\s*)(Speaker\s+\d+)(:\s*)(.*)$")
    out_lines = []
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if not m:
            out_lines.append(line)
            continue
        prefix, spk, colon, rest = m.groups()
        role = roles.get(spk, "UNKNOWN")
        if role == "UNKNOWN":
            out_lines.append(line)
        else:
            out_lines.append(f"{prefix}{role} ({spk}){colon}{rest}")
    return "\n".join(out_lines) + ("\n" if txt.endswith("\n") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-in", required=True, help="Input diarized JSON file")
    ap.add_argument("--txt-in", required=True, help="Input diarized TXT file")
    ap.add_argument("--json-out", required=True, help="Output labeled JSON file")
    ap.add_argument("--txt-out", required=True, help="Output labeled TXT file")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model name")
    ap.add_argument("--max-snippet-segments", type=int, default=240)
    args = ap.parse_args()

    json_in = Path(args.json_in)
    txt_in = Path(args.txt_in)

    obj = json.loads(json_in.read_text(encoding="utf-8"))
    segments = obj.get("segments", [])
    if not segments:
        raise SystemExit("No segments found in JSON.")

    top2 = choose_top_speakers_by_talktime(segments, k=2)
    if len(top2) < 2:
        raise SystemExit("Need at least 2 speakers to label host vs guest.")

    excerpt = build_excerpt(segments, top2, max_segments=args.max_snippet_segments)
    roles_top2, meta = gpt_label_roles(excerpt, top2, model=args.model)

    # Mark any other speakers UNKNOWN
    all_speakers = sorted({(s.get("speaker_label") or f"Speaker {s.get('speaker_id')}") for s in segments})
    roles = {spk: roles_top2.get(spk, "UNKNOWN") for spk in all_speakers}

    labeled_json = rewrite_json(obj, roles)
    labeled_txt = rewrite_txt(txt_in.read_text(encoding="utf-8"), roles)

    Path(args.json_out).write_text(json.dumps(labeled_json, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.txt_out).write_text(labeled_txt, encoding="utf-8")

    print("Done.")
    print("Roles:", roles_top2)
    print(f"Confidence: {meta['confidence']:.2f}")
    print(f"Wrote JSON -> {args.json_out}")
    print(f"Wrote TXT  -> {args.txt_out}")


if __name__ == "__main__":
    main()

