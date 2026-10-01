#!/usr/bin/env python3
"""
Parse diarized transcript lines like:

[00:00:00.351 - 00:00:03.234] SPEAKER_01 (male, HOST): What's your message...
[00:00:09.479 - 00:00:11.640] SPEAKER_00 (male, GUEST): You know...

Then detect overlapping speech ("interruptions"):
- Overlap event when B.start < A.end and overlap >= overlap_threshold seconds
- Attribute direction as: interrupter = B, interrupted = A

Outputs:
- overlap_events.csv
- overlap_matrix.csv   (grouped by ROLE_gender, e.g., HOST_male -> GUEST_female)

Usage:
  python detect_overlaps_from_format.py --input transcript.txt --out_dir out
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict


LINE_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s+"
    r"(?P<speaker>SPEAKER_\d+)\s+\((?P<gender>[^,]+),\s*(?P<role>[^)]+)\):\s*(?P<text>.*)$"
)

BACKCHANNEL_RE = re.compile(
    r"^\s*(yeah|yep|mm+[- ]?hmm+|uh[- ]?huh|right|exactly|totally|sure|ok(ay)?|mhm)\s*$",
    re.IGNORECASE
)


@dataclass
class Turn:
    episode_id: str
    speaker_id: str
    gender: str
    role: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def group(self) -> str:
        # e.g., HOST_male, GUEST_female
        return f"{self.role.upper()}_{self.gender.lower()}"


def ts_to_seconds(ts: str) -> float:
    # "HH:MM:SS.mmm"
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_file(path: str, episode_id: str) -> List[Turn]:
    turns: List[Turn] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                # Skip non-matching lines quietly (or raise if you prefer)
                continue
            start = ts_to_seconds(m.group("start"))
            end = ts_to_seconds(m.group("end"))
            if end < start:
                start, end = end, start

            turns.append(
                Turn(
                    episode_id=episode_id,
                    speaker_id=m.group("speaker").strip(),
                    gender=m.group("gender").strip().lower(),
                    role=m.group("role").strip().upper(),
                    start=float(start),
                    end=float(end),
                    text=m.group("text").strip()
                )
            )
    # Sort by start time
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def is_backchannel(text: str, max_words: int = 4) -> bool:
    if not text:
        return False
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) <= max_words and BACKCHANNEL_RE.match(text.strip()):
        return True
    return False


def detect_overlap_events(
    turns: List[Turn],
    overlap_threshold: float = 0.50,
    merge_window: float = 0.50,
    ignore_same_speaker: bool = True,
) -> List[dict]:
    """
    Sweep-line overlap detection.
    For each incoming turn B, compare with "active" turns A where A.end > B.start.
    If B.start occurs during A and overlap >= threshold => B interrupts A.
    """
    events: List[dict] = []
    active: List[Turn] = []

    for B in turns:
        active = [A for A in active if A.end > B.start]

        for A in active:
            if ignore_same_speaker and A.speaker_id == B.speaker_id:
                continue
            if B.start < A.end and B.start >= A.start:
                overlap_sec = min(A.end, B.end) - B.start
                if overlap_sec >= overlap_threshold:
                    events.append({
                        "episode_id": B.episode_id,
                        "interrupter_speaker": B.speaker_id,
                        "interrupted_speaker": A.speaker_id,
                        "interrupter_group": B.group,
                        "interrupted_group": A.group,
                        "start": round(B.start, 3),
                        "end": round(min(A.end, B.end), 3),
                        "overlap_sec": round(overlap_sec, 3),
                        "is_backchannel": int(is_backchannel(B.text)),
                        "interrupter_text": B.text,
                        "interrupted_text": A.text,
                    })

        active.append(B)

    # Merge near-duplicate events caused by diarization fragmentation
    if not events:
        return events

    events.sort(key=lambda e: (
        e["episode_id"], e["interrupter_speaker"], e["interrupted_speaker"], e["start"]
    ))

    merged = []
    prev = events[0]
    for e in events[1:]:
        same_pair = (
            e["episode_id"] == prev["episode_id"]
            and e["interrupter_speaker"] == prev["interrupter_speaker"]
            and e["interrupted_speaker"] == prev["interrupted_speaker"]
        )
        close = (e["start"] - prev["end"]) <= merge_window
        if same_pair and close:
            prev["end"] = max(prev["end"], e["end"])
            prev["overlap_sec"] = round(prev["overlap_sec"] + e["overlap_sec"], 3)
            prev["is_backchannel"] = int(prev["is_backchannel"] and e["is_backchannel"])
        else:
            merged.append(prev)
            prev = e
    merged.append(prev)

    return merged


def write_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        # Write empty file with a minimal header
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_group_matrix(events: List[dict], include_backchannels: bool = False) -> Tuple[List[str], List[List[int]]]:
    groups = set()
    for e in events:
        if (not include_backchannels) and int(e["is_backchannel"]) == 1:
            continue
        groups.add(e["interrupter_group"])
        groups.add(e["interrupted_group"])
    groups = sorted(groups)

    idx = {g: i for i, g in enumerate(groups)}
    mat = [[0 for _ in groups] for __ in groups]

    for e in events:
        if (not include_backchannels) and int(e["is_backchannel"]) == 1:
            continue
        i = idx[e["interrupter_group"]]
        j = idx[e["interrupted_group"]]
        mat[i][j] += 1

    return groups, mat


def write_matrix(groups: List[str], mat: List[List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from\\to"] + groups)
        for i, g in enumerate(groups):
            w.writerow([g] + mat[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to transcript .txt with your format.")
    ap.add_argument("--episode_id", default="ep001", help="Episode ID label to store in outputs.")
    ap.add_argument("--out_dir", default="out", help="Output directory.")
    ap.add_argument("--overlap_threshold", type=float, default=0.50, help="Minimum overlap seconds.")
    ap.add_argument("--merge_window", type=float, default=0.50, help="Merge duplicate events within this gap.")
    ap.add_argument("--include_backchannels", action="store_true", help="Include backchannels in matrix.")
    args = ap.parse_args()

    turns = parse_file(args.input, episode_id=args.episode_id)
    events = detect_overlap_events(
        turns,
        overlap_threshold=args.overlap_threshold,
        merge_window=args.merge_window
    )

    events_path = os.path.join(args.out_dir, "overlap_events.csv")
    matrix_path = os.path.join(args.out_dir, "overlap_matrix.csv")

    write_csv(events, events_path)

    groups, mat = build_group_matrix(events, include_backchannels=args.include_backchannels)
    write_matrix(groups, mat, matrix_path)

    print(f"[OK] Parsed turns: {len(turns)}")
    print(f"[OK] Overlap events: {len(events)} (threshold={args.overlap_threshold}s)")
    print(f"[OK] Wrote: {events_path}")
    print(f"[OK] Wrote: {matrix_path}")


if __name__ == "__main__":
    main()