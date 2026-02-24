#!/usr/bin/env python3
"""
Convert Deepgram pre-recorded transcription JSON into a diarized TXT format:

[HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker 1: ...

Assumes you requested --diarize --utterances in Deepgram.
If utterances are missing, it will fall back to word-level diarization if present.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple, Optional


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
    # Your example starts at "Speaker 1", so +1
    return f"Speaker {speaker_id + 1}"


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


def words_to_segments(words: List[Dict[str, Any]], gap_s: float = 0.8) -> List[Tuple[float, float, int, str]]:
    """
    Build speaker segments from word-level diarization.
    Groups consecutive words with same speaker; splits if there's a long gap.
    Returns list of (start, end, speaker_id, text).
    """
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


def compute_stats(segments: List[Tuple[float, float, int, str]]) -> Dict[str, Any]:
    talk_time = {}
    turns = {}
    for start, end, spk, _ in segments:
        dur = max(0.0, end - start)
        talk_time[spk] = talk_time.get(spk, 0.0) + dur
        turns[spk] = turns.get(spk, 0) + 1

    # sort speakers by total talk time desc
    speakers_sorted = sorted(talk_time.keys(), key=lambda s: talk_time[s], reverse=True)
    return {
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing *.deepgram.json")
    ap.add_argument("--out-dir", required=True, help="Directory to write diarized *.txt")
    ap.add_argument("--gap-s", type=float, default=0.8, help="Gap threshold (sec) for word-based segmentation.")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.deepgram.json"))
    if not json_files:
        print(f"No *.deepgram.json files found in {in_dir}")
        return 2

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))

        utterances = get_utterances(data)
        if utterances:
            segments = utterances_to_segments(utterances)
        else:
            words = get_words(data)
            segments = words_to_segments(words, gap_s=args.gap_s)

        # Write diarized text
        lines = []
        for start, end, spk, text in segments:
            lines.append(f"[{sec_to_timestamp(start)} - {sec_to_timestamp(end)}] {speaker_label(spk)}: {text}")

        txt_path = out_dir / jf.name.replace(".deepgram.json", ".diarized.txt")
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Write stats
        stats = compute_stats(segments)
        stats_path = out_dir / jf.name.replace(".deepgram.json", ".stats.json")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        print(f"Wrote: {txt_path.name} (+ {stats_path.name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
