#!/usr/bin/env python3
"""
apply_host_overrides.py

Manually assign a known host to a diarized speaker when the speaker library
match failed in detect_gender_v5.py. Rewrites the episode's gender JSON/TXT so
the speaker carries the host's name with source=manual, which
labeled_transcript_v12.py treats as the host.

Re-run this after any re-run of detect_gender_v5.py, since that overwrites
the gender JSON.

Usage:
    python src/transcript_creation/apply_host_overrides.py \
        --overrides data/host_overrides.json \
        --whisperx_dir data/outputs/whisperx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_gender_v5 import (  # noqa: E402
    apply_gender_to_json,
    build_txt_lines,
    load_json,
    save_json,
    save_txt,
)


def main():
    ap = argparse.ArgumentParser(description="Apply manual host overrides to gender JSON/TXT")
    ap.add_argument("--overrides", default="data/host_overrides.json",
                    help='JSON: {"ep_369": {"speaker": "SPEAKER_00", "name": ..., "gender": ...}}')
    ap.add_argument("--whisperx_dir", default="data/outputs/whisperx")
    ap.add_argument("--speaker_key", default=None)
    args = ap.parse_args()

    overrides = load_json(Path(args.overrides))
    whisperx_dir = Path(args.whisperx_dir)

    for ep, ov in sorted(overrides.items()):
        json_path = whisperx_dir / f"{ep}_whisperx_diarized.gender.json"
        txt_path = whisperx_dir / f"{ep}_whisperx_diarized.gender.txt"
        if not json_path.exists():
            print(f"WARNING: {json_path} not found, skipping {ep}", file=sys.stderr)
            continue

        data = load_json(json_path)
        mapping = data.get("speaker_gender_mapping", {})
        spk = ov["speaker"]
        if spk not in mapping:
            print(f"WARNING: {spk} not in {ep} speaker_gender_mapping, skipping", file=sys.stderr)
            continue

        # Drop library matches on other speakers so there is exactly one host
        for other, info in mapping.items():
            if other != spk and info.get("source") in ("library", "manual"):
                print(f"WARNING: {ep} {other} was matched to {info.get('name')}; "
                      f"clearing name because {spk} is the manual host", file=sys.stderr)
                info.pop("name", None)
                info["source"] = "acoustic_override_cleared"
                info["needs_review"] = True

        mapping[spk] = {
            "name": ov["name"],
            "gender": ov["gender"],
            "source": "manual",
            "confidence": 1.0,
            "needs_review": False,
            "previous": mapping[spk],
        }

        # apply_gender_to_json only sets speaker_name when a name exists, so
        # clear stale names first.
        for seg in data.get("segments", []):
            seg.pop("speaker_name", None)
        data = apply_gender_to_json(data, mapping, args.speaker_key)
        save_json(json_path, data)
        save_txt(txt_path, build_txt_lines(data, args.speaker_key))
        print(f"{ep}: {spk} → {ov['name']} ({ov['gender']}, source=manual)")


if __name__ == "__main__":
    main()
