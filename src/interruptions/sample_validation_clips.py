#!/usr/bin/env python3
"""
sample_validation_clips.py

Step 3 of the interruption analysis: sample overlap events to check by ear.
Draws up to --per_stratum events per (type x dyad) stratum from
results/interruptions/interruption_events.csv. For each one it cuts an audio
clip (--pad s before the overlap to --pad s after the later turn end, capped at
--max_clip s) and writes a CSV with an empty `human_label` column.

Suggested labels: successful_interruption, unsuccessful_interruption,
backchannel, terminal_overlap, no_overlap (diarization error),
wrong_speaker (who-overlapped-whom is wrong).

The sample is split into a `tune` half (to adjust thresholds in
detect_interruptions.py) and a `test` half (for reporting precision).

Clips are only cut for episodes whose audio is in --audio_dir; the rest are
listed with clip_path empty.

Usage:
  python src/interruptions/sample_validation_clips.py
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Sample interruption events and cut audio clips for validation")
    ap.add_argument("--events", default="results/interruptions/interruption_events.csv")
    ap.add_argument("--audio_dir", default="data/raw_audio")
    ap.add_argument("--out_dir", default="results/interruptions/validation")
    ap.add_argument("--per_stratum", type=int, default=10)
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--max_clip", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    ev = ev[ev.overlapper_role.isin(["HOST", "GUEST"]) & ev.overlapped_role.isin(["HOST", "GUEST"])]
    sample = (ev.sample(frac=1, random_state=args.seed)
                .groupby(["type", "dyad"]).head(args.per_stratum)
                .reset_index(drop=True))
    rng = np.random.default_rng(args.seed)
    sample["split"] = rng.permutation(np.arange(len(sample)) % 2 == 0)
    sample["split"] = sample.split.map({True: "tune", False: "test"})

    clip_dir = os.path.join(args.out_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    paths = []
    for i, r in sample.iterrows():
        audio = os.path.join(args.audio_dir, f"{r.episode_id}.wav")
        if not os.path.exists(audio):
            paths.append("")
            continue
        start = max(r.overlap_start - args.pad, 0.0)
        end = min(max(r.overlapped_turn_end, r.overlapper_turn_end) + args.pad, start + args.max_clip)
        out = os.path.join(clip_dir, f"{i:04d}_{r.episode_id}_{r.overlap_start:.1f}.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-i", audio, "-ac", "1", "-ar", "16000", out], check=True)
        paths.append(out)
    sample["clip_path"] = paths
    sample["clip_offset_of_overlap"] = (sample.overlap_start - (sample.overlap_start - args.pad).clip(lower=0)).round(2)
    sample["human_label"] = ""
    sample["notes"] = ""

    out_csv = os.path.join(args.out_dir, "validation_sample_to_label.csv")
    sample.to_csv(out_csv, index=False)
    print(sample.groupby(["type", "dyad"]).size().unstack(fill_value=0))
    print(f"{(sample.clip_path != '').sum()}/{len(sample)} clips cut. Wrote {out_csv}")


if __name__ == "__main__":
    main()
