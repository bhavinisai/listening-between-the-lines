#!/usr/bin/env python3
"""
overlap_diarization.py

Step 1 of the interruption analysis (GPU). Re-runs pyannote speaker diarization
on one episode and keeps the *overlapping* output. WhisperX only keeps the
exclusive (one speaker at a time) version, so the existing transcripts contain
no overlapping speech.

Uses the same pipeline WhisperX used (pyannote/speaker-diarization-community-1)
with 2 speakers, so speaker clusters match the existing transcripts as closely
as possible. The new SPEAKER_xx labels are mapped to the old HOST/GUEST labels
later, in detect_interruptions.py.

Writes data/outputs/overlap/<stem>_overlap_diarization.json:
  {"episode_id", "model", "num_speakers", "duration",
   "diarization":           [{"start", "end", "speaker"}, ...],   # with overlap
   "exclusive_diarization": [{"start", "end", "speaker"}, ...]}   # without

Usage:
  python src/interruptions/overlap_diarization.py data/raw_audio/ep_001.wav \
      --output_dir data/outputs/overlap --hf_token "$HF_TOKEN"
"""

import argparse
import json
import os
from pathlib import Path

import torch
import whisperx
from pyannote.audio import Pipeline

SAMPLE_RATE = 16000


def annotation_to_list(annotation):
    return [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
            for seg, _, spk in annotation.itertracks(yield_label=True)]


def main():
    ap = argparse.ArgumentParser(description="Overlap-preserving pyannote diarization for one episode")
    ap.add_argument("audio_file")
    ap.add_argument("--output_dir", default="data/outputs/overlap")
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--num_speakers", type=int, default=2)
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    audio_path = Path(args.audio_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}_overlap_diarization.json"
    if out_path.exists() and not args.overwrite:
        print(f"Exists, skipping: {out_path}")
        return

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Pass --hf_token or set HF_TOKEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(args.model, token=hf_token).to(device)

    # Load through ffmpeg the same way WhisperX does, so timestamps line up
    audio = whisperx.load_audio(str(audio_path))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE},
                      num_speakers=args.num_speakers)

    # pyannote >= 4 returns DiarizeOutput; older versions return an Annotation
    if hasattr(output, "speaker_diarization"):
        diar, excl = output.speaker_diarization, output.exclusive_speaker_diarization
    else:
        diar, excl = output, None

    result = {
        "episode_id": audio_path.stem,
        "model": args.model,
        "num_speakers": args.num_speakers,
        "duration": round(len(audio) / SAMPLE_RATE, 3),
        "diarization": annotation_to_list(diar),
        "exclusive_diarization": annotation_to_list(excl) if excl is not None else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    overlap = diar.get_overlap().duration()
    print(f"{audio_path.stem}: {len(result['diarization'])} turns, "
          f"{overlap:.1f}s overlapped speech ({100 * overlap / result['duration']:.2f}% of audio)")


if __name__ == "__main__":
    main()
