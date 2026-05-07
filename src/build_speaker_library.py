#!/usr/bin/env python3
"""
build_speaker_library.py

Build a speaker embedding library from a set of known host .wav files.
Each host is enrolled with their name and gender, and a voice embedding
is computed and stored for later matching against podcast episodes.

Requirements:
    pip install pyannote.audio torch torchaudio

Usage:
    python build_speaker_library.py \
        --hosts_dir data/hosts/ \
        --output_library data/speaker_library.json

Expected directory structure:
    data/hosts/
        john_doe.wav
        jane_smith.wav
        bob_jones.wav
        ...

The script will prompt you to enter the gender for each host interactively.

Alternatively, pass a metadata CSV to skip interactive prompts:
    python build_speaker_library.py \
        --hosts_dir data/hosts/ \
        --output_library data/speaker_library.json \
        --metadata data/hosts/metadata.csv

metadata.csv format (no header required):
    filename,name,gender
    john_doe.wav,John Doe,male
    jane_smith.wav,Jane Smith,female
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedding_model():
    """Load pyannote speaker embedding model."""
    try:
        from pyannote.audio import Model
        from pyannote.audio import Inference
    except ImportError:
        raise RuntimeError(
            "pyannote.audio is not installed.\n"
            "Install with: pip install pyannote.audio torch torchaudio"
        )

    print("Loading speaker embedding model...")
    model = Model.from_pretrained("pyannote/embedding",
                                  use_auth_token=False)
    inference = Inference(model, window="whole")
    return inference


def compute_embedding(inference, audio_path: Path) -> list[float]:
    import torch
    import torchaudio

    # Load audio manually to bypass torchcodec
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Resample to 16kHz if needed (pyannote expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Pass as dict directly — bypasses pyannote's broken audio decoder
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    embedding = inference(audio_input)
    vec = np.array(embedding).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load host metadata from a CSV file.

    Returns:
        {filename: {"name": ..., "gender": ...}}
    """
    metadata = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            name = row.get("name", "").strip()
            gender = row.get("gender", "").strip().lower()

            if not filename:
                continue
            if gender not in {"male", "female"}:
                print(f"  WARNING: Invalid gender '{gender}' for {filename} — skipping")
                continue

            metadata[filename] = {"name": name, "gender": gender}
    return metadata


def prompt_host_info(wav_path: Path) -> Dict[str, str]:
    """Interactively prompt for host name and gender."""
    print(f"\n  File: {wav_path.name}")
    name = input("  Enter host name: ").strip()
    while True:
        gender = input("  Enter gender (male/female): ").strip().lower()
        if gender in {"male", "female"}:
            break
        print("  Please enter 'male' or 'female'")
    return {"name": name, "gender": gender}


# ---------------------------------------------------------------------------
# Library building
# ---------------------------------------------------------------------------

def build_library(
    hosts_dir: Path,
    output_path: Path,
    metadata_csv: Optional[Path] = None,
) -> None:
    wav_files = sorted(hosts_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {hosts_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files)} host audio files:\n")
    for f in wav_files:
        print(f"  {f.name}")

    # Load metadata if provided
    metadata: Dict[str, Dict[str, str]] = {}
    if metadata_csv:
        print(f"\nLoading metadata from {metadata_csv}")
        metadata = load_metadata_csv(metadata_csv)

    # Load embedding model once
    inference = load_embedding_model()

    library: Dict[str, Any] = {"speakers": []}

    for wav_path in wav_files:
        print(f"\nProcessing: {wav_path.name}")

        # Get host info
        if wav_path.name in metadata:
            info = metadata[wav_path.name]
            print(f"  Name:   {info['name']}")
            print(f"  Gender: {info['gender']}")
        else:
            info = prompt_host_info(wav_path)

        # Compute embedding
        print(f"  Computing voice embedding...")
        try:
            embedding = compute_embedding(inference, wav_path)
        except Exception as e:
            print(f"  ERROR computing embedding: {e} — skipping")
            continue

        entry = {
            "name": info["name"],
            "gender": info["gender"],
            "source_file": wav_path.name,
            "embedding": embedding,
        }
        library["speakers"].append(entry)
        print(f"  Done. Embedding dim: {len(embedding)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Library saved: {output_path}")
    print(f"Enrolled speakers ({len(library['speakers'])}):")
    for entry in library["speakers"]:
        print(f"  {entry['name']} ({entry['gender']}) — {entry['source_file']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a speaker embedding library from host .wav files"
    )
    ap.add_argument(
        "--hosts_dir",
        required=True,
        help="Directory containing one .wav file per host",
    )
    ap.add_argument(
        "--output_library",
        required=True,
        help="Path to save the speaker library JSON",
    )
    ap.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV with columns: filename, name, gender (skips interactive prompts)",
    )
    args = ap.parse_args()

    hosts_dir = Path(args.hosts_dir)
    output_path = Path(args.output_library)
    metadata_csv = Path(args.metadata) if args.metadata else None

    if not hosts_dir.exists():
        print(f"ERROR: hosts_dir not found: {hosts_dir}")
        sys.exit(1)

    if metadata_csv and not metadata_csv.exists():
        print(f"ERROR: metadata CSV not found: {metadata_csv}")
        sys.exit(1)

    build_library(hosts_dir, output_path, metadata_csv)


if __name__ == "__main__":
    main()
