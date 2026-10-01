#!/usr/bin/env python3
"""
dataset_stats.py

Compute basic statistics from host_guest JSON files:
  - Total episodes
  - Episodes by host gender (male host / female host / unknown)
  - For each host gender: how many male/female/unknown guests
  - Episode count per known host (from speaker library matches)
  - Episodes where gender could not be determined

Usage:
    python src/dataset_stats.py \
        --json_dir data/outputs/whisperx/ \
        --output_csv data/outputs/dataset_stats.csv
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_host_guest_info(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract host and guest gender info from a host_guest JSON file.

    Returns:
    {
        "host_speaker":   "SPEAKER_00",
        "host_gender":    "male",
        "host_name":      "Ranveer A",        # if matched from library
        "host_source":    "library",
        "guest_speaker":  "SPEAKER_01",
        "guest_gender":   "female",
        "guest_name":     "SPEAKER_01",
        "guest_source":   "acoustic",
    }
    """
    segments = data.get("segments", [])
    speaker_role_mapping = data.get("speaker_role_mapping", {})
    speaker_gender_mapping = data.get("speaker_gender_mapping", {})

    if not speaker_role_mapping or not speaker_gender_mapping:
        return None

    host_id = None
    guest_ids = []

    for speaker, role in speaker_role_mapping.items():
        if role == "HOST":
            host_id = speaker
        elif role == "GUEST":
            guest_ids.append(speaker)

    if not host_id:
        return None

    def get_info(speaker_id):
        info = speaker_gender_mapping.get(speaker_id, {})
        gender = info.get("gender", "unknown")
        source = info.get("source", "none")
        name = info.get("name", speaker_id)  # real name if library match
        return gender, source, name

    host_gender, host_source, host_name = get_info(host_id)

    # Handle multiple guests (take first guest for simplicity)
    guest_id = guest_ids[0] if guest_ids else None
    if guest_id:
        guest_gender, guest_source, guest_name = get_info(guest_id)
    else:
        guest_gender, guest_source, guest_name = "unknown", "none", "unknown"

    return {
        "host_speaker":  host_id,
        "host_gender":   host_gender,
        "host_name":     host_name,
        "host_source":   host_source,
        "guest_speaker": guest_id,
        "guest_gender":  guest_gender,
        "guest_name":    guest_name,
        "guest_source":  guest_source,
    }


def compute_stats(json_dir: Path) -> Dict[str, Any]:
    json_files = sorted(json_dir.glob("*_whisperx_diarized.gender.host_guest.json"))

    if not json_files:
        # Try alternate naming pattern
        json_files = sorted(json_dir.glob("*.host_guest.json"))

    if not json_files:
        raise FileNotFoundError(f"No host_guest JSON files found in {json_dir}")

    print(f"Found {len(json_files)} episode JSON files\n")

    total_episodes = 0
    skipped = []

    # Host gender counts
    host_gender_counts = defaultdict(int)

    # For each host gender: guest gender breakdown
    # e.g. guest_breakdown["male"]["female"] = 12
    guest_breakdown = defaultdict(lambda: defaultdict(int))

    # Per known host episode counts
    host_episode_counts = defaultdict(int)

    # Per row data for CSV
    rows = []

    for json_path in json_files:
        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"  ERROR reading {json_path.name}: {e}")
            skipped.append(json_path.name)
            continue

        info = get_host_guest_info(data)
        if info is None:
            print(f"  SKIP (missing role/gender mapping): {json_path.name}")
            skipped.append(json_path.name)
            continue

        total_episodes += 1
        host_gender = info["host_gender"]
        guest_gender = info["guest_gender"]
        host_name = info["host_name"]

        host_gender_counts[host_gender] += 1
        guest_breakdown[host_gender][guest_gender] += 1

        # Count per known host (only if matched from library)
        if info["host_source"] == "library":
            host_episode_counts[host_name] += 1
        else:
            host_episode_counts["Unknown Host"] += 1

        rows.append({
            "episode":       json_path.stem,
            "host_speaker":  info["host_speaker"],
            "host_name":     info["host_name"],
            "host_gender":   info["host_gender"],
            "host_source":   info["host_source"],
            "guest_speaker": info["guest_speaker"],
            "guest_name":    info["guest_name"],
            "guest_gender":  info["guest_gender"],
            "guest_source":  info["guest_source"],
        })

    return {
        "total_episodes":     total_episodes,
        "skipped":            skipped,
        "host_gender_counts": dict(host_gender_counts),
        "guest_breakdown":    {k: dict(v) for k, v in guest_breakdown.items()},
        "host_episode_counts": dict(host_episode_counts),
        "rows":               rows,
    }


def print_stats(stats: Dict[str, Any]) -> None:
    print("=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)

    print(f"\nTotal episodes processed: {stats['total_episodes']}")
    if stats["skipped"]:
        print(f"Skipped (missing data):   {len(stats['skipped'])}")
        for s in stats["skipped"]:
            print(f"  - {s}")

    # Host gender breakdown
    print(f"\n--- Episodes by Host Gender ---")
    for gender in ["male", "female", "unknown"]:
        count = stats["host_gender_counts"].get(gender, 0)
        pct = count / stats["total_episodes"] * 100 if stats["total_episodes"] > 0 else 0
        print(f"  {gender.capitalize():<10} {count:>4} episodes  ({pct:.1f}%)")

    # Guest breakdown per host gender
    print(f"\n--- Guest Gender by Host Gender ---")
    for host_gender in ["male", "female", "unknown"]:
        breakdown = stats["guest_breakdown"].get(host_gender, {})
        if not breakdown:
            continue
        total_hg = sum(breakdown.values())
        print(f"\n  Host: {host_gender.upper()} ({total_hg} episodes)")
        for guest_gender in ["male", "female", "unknown"]:
            count = breakdown.get(guest_gender, 0)
            pct = count / total_hg * 100 if total_hg > 0 else 0
            print(f"    Guest {guest_gender:<10} {count:>4} episodes  ({pct:.1f}%)")

    # Per host episode counts
    print(f"\n--- Episodes per Host ---")
    for host, count in sorted(stats["host_episode_counts"].items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {host:<25} {count:>4} episodes")

    print("\n" + "=" * 55)


def save_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode", "host_speaker", "host_name", "host_gender", "host_source",
        "guest_speaker", "guest_name", "guest_gender", "guest_source"
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode CSV saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Compute dataset statistics from host_guest JSON files"
    )
    ap.add_argument(
        "--json_dir",
        required=True,
        help="Directory containing *.host_guest.json files"
    )
    ap.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save per-episode CSV"
    )
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"Directory not found: {json_dir}")

    stats = compute_stats(json_dir)
    print_stats(stats)

    if args.output_csv:
        save_csv(stats["rows"], Path(args.output_csv))


if __name__ == "__main__":
    main()
