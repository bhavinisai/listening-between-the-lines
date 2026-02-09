#!/usr/bin/env python3
"""
Detect speaker gender using voice analysis.

Uses inaSpeechSegmenter for gender detection from audio.

Install:
  pip install inaSpeechSegmenter tensorflow

Usage:
  python detect_speaker_gender.py \
    --audio episode.wav \
    --json episode.whisperx.json \
    --output episode.with_gender.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def detect_gender_inaspeech(audio_path):
    """
    Use inaSpeechSegmenter to detect gender from audio.
    Returns list of (start, end, gender) tuples.
    """
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError:
        print("Error: inaSpeechSegmenter not installed")
        print("Install with: pip install inaSpeechSegmenter tensorflow")
        return None
    
    print("Loading gender detection model...")
    seg = Segmenter()
    
    print(f"Analyzing audio: {audio_path}")
    segmentation = seg(str(audio_path))
    
    # segmentation is list of (label, start, end)
    # labels: 'male', 'female', 'noEnergy', 'noise', 'music'
    gender_segments = []
    for label, start, end in segmentation:
        if label in ['male', 'female']:
            gender_segments.append({
                'start': start,
                'end': end,
                'gender': label
            })
    
    return gender_segments


def overlap_duration(a_start, a_end, b_start, b_end):
    """Calculate overlap duration between two time segments."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def assign_gender_to_speakers(whisperx_segments, gender_segments):
    """
    Assign gender to each speaker based on overlap with gender segments.
    Returns dict: {speaker_label: gender}
    """
    # For each speaker, accumulate overlap time with male/female segments
    speaker_gender_time = defaultdict(lambda: {'male': 0.0, 'female': 0.0})
    
    for wseg in whisperx_segments:
        speaker = wseg.get('speaker', 'Unknown')
        if speaker == 'Unknown':
            continue
        
        w_start = wseg['start']
        w_end = wseg['end']
        
        # Check overlap with each gender segment
        for gseg in gender_segments:
            overlap = overlap_duration(w_start, w_end, gseg['start'], gseg['end'])
            if overlap > 0:
                speaker_gender_time[speaker][gseg['gender']] += overlap
    
    # Determine gender for each speaker
    speaker_gender = {}
    for speaker, times in speaker_gender_time.items():
        total_male = times['male']
        total_female = times['female']
        total = total_male + total_female
        
        if total == 0:
            speaker_gender[speaker] = 'unknown'
        elif total_male > total_female:
            confidence = total_male / total
            speaker_gender[speaker] = 'male' if confidence > 0.6 else 'unknown'
        else:
            confidence = total_female / total
            speaker_gender[speaker] = 'female' if confidence > 0.6 else 'unknown'
    
    return speaker_gender


def main():
    ap = argparse.ArgumentParser(description="Detect speaker gender from audio")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--json", required=True, help="WhisperX JSON with speaker labels")
    ap.add_argument("--output", help="Output JSON with gender info (optional)")
    ap.add_argument("--method", default="inaspeech", 
                    choices=["inaspeech", "pitch"],
                    help="Gender detection method")
    args = ap.parse_args()
    
    audio_path = Path(args.audio)
    json_path = Path(args.json)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    # Load WhisperX segments
    print(f"Loading transcript: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']
    
    # Detect gender from audio
    if args.method == "inaspeech":
        gender_segments = detect_gender_inaspeech(audio_path)
        if gender_segments is None:
            return
    else:
        print(f"Method {args.method} not implemented yet")
        return
    
    print(f"\nFound {len(gender_segments)} gender segments")
    
    # Assign gender to speakers
    print("\nAssigning gender to speakers...")
    speaker_gender = assign_gender_to_speakers(segments, gender_segments)
    
    print("\nResults:")
    for speaker, gender in speaker_gender.items():
        print(f"  {speaker}: {gender}")
    
    # Add gender to segments
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        seg['gender'] = speaker_gender.get(speaker, 'unknown')
    
    # Save output
    output_data = {
        'segments': segments,
        'speaker_gender': speaker_gender,
        'gender_segments': gender_segments
    }
    
    output_path = Path(args.output) if args.output else json_path.parent / f"{json_path.stem}.with_gender.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved output: {output_path}")


if __name__ == "__main__":
    main()
