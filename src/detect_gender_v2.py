#!/usr/bin/env python3
"""
Detect speaker gender using voice analysis and update Deepgram host_guest files.

Uses inaSpeechSegmenter for gender detection from audio.

Install:
  pip install inaSpeechSegmenter tensorflow

Usage:
  python detect_gender_v2.py \
    --audio data/raw_audio/ep_002.wav \
    --input-json data/outputs/ep_002.sentences.host_guest.json \
    --output-json data/outputs/ep_002.sentences.host_guest.gender.json \
    --output-txt data/outputs/ep_002.sentences.host_guest.gender.txt
"""

from __future__ import annotations

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
    return max(0, min(a_end, b_end) - max(a_start, b_end))


def assign_gender_to_speakers(deepgram_segments, gender_segments):
    """
    Assign gender to each speaker based on overlap with gender segments.
    Returns dict: {speaker_raw: gender}
    """
    # For each speaker, accumulate overlap time with male/female segments
    speaker_gender_time = defaultdict(lambda: {'male': 0.0, 'female': 0.0})
    
    for dseg in deepgram_segments:
        speaker = dseg.get('speaker_raw', 'Unknown')
        if speaker == 'Unknown':
            continue
        
        d_start = dseg['start']
        d_end = dseg['end']
        
        # Check overlap with each gender segment
        for gseg in gender_segments:
            overlap = overlap_duration(d_start, d_end, gseg['start'], gseg['end'])
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


def update_json_with_gender(json_path, speaker_gender, output_path):
    """Update JSON file with gender information."""
    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add gender to each segment
    for seg in data['segments']:
        speaker = seg.get('speaker_raw', 'Unknown')
        seg['gender'] = speaker_gender.get(speaker, 'unknown')
    
    # Add speaker gender mapping
    data['speaker_gender'] = speaker_gender
    
    # Save updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated JSON: {output_path}")


def update_txt_with_gender(txt_path, json_path, speaker_gender, output_path):
    """Update TXT file with gender information."""
    print(f"Loading TXT: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Load JSON to get speaker mapping
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Create mapping from role to gender
    role_gender_map = {}
    for seg in json_data['segments']:
        role = seg.get('speaker_role', 'Unknown')
        speaker = seg.get('speaker_raw', 'Unknown')
        gender = speaker_gender.get(speaker, 'unknown')
        role_gender_map[role] = gender
    
    # Update lines with gender info
    updated_lines = []
    for line in lines:
        line = line.strip()
        if line and '[' in line and ']' in line and ':' in line:
            # Parse line format: [00:00:00] HOST: text
            parts = line.split(':', 1)
            if len(parts) >= 2:
                role_part = parts[0].strip()  # [00:00:00] HOST
                text_part = parts[1].strip()  # text
                
                # Extract role from role_part
                if ']' in role_part:
                    timestamp = role_part.split(']')[0] + ']'
                    role = role_part.split(']')[1].strip()
                    gender = role_gender_map.get(role, 'unknown')
                    
                    # Add gender info
                    updated_line = f"{timestamp} {role} ({gender}): {text_part}"
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    # Add gender summary at the end
    gender_summary = "\n\n# Speaker Gender Detection:\n"
    for role, gender in role_gender_map.items():
        gender_summary += f"# {role}: {gender}\n"
    
    updated_lines.append(gender_summary)
    
    # Save updated TXT
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Updated TXT: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Detect speaker gender from audio and update Deepgram host_guest files")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--input-json", required=True, help="Input Deepgram host_guest JSON file path")
    ap.add_argument("--output-json", required=True, help="Output JSON file path with gender info")
    ap.add_argument("--output-txt", required=True, help="Output TXT file path with gender info")
    ap.add_argument("--method", default="inaspeech", 
                    choices=["inaspeech", "pitch"],
                    help="Gender detection method")
    args = ap.parse_args()
    
    audio_path = Path(args.audio)
    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    output_txt_path = Path(args.output_txt)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    if not input_json_path.exists():
        print(f"Error: Input JSON file not found: {input_json_path}")
        return 1
    
    # Load Deepgram segments
    print(f"Loading Deepgram transcript: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']
    
    # Detect gender from audio
    if args.method == "inaspeech":
        gender_segments = detect_gender_inaspeech(audio_path)
        if gender_segments is None:
            return 1
    else:
        print(f"Method {args.method} not implemented yet")
        return 1
    
    print(f"\nFound {len(gender_segments)} gender segments")
    
    # Assign gender to speakers
    print("\nAssigning gender to speakers...")
    speaker_gender = assign_gender_to_speakers(segments, gender_segments)
    
    print("\nResults:")
    for speaker, gender in speaker_gender.items():
        print(f"  {speaker}: {gender}")
    
    # Update JSON file
    update_json_with_gender(input_json_path, speaker_gender, output_json_path)
    
    # Update TXT file (if input TXT exists)
    input_txt_path = input_json_path.parent / f"{input_json_path.stem}.txt"
    if input_txt_path.exists():
        update_txt_with_gender(input_txt_path, input_json_path, speaker_gender, output_txt_path)
    else:
        print(f"Warning: TXT file not found: {input_txt_path}")
    
    print(f"\nGender detection complete!")
    print(f"Updated files:")
    print(f"  JSON: {output_json_path}")
    print(f"  TXT: {output_txt_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

