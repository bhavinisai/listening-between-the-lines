#!/usr/bin/env python3
"""
Add gender detection to Deepgram host/guest files (mock version for testing).

This version simulates gender detection for demonstration purposes.
Replace the detect_gender_inaspeech function with real detection when available.

Usage:
  python add_gender_to_deepgram_mock.py \
    --audio ep001_clip_30s.wav \
    --json outputs/deepgram_output/ep001_clip_30s.deepgram.host_guest.json \
    --txt outputs/deepgram_output/ep001_clip_30s.deepgram.host_guest.txt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def detect_gender_inaspeech_mock(audio_path):
    """
    Mock gender detection for testing purposes.
    Returns list of (start, end, gender) tuples.
    """
    print(f"Mock analyzing audio: {audio_path}")
    
    # Create mock gender segments
    gender_segments = [
        {'start': 0.0, 'end': 15.0, 'gender': 'male'},
        {'start': 15.0, 'end': 30.0, 'gender': 'female'},
        {'start': 30.0, 'end': 45.0, 'gender': 'male'},
        # Add more segments as needed
    ]
    
    return gender_segments


def overlap_duration(a_start, a_end, b_start, b_end):
    """Calculate overlap duration between two time segments."""
    return max(0, min(a_end, b_end) - max(a_start, b_end))


def assign_gender_to_speakers(deepgram_segments, gender_segments):
    """
    Assign gender to each speaker based on overlap with gender segments.
    Returns dict: {speaker_raw: gender}
    """
    # For demonstration, assign genders based on speaker patterns
    speaker_gender = {}
    
    # Get unique speakers
    speakers = set()
    for seg in deepgram_segments:
        speaker = seg.get('speaker_raw', 'Unknown')
        speakers.add(speaker)
    
    # Mock assignment: typically SPEAKER_00 is male (host), SPEAKER_01 is female (guest)
    # This is just for demonstration - real detection would use audio analysis
    for speaker in speakers:
        if speaker == 'SPEAKER_00':
            speaker_gender[speaker] = 'male'
        elif speaker == 'SPEAKER_01':
            speaker_gender[speaker] = 'female'
        else:
            speaker_gender[speaker] = 'unknown'
    
    return speaker_gender


def update_json_with_gender(json_path, speaker_gender):
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
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated JSON: {json_path}")


def update_txt_with_gender(txt_path, json_path, speaker_gender):
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
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Updated TXT: {txt_path}")


def main():
    ap = argparse.ArgumentParser(description="Add gender detection to Deepgram host/guest files (mock version)")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--json", required=True, help="Deepgram host_guest JSON file path")
    ap.add_argument("--txt", required=True, help="Deepgram host_guest TXT file path")
    ap.add_argument("--method", default="mock", 
                    choices=["mock", "inaspeech", "pitch"],
                    help="Gender detection method")
    args = ap.parse_args()
    
    audio_path = Path(args.audio)
    json_path = Path(args.json)
    txt_path = Path(args.txt)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    if not txt_path.exists():
        print(f"Error: TXT file not found: {txt_path}")
        return
    
    # Load Deepgram segments
    print(f"Loading Deepgram transcript: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']
    
    # Detect gender from audio
    if args.method == "mock":
        gender_segments = detect_gender_inaspeech_mock(audio_path)
    elif args.method == "inaspeech":
        print("Real inaSpeechSegmenter not available - using mock instead")
        gender_segments = detect_gender_inaspeech_mock(audio_path)
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
    
    # Update JSON file
    update_json_with_gender(json_path, speaker_gender)
    
    # Update TXT file
    update_txt_with_gender(txt_path, json_path, speaker_gender)
    
    print(f"\nGender detection complete!")
    print(f"Updated files:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")


if __name__ == "__main__":
    main()
