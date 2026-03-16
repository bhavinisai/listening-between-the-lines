#!/usr/bin/env python3
"""
Label speakers as HOST/GUEST based on gender-labeled simple diarization output.

Works with the format from detect_gender_simple.py output.

Usage:
  python host_guest_simple.py \
    --input data/outputs/gender/ep_001.diarized.gender.json \
    --output data/outputs/gender/ep_001.host_guest.json \
    --output-txt data/outputs/gender/ep_001.host_guest.txt
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_simple_diarized_json(json_path):
    """Load simple diarized JSON format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def identify_host_guest(segments):
    """
    Identify HOST and GUEST based on:
    1. Speaking time (most = HOST)
    2. Gender (if both same gender, use speaking time)
    3. Early speaking (who speaks first)
    """
    if not segments:
        return {}
    
    # Calculate speaking time per speaker
    speaker_stats = defaultdict(lambda: {'duration': 0.0, 'segments': [], 'gender': 'unknown'})
    
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        duration = seg.get('end', 0) - seg.get('start', 0)
        gender = seg.get('gender', 'unknown')
        
        speaker_stats[speaker_id]['duration'] += duration
        speaker_stats[speaker_id]['segments'].append(seg)
        speaker_stats[speaker_id]['gender'] = gender
    
    # Sort speakers by speaking time
    speakers_by_time = sorted(
        speaker_stats.items(),
        key=lambda x: x[1]['duration'],
        reverse=True
    )
    
    # Identify host (most speaking time)
    host_id = speakers_by_time[0][0]
    
    # Assign roles
    roles = {}
    for speaker_id, stats in speaker_stats.items():
        if speaker_id == host_id:
            roles[speaker_id] = 'HOST'
        else:
            roles[speaker_id] = 'GUEST'
    
    return roles


def add_host_guest_labels(segments, roles):
    """Add HOST/GUEST labels to segments."""
    labeled_segments = []
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        speaker_label = seg.get('speaker_label', f"Speaker {speaker_id + 1}")
        gender = seg.get('gender', 'unknown')
        role = roles.get(speaker_id, 'UNKNOWN')
        
        labeled_seg = {
            'speaker_id': speaker_id,
            'speaker_label': speaker_label,
            'gender': gender,
            'role': role,
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'text': seg.get('text', ''),
            'confidence': seg.get('confidence', 0.0)
        }
        labeled_segments.append(labeled_seg)
    
    return labeled_segments


def write_host_guest_json(segments, roles, output_path):
    """Write JSON with HOST/GUEST labels."""
    # Get unique speakers
    speakers = list(set(seg['speaker_id'] for seg in segments))
    speakers.sort()
    
    # Create speaker info
    speaker_info = []
    for speaker_id in speakers:
        # Find gender and role for this speaker
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
        if speaker_segments:
            gender = speaker_segments[0]['gender']
            role = roles.get(speaker_id, 'UNKNOWN')
        else:
            gender = 'unknown'
            role = 'UNKNOWN'
        
        speaker_info.append({
            'speaker_id': speaker_id,
            'speaker_label': f"Speaker {speaker_id + 1}",
            'gender': gender,
            'role': role
        })
    
    # Create output structure
    output_data = {
        'total_segments': len(segments),
        'speakers': speaker_info,
        'segments': segments,
        'host_guest_mapping': roles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Written: {output_path}")


def write_host_guest_txt(segments, output_path):
    """Write TXT with HOST/GUEST labels."""
    lines = []
    
    for seg in segments:
        start_min = int(seg['start'] // 60)
        start_sec = int(seg['start'] % 60)
        timestamp = f"{start_min:02d}:{start_sec:02d}"
        
        role = seg['role']
        gender = seg['gender']
        text = seg['text'].strip()
        
        if text:
            lines.append(f"[{timestamp}] {role} ({gender}): {text}")
    
    # Add summary at the end
    lines.append("")
    lines.append("# Speaker Summary:")
    
    # Get unique speakers
    speakers = list(set(seg['speaker_id'] for seg in segments))
    speakers.sort()
    
    for speaker_id in speakers:
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
        if speaker_segments:
            role = speaker_segments[0]['role']
            gender = speaker_segments[0]['gender']
            speaker_label = f"Speaker {speaker_id + 1}"
            duration = sum(s['end'] - s['start'] for s in speaker_segments)
            lines.append(f"# {speaker_label} = {role} ({gender}) - {duration:.1f}s total")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Label speakers as HOST/GUEST from simple diarized output")
    parser.add_argument("--input", required=True, help="Input gender-labeled JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--output-txt", required=True, help="Output TXT file path")
    parser.add_argument("--method", default="speaking_time", 
                    choices=["speaking_time", "first_speaker", "gender_based"],
                    help="Method for host identification")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_json_path = Path(args.output)
    output_txt_path = Path(args.output_txt)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Load input data
    print(f"Loading: {input_path}")
    data = load_simple_diarized_json(input_path)
    
    segments = data.get('segments', [])
    if not segments:
        print("Error: No segments found in input file")
        return 1
    
    print(f"Found {len(segments)} segments")
    
    # Identify host and guest
    print("Identifying HOST/GUEST roles...")
    roles = identify_host_guest(segments)
    
    print("Results:")
    for speaker_id, role in roles.items():
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
        if speaker_segments:
            gender = speaker_segments[0]['gender']
            duration = sum(s['end'] - s['start'] for s in speaker_segments)
            print(f"  Speaker {speaker_id + 1}: {role} ({gender}) - {duration:.1f}s")
    
    # Add labels to segments
    labeled_segments = add_host_guest_labels(segments, roles)
    
    # Write outputs
    print("Writing outputs...")
    write_host_guest_json(labeled_segments, roles, output_json_path)
    write_host_guest_txt(labeled_segments, output_txt_path)
    
    print(f"\n✅ Complete!")
    print(f"📄 JSON: {output_json_path}")
    print(f"📄 TXT: {output_txt_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

