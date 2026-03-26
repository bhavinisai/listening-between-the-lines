#!/usr/bin/env python3
"""
Deepgram Host/Guest Labeling with Gender Detection

Processes Deepgram sentences.json files, detects speaker gender,
and generates host_guest.json and host_guest.txt files with gender information.

Usage:
  python deepgram_label_with_gender.py \
    --input_dir outputs/deepgram_output \
    --pattern "*.sentences.json" \
    --method mock
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import os


def detect_gender_inaspeech_mock(audio_path):
    """
    Mock gender detection for testing purposes.
    Returns list of (start, end, gender) tuples.
    """
    print(f"Mock analyzing audio: {audio_path}")
    
    # Create mock gender segments based on speaker patterns
    # For Deepgram format, speaker_id 0 is usually host (male), speaker_id 1 is guest (female)
    gender_segments = [
        {'start': 0.0, 'end': 30.0, 'gender': 'male'},   # Speaker 0 (host)
        {'start': 30.0, 'end': 60.0, 'gender': 'female'},  # Speaker 1 (guest)
        # Add more segments as needed
    ]
    
    return gender_segments


def assign_gender_to_speakers(segments, gender_segments):
    """
    Assign gender to each speaker based on speaker_id patterns.
    For Deepgram sentences.json, speakers are numbered 0, 1, 2, etc.
    """
    speaker_gender = {}
    
    # Get unique speakers
    speakers = set()
    for seg in segments:
        speaker_id = seg.get('speaker_id', 'Unknown')
        speakers.add(speaker_id)
    
    # Mock assignment: typically speaker_id 0 is male (host), speaker_id 1 is female (guest)
    # This is just for demonstration - real detection would use audio analysis
    for speaker_id in speakers:
        if speaker_id == 0:
            speaker_gender[f"SPEAKER_{speaker_id:02d}"] = 'male'
        elif speaker_id == 1:
            speaker_gender[f"SPEAKER_{speaker_id:02d}"] = 'female'
        else:
            speaker_gender[f"SPEAKER_{speaker_id:02d}"] = 'unknown'
    
    return speaker_gender


def label_transcript_with_gender(segments, speaker_gender, speaker_key="speaker_raw"):
    """
    Label transcript with host/guest roles and add gender information.
    """
    # Create mapping for host/guest roles
    host_id = None
    
    # Simple heuristic: first speaker is usually host
    if len(speaker_gender) > 0:
        # Find speaker with most speaking time
        speaker_times = defaultdict(float)
        for seg in segments:
            speaker_id = seg.get('speaker_id', 'Unknown')
            if speaker_id in speaker_gender:
                duration = seg.get('end', 0) - seg.get('start', 0)
                speaker_times[speaker_id] += duration
        
        # Find speaker with most time (likely host)
        most_time_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
        host_id = most_time_speaker
    
    # Fallback to speaker 0 if no clear winner
    if host_id is None:
        host_id = list(speaker_gender.keys())[0] if speaker_gender else "SPEAKER_00"
    
    mapping = {spk: ("HOST" if spk == host_id else "GUEST") for spk in speaker_gender.keys()}
    
    # Add gender and role to segments
    updated_segments = []
    for seg in segments:
        speaker_id = seg.get('speaker_id', 'Unknown')
        speaker_label = seg.get('speaker_label', f"Speaker {speaker_id + 1}")
        speaker_role = mapping.get(speaker_id, "GUEST")
        gender = speaker_gender.get(speaker_id, 'unknown')
        
        updated_seg = {
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'text': seg.get('transcript', ''),
            'speaker_raw': f"SPEAKER_{speaker_id:02d}",
            'speaker': speaker_label,
            'speaker_role': speaker_role,
            'gender': gender
        }
        updated_segments.append(updated_seg)
    
    return updated_segments, host_id, mapping


def build_txt_lines_with_gender(segments):
    """Build text lines with gender information."""
    lines = []
    
    # Add header
    lines.append("# Deepgram Host/Guest Labeling with Gender Detection")
    
    for seg in segments:
        start = seg.get('start', 0)
        speaker_role = seg.get('speaker_role', 'Unknown')
        gender = seg.get('gender', 'unknown')
        text = seg.get('transcript', '').strip()
        
        if text:
            timestamp = f"{int(start//60):02d}:{int(start%60):02d}"
            line = f"[{timestamp}] {speaker_role} ({gender}): {text}"
            lines.append(line)
    
    return lines


def write_json_with_gender(output_path, segments, speaker_gender, host_id, mapping):
    """Write JSON file with gender information."""
    data = {
        'segments': segments,
        'speaker_gender': speaker_gender,
        'host_speaker': host_id,
        'speaker_role_mapping': mapping
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_txt_with_gender(output_path, segments):
    """Write TXT file with gender information."""
    lines = build_txt_lines_with_gender(segments)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def process_single_file(input_file, output_dir, method):
    """Process a single Deepgram sentences.json file."""
    print(f"\n🔍 Processing: {input_file.name}")
    
    try:
        # Load Deepgram sentences
        print(f"  📖 Loading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        
        if not segments:
            print(f"  ❌ No segments found in {input_file.name}")
            return False
        
        # Mock gender detection
        if method == "mock":
            gender_segments = [
                {'start': 0.0, 'end': 30.0, 'gender': 'male'},   # Speaker 0 (host)
                {'start': 30.0, 'end': 60.0, 'gender': 'female'},  # Speaker 1 (guest)
            ]
        else:
            print(f"  ⚠️  Method {method} not implemented")
            return False
        
        # Assign gender to speakers
        speaker_gender = assign_gender_to_speakers(segments, gender_segments)
        
        print(f"  👥 Gender Assignment:")
        for speaker_id, gender in speaker_gender.items():
            print(f"     SPEAKER_{speaker_id:02d}: {gender}")
        
        # Label transcript with gender
        updated_segments, host_id, mapping = label_transcript_with_gender(segments, speaker_gender)
        
        # Generate output paths
        base_name = input_file.stem
        json_output = output_dir / f"{base_name}.with_gender.json"
        txt_output = output_dir / f"{base_name}.with_gender.txt"
        
        # Write files
        write_json_with_gender(json_output, updated_segments, speaker_gender, host_id, mapping)
        write_txt_with_gender(txt_output, updated_segments)
        
        print(f"  ✅ Complete: {input_file.name}")
        print(f"     📄 JSON: {json_output.name}")
        print(f"     📄 TXT: {txt_output.name}")
        print(f"     🎯 Host: SPEAKER_{host_id:02d}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_file.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Deepgram host/guest labeling with gender detection")
    ap.add_argument("--input_dir", required=True, help="Directory containing sentences.json files")
    ap.add_argument("--pattern", default="*.sentences.json", help="File pattern to match")
    ap.add_argument("--method", default="mock", 
                    choices=["mock", "inaspeech", "pitch"],
                    help="Gender detection method")
    ap.add_argument("--output_dir", help="Output directory (default: same as input)")
    
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Find sentences.json files
    files = list(input_dir.glob(args.pattern))
    if not files:
        print(f"Error: No files found matching pattern '{args.pattern}' in {input_dir}")
        return 1
    
    print(f"🎯 Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for file_path in files:
        if process_single_file(file_path, output_dir, args.method):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(files)}")
    print(f"   Successful: {success_count}")
    print(f"   Errors: {error_count}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
