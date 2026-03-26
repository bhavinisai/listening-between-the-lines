#!/usr/bin/env python3
"""
Process Deepgram sentences.json files to add gender detection.

This script reads Deepgram sentences.json files, detects speaker gender,
and updates the files with gender information.

Usage:
  python process_sentences_with_gender.py \
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
    # For Deepgram format, we'll use speaker_id patterns
    gender_segments = [
        {'start': 0.0, 'end': 30.0, 'gender': 'male'},   # Speaker 0 (usually host)
        {'start': 30.0, 'end': 60.0, 'gender': 'female'},  # Speaker 1 (usually guest)
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


def update_sentences_json_with_gender(json_path, speaker_gender):
    """Update sentences.json file with gender information."""
    print(f"Loading sentences JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add gender to each segment
    for seg in data['segments']:
        speaker_id = seg.get('speaker_id', 'Unknown')
        seg['gender'] = speaker_gender.get(f"SPEAKER_{speaker_id:02d}", 'unknown')
    
    # Add speaker gender mapping
    data['speaker_gender'] = speaker_gender
    
    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated JSON: {json_path}")


def create_host_guest_txt_from_sentences(json_path, speaker_gender, output_path):
    """Create host_guest.txt from sentences.json with gender info."""
    print(f"Creating host_guest TXT from: {json_path}")
    
    # Load sentences JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']
    
    # Create host_guest.txt format
    lines = []
    
    # Add header
    lines.append("# Speaker Gender Detection Results")
    
    # Process each segment
    for seg in segments:
        speaker_id = seg.get('speaker_id', 'Unknown')
        start = seg.get('start', 0.0)
        text = seg.get('transcript', '').strip()
        gender = speaker_gender.get(f"SPEAKER_{speaker_id:02d}", 'unknown')
        
        # Format: [00:00:00] HOST (male): text
        timestamp = f"{int(start//60):02d}:{int(start%60):02d}"
        role = "HOST" if seg.get('speaker_role') == "HOST" else "GUEST"
        line = f"[{timestamp}] {role} ({gender}): {text}"
        lines.append(line)
    
    # Add gender summary at the end
    lines.append("")
    lines.append("# Speaker Gender Summary:")
    for speaker_id, gender in speaker_gender.items():
        speaker_label = f"Speaker {speaker_id + 1}"
        lines.append(f"# {speaker_label}: {gender}")
    
    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Created host_guest TXT: {output_path}")


def process_single_file(input_file, output_dir, method):
    """Process a single sentences.json file."""
    print(f"\nProcessing: {input_file.name}")
    
    # Determine output paths
    base_name = input_file.stem
    json_path = output_dir / f"{base_name}.with_gender.json"
    txt_path = output_dir / f"{base_name}.with_gender.txt"
    
    # Load segments
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get('segments', [])
    
    if not segments:
        print(f"  No segments found in {input_file}")
        return False
    
    # Detect gender (mock for now)
    if method == "mock":
        gender_segments = detect_gender_inaspeech_mock(str(input_file).replace('.sentences.json', '.wav'))
    else:
        print(f"Method {method} not implemented yet")
        return False
    
    # Assign gender to speakers
    speaker_gender = assign_gender_to_speakers(segments, gender_segments)
    
    print("\nGender Assignment:")
    for speaker, gender in speaker_gender.items():
        print(f"  {speaker}: {gender}")
    
    # Update JSON file
    update_sentences_json_with_gender(input_file, speaker_gender)
    
    # Create host_guest.txt
    create_host_guest_txt_from_sentences(input_file, speaker_gender, txt_path)
    
    print(f"✅ Complete: {input_file.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Process Deepgram sentences.json files with gender detection")
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
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            if process_single_file(file_path, output_dir, args.method):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total files: {len(files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
