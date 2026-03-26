#!/usr/bin/env python3
"""
Batch gender detection for simple diarization output.

Uses inaSpeechSegmenter for gender detection from audio.

Install:
  pip install inaSpeechSegmenter tensorflow

Usage:
  python detect_gender_batch.py \
    --audio-dir data/raw_audio \
    --json-dir outputs \
    --output-dir outputs/gender
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict


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


def assign_gender_to_speakers(segments, gender_segments):
    """
    Assign gender to each speaker based on overlap with gender segments.
    Returns dict: {speaker_id: gender}
    """
    # For each speaker, accumulate overlap time with male/female segments
    speaker_gender_time = defaultdict(lambda: {'male': 0.0, 'female': 0.0})
    
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        
        seg_start = seg['start']
        seg_end = seg['end']
        
        # Check overlap with each gender segment
        for gseg in gender_segments:
            overlap = overlap_duration(seg_start, seg_end, gseg['start'], gseg['end'])
            if overlap > 0:
                speaker_gender_time[speaker_id][gseg['gender']] += overlap
    
    # Determine gender for each speaker
    speaker_gender = {}
    for speaker_id, times in speaker_gender_time.items():
        total_male = times['male']
        total_female = times['female']
        total = total_male + total_female
        
        if total == 0:
            speaker_gender[speaker_id] = 'unknown'
        elif total_male > total_female:
            confidence = total_male / total
            speaker_gender[speaker_id] = 'male' if confidence > 0.6 else 'unknown'
        else:
            confidence = total_female / total
            speaker_gender[speaker_id] = 'female' if confidence > 0.6 else 'unknown'
    
    return speaker_gender


def process_single_file(audio_path, json_path, output_dir):
    """Process a single audio file and its JSON."""
    print(f"\n🎵 Processing: {audio_path.name}")
    
    # Check if audio file exists
    if not audio_path.exists():
        print(f"  ❌ Audio file not found: {audio_path}")
        return False
    
    # Check if JSON file exists
    if not json_path.exists():
        print(f"  ❌ JSON file not found: {json_path}")
        return False
    
    # Generate output paths
    stem = audio_path.stem
    output_json = output_dir / f"{stem}.diarized.gender.json"
    output_txt = output_dir / f"{stem}.diarized.gender.txt"
    
    # Skip if outputs already exist
    if output_json.exists() and output_txt.exists():
        print(f"  ⏭  Skipping (already processed)")
        return True
    
    # Load diarized segments
    print(f"  📖 Loading JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    if not segments:
        print(f"  ❌ No segments found in JSON")
        return False
    
    # Detect gender from audio
    print(f"  🔍 Detecting gender...")
    gender_segments = detect_gender_inaspeech(audio_path)
    if gender_segments is None:
        return False
    
    print(f"  📊 Found {len(gender_segments)} gender segments")
    
    # Assign gender to speakers
    print(f"  👥 Assigning gender to speakers...")
    speaker_gender = assign_gender_to_speakers(segments, gender_segments)
    
    print(f"  🎯 Results:")
    for speaker_id, gender in speaker_gender.items():
        print(f"     Speaker {speaker_id + 1}: {gender}")
    
    # Add gender to segments
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        seg['gender'] = speaker_gender.get(speaker_id, 'unknown')
    
    # Add speaker gender mapping
    data['speaker_gender'] = speaker_gender
    
    # Update speakers list
    for speaker_info in data.get('speakers', []):
        speaker_id = speaker_info.get('speaker_id', 0)
        speaker_info['gender'] = speaker_gender.get(speaker_id, 'unknown')
    
    # Save updated JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Create TXT file
    print(f"  📝 Creating TXT file...")
    txt_lines = []
    
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        gender = seg.get('gender', 'unknown')
        speaker_label = seg.get('speaker_label', f"Speaker {speaker_id + 1}")
        
        start_min = int(seg["start"] // 60)
        start_sec = int(seg["start"] % 60)
        timestamp = f"{start_min:02d}:{start_sec:02d}"
        
        text = seg.get('text', '').strip()
        if text:
            txt_lines.append(f"[{timestamp}] {speaker_label} ({gender}): {text}")
    
    # Add gender summary
    txt_lines.append("")
    txt_lines.append("# Speaker Gender Detection:")
    for speaker_id, gender in speaker_gender.items():
        txt_lines.append(f"# Speaker {speaker_id + 1}: {gender}")
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))
    
    print(f"  ✅ Complete: {audio_path.name}")
    print(f"     📄 JSON: {output_json.name}")
    print(f"     📄 TXT: {output_txt.name}")
    
    return True


def main():
    ap = argparse.ArgumentParser(description="Batch gender detection for simple diarization files")
    ap.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    ap.add_argument("--json-dir", required=True, help="Directory containing diarized JSON files")
    ap.add_argument("--output-dir", required=True, help="Output directory for gender files")
    ap.add_argument("--pattern", default="*.diarized.json", help="JSON file pattern to match")
    ap.add_argument("--method", default="inaspeech", 
                    choices=["inaspeech", "pitch"],
                    help="Gender detection method")
    args = ap.parse_args()
    
    audio_dir = Path(args.audio_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    
    # Check directories
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return 1
    
    if not json_dir.exists():
        print(f"Error: JSON directory not found: {json_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find JSON files
    json_files = list(json_dir.glob(args.pattern))
    if not json_files:
        print(f"Error: No JSON files found matching pattern '{args.pattern}' in {json_dir}")
        return 1
    
    print(f"🎯 Found {len(json_files)} JSON files to process")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        # Extract base name without .diarized
        base_name = json_file.stem.replace('.diarized', '')
        
        # Find corresponding audio file
        audio_file = None
        for ext in ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']:
            potential_audio = audio_dir / f"{base_name}{ext}"
            if potential_audio.exists():
                audio_file = potential_audio
                break
        
        if audio_file is None:
            print(f"  ❌ Audio file not found for {json_file.name} (tried: {base_name}.wav, .mp3, .mp4, .m4a, .flac, .ogg)")
            error_count += 1
            continue
        
        if process_single_file(audio_file, json_file, output_dir):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(json_files)}")
    print(f"   Successful: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Skipped: {len(json_files) - success_count - error_count}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())

