#!/usr/bin/env python3
"""
Simple test to process Deepgram sentences.json files.
"""

import json
import argparse
from pathlib import Path


def process_single_file(input_file, output_dir):
    """Process a single Deepgram sentences.json file."""
    print(f"\n🔍 Processing: {input_file.name}")
    
    try:
        # Load Deepgram sentences
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        
        if not segments:
            print(f"  ❌ No segments found in {input_file.name}")
            return False
        
        # Simple mock gender assignment
        speaker_gender = {
            0: 'male',   # Speaker 0 (host)
            1: 'female'   # Speaker 1 (guest)
        }
        
        # Add gender to segments
        updated_segments = []
        for seg in segments:
            speaker_id = seg.get('speaker_id', 'Unknown')
            gender = speaker_gender.get(speaker_id, 'unknown')
            seg_copy = seg.copy()  # Make a copy to avoid modifying original
            seg_copy['gender'] = gender
            updated_segments.append(seg_copy)
        
        # Generate output paths
        base_name = input_file.stem
        json_output = output_dir / f"{base_name}.test_gender.json"
        txt_output = output_dir / f"{base_name}.test_gender.txt"
        
        # Write JSON
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                'segments': updated_segments,
                'speaker_gender': speaker_gender
            }, f, ensure_ascii=False, indent=2)
        
        # Write TXT
        lines = []
        for seg in updated_segments:
            start = seg.get('start', 0)
            speaker_role = seg.get('speaker_role', 'Unknown')
            gender = seg.get('gender', 'unknown')
            text = seg.get('transcript', '').strip()
            
            if text:
                timestamp = f"{int(start//60):02d}:{int(start%60):02d}"
                line = f"[{timestamp}] {speaker_role} ({gender}): {text}"
                lines.append(line)
        
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  ✅ Complete: {input_file.name}")
        print(f"     📄 JSON: {json_output.name}")
        print(f"     📄 TXT: {txt_output.name}")
        print(f"     🎯 Host: SPEAKER_00 (male)")
        print(f"     🎯 Guest: SPEAKER_01 (female)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_file.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Test Deepgram sentences.json processing")
    ap.add_argument("--input_file", required=True, help="Single input file")
    ap.add_argument("--output_dir", help="Output directory (default: same as input)")
    
    args = ap.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else input_file.parent
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    return 0 if process_single_file(input_file, output_dir) else 1


if __name__ == "__main__":
    exit(main())
