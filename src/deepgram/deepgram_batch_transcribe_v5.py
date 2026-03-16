#!/usr/bin/env python3
"""
Simple Deepgram Diarization Script
Focus on accurate 2-speaker detection and labeling.
"""

import argparse
import json
import os
import pathlib
import requests
import sys
from typing import Dict, Any, List, Tuple


def transcribe_audio(audio_path: str, api_key: str) -> Dict[str, Any]:
    """Transcribe audio with Deepgram API."""
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    
    params = {
        "model": "nova-2",
        "language": "en",
        "punctuate": "true",
        "utterances": "true",
        "diarize": "true",
        "smart_format": "true"
    }
    
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, params=params, data=audio_file)
            
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return {"error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"error": str(e)}


def extract_speakers(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and clean speaker segments."""
    utterances = response.get("results", {}).get("utterances", [])
    
    if not utterances:
        print("No utterances found in response")
        return []
    
    # Filter and clean segments
    segments = []
    for utterance in utterances:
        if utterance.get("transcript", "").strip():
            segments.append({
                "speaker_id": int(utterance.get("speaker", 0)),
                "speaker_label": f"Speaker {int(utterance.get('speaker', 0)) + 1}",
                "start": float(utterance.get("start", 0.0)),
                "end": float(utterance.get("end", 0.0)),
                "text": utterance.get("transcript", "").strip(),
                "confidence": float(utterance.get("confidence", 0.0))
            })
    
    return segments


def consolidate_speakers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Consolidate to exactly 2 speakers with confidence scoring."""
    if not segments:
        return []
    
    # Group by original speaker ID
    speaker_groups = {}
    for seg in segments:
        speaker_id = seg["speaker_id"]
        if speaker_id not in speaker_groups:
            speaker_groups[speaker_id] = []
        speaker_groups[speaker_id].append(seg)
    
    # Calculate speaking time for each speaker
    speaker_stats = {}
    for speaker_id, segs in speaker_groups.items():
        total_time = sum(seg["end"] - seg["start"] for seg in segs)
        avg_confidence = sum(seg["confidence"] for seg in segs) / len(segs)
        
        speaker_stats[speaker_id] = {
            "total_time": total_time,
            "utterance_count": len(segs),
            "avg_confidence": avg_confidence,
            "segments": segs
        }
    
    # Sort by speaking time (most speaking first)
    sorted_speakers = sorted(speaker_stats.keys(), 
                          key=lambda x: speaker_stats[x]["total_time"], 
                          reverse=True)
    
    # Take top 2 speakers
    top_speakers = sorted_speakers[:2]
    
    # Re-map to Speaker 1, Speaker 2
    consolidated = []
    for i, original_id in enumerate(top_speakers):
        for seg in speaker_stats[original_id]["segments"]:
            consolidated.append({
                "speaker_id": i,  # 0 or 1
                "speaker_label": f"Speaker {i + 1}",  # Speaker 1, Speaker 2
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "confidence": seg["confidence"]
            })
    
    # Sort by time
    consolidated.sort(key=lambda x: x["start"])
    
    return consolidated


def write_diarized_txt(segments: List[Dict[str, Any]], output_path: str):
    """Write diarized text file."""
    lines = []
    for seg in segments:
        start_min = int(seg["start"] // 60)
        start_sec = int(seg["start"] % 60)
        timestamp = f"{start_min:02d}:{start_sec:02d}"
        
        speaker_label = seg["speaker_label"]  # Speaker 1, Speaker 2
        text = seg["text"]
        
        lines.append(f"[{timestamp}] {speaker_label}: {text}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Written: {output_path}")


def write_json(segments: List[Dict[str, Any]], output_path: str):
    """Write JSON file with speaker info."""
    data = {
        "total_segments": len(segments),
        "speakers": [
            {
                "speaker_id": i,
                "speaker_label": f"Speaker {i + 1}"
            }
            for i in range(2)
        ],
        "segments": segments
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple 2-speaker diarization with Deepgram")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Single input audio file")
    src.add_argument("--input-dir", help="Directory of audio files")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--api-key", help="Deepgram API key (or set DEEPGRAM_API_KEY)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Error: Set DEEPGRAM_API_KEY environment variable or use --api-key")
        return 1
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get audio files
    audio_files = []
    if args.input:
        # Single file mode
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1
        audio_files = [input_path]
    elif args.input_dir:
        # Directory mode
        input_dir = pathlib.Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1
        
        # Find audio files
        audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg'}
        audio_files = sorted([f for f in input_dir.iterdir() 
                          if f.suffix.lower() in audio_extensions and f.is_file()])
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return 1
    else:
        print("Error: Must specify --input or --input-dir")
        return 1
    
    print(f"Processing {len(audio_files)} audio files...")
    print(f"Output directory: {output_dir}")
    
    success_count = 0
    error_count = 0
    
    # Process each file
    for audio_path in audio_files:
        print(f"\n🎵 Processing: {audio_path.name}")
        
        # Generate output paths
        stem = audio_path.stem
        txt_output = output_dir / f"{stem}.diarized.txt"
        json_output = output_dir / f"{stem}.diarized.json"
        
        # Skip if outputs exist
        if txt_output.exists() and json_output.exists():
            print(f"⏭  Skipping {audio_path.name} (already processed)")
            success_count += 1
            continue
        
        # Transcribe
        response = transcribe_audio(str(audio_path), api_key)
        
        if "error" in response:
            print(f"❌ Transcription failed: {response['error']}")
            error_count += 1
            continue
        
        # Extract and process speakers
        segments = extract_speakers(response)
        
        if not segments:
            print(f"⚠️  No valid segments found in {audio_path.name}")
            error_count += 1
            continue
        
        print(f"📊 Found {len(segments)} raw segments")
        
        # Consolidate to 2 speakers
        consolidated = consolidate_speakers(segments)
        
        print(f"👥 Consolidated to 2 speakers:")
        for i in range(2):
            speaker_segments = [s for s in consolidated if s["speaker_id"] == i]
            if speaker_segments:
                total_time = sum(s["end"] - s["start"] for s in speaker_segments)
                print(f"   Speaker {i + 1}: {len(speaker_segments)} segments, {total_time:.1f}s total")
        
        # Write outputs
        write_diarized_txt(consolidated, str(txt_output))
        write_json(consolidated, str(json_output))
        
        print(f"✅ Complete: {audio_path.name}")
        success_count += 1
    
    # Summary
    print(f"\n� Summary:")
    print(f"   Total files: {len(audio_files)}")
    print(f"   Successful: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Skipped: {len(audio_files) - success_count - error_count}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

