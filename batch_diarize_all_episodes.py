#!/usr/bin/env python3
"""
Batch diarization script - processes all WAV files in a directory.

Usage:
  python batch_diarize_all_episodes.py \
    --audio_dir data/raw_audio \
    --outdir outputs \
    --pattern "Episode*.wav" \
    --model medium
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import shutil


def fmt_ts(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def patch_torch_load():
    """Patch torch.load to allow unsafe loading for PyTorch 2.8+"""
    import torch
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load


def ensure_ffmpeg_on_path() -> str:
    """Ensure ffmpeg is discoverable."""
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        str(Path.home() / "bin" / "ffmpeg"),
    ]
    ffmpeg = next((p for p in candidates if p and Path(p).exists()), None)
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found. Install it or add to PATH.")
    ffdir = str(Path(ffmpeg).parent)
    os.environ["PATH"] = f"{ffdir}:{os.environ.get('PATH','')}"
    return ffmpeg


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker_to_segment(seg, turns, min_overlap=0.10):
    s0, s1 = float(seg["start"]), float(seg["end"])
    best_spk, best_ov = "Unknown", 0.0
    for t in turns:
        ov = overlap(s0, s1, t["start"], t["end"])
        if ov > best_ov:
            best_ov = ov
            best_spk = t["speaker"]
    return best_spk if best_ov >= min_overlap else "Unknown"


def build_speaker_mapping(turns, top_k=2):
    dur = {}
    first = {}
    for t in turns:
        spk = t["speaker"]
        dur[spk] = dur.get(spk, 0.0) + max(0.0, (t["end"] - t["start"]))
        first.setdefault(spk, t["start"])
    
    if not dur:
        return {}
    
    top = sorted(dur.keys(), key=lambda s: dur[s], reverse=True)[:top_k]
    top_sorted = sorted(top, key=lambda s: first.get(s, 1e9))
    
    mapping = {}
    if len(top_sorted) >= 1:
        mapping[top_sorted[0]] = "Speaker 1"
    if len(top_sorted) >= 2:
        mapping[top_sorted[1]] = "Speaker 2"
    return mapping


def process_single_file(
    audio_path: Path,
    outdir: Path,
    asr_model,
    align_model,
    align_metadata,
    diarize_pipeline,
    device,
    args,
):
    """Process a single audio file."""
    print(f"\n{'='*70}")
    print(f"Processing: {audio_path.name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        import whisperx
        
        # Load audio
        print("[1/5] Loading audio...")
        audio_arr = whisperx.load_audio(str(audio_path))
        
        # Transcribe
        print("[2/5] Transcribing...")
        asr_kwargs = {}
        if args.language:
            asr_kwargs["language"] = args.language
        asr_result = asr_model.transcribe(audio_arr, **asr_kwargs)
        
        # Align
        print("[3/5] Aligning...")
        result = whisperx.align(
            asr_result["segments"],
            align_model,
            align_metadata,
            audio_arr,
            device
        )
        
        # Diarize
        print("[4/5] Diarizing...")
        diar = diarize_pipeline(
            str(audio_path),
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        
        turns = []
        for turn, _, spk in diar.itertracks(yield_label=True):
            turns.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": spk
            })
        
        # Assign speakers
        print("[5/5] Assigning speakers...")
        for seg in result["segments"]:
            seg["speaker_raw"] = assign_speaker_to_segment(
                seg, turns, min_overlap=args.min_overlap
            )
        
        mapping = build_speaker_mapping(turns, top_k=2)
        
        for seg in result["segments"]:
            raw = seg.get("speaker_raw", "Unknown")
            seg["speaker"] = mapping.get(raw, "Unknown")
        
        # Save JSON
        json_path = outdir / f"{audio_path.stem}.whisperx.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save transcript
        txt_lines = []
        for seg in result.get("segments", []):
            spk = seg.get("speaker", "Unknown")
            st = fmt_ts(float(seg["start"]))
            et = fmt_ts(float(seg["end"]))
            text = (seg.get("text") or "").strip()
            if text:
                txt_lines.append(f"[{st} - {et}] {spk}: {text}")
        
        txt_path = outdir / f"{audio_path.stem}.speaker_transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))
        
        elapsed = time.time() - start_time
        print(f"✓ Success! Time: {elapsed/60:.1f} minutes")
        print(f"  JSON: {json_path.name}")
        print(f"  TXT : {txt_path.name}")
        
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ FAILED! Error: {e}")
        print(f"  Time: {elapsed:.1f}s")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Directory containing WAV files")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--pattern", default="Episode*.wav", help="File pattern to match")
    ap.add_argument("--model", default="medium", help="Whisper model")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--compute_type", default=None, help="float16 / int8")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language (e.g., en)")
    ap.add_argument("--min_overlap", type=float, default=0.10)
    ap.add_argument("--start", type=int, default=None, help="Start from episode N")
    ap.add_argument("--end", type=int, default=None, help="End at episode N")
    args = ap.parse_args()
    
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        sys.exit(1)
    
    # Find all matching audio files
    audio_files = sorted(audio_dir.glob(args.pattern))
    
    if not audio_files:
        print(f"Error: No files matching '{args.pattern}' found in {audio_dir}")
        sys.exit(1)
    
    # Apply start/end filters
    if args.start is not None or args.end is not None:
        filtered_files = []
        for f in audio_files:
            try:
                # Extract episode number from filename like Episode01_...
                ep_num = int(f.stem.split('_')[0].replace('Episode', ''))
                if args.start is not None and ep_num < args.start:
                    continue
                if args.end is not None and ep_num > args.end:
                    continue
                filtered_files.append(f)
            except (ValueError, IndexError):
                # If we can't parse episode number, include the file
                filtered_files.append(f)
        audio_files = filtered_files
    
    print(f"Found {len(audio_files)} files to process")
    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {f.name}")
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir.absolute()}")
    
    # HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Error: Set HF_TOKEN for pyannote diarization")
        sys.exit(1)
    
    # Setup
    ffmpeg = ensure_ffmpeg_on_path()
    print(f"ffmpeg: {ffmpeg}")
    
    import torch
    patch_torch_load()
    
    import whisperx
    from pyannote.audio import Pipeline
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"device={device} compute_type={compute_type} model={args.model}")
    
    # Load models once
    print("\n" + "="*70)
    print("LOADING MODELS (this happens once for all files)")
    print("="*70)
    
    print("[1/3] Loading WhisperX ASR model...")
    asr_model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )
    
    print("[2/3] Loading alignment model...")
    lang = args.language if args.language else "en"
    align_model, align_metadata = whisperx.load_align_model(
        language_code=lang,
        device=device
    )
    
    print("[3/3] Loading diarization pipeline...")
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    try:
        diarize_pipeline.to(torch.device(device))
    except Exception:
        pass
    
    print("\n" + "="*70)
    print(f"PROCESSING {len(audio_files)} FILES")
    print("="*70)
    
    # Process all files
    results = []
    total_start = time.time()
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]")
        success, elapsed = process_single_file(
            audio_path,
            outdir,
            asr_model,
            align_model,
            align_metadata,
            diarize_pipeline,
            device,
            args,
        )
        results.append({
            "file": audio_path.name,
            "success": success,
            "time_seconds": elapsed,
            "time_minutes": elapsed/60
        })
    
    # Summary
    total_time = time.time() - total_start
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files:    {len(results)}")
    print(f"Successful:     {success_count}")
    print(f"Failed:         {fail_count}")
    print(f"Total time:     {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Avg time/file:  {total_time/len(results)/60:.1f} minutes")
    print(f"Output dir:     {outdir.absolute()}")
    
    if fail_count > 0:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}")
    
    # Save summary
    summary_path = outdir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": len(results),
            "successful": success_count,
            "failed": fail_count,
            "total_time_minutes": total_time/60,
            "total_time_hours": total_time/3600,
            "avg_time_per_file_minutes": total_time/len(results)/60,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed summary saved: {summary_path.name}")


if __name__ == "__main__":
    main()
