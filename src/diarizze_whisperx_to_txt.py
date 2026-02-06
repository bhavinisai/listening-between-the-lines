#!/usr/bin/env python3
"""
WhisperX transcript with PyTorch 2.8 compatibility patch.

This script works around PyTorch 2.8's stricter pickle security by patching torch.load
"""

import os
import json
import argparse
from pathlib import Path
import shutil
import time


def fmt_ts(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def ensure_ffmpeg_on_path() -> str:
    """Ensure ffmpeg is available."""
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    ffmpeg = next((p for p in candidates if p and Path(p).exists()), None)
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found. Install it or ensure it's on PATH.")
    ffdir = str(Path(ffmpeg).parent)
    os.environ["PATH"] = f"{ffdir}:{os.environ.get('PATH','')}"
    return ffmpeg


def patch_torch_load():
    """Patch torch.load to allow unsafe loading for PyTorch 2.8+"""
    import torch
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # Force weights_only=False for PyTorch 2.8+
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    print("[info] Patched torch.load to allow legacy model loading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to wav")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="medium", help="Whisper model")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--compute_type", default=None, help="float16 / int8")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--language", default="en", help="Language hint")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN for diarization.")

    ffmpeg = ensure_ffmpeg_on_path()
    print(f"[info] ffmpeg: {ffmpeg}")

    # CRITICAL: Patch torch.load BEFORE importing whisperx
    import torch
    patch_torch_load()
    
    import whisperx

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

    print(f"[info] device={device} compute_type={compute_type} model={args.model}")

    # 1) ASR
    t0 = time.time()
    print("[info] 1) loading WhisperX model...")
    asr_model = whisperx.load_model(args.model, device=device, compute_type=compute_type)

    print("[info] 2) loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    asr_kwargs = {}
    if args.language:
        asr_kwargs["language"] = args.language

    print("[info] 3) transcribing...")
    result = asr_model.transcribe(audio, **asr_kwargs)
    print(f"[done] transcribe: {time.time() - t0:.1f}s")

    # 2) Alignment
    t1 = time.time()
    print("[info] 4) aligning...")
    lang = result.get("language") or (args.language if args.language else "en")
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    print(f"[done] align: {time.time() - t1:.1f}s")

    # 3) Diarization
    t2 = time.time()
    print("[info] 5) diarizing...")
    try:
        from pyannote.audio import Pipeline
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(torch.device(device))
        diarization_segments = diarize_model(
            audio,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
    except Exception as e:
        print(f"[warning] Diarization failed: {e}")
        diarization_segments = None
    
    print(f"[done] diarize: {time.time() - t2:.1f}s")

    # 4) Assign speakers
    t3 = time.time()
    print("[info] 6) assigning speakers...")
    if diarization_segments is not None:
        result = whisperx.assign_word_speakers(diarization_segments, result)
    else:
        print("[warning] Skipping speaker assignment")
    print(f"[done] assign speakers: {time.time() - t3:.1f}s")

    # Save outputs
    json_path = outdir / f"{audio_path.stem}.whisperx.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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

    print(f"[done] JSON: {json_path}")
    print(f"[done] TXT : {txt_path}")


if __name__ == "__main__":
    main()
