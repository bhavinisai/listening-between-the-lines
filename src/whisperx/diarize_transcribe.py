import os
import json
import argparse
from pathlib import Path

import torch
import whisperx


def format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe + diarize a WAV file using WhisperX + pyannote (labels speakers)."
    )
    ap.add_argument("--audio", required=True, help="Path to input WAV file")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument(
        "--model",
        default="medium",
        help="Whisper model size: tiny/base/small/medium/large-v2/large-v3",
    )
    ap.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., en). If omitted, Whisper will detect.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cpu or cuda. If omitted, auto-detect.",
    )
    ap.add_argument(
        "--compute_type",
        default=None,
        help="e.g., int8 (cpu), float16 (cuda). If omitted, chosen automatically.",
    )
    ap.add_argument(
        "--min_speakers",
        type=int,
        default=2,
        help="Min number of speakers (set 2 for your case)",
    )
    ap.add_argument(
        "--max_speakers",
        type=int,
        default=2,
        help="Max number of speakers (set 2 for your case)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for ASR (reduce if you hit OOM on GPU).",
    )
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Hugging Face token (often required for pyannote diarization models)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Set env var HUGGINGFACE_TOKEN (or HF_TOKEN).\n"
            "Example:\n"
            "  export HUGGINGFACE_TOKEN='hf_...'\n"
        )

    # Device auto-detect (FIXED: do NOT use whisperx.utils.get_device)
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute type defaults
    if args.compute_type:
        compute_type = args.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"[info] audio        : {audio_path}")
    print(f"[info] outdir       : {outdir}")
    print(f"[info] device       : {device}")
    print(f"[info] compute_type  : {compute_type}")
    print(f"[info] model        : {args.model}")
    print(f"[info] speakers     : {args.min_speakers}..{args.max_speakers}")
    print(f"[info] batch_size   : {args.batch_size}")
    if device == "cuda":
        print(f"[info] cuda device  : {torch.cuda.get_device_name(0)}")

    # 1) Load audio
    audio = whisperx.load_audio(str(audio_path))

    # 2) ASR (transcription)
    asr_model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=compute_type,
        language=args.language,
    )
    result = asr_model.transcribe(audio, batch_size=args.batch_size)

    # Detected language (if not provided)
    lang = result.get("language", args.language) or "en"
    print(f"[info] detected language: {lang}")

    # 3) Alignment (word-level timestamps helps diarization merge quality)
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # 4) Diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(
        audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # 5) Assign speakers to words/segments
    final_result = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    # Save JSON output
    json_path = outdir / f"{audio_path.stem}.whisperx_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # Save readable speaker transcript
    txt_path = outdir / f"{audio_path.stem}.speaker_transcript.txt"

    lines = []
    segments = final_result.get("segments", [])
    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{format_ts(start)} - {format_ts(end)}] {spk}: {text}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[done] JSON saved: {json_path}")
    print(f"[done] TXT  saved: {txt_path}")


if __name__ == "__main__":
    main()
