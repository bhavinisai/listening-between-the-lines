import os
import json
import argparse
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline


def sec_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def normalize_speaker_name(speaker_label: str) -> str:
    if not speaker_label:
        return "Speaker Unknown"
    if speaker_label.startswith("SPEAKER_"):
        try:
            idx = int(speaker_label.split("_")[-1])
            return f"Speaker {idx + 1}"
        except Exception:
            pass
    return speaker_label.replace("_", " ").title()


def write_formatted_transcript(segments, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = sec_to_timestamp(float(seg.get("start", 0.0)))
            end = sec_to_timestamp(float(seg.get("end", 0.0)))
            text = seg.get("text", "").strip()
            speaker = normalize_speaker_name(seg.get("speaker", "Speaker Unknown"))
            if text:
                f.write(f"[{start} - {end}] {speaker}: {text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="large-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Pass --hf_token or set HF_TOKEN")

    device = "cuda"
    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(args.model, device, compute_type=args.compute_type)
    result = model.transcribe(audio, batch_size=args.batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(
        audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    base = audio_path.stem
    with open(output_dir / f"{base}_whisperx_diarized.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    write_formatted_transcript(result["segments"], output_dir / f"{base}_speaker_transcript.txt")


if __name__ == "__main__":
    main()
