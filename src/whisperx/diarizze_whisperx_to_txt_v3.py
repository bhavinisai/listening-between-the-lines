#!/usr/bin/env python3
"""
WhisperX ASR + Alignment + Pyannote Diarization -> Speaker 1 / Speaker 2 transcript.

Key improvements over v2:
- merge_short_turns(): merges very short turns that are likely mis-assigned
- smooth_speaker_transitions(): fixes isolated speaker assignments during fast back-and-forth

Requirements:
  pip install whisperx pyannote.audio torch torchaudio soundfile
  export HF_TOKEN="hf_..."

Run:
  python diarizze_whisperx_to_txt_v4.py \
    --audio data/raw_audio/episode.wav \
    --outdir outputs \
    --model medium \
    --compute_type int8 \
    --language en

Outputs:
  - outputs/<stem>.whisperx.json
  - outputs/<stem>.speaker_transcript.txt
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
import time


def fmt_ts(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def patch_torch_load():
    """Patch torch.load to allow unsafe loading for PyTorch 2.6+"""
    import torch
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    print("[info] Patched torch.load for PyTorch 2.6+ compatibility")


def ensure_ffmpeg_on_path() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        str(Path.home() / "bin" / "ffmpeg"),
    ]
    ffmpeg = next((p for p in candidates if p and Path(p).exists()), None)
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found. Install it or add it to PATH.")
    ffdir = str(Path(ffmpeg).parent)
    os.environ["PATH"] = f"{ffdir}:{os.environ.get('PATH', '')}"
    return ffmpeg


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_short_turns(turns, min_turn_duration=0.5):
    """
    Merge very short turns into the previous turn.
    Short turns during fast back-and-forth are often mis-assigned.

    e.g:
    Speaker 1: 0.0 - 5.0
    Speaker 1: 5.1 - 5.3  <- too short, likely noise
    Speaker 2: 5.4 - 10.0
    ->
    Speaker 1: 0.0 - 5.3  <- merged
    Speaker 2: 5.4 - 10.0
    """
    if not turns:
        return turns

    merged = [turns[0].copy()]
    for turn in turns[1:]:
        duration = turn['end'] - turn['start']
        prev = merged[-1]

        if duration < min_turn_duration:
            # Extend previous turn to cover this short one
            prev['end'] = turn['end']
        else:
            merged.append(turn.copy())

    print(f"[info] merge_short_turns: {len(turns)} -> {len(merged)} turns")
    return merged


def smooth_speaker_transitions(turns, context_window=3):
    """
    Fix isolated speaker assignments during rapid turn-taking.

    If a speaker appears only once surrounded by the other speaker,
    it's likely a mis-assignment. Reassign to the dominant neighbor.

    e.g:
    S1, S1, S2, S1, S1  <- S2 is isolated, likely wrong
    ->
    S1, S1, S1, S1, S1  <- reassigned
    """
    if len(turns) < 3:
        return turns

    smoothed = [t.copy() for t in turns]

    for i in range(1, len(smoothed) - 1):
        current = smoothed[i]['speaker']

        # Get neighboring speakers within context window
        start_idx = max(0, i - context_window)
        end_idx = min(len(smoothed), i + context_window + 1)
        neighbors = [smoothed[j]['speaker'] for j in range(start_idx, end_idx) if j != i]

        # Count how often each neighbor speaker appears
        neighbor_counts = defaultdict(int)
        for spk in neighbors:
            neighbor_counts[spk] += 1

        dominant_neighbor = max(neighbor_counts, key=neighbor_counts.get)
        dominant_count = neighbor_counts[dominant_neighbor]
        total_neighbors = len(neighbors)

        # Reassign if current speaker is isolated among neighbors
        if current != dominant_neighbor and dominant_count >= (total_neighbors * 0.75):
            smoothed[i]['speaker'] = dominant_neighbor

    changes = sum(1 for o, s in zip(turns, smoothed) if o['speaker'] != s['speaker'])
    print(f"[info] smooth_speaker_transitions: reassigned {changes} turns")
    return smoothed


def assign_speaker_to_segment(seg, turns, min_overlap=0.10):
    """Assign best matching speaker to a segment by overlap."""
    s0, s1 = float(seg["start"]), float(seg["end"])
    best_spk, best_ov = "Unknown", 0.0
    for t in turns:
        ov = overlap(s0, s1, t["start"], t["end"])
        if ov > best_ov:
            best_ov = ov
            best_spk = t["speaker"]
    return best_spk if best_ov >= min_overlap else "Unknown"


def build_speaker_mapping(turns, top_k=2):
    """Map raw diarization labels to Speaker 1 / Speaker 2."""
    dur, first = {}, {}
    for t in turns:
        spk = t["speaker"]
        dur[spk] = dur.get(spk, 0.0) + max(0.0, t["end"] - t["start"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="medium", help="Whisper model (small/medium/large-v3)")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if not set)")
    ap.add_argument("--compute_type", default=None, help="float16 / int8")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language code")
    ap.add_argument("--min_overlap", type=float, default=0.10,
                    help="Min overlap (sec) to assign a speaker")
    ap.add_argument("--min_turn_duration", type=float, default=0.5,
                    help="Turns shorter than this (sec) are merged (default: 0.5)")
    ap.add_argument("--context_window", type=int, default=3,
                    help="Context window for smoothing isolated speaker assignments (default: 3)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN for pyannote diarization.")

    ffmpeg = ensure_ffmpeg_on_path()
    print(f"[info] ffmpeg: {ffmpeg}")

    import torch
    patch_torch_load()

    import whisperx
    from pyannote.audio import Pipeline

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"[info] device={device} compute_type={compute_type} model={args.model}")

    # 1) ASR
    t0 = time.time()
    print("[info] 1) loading WhisperX model...")
    asr_model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )

    print("[info] 2) loading audio...")
    audio_arr = whisperx.load_audio(str(audio_path))

    asr_kwargs = {}
    if args.language:
        asr_kwargs["language"] = args.language

    print("[info] 3) transcribing...")
    asr_result = asr_model.transcribe(audio_arr, **asr_kwargs)
    print(f"[done] transcribe: {time.time()-t0:.1f}s")

    # 2) Alignment
    t1 = time.time()
    print("[info] 4) aligning...")
    lang = asr_result.get("language") or (args.language if args.language else "en")
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(asr_result["segments"], align_model, metadata, audio_arr, device)
    print(f"[done] align: {time.time()-t1:.1f}s")

    # 3) Diarization
    t2 = time.time()
    print("[info] 5) diarizing (pyannote)...")
    diarize = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    try:
        diarize.to(torch.device(device))
    except Exception:
        pass

    diar = diarize(
        str(audio_path),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    turns = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": spk})

    print(f"[done] diarize: {time.time()-t2:.1f}s | turns={len(turns)}")

    # Post-process turns to fix rapid turn-taking issues
    turns = merge_short_turns(turns, min_turn_duration=args.min_turn_duration)
    turns = smooth_speaker_transitions(turns, context_window=args.context_window)

    # 4) Assign speakers by overlap
    t3 = time.time()
    print("[info] 6) assigning speakers...")
    for seg in result["segments"]:
        seg["speaker_raw"] = assign_speaker_to_segment(seg, turns, args.min_overlap)

    mapping = build_speaker_mapping(turns, top_k=2)
    for seg in result["segments"]:
        raw = seg.get("speaker_raw", "Unknown")
        seg["speaker"] = mapping.get(raw, "Unknown")

    print(f"[done] assign speakers: {time.time()-t3:.1f}s")

    # Save JSON
    json_path = outdir / f"{audio_path.stem}.whisperx.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save readable transcript
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
