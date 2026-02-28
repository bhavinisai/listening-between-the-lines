#!/usr/bin/env python3
"""
WhisperX ASR + alignment + Pyannote diarization -> Speaker 1 / Speaker 2 transcript.

Key design choice:
- We DO NOT rely on whisperx.assign_word_speakers() (often brittle across versions).
- Instead, we:
  1) run pyannote diarization on the audio FILE
  2) extract diarization turns
  3) assign a speaker to each Whisper segment by maximum time overlap
  4) map the top-2 diarization speakers -> Speaker 1 / Speaker 2 (stable ordering)

Requirements:
  pip install whisperx pyannote.audio torch torchaudio soundfile
  export HF_TOKEN="hf_..."   (must have access to pyannote diarization models)
  ffmpeg must be installed and on PATH (we also try common locations).

Run:
  python diarize_whisperx_to_txt.py \
    --audio data/raw_audio/Sunita_Williams_clip_60s.wav \
    --outdir outputs \
    --model medium \
    --compute_type int8 \
    --language en \
    --min_speakers 2 --max_speakers 4
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


def patch_torch_load():
    """Patch torch.load to allow unsafe loading for PyTorch 2.8+"""
    import torch
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # Force weights_only=False for PyTorch 2.8+
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    print("[info] Patched torch.load for PyTorch 2.8+ compatibility")


def ensure_ffmpeg_on_path() -> str:
    """
    Ensure an ffmpeg executable is discoverable even in a stripped environment.
    Prefers:
      1) whatever `shutil.which("ffmpeg")` finds
      2) /usr/local/bin/ffmpeg
      3) /opt/homebrew/bin/ffmpeg
      4) ~/bin/ffmpeg (common on clusters)
    """
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        str(Path.home() / "bin" / "ffmpeg"),
    ]
    ffmpeg = next((p for p in candidates if p and Path(p).exists()), None)
    if not ffmpeg:
        raise FileNotFoundError(
            "ffmpeg not found. Install it (e.g., brew install ffmpeg) or add it to PATH."
        )
    ffdir = str(Path(ffmpeg).parent)
    os.environ["PATH"] = f"{ffdir}:{os.environ.get('PATH','')}"
    return ffmpeg


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker_to_segment(seg, turns, min_overlap=0.10):
    """
    seg: whisper segment dict with "start","end"
    turns: list of {"start","end","speaker"} from diarization
    """
    s0, s1 = float(seg["start"]), float(seg["end"])
    best_spk, best_ov = "Unknown", 0.0
    for t in turns:
        ov = overlap(s0, s1, t["start"], t["end"])
        if ov > best_ov:
            best_ov = ov
            best_spk = t["speaker"]
    return best_spk if best_ov >= min_overlap else "Unknown"


def build_speaker_mapping(turns, top_k=2):
    """
    Map diarization raw speaker labels to 'Speaker 1' / 'Speaker 2'.

    - First choose top_k speakers by total speaking time (filters out tiny clusters).
    - Then order those top speakers by who appears first (stable ordering).
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to wav/audio file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="medium", help="Whisper model (e.g., small, medium, large-v3)")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if not set)")
    ap.add_argument("--compute_type", default=None, help="float16 / int8 / int8_float16 (auto if not set)")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language hint (e.g., en). Use '' to auto-detect.")
    ap.add_argument("--min_overlap", type=float, default=0.10, help="Min overlap (sec) to assign a speaker.")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # HF token for pyannote diarization
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_TOKEN) for pyannote diarization.")

    ffmpeg = ensure_ffmpeg_on_path()
    print(f"[info] ffmpeg: {ffmpeg}")

    # CRITICAL: Patch torch.load BEFORE importing torch/whisperx
    import torch
    patch_torch_load()
    
    import whisperx
    from pyannote.audio import Pipeline

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    print(f"[info] device={device} compute_type={compute_type} model={args.model}")

    # 1) WhisperX ASR
    t0 = time.time()
    print("[info] 1) loading WhisperX model...")
    asr_model = whisperx.load_model(
    args.model,
    device=device,
    compute_type=compute_type,
    vad_method="silero",   # <-- avoids pyannote VAD + torch.load issue
)

    print("[info] 2) loading audio...")
    audio_arr = whisperx.load_audio(str(audio_path))

    asr_kwargs = {}
    if args.language:
        asr_kwargs["language"] = args.language

    print("[info] 3) transcribing...")
    asr_result = asr_model.transcribe(audio_arr, **asr_kwargs)
    print(f"[done] transcribe: {time.time()-t0:.1f}s")

    # 2) Alignment (word timestamps + better segment timing)
    t1 = time.time()
    print("[info] 4) aligning...")
    lang = asr_result.get("language") or (args.language if args.language else "en")
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(asr_result["segments"], align_model, metadata, audio_arr, device)
    print(f"[done] align: {time.time()-t1:.1f}s")

    # 3) Diarization (pyannote) on FILE PATH (most reliable)
    t2 = time.time()
    print("[info] 5) diarizing (pyannote)...")
    diarize = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    # Some environments support .to(device); safe to try:
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

    # 4) Assign speaker to each Whisper segment by overlap
    t3 = time.time()
    print("[info] 6) assigning speakers by overlap...")
    for seg in result["segments"]:
        seg["speaker_raw"] = assign_speaker_to_segment(seg, turns, min_overlap=args.min_overlap)

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
