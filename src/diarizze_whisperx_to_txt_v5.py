#!/usr/bin/env python3
"""
WhisperX ASR + alignment + Pyannote diarization -> Host / Guest transcript.

Role mapping without host_ref:
- Uses heuristic host scoring (questions, framing phrases, short interjections).
- Falls back to speaking time if heuristic is inconclusive.

Requirements:
  pip install whisperx pyannote.audio torch torchaudio soundfile
  export HF_TOKEN="hf_..."   (must have access to pyannote diarization models)
  ffmpeg must be installed and on PATH.

Run:
  python diarize_whisperx_to_txt.py \
    --audio path/to/audio.wav \
    --outdir outputs \
    --model medium \
    --compute_type int8 \
    --language en \
    --min_speakers 2 --max_speakers 4 \
    --role_mode heuristic
"""

import os
import json
import argparse
from pathlib import Path
import shutil
import time
import re


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
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    print("[info] Patched torch.load for PyTorch 2.8+ compatibility")


def ensure_ffmpeg_on_path() -> str:
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
    s0, s1 = float(seg["start"]), float(seg["end"])
    best_spk, best_ov = "Unknown", 0.0
    for t in turns:
        ov = overlap(s0, s1, t["start"], t["end"])
        if ov > best_ov:
            best_ov = ov
            best_spk = t["speaker"]
    return best_spk if best_ov >= min_overlap else "Unknown"


def top_speakers_by_time(turns, top_k=2):
    dur = {}
    for t in turns:
        spk = t["speaker"]
        dur[spk] = dur.get(spk, 0.0) + max(0.0, (t["end"] - t["start"]))
    top = sorted(dur.keys(), key=lambda s: dur[s], reverse=True)[:top_k]
    return top, dur


def first_appearance(turns):
    first = {}
    for t in turns:
        first.setdefault(t["speaker"], t["start"])
    return first


# ---------------- Role heuristics ---------------- #

HOST_PHRASES = [
    r"\btoday'?s episode\b",
    r"\bwelcome\b",
    r"\bsubscribe\b",
    r"\bcomment(s)?\b",
    r"\bshare (it|this)\b",
    r"\benjoy (the )?show\b",
    r"\bthank you for (watching|listening)\b",
    r"\bwe spoke to\b",
    r"\bintroduc(e|ing)\b",
]

QUESTION_CUES = [
    r"\?$",
    r"\bwhy\b",
    r"\bhow\b",
    r"\bwhat\b",
    r"\bwhen\b",
    r"\bwhere\b",
    r"\bdo you\b",
    r"\bcan you\b",
    r"\btell me\b",
    r"\bhelp me\b",
    r"\bwalk me through\b",
]

SHORT_INTERJECTIONS = [
    r"^\s*(yeah|yes|right|okay|ok|hmm|interesting|true|got it)\s*[\.\!]*\s*$"
]


def host_score_for_text(text: str) -> float:
    """
    Score a segment as "host-like".
    Weighted features:
      - Questions (strong)
      - Episode framing phrases (strong)
      - Short interjections (weak-medium)
    """
    if not text:
        return 0.0
    t = text.strip().lower()

    score = 0.0

    # questions
    for pat in QUESTION_CUES:
        if re.search(pat, t):
            score += 2.0
            break
    if t.endswith("?"):
        score += 2.0

    # framing
    for pat in HOST_PHRASES:
        if re.search(pat, t):
            score += 3.0

    # short interjections
    for pat in SHORT_INTERJECTIONS:
        if re.match(pat, t):
            score += 0.75
            break

    return score


def map_roles_no_ref(result_segments, turns, role_mode="heuristic"):
    """
    Returns mapping raw_speaker -> "Host"/"Guest" for top-2 speakers.
    """
    top2, dur = top_speakers_by_time(turns, top_k=2)
    if len(top2) < 2:
        return {top2[0]: "Host"} if top2 else {}

    spkA, spkB = top2[0], top2[1]
    first = first_appearance(turns)

    if role_mode == "time":
        # host = more speaking time
        host = spkA if dur.get(spkA, 0) >= dur.get(spkB, 0) else spkB
        guest = spkB if host == spkA else spkA
        return {host: "Host", guest: "Guest"}

    if role_mode == "first":
        # host = who appears first
        host = spkA if first.get(spkA, 1e9) <= first.get(spkB, 1e9) else spkB
        guest = spkB if host == spkA else spkA
        return {host: "Host", guest: "Guest"}

    # heuristic mode
    score = {spkA: 0.0, spkB: 0.0}
    for seg in result_segments:
        raw = seg.get("speaker_raw")
        if raw not in score:
            continue
        txt = (seg.get("text") or "")
        score[raw] += host_score_for_text(txt)

    # Decide with thresholds
    a, b = score[spkA], score[spkB]
    # If one clearly higher, choose it
    if abs(a - b) >= 2.5:
        host = spkA if a > b else spkB
    else:
        # fallback to speaking time if heuristic weak/tied
        host = spkA if dur.get(spkA, 0) >= dur.get(spkB, 0) else spkB

    guest = spkB if host == spkA else spkA
    return {host: "Host", guest: "Guest"}


# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to wav/audio file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="medium", help="Whisper model")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if not set)")
    ap.add_argument("--compute_type", default=None, help="float16 / int8 / int8_float16 (auto if not set)")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language hint (e.g., en). Use '' to auto-detect.")
    ap.add_argument("--min_overlap", type=float, default=0.10, help="Min overlap (sec) to assign a speaker.")
    ap.add_argument("--role_mode", default="heuristic", choices=["heuristic", "time", "first"],
                    help="How to map top-2 diarization speakers to Host/Guest.")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_TOKEN) for pyannote diarization.")

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

    # 4) Assign raw speaker to segments by overlap
    t3 = time.time()
    print("[info] 6) assigning speakers by overlap...")
    for seg in result["segments"]:
        seg["speaker_raw"] = assign_speaker_to_segment(seg, turns, min_overlap=args.min_overlap)

    # 5) Map raw speakers -> Host/Guest (top-2 only)
    role_map = map_roles_no_ref(result["segments"], turns, role_mode=args.role_mode)

    for seg in result["segments"]:
        raw = seg.get("speaker_raw", "Unknown")
        seg["speaker"] = role_map.get(raw, "Other")  # Other = any extra diarization speaker beyond top-2

    print(f"[done] role mapping ({args.role_mode}): {role_map}")
    print(f"[done] assign speakers: {time.time()-t3:.1f}s")

    # Save JSON
    json_path = outdir / f"{audio_path.stem}.whisperx.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save readable transcript
    txt_lines = []
    for seg in result.get("segments", []):
        role = seg.get("speaker", "Other")
        st = fmt_ts(float(seg["start"]))
        et = fmt_ts(float(seg["end"]))
        text = (seg.get("text") or "").strip()
        if text:
            txt_lines.append(f"[{st} - {et}] {role}: {text}")

    txt_path = outdir / f"{audio_path.stem}.host_guest_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))

    print(f"[done] JSON: {json_path}")
    print(f"[done] TXT : {txt_path}")


if __name__ == "__main__":
    main()

