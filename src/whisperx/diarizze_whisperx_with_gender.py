#!/usr/bin/env python3
"""
Unified Podcast Processing Pipeline

Combines:
1. WhisperX ASR + Alignment + Pyannote Diarization -> Speaker 1 / Speaker 2
2. Host/Guest Identification via keyword detection (sponsor, YouTube CTAs)
3. Gender Detection via pitch analysis (librosa)

Requirements:
  pip install whisperx pyannote.audio torch torchaudio soundfile librosa
  export HF_TOKEN="hf_..."

Run:
python src/diarizze_whisperx_with_gender.py \
  --audio data/raw_audio/Sunita_Williams_clip_60s.wav \
  --outdir outputs/whisperx_output \
  --model medium \
  --compute_type int8 \
  --language en

Outputs:
  - outputs/<stem>.whisperx.json               (raw diarization)
  - outputs/<stem>.speaker_transcript.txt      (Speaker 1/2 labels)
  - outputs/<stem>.labeled.json                (Host/Guest labels)
  - outputs/<stem>.labeled.txt                 (Host/Guest readable transcript)
  - outputs/<stem>.final.txt                   (Host/Guest + Gender labels)
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
import time
import numpy as np


# ============================================================
# KEYWORD LISTS FOR HOST DETECTION
# ============================================================

SPONSOR_KEYWORDS = [
    "sponsor", "brought to you by", "today's sponsor",
    "this episode is sponsored", "this podcast is brought",
    "promo code", "discount code", "coupon code",
    "percent off", "% off", "discount",
    "audible", "squarespace", "hellofresh", "betterhelp",
    "nordvpn", "athletic greens",
    "before we get started", "before we dive in",
    "quick word from our sponsor", "message from our sponsor",
    "thanks to our sponsor",
]

YOUTUBE_HOST_KEYWORDS = [
    "subscribe", "hit the subscribe button", "subscribe to the channel",
    "subscribe to my channel", "don't forget to subscribe",
    "like and subscribe", "subscribe below", "click subscribe",
    "hit the bell", "notification bell", "smash that subscribe",
    "leave a like", "hit the like button", "thumbs up",
    "check out the description", "link in the description",
    "patreon", "support the channel", "support the show",
]

INTRO_OUTRO_KEYWORDS = [
    "welcome to", "welcome back", "thanks for listening",
    "that's all for today", "see you next time",
    "don't forget to subscribe", "leave a review",
    "if you enjoyed this episode",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def fmt_ts(t: float) -> str:
    """Format float seconds as HH:MM:SS.mmm"""
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
    """Ensure ffmpeg is discoverable."""
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


def contains_keywords(text, keywords):
    """Check if text contains any keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def overlap(a0, a1, b0, b1):
    """Calculate overlap between two time intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))


# ============================================================
# STEP 1: DIARIZATION
# ============================================================

def run_diarization(audio_path, args, hf_token):
    """
    Run WhisperX ASR + Alignment + Pyannote Diarization.
    Returns result dict with segments labeled Speaker 1 / Speaker 2.
    """
    import torch
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
    return result


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


# ============================================================
# STEP 2: HOST/GUEST IDENTIFICATION
# ============================================================

def detect_host_segments(segments):
    """
    Detect segments that indicate who the host is.
    Priority: YouTube CTAs > Sponsor > Intro/Outro
    Returns (indices, detection_type)
    """
    youtube_segs, sponsor_segs, intro_segs = [], [], []

    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        if contains_keywords(text, YOUTUBE_HOST_KEYWORDS):
            youtube_segs.append(i)
        elif contains_keywords(text, SPONSOR_KEYWORDS):
            sponsor_segs.append(i)
        elif contains_keywords(text, INTRO_OUTRO_KEYWORDS):
            intro_segs.append(i)

    if youtube_segs:
        return youtube_segs, "youtube_cta"
    elif sponsor_segs:
        return sponsor_segs, "sponsor"
    elif intro_segs:
        return intro_segs, "intro_outro"
    return [], "none"


def identify_host(segments, host_indices, detection_type):
    """Identify host speaker label based on detected segments."""
    if not host_indices:
        return identify_host_fallback(segments)

    speaker_counts = defaultdict(int)
    for idx in host_indices:
        speaker = segments[idx].get('speaker', 'Unknown')
        if speaker != 'Unknown':
            speaker_counts[speaker] += 1

    if not speaker_counts:
        return identify_host_fallback(segments)

    host = max(speaker_counts, key=speaker_counts.get)
    total = sum(speaker_counts.values())
    confidence = speaker_counts[host] / total if total > 0 else 0.5

    if detection_type == "youtube_cta":
        confidence = min(0.95, confidence + 0.2)
        method = f"youtube_cta ({len(host_indices)} subscribe/like mentions)"
    elif detection_type == "sponsor":
        method = f"sponsor_detection ({len(host_indices)} ad segments)"
    else:
        method = f"intro_outro ({len(host_indices)} segments)"

    return host, confidence, method


def identify_host_fallback(segments):
    """Fallback: identify host by speaking time and position."""
    if not segments:
        return "Speaker 1", 0.5, "default"

    first_speaker = segments[0].get('speaker', 'Unknown')
    last_speaker = segments[-1].get('speaker', 'Unknown')

    speaking_time = defaultdict(float)
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        if speaker != 'Unknown':
            speaking_time[speaker] += seg.get('end', 0) - seg.get('start', 0)

    if first_speaker == last_speaker and first_speaker != 'Unknown':
        return first_speaker, 0.7, "speaks_first_and_last"

    total_time = sum(speaking_time.values())
    if total_time > 0:
        dominant = max(speaking_time, key=speaking_time.get)
        ratio = speaking_time[dominant] / total_time
        if ratio > 0.6:
            return dominant, ratio, f"dominant_speaker ({ratio:.1%})"

    return first_speaker, 0.5, "speaks_first (fallback)"


def relabel_host_guest(segments, host_label):
    """Relabel Speaker 1/2 as Host/Guest."""
    guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
    relabeled = []
    for seg in segments:
        new_seg = seg.copy()
        speaker = seg.get('speaker', 'Unknown')
        if speaker == host_label:
            new_seg['speaker'] = 'Host'
            new_seg['speaker_original'] = speaker
        elif speaker == guest_label:
            new_seg['speaker'] = 'Guest'
            new_seg['speaker_original'] = speaker
        else:
            new_seg['speaker'] = 'Unknown'
        relabeled.append(new_seg)
    return relabeled


# ============================================================
# STEP 3: GENDER DETECTION
# ============================================================

def extract_pitch_for_segment(audio, sr, start_time, end_time):
    """Extract median pitch for a time segment using librosa."""
    import librosa

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]

    if len(segment) < 512:
        return None

    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    return np.median(pitch_values) if pitch_values else None


def classify_gender_by_pitch(pitch):
    """Classify gender based on pitch frequency."""
    if pitch is None:
        return 'unknown'
    if pitch < 165:
        return 'male'
    elif pitch > 180:
        return 'female'
    return 'ambiguous'


def detect_gender(audio_path, segments, sample_size=10):
    """Analyze gender for each speaker using pitch analysis."""
    import librosa

    print(f"[info] Loading audio for gender detection...")
    audio, sr = librosa.load(str(audio_path), sr=None)

    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        if speaker != 'Unknown':
            speaker_segments[speaker].append(seg)

    speaker_results = {}

    for speaker, segs in speaker_segments.items():
        print(f"[info] Analyzing gender for {speaker}...")

        if len(segs) <= sample_size:
            sample_segs = segs
        else:
            indices = np.linspace(0, len(segs)-1, sample_size, dtype=int)
            sample_segs = [segs[i] for i in indices]

        pitches = []
        for seg in sample_segs:
            pitch = extract_pitch_for_segment(audio, sr, seg['start'], seg['end'])
            if pitch is not None:
                pitches.append(pitch)

        if not pitches:
            speaker_results[speaker] = {'gender': 'unknown', 'avg_pitch': None, 'confidence': 0.0}
            continue

        avg_pitch = float(np.mean(pitches))
        std_pitch = float(np.std(pitches))
        gender = classify_gender_by_pitch(avg_pitch)

        if gender in ['male', 'female']:
            consistency = 1.0 - min(std_pitch / avg_pitch, 1.0)
            if gender == 'male':
                distance = max(0, 165 - avg_pitch) / 80
            else:
                distance = max(0, avg_pitch - 180) / 75
            confidence = float((consistency + distance) / 2)
        else:
            confidence = 0.3

        speaker_results[speaker] = {
            'gender': gender,
            'avg_pitch': avg_pitch,
            'std_pitch': std_pitch,
            'confidence': confidence,
            'sample_count': len(pitches)
        }

        print(f"  {speaker}: {gender.upper()} (avg pitch: {avg_pitch:.1f} Hz, confidence: {confidence:.1%})")

    return speaker_results


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_transcript(segments, output_path, include_gender=False):
    """Save readable transcript to text file."""
    lines = []
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        gender = seg.get('gender', '')
        start = fmt_ts(float(seg['start']))
        end = fmt_ts(float(seg['end']))
        text = (seg.get('text') or '').strip()

        if text:
            if include_gender and gender and gender != 'unknown':
                lines.append(f"[{start} - {end}] {speaker} ({gender}): {text}")
            else:
                lines.append(f"[{start} - {end}] {speaker}: {text}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def save_json(data, output_path):
    """Save JSON output."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Unified Podcast Processing Pipeline")

    # Audio
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")

    # Diarization
    ap.add_argument("--model", default="medium", help="Whisper model (small/medium/large-v3)")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not set)")
    ap.add_argument("--compute_type", default=None, help="float16 / int8")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language code")
    ap.add_argument("--min_overlap", type=float, default=0.10)

    # Host detection
    ap.add_argument("--show_host_segments", action="store_true",
                    help="Print detected host-identifying segments")
    ap.add_argument("--skip_host_detection", action="store_true",
                    help="Skip host/guest identification")

    # Gender detection
    ap.add_argument("--skip_gender", action="store_true",
                    help="Skip gender detection")
    ap.add_argument("--gender_sample_size", type=int, default=10,
                    help="Number of segments to sample for gender detection")

    args = ap.parse_args()

    # Setup
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

    # Patch torch BEFORE importing whisperx
    import torch
    patch_torch_load()

    total_start = time.time()

    # -------------------------------------------------------
    # STEP 1: DIARIZATION
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: DIARIZATION")
    print("="*60)

    result = run_diarization(audio_path, args, hf_token)
    segments = result["segments"]

    print(f"[done] Diarization complete: {len(segments)} segments")

    # -------------------------------------------------------
    # STEP 2: HOST/GUEST IDENTIFICATION
    # -------------------------------------------------------
    if not args.skip_host_detection:
        print("\n" + "="*60)
        print("STEP 2: HOST/GUEST IDENTIFICATION")
        print("="*60)

        host_indices, detection_type = detect_host_segments(segments)

        if detection_type == "youtube_cta":
            print(f"[info] Found {len(host_indices)} YouTube CTA segments")
        elif detection_type == "sponsor":
            print(f"[info] Found {len(host_indices)} sponsor segments")
        elif detection_type == "intro_outro":
            print(f"[info] Found {len(host_indices)} intro/outro segments")
        else:
            print("[info] No host-identifying keywords found, using fallback")

        if args.show_host_segments and host_indices:
            print(f"\nDetected {detection_type.upper()} segments:")
            for idx in host_indices[:5]:
                seg = segments[idx]
                speaker = seg.get('speaker', 'Unknown')
                text = seg.get('text', '')[:100]
                print(f"  [{idx}] {speaker}: {text}...")

        host_label, confidence, method = identify_host(segments, host_indices, detection_type)
        guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"

        print(f"[info] Host: {host_label} (confidence: {confidence:.1%}, method: {method})")
        print(f"[info] Guest: {guest_label}")

        segments = relabel_host_guest(segments, host_label)

        host_count = sum(1 for s in segments if s['speaker'] == 'Host')
        guest_count = sum(1 for s in segments if s['speaker'] == 'Guest')
        print(f"[info] Host segments: {host_count}, Guest segments: {guest_count}")

        # Store metadata for final output
        host_metadata = {
            'host_original_label': host_label,
            'guest_original_label': guest_label,
            'confidence': confidence,
            'method': method,
            'detection_type': detection_type
        }

    # -------------------------------------------------------
    # STEP 3: GENDER DETECTION
    # -------------------------------------------------------
    if not args.skip_gender:
        print("\n" + "="*60)
        print("STEP 3: GENDER DETECTION")
        print("="*60)

        try:
            speaker_gender = detect_gender(audio_path, segments, args.gender_sample_size)

            # Add gender to segments
            for seg in segments:
                speaker = seg.get('speaker', 'Unknown')
                if speaker in speaker_gender:
                    seg['gender'] = speaker_gender[speaker]['gender']
                    seg['gender_confidence'] = speaker_gender[speaker]['confidence']

            # Print results
            print("\nGender Detection Results:")
            print("="*40)
            for speaker, result in speaker_gender.items():
                gender = result['gender']
                pitch = result.get('avg_pitch')
                conf = result['confidence']
                print(f"  {speaker}: {gender.upper()}")
                if pitch:
                    print(f"    Avg pitch: {pitch:.1f} Hz")
                print(f"    Confidence: {conf:.1%}")

        except ImportError:
            print("[warning] librosa not installed, skipping gender detection")
            print("  Install with: pip install librosa soundfile")
            speaker_gender = {}

    # -------------------------------------------------------
    # SAVE FINAL OUTPUTS ONLY
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING FINAL OUTPUTS")
    print("="*60)

    final_txt_path = outdir / f"{audio_path.stem}.final.txt"
    final_json_path = outdir / f"{audio_path.stem}.final.json"

    # Build final JSON with all metadata
    final_data = {'segments': segments}

    if not args.skip_host_detection:
        final_data['host_identification'] = host_metadata

    if not args.skip_gender and speaker_gender:
        final_data['speaker_gender_analysis'] = speaker_gender

    save_transcript(segments, final_txt_path, include_gender=not args.skip_gender)
    save_json(final_data, final_json_path)

    print(f"[done] Saved: {final_txt_path}")
    print(f"[done] Saved: {final_json_path}")

    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nOutput files in: {outdir}")
    print(f"  {audio_path.stem}.final.txt   (final transcript)")
    print(f"  {audio_path.stem}.final.json  (full data with metadata)")


if __name__ == "__main__":
    main()

