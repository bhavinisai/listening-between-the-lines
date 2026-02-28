#!/usr/bin/env python3
"""
Combined Pipeline: Diarization + Host/Guest Identification

Does everything in one script:
1. WhisperX transcription
2. Alignment
3. Pyannote diarization → Speaker 1 / Speaker 2
4. Host/Guest identification (question detection, YouTube CTAs, sponsors)
5. Output final transcript with Host/Guest labels

Requirements:
  pip install whisperx pyannote.audio torch torchaudio soundfile
  export HF_TOKEN="hf_..."

Usage:
  # Interview-style podcast
  python src/diarize_and_label.py \
    --audio data/raw_audio/Episode18-clip.wav \
    --outdir outputs \
    --model medium \
    --interview-style
    
  # Conversational podcast
  python podcast_diarize_and_label.py \
    --audio data/raw_audio/episode.wav \
    --outdir outputs \
    --model medium
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import shutil
import time


# ============================================================
# KEYWORD LISTS FOR HOST DETECTION
# ============================================================

YOUTUBE_HOST_KEYWORDS = [
    "subscribe", "hit the subscribe button", "subscribe to the channel",
    "subscribe to my channel", "don't forget to subscribe",
    "like and subscribe", "subscribe below", "click subscribe",
    "hit the bell", "notification bell", "smash that subscribe",
    "leave a like", "hit the like button", "thumbs up",
    "check out the description", "link in the description",
    "patreon", "support the channel", "support the show",
]

SPONSOR_KEYWORDS = [
    "sponsor", "brought to you by", "today's sponsor",
    "this episode is sponsored", "this podcast is brought",
    "promo code", "discount code", "coupon code",
    "percent off", "% off", "discount",
    "audible", "squarespace", "hellofresh", "betterhelp",
    "nordvpn", "athletic greens",
    "before we get started", "before we dive in",
    "quick word from our sponsor", "message from our sponsor",
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


def overlap(a0, a1, b0, b1):
    """Calculate overlap between two time intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def contains_keywords(text, keywords):
    """Check if text contains any keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ============================================================
# STEP 1: DIARIZATION
# ============================================================

def merge_short_turns(turns, min_turn_duration=0.3):
    """Merge very short turns into the previous turn."""
    if not turns:
        return turns

    merged = [turns[0].copy()]
    for turn in turns[1:]:
        duration = turn['end'] - turn['start']
        prev = merged[-1]
        if duration < min_turn_duration:
            prev['end'] = turn['end']
        else:
            merged.append(turn.copy())

    print(f"[info] merge_short_turns: {len(turns)} -> {len(merged)} turns")
    return merged


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


def run_diarization(audio_path, args, hf_token):
    """
    Run WhisperX ASR + Alignment + Pyannote Diarization.
    Returns segments with Speaker 1 / Speaker 2 labels.
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

    # Post-process turns
    turns = merge_short_turns(turns, min_turn_duration=args.min_turn_duration)

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
    return result["segments"]


# ============================================================
# STEP 2: HOST/GUEST IDENTIFICATION
# ============================================================

def count_questions_by_speaker(segments):
    """Count how many questions each speaker asks."""
    question_starters = [
        "what", "why", "how", "when", "where", "who", "which",
        "can you", "could you", "would you", "will you", "should",
        "do you", "did you", "does", "have you", "has",
        "is it", "are you", "was", "were",
        "tell me", "explain",
    ]
    
    speaker_questions = defaultdict(int)
    
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '').strip()
        
        if not text or speaker == 'Unknown':
            continue
        
        text_lower = text.lower()
        
        is_question = text.endswith('?')
        
        if not is_question:
            for starter in question_starters:
                if text_lower.startswith(starter):
                    is_question = True
                    break
        
        if is_question:
            speaker_questions[speaker] += 1
    
    return speaker_questions


def detect_host_segments(segments):
    """Detect YouTube CTA and sponsor segments."""
    youtube_segs = []
    sponsor_segs = []
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        
        if contains_keywords(text, YOUTUBE_HOST_KEYWORDS):
            youtube_segs.append(i)
        elif contains_keywords(text, SPONSOR_KEYWORDS):
            sponsor_segs.append(i)
    
    if youtube_segs:
        return youtube_segs, "youtube_cta"
    elif sponsor_segs:
        return sponsor_segs, "sponsor"
    
    return [], "none"


def identify_host(segments, interview_style=False, force_host=None):
    """
    Identify who is the host.
    Returns (host_label, guest_label, confidence, method)
    """
    # Manual override
    if force_host:
        host_label = force_host
        guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
        return host_label, guest_label, 1.0, "manual_override"
    
    # Priority 1: Question detection
    speaker_questions = count_questions_by_speaker(segments)
    
    if speaker_questions:
        max_questions = max(speaker_questions.values())
        total_questions = sum(speaker_questions.values())
        
        if total_questions >= 5:
            question_asker = max(speaker_questions, key=speaker_questions.get)
            ratio = max_questions / total_questions
            
            if ratio > 0.6:
                if interview_style:
                    # Interview: question asker = host
                    host_label = question_asker
                else:
                    # Conversational: question asker = guest
                    host_label = "Speaker 1" if question_asker == "Speaker 2" else "Speaker 2"
                
                guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
                method = f"questions ({speaker_questions[question_asker]} asked, {ratio:.1%})"
                return host_label, guest_label, ratio, method
    
    # Priority 2: YouTube CTAs / Sponsors
    host_indices, detection_type = detect_host_segments(segments)
    
    if host_indices:
        speaker_counts = defaultdict(int)
        for idx in host_indices:
            speaker = segments[idx].get('speaker', 'Unknown')
            if speaker != 'Unknown':
                speaker_counts[speaker] += 1
        
        if speaker_counts:
            host_label = max(speaker_counts, key=speaker_counts.get)
            guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
            total = sum(speaker_counts.values())
            confidence = speaker_counts[host_label] / total
            
            if detection_type == "youtube_cta":
                confidence = min(0.95, confidence + 0.2)
                method = f"youtube_cta ({len(host_indices)} mentions)"
            else:
                method = f"sponsor ({len(host_indices)} segments)"
            
            return host_label, guest_label, confidence, method
    
    # Priority 3: Fallback (speaks first and last)
    if not segments:
        return "Speaker 1", "Speaker 2", 0.5, "default"
    
    first_speaker = segments[0].get('speaker', 'Unknown')
    last_speaker = segments[-1].get('speaker', 'Unknown')
    
    if first_speaker == last_speaker and first_speaker != 'Unknown':
        guest_label = "Speaker 1" if first_speaker == "Speaker 2" else "Speaker 2"
        return first_speaker, guest_label, 0.7, "speaks_first_and_last"
    
    # Last resort: first speaker
    guest_label = "Speaker 1" if first_speaker == "Speaker 2" else "Speaker 2"
    return first_speaker, guest_label, 0.5, "speaks_first"


def relabel_transcript(segments, host_label):
    """Relabel speakers as Host/Guest."""
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
# SAVE OUTPUTS
# ============================================================

def save_transcript(segments, output_path):
    """Save readable transcript."""
    lines = []
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        start = fmt_ts(float(seg['start']))
        end = fmt_ts(float(seg['end']))
        text = (seg.get('text') or '').strip()
        
        if text:
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
    ap = argparse.ArgumentParser(description="Combined Diarization + Host/Guest Identification")
    
    # Audio input
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    
    # Diarization params
    ap.add_argument("--model", default="medium", help="Whisper model (small/medium/large-v3)")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if not set)")
    ap.add_argument("--compute_type", default=None, help="float16 / int8")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--language", default="en", help="Language code")
    ap.add_argument("--min_overlap", type=float, default=0.10)
    ap.add_argument("--min_turn_duration", type=float, default=0.3,
                    help="Turns shorter than this are merged (default: 0.3)")
    
    # Host/Guest identification
    ap.add_argument("--interview-style", action="store_true",
                    help="Interview podcast: question asker = host")
    ap.add_argument("--force-host", choices=["Speaker 1", "Speaker 2"],
                    help="Manually specify host")
    
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
    
    import torch
    patch_torch_load()
    
    total_start = time.time()
    
    # -------------------------------------------------------
    # STEP 1: DIARIZATION
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: DIARIZATION")
    print("="*60)
    
    segments = run_diarization(audio_path, args, hf_token)
    
    # -------------------------------------------------------
    # STEP 2: HOST/GUEST IDENTIFICATION
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: HOST/GUEST IDENTIFICATION")
    print("="*60)
    
    # Show question counts
    speaker_questions = count_questions_by_speaker(segments)
    if speaker_questions:
        print(f"\nQuestion counts:")
        for speaker, count in sorted(speaker_questions.items()):
            print(f"  {speaker}: {count} questions")
    
    host_label, guest_label, confidence, method = identify_host(
        segments, 
        interview_style=args.interview_style,
        force_host=args.force_host
    )
    
    print(f"\nIdentification result:")
    print(f"  Host: {host_label} (confidence: {confidence:.1%})")
    print(f"  Guest: {guest_label}")
    print(f"  Method: {method}")
    
    if args.interview_style:
        print(f"  Mode: Interview-style (question asker = host)")
    
    # Relabel segments
    segments = relabel_transcript(segments, host_label)
    
    host_count = sum(1 for s in segments if s['speaker'] == 'Host')
    guest_count = sum(1 for s in segments if s['speaker'] == 'Guest')
    print(f"  Host segments: {host_count}")
    print(f"  Guest segments: {guest_count}")
    
    # -------------------------------------------------------
    # SAVE FINAL OUTPUTS
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    final_txt = outdir / f"{audio_path.stem}.final.txt"
    final_json = outdir / f"{audio_path.stem}.final.json"
    
    save_transcript(segments, final_txt)
    save_json({
        'segments': segments,
        'host_identification': {
            'host_label': host_label,
            'guest_label': guest_label,
            'confidence': confidence,
            'method': method,
            'interview_style': args.interview_style
        }
    }, final_json)
    
    print(f"[done] {final_txt}")
    print(f"[done] {final_json}")
    
    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Output: {final_txt}")


if __name__ == "__main__":
    main()
