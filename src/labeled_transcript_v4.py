#!/usr/bin/env python3
"""
Identify host vs guest in podcast transcripts.

Priority Order:
1. YouTube CTAs (subscribe to this channel, I'll see you next time, etc.)
2. Host-specific phrases (welcome to, thanks for joining, etc.)
3. Sponsor/ad segments
4. Question count (question asker = host, no flip logic)
5. Fallback (speaks first and last, speaks first)

Usage:
  python identify_host_guest_v5.py \
    --json outputs/episode.whisperx.json

  # Force manual override if auto-detection is wrong
  python identify_host_guest_v5.py \
    --json outputs/episode.whisperx.json \
    --force-host "Speaker 1"
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# ============================================================
# KEYWORD LISTS
# ============================================================

# Host-specific phrases (very reliable)
HOST_PHRASES = [
    # Welcoming the guest
    "welcome to", "welcome back", "welcome back to",
    "thanks for joining", "glad you could join", "great to have you",
    "thank you for being here", "thank you so much for being here",
    "thanks for coming on", "thanks for being on the show",

    # Opening the episode
    "today we have", "today i'm joined by", "today i'm talking to",
    "my guest today", "joining me today",

    # Closing the show
    "thanks for listening", "thank you for listening",
    "that's all for today", "that's it for today",
    "see you next time", "until next time", "catch you next time",
    "i'll see you next time",
    "don't forget to subscribe", "make sure to subscribe",
]

# YouTube CTAs - strict phrases only (avoids false positives)
YOUTUBE_HOST_KEYWORDS = [
    "subscribe to this channel", "subscribe to the channel",
    "subscribe to my channel", "hit the subscribe button",
    "like and subscribe", "subscribe below", "click subscribe",
    "smash that subscribe", "don't forget to subscribe",
    "hit the bell", "notification bell",
    "leave a like", "hit the like button",
    "check out the description", "link in the description",
    "support the channel", "support the show",
    "patreon",
]

# Sponsor/ad keywords
SPONSOR_KEYWORDS = [
    "sponsor", "brought to you by", "today's sponsor",
    "this episode is sponsored", "this podcast is brought",
    "promo code", "discount code", "coupon code",
    "percent off", "% off",
    "audible", "squarespace", "hellofresh", "betterhelp",
    "nordvpn", "athletic greens",
    "quick word from our sponsor", "message from our sponsor",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_transcript(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def contains_keywords(text, keywords):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


# ============================================================
# DETECTION METHODS
# ============================================================

def detect_host_segments(segments):
    """
    Scan all segments for host-identifying keywords.
    Priority: YouTube CTAs > Host Phrases > Sponsors
    Returns (matching_indices, detection_type)
    """
    youtube_segs = []
    host_phrase_segs = []
    sponsor_segs = []

    for i, seg in enumerate(segments):
        text = seg.get('text', '')

        if contains_keywords(text, YOUTUBE_HOST_KEYWORDS):
            youtube_segs.append(i)
        elif contains_keywords(text, HOST_PHRASES):
            host_phrase_segs.append(i)
        elif contains_keywords(text, SPONSOR_KEYWORDS):
            sponsor_segs.append(i)

    if youtube_segs:
        return youtube_segs, "youtube_cta"
    elif host_phrase_segs:
        return host_phrase_segs, "host_phrases"
    elif sponsor_segs:
        return sponsor_segs, "sponsor"

    return [], "none"


def count_questions_by_speaker(segments):
    """
    Count how many questions each speaker asks.
    A segment is a question if it:
    - Ends with "?"
    - OR starts with a question word
    Returns dict: {speaker: count}
    """
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


def identify_host(segments, force_host=None):
    """
    Identify who is the host.
    Returns (host_label, guest_label, confidence, method, detection_type)
    """
    # Manual override
    if force_host:
        host_label = force_host
        guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
        return host_label, guest_label, 1.0, "manual_override", "manual"

    # Priority 1-3: Keyword detection (YouTube CTAs, Host Phrases, Sponsors)
    # Most reliable signals - only hosts say these things
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
            elif detection_type == "host_phrases":
                confidence = min(0.95, confidence + 0.25)
                method = f"host_phrases ({len(host_indices)} phrases)"
            else:
                method = f"sponsor ({len(host_indices)} segments)"

            return host_label, guest_label, confidence, method, detection_type

    # Priority 4: Question count (only when no keywords found)
    # Question asker = host, no flip logic
    # Use --force-host to override if wrong
    speaker_questions = count_questions_by_speaker(segments)

    if speaker_questions:
        max_questions = max(speaker_questions.values())
        total_questions = sum(speaker_questions.values())

        if total_questions >= 5:
            question_asker = max(speaker_questions, key=speaker_questions.get)
            speakers = list(speaker_questions.keys())
            other = [s for s in speakers if s != question_asker]
            other_count = speaker_questions[other[0]] if other else 0
            ratio = max_questions / total_questions
            diff = max_questions - other_count

            if ratio > 0.55 and diff >= 5:
                host_label = question_asker
                guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
                method = f"questions ({max_questions} vs {other_count}, {ratio:.1%})"
                return host_label, guest_label, ratio, method, "questions"

    # Priority 5: Fallback
    if not segments:
        return "Speaker 1", "Speaker 2", 0.5, "default", "fallback"

    first_speaker = segments[0].get('speaker', 'Unknown')
    last_speaker = segments[-1].get('speaker', 'Unknown')

    if first_speaker == last_speaker and first_speaker != 'Unknown':
        guest_label = "Speaker 1" if first_speaker == "Speaker 2" else "Speaker 2"
        return first_speaker, guest_label, 0.7, "speaks_first_and_last", "fallback"

    guest_label = "Speaker 1" if first_speaker == "Speaker 2" else "Speaker 2"
    return first_speaker, guest_label, 0.5, "speaks_first", "fallback"


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

def format_timestamp(t):
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def save_labeled_transcript(segments, output_path):
    lines = []
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg.get('text', '').strip()
        if text:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def save_labeled_json(segments, output_path, metadata):
    output = {'segments': segments, 'host_identification': metadata}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Identify host vs guest in podcast")
    ap.add_argument("--json", required=True, help="WhisperX JSON output")
    ap.add_argument("--output_txt", help="Output labeled transcript (optional)")
    ap.add_argument("--output_json", help="Output labeled JSON (optional)")
    ap.add_argument("--show_segments", action="store_true",
                    help="Print detected host-identifying segments")
    ap.add_argument("--force-host", choices=["Speaker 1", "Speaker 2"],
                    help="Manually specify which speaker is the host")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    print(f"Loading transcript: {json_path}")
    segments = load_transcript(json_path)
    print(f"  Total segments: {len(segments)}")

    # Show question counts for info
    speaker_questions = count_questions_by_speaker(segments)
    if speaker_questions:
        print(f"\nQuestion counts:")
        for speaker, count in sorted(speaker_questions.items()):
            print(f"  {speaker}: {count} questions")

    print("\n" + "="*60)
    print("HOST/GUEST IDENTIFICATION")
    print("="*60)

    host_label, guest_label, confidence, method, detection_type = identify_host(
        segments,
        force_host=args.force_host
    )

    # Show detected segments if requested
    if args.show_segments and detection_type in ["youtube_cta", "host_phrases", "sponsor"]:
        host_indices, _ = detect_host_segments(segments)
        if host_indices:
            print(f"\nDetected {detection_type.upper()} segments:")
            for idx in host_indices[:5]:
                seg = segments[idx]
                speaker = seg.get('speaker', 'Unknown')
                text = seg.get('text', '')[:100]
                print(f"  [{idx}] {speaker}: {text}...")
            if len(host_indices) > 5:
                print(f"  ... and {len(host_indices) - 5} more")

    print(f"\nIdentification result:")
    print(f"  Host:           {host_label} (confidence: {confidence:.1%})")
    print(f"  Guest:          {guest_label}")
    print(f"  Method:         {method}")
    print(f"  Detection type: {detection_type}")

    print("\nRelabeling transcript...")
    relabeled_segments = relabel_transcript(segments, host_label)

    host_count = sum(1 for s in relabeled_segments if s['speaker'] == 'Host')
    guest_count = sum(1 for s in relabeled_segments if s['speaker'] == 'Guest')
    print(f"  Host segments:  {host_count}")
    print(f"  Guest segments: {guest_count}")

    metadata = {
        'host_original_label': host_label,
        'guest_original_label': guest_label,
        'confidence': confidence,
        'method': method,
        'detection_type': detection_type,
    }

    if args.output_txt:
        save_labeled_transcript(relabeled_segments, Path(args.output_txt))
        print(f"\nSaved: {args.output_txt}")

    if args.output_json:
        save_labeled_json(relabeled_segments, Path(args.output_json), metadata)
        print(f"Saved: {args.output_json}")

    if not args.output_txt and not args.output_json:
        auto_txt = json_path.parent / f"{json_path.stem}.labeled.txt"
        auto_json = json_path.parent / f"{json_path.stem}.labeled.json"
        save_labeled_transcript(relabeled_segments, auto_txt)
        save_labeled_json(relabeled_segments, auto_json, metadata)
        print(f"\nSaved outputs:")
        print(f"  {auto_txt}")
        print(f"  {auto_json}")


if __name__ == "__main__":
    main()
