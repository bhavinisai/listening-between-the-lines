#!/usr/bin/env python3
"""
Identify host vs guest in podcast transcripts.

Priority Order:
1. Question count (>60% AND at least 5 more questions than other speaker)
2. Host-specific phrases (welcome, thanks for joining, hope you enjoyed)
3. YouTube CTAs (strict phrases only, not loose words like "subscribe")
4. Sponsor/ad segments
5. Fallback heuristics (speaks first and last, speaks first)

Usage:
  # Interview-style (interviewer = host)
  python identify_host_guest_v5.py \
    --json outputs/episode.whisperx.json \
    --interview-style

  # Conversational (default)
  python identify_host_guest_v5.py \
    --json outputs/episode.whisperx.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# ============================================================
# KEYWORD LISTS
# ============================================================

# Host-specific phrases (HIGH PRIORITY - very reliable)
HOST_PHRASES = [
    # Welcoming
    "welcome to", "welcome back", "welcome back to",
    "thanks for joining", "glad you could join", "great to have you",
    "thank you for being here", "thank you so much for being here",
    "thanks for coming on", "thanks for being on the show",

    # Opening
    "today we have", "today i'm joined by", "today i'm talking to",
    "my guest today", "i'm here with", "joining me today",

    # Closing
    "thanks for listening", "thank you for listening",
    "hope you enjoyed", "hope you had fun", "hope you learned",
    "that's all for today", "that's it for today",
    "see you next time", "until next time", "catch you next time",
    "don't forget to subscribe", "make sure to subscribe",

    # Transition/Control
    "let's dive in", "let's get into it", "let's talk about",
    "i want to ask you", "i wanted to ask", "can i ask you",
]

# YouTube CTAs - STRICT phrases only (avoids false positives like "subscribed to")
YOUTUBE_HOST_KEYWORDS = [
    "hit the subscribe button", "subscribe to the channel",
    "subscribe to my channel", "don't forget to subscribe",
    "like and subscribe", "subscribe below", "click subscribe",
    "smash that subscribe", "hit the bell", "notification bell",
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
    "percent off", "% off", "discount",
    "audible", "squarespace", "hellofresh", "betterhelp",
    "nordvpn", "athletic greens",
    "before we get started", "before we dive in",
    "quick word from our sponsor", "message from our sponsor",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_transcript(json_path):
    """Load WhisperX JSON output."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def contains_keywords(text, keywords):
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


# ============================================================
# DETECTION METHODS
# ============================================================

def count_questions_by_speaker(segments):
    """
    Count how many questions each speaker asks.
    Returns dict: {speaker: question_count}
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

        # Check if ends with question mark
        is_question = text.endswith('?')

        # Or starts with question word
        if not is_question:
            for starter in question_starters:
                if text_lower.startswith(starter):
                    is_question = True
                    break

        if is_question:
            speaker_questions[speaker] += 1

    return speaker_questions


def detect_host_segments(segments):
    """
    Detect segments that indicate who the host is.
    Returns (indices, detection_type)
    Priority: Host Phrases > YouTube CTAs > Sponsor
    """
    host_phrase_segs = []
    youtube_segs = []
    sponsor_segs = []

    for i, seg in enumerate(segments):
        text = seg.get('text', '')

        if contains_keywords(text, HOST_PHRASES):
            host_phrase_segs.append(i)
        elif contains_keywords(text, YOUTUBE_HOST_KEYWORDS):
            youtube_segs.append(i)
        elif contains_keywords(text, SPONSOR_KEYWORDS):
            sponsor_segs.append(i)

    if host_phrase_segs:
        return host_phrase_segs, "host_phrases"
    elif youtube_segs:
        return youtube_segs, "youtube_cta"
    elif sponsor_segs:
        return sponsor_segs, "sponsor"

    return [], "none"


def identify_host(segments, interview_style=False, force_host=None):
    """
    Identify who is the host.
    Returns (host_label, guest_label, confidence, method, detection_type)
    """
    # Manual override
    if force_host:
        host_label = force_host
        guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
        return host_label, guest_label, 1.0, "manual_override", "manual"

    # Priority 1: Question detection
    # Requires >60% ratio AND at least 5 more questions than the other speaker
    # This avoids false positives when both speakers ask similar numbers of questions
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

            if ratio > 0.6 and diff >= 5:
                if interview_style:
                    host_label = question_asker
                else:
                    host_label = "Speaker 1" if question_asker == "Speaker 2" else "Speaker 2"

                guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
                method = f"questions ({max_questions} vs {other_count}, {ratio:.1%})"
                return host_label, guest_label, ratio, method, "questions"

    # Priority 2-4: Keyword detection (Host Phrases, YouTube CTAs, Sponsors)
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

            if detection_type == "host_phrases":
                confidence = min(0.95, confidence + 0.25)
                method = f"host_phrases ({len(host_indices)} phrases)"
            elif detection_type == "youtube_cta":
                confidence = min(0.95, confidence + 0.2)
                method = f"youtube_cta ({len(host_indices)} mentions)"
            else:
                method = f"sponsor ({len(host_indices)} segments)"

            return host_label, guest_label, confidence, method, detection_type

    # Priority 5: Fallback (speaks first and last)
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
    """Relabel speakers as 'Host' and 'Guest'."""
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
    ap.add_argument("--interview-style", action="store_true",
                    help="Interview podcast: person asking questions is HOST")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    print(f"Loading transcript: {json_path}")
    segments = load_transcript(json_path)
    print(f"  Total segments: {len(segments)}")

    # Show question statistics
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
        interview_style=args.interview_style,
        force_host=args.force_host
    )

    if args.show_segments and detection_type in ["host_phrases", "youtube_cta", "sponsor"]:
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
    if args.interview_style:
        print(f"  Mode:           Interview-style (question asker = host)")

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
        'interview_style': args.interview_style
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
