#!/usr/bin/env python3
"""
Identify host vs guest in podcast transcripts.

Strategy:
1. Detect sponsor/ad segments using keyword matching
2. Identify which speaker appears in sponsor segments
3. Label that speaker as "Host", the other as "Guest"
4. Re-label the transcript

Usage:
  python identify_host_guest.py \
    --transcript outputs/episode_01.speaker_transcript.txt \
    --json outputs/episode_01.whisperx.json \
    --output outputs/episode_01.labeled.txt
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


# Keywords that indicate sponsor/ad segments
SPONSOR_KEYWORDS = [
    # Direct mentions
    "sponsor", "brought to you by", "today's sponsor",
    "this episode is sponsored", "this podcast is brought",
    
    # Common sponsor phrases
    "promo code", "discount code", "coupon code",
    "visit", "go to", "check out",
    "percent off", "% off", "discount",
    
    # Specific sponsors (customize for your podcast)
    "audible", "squarespace", "hellofresh", "betterhelp",
    "nord vpn", "nordvpn", "athletic greens",
    
    # Ad structure
    "before we get started", "before we dive in",
    "quick word from our sponsor", "message from our sponsor",
    "thanks to our sponsor",
]

# Intro/outro phrases (also usually host)
INTRO_OUTRO_KEYWORDS = [
    "welcome to", "welcome back", "thanks for listening",
    "that's all for today", "see you next time",
    "don't forget to subscribe", "leave a review",
    "if you enjoyed this episode",
]

# YouTube-specific host indicators (VERY RELIABLE)
YOUTUBE_HOST_KEYWORDS = [
    "subscribe", "hit the subscribe button", "subscribe to the channel",
    "subscribe to my channel", "don't forget to subscribe",
    "like and subscribe", "subscribe below", "click subscribe",
    "hit the bell", "notification bell", "smash that subscribe",
    "leave a like", "hit the like button", "thumbs up",
    "check out the description", "link in the description",
    "patreon", "support the channel", "support the show",
]


def load_transcript(json_path):
    """Load WhisperX JSON output."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments']


def contains_keywords(text, keywords):
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def detect_sponsor_segments(segments):
    """
    Identify which segments are likely from the host.
    Returns list of segment indices and detection type.
    """
    youtube_segments = []  # High confidence - YouTube CTAs
    sponsor_segments = []  # Medium confidence - sponsor reads
    intro_outro_segments = []  # Medium confidence - intro/outro
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        
        # Check for YouTube-specific keywords (HIGHEST PRIORITY)
        if contains_keywords(text, YOUTUBE_HOST_KEYWORDS):
            youtube_segments.append(i)
        
        # Check for sponsor keywords
        elif contains_keywords(text, SPONSOR_KEYWORDS):
            sponsor_segments.append(i)
        
        # Check intro/outro
        elif contains_keywords(text, INTRO_OUTRO_KEYWORDS):
            intro_outro_segments.append(i)
    
    # Return YouTube segments if found (most reliable)
    if youtube_segments:
        return youtube_segments, "youtube_cta"
    # Otherwise sponsor segments
    elif sponsor_segments:
        return sponsor_segments, "sponsor"
    # Otherwise intro/outro
    elif intro_outro_segments:
        return intro_outro_segments, "intro_outro"
    
    return [], "none"


def identify_host(segments, sponsor_segment_indices, detection_type):
    """
    Identify host based on who appears in sponsor segments.
    
    Returns:
        host_label: "Speaker 1" or "Speaker 2"
        confidence: float 0-1
        method: str describing how host was identified
    """
    if not sponsor_segment_indices:
        # Fallback: whoever speaks first and last
        return identify_host_fallback(segments)
    
    # Count appearances in sponsor segments
    speaker_counts = defaultdict(int)
    
    for idx in sponsor_segment_indices:
        speaker = segments[idx].get('speaker', 'Unknown')
        if speaker != 'Unknown':
            speaker_counts[speaker] += 1
    
    if not speaker_counts:
        return identify_host_fallback(segments)
    
    # Host is whoever appears most in sponsor segments
    host = max(speaker_counts, key=speaker_counts.get)
    total = sum(speaker_counts.values())
    confidence = speaker_counts[host] / total if total > 0 else 0.5
    
    # Adjust confidence based on detection type
    if detection_type == "youtube_cta":
        confidence = min(0.95, confidence + 0.2)  # Very high confidence
        method_desc = f"youtube_cta ({len(sponsor_segment_indices)} subscribe/like mentions)"
    elif detection_type == "sponsor":
        method_desc = f"sponsor_detection ({len(sponsor_segment_indices)} ad segments)"
    elif detection_type == "intro_outro":
        method_desc = f"intro_outro ({len(sponsor_segment_indices)} segments)"
    else:
        method_desc = "keyword_detection"
    
    return host, confidence, method_desc


def identify_host_fallback(segments):
    """
    Fallback method: identify host by:
    1. Who speaks first
    2. Who speaks last
    3. Who has more total speaking time
    """
    if not segments:
        return "Speaker 1", 0.5, "default"
    
    first_speaker = segments[0].get('speaker', 'Unknown')
    last_speaker = segments[-1].get('speaker', 'Unknown')
    
    # Count total speaking time
    speaking_time = defaultdict(float)
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        if speaker != 'Unknown':
            duration = seg.get('end', 0) - seg.get('start', 0)
            speaking_time[speaker] += duration
    
    # Host usually speaks first AND last
    if first_speaker == last_speaker and first_speaker != 'Unknown':
        return first_speaker, 0.7, "speaks_first_and_last"
    
    # Or has significantly more speaking time (>60%)
    total_time = sum(speaking_time.values())
    if total_time > 0:
        dominant_speaker = max(speaking_time, key=speaking_time.get)
        ratio = speaking_time[dominant_speaker] / total_time
        if ratio > 0.6:
            return dominant_speaker, ratio, f"dominant_speaker ({ratio:.1%})"
    
    # Default to whoever speaks first
    return first_speaker, 0.5, "speaks_first (fallback)"


def relabel_transcript(segments, host_label):
    """
    Relabel speakers as 'Host' and 'Guest'.
    Returns new segments list.
    """
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


def format_timestamp(t):
    """Format timestamp as HH:MM:SS.mmm"""
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def save_labeled_transcript(segments, output_path):
    """Save transcript with Host/Guest labels."""
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
    """Save JSON with Host/Guest labels and metadata."""
    output = {
        'segments': segments,
        'host_identification': metadata
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Identify host vs guest in podcast")
    ap.add_argument("--json", required=True, help="WhisperX JSON output")
    ap.add_argument("--output_txt", help="Output labeled transcript (optional)")
    ap.add_argument("--output_json", help="Output labeled JSON (optional)")
    ap.add_argument("--show_sponsor_segments", action="store_true",
                    help="Print detected sponsor segments")
    args = ap.parse_args()
    
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    # Load transcript
    print(f"Loading transcript: {json_path}")
    segments = load_transcript(json_path)
    print(f"  Total segments: {len(segments)}")
    
    # Detect sponsor segments
    print("\nDetecting host-identifying segments...")
    sponsor_indices, detection_type = detect_sponsor_segments(segments)
    
    if detection_type == "youtube_cta":
        print(f"  Found {len(sponsor_indices)} YouTube CTA segments (subscribe/like/bell)")
    elif detection_type == "sponsor":
        print(f"  Found {len(sponsor_indices)} sponsor/ad segments")
    elif detection_type == "intro_outro":
        print(f"  Found {len(sponsor_indices)} intro/outro segments")
    else:
        print(f"  No host-identifying keywords found")
    
    if args.show_sponsor_segments and sponsor_indices:
        print(f"\n{detection_type.upper()} segments:")
        for idx in sponsor_indices[:5]:  # Show first 5
            seg = segments[idx]
            speaker = seg.get('speaker', 'Unknown')
            text = seg.get('text', '')[:100]
            print(f"  [{idx}] {speaker}: {text}...")
        if len(sponsor_indices) > 5:
            print(f"  ... and {len(sponsor_indices) - 5} more")
    
    # Identify host
    print("\nIdentifying host...")
    host_label, confidence, method = identify_host(segments, sponsor_indices, detection_type)
    guest_label = "Speaker 1" if host_label == "Speaker 2" else "Speaker 2"
    
    print(f"  Host: {host_label} (confidence: {confidence:.1%})")
    print(f"  Guest: {guest_label}")
    print(f"  Method: {method}")
    
    # Relabel transcript
    print("\nRelabeling transcript...")
    relabeled_segments = relabel_transcript(segments, host_label)
    
    # Count segments per speaker
    host_count = sum(1 for s in relabeled_segments if s['speaker'] == 'Host')
    guest_count = sum(1 for s in relabeled_segments if s['speaker'] == 'Guest')
    print(f"  Host segments: {host_count}")
    print(f"  Guest segments: {guest_count}")
    
    # Save outputs
    metadata = {
        'host_original_label': host_label,
        'guest_original_label': guest_label,
        'confidence': confidence,
        'method': method,
        'sponsor_segments_detected': len(sponsor_indices)
    }
    
    if args.output_txt:
        output_txt = Path(args.output_txt)
        save_labeled_transcript(relabeled_segments, output_txt)
        print(f"\nSaved labeled transcript: {output_txt}")
    
    if args.output_json:
        output_json = Path(args.output_json)
        save_labeled_json(relabeled_segments, output_json, metadata)
        print(f"Saved labeled JSON: {output_json}")
    
    # Auto-generate output paths if not specified
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
