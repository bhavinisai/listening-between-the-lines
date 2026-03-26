#!/usr/bin/env python3
"""
LLM-powered host/guest labeling for gender-labeled simple diarization output.

Uses OpenRouter LLM to intelligently identify HOST and GUEST roles.

Usage:
  export OPENROUTER_API_KEY="..."
  
  python host_guest_llm.py \
    --input data/outputs/gender/ep_001.diarized.gender.json \
    --output data/outputs/gender/ep_001.host_guest.json \
    --output-txt data/outputs/gender/ep_001.host_guest.txt \
    --model openrouter/free
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    app_title: str,
    http_referer: str,
    want_json_object: bool,
    debug_dump_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call OpenRouter API for LLM inference."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_title,
        "HTTP-Referer": http_referer,
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if want_json_object:
        payload["response_format"] = {"type": "json_object"}

    last_err: Optional[Exception] = None
    last_json: Dict[str, Any] = {}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)

            if resp.status_code == 404 and "No endpoints found" in resp.text:
                payload["model"] = "openrouter/free"
                resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)

            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1200]}")

            last_json = resp.json()
            content = last_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if debug_dump_path:
                debug_dump_path.write_text(json.dumps(last_json, indent=2), encoding="utf-8")
            
            return content, last_json

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            else:
                raise

    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_err}")


def extract_json_from_raw(raw: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    # Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract first JSON object
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    preview = raw[:1500].replace("\n", "\\n")
    raise ValueError(
        "Could not find JSON object in model output. "
        f"Model output preview (first 1500 chars): {preview}"
    )


def build_llm_prompt(segments: List[Dict[str, Any]], filename_hint: str) -> List[Dict[str, str]]:
    """Build LLM prompt with transcript snippets and gender info."""
    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        speaker_segments[speaker_id].append(seg)
    
    # Create transcript snippets
    blocks = []
    for speaker_id, segs in speaker_segments.items():
        # Get gender info
        gender = segs[0].get('gender', 'unknown')
        speaker_label = segs[0].get('speaker_label', f"Speaker {speaker_id + 1}")
        
        # Get first few examples (max 5 per speaker)
        examples = segs[:5]
        ex_lines = []
        for seg in examples:
            text = seg.get('text', '').strip()
            if text:
                start_min = int(seg.get('start', 0) // 60)
                start_sec = int(seg.get('start', 0) % 60)
                timestamp = f"{start_min:02d}:{start_sec:02d}"
                ex_lines.append(f"[{timestamp}] {text}")
        
        blocks.append(f"{speaker_label} ({gender}):\n" + "\n".join(ex_lines))

    snippet_text = "\n\n".join(blocks)

    user_msg = f"""
You are labeling speakers in a podcast transcript.

Task:
- Decide which speaker is the HOST (the interviewer / show runner).
- All other speakers should be GUEST.
- There must be exactly ONE HOST.
- Consider speaking patterns, gender roles, and content.

Input transcript file hint: {filename_hint}

Return ONLY valid JSON in this schema:
{{
  "host_speaker_id": <speaker_id>,
  "mapping": {{
     "<speaker_id>": "HOST" | "GUEST"
  }},
  "reasoning_brief": {{
     "<speaker_id>": "<1 sentence reason>"
  }},
  "confidence": {{
     "<speaker_id>": 0.0-1.0
  }}
}}

Rules:
- Use the speaker_id numbers (0, 1, etc.) from the input
- Consider both speaking patterns and gender information
- Output JSON only.

Transcript snippets:
{snippet_text}
""".strip()

    return [
        {"role": "system", "content": "Output ONLY JSON. No markdown. No extra text."},
        {"role": "user", "content": user_msg},
    ]


def load_simple_diarized_json(json_path):
    """Load simple diarized JSON format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_llm_labels(segments: List[Dict[str, Any]], llm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply LLM-determined labels to segments."""
    mapping = llm_result.get('mapping', {})
    reasoning = llm_result.get('reasoning_brief', {})
    confidence = llm_result.get('confidence', {})
    
    labeled_segments = []
    for seg in segments:
        speaker_id = seg.get('speaker_id', 0)
        role = mapping.get(str(speaker_id), 'UNKNOWN')
        reason = reasoning.get(str(speaker_id), 'No reasoning provided')
        conf = confidence.get(str(speaker_id), 0.5)
        
        labeled_seg = {
            'speaker_id': speaker_id,
            'speaker_label': seg.get('speaker_label', f"Speaker {speaker_id + 1}"),
            'gender': seg.get('gender', 'unknown'),
            'role': role,
            'reasoning': reason,
            'confidence': conf,
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'text': seg.get('text', ''),
            'original_confidence': seg.get('confidence', 0.0)
        }
        labeled_segments.append(labeled_seg)
    
    return labeled_segments


def write_host_guest_json(segments: List[Dict[str, Any]], llm_result: Dict[str, Any], output_path: Path):
    """Write JSON with LLM-determined HOST/GUEST labels."""
    # Get unique speakers
    speakers = list(set(seg['speaker_id'] for seg in segments))
    speakers.sort()
    
    # Create speaker info
    speaker_info = []
    for speaker_id in speakers:
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
        if speaker_segments:
            seg = speaker_segments[0]
            speaker_info.append({
                'speaker_id': speaker_id,
                'speaker_label': seg['speaker_label'],
                'gender': seg['gender'],
                'role': seg['role'],
                'reasoning': seg['reasoning'],
                'confidence': seg['confidence']
            })
    
    # Create output structure
    output_data = {
        'total_segments': len(segments),
        'speakers': speaker_info,
        'segments': segments,
        'llm_result': llm_result,
        'host_speaker_id': llm_result.get('host_speaker_id'),
        'host_guest_mapping': llm_result.get('mapping', {})
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Written: {output_path}")


def write_host_guest_txt(segments: List[Dict[str, Any]], output_path: Path):
    """Write TXT with LLM-determined HOST/GUEST labels."""
    lines = []
    
    for seg in segments:
        start_min = int(seg['start'] // 60)
        start_sec = int(seg['start'] % 60)
        timestamp = f"{start_min:02d}:{start_sec:02d}"
        
        role = seg['role']
        gender = seg['gender']
        text = seg['text'].strip()
        
        if text:
            lines.append(f"[{timestamp}] {role} ({gender}): {text}")
    
    # Add summary at the end
    lines.append("")
    lines.append("# LLM Speaker Analysis:")
    
    # Get unique speakers
    speakers = list(set(seg['speaker_id'] for seg in segments))
    speakers.sort()
    
    for speaker_id in speakers:
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
        if speaker_segments:
            seg = speaker_segments[0]
            role = seg['role']
            gender = seg['gender']
            speaker_label = seg['speaker_label']
            reasoning = seg['reasoning']
            confidence = seg['confidence']
            duration = sum(s['end'] - s['start'] for s in speaker_segments)
            lines.append(f"# {speaker_label} = {role} ({gender}) - {confidence:.2f} confidence")
            lines.append(f"#   Reason: {reasoning}")
            lines.append(f"#   Speaking time: {duration:.1f}s")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM-powered host/guest labeling for gender-labeled transcripts")
    parser.add_argument("--input", required=True, help="Input gender-labeled JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--output-txt", required=True, help="Output TXT file path")
    parser.add_argument("--model", default="openrouter/free", help="OpenRouter model")
    parser.add_argument("--max-segments-per-speaker", type=int, default=10, help="Max segments per speaker in prompt")
    parser.add_argument("--debug-dump-raw", action="store_true", help="Dump raw LLM response")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY")
        return 1
    
    input_path = Path(args.input)
    output_json_path = Path(args.output)
    output_txt_path = Path(args.output_txt)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Load input data
    print(f"Loading: {input_path}")
    data = load_simple_diarized_json(input_path)
    
    segments = data.get('segments', [])
    if not segments:
        print("Error: No segments found in input file")
        return 1
    
    print(f"Found {len(segments)} segments")
    
    # Limit segments for prompt
    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker_segments[seg['speaker_id']].append(seg)
    
    limited_segments = []
    for speaker_id, segs in speaker_segments.items():
        limited_segments.extend(segs[:args.max_segments_per_speaker])
    
    print(f"Using {len(limited_segments)} segments for LLM analysis")
    
    # Build LLM prompt
    print("Building LLM prompt...")
    messages = build_llm_prompt(limited_segments, input_path.name)
    
    # Call LLM
    print("Calling LLM for host/guest identification...")
    debug_path = Path("debug_llm_response.json") if args.debug_dump_raw else None
    
    try:
        llm_response, full_json = openrouter_chat(
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            timeout_s=60,
            retries=3,
            app_title="Host Guest Labeling",
            http_referer="https://github.com/bhavinisai/listening-between-the-lines",
            want_json_object=True,
            debug_dump_path=debug_path
        )
    except Exception as e:
        print(f"LLM call failed: {e}")
        return 1
    
    # Parse LLM response
    try:
        llm_result = extract_json_from_raw(llm_response)
        print("LLM response parsed successfully")
        
        # Show results
        host_id = llm_result.get('host_speaker_id')
        mapping = llm_result.get('mapping', {})
        reasoning = llm_result.get('reasoning_brief', {})
        
        print(f"Host speaker: {host_id}")
        for speaker_id, role in mapping.items():
            reason = reasoning.get(speaker_id, 'No reasoning')
            print(f"  Speaker {speaker_id}: {role} - {reason}")
        
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        if args.debug_dump_raw:
            print(f"Raw response saved to: {debug_path}")
        return 1
    
    # Apply labels to all segments
    labeled_segments = apply_llm_labels(segments, llm_result)
    
    # Write outputs
    print("Writing outputs...")
    write_host_guest_json(labeled_segments, llm_result, output_json_path)
    write_host_guest_txt(labeled_segments, output_txt_path)
    
    print(f"\n✅ Complete!")
    print(f"📄 JSON: {output_json_path}")
    print(f"📄 TXT: {output_txt_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
