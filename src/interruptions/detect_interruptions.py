#!/usr/bin/env python3
"""
detect_interruptions.py

Step 2 of the interruption analysis (CPU, all episodes). Reads the overlapping
diarization from overlap_diarization.py together with the existing labeled
transcript, and finds and classifies overlap events.

1. Speaker mapping. Each pyannote speaker is matched to a transcript speaker
   (and so to HOST/GUEST plus gender) by how much its turns overlap in time with
   that speaker's WhisperX words (Hungarian assignment). Purity = share of the
   pyannote speaker's word-covered time that falls on its matched speaker.
   Episodes where a HOST or GUEST mapping has purity < --min_purity are flagged
   and left out of the per-episode rates.

2. Turns. A speaker's pyannote turns separated by less than --merge_gap seconds
   are merged into one turn, so short pauses don't count as giving up the floor.

3. Overlap events. B overlaps A when B's turn starts while A's turn is running.
   B is the overlapper and A the overlapped speaker. Each event is classified:
     simultaneous_start        both turns start within --simul_window s; no clear overlapper
     (dropped)                 overlap shorter than --min_overlap s
     backchannel               B's turn ends inside A's and is short (<= --bc_max_dur s),
                               or B's transcribed words are only backchannel tokens
     unsuccessful_interruption B's turn ends inside A's turn; A keeps the floor
     terminal_overlap          B outlasts A, but the overlap is <= --terminal_max_overlap s
                               (B starting as A finishes, not cutting A off)
     successful_interruption   B outlasts A after a longer overlap; B takes the floor

   The transcript words in and around the event are written out too, to help
   with checking events by ear (sample_validation_clips.py). During overlap,
   WhisperX usually transcribes only the louder speaker, so B's words may be missing.

Outputs (in --out_dir):
  interruption_events.csv            one row per overlap event
  interruption_rates_by_episode.csv  one row per episode: host/guest counts and rates
  episode_qc.csv                     speaker mapping purity and overlap totals per episode

Rates are given per 10 minutes of the *overlapped* speaker's speaking time
(their time on the floor is the opportunity to be interrupted).

Usage:
  python src/interruptions/detect_interruptions.py
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BACKCHANNEL_RE = re.compile(
    r"^(yeah|yes|yep|yup|mm+|mm+[- ]?hmm+|hmm+|uh[- ]?huh|right|exactly|totally|true|"
    r"sure|ok(ay)?|mhm|wow|oh|ah|haan|achha|accha|ji|correct|absolutely|of course)$",
    re.IGNORECASE,
)
TYPES = ["successful_interruption", "unsuccessful_interruption", "terminal_overlap",
         "backchannel", "simultaneous_start"]


def load_words(transcript):
    """Timed WhisperX words as (start, end, speaker, word), sorted by start."""
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append((float(w["start"]), float(w["end"]), w.get("speaker", seg.get("speaker")), w["word"]))
    words.sort(key=lambda w: w[0])
    return words


def merge_turns(turns, gap):
    """Merge each speaker's turns separated by less than `gap` seconds."""
    merged = []
    for spk in sorted({t["speaker"] for t in turns}):
        cur = None
        for t in sorted((t for t in turns if t["speaker"] == spk), key=lambda t: t["start"]):
            if cur and t["start"] - cur["end"] < gap:
                cur["end"] = max(cur["end"], t["end"])
            else:
                if cur:
                    merged.append(cur)
                cur = {"start": t["start"], "end": t["end"], "speaker": spk}
        if cur:
            merged.append(cur)
    merged.sort(key=lambda t: (t["start"], t["end"]))
    return merged


def map_speakers(turns, words):
    """Map pyannote speakers to transcript speakers by word-time overlap."""
    p_spk = sorted({t["speaker"] for t in turns})
    o_spk = sorted({w[2] for w in words if w[2]})
    if not p_spk or not o_spk:
        return {}
    w_start = np.array([w[0] for w in words])
    m = np.zeros((len(p_spk), len(o_spk)))
    for t in turns:
        i = p_spk.index(t["speaker"])
        lo = max(np.searchsorted(w_start, t["start"]) - 1, 0)
        hi = np.searchsorted(w_start, t["end"])
        for ws, we, spk, _ in words[lo:hi]:
            ov = min(we, t["end"]) - max(ws, t["start"])
            if ov > 0 and spk:
                m[i, o_spk.index(spk)] += ov
    rows, cols = linear_sum_assignment(-m)
    mapping = {}
    for r, c in zip(rows, cols):
        total = m[r].sum()
        mapping[p_spk[r]] = {"orig_speaker": o_spk[c], "purity": m[r, c] / total if total else 0.0,
                             "matched_sec": m[r, c]}
    return mapping


def words_in(words, w_start, start, end, speaker=None, max_words=40):
    lo = max(np.searchsorted(w_start, start) - 1, 0)
    hi = np.searchsorted(w_start, end)
    out = [w[3] for w in words[lo:hi] if w[1] > start and w[0] < end and (speaker is None or w[2] == speaker)]
    return " ".join(out[:max_words])


def is_backchannel_text(text, max_words=4):
    toks = re.findall(r"[A-Za-z'-]+", text.lower())
    return 0 < len(toks) <= max_words and all(BACKCHANNEL_RE.match(t) for t in toks)


def detect_events(turns, args):
    """Sweep over merged turns; yield (A = overlapped turn, B = overlapping turn, type, overlap)."""
    active = []
    for b in turns:
        active = [a for a in active if a["end"] > b["start"]]
        for a in active:
            if a["speaker"] == b["speaker"]:
                continue
            overlap = min(a["end"], b["end"]) - b["start"]
            if b["start"] - a["start"] < args.simul_window:
                yield a, b, "simultaneous_start", overlap
                continue
            if overlap < args.min_overlap:
                continue
            if b["end"] <= a["end"]:
                kind = "backchannel" if b["end"] - b["start"] <= args.bc_max_dur else "unsuccessful_interruption"
            elif overlap <= args.terminal_max_overlap:
                kind = "terminal_overlap"
            else:
                kind = "successful_interruption"
            yield a, b, kind, overlap
        active.append(b)


def process_episode(diar_path, transcript_path, args):
    diar = json.load(open(diar_path))
    transcript = json.load(open(transcript_path))
    ep = diar["episode_id"]
    role_of = transcript["speaker_role_mapping"]
    gender_of = {s: v.get("gender") for s, v in transcript["speaker_gender_mapping"].items()}

    words = load_words(transcript)
    w_start = np.array([w[0] for w in words])
    mapping = map_speakers(diar["diarization"], words)

    def label(p):
        o = mapping.get(p, {}).get("orig_speaker")
        return o, role_of.get(o, "OTHER"), gender_of.get(o)

    turns = merge_turns(diar["diarization"], args.merge_gap)

    events = []
    for a, b, kind, overlap in detect_events(turns, args):
        a_orig, a_role, a_gender = label(a["speaker"])
        b_orig, b_role, b_gender = label(b["speaker"])
        b_text = words_in(words, w_start, b["start"], b["end"], speaker=b_orig)
        if kind == "unsuccessful_interruption" and is_backchannel_text(b_text):
            kind = "backchannel"
        events.append({
            "episode_id": ep, "type": kind,
            "overlapper_role": b_role, "overlapper_gender": b_gender,
            "overlapped_role": a_role, "overlapped_gender": a_gender,
            "overlap_start": round(b["start"], 3), "overlap_dur": round(overlap, 3),
            "overlapped_turn_start": round(a["start"], 3), "overlapped_turn_end": round(a["end"], 3),
            "overlapper_turn_start": round(b["start"], 3), "overlapper_turn_end": round(b["end"], 3),
            "overlapped_context": words_in(words, w_start, b["start"] - args.context_sec, b["start"], speaker=a_orig),
            "overlapper_text": b_text,
        })

    # speaking time per role from the merged overlapping turns
    speak_sec = {"HOST": 0.0, "GUEST": 0.0}
    for t in turns:
        role = label(t["speaker"])[1]
        if role in speak_sec:
            speak_sec[role] += t["end"] - t["start"]

    purity = {label(p)[1]: v["purity"] for p, v in mapping.items()}
    roles_found = sorted(purity)
    qc = {
        "episode_id": ep,
        "duration_min": diar["duration"] / 60,
        "n_pyannote_speakers": len({t["speaker"] for t in diar["diarization"]}),
        "roles_mapped": "+".join(roles_found),
        "host_purity": purity.get("HOST", np.nan),
        "guest_purity": purity.get("GUEST", np.nan),
        "n_events": len(events),
        "overlap_sec_total": sum(e["overlap_dur"] for e in events),
    }
    qc["mapping_ok"] = bool(roles_found == ["GUEST", "HOST"]
                            and min(qc["host_purity"], qc["guest_purity"]) >= args.min_purity)

    rates = {"episode_id": ep, "host_speak_min": speak_sec["HOST"] / 60, "guest_speak_min": speak_sec["GUEST"] / 60}
    for who, other in [("host", "GUEST"), ("guest", "HOST")]:
        other_min = speak_sec[other] / 60
        for kind in TYPES:
            n = sum(1 for e in events if e["type"] == kind
                    and e["overlapper_role"] == who.upper() and e["overlapped_role"] == other)
            rates[f"{who}_{kind}_n"] = n
            rates[f"{who}_{kind}_per10"] = 10 * n / other_min if other_min > 0 else np.nan
    return events, rates, qc


def main():
    ap = argparse.ArgumentParser(description="Detect and classify interruptions from overlapping diarization")
    ap.add_argument("--diar_dir", default="data/outputs/overlap")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--dyads", default="results/dyads/speaking_time_dyads_v2.csv")
    ap.add_argument("--out_dir", default="results/interruptions")
    ap.add_argument("--merge_gap", type=float, default=0.5, help="merge same-speaker turns closer than this (s)")
    ap.add_argument("--min_overlap", type=float, default=0.3, help="ignore overlaps shorter than this (s)")
    ap.add_argument("--simul_window", type=float, default=0.2, help="turn starts this close = simultaneous start (s)")
    ap.add_argument("--bc_max_dur", type=float, default=1.0, help="max length of a backchannel turn (s)")
    ap.add_argument("--terminal_max_overlap", type=float, default=0.5,
                    help="overlap at A's turn end up to this long = terminal overlap, not interruption (s)")
    ap.add_argument("--min_purity", type=float, default=0.7, help="min speaker-mapping purity to keep an episode")
    ap.add_argument("--context_sec", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_events, all_rates, all_qc = [], [], []
    diar_files = sorted(glob.glob(os.path.join(args.diar_dir, "*_overlap_diarization.json")))
    if not diar_files:
        raise SystemExit(f"No *_overlap_diarization.json files in {args.diar_dir}")
    for path in diar_files:
        ep = os.path.basename(path).replace("_overlap_diarization.json", "")
        tpath = os.path.join(args.transcript_dir, f"{ep}_whisperx_diarized.gender.host_guest.json")
        if not os.path.exists(tpath):
            print(f"[skip] no transcript for {ep}")
            continue
        events, rates, qc = process_episode(path, tpath, args)
        all_events += events
        all_rates.append(rates)
        all_qc.append(qc)

    events = pd.DataFrame(all_events)
    rates = pd.DataFrame(all_rates)
    qc = pd.DataFrame(all_qc)

    dy = pd.read_csv(args.dyads)
    dy["episode_id"] = dy.episode_id.str.split("_whisperx").str[0]
    dy = dy[["episode_id", "host_name", "dyad", "host_gender", "guest_gender"]]
    events = events.merge(dy, on="episode_id", how="left")
    rates = rates.merge(dy, on="episode_id", how="left").merge(qc[["episode_id", "mapping_ok"]], on="episode_id")
    n_bad = (~rates.mapping_ok).sum()
    rates = rates[rates.mapping_ok].drop(columns="mapping_ok")

    events.to_csv(os.path.join(args.out_dir, "interruption_events.csv"), index=False)
    rates.to_csv(os.path.join(args.out_dir, "interruption_rates_by_episode.csv"), index=False)
    qc.to_csv(os.path.join(args.out_dir, "episode_qc.csv"), index=False)

    print(f"Episodes processed: {len(qc)}  kept: {len(rates)}  dropped for speaker mapping: {n_bad}")
    print(events.groupby(["type", "overlapper_role"]).size().unstack(fill_value=0))
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
