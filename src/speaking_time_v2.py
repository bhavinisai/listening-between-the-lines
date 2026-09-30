#!/usr/bin/env python3
"""
speaking_time_v2.py

Total speaking time and share of speaking time per speaker per episode, from
the WhisperX segment timestamps, joined to the host->guest gender dyads
(episode_dyads.py) and the host identity used in dyad_analysis_v2.py.

Speaking time of a speaker = sum of (end - start) over their diarized
segments. As in compute_bias_features_v2.py, segments without a speaker are
dropped and segments starting in the first/last --exclude_edges_sec seconds
of the episode (intro/outro) are excluded. The host and the guest of an
episode are the HOST / GUEST speakers with the most words, as in
episode_dyads.py. Host share = host time / (host time + guest time).

Outputs:
  results/features/speaking_time_v2.csv        one row per (episode, speaker)
  results/dyads/speaking_time_dyads_v2.csv     one row per episode: dyad, host,
                                               host/guest minutes, host share
  results/dyads/speaking_time_by_dyad_v2.csv   descriptives per dyad
  results/dyads/speaking_time_by_host_v2.csv   descriptives per host x guest gender

Usage:
  python src/speaking_time_v2.py
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

from dyad_analysis_v2 import DYAD_ORDER, host_names


def speaker_times(path, exclude_edges_sec):
    """Per speaker: role, gender, speaking seconds, word count, segment count."""
    with open(path, "r", encoding="utf-8") as f:
        segs = [s for s in json.load(f).get("segments", []) if (s.get("text") or "").strip()]
    if not segs:
        return {}, 0.0
    ep_end = max(float(s.get("end") or s.get("start") or 0.0) for s in segs)
    lo, hi = exclude_edges_sec, ep_end - exclude_edges_sec

    out = defaultdict(lambda: {"role": None, "gender": None, "seconds": 0.0, "words": 0, "segments": 0})
    for s in segs:
        spk = s.get("speaker")
        if not spk:
            continue
        start, end = float(s.get("start") or 0.0), float(s.get("end") or 0.0)
        if exclude_edges_sec > 0 and not (lo <= start <= hi):
            continue
        a = out[spk]
        a["role"] = a["role"] or (s.get("speaker_role") or "UNKNOWN").upper()
        a["gender"] = a["gender"] or (s.get("gender") or "unknown").lower()
        a["seconds"] += max(0.0, end - start)
        a["words"] += len(s["text"].split())
        a["segments"] += 1
    return out, ep_end


def main():
    ap = argparse.ArgumentParser(description="Speaking time per speaker per episode")
    ap.add_argument("--input_glob",
                    default="data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json")
    ap.add_argument("--dyads", default="results/features/episode_dyads.csv")
    ap.add_argument("--exclude_edges_sec", type=float, default=60.0)
    ap.add_argument("--out_speakers", default="results/features/speaking_time_v2.csv")
    ap.add_argument("--dyad_dir", default="results/dyads")
    args = ap.parse_args()

    dy = pd.read_csv(args.dyads)
    keep = set(dy.episode_id)
    rows = []
    for path in sorted(glob.glob(args.input_glob)):
        eid = os.path.splitext(os.path.basename(path))[0]
        if eid not in keep:
            continue
        spk, ep_end = speaker_times(path, args.exclude_edges_sec)
        for sid, a in spk.items():
            rows.append({"episode_id": eid, "speaker_id": sid, "role": a["role"], "gender": a["gender"],
                         "speaking_sec": a["seconds"], "speaking_min": a["seconds"] / 60,
                         "word_count": a["words"], "segments": a["segments"],
                         "episode_min": ep_end / 60})
    sp = pd.DataFrame(rows)

    # Host and guest = top-word-count HOST / GUEST speaker (as in episode_dyads.py)
    top = (sp[sp.role.isin(["HOST", "GUEST"])]
           .sort_values(["episode_id", "role", "word_count"], ascending=[True, True, False])
           .drop_duplicates(["episode_id", "role"]))
    hg_total = top.groupby("episode_id").speaking_sec.sum()
    sp["share_of_host_guest_time"] = sp.apply(
        lambda r: r.speaking_sec / hg_total[r.episode_id]
        if (r.episode_id, r.speaker_id) in set(zip(top.episode_id, top.speaker_id)) else float("nan"), axis=1)
    sp["share_of_all_speech"] = sp.speaking_sec / sp.groupby("episode_id").speaking_sec.transform("sum")
    os.makedirs(os.path.dirname(args.out_speakers) or ".", exist_ok=True)
    sp.to_csv(args.out_speakers, index=False)

    host = top[top.role == "HOST"].set_index("episode_id")
    guest = top[top.role == "GUEST"].set_index("episode_id")
    ep = dy[["episode_id", "dyad", "host_gender", "guest_gender"]].copy()
    ep.insert(1, "host_name", ep.episode_id.map(host_names(args.input_glob)))
    ep["host_min"] = ep.episode_id.map(host.speaking_min)
    ep["guest_min"] = ep.episode_id.map(guest.speaking_min)
    ep["host_guest_min"] = ep.host_min + ep.guest_min
    ep["other_speakers_min"] = ep.episode_id.map(
        sp.groupby("episode_id").speaking_min.sum()) - ep.host_guest_min
    ep["episode_min"] = ep.episode_id.map(sp.groupby("episode_id").episode_min.first())
    ep["host_time_share"] = ep.host_min / ep.host_guest_min
    ep["host_word_share"] = ep.episode_id.map(host.word_count) / (
        ep.episode_id.map(host.word_count) + ep.episode_id.map(guest.word_count))
    os.makedirs(args.dyad_dir, exist_ok=True)
    ep.to_csv(f"{args.dyad_dir}/speaking_time_dyads_v2.csv", index=False)

    q = lambda p: (lambda s: s.quantile(p))
    by_dyad = ep.groupby("dyad").agg(
        episodes=("episode_id", "size"),
        host_share_mean=("host_time_share", "mean"), host_share_median=("host_time_share", "median"),
        host_share_q1=("host_time_share", q(.25)), host_share_q3=("host_time_share", q(.75)),
        host_min_median=("host_min", "median"), guest_min_median=("guest_min", "median"),
        host_guest_min_median=("host_guest_min", "median")).reindex(DYAD_ORDER)
    by_dyad.to_csv(f"{args.dyad_dir}/speaking_time_by_dyad_v2.csv")

    m = ep[ep.host_name.notna()]
    by_host = m.pivot_table(index=["host_name", "host_gender"], columns="guest_gender",
                            values="host_time_share", aggfunc=["size", "median"])
    by_host.columns = [f"{'episodes' if a == 'size' else 'median_host_share'}_{b}_guest" for a, b in by_host.columns]
    by_host = by_host.sort_index(level="host_gender")
    by_host.to_csv(f"{args.dyad_dir}/speaking_time_by_host_v2.csv")

    pd.set_option("display.width", 200)
    print(f"{len(ep)} episodes; {ep.host_name.notna().sum()} with a library-matched host")
    print("\nBy dyad:")
    print(by_dyad.round(3).to_string())
    print("\nBy host (median host share):")
    print(by_host.round(3).to_string())
    print(f"\nTime share vs word share: r = {ep.host_time_share.corr(ep.host_word_share):.3f}")


if __name__ == "__main__":
    main()
