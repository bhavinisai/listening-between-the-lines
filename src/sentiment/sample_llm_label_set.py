#!/usr/bin/env python3
"""
Sample turns for LLM labeling, from episodes that are NOT in the gold sheet (so
the human gold set stays a clean test set). Oversamples turns the zero-shot
scorers consider polar / intense, so rare classes (negative, positive, high
arousal) are represented, and writes blind chunks of --chunk rows.

Usage:
    python src/sentiment/sample_llm_label_set.py --out_dir results/sentiment/llm_labels
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/sentiment/turn_scores.csv")
    ap.add_argument("--gold", default="results/sentiment/gold_sheet.csv")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--max_per_episode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/sentiment/llm_labels")
    ap.add_argument("--exclude", help="sample_meta.csv of an earlier draw; its (episode, turn) pairs are not drawn again")
    ap.add_argument("--quota", default="MALE->MALE:800,MALE->FEMALE:600,FEMALE->MALE:300,FEMALE->FEMALE:300")
    ap.add_argument("--prefix", default="t")
    a = ap.parse_args()

    df = pd.read_csv(a.scores)
    gold_eps = set(pd.read_csv(a.gold).episode_id)
    df = df[~df.episode_id.isin(gold_eps) & df.dyad.ne("MALE->UNKNOWN") & df.word_count.between(5, 150)].copy()
    prev = df[["episode_id", "turn_idx", "text"]].rename(columns={"turn_idx": "prev_turn_idx", "text": "prev_text"})
    df = df.merge(prev, on=["episode_id", "prev_turn_idx"], how="left")
    if a.exclude:
        old = pd.read_csv(a.exclude)[["episode_id", "turn_idx"]].assign(_x=1)
        df = df.merge(old, on=["episode_id", "turn_idx"], how="left")
        df = df[df._x.isna()].drop(columns="_x")
    df["turn_id"] = [f"{a.prefix}{i:05d}" for i in range(len(df))]

    # Per-dyad quotas (few female-host episodes exist outside the gold set), each
    # split into 20% likely negative / 20% likely positive / 20% high arousal /
    # 40% random. Small dyads get a higher per-episode cap.
    quota = {k: int(v) for k, v in (x.rsplit(":", 1) for x in a.quota.split(","))}
    cap = {"MALE->MALE": a.max_per_episode, "MALE->FEMALE": 35, "FEMALE->MALE": 40, "FEMALE->FEMALE": 40}
    df = df.sample(frac=1, random_state=a.seed)
    take = []
    for dy, n in quota.items():
        d = df[df.dyad == dy]
        got = []
        def draw(pool, k):
            used = {t for x in got for t in x.turn_id}
            pool = pool[~pool.turn_id.isin(used)].groupby("episode_id").head(cap[dy])
            got.append(pool.head(k))
        draw(d[(d.zs_valence < -0.4) & (d.ft_neg > 0.25)], int(.2 * n))
        draw(d[(d.zs_valence > 0.6) & (d.ft_pos > 0.4)], int(.2 * n))
        draw(d[d.arousal > df.arousal.quantile(.9)], int(.2 * n))
        draw(d, n - sum(len(x) for x in got))
        take.extend(got)
    s = pd.concat(take).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True)
    s.drop(columns=[c for c in s.columns if c.startswith(("ft_", "zs_", "emo_", "arousal", "valence"))]).to_csv(
        f"{a.out_dir}/sample_meta.csv", index=False)
    for i in range(0, len(s), a.chunk):
        s.iloc[i:i + a.chunk][["turn_id", "prev_text", "text"]].to_csv(f"{a.out_dir}/chunk_{i // a.chunk:02d}.csv", index=False)
    print(len(s), "turns,", s.episode_id.nunique(), "episodes,", s.dyad.value_counts().to_dict())


if __name__ == "__main__":
    main()
