#!/usr/bin/env python3
"""
Draw the data-scaling sample: ~3,217 additional LLM-label candidates from TRAIN episodes only
(120 per llm_ft2/labels_majority_split.csv). Never touches validation / test / human-gold
episodes, labeled turns or duplicate texts. Original labels and split assignments are not modified.

Strategy (seeded, sequential draw without replacement, per dyad group):
    FH_long  every remaining female-host turn (>150 words; none shorter are left unlabeled)
    L        low-arousal candidates       ar_low  > 0.15
    H        high-arousal candidates      ar_high > 0.15
    N        negative-valence candidates  ft_neg  > 0.50
    P        positive-valence candidates  ft_pos  > 0.50
    R        uniform random from what is left of the group's pool (calibration sample)
The proxy columns come from turn_scores_llm2.csv (current model) and are only used to select;
they are never shown to the labelers. Within a stratum the draw is uniform, so extreme turns
are not over-represented. Per-episode caps limit concentration.

Turns the 256-token model cannot read (>250 tokens, i.e. the long female-host turns) are
condensed ONCE, documented, and the LLM labels the same text the model will see:
    head + " [...] " + tail, equal word counts, largest k with <= 250 XLM-R tokens.
The original text is kept in text_full.

Writes results/sentiment/llm_labels3/: sample_meta.csv, pilot_chunk_*.csv, chunk_*.csv,
pilot_fulltext_check.csv, manifest.json.

Usage:
    python src/sentiment/sample_scaling_set.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

KEY = ["episode_id", "turn_idx"]
FEMALE_HOST = ["FEMALE->FEMALE", "FEMALE->MALE"]
RATIO = dict(L=600, H=550, N=650, P=300, R=1117)           # proposed 3,217 split, rescaled per group
THR = dict(L=("ar_low", 0.15), H=("ar_high", 0.15), N=("ft_neg", 0.50), P=("ft_pos", 0.50))
MAX_TOK = 250


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def condense(text, tok):
    n = lambda s: len(tok(s)["input_ids"])
    if n(text) <= MAX_TOK:
        return text, False
    w = text.split()
    lo, hi = 2, len(w)
    while lo < hi:                                     # largest k with head(k/2)+tail(k/2) <= MAX_TOK
        k = (lo + hi + 1) // 2
        s = " ".join(w[:(k + 1) // 2]) + " [...] " + " ".join(w[-(k // 2):])
        lo, hi = (k, hi) if n(s) <= MAX_TOK else (lo, k - 1)
    return " ".join(w[:(lo + 1) // 2]) + " [...] " + " ".join(w[-(lo // 2):]), True


def draw(pool, strata_q, cap, rng, group):
    """Sequential stratified draw; shortfalls fall through to R. Returns rows with stratum + probability."""
    taken, out, ep_count = set(), [], {}
    short = 0
    for s in ["L", "H", "N", "P", "R"]:
        q = strata_q[s] + (short if s == "R" else 0)
        if s == "R":
            cand = pool[~pool.index.isin(taken)]
            col, thr = None, None
        else:
            col, thr = THR[s]
            cand = pool[(pool[col] > thr) & ~pool.index.isin(taken)]
        size = len(cand)
        got = 0
        for i in rng.permutation(cand.index.values):
            if got >= q:
                break
            e = pool.at[i, "episode_id"]
            if ep_count.get(e, 0) >= cap:
                continue
            ep_count[e] = ep_count.get(e, 0) + 1
            taken.add(i); got += 1
            out.append(dict(idx=i, stratum=s, group=group, stratum_pool=size, stratum_quota=q))
        short = q - got if s != "R" else 0
        if s == "R" and got < q:
            print(f"  WARNING {group}: random stratum short by {q - got}")
    o = pd.DataFrame(out)
    o["incl_prob"] = np.minimum(1.0, o.stratum_quota / o.stratum_pool)       # approx. inclusion probability within stratum
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_new", type=int, default=3217)
    ap.add_argument("--mf_new", type=int, default=950)
    ap.add_argument("--cap_mm", type=int, default=40)
    ap.add_argument("--cap_mf", type=int, default=90)
    ap.add_argument("--pilot", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out_dir", default="results/sentiment/llm_labels3")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("models/hf_converted/cardiff_xlmr_multi")

    T = pd.read_csv("results/sentiment/sentiment_turns.csv")
    D = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
    G = pd.read_csv("results/sentiment/gold_sheet_labeled.csv")
    S = pd.read_csv("results/sentiment/turn_scores_llm2.csv", usecols=KEY + ["ft_neg", "ft_pos", "ar_low", "ar_high"])
    split = D.groupby("episode_id").split.first()
    T["split"] = T.episode_id.map(split)
    assert not T[T.split == "train"].episode_id.isin(G.episode_id).any()
    prev = T[KEY + ["text"]].rename(columns={"turn_idx": "prev_turn_idx", "text": "prev_text"})
    T = T.merge(prev, left_on=["episode_id", "prev_turn_idx"], right_on=["episode_id", "prev_turn_idx"], how="left")
    P = T[(T.split == "train") & T.dyad.ne("MALE->UNKNOWN")].merge(D[KEY].assign(_l=1), on=KEY, how="left")
    P = P[P._l.isna()].drop(columns="_l")
    n0 = len(P)
    P = P[P.word_count.between(5, 150) | (P.dyad.isin(FEMALE_HOST) & (P.word_count > 150))]
    P = P[~P.text.isin(set(D.text) | set(G.text))]
    P = P[~P.text.duplicated(keep="first")].merge(S, on=KEY, how="left").reset_index(drop=True)
    print(f"train turns unlabeled {n0} -> eligible unique {len(P)}")
    P["group"] = np.where(P.dyad.isin(FEMALE_HOST), "FH", np.where(P.dyad == "MALE->FEMALE", "MF", "MM"))
    print(P.group.value_counts().to_dict())

    fh = P[P.group == "FH"].assign(stratum="FH_long", stratum_pool=lambda x: len(x), stratum_quota=lambda x: len(x), incl_prob=1.0)
    q_rest = a.total_new - len(fh)
    mf_cap_feas = int(P[P.group == "MF"].groupby("episode_id").size().clip(upper=a.cap_mf).sum())
    mf_q = min(a.mf_new, mf_cap_feas)
    mm_q = q_rest - mf_q
    mm_feas = int(P[P.group == "MM"].groupby("episode_id").size().clip(upper=a.cap_mm).sum())
    print(f"FH forced {len(fh)}; MF target {mf_q} (cap-feasible {mf_cap_feas}); MM target {mm_q} (cap-feasible {mm_feas})")
    assert mm_q <= mm_feas

    def quotas(total):
        tot = sum(RATIO.values())
        q = {s: int(round(total * v / tot)) for s, v in RATIO.items()}
        q["R"] += total - sum(q.values())
        return q

    parts = [fh]
    for g, tot, cap in (("MF", mf_q, a.cap_mf), ("MM", None, a.cap_mm)):
        if g == "MM":
            tot = a.total_new - sum(len(x) for x in parts)       # MM absorbs any MF shortfall caused by the episode cap
            assert tot <= mm_feas
        sub = P[P.group == g]
        r = draw(sub, quotas(tot), cap, rng, g)
        parts.append(sub.loc[r.idx.values].reset_index(drop=True).assign(
            stratum=r.stratum.values, stratum_pool=r.stratum_pool.values, stratum_quota=r.stratum_quota.values,
            incl_prob=r.incl_prob.values))
    sel = pd.concat(parts, ignore_index=True)
    sel["text_full"] = sel.text
    cd = [condense(t, tok) for t in sel.text]
    sel["text"], sel["truncated"] = [c[0] for c in cd], [c[1] for c in cd]
    sel["n_tokens"] = [len(tok(t)["input_ids"]) for t in sel.text]
    sel["prev_text"] = sel.prev_text.fillna("").map(lambda s: " ".join(s.split()[-100:]))      # context capped at its last 100 words
    sel = sel.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    sel["turn_id"] = [f"v{i:05d}" for i in range(len(sel))]

    # pilot: stratified by (group, stratum), always includes female-host long turns
    sel["pilot"] = False
    per = (sel.groupby(["group", "stratum"]).size() / len(sel) * a.pilot).round().astype(int).clip(lower=1)
    for (g, s), k in per.items():
        idx = sel[(sel.group == g) & (sel.stratum == s)].sample(min(k, ((sel.group == g) & (sel.stratum == s)).sum()), random_state=a.seed).index
        sel.loc[idx, "pilot"] = True
    keep = ["turn_id", "prev_text", "text"]
    pil, main_ = sel[sel.pilot], sel[~sel.pilot]
    n_pc = int(np.ceil(len(pil) / 84))
    for i, ix in enumerate(np.array_split(np.arange(len(pil)), n_pc)):
        pil.iloc[ix][keep].to_csv(f"{a.out_dir}/pilot_chunk_{i:02d}.csv", index=False)
    for i, ix in enumerate(np.array_split(np.arange(len(main_)), int(np.ceil(len(main_) / a.chunk)))):
        main_.iloc[ix][keep].to_csv(f"{a.out_dir}/chunk_{i:02d}.csv", index=False)
    trunc_pilot = pil[pil.truncated]
    trunc_pilot[["turn_id", "prev_text"]].assign(text=trunc_pilot.text_full).to_csv(f"{a.out_dir}/pilot_fulltext_check.csv", index=False)
    sel.to_csv(f"{a.out_dir}/sample_meta.csv", index=False)

    prompt = "documentation/sentiment_llm_labeling_prompt_v1.md"
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=a.seed, script="src/sentiment/sample_scaling_set.py",
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        prompt_file=prompt, prompt_sha256=sha(prompt), labeling_model_alias="sonnet", temperature="platform default (not controllable for subagents)",
        output_schema="CSV header turn_id,valence,arousal; one row per input row; lowercase labels; valence in {negative,neutral,positive}; arousal in {low,medium,high}",
        passes="3 independent passes (A,B,C), same prompt, same model; majority = >=2 of 3",
        truncation=f"turns >{MAX_TOK} XLM-R tokens condensed to head+' [...] '+tail (equal word counts); labeler sees the condensed text; text_full retained",
        prev_text_cap="last 100 words", eligible_pool=len(P), eligible_by_group=P.group.value_counts().to_dict(),
        quotas=dict(FH_forced=len(fh), MF_target=mf_q, MM_target_before_absorb=mm_q, ratio=RATIO), thresholds={k: list(v) for k, v in THR.items()},
        caps=dict(MM=a.cap_mm, MF=a.cap_mf), n_selected=len(sel), n_pilot=int(sel.pilot.sum()), n_main=int((~sel.pilot).sum()),
        n_episodes=int(sel.episode_id.nunique()), n_truncated=int(sel.truncated.sum()),
        strata_counts=sel.groupby(["group", "stratum"]).size().rename(lambda x: x).reset_index().assign(
            k=lambda x: x.group + "/" + x.stratum).set_index("k")[0].to_dict(),
        excluded="validation, test, human-gold episodes; already-labeled turns; duplicate texts")
    json.dump(manifest, open(f"{a.out_dir}/manifest.json", "w"), indent=1, default=str)
    print(json.dumps({k: manifest[k] for k in ["n_selected", "n_pilot", "n_main", "n_episodes", "n_truncated", "strata_counts"]}, indent=1))
    print("max per episode", sel.groupby("episode_id").size().max(), "episodes by group", sel.groupby("group").episode_id.nunique().to_dict())
    print("tokens max", sel.n_tokens.max())


if __name__ == "__main__":
    main()
