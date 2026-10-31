#!/usr/bin/env python3
"""
Step 4: prepare the human validation study of topic initiation and follow-up.
This script prepares materials only. It produces no labels of any kind.

Sampling frame: the 17,046 initiation-response pairs from the 131 mixed-gender
episodes, under the Report 1 (original) definition.

Design: 16 strata = dyad (M->F, F->M) x initiator role (host, guest)
x automated follow-up (yes, no) x initiator length (<= 8 words, > 8 words;
8 = median initiator length in the mixed sample's host turns). Gender is fixed
by dyad x role. 11 pairs are drawn per stratum without replacement
(176 items), at most 3 per episode, no repeated (episode, turn). Each item
carries its inclusion probability n_h / N_h and design weight N_h / n_h.

Outputs (results/topic_control_revision/human_validation/):
    manifest/sampling_frame_summary.csv   N_h, n_h, pi_h per stratum
    manifest/validation_manifest.csv      MASTER key (automated labels etc.)
                                          -- annotators must NOT open it
    annotator_1_sheet.csv, annotator_2_sheet.csv
                                          blinded, independently shuffled
    adjudication_template.csv             empty template
    sampling_metadata.json                seed, inputs, design
"""

import json
import os

import numpy as np
import pandas as pd

from common import (DATA, MIXED, SEED, TRANSCRIPT_DIR, VALID, build_pairs_with_text,
                    is_backchannel, merge_consecutive_same_speaker, parse_file)

PER_STRATUM = 11
MAX_PER_EPISODE = 3
LEN_CUT = 8
MAX_WORDS_SHOWN = 220


def clip(text, n=MAX_WORDS_SHOWN):
    w = str(text).split()
    if len(w) <= n:
        return " ".join(w), False
    head, tail = int(n * 0.65), n - int(n * 0.65)
    return " ".join(w[:head]) + " […] " + " ".join(w[-tail:]), True


def refuse_if_annotated():
    for k in (1, 2):
        p = os.path.join(VALID, f"annotator_{k}_sheet.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
            if (d[["Q1_new_topic", "Q2_engagement"]] != "").any().any():
                raise SystemExit(f"{p} already contains human labels; refusing to overwrite it.")


def main():
    refuse_if_annotated()
    rng = np.random.default_rng(SEED)
    rb = build_pairs_with_text()
    rb = rb.copy()
    rb["len_band"] = np.where(rb["initiator_word_count"] <= LEN_CUT, "short", "long")
    rb["stratum"] = (rb["dyad"].map({"MALE->FEMALE": "MF", "FEMALE->MALE": "FM"}) + "_"
                     + rb["initiator_role"].str.lower() + "_fu" + rb["followed_up"].astype(str)
                     + "_" + rb["len_band"])
    frame = rb.groupby("stratum").size().rename("N_h")

    # Draw stratum by stratum in a random order of candidates, honouring the
    # per-episode cap across strata.
    per_ep = {}
    picked = []
    for s in sorted(frame.index):
        cand = rb[rb["stratum"] == s].sample(frac=1.0, random_state=int(rng.integers(1e9)))
        take = []
        for idx, r in cand.iterrows():
            if per_ep.get(r["episode_id"], 0) >= MAX_PER_EPISODE:
                continue
            take.append(idx)
            per_ep[r["episode_id"]] = per_ep.get(r["episode_id"], 0) + 1
            if len(take) == PER_STRATUM:
                break
        picked.extend(take)
    s = rb.loc[picked].copy()
    assert not s.duplicated(["episode_id", "init_idx"]).any()
    n_h = s.groupby("stratum").size().rename("n_h")
    summ = pd.concat([frame, n_h], axis=1).fillna(0)
    summ["inclusion_prob"] = summ["n_h"] / summ["N_h"]
    summ["design_weight"] = summ["N_h"] / summ["n_h"]

    # Context: the two turns before the initiation (same filtered turn sequence
    # that the measure uses).
    ctx = {}
    for ep in s["episode_id"].unique():
        turns = merge_consecutive_same_speaker(parse_file(os.path.join(TRANSCRIPT_DIR, f"{ep}.txt"), ep))
        ctx[ep] = [t for t in turns if not is_backchannel(t.text)]

    s = s.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    s["item_id"] = [f"TC{i + 1:03d}" for i in range(len(s))]
    sheet_rows, trunc = [], []
    for _, r in s.iterrows():
        T = ctx[r["episode_id"]]
        i, j = int(r["init_idx"]), int(r["resp_idx"])
        a_spk = T[i].speaker_id

        def lab(t):
            return "A" if t.speaker_id == a_spk else "B"
        prev = []
        for k in [i - 2, i - 1]:
            if k >= 0:
                txt, _ = clip(T[k].text, 120)
                prev.append(f"[{lab(T[k])}] {txt}")
        between = [f"[{lab(T[k])}] {clip(T[k].text, 60)[0]}" for k in range(i + 1, j)]
        init_txt, t1 = clip(T[i].text)
        resp_txt, t2 = clip(T[j].text)
        trunc.append(t1 or t2)
        sheet_rows.append({
            "item_id": r["item_id"],
            "context_before": "  ||  ".join(prev) if prev else "(start of episode)",
            "A_initiating_turn": init_txt,
            "turns_between": "  ||  ".join(between),
            "B_next_turn": resp_txt,
            "Q1_new_topic": "", "Q2_engagement": "", "confidence_1to3": "", "notes": "",
        })
    s["text_truncated_on_sheet"] = trunc
    sheet = pd.DataFrame(sheet_rows)

    os.makedirs(os.path.join(VALID, "manifest"), exist_ok=True)
    keep = ["item_id", "episode_id", "init_idx", "resp_idx", "initiator_start_sec", "dyad",
            "initiator_role", "initiator_gender", "stratum", "len_band", "initiator_word_count",
            "responder_word_count", "incoming_similarity", "init_cutoff", "similarity",
            "followup_threshold", "followed_up", "text_truncated_on_sheet"]
    man = s[keep].merge(summ[["N_h", "n_h", "inclusion_prob", "design_weight"]],
                        left_on="stratum", right_index=True)
    man = man.sort_values("item_id")
    man.to_csv(os.path.join(VALID, "manifest", "validation_manifest.csv"), index=False)
    summ.reset_index().rename(columns={"index": "stratum"}).to_csv(
        os.path.join(VALID, "manifest", "sampling_frame_summary.csv"), index=False)

    for k, seed in [(1, SEED + 1), (2, SEED + 2)]:
        sheet.sample(frac=1.0, random_state=seed).to_csv(
            os.path.join(VALID, f"annotator_{k}_sheet.csv"), index=False)
    adj = pd.DataFrame(columns=["item_id", "annotator_1_Q1", "annotator_2_Q1", "annotator_1_Q2",
                                "annotator_2_Q2", "final_Q1", "final_Q2", "adjudicated_by", "rationale"])
    adj.to_csv(os.path.join(VALID, "adjudication_template.csv"), index=False)
    meta = {"seed": SEED, "per_stratum": PER_STRATUM, "max_per_episode": MAX_PER_EPISODE,
            "length_cut_words": LEN_CUT, "max_words_shown": MAX_WORDS_SHOWN,
            "n_items": len(s), "n_episodes": int(s["episode_id"].nunique()),
            "frame": "17,046 mixed-gender pairs, Report 1 definition",
            "input": os.path.relpath(os.path.join(DATA, "mixed_pairs_with_text.csv"), VALID)}
    with open(os.path.join(VALID, "sampling_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(summ)
    print(f"[OK] {len(s)} items from {s['episode_id'].nunique()} episodes; "
          f"{sum(trunc)} items have truncated text on the sheet")


if __name__ == "__main__":
    main()
