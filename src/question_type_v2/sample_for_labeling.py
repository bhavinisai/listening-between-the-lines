#!/usr/bin/env python3
"""
v2 Step 2-3: host-level split + sampling of host questions for LLM labeling.

Split (by host, so no host appears in both train/dev and test):
    test hosts  = Samdish Bhatia, Faye Dsouza, plus a random half of the
                  single-episode "unknown_*" hosts (seeded).
    dev  hosts  = everyone else (used for training + host-grouped CV).

Leakage controls:
    * exact-duplicate questions (normalised text) are assigned to ONE split: a
      text seen in any test host is test-only and removed from dev.
    * the 120 human-labeled gold questions are excluded from both samples
      (they are a separate evaluation set).
    * questions with fewer than 2 words are dropped.

Samples:
    test : N_TEST random questions from the test pool (natural distribution,
           so test macro-F1 reflects deployment).
    dev  : (a) host-asked rows of the original 1,280 seed set that fall in dev
           hosts (to be relabeled under the v2 rules), plus
           (b) per-v1-predicted-type quotas, oversampling the weak classes,
           plus (c) a few regex-flagged challenge/leading candidates.

All sampled rows are shuffled together (labelers cannot tell test from dev and
never see v1 predictions) and written as chunks for the labelers.

Does not modify any v1 file.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

TEST_NAMED = ["Samdish Bhatia", "Faye Dsouza"]
QUOTA = {"challenge": 500, "professional": 400, "personal": 400,
         "leading": 250, "open": 250, "closed": 250}
N_FLAGGED = 150
N_TEST = 800
CHUNK = 600

FLAG_RE = re.compile(
    r"\b(?:but|really|isn't it|aren't you|don't you|didn't you|wouldn't you|"
    r"how can you|how could you|prove|evidence|surely|hypocri|you said|you told|"
    r"i thought you)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/question_type_v2/pool_host_questions.csv")
    ap.add_argument("--v1", default="results/question_type_analysis/question_classification.csv")
    ap.add_argument("--seed_labeled", default="results/question_type_analysis/question_type_seed_labeled.csv")
    ap.add_argument("--gold", default="gold_work/gold_full.csv")
    ap.add_argument("--out_dir", default="results/question_type_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pool = pd.read_csv(args.pool)
    v1 = pd.read_csv(args.v1, usecols=["episode_id", "start", "text", "asker_role", "predicted_type", "confidence"])
    v1 = v1[v1.asker_role == "HOST"].rename(columns={"predicted_type": "v1_pred", "confidence": "v1_conf"})
    v1["start"] = v1["start"].round(3)
    v1 = v1.drop_duplicates(["episode_id", "start", "text"])
    pool["start_r"] = pool["start"].round(3)
    pool = pool.merge(v1[["episode_id", "start", "text", "v1_pred", "v1_conf"]].rename(columns={"start": "start_r"}),
                      on=["episode_id", "start_r", "text"], how="left")
    print(f"[info] v1 prediction joined for {pool.v1_pred.notna().mean():.1%} of pool")

    # ---- host split
    unk = sorted(h for h in pool.host_id.unique() if h.startswith("unknown_"))
    test_unk = set(rng.choice(unk, size=len(unk) // 2, replace=False))
    test_hosts = set(TEST_NAMED) | test_unk
    pool["split"] = np.where(pool.host_id.isin(test_hosts), "test", "dev")
    hs = pool.groupby("host_id").agg(split=("split", "first"), n_questions=("qid", "size"),
                                      n_episodes=("episode_id", "nunique"), host_gender=("host_gender", "first"))
    hs.sort_values("n_questions", ascending=False).to_csv(os.path.join(args.out_dir, "host_split.csv"))

    # ---- exclusions / dedupe
    gold = pd.read_csv(args.gold)
    gold_norm = set(gold.text.map(lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()))
    pool["n_words"] = pool.text.str.split().str.len()
    pool = pool[(pool.n_words >= 2) & ~pool.text_norm.isin(gold_norm)].copy()
    test_texts = set(pool.loc[pool.split == "test", "text_norm"])
    pool = pool[~((pool.split == "dev") & pool.text_norm.isin(test_texts))]   # cross-split duplicates -> test only
    pool = pool.drop_duplicates(["split", "text_norm"]).reset_index(drop=True)
    print(pool.groupby("split").size().rename("unique questions after exclusions").to_string())

    # ---- test sample
    tp = pool[pool.split == "test"]
    test_s = tp.sample(n=min(N_TEST, len(tp)), random_state=args.seed).assign(source="test_random")

    # ---- dev sample
    dev = pool[pool.split == "dev"]
    seed = pd.read_csv(args.seed_labeled)
    seed = seed[seed.asker_role == "HOST"]
    seed_keys = set(zip(seed.episode_id, seed.text))
    dev_seed = dev[[(e, t) in seed_keys for e, t in zip(dev.episode_id, dev.text)]].assign(source="seed_host_relabel")
    taken = set(dev_seed.qid)
    parts = [dev_seed]
    rest = dev[~dev.qid.isin(taken)]
    flagged = rest[rest.text.str.contains(FLAG_RE)]
    fl = flagged.sample(n=min(N_FLAGGED, len(flagged)), random_state=args.seed).assign(source="regex_flagged")
    parts.append(fl)
    taken |= set(fl.qid)
    rest = rest[~rest.qid.isin(taken)]
    for t, n in QUOTA.items():
        sub = rest[rest.v1_pred == t]
        parts.append(sub.sample(n=min(n, len(sub)), random_state=args.seed).assign(source=f"v1pred_{t}"))
    dev_s = pd.concat(parts).drop_duplicates("qid")

    allq = pd.concat([test_s, dev_s]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    allq["chunk"] = allq.index // CHUNK
    manifest_cols = ["qid", "episode_id", "host_id", "host_gender", "guest_gender", "split", "source",
                     "v1_pred", "v1_conf", "text", "prev_sentence", "prev_turn_tail", "chunk"]
    allq[manifest_cols].to_csv(os.path.join(args.out_dir, "labeling_manifest.csv"), index=False)

    os.makedirs(os.path.join(args.out_dir, "label_chunks"), exist_ok=True)
    for c, g in allq.groupby("chunk"):
        g = g.copy()
        g["prev_turn_tail"] = g.prev_turn_tail.fillna("").map(lambda s: " ".join(str(s).split()[-30:]))
        g[["qid", "text", "prev_sentence", "prev_turn_tail"]].fillna("").to_csv(
            os.path.join(args.out_dir, "label_chunks", f"pass1_chunk_{c}.csv"), index=False)

    print(f"[OK] {len(allq)} questions to label in {allq.chunk.nunique()} chunks")
    print(allq.groupby(["split", "source"]).size().to_string())
    print("dev hosts:", dev_s.host_id.nunique(), " test hosts:", test_s.host_id.nunique())
    assert not (set(test_s.text_norm) & set(dev_s.text_norm)), "text leakage test/dev"


if __name__ == "__main__":
    main()
