#!/usr/bin/env python3
"""
Assemble the scaling dataset from the 3-pass LLM labels in results/sentiment/llm_labels3 and the
original train labels. Original files are read only; everything is written to
results/sentiment/llm_labels3/.

Outputs
    labels_new_all.csv         every new turn with its 3 passes, majority labels, agreement flags, provenance columns
    scaling_train_sets.csv     original train rows + accepted new rows, with nested-membership flags
                               (in_1783 = original train, in_3000 ⊂ in_5000 = all) and dyad-restoring weights
    scaling_summary.json       counts, class and dyad distributions per set, leakage checks, agreement
    manifest.json              (updated) pass-file hashes and final counts

Nested subsets: in_1783 = the original training rows; in_3000 adds a random ~1,217 of the accepted new
rows, drawn stratified by (group, stratum) so the added data has the same composition as the full
expansion; in_5000 = everything. Validation and test rows are never included.

Weights: per set, w_dyad = (original train dyad share) / (that set's dyad share), clipped to [0.5, 3],
rescaled to mean 1. This restores the original dyad mix without inventing female-host turns.

Usage:
    python src/sentiment/build_scaling_dataset.py
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

BASE = "results/sentiment/llm_labels3"
PASSES = "results/sentiment/llm_labels_passes/llm_labels3"
SEED = 43
TARGET_MID = 3000


def maj(row, cols):
    v = [row[c] for c in cols if isinstance(row[c], str)]
    top = max(set(v), key=v.count) if v else None
    return top if top is not None and v.count(top) >= 2 else np.nan


def load_pass(p):
    fs = sorted(glob.glob(f"{PASSES}/{p}_pilot_*.csv") + glob.glob(f"{PASSES}/{p}_[0-9][0-9].csv"))
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("turn_id").set_index("turn_id"), fs


def main():
    meta = pd.read_csv(f"{BASE}/sample_meta.csv").set_index("turn_id")
    files = {}
    for p in "ABC":
        L, fs = load_pass(p)
        files[p] = fs
        meta[f"valence_{p}"], meta[f"arousal_{p}"] = L.valence, L.arousal
    d = meta.reset_index()
    for t, ok in (("valence", {"negative", "neutral", "positive"}), ("arousal", {"low", "medium", "high"})):
        cols = [f"{t}_{p}" for p in "ABC"]
        bad = int((~d[cols].isin(ok)).sum().sum())
        assert bad == 0, f"{t}: {bad} unparseable labels (including missing)"
        d[f"{t}_maj"] = d.apply(lambda r: maj(r, cols), axis=1)
        d[f"{t}_all3"] = d[cols].nunique(axis=1).eq(1)
    d["accepted_valence"], d["accepted_arousal"] = d.valence_maj.notna(), d.arousal_maj.notna()
    d.to_csv(f"{BASE}/labels_new_all.csv", index=False)

    # --- original train rows
    O = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
    tr = O[O.split == "train"].copy()
    tr["origin"], tr["stratum"], tr["group"], tr["truncated"], tr["incl_prob"] = "original", "original", "original", False, np.nan
    new = d[d.accepted_valence | d.accepted_arousal].copy()
    new["origin"], new["split"] = "new", "train"
    keep = ["turn_id", "episode_id", "turn_idx", "dyad", "role", "gender", "word_count", "text", "prev_text", "split", "origin", "stratum", "group",
            "truncated", "incl_prob", "valence_maj", "arousal_maj", "valence_all3", "arousal_all3", "valence_A", "valence_B", "valence_C",
            "arousal_A", "arousal_B", "arousal_C"]
    S = pd.concat([tr[keep], new[keep + []]], ignore_index=True)

    # --- leakage checks
    eps_tr = set(O[O.split == "train"].episode_id)
    eps_other = set(O[O.split != "train"].episode_id)
    G = pd.read_csv("results/sentiment/gold_sheet_labeled.csv")
    chk = dict(new_turns=len(d), new_not_in_train_episodes=int((~d.episode_id.isin(eps_tr)).sum()),
               new_in_val_test_episodes=int(d.episode_id.isin(eps_other).sum()), new_in_gold_episodes=int(d.episode_id.isin(G.episode_id).sum()),
               new_text_dup_within=int(d.text_full.duplicated().sum()), new_text_in_original=int(d.text_full.isin(O.text).sum()),
               new_text_in_gold=int(d.text_full.isin(G.text).sum()),
               new_key_in_original=int(d.merge(O[["episode_id", "turn_idx"]], on=["episode_id", "turn_idx"]).shape[0]),
               duplicate_turn_ids=int(S.turn_id.duplicated().sum()))

    # --- nested subsets
    rng = np.random.default_rng(SEED)
    S["in_1783"] = S.origin.eq("original")
    new_idx = S.index[S.origin == "new"]
    need = TARGET_MID - int(S.in_1783.sum())
    strata = S.loc[new_idx].groupby(["group", "stratum"]).groups
    pick = []
    for _, idx in strata.items():
        k = int(round(need * len(idx) / len(new_idx)))
        pick += list(rng.choice(np.array(idx), size=min(k, len(idx)), replace=False))
    pick = pick[:need] if len(pick) > need else pick
    S["in_3000"] = S.in_1783 | S.index.isin(pick)
    S["in_5000"] = True
    assert (S.in_1783 <= S.in_3000).all() and (S.in_3000 <= S.in_5000).all()

    # --- dyad-restoring weights (target = original train mix)
    tgt = tr.dyad.value_counts(normalize=True)
    for s in ("1783", "3000", "5000"):
        m = S[f"in_{s}"]
        share = S[m].dyad.value_counts(normalize=True)
        w = (S.dyad.map(tgt) / S.dyad.map(share)).clip(0.5, 3.0)
        w = w / w[m].mean()
        S[f"w_dyad_{s}"] = np.where(m, w, np.nan)
    S.to_csv(f"{BASE}/scaling_train_sets.csv", index=False)

    # --- summary
    summ = dict(leakage=chk, passes=files and {p: len(f) for p, f in files.items()})
    summ["accepted_new"] = dict(valence=int(d.accepted_valence.sum()), arousal=int(d.accepted_arousal.sum()),
                                no_majority_valence=int((~d.accepted_valence).sum()), no_majority_arousal=int((~d.accepted_arousal).sum()))
    sets = {}
    for s in ("1783", "3000", "5000"):
        x = S[S[f"in_{s}"]]
        sets[s] = dict(rows=len(x), episodes=int(x.episode_id.nunique()),
                       valence=x.valence_maj.value_counts().to_dict(), arousal=x.arousal_maj.value_counts().to_dict(),
                       dyad_counts=x.dyad.value_counts().to_dict(), dyad_share=x.dyad.value_counts(normalize=True).round(3).to_dict(),
                       weighted_dyad_share=(x.groupby("dyad")[f"w_dyad_{s}"].sum() / x[f"w_dyad_{s}"].sum()).round(3).to_dict(),
                       truncated_rows=int(x.truncated.fillna(False).astype(bool).sum()),
                       max_rows_one_episode=int(x.groupby("episode_id").size().max()))
    summ["sets"] = sets
    summ["new_by_dyad"] = d.dyad.value_counts().to_dict()
    summ["new_valence"] = d.valence_maj.value_counts().to_dict()
    summ["new_arousal"] = d.arousal_maj.value_counts().to_dict()
    summ["new_valence_by_stratum"] = {f"{g}/{s}": x.valence_maj.value_counts().to_dict() for (g, s), x in d.groupby(["group", "stratum"])}
    summ["new_arousal_by_stratum"] = {f"{g}/{s}": x.arousal_maj.value_counts().to_dict() for (g, s), x in d.groupby(["group", "stratum"])}
    json.dump(summ, open(f"{BASE}/scaling_summary.json", "w"), indent=1, default=str)

    # --- manifest update
    man = json.load(open(f"{BASE}/manifest.json"))
    man["pass_files"] = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()
                         for f in sorted(glob.glob(f"{PASSES}/*.csv"))}
    man["accepted_new"] = summ["accepted_new"]
    man["nested_sets"] = {k: dict(rows=v["rows"]) for k, v in sets.items()}
    man["dyad_weighting"] = "w_dyad_<size> = original train dyad share / set dyad share, clipped [0.5,3], mean 1"
    json.dump(man, open(f"{BASE}/manifest.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
