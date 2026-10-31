#!/usr/bin/env python3
"""
Valence stratum ablation, step 1 (CPU): preprocessing audit + subset construction.

Condition A: original 1,783 + targeted-stratum new turns (strata L, H, N, P).
Condition B: original 1,783 + random-stratum new turns (stratum R).
Female-host long turns (FH_long) are excluded from BOTH so the comparison is not confounded by truncated text.
B uses every random-stratum turn with a valence majority label. A is drawn from the targeted pool with the same
size, the same dyad counts and (as far as the pool allows) the same per-episode counts as B.

Writes results/sentiment/llm_scaling_stratum/: subsets.csv, composition.json, preprocessing_audit.json.
Original files are only read.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = "results/sentiment/llm_labels3"
OUT = "results/sentiment/llm_scaling_stratum"
SEED = 43
os.makedirs(OUT, exist_ok=True)

S = pd.read_csv(f"{BASE}/scaling_train_sets.csv")
O = pd.read_csv("results/sentiment/llm_ft2/labels_majority_split.csv")
log = {(r["task"], r["size"], r["seed"]): r for r in json.load(open("results/sentiment/llm_scaling_exp/run_log.json"))}

# ------------------------------------------------------------------ audit
orig = S[S.origin == "original"]
chk = orig.merge(O[O.split == "train"][["turn_id", "text", "prev_text"]], on="turn_id", suffixes=("", "_o"))
new = S[S.origin == "new"]
audit = dict(
    valence_uses_previous_turn=False,
    previous_turn_cap_affects_valence=False,
    note_prev_cap="scaling run capped previous-turn text to its last 100 words for all rows; the valence variant is no-context so prev_text is never read",
    original_rows_text_identical_to_baseline_file=bool((chk.text == chk.text_o).all()),
    original_rows_with_prev_text_over_100_words_in_baseline=int(O[O.split == "train"].prev_text.fillna("").str.split().str.len().gt(100).sum()),
    new_rows_truncated_total=int(new.truncated.astype(bool).sum()),
    new_rows_truncated_by_stratum=new[new.truncated.astype(bool)].stratum.value_counts().to_dict(),
    settings_identical_to_baseline=dict(base="cardiff_xlmr_multi", lr=2e-5, epochs=4, batch=16, max_len=256, weight_pow=0.5,
                                        optimizer="AdamW wd=0.01, 10% warmup, grad-clip 1.0, bf16 autocast", variant_valence="no_context"),
    differences_vs_baseline=[
        "checkpoint = best-validation-macro-F1 epoch (baseline llm_pipeline.py used the fixed last epoch)",
        "3 seeds (0,1,2) per condition (baseline: seed 0 only)",
        "training rows ordered original-then-new in the DataFrame (the sampler shuffles each epoch, so order does not matter)",
        "baseline file llm_ft2 trained on 1,782 valence rows; scaling 1,783-set also 1,782 rows (1 row had no valence majority)",
        "class weights are recomputed from each training set's own class counts (same rule as baseline)",
    ],
    scaling_vs_baseline_val_macro_f1_seed0=dict(
        baseline_llm_ft2_last_epoch=0.7703, scaling_1783_seed0_last_epoch=log[("valence", 1783, 0)]["val_final_epoch"]["macro_f1"],
        scaling_1783_seed0_best_epoch=log[("valence", 1783, 0)]["val_best"]["macro_f1"]),
    evaluation="256 validation turns / 18 episodes (valence majority labels); test and human gold are not read for evaluation",
)
json.dump(audit, open(f"{OUT}/preprocessing_audit.json", "w"), indent=1, default=str)

# ------------------------------------------------------------------ subsets
N = new[new.valence_maj.notna()].copy()
B = N[N.stratum == "R"].copy()
pool = N[N.stratum.isin(["L", "H", "N", "P"])].copy()
rng = np.random.default_rng(SEED)

picked = []
ep_counts = B.groupby("episode_id").size()
for ep, c in ep_counts.items():                       # per-episode quota = B's count in that episode
    avail = pool[pool.episode_id == ep]
    take = min(c, len(avail))
    if take:
        picked += list(rng.choice(avail.index.values, size=take, replace=False))
taken = set(picked)
short = len(B) - len(picked)
for dyad, c in B.dyad.value_counts().items():         # fill any shortfall from the same dyad, keeping dyad counts equal to B's
    have = sum(pool.loc[i, "dyad"] == dyad for i in picked)
    need = c - have
    if need > 0:
        rest = pool[(pool.dyad == dyad) & ~pool.index.isin(taken)]
        extra = rng.choice(rest.index.values, size=min(need, len(rest)), replace=False)
        picked += list(extra); taken |= set(extra)
A = pool.loc[picked].copy()

rows = [S[S.origin == "original"].assign(condition="base")]
rows += [A.assign(condition="A_targeted"), B.assign(condition="B_random")]
sub = pd.concat(rows, ignore_index=True)
sub["valence_maj"] = sub.valence_maj
sub.to_csv(f"{OUT}/subsets.csv", index=False)


def comp(x):
    return dict(rows=len(x), episodes=int(x.episode_id.nunique()), dyad=x.dyad.value_counts().to_dict(),
                valence=x.valence_maj.value_counts().to_dict(),
                valence_share=x.valence_maj.value_counts(normalize=True).round(3).to_dict(),
                arousal=x.arousal_maj.value_counts().to_dict(), strata=x.stratum.value_counts().to_dict(),
                truncated=int(x.truncated.astype(bool).sum()), mean_words=round(float(x.text.str.split().str.len().mean()), 1),
                max_rows_one_episode=int(x.groupby("episode_id").size().max()))


ea, eb = A.groupby("episode_id").size(), B.groupby("episode_id").size()
eps = sorted(set(ea.index) | set(eb.index))
l1 = float(sum(abs(ea.get(e, 0) - eb.get(e, 0)) for e in eps) / 2)
G = pd.read_csv("results/sentiment/gold_sheet_labeled.csv", usecols=["episode_id", "text"])
V = O[O.split == "val"]
report = dict(
    new_new_valence_labeled=len(N), excluded_FH_long=int((N.stratum == "FH_long").sum()), A=comp(A), B=comp(B),
    episode_count_mismatch_turns=l1, episode_count_mismatch_share=round(l1 / len(B), 4),
    A_B_overlap_turns=len(set(A.turn_id) & set(B.turn_id)), A_in_B_episodes_only=bool(set(A.episode_id) <= set(B.episode_id)),
    leakage=dict(new_in_val_episodes=int(sub.episode_id.isin(V.episode_id).sum()),
                 new_in_test_episodes=int(sub.episode_id.isin(O[O.split == "test"].episode_id).sum()),
                 new_in_gold_episodes=int(sub.episode_id.isin(G.episode_id).sum()),
                 text_in_val=int(sub.text.isin(V.text).sum()), duplicate_turn_ids_within_condition=int(
                     sub.groupby("condition").turn_id.apply(lambda s: s.duplicated().sum()).sum())),
    sizes={k: int((sub.condition == k).sum()) for k in ("base", "A_targeted", "B_random")},
    A_shortfall_filled_from_other_episodes=short)
json.dump(report, open(f"{OUT}/composition.json", "w"), indent=1, default=str)
print(json.dumps(audit, indent=1, default=str)); print(json.dumps(report, indent=1, default=str))
