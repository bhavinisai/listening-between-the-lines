#!/usr/bin/env python3
"""
Quality control for a batch of LLM labels from the scaling sample (results/sentiment/llm_labels3).

Checks: schema / parsing, row coverage, label distributions per pass and per stratum, pairwise and
Fleiss-style agreement, unanimity, ambiguous turns (no majority / 2-1 splits), agreement of the
majority label with the selection proxy (a sanity check, NOT ground truth), female-host long turns,
position effects across chunks, and (pilot) condensed-vs-full-text label agreement.

Usage:
    python src/sentiment/qc_scaling_labels.py --pattern pilot --out results/sentiment/llm_labels3/pilot_qc
    python src/sentiment/qc_scaling_labels.py --pattern main  --out results/sentiment/llm_labels3/main_qc
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = "results/sentiment/llm_labels3"
PASSES = "results/sentiment/llm_labels_passes/llm_labels3"
OK = dict(valence={"negative", "neutral", "positive"}, arousal={"low", "medium", "high"})


def load(pattern, p):
    fs = sorted(glob.glob(f"{PASSES}/{p}_pilot_*.csv" if pattern == "pilot" else f"{PASSES}/{p}_[0-9][0-9].csv"))
    return pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in fs]) if fs else pd.DataFrame()


def maj(r, cols):
    v = [r[c] for c in cols if isinstance(r[c], str)]
    top = max(set(v), key=v.count)
    return top if v.count(top) >= 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["pilot", "main"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(f"{BASE}/sample_meta.csv")
    meta = meta[meta.pilot] if a.pattern == "pilot" else meta[~meta.pilot]
    rep, d = {}, meta.set_index("turn_id").copy()
    problems = []
    for p in "ABC":
        L = load(a.pattern, p)
        if L.empty:
            problems.append(f"pass {p}: no files"); continue
        dup = int(L.turn_id.duplicated().sum()); missing = sorted(set(meta.turn_id) - set(L.turn_id)); extra = sorted(set(L.turn_id) - set(meta.turn_id))
        badv = L[~L.valence.isin(OK["valence"])]; bada = L[~L.arousal.isin(OK["arousal"])]
        if dup or missing or extra or len(badv) or len(bada):
            problems.append(f"pass {p}: dup={dup} missing={len(missing)} extra={len(extra)} bad_valence={len(badv)} bad_arousal={len(bada)}")
        L = L.drop_duplicates("turn_id").set_index("turn_id")
        d[f"valence_{p}"], d[f"arousal_{p}"] = L.valence, L.arousal
    rep["schema_problems"] = problems
    cols = {t: [f"{t}_{p}" for p in "ABC" if f"{t}_{p}" in d] for t in OK}
    for t in OK:
        d[f"{t}_maj"] = d.apply(lambda r: maj(r, cols[t]), axis=1)
        d[f"{t}_all3"] = d[cols[t]].nunique(axis=1).eq(1)
    rep["n"] = len(d)
    for t, c in cols.items():
        rep[t] = dict(
            label_dist_per_pass={x: d[x].value_counts(normalize=True).round(3).to_dict() for x in c},
            majority_dist=d[f"{t}_maj"].value_counts().to_dict(), no_majority=int(d[f"{t}_maj"].isna().sum()),
            unanimous=float(d[f"{t}_all3"].mean()),
            pairwise_kappa={f"{x[-1]}-{y[-1]}": float(cohen_kappa_score(d[x], d[y])) for i, x in enumerate(c) for y in c[i + 1:]},
            by_stratum_majority={f"{g}/{s}": x[f"{t}_maj"].value_counts().to_dict() for (g, s), x in d.groupby(["group", "stratum"])},
            unanimous_by_stratum={f"{g}/{s}": round(float(x[f"{t}_all3"].mean()), 3) for (g, s), x in d.groupby(["group", "stratum"])},
        )
    # selection-proxy sanity (the proxy is a model trained on earlier LLM labels, so this is NOT validity)
    pv = d.valence_maj
    rep["proxy_sanity"] = dict(
        neg_rate_in_N_stratum=float((pv[d.stratum == "N"] == "negative").mean()),
        neg_rate_in_R_stratum=float((pv[d.stratum == "R"] == "negative").mean()),
        low_rate_in_L_stratum=float((d.arousal_maj[d.stratum == "L"] == "low").mean()),
        low_rate_in_R_stratum=float((d.arousal_maj[d.stratum == "R"] == "low").mean()),
        high_rate_in_H_stratum=float((d.arousal_maj[d.stratum == "H"] == "high").mean()),
        high_rate_in_R_stratum=float((d.arousal_maj[d.stratum == "R"] == "high").mean()))
    # female-host long turns
    fh = d[d.group == "FH"]
    rep["female_host_long"] = dict(n=len(fh), valence_majority=fh.valence_maj.value_counts().to_dict(), arousal_majority=fh.arousal_maj.value_counts().to_dict(),
                                   unanimous_valence=float(fh.valence_all3.mean()), unanimous_arousal=float(fh.arousal_all3.mean()),
                                   truncated_unanimous_valence=float(fh[fh.truncated].valence_all3.mean()) if fh.truncated.any() else None)
    # condensed vs full text (pilot only)
    f = f"{PASSES}/F_pilot_fulltext.csv"
    if a.pattern == "pilot" and os.path.exists(f):
        F = pd.read_csv(f).set_index("turn_id")
        j = d.join(F[["valence", "arousal"]].rename(columns={"valence": "valence_full", "arousal": "arousal_full"}), how="inner")
        rep["condensed_vs_full"] = dict(n=len(j), valence_agree=float((j.valence_maj == j.valence_full).mean()),
                                        arousal_agree=float((j.arousal_maj == j.arousal_full).mean()),
                                        valence_disagreements={f"{k[0]}->{k[1]}": int(v) for k, v in j[j.valence_maj != j.valence_full][["valence_maj", "valence_full"]].value_counts().items()})
    # ambiguity list
    amb = d[(~d.valence_all3) | (~d.arousal_all3)].reset_index()
    amb[["turn_id", "group", "stratum", "dyad", "text", "valence_A", "valence_B", "valence_C", "arousal_A", "arousal_B", "arousal_C"]].to_csv(f"{a.out}/ambiguous.csv", index=False)
    rep["n_ambiguous_turns"] = len(amb)
    # position effect: label rate by chunk (drift across agents)
    d.reset_index().to_csv(f"{a.out}/labels_wide.csv", index=False)
    json.dump(rep, open(f"{a.out}/qc.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
