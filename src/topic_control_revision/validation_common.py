"""
Shared loading/scoring helpers for the topic-control human validation
(scripts 05-07). These scripts only read labels that human annotators typed
into the sheets; they never create labels.
"""

import os

import numpy as np
import pandas as pd

from common import SEED, VALID

Q1_CODES = {"NEW": 1, "SHIFT": 1, "SAME": 0}          # UNCLEAR / NA -> excluded
Q1_STRICT = {"NEW": 1, "SHIFT": 0, "SAME": 0}
Q2_CODES = {"ENGAGE": 1, "MINIMAL": 0, "NONE": 0}     # UNCLEAR / NA -> excluded
VALID_Q1 = {"NEW", "SHIFT", "SAME", "UNCLEAR", "NA"}
VALID_Q2 = {"ENGAGE", "MINIMAL", "NONE", "UNCLEAR", "NA"}


def vdir(base=None):
    return base or VALID


def read_sheet(path, k):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for q, ok in [("Q1_new_topic", VALID_Q1), ("Q2_engagement", VALID_Q2)]:
        df[q] = df[q].str.strip().str.upper()
        bad = df[(df[q] != "") & (~df[q].isin(ok))]
        if len(bad):
            raise SystemExit(f"annotator {k}: invalid {q} codes {sorted(bad[q].unique())} "
                             f"(allowed: {sorted(ok)})")
    return df[["item_id", "Q1_new_topic", "Q2_engagement", "confidence_1to3"]].rename(
        columns={"Q1_new_topic": f"a{k}_Q1", "Q2_engagement": f"a{k}_Q2", "confidence_1to3": f"a{k}_conf"})


def load_annotations(base=None, require_complete=True):
    d = vdir(base)
    a1 = read_sheet(os.path.join(d, "annotator_1_sheet.csv"), 1)
    a2 = read_sheet(os.path.join(d, "annotator_2_sheet.csv"), 2)
    man = pd.read_csv(os.path.join(d, "manifest", "validation_manifest.csv"))
    df = man.merge(a1, on="item_id").merge(a2, on="item_id")
    done = (df[["a1_Q1", "a1_Q2", "a2_Q1", "a2_Q2"]] != "").all(axis=1)
    if not done.any():
        raise SystemExit("No completed annotations found yet. Both annotators must fill "
                         "Q1_new_topic and Q2_engagement before these scripts can run.")
    if require_complete and not done.all():
        print(f"WARNING: {(~done).sum()} of {len(df)} items are not yet labeled by both "
              "annotators; they are excluded.")
    return df[done].copy()


def final_labels(df, base=None):
    """Agreed label, else the adjudicated label; unresolved disagreements -> NaN."""
    d = vdir(base)
    adj_path = os.path.join(d, "adjudication_completed.csv")
    adj = (pd.read_csv(adj_path, dtype=str, keep_default_na=False).set_index("item_id")
           if os.path.exists(adj_path) else pd.DataFrame())
    for q in ["Q1", "Q2"]:
        agree = df[f"a1_{q}"] == df[f"a2_{q}"]
        final = df[f"a1_{q}"].where(agree, "")
        if len(adj) and f"final_{q}" in adj:
            fill = df["item_id"].map(adj[f"final_{q}"].str.strip().str.upper())
            final = final.where(agree, fill.fillna(""))
        df[f"final_{q}"] = final
    df["human_new_topic"] = df["final_Q1"].map(Q1_CODES)
    df["human_new_topic_strict"] = df["final_Q1"].map(Q1_STRICT)
    df["human_engaged"] = df["final_Q2"].map(Q2_CODES)
    return df


def strat_boot(df, stat, n=2000, seed=SEED):
    """Percentile CI by resampling items within each sampling stratum."""
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in df.groupby("stratum")]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        v = stat(df.loc[idx])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x)
    return np.sum(x[ok] * w[ok]) / np.sum(w[ok]) if ok.any() else np.nan
