#!/usr/bin/env python3
"""
Evaluate the question-type classifier against the manually labeled gold sheet.

The gold sample is stratified on the classifier's *predicted* type (20
questions per type), so overall agreement is not population accuracy. This
script reports:

    1. precision per predicted type (the quantity the stratified sample
       estimates directly) with Wilson 95% intervals
    2. a confusion matrix (gold label x predicted type)
    3. recall / F1 per gold label, as a rough guide only: gold was not sampled
       by true label, so these are unweighted and small-n for rare classes
    4. a population-weighted accuracy: per-type precision weighted by how often
       the classifier predicts each type across the full corpus

Input:
    gold_work/gold_full.csv            sampled questions + predicted_type
    gold_work/gold_sheet_labeled.csv   gold_id, text, label (manual)
    results/question_type_analysis/question_classification.csv

Output (in --out_dir):
    gold_eval_precision_by_predicted.csv
    gold_eval_confusion_matrix.csv
    gold_eval_per_gold_label.csv
    gold_eval_summary.txt
"""

import argparse
import math
from pathlib import Path

import pandas as pd

CLASS_LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold_full", default="gold_work/gold_full.csv")
    ap.add_argument("--gold_labels", default="gold_work/gold_sheet_labeled.csv")
    ap.add_argument("--classified", default="results/question_type_analysis/question_classification.csv")
    ap.add_argument("--out_dir", default="results/question_type_analysis")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.gold_full)
    lab = pd.read_csv(args.gold_labels)[["gold_id", "label"]]
    df = full.merge(lab, on="gold_id", how="left")
    if df["label"].isna().any():
        raise SystemExit(f"{int(df['label'].isna().sum())} gold rows are unlabeled")
    df["label"] = df["label"].str.strip().str.lower()
    df["predicted_type"] = df["predicted_type"].str.strip().str.lower()
    df["match"] = df["label"] == df["predicted_type"]

    pop = pd.read_csv(args.classified, usecols=["predicted_type"])["predicted_type"].str.lower()
    pop_share = pop.value_counts(normalize=True)

    # 1. precision by predicted type
    rows = []
    for t in CLASS_LABELS:
        sub = df[df["predicted_type"] == t]
        k, n = int(sub["match"].sum()), len(sub)
        lo, hi = wilson(k, n)
        rows.append({
            "predicted_type": t, "n_gold": n, "n_agree": k,
            "precision": k / n if n else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "corpus_share": float(pop_share.get(t, 0.0)),
        })
    prec = pd.DataFrame(rows)
    prec.to_csv(out / "gold_eval_precision_by_predicted.csv", index=False)

    # 2. confusion matrix
    cm = pd.crosstab(
        df["label"].astype(pd.CategoricalDtype(CLASS_LABELS)),
        df["predicted_type"].astype(pd.CategoricalDtype(CLASS_LABELS)),
        rownames=["gold"], colnames=["predicted"], dropna=False,
    )
    cm.to_csv(out / "gold_eval_confusion_matrix.csv")

    # 3. per gold label (unweighted, indicative only)
    rows = []
    for t in CLASS_LABELS:
        g = df[df["label"] == t]
        tp = int((g["predicted_type"] == t).sum())
        pred_n = int((df["predicted_type"] == t).sum())
        r = tp / len(g) if len(g) else float("nan")
        p = tp / pred_n if pred_n else float("nan")
        f1 = 2 * p * r / (p + r) if p and r and not math.isnan(p + r) else float("nan")
        rows.append({"gold_label": t, "n_gold": len(g), "tp": tp, "recall": r, "precision": p, "f1": f1})
    per = pd.DataFrame(rows)
    per.to_csv(out / "gold_eval_per_gold_label.csv", index=False)

    # 4. population-weighted accuracy
    weighted = float((prec["precision"] * prec["corpus_share"]).sum() / prec["corpus_share"].sum())
    raw = float(df["match"].mean())

    lines = [
        f"Gold questions: {len(df)}",
        f"Raw agreement (equal weight per predicted type): {raw:.3f}",
        f"Population-weighted accuracy (precision x corpus predicted share): {weighted:.3f}",
        "",
        "Precision by predicted type (Wilson 95% CI):",
    ]
    for _, r in prec.iterrows():
        lines.append(
            f"  {r.predicted_type:<13} {int(r.n_agree):>2}/{int(r.n_gold):<2} "
            f"{r.precision:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]  corpus share {r.corpus_share:.3f}"
        )
    lines += ["", "Confusion matrix (rows = gold, cols = predicted):", cm.to_string(), "",
              "Per gold label (unweighted; gold not sampled by true label):", per.to_string(index=False)]
    text = "\n".join(lines)
    (out / "gold_eval_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
