#!/usr/bin/env python3
"""
Independent audit of the sentiment figures. Re-derives every plotted number from the raw saved files (validation probabilities,
labels, run logs, training-label table, bootstrap CSVs) and compares it with the plotted-data tables, then checks file formats,
naming, size and caption wording. Writes results/sentiment/figures/quality_audit.md.

    python src/sentiment/figures/verify_figures.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, "src/sentiment")
R = "results/sentiment"
SC, ST, EN, AU, L3 = f"{R}/llm_scaling_exp", f"{R}/llm_scaling_stratum", f"{R}/llm_ensemble_check", f"{R}/audit", f"{R}/llm_labels3"
OUT = f"{R}/figures"
DATA = f"{OUT}/data"
CLS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
EXPECTED = ["performance_by_training_size", "valence_confusion_matrix", "arousal_confusion_matrix", "valence_data_comparison", "valence_ablation_results", "per_class_performance", "training_label_distribution",
            "single_model_vs_ensemble", "performance_by_gender_dyad", "per_episode_accuracy_change", "earlier_models_test_vs_human_gold"]
checks = []


def ok(name, cond, detail=""):
    checks.append((bool(cond), name, detail))


def close(a, b, tol=1e-9):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


log_sc = {(r["task"], r["size"], r["seed"]): r for r in json.load(open(f"{SC}/run_log.json"))}
log_st = {(r["cond"], r["seed"]): r for r in json.load(open(f"{ST}/run_log.json"))}
mf = lambda y, p: f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)
probs = lambda t, size, seed: np.load(f"{SC}/valprobs_{t}_{size}_seed{seed}.npy")[log_sc[(t, size, seed)]["best_epoch"] - 1]
RET = {"valence": (1783, 2), "arousal": (5000, 2)}

# ---- inputs
try:
    from llm_pipeline import CLASSES as TRAIN_CLASSES
    ok("class order identical to training code (llm_pipeline.CLASSES)", TRAIN_CLASSES == CLS)
except Exception as e:                                       # pragma: no cover
    ok("class order identical to training code", False, repr(e))
lab = {t: pd.read_csv(f"{SC}/val_labels_{t}.csv") for t in CLS}
for t in CLS:
    ok(f"{t}: validation labels = 256 turns / 18 episodes", len(lab[t]) == 256 and lab[t].episode_id.nunique() == 18)
ok("valence validation labels identical across scaling and ablation experiments", lab["valence"].equals(pd.read_csv(f"{ST}/val_labels_valence.csv")))
O = pd.read_csv(f"{R}/llm_ft2/labels_majority_split.csv")
V = O[O.split == "val"]
ok("validation turn_ids = original validation split (no test / gold rows)",
   set(lab["valence"].turn_id) == set(V[V.valence_maj.notna()].turn_id) and set(lab["arousal"].turn_id) == set(V[V.arousal_maj.notna()].turn_id))
dev = json.load(open(f"{SC}/development_selected.json"))
for t, (s, sd) in RET.items():
    ok(f"{t}: retained checkpoint = {t}_{s}_seed{sd} and exists", dev[t]["checkpoint"].endswith(f"{t}_{s}_seed{sd}") and os.path.exists(f"models/llm_scaling/{t}_{s}_seed{sd}/model.safetensors"))
    ok(f"{t}: recomputed val macro-F1 of retained checkpoint = logged value", close(mf(lab[t].y, probs(t, s, sd).argmax(1)), log_sc[(t, s, sd)]["val_best"]["macro_f1"]),
       f"{mf(lab[t].y, probs(t, s, sd).argmax(1)):.4f}")


# ---- confusion matrices
for t, (s_, sd) in RET.items():
    c = pd.read_csv(f"{DATA}/{t}_confusion_matrix.csv")
    cmx = confusion_matrix(lab[t].y, probs(t, s_, sd).argmax(1), labels=[0, 1, 2])
    saved = pd.read_csv(f"{EN}/confusion_{t}_single.csv", index_col=0).values
    plotted = np.array([[c[(c.true_label.str.lower() == a) & (c.predicted_label.str.lower() == b)]["count"].iloc[0] for b in CLS[t]] for a in CLS[t]])
    ok(f"{t}_confusion_matrix: plotted counts = recomputed confusion = saved llm_ensemble_check confusion", (plotted == cmx).all() and (cmx == saved).all())
    ok(f"{t}_confusion_matrix: row sums = class supports {np.bincount(lab[t].y).tolist()}; row percentages sum to 100",
       (plotted.sum(1) == np.bincount(lab[t].y)).all() and close(c.groupby("true_label").row_percent.sum().values, 100, 1e-6))

# ---- performance_by_training_size
d = pd.read_csv(f"{DATA}/performance_by_training_size.csv")
good = all(close(d[(d.task == t) & (d["size"] == n) & (d.seed == s)].val_macro_f1.iloc[0], mf(lab[t].y, probs(t, n, s).argmax(1)))
           for t in CLS for n in (1783, 3000, 5000) for s in (0, 1, 2))
S = pd.read_csv(f"{SC}/summary_by_size.csv")
for t in CLS:
    for n in (1783, 3000, 5000):
        v = d[(d.task == t) & (d["size"] == n)].val_macro_f1.values
        r = S[(S.task == t) & (S["size"] == n)].iloc[0]
        good &= close(v.mean(), r.macro_f1_mean) and close(v.std(ddof=1), r.macro_f1_sd)
ok("performance_by_training_size: 18 per-seed scores recomputed; mean / SD (ddof=1) match summary_by_size.csv", good)

# ---- per_class_performance
c5, comp = pd.read_csv(f"{DATA}/per_class_performance.csv"), pd.read_csv(f"{EN}/comparison.csv")
for t, (s, sd) in RET.items():
    y, p = lab[t].y.values, probs(t, s, sd).argmax(1)
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    x = c5[c5.task == t].reset_index(drop=True)
    c = comp[(comp.task == t) & (comp.model == "selected single")].iloc[0]
    ok(f"per_class_performance {t}: P/R/F1/support = recomputed = comparison.csv; classes in order {CLS[t]}",
       [k.lower() for k in x.cls] == CLS[t] and all(close(x.precision[i], pr[i]) and close(x.recall[i], rc[i]) and close(x.f1[i], f[i]) and int(x.support[i]) == sup[i]
                                                    and close(c[f"{k}_P"], pr[i]) and close(c[f"{k}_F1"], f[i]) for i, k in enumerate(CLS[t])))
    ok(f"per_class_performance {t}: confusion row sums = supports {np.bincount(y).tolist()}", (confusion_matrix(y, p, labels=[0, 1, 2]).sum(1) == sup).all())

# ---- valence_data_comparison / valence_ablation_results
a3, sm = pd.read_csv(f"{DATA}/valence_data_comparison.csv"), pd.read_csv(f"{ST}/summary.csv").set_index("condition")
yv, good = lab["valence"].y.values, True
for cond in ("base", "A_targeted", "B_random"):
    for seed in range(5):
        pb = probs("valence", 1783, seed) if (cond == "base" and seed < 3) else np.load(f"{ST}/valprobs_{cond}_seed{seed}.npy")[log_st[(cond, seed)]["best_epoch"] - 1]
        rc = precision_recall_fscore_support(yv, pb.argmax(1), labels=[0, 1, 2], zero_division=0)[1][0]
        r = a3[(a3.condition == cond) & (a3.seed == seed)].iloc[0]
        good &= close(r.macro_f1, mf(yv, pb.argmax(1))) and close(r.neg_recall, rc)
    v = a3[a3.condition == cond]
    good &= close(v.macro_f1.mean(), sm.loc[cond, "macro_f1_mean"]) and close(v.macro_f1.std(ddof=1), sm.loc[cond, "macro_f1_sd"])
    good &= close(v.neg_recall.mean(), sm.loc[cond, "neg_r_mean"]) and close(v.neg_recall.std(ddof=1), sm.loc[cond, "neg_r_sd"])
ok("valence_data_comparison: 15 per-seed scores recomputed; means / SD match ablation summary.csv", good)
ok("valence_data_comparison: 5 matched seeds (0-4) per condition", all(sorted(a3[a3.condition == c].seed) == [0, 1, 2, 3, 4] for c in ("base", "A_targeted", "B_random")))
ok("valence_ablation_results: plotted bootstrap differences identical to saved bootstrap_differences.csv", pd.read_csv(f"{DATA}/valence_ablation_results.csv").equals(pd.read_csv(f"{ST}/bootstrap_differences.csv")))

# ---- training_label_distribution
S6, f6 = pd.read_csv(f"{L3}/scaling_train_sets.csv"), pd.read_csv(f"{DATA}/training_label_distribution.csv")
ok("training sets nested with 1,783 / 3,000 / 5,000 rows", (S6.in_1783 <= S6.in_3000).all() and (S6.in_3000 <= S6.in_5000).all() and [int(S6[f"in_{n}"].sum()) for n in (1783, 3000, 5000)] == [1783, 3000, 5000])
good = True
for t in CLS:
    for n in (1783, 3000, 5000):
        x = S6[S6[f"in_{n}"]][f"{t}_maj"]
        nl = int(x.notna().sum())
        for c in CLS[t]:
            r = f6[(f6.task == t) & (f6["size"] == n) & (f6.cls.str.lower() == c)].iloc[0]
            good &= int(r["count"]) == int((x == c).sum()) and int(r.labeled_rows) == nl and int(r.rows_without_majority) == n - nl
        good &= int(f6[(f6.task == t) & (f6["size"] == n)]["count"].sum()) == nl
ok("training_label_distribution: counts, denominators and rows without majority label recomputed from scaling_train_sets.csv", good)
dv = json.load(open(f"{SC}/data_verification.json"))["sets"]
ok("training_label_distribution: denominators match data_verification.json", all(int(f6[(f6.task == t) & (f6["size"] == n)].labeled_rows.iloc[0]) == dv[str(n)][f"{t}_rows"] for t in CLS for n in (1783, 3000, 5000)))

# ---- single_model_vs_ensemble
a7, b7 = pd.read_csv(f"{DATA}/single_model_vs_ensemble_macro_f1.csv"), pd.read_csv(f"{DATA}/single_model_vs_ensemble_differences.csv")
ok("single_model_vs_ensemble: macro-F1 bars = comparison.csv", all(close(r.macro_f1, comp[(comp.task == r.task) & (comp.model == r.model)].macro_f1.iloc[0]) for r in a7.itertuples()))
good = True
for t in CLS:
    bt = pd.read_csv(f"{EN}/bootstrap_{t}.csv")
    for ref in ("selected single", "mean single"):
        s = bt[(bt.ensemble_minus == ref) & (bt.metric == "macro_f1")].iloc[0]
        r = b7[(b7.task == t) & (b7.ensemble_minus == ref) & (b7.metric == "macro_f1")].iloc[0]
        good &= close(s["diff"], r["diff"]) and close(s.ci_lo, r.ci_lo) and close(s.ci_hi, r.ci_hi)
ok("single_model_vs_ensemble: plotted differences / CI = saved bootstrap_{task}.csv", good)
dec = json.load(open(f"{EN}/decision.json"))
ok("single_model_vs_ensemble: neither ensemble selected (decision.json)", not dec["valence"]["adopt_ensemble"] and not dec["arousal"]["adopt_ensemble"])

# ---- per_episode_accuracy_change
s1, stab = pd.read_csv(f"{DATA}/per_episode_accuracy_change.csv"), {r["task"]: r for r in json.load(open(f"{SC}/stability.json"))}
good = True
for t in CLS:
    x = s1[s1.task == t]
    good &= len(x) == 18 and int(x.n_turns.sum()) == 256 and close(x.delta, x.acc_5000 - x.acc_1783)
    good &= int((x.delta > 1e-9).sum()) == stab[t]["episodes_improved"] and int((x.delta < -1e-9).sum()) == stab[t]["episodes_worse"]
ok("per_episode_accuracy_change: 18 episodes / 256 turns per task; delta = acc_5000 - acc_1783; improved/worse counts match stability.json", good)

# ---- performance_by_gender_dyad (metric convention)
dy = pd.read_csv(f"{DATA}/performance_by_gender_dyad.csv")
good_val, good_flag, good_nan, good_n = True, True, True, True
for t, (s, sd) in RET.items():
    p = probs(t, s, sd).argmax(1)
    x = dy[dy.task == t]
    good_n &= int(x.n.sum()) == 256
    for r in x.itertuples():
        m = (lab[t].dyad == r.dyad).values
        y = lab[t].y.values[m]
        cnt = np.bincount(y, minlength=3)
        good_n &= int(r.n) == int(m.sum()) and r.class_counts == str(cnt.tolist())
        if (cnt == 0).any():
            good_nan &= bool(np.isnan(r.macro_f1_3class)) and r.status.startswith("undefined")
        else:
            good_val &= close(r.macro_f1_3class, mf(y, p[m]))
            good_flag &= r.status.startswith("unreliable") == bool((cnt < 5).any())
ok("performance_by_gender_dyad: n and class counts recomputed from validation labels; n sums to 256 per task", good_n)
ok("performance_by_gender_dyad: plotted values = macro-F1 over ALL three classes (same class definition in every subgroup)", good_val)
ok("performance_by_gender_dyad: subgroups with an absent class are NaN / 'undefined' (no value fabricated); classes with <5 turns flagged unreliable", good_nan and good_flag)
ok("performance_by_gender_dyad: undefined subgroup(s) = " + ", ".join(f"{r.task} {r.dyad}" for r in dy[dy.status.str.startswith("undefined")].itertuples()), True)

# ---- earlier_models_test_vs_human_gold
s3, M = pd.read_csv(f"{DATA}/earlier_models_test_vs_human_gold.csv"), pd.read_csv(f"{AU}/metrics_full.csv")
ok("earlier_models_test_vs_human_gold: values = saved audit/metrics_full.csv (historical; read only)",
   all(close(r.macro_f1, M[(M.task == r.task) & (M.eval_set == r.eval_set)].macro_f1.iloc[0]) for r in s3.itertuples()))

# ---- files, naming, resolution, size
pngs = sorted(glob.glob(f"{OUT}/*.png"))
stems = [os.path.basename(f)[:-4] for f in pngs]
ok("exactly the 11 expected PNG figures (incl. valence and arousal confusion matrices)", sorted(stems) == sorted(EXPECTED), ", ".join(stems))
ok("all figure filenames are snake_case", all(re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", s) for s in stems))
ok("no PDF / SVG files in the figure directory", not glob.glob(f"{OUT}/*.pdf") and not glob.glob(f"{OUT}/*.svg") and not glob.glob(f"{OUT}/**/*.pdf", recursive=True) and not glob.glob(f"{OUT}/**/*.svg", recursive=True))
dpis, widths = [], []
for f in pngs:
    im = Image.open(f)
    dpis.append(round(im.info.get("dpi", (0, 0))[0])); widths.append(im.size[0] / 300)
ok("all PNGs saved at 300 dpi", all(x >= 299 for x in dpis), str(sorted(set(dpis))))
conf_w = {os.path.basename(f)[:-4]: im_w for f, im_w in zip(pngs, widths)}
ok("all multi-panel figures are 6.5 inches wide; the two single-panel confusion matrices share one size (4.5 in)",
   all(abs(w - (4.5 if "confusion" in k else 6.5)) < 0.02 for k, w in conf_w.items()), str(sorted({round(w, 2) for w in widths})))
hits = [f for f in glob.glob("src/sentiment/figures/*.py") + glob.glob(f"{OUT}/*.md") if "report2" in open(f).read().lower() and not f.endswith("verify_figures.py")]
ok("no legacy report-folder naming in the figure scripts or outputs; the legacy source and results folders no longer exist",
   not hits and not os.path.exists("src/report2") and not os.path.exists("results/report2"), ", ".join(hits))

# ---- captions
cap = f"{OUT}/captions.md"
if os.path.exists(cap):
    txt = open(cap).read()
    low = txt.lower()
    ok("captions: one caption per figure (11)", len(re.findall(r"^\*\*Figure \d+\.", txt, flags=re.M)) == 11)
    ok("captions contain no 'significant' / p-value claims", "significan" not in low and "p <" not in low and "p<" not in low)
    ok("captions state the evaluation split for every figure", len(re.findall(r"validation set|training set|training sets|held-out", low)) >= 11)

n_pass = sum(c for c, *_ in checks)
lines = ["# Sentiment figures — quality audit", "", f"Independent recomputation from raw saved files: **{n_pass}/{len(checks)} checks passed**.", "",
         "Title hygiene (no checkpoint names, seed IDs, directory names or methodological text) is enforced inside `make_figures.py` (every title is asserted against a forbidden-pattern list before saving).", ""]
lines += [f"- {'PASS' if c else '**FAIL**'} — {n}" + (f" ({dd})" if dd else "") for c, n, dd in checks]
open(f"{OUT}/quality_audit.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if n_pass == len(checks) else 1)
