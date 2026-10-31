#!/usr/bin/env python3
"""
Validation-only macro-F1 optimisation of the LLM-label models, from the per-seed probabilities
written by llm_seed_predictions.py. Nothing here retrains a model.

Experiments (all selected on the validation split ONLY):
    E0  seed-0 argmax (reproduces llm_pipeline.py)
    E1  mean probability over seeds (per variant)
    E2  mean probability over seeds and both variants (context + no-context)
    E3  prior correction  p / prior^alpha  (1 parameter)
    E4  class-specific logit biases on the minority classes (2 parameters)

Overfitting guards (validation has only ~18 episodes): biases are chosen by the mean macro-F1
over episode-cluster bootstrap resamples (not the single noisy argmax), parameters live on a
coarse grid, and every tuned step is scored by nested leave-episodes-out CV inside validation.
A tuned step is kept only if its nested-CV gain is positive.

Test and human-gold are NOT used for any choice; they are scored once at the end for the
single chosen configuration and written to a separate file.

Usage:
    python src/sentiment/tune_macro_f1.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

IN, OUT = "results/sentiment/llm_ft2_seeds", "results/sentiment/tuning"
CLS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
SEEDS, VARIANTS = [0, 1, 2], ["no_context", "context"]
GRID = np.arange(-1.5, 1.51, 0.25)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def load(task, variant, split):
    ps = [pd.read_csv(f"{IN}/{task}_{variant}_seed{s}_{split}.csv") for s in SEEDS]
    return ps[0][[c for c in ps[0].columns if not c.startswith("p")]], np.stack([p[["p0", "p1", "p2"]].values for p in ps])


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2])


def apply(P, bias):
    return (np.log(np.clip(P, 1e-9, 1)) + bias).argmax(1)


def boot_idx(groups, B):
    """Episode-cluster bootstrap as a (B, n) matrix of per-row resample counts."""
    eps = np.unique(groups)
    pos = {e: i for i, e in enumerate(eps)}
    col = np.array([pos[g] for g in groups])
    W = np.zeros((B, len(groups)))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps))[col]
    return W


def boot_mf1(W, y, pr):
    """Mean macro-F1 over bootstrap weight rows W for predictions pr (classes 0..2, all present)."""
    f = []
    for c in range(3):
        tp = W @ ((pr == c) & (y == c)); fp = W @ ((pr == c) & (y != c)); fn = W @ ((pr != c) & (y == c))
        f.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), 0.0))
    return float(np.mean(np.mean(f, 0)))


def tune_bias(P, y, groups, B=150):
    """Grid-search the two minority-class logit biases on bootstrap-averaged macro-F1."""
    bs = boot_idx(groups, B)
    best, best_b = -1, np.zeros(3)
    for b0 in GRID:
        for b2 in GRID:
            b = np.array([b0, 0.0, b2])
            pr = apply(P, b)
            s = boot_mf1(bs, y, pr)
            if s > best + 1e-9 or (abs(s - best) <= 1e-9 and abs(b0) + abs(b2) < np.abs(best_b).sum()):
                best, best_b = s, b
    return best_b, best


def tune_alpha(P, y, groups, prior, B=150):
    bs = boot_idx(groups, B)
    best, best_a = -1, 0.0
    for a in np.arange(0, 1.01, 0.1):
        pr = apply(P, -a * np.log(prior))
        s = boot_mf1(bs, y, pr)
        if s > best + 1e-9:
            best, best_a = s, a
    return best_a, best


def nested_cv(P, y, groups, fit, make_bias, folds=6):
    """Out-of-fold macro-F1 when the tuning is fit on the other validation episodes."""
    oof = np.zeros(len(y), int)
    for tr, te in GroupKFold(folds).split(P, y, groups):
        oof[te] = apply(P[te], make_bias(fit(P[tr], y[tr], groups[tr])))
    return mf1(y, oof), oof


def per_episode_gain(y, groups, base, new):
    g = []
    for e in np.unique(groups):
        m = groups == e
        g.append(mf1(y[m], new[m]) - mf1(y[m], base[m]))
    g = np.array(g)
    return int((g > 0).sum()), int((g < 0).sum()), int((g == 0).sum())


def report(y, p, cls):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    r = dict(acc=accuracy_score(y, p), macro_f1=mf1(y, p), weighted_f1=f1_score(y, p, average="weighted"))
    for i, c in enumerate(cls):
        r.update({f"{c}_P": pr[i], f"{c}_R": rc[i], f"{c}_F1": f[i], f"{c}_n": int(sup[i])})
    return r


rows, chosen, finals = [], {}, []
for task, cls in CLS.items():
    meta, _ = load(task, VARIANTS[0], "val")
    y, groups = meta.y.values, meta.episode_id.values
    prior = np.bincount(y, minlength=3) / len(y)
    V, T, G = {}, {}, {}
    for v in VARIANTS:
        for name, d in (("val", V), ("test", T), ("gold", G)):
            m, P = load(task, v, name)
            d[v] = (m, P)
    cand = {}                                   # name -> (val probs, test probs, gold probs)
    for v in VARIANTS:
        cand[f"E0 {v} seed0"] = tuple(D[v][1][0] for D in (V, T, G))
        cand[f"E1 {v} seed-mean"] = tuple(D[v][1].mean(0) for D in (V, T, G))
    cand["E2 both variants x seeds"] = tuple(np.mean([D[v][1].mean(0) for v in VARIANTS], 0) for D in (V, T, G))

    base_name = f"E0 {'no_context' if task == 'valence' else 'context'} seed0"   # variant llm_pipeline.py chose
    base_pred = cand[base_name][0].argmax(1)
    base_f1 = mf1(y, base_pred)

    stage = []
    for name, (Pv, Pt, Pg) in cand.items():
        t0 = time.time()
        stage.append((name, Pv, Pt, Pg, np.zeros(3), time.time() - t0))
        # E3 prior correction
        t0 = time.time()
        a, _ = tune_alpha(Pv, y, groups, prior)
        cv, _ = nested_cv(Pv, y, groups, lambda P, yy, gg: tune_alpha(P, yy, gg, prior)[0],
                          lambda aa: -aa * np.log(prior))
        stage.append((name + f" + E3 prior a={a:.1f}", Pv, Pt, Pg, -a * np.log(prior), time.time() - t0, cv))
        # E4 minority logit biases
        t0 = time.time()
        b, _ = tune_bias(Pv, y, groups)
        cv, _ = nested_cv(Pv, y, groups, lambda P, yy, gg: tune_bias(P, yy, gg, B=60)[0], lambda bb: bb)
        stage.append((name + f" + E4 bias=({b[0]:+.2f},{b[2]:+.2f})", Pv, Pt, Pg, b, time.time() - t0, cv))

    for s in stage:
        name, Pv, Pt, Pg, b, rt = s[:6]
        nested = s[6] if len(s) > 6 else np.nan
        pred = apply(Pv, b)
        up, dn, eq = per_episode_gain(y, groups, base_pred, pred)
        r = dict(task=task, config=name, runtime_s=round(rt, 1), val_macro_f1_original=base_f1,
                 val_macro_f1=mf1(y, pred), val_acc_original=accuracy_score(y, base_pred),
                 val_acc=accuracy_score(y, pred), nested_cv_macro_f1=nested,
                 episodes_better=up, episodes_worse=dn, episodes_same=eq, **{f"val_{k}": v for k, v in report(y, pred, cls).items()
                                                                           if k not in ("acc", "macro_f1")})
        rows.append(r)

    # selection: highest validation macro-F1 among untuned ensembles, tuned steps only if nested CV > untuned val F1 of the same base
    df = pd.DataFrame([r for r in rows if r["task"] == task])
    ens = df[~df.config.str.contains(r"\+ E")]
    best_ens = ens.loc[ens.val_macro_f1.idxmax()]
    tuned = df[df.config.str.startswith(best_ens.config + " +")]
    ok = tuned[tuned.nested_cv_macro_f1 > best_ens.val_macro_f1]
    pick = (ok.loc[ok.nested_cv_macro_f1.idxmax()] if len(ok) else best_ens)
    chosen[task] = dict(config=pick.config, val_macro_f1=float(pick.val_macro_f1),
                        nested_cv_macro_f1=None if pd.isna(pick.get("nested_cv_macro_f1", np.nan)) else float(pick.nested_cv_macro_f1),
                        original_val_macro_f1=float(base_f1), original_config=base_name)

    # one-time scoring of the single selected configuration on test / human gold
    name = pick.config
    S = next(s for s in stage if s[0] == name)
    _, Pv, Pt, Pg, b = S[:5]
    for split, P, D in (("val", Pv, V), ("test", Pt, T), ("gold", Pg, G)):
        yy = D[VARIANTS[0]][0].y.values
        for lab, bb, PP in (("original (seed0, argmax)", np.zeros(3), cand[base_name][("val", "test", "gold").index(split)]),
                            ("selected", b, P)):
            pr = apply(PP, bb)
            finals.append(dict(task=task, eval_set=split, config=lab, **report(yy, pr, cls)))
            cm = confusion_matrix(yy, pr, labels=[0, 1, 2])
            pd.DataFrame(cm, index=[f"true_{c}" for c in cls], columns=[f"pred_{c}" for c in cls]).to_csv(
                f"{OUT}/confusion_{task}_{split}_{'selected' if lab == 'selected' else 'original'}.csv")
    chosen[task]["bias"] = [float(x) for x in b]

pd.DataFrame(rows).to_csv(f"{OUT}/val_experiments.csv", index=False)
pd.DataFrame(finals).to_csv(f"{OUT}/final_selected_vs_original.csv", index=False)
json.dump(chosen, open(f"{OUT}/chosen_config.json", "w"), indent=1)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
print(pd.DataFrame(rows)[["task", "config", "runtime_s", "val_macro_f1_original", "val_macro_f1", "val_acc_original", "val_acc",
                          "nested_cv_macro_f1", "episodes_better", "episodes_worse"]].round(3).to_string())
print(json.dumps(chosen, indent=1))
print(pd.DataFrame(finals).round(3).to_string())
