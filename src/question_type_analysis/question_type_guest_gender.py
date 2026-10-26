#!/usr/bin/env python3
"""
Test whether the distribution of question types the HOST asks differs by
GUEST gender (male vs female guests), pooled over host gender, and check that
the result survives accounting for repeated hosts and episodes.

question_type_analysis.py compares the four host x guest dyads; this script
answers the simpler question "do female and male guests get different kinds of
questions?" using the classifications that script already wrote, so it needs
no GPU or classifier.

Questions are not independent: they are nested within episodes, and a handful
of hosts ask most of them (Raj Shamani alone asks ~39% of host questions). A
plain chi-square ignores this and overstates certainty, so three levels of
test are reported:

    1. chi_square_*         naive pooled chi-square (question-level, no clustering)
    2. episode_mwu_*        per-episode share of each type, female- vs
                            male-guest episodes (Mann-Whitney U); each episode
                            counts once
    3. gee_*                per-type GEE logistic regression, outcome
                            P(type == X), exchangeable correlation:
        gee_ep_*            ~ is_female_guest, clustered by episode
        gee_host_*          ~ is_female_guest, clustered by host
        gee_hostfe_*        ~ is_female_guest + host fixed effects, clustered
                            by episode (compares female vs male guests of the
                            SAME host, so differences between hosts can't
                            drive the result)

Host identity is the named asker_speaker (Raj Shamani, Ranveer A, ...). Hosts
the diarizer left as SPEAKER_XX are unidentifiable across episodes, so each
such episode is treated as its own host.

Bonferroni alpha is 0.05 / n_types, applied to each family of per-type tests.

Input:
    results/question_type_analysis/question_classification.csv

Output (in --out_dir):
    question_type_by_guest_gender.csv   counts and shares by guest gender
    question_type_guest_gender_stats.csv

Usage:
    python src/question_type_analysis/question_type_guest_gender.py
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

TYPES = ["closed", "open", "leading", "personal", "professional", "challenge"]


def host_id(row):
    spk = str(row["asker_speaker"])
    return f"unknown_{row['episode_id']}" if spk.startswith("SPEAKER_") else spk


def gee_row(test, formula, data, groups, term="is_female_guest", note=""):
    try:
        res = smf.gee(
            formula, groups=groups, data=data,
            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    except Exception as e:  # convergence / separation
        return {"test": test, "statistic": np.nan, "p_value": np.nan, "dof": np.nan,
                "notes": f"{note} FAILED: {type(e).__name__}"}
    coef, se = res.params[term], res.bse[term]
    lo, hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
    return {
        "test": test, "statistic": res.tvalues[term], "p_value": res.pvalues[term],
        "dof": data[groups].nunique(),
        "notes": f"{note} OR={np.exp(coef):.3f} (95% CI {lo:.3f}-{hi:.3f}), coef={coef:.4f}, "
                 f"se={se:.4f}, n_clusters={data[groups].nunique()}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified",
                    default="results/question_type_analysis/question_classification.csv")
    ap.add_argument("--out_dir", default="results/question_type_analysis")
    args = ap.parse_args()

    df = pd.read_csv(args.classified)
    q = df[(df["asker_role"] == "HOST") & df["guest_gender"].isin(["male", "female"])].copy()
    q["host_id"] = q.apply(host_id, axis=1)
    q["is_female_guest"] = (q["guest_gender"] == "female").astype(int)
    q["is_male_host"] = (q["host_gender"] == "male").astype(int)
    n_types = q["predicted_type"].nunique()
    alpha = 0.05 / n_types
    print(f"[OK] {len(q)} host questions, {q['episode_id'].nunique()} episodes, "
          f"{q['host_id'].nunique()} host ids "
          f"({(~q['host_id'].str.startswith('unknown_')).sum()} questions from named hosts)")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- distribution table ---
    counts = pd.crosstab(q["predicted_type"], q["guest_gender"]).reindex(TYPES)
    shares = pd.crosstab(q["predicted_type"], q["guest_gender"], normalize="columns").reindex(TYPES)
    summary = counts.add_suffix("_n").join(shares.add_suffix("_share")).reset_index()
    summary_path = os.path.join(args.out_dir, "question_type_by_guest_gender.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.round(3))

    rows = []

    # 1. naive chi-square
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    rows.append({"test": "chi_square_omnibus_type_by_guest_gender", "statistic": chi2,
                 "p_value": p, "dof": dof,
                 "notes": f"NAIVE (no clustering); n={len(q)}; H0: type independent of guest gender"})
    for t in TYPES:
        two = pd.crosstab(q["predicted_type"] == t, q["guest_gender"])
        c, pv, d, _ = stats.chi2_contingency(two)
        rows.append({"test": f"chi_square_{t}_vs_rest_by_guest_gender", "statistic": c,
                     "p_value": pv, "dof": d,
                     "notes": f"NAIVE; bonferroni_alpha={alpha:.4f}; share female={shares.loc[t, 'female']:.4f}, "
                              f"male={shares.loc[t, 'male']:.4f}"})

    # 2. episode-level Mann-Whitney
    ep = (q.groupby(["episode_id", "guest_gender"]).predicted_type
          .value_counts(normalize=True).unstack(fill_value=0).reset_index())
    n_f = (ep["guest_gender"] == "female").sum()
    n_m = (ep["guest_gender"] == "male").sum()
    for t in TYPES:
        a = ep.loc[ep["guest_gender"] == "female", t]
        b = ep.loc[ep["guest_gender"] == "male", t]
        u, pv = stats.mannwhitneyu(a, b)
        rows.append({"test": f"episode_mwu_{t}", "statistic": u, "p_value": pv, "dof": np.nan,
                     "notes": f"one value per episode ({n_f} female-guest, {n_m} male-guest episodes); "
                              f"bonferroni_alpha={alpha:.4f}; mean share female={a.mean():.4f}, male={b.mean():.4f}"})

    # 3. GEE logistic regressions per type
    named_hosts = q["host_id"].value_counts()
    for t in TYPES:
        q["y"] = (q["predicted_type"] == t).astype(int)
        rows.append(gee_row(f"gee_ep_{t}", "y ~ is_female_guest", q, "episode_id",
                            note=f"clustered by episode; bonferroni_alpha={alpha:.4f};"))
        rows.append(gee_row(f"gee_host_{t}", "y ~ is_female_guest", q, "host_id",
                            note=f"clustered by host; bonferroni_alpha={alpha:.4f};"))
        rows.append(gee_row(f"gee_hostfe_{t}", "y ~ is_female_guest + C(host_id)", q, "episode_id",
                            note=f"host fixed effects, clustered by episode; bonferroni_alpha={alpha:.4f};"))

    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(args.out_dir, "question_type_guest_gender_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Wrote {summary_path}\n[OK] Wrote {stats_path}")
    with pd.option_context("display.max_colwidth", 140, "display.width", 250):
        print(stats_df[["test", "p_value", "notes"]].to_string())


if __name__ == "__main__":
    main()
