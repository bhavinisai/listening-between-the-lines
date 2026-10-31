#!/usr/bin/env python3
"""
Step 2 of the topic-control revision: prespecified sensitivity analyses and
corrected versions of Report 1 Tables 10-13.

Specifications (outcome for A-F: followed_up 0/1 under the Report 1 definition;
mixed-gender episodes only):
  A  Original GEE: logit, exchangeable, clustered by episode;
     male + host + raw initiator word count
  B  A without any length covariate
  C  A with log(1 + initiator words) instead of raw words
  D  C plus log(1 + responder words)   [conditions on part of the response]
  E  A with every episode weighted equally (weights 1/n_pairs, independence)
  F  A clustered by host (independence; Mancl-DeRouen bias-reduced SEs).
     Only 4 male-host clusters -> reported for transparency, not inference.
  G  Episode level: host-minus-guest follow-up gap, M->F vs F->M episodes
     (the role-adjusted gender contrast without any turn-level model)

Plus: leave-one-host-out influence check for A and C, corrected Table 10
(episode-cluster bootstrap CIs), Table 11 (threshold sensitivity, with
within-role gaps added), Table 12 (unrounded, per-100-word units), Table 13.

Usage:
    python src/topic_control_revision/02_sensitivity_and_tables.py
"""

import json
import os
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from common import (MIXED, ORIG_EMB_TURNS, ORIG_STATS, ORIG_TURNS, OUT, SEED, TABLES,
                    load_original_mixed)

N_BOOT = 5000


# ---------------------------------------------------------------- formatting
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def fmt_or(est, lo, hi):
    return f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


def write_table(df, name, caption):
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    cols = list(df.columns)
    md = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    with open(os.path.join(TABLES, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    def esc(s):
        return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                .replace("→", r"$\rightarrow$").replace("−", "$-$").replace("<", r"$<$")
                .replace("≤", r"$\leq$").replace("×", r"$\times$"))
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small",
           rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
           r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        tex.append(" & ".join(esc(v) for v in r.values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES, f"{name}.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


# ---------------------------------------------------------------- models
def fit_gee(m, formula, groups, cov="exchangeable", weights=None, cov_type="robust"):
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cs = sm.cov_struct.Exchangeable() if cov == "exchangeable" else sm.cov_struct.Independence()
        res = smf.gee(formula, groups=groups, data=m, family=sm.families.Binomial(),
                      cov_struct=cs, weights=weights).fit(cov_type=cov_type)
        caught = sorted({str(x.message).split("\n")[0][:80] for x in w
                         if "overflow" not in str(x.message)})
    return res, caught


def or_row(res, term, df_t=None):
    b, s = res.params[term], res.bse[term]
    if df_t is None:
        z, crit = b / s, 1.96
        p = 2 * stats.norm.sf(abs(z))
    else:
        crit = stats.t.ppf(0.975, df_t)
        p = 2 * stats.t.sf(abs(b / s), df_t)
    return np.exp(b), np.exp(b - crit * s), np.exp(b + crit * s), p, b, s


def episode_gap_table(m):
    pr = (m.groupby(["episode_id", "dyad", "initiator_role"])["followed_up"].mean()
          .unstack().reset_index())
    pr["host_minus_guest"] = (pr["HOST"] - pr["GUEST"]) * 100
    return pr


def boot_diff_means(a, b, rng, n=10000):
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def sensitivity(m):
    rng = np.random.default_rng(SEED)
    rows, warn_log = [], {}
    n_eps, n_pairs = m["episode_id"].nunique(), len(m)
    n_hosts = m["host_id"].nunique()
    named = m.loc[~m["host_id"].str.startswith("unnamed:"), ["host_id", "dyad"]].drop_duplicates()
    n_named = f"{n_hosts} host clusters ({(named.dyad == 'MALE->FEMALE').sum()} male, " \
              f"{(named.dyad == 'FEMALE->MALE').sum()} named female, " \
              f"{m.loc[m.host_id.str.startswith('unnamed:'), 'episode_id'].nunique()} unnamed female)"
    base = "followed_up ~ is_male_initiator + is_host_initiator"
    specs = [
        ("A", "Original (Report 1)", base + " + wc100", "episode_id", "exchangeable", None, "robust",
         "Male + host + initiator words (raw, per 100)", "Episode clusters, exchangeable, robust SE"),
        ("B", "No length adjustment", base, "episode_id", "exchangeable", None, "robust",
         "Male + host", "Episode clusters, exchangeable, robust SE"),
        ("C", "Log initiator length", base + " + log_init_wc", "episode_id", "exchangeable", None, "robust",
         "Male + host + log(1+initiator words)", "Episode clusters, exchangeable, robust SE"),
        ("D", "Log initiator + responder length", base + " + log_init_wc + log_resp_wc", "episode_id",
         "exchangeable", None, "robust",
         "Male + host + log(1+initiator words) + log(1+responder words)",
         "Episode clusters, exchangeable, robust SE"),
        ("E", "Equal episode weights", base + " + wc100", "episode_id", "independence",
         (1.0 / m.groupby("episode_id")["followed_up"].transform("size")).values, "robust",
         "As A", "Episode clusters, independence, weight 1/n per episode"),
        ("F", "Host-clustered SE", base + " + wc100", "host_id", "independence", None, "bias_reduced",
         "As A", f"{n_hosts} host clusters, independence, bias-reduced SE, t(df={n_hosts - 1})"),
    ]
    fits = {}
    for code, label, f, grp, cov, w, ct, covs, clus in specs:
        res, wl = fit_gee(m, f, grp, cov, w, ct)
        fits[code] = res
        warn_log[code] = wl
        df_t = n_hosts - 1 if code == "F" else None
        mo = or_row(res, "is_male_initiator", df_t)
        ho = or_row(res, "is_host_initiator", df_t)
        rows.append({
            "spec": code, "label": label,
            "estimand": "Role-adjusted log-odds ratio of follow-up, male vs female initiator "
                        "(population-averaged)" + ("; conditional on reply length" if code == "D" else ""),
            "outcome": "followed_up (0/1, Report 1 definition)",
            "covariates": covs, "clustering_weighting": clus,
            "n_episodes": n_eps, "n_pairs": n_pairs, "n_host_clusters": n_hosts,
            "male_OR": mo[0], "male_lo": mo[1], "male_hi": mo[2], "male_p": mo[3],
            "host_OR": ho[0], "host_lo": ho[1], "host_hi": ho[2], "host_p": ho[3],
            "use_for_inference": "No (only 4 male-host clusters)" if code == "F" else "Yes",
            "warnings": "; ".join(wl) if wl else "none",
        })

    # G: episode-level contrast
    pr = episode_gap_table(m)
    mf = pr.loc[pr.dyad == "MALE->FEMALE", "host_minus_guest"]
    fm = pr.loc[pr.dyad == "FEMALE->MALE", "host_minus_guest"]
    wt = stats.ttest_ind(mf, fm, equal_var=False)
    mw = stats.mannwhitneyu(mf, fm, alternative="two-sided")
    lo, hi = boot_diff_means(mf, fm, rng)
    diff = mf.mean() - fm.mean()
    rows.append({
        "spec": "G", "label": "Episode-level gap comparison",
        "estimand": "Difference in mean host-minus-guest follow-up gap (pp), M→F minus F→M episodes; "
                    "half of it is the role-adjusted male-minus-female difference",
        "outcome": "Per-episode follow-up rate (Report 1 definition)",
        "covariates": "None", "clustering_weighting": "One value per episode; Welch t; bootstrap CI (10,000)",
        "n_episodes": len(pr), "n_pairs": n_pairs, "n_host_clusters": n_hosts,
        "male_OR": np.nan, "male_lo": np.nan, "male_hi": np.nan, "male_p": wt.pvalue,
        "host_OR": np.nan, "host_lo": np.nan, "host_hi": np.nan, "host_p": np.nan,
        "use_for_inference": "Yes (descriptive; still confounded with host identity)",
        "warnings": "none",
        "gap_MF_pp": mf.mean(), "gap_FM_pp": fm.mean(), "diff_pp": diff, "diff_lo": lo, "diff_hi": hi,
        "implied_male_minus_female_pp": diff / 2, "mannwhitney_p": mw.pvalue,
    })
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(TABLES, "sensitivity_full.csv"), index=False)

    compact = []
    for r in rows:
        if r["spec"] == "G":
            male = (f"Gap M→F {r['gap_MF_pp']:.1f} vs F→M {r['gap_FM_pp']:.1f} pp; "
                    f"difference {r['diff_pp']:.1f} [{r['diff_lo']:.1f}, {r['diff_hi']:.1f}]")
            compact.append({"Spec": "G", "Model": r["label"], "Unit": "Episode (n = 131)",
                            "Male initiator": male, "p (male)": fmt_p(r["male_p"]),
                            "Host initiator": "—", "Use": "Yes"})
        else:
            compact.append({"Spec": r["spec"], "Model": r["label"],
                            "Unit": f"Pair, clustered by {'host' if r['spec'] == 'F' else 'episode'}",
                            "Male initiator": fmt_or(r["male_OR"], r["male_lo"], r["male_hi"]),
                            "p (male)": fmt_p(r["male_p"]),
                            "Host initiator": fmt_or(r["host_OR"], r["host_lo"], r["host_hi"]),
                            "Use": "Transparency only" if r["spec"] == "F" else "Yes"})
    write_table(pd.DataFrame(compact), "table_S1_sensitivity",
                "Sensitivity of the gender and role estimates for topic follow-up in the 131 "
                "mixed-gender episodes (17,046 initiation-response pairs). A-F: GEE logistic "
                "regression, odds ratios with 95% CI. G: episode-level comparison, percentage points.")

    # Leave-one-host-out (named hosts with >= 4 episodes)
    big = (m[~m.host_id.str.startswith("unnamed:")].groupby("host_id")["episode_id"].nunique())
    big = big[big >= 4].index
    lo_rows = []
    for h in big:
        sub = m[m.host_id != h]
        for code, f in [("A", base + " + wc100"), ("C", base + " + log_init_wc")]:
            res, _ = fit_gee(sub, f, "episode_id")
            o = or_row(res, "is_male_initiator")
            lo_rows.append({"dropped_host": h, "host_gender": "male" if (m[m.host_id == h].dyad.iloc[0] == "MALE->FEMALE") else "female",
                            "episodes_dropped": int(m[m.host_id == h].episode_id.nunique()), "spec": code,
                            "male_OR": round(o[0], 3), "lo": round(o[1], 3), "hi": round(o[2], 3), "p": o[3]})
    pd.DataFrame(lo_rows).to_csv(os.path.join(TABLES, "leave_one_host_out.csv"), index=False)
    return full, fits, warn_log


# ---------------------------------------------------------------- Table 10
def cluster_boot_pooled(m, rng, n=N_BOOT):
    eps = m["episode_id"].unique()
    agg = m.groupby(["episode_id", "initiator_gender"])["followed_up"].agg(["sum", "size"]).unstack(fill_value=0)
    S, N = agg["sum"].loc[eps], agg["size"].loc[eps]
    out = {"female": [], "male": [], "diff": []}
    for _ in range(n):
        idx = rng.integers(0, len(eps), len(eps))
        f = S["female"].values[idx].sum() / N["female"].values[idx].sum()
        mm = S["male"].values[idx].sum() / N["male"].values[idx].sum()
        out["female"].append(f)
        out["male"].append(mm)
        out["diff"].append(mm - f)
    return {k: np.percentile(np.array(v) * 100, [2.5, 97.5]) for k, v in out.items()}


def table10(m, rng):
    pooled = m.groupby("initiator_gender")["followed_up"].agg(["mean", "size"])
    ci = cluster_boot_pooled(m, rng)
    pe = m.groupby(["episode_id", "initiator_gender"])["followed_up"].mean().unstack() * 100
    d = pe["male"] - pe["female"]
    rows = []
    for g in ["female", "male"]:
        x = pe[g]
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        rows.append({"Initiator": g.capitalize(), "Initiations": f"{int(pooled.loc[g, 'size']):,}",
                     "Pooled rate [episode-bootstrap 95% CI]":
                         f"{pooled.loc[g, 'mean'] * 100:.1f}% [{ci[g][0]:.1f}, {ci[g][1]:.1f}]",
                     "Per-episode mean [95% CI]": f"{x.mean():.1f}% [{x.mean() - h:.1f}, {x.mean() + h:.1f}]",
                     "Per-episode median": f"{x.median():.1f}%"})
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    gap = (pooled.loc["male", "mean"] - pooled.loc["female", "mean"]) * 100
    rows.append({"Initiator": "Male − female", "Initiations": "",
                 "Pooled rate [episode-bootstrap 95% CI]": f"{gap:+.1f} pp [{ci['diff'][0]:+.1f}, {ci['diff'][1]:+.1f}]",
                 "Per-episode mean [95% CI]": f"{d.mean():+.1f} pp [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}]",
                 "Per-episode median": f"{d.median():+.1f} pp"})
    write_table(pd.DataFrame(rows), "table10_corrected",
                "Follow-up rate by initiator gender in the 131 mixed-gender episodes: pooled over "
                "initiations and averaged per episode. Neither column adjusts for role.")
    return ci


# ---------------------------------------------------------------- Table 11
def table11(m, rng, n=2000):
    thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    eps = m["episode_id"].unique()
    pos = {e: i for i, e in enumerate(eps)}
    keys = [("all", None), ("host", "HOST"), ("guest", "GUEST")]
    cnt = {}
    for k, role in keys:
        sub = m if role is None else m[m.initiator_role == role]
        for g in ["female", "male"]:
            A = np.zeros((len(eps), len(thresholds)))
            T = np.zeros(len(eps))
            for e, sims in sub[sub.initiator_gender == g].groupby("episode_id")["similarity"]:
                A[pos[e]] = (sims.values[:, None] > np.array(thresholds)[None, :]).sum(0)
                T[pos[e]] = len(sims)
            cnt[(k, g)] = (A, T)

    def gaps(idx):
        out = {}
        for k, _ in keys:
            (Af, Tf), (Am, Tm) = cnt[(k, "female")], cnt[(k, "male")]
            fr = Af[idx].sum(0) / Tf[idx].sum()
            mr = Am[idx].sum(0) / Tm[idx].sum()
            out[k] = (fr, mr, (mr - fr) * 100)
        return out

    point = gaps(np.arange(len(eps)))
    boots = {k: [] for k, _ in keys}
    for _ in range(n):
        g = gaps(rng.integers(0, len(eps), len(eps)))
        for k, _ in keys:
            boots[k].append(g[k][2])
    ci = {k: np.percentile(np.array(v), [2.5, 97.5], axis=0) for k, v in boots.items()}
    rows = []
    for i, t in enumerate(thresholds):
        fr, mr, gp = point["all"]
        rows.append({"Threshold": f"{t:.2f}", "Female": f"{fr[i] * 100:.1f}%", "Male": f"{mr[i] * 100:.1f}%",
                     "Raw gap, pp [95% CI]": f"{gp[i]:+.1f} [{ci['all'][0][i]:+.1f}, {ci['all'][1][i]:+.1f}]",
                     "Host-initiated gap, pp [95% CI]":
                         f"{point['host'][2][i]:+.1f} [{ci['host'][0][i]:+.1f}, {ci['host'][1][i]:+.1f}]",
                     "Guest-initiated gap, pp [95% CI]":
                         f"{point['guest'][2][i]:+.1f} [{ci['guest'][0][i]:+.1f}, {ci['guest'][1][i]:+.1f}]"})
    write_table(pd.DataFrame(rows), "table11_corrected",
                "Pooled follow-up rate at fixed TF-IDF similarity thresholds (131 mixed-gender "
                "episodes). Gaps are male minus female; 95% CIs from 2,000 episode-level bootstrap "
                "resamples. The raw gap is not adjusted for role; the host- and guest-initiated "
                "gaps compare men and women within the same role, which means comparing "
                "different episodes and hosts.")


# ---------------------------------------------------------------- Table 12
def table12(m, fits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = m.copy()
        d["speaker_key"] = d["episode_id"] + "::" + d["initiator_speaker"]
        mx = MixedLM.from_formula("similarity ~ is_male_initiator + wc100 + is_host_initiator",
                                  groups="episode_id", vc_formula={"speaker": "0 + C(speaker_key)"},
                                  data=d).fit(reml=True)
    ci = mx.conf_int()
    resid = mx.resid
    st = pd.read_csv(ORIG_STATS)
    vb = st[st.test == "glmm_followedup_is_male_initiator"].iloc[0]
    rows = []
    A = fits["A"]
    for term, label in [("is_male_initiator", "Male initiator (vs female)"),
                        ("is_host_initiator", "Host initiator (vs guest)"),
                        ("wc100", "Initiator turn length, per 100 words")]:
        o = or_row(A, term)
        rows.append({"Predictor": label, "GEE OR [95% CI]": fmt_or(*o[:3]), "GEE p": fmt_p(o[3]),
                     "Mixed model β ×100 [95% CI]":
                         f"{mx.params[term] * 100:.2f} [{ci.loc[term, 0] * 100:.2f}, {ci.loc[term, 1] * 100:.2f}]",
                     "Mixed model p": fmt_p(mx.pvalues[term])})
    write_table(pd.DataFrame(rows), "table12_corrected",
                "Predictors of topic follow-up in the 131 mixed-gender episodes (17,046 pairs). "
                "GEE: binary follow-up, logit link, exchangeable correlation, robust SEs clustered "
                "by episode; population-averaged odds ratios. Mixed model: a different outcome, the "
                "continuous TF-IDF similarity of the reply, with random intercepts for episode and "
                "speaker; β is shown ×100 (similarity points per unit).")
    diag = {"mixedlm_converged": bool(mx.converged), "resid_skew": float(stats.skew(resid)),
            "share_similarity_zero": float((m.similarity == 0).mean()),
            "vb_glmm_male_logodds": float(vb["notes"].split("coef=")[1].split(" ")[0]),
            "vb_glmm_male_OR": float(np.exp(float(vb["notes"].split("coef=")[1].split(" ")[0])))}
    for term in ["is_male_initiator", "is_host_initiator", "wc100"]:
        diag[f"mixedlm_{term}_coef"] = float(mx.params[term])
        diag[f"mixedlm_{term}_p"] = float(mx.pvalues[term])
    return diag


def table13(m):
    emb = pd.read_csv(ORIG_EMB_TURNS)
    em = emb[emb.dyad.isin(MIXED)]
    rows = []
    for label, df in [("TF-IDF", m), ("Sentence embeddings", em)]:
        pe = df.groupby(["episode_id", "initiator_gender"])["followed_up"].mean().unstack() * 100
        d = pe["male"] - pe["female"]
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        t = stats.ttest_rel(pe["male"], pe["female"])
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=(10000, len(d)))
        perm = (np.sum(np.abs((d.values[None, :] * signs).mean(1)) >= abs(d.mean())) + 1) / 10001
        rows.append({"Similarity": label, "Pairs": f"{len(df):,}", "Female": f"{pe['female'].mean():.1f}%",
                     "Male": f"{pe['male'].mean():.1f}%",
                     "Gap, pp [95% CI]": f"{d.mean():+.1f} [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}]",
                     "Paired t (p)": f"{t.statistic:.2f} ({fmt_p(t.pvalue)})", "Permutation p": fmt_p(perm)})
    write_table(pd.DataFrame(rows), "table13_corrected",
                "Within-episode comparison of the male and female speaker in the 131 mixed-gender "
                "episodes, under TF-IDF and sentence-embedding similarity (episode means; paired t "
                "and 10,000 sign-flip permutations). Not adjusted for role. Each measure selects "
                "its own set of initiations.")


def main():
    m = load_original_mixed()
    rng = np.random.default_rng(SEED)
    full, fits, warn_log = sensitivity(m)
    table10(m, rng)
    table11(m, rng)
    diag = table12(m, fits)
    table13(m)
    pr = episode_gap_table(m)
    pr.to_csv(os.path.join(TABLES, "episode_host_guest_gaps.csv"), index=False)
    meta = {
        "python": platform.python_version(), "statsmodels": statsmodels.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "seed": SEED, "cluster_bootstrap_reps_table10": N_BOOT, "bootstrap_reps_table11": 2000,
        "inputs": [os.path.relpath(p, OUT) for p in [ORIG_TURNS, ORIG_STATS, ORIG_EMB_TURNS]],
        "gee_warnings": warn_log, "model_diagnostics": diag,
    }
    with open(os.path.join(OUT, "run_metadata_02.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(pd.read_csv(os.path.join(TABLES, "table_S1_sensitivity.csv")).to_string())
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
