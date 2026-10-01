#!/usr/bin/env python3
"""
speaking_time_mixed_model_v2.py

Linear mixed-effects model of the host's share of speaking time
(results/dyads/speaking_time_dyads_v2.csv, from src/speaking_time/speaking_time_v2.py):

    host_time_share ~ host_female * guest_female + (1 | host)

One row per episode. The random intercept is per host (show), the level at
which show-to-show variation lives: an episode-level intercept is not
identifiable with one observation per episode. host_female x guest_female
spans the four dyads (MM, MF, FM, FF). Guest gender varies within each host,
so its effect is estimated within host; host gender does not, so its effect
compares the 4 female-hosted and 5 male-hosted shows.

Inference: REML fit (statsmodels MixedLM) with t tests on between-within
degrees of freedom, a small-sample correction for few clusters:
host-level terms (intercept, host_female) get n_hosts - 2 df; within-host
terms (guest_female, interaction) get n_episodes - n_hosts - 2 df.
Nested models are compared with likelihood-ratio tests on ML fits.

Also written: estimated marginal means per dyad, pairwise dyad contrasts
(Holm-corrected), variance components / ICC / R2 (Nakagawa), a logit-share
sensitivity fit, leave-one-host-out, the word-share model, the balanced
200-episode subset, and diagnostic and summary figures.

If the host variance is estimated at zero (a singular fit), the mixed model
no longer adjusts for host, so its guest-gender estimate becomes a pooled
comparison. Every robustness row therefore also reports the within-host
guest-gender effect from an OLS model with host fixed effects, and the
rank-based within-host permutation test (within_host_test_v2.py) is run on
the full data and the balanced subset.

Episodes whose host did not match the speaker library have no host label
and are excluded.

Usage:
  python src/stat_analysis/speaking_time_mixed_model_v2.py
"""

import argparse
import os
import sys
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversational_style"))
from dyad_analysis_v2 import DYAD_ORDER  # noqa: E402
from within_host_test_v2 import within_host_guest_gender  # noqa: E402

FORMULA = "{y} ~ host_female * guest_female"
TERMS = ["Intercept", "host_female", "guest_female", "host_female:guest_female"]
HOST_LEVEL = {"Intercept", "host_female"}
CELLS = {"MALE->MALE": (0, 0), "MALE->FEMALE": (0, 1), "FEMALE->MALE": (1, 0), "FEMALE->FEMALE": (1, 1)}
AB = {"MALE->MALE": "MM", "MALE->FEMALE": "MF", "FEMALE->MALE": "FM", "FEMALE->FEMALE": "FF"}


def prep(df):
    d = df[df.host_name.notna()].copy()
    d["host_female"] = (d.host_gender == "female").astype(int)
    d["guest_female"] = (d.guest_gender == "female").astype(int)
    return d


def fit(d, y, reml=True, formula=FORMULA):
    """MixedLM fit, trying several optimizers (few groups make this fragile).
    Returns the first converged fit; if none converges (typically when the
    host variance sits on the zero boundary), the highest-likelihood fit with
    finite standard errors, marked with .fallback = True."""
    candidates = []
    for method in ["lbfgs", "bfgs", "powell", "nm"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(formula.format(y=y), d, groups=d["host_name"]).fit(
                    reml=reml, method=method, maxiter=2000)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not (np.isfinite(m.llf) and np.all(np.isfinite(m.bse_fe))):
            continue
        if m.converged:
            m.fallback = False
            return m
        candidates.append(m)
    if not candidates:
        raise RuntimeError(f"MixedLM failed for {y} ({len(d)} episodes)")
    m = max(candidates, key=lambda r: r.llf)
    m.fallback = True
    return m


def dfs(d):
    h, n = d.host_name.nunique(), len(d)
    return {t: (h - 2 if t in HOST_LEVEL else n - h - 2) for t in TERMS}


def fixed_table(m, d):
    df_ = dfs(d)
    fe, se = m.fe_params, m.bse_fe
    rows = []
    for t in TERMS:
        tval = fe[t] / se[t]
        q = stats.t.ppf(0.975, df_[t])
        rows.append({"term": t, "estimate": fe[t], "se": se[t], "df": df_[t], "t": tval,
                     "p": 2 * stats.t.sf(abs(tval), df_[t]),
                     "ci_low": fe[t] - q * se[t], "ci_high": fe[t] + q * se[t]})
    return pd.DataFrame(rows)


def cell_vec(hf, gf):
    return np.array([1, hf, gf, hf * gf], dtype=float)


def emmeans(m, d):
    cov = m.cov_params().loc[TERMS, TERMS].to_numpy()
    beta = m.fe_params[TERMS].to_numpy()
    df_host = d.host_name.nunique() - 2
    rows = []
    for dy in DYAD_ORDER:
        L = cell_vec(*CELLS[dy])
        est, se = L @ beta, np.sqrt(L @ cov @ L)
        q = stats.t.ppf(0.975, df_host)
        rows.append({"dyad": dy, "estimate": est, "se": se, "ci_low": est - q * se, "ci_high": est + q * se,
                     "n_episodes": int(((d.host_female == CELLS[dy][0]) & (d.guest_female == CELLS[dy][1])).sum())})
    return pd.DataFrame(rows)


def contrasts(m, d):
    cov = m.cov_params().loc[TERMS, TERMS].to_numpy()
    beta = m.fe_params[TERMS].to_numpy()
    h, n = d.host_name.nunique(), len(d)
    rows = []
    for a, b in combinations(DYAD_ORDER, 2):
        L = cell_vec(*CELLS[a]) - cell_vec(*CELLS[b])
        est, se = L @ beta, np.sqrt(L @ cov @ L)
        within = CELLS[a][0] == CELLS[b][0]   # same host gender -> within-host contrast
        df_ = n - h - 2 if within else h - 2
        tval = est / se
        rows.append({"contrast": f"{AB[a]} - {AB[b]}", "estimate": est, "se": se, "df": df_,
                     "type": "within host (guest gender)" if within else "between hosts",
                     "t": tval, "p": 2 * stats.t.sf(abs(tval), df_)})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out.p, method="holm")[1]
    return out


def variance_components(m, d, y):
    v_host = float(m.cov_re.iloc[0, 0])
    v_res = float(m.scale)
    X = np.column_stack([cell_vec(hf, gf) for hf, gf in zip(d.host_female, d.guest_female)]).T
    v_fix = float(np.var(X @ m.fe_params[TERMS].to_numpy()))
    total = v_fix + v_host + v_res
    ml_mixed = fit(d, y, reml=False)
    ml_ols = smf.ols(FORMULA.format(y=y), d).fit()
    lr_re = 2 * (ml_mixed.llf - ml_ols.llf)
    # Variance on the boundary under H0: p is half the chi2(1) tail
    p_re = 0.5 * stats.chi2.sf(max(lr_re, 0), 1)
    return pd.DataFrame([
        {"quantity": "host (show) variance", "value": v_host},
        {"quantity": "residual (episode) variance", "value": v_res},
        {"quantity": "ICC (share of residual+host variance due to host)", "value": v_host / (v_host + v_res)},
        {"quantity": "R2 marginal (fixed effects)", "value": v_fix / total},
        {"quantity": "R2 conditional (fixed + host)", "value": (v_fix + v_host) / total},
        {"quantity": "LRT host random effect: chi2", "value": lr_re},
        {"quantity": "LRT host random effect: p (boundary-corrected)", "value": p_re},
        {"quantity": "singular fit (host variance < 1e-6)", "value": float(v_host < 1e-6)},
    ])


def host_fe(d, y):
    """Within-host guest-gender effect for male hosts (and its difference for
    female hosts) from OLS with host fixed effects."""
    f = smf.ols(f"{y} ~ C(host_name) + guest_female + guest_female:host_female", d).fit()
    return (f.params["guest_female"], f.pvalues["guest_female"],
            f.params["guest_female:host_female"], f.pvalues["guest_female:host_female"])


def interaction_lrt(d, y):
    full = fit(d, y, reml=False)
    red = fit(d, y, reml=False, formula="{y} ~ host_female + guest_female")
    lr = 2 * (full.llf - red.llf)
    return lr, stats.chi2.sf(lr, 1)


def leave_one_host_out(d, y):
    rows = []
    for h in sorted(d.host_name.unique()):
        dd = d[d.host_name != h]
        m = fit(dd, y)
        ft = fixed_table(m, dd).set_index("term")
        rows.append({"dropped_host": h, "episodes": len(dd), "converged": not m.fallback,
                     "guest_female": ft.loc["guest_female", "estimate"], "p_guest_female": ft.loc["guest_female", "p"],
                     "interaction": ft.loc["host_female:guest_female", "estimate"],
                     "p_interaction": ft.loc["host_female:guest_female", "p"],
                     "host_female": ft.loc["host_female", "estimate"], "p_host_female": ft.loc["host_female", "p"]})
    return pd.DataFrame(rows)


def figures(d, m, emm, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    hosts = sorted(d.host_name.unique())
    cmap = plt.get_cmap("tab10")
    color = {h: cmap(i) for i, h in enumerate(hosts)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, dy in enumerate(DYAD_ORDER):
        g = d[d.dyad == dy]
        ax.boxplot(g.host_time_share, positions=[i], widths=0.5, showfliers=False,
                   medianprops={"color": "black"})
        for h, gg in g.groupby("host_name"):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(gg)), gg.host_time_share, s=10, alpha=0.6,
                       color=color[h], label=h if i == 0 or h not in d[d.dyad.isin(DYAD_ORDER[:i])].host_name.values else None)
        r = emm[emm.dyad == dy].iloc[0]
        ax.errorbar(i + 0.33, r.estimate, yerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                    fmt="D", color="black", capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{AB[x]}\n(n={int((d.dyad == x).sum())})" for x in DYAD_ORDER])
    ax.set_ylabel("Host share of speaking time")
    ax.set_title("Host share of speaking time by dyad (points = episodes, colored by host;\n"
                 "diamonds = model estimate with 95% CI)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/speaking_time_share_by_dyad.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    resid = m.resid
    stats.probplot(resid, dist="norm", plot=axes[0])
    axes[0].set_title("Residual QQ plot")
    axes[1].scatter(m.fittedvalues, resid, s=8, alpha=0.5)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Fitted host share")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs fitted")
    re = pd.Series({h: v.iloc[0] for h, v in m.random_effects.items()}).sort_values()
    axes[2].barh(re.index, re.values, color=[color[h] for h in re.index])
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_title("Host random intercepts")
    axes[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/speaking_time_model_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Mixed model of host speaking-time share")
    ap.add_argument("--input", default="results/dyads/speaking_time_dyads_v2.csv")
    ap.add_argument("--balanced", default="results/features/balanced_200_episodes.csv")
    ap.add_argument("--out_prefix", default="results/stat_analysis/speaking_time/speaking_time_mixed_model_v2")
    ap.add_argument("--fig_dir", default="results/figures/speaking_time")
    args = ap.parse_args()

    d = prep(pd.read_csv(args.input))
    d["logit_host_time_share"] = np.log(d.host_time_share / (1 - d.host_time_share))
    y = "host_time_share"
    m = fit(d, y)
    fe = fixed_table(m, d)
    emm = emmeans(m, d)
    con = contrasts(m, d)
    vc = variance_components(m, d, y)
    lr_int, p_int = interaction_lrt(d, y)

    # Sensitivity and robustness fits (guest-gender and interaction terms)
    rob = []
    for name, dd, yy in [
        ("main: time share", d, y),
        ("logit(time share)", d, "logit_host_time_share"),
        ("word share instead of time share", d, "host_word_share"),
        ("balanced 200-episode subset", d[d.episode_id.isin(pd.read_csv(args.balanced).episode_id)], y),
    ]:
        ft = fixed_table(fit(dd, yy), dd).set_index("term")
        g, gp, i, ip = host_fe(dd, yy)
        rob.append({"model": name, "episodes": len(dd), "hosts": dd.host_name.nunique(),
                    **{f"{t}_est": ft.loc[t, "estimate"] for t in TERMS[1:]},
                    **{f"{t}_p": ft.loc[t, "p"] for t in TERMS[1:]},
                    "hostFE_guest_female_est": g, "hostFE_guest_female_p": gp,
                    "hostFE_interaction_est": i, "hostFE_interaction_p": ip})
    rob = pd.DataFrame(rob)
    bal = d[d.episode_id.isin(pd.read_csv(args.balanced).episode_id)]
    rank = pd.concat([within_host_guest_gender(dd, [y], 10000, 42).assign(data=name)
                      for name, dd in [("all episodes", d), ("balanced 200-episode subset", bal)]])
    rank = rank[["data", "rank_diff_female_minus_male", "p_perm", "n_hosts", "n_episodes"]]
    loho = leave_one_host_out(d, y)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    fe.to_csv(f"{args.out_prefix}_fixed_effects.csv", index=False)
    emm.to_csv(f"{args.out_prefix}_emmeans.csv", index=False)
    con.to_csv(f"{args.out_prefix}_contrasts.csv", index=False)
    vc = pd.concat([vc, pd.DataFrame([{"quantity": "LRT interaction: chi2", "value": lr_int},
                                      {"quantity": "LRT interaction: p", "value": p_int},
                                      {"quantity": "episodes", "value": len(d)},
                                      {"quantity": "hosts", "value": d.host_name.nunique()}])])
    vc.to_csv(f"{args.out_prefix}_variance.csv", index=False)
    rob.to_csv(f"{args.out_prefix}_robustness.csv", index=False)
    loho.to_csv(f"{args.out_prefix}_leave_one_host_out.csv", index=False)
    rank.to_csv(f"{args.out_prefix}_rank_within_host.csv", index=False)
    figures(d, m, emm, args.fig_dir)

    pd.set_option("display.width", 220)
    print(f"{len(d)} episodes, {d.host_name.nunique()} hosts\n")
    print(m.summary())
    print("Fixed effects (between-within df):")
    print(fe.round(4).to_string(index=False))
    print("\nEstimated host share per dyad:")
    print(emm.round(4).to_string(index=False))
    print("\nPairwise contrasts:")
    print(con.round(4).to_string(index=False))
    print("\nVariance components and model comparisons:")
    print(vc.round(4).to_string(index=False))
    print("\nSensitivity / robustness:")
    print(rob.round(4).to_string(index=False))
    print("\nRank-based within-host test:")
    print(rank.round(4).to_string(index=False))
    print("\nLeave one host out:")
    print(loho.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
