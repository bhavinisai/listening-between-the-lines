#!/usr/bin/env python3
"""
Final guest-gender / host-gender analysis of HOST question types.

Unit of inference: questions are nested in episodes and a few hosts ask most of
them, so no test treats questions as independent.

Models (per question type, outcome = 1[type == X]):
    M1  GEE logit, exchangeable, clustered by episode            ~ guest_female
    M2  GEE logit, host fixed effects, clustered by episode      ~ guest_female + C(host)
        (PRIMARY: compares female vs male guests of the SAME host)
    M3  GEE logit, clustered by host (35 clusters; fragile)      ~ guest_female
    M4  GEE logit, host FE: guest_female + guest_female:host_female
    M5  GEE logit clustered by host                              ~ host_female + guest_female
    M6  Episode-level mixed model: share ~ guest_female, host random intercept
Bonferroni alpha = 0.05 / 6 per family.

Within-host contrasts: hosts with >= MIN_EP episodes of each guest gender;
difference in mean per-episode share (female - male guests) with an
episode-bootstrap CI.

Sensitivity (all use M2 unless noted):
    S1  drop predictions below confidence 0.5 / 0.6 / 0.7
    S2  soft labels: fractional outcome = predicted probability
    S3  multiple imputation: labels drawn from predicted probabilities
        (N_MI draws, Rubin's rules), personal + professional
    S4  misclassification correction with a held-out confusion matrix
        (host-standardised female-minus-male share difference, bootstrap CI
        over episodes and over the confusion-matrix sample)

Inputs:
    --pred       qid, p_<class>, pred, confidence  (predict_v2.py output)
    --pool       pool_host_questions.csv (episode_id, host_id, host_gender, guest_gender)
    --episodes   results/features/all_episodes.csv  (for coverage table)
    --confusion  CSV with columns true,pred (held-out reference vs prediction) for S4
Outputs (in --out_dir): coverage.csv, shares_*.csv, gee_main.csv,
    within_host.csv, sens_threshold.csv, sens_soft.csv, sens_mi.csv,
    sens_corrected.csv, analysis_log.txt
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
T = ["closed", "open", "leading", "personal", "professional", "challenge"]
ALPHA = 0.05 / 6
MIN_EP = 5
N_MI = 30
B_BOOT = 400


def gee_fit(formula, data, groups, term):
    try:
        r = smf.gee(formula, groups=groups, data=data, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        b, se = r.params[term], r.bse[term]
        return dict(coef=b, se=se, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                    p=r.pvalues[term], n_clusters=int(data[groups].nunique()), ok=True)
    except Exception as e:  # separation / convergence
        return dict(coef=np.nan, se=np.nan, OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n_clusters=0, ok=False, err=type(e).__name__)


def load(args):
    pool = pd.read_csv(args.pool, usecols=["qid", "episode_id", "host_id", "host_gender", "guest_gender"])
    pr = pd.read_csv(args.pred)
    q = pool.merge(pr, on="qid", validate="one_to_one")
    q = q[q.guest_gender.isin(["male", "female"])].copy()
    q["gf"] = (q.guest_gender == "female").astype(int)
    q["hf"] = (q.host_gender == "female").astype(int)
    for t in T:
        q[f"y_{t}"] = (q.pred == t).astype(int)
    return q


def coverage(args, q_all, log):
    e = pd.read_csv(args.episodes)
    e["has_transcript"] = e.episode_id.map(lambda x: os.path.exists(os.path.join(args.transcript_dir, f"{x}.txt")))
    qe = q_all.groupby("episode_id").size()
    e["n_host_questions"] = e.episode_id.map(qe).fillna(0).astype(int)
    rows = {
        "episodes_in_corpus": len(e),
        "with_transcript": int(e.has_transcript.sum()),
        "with_>=1_host_question": int((e.n_host_questions > 0).sum()),
        "guest_gender_unknown": int((~e.guest_gender.isin(["male", "female"])).sum()),
        "host_gender_unknown": int((~e.host_gender.isin(["male", "female"])).sum()),
        "episodes_in_gender_tests": int(((e.n_host_questions > 0) & e.guest_gender.isin(["male", "female"])).sum()),
        "host_questions_total": int(len(q_all)),
        "host_questions_in_gender_tests": int(e[e.guest_gender.isin(["male", "female"])].n_host_questions.sum()),
        "female_guest_episodes": int(((e.guest_gender == "female") & (e.n_host_questions > 0)).sum()),
        "male_guest_episodes": int(((e.guest_gender == "male") & (e.n_host_questions > 0)).sum()),
        "distinct_hosts": int(q_all.host_id.nunique()),
    }
    pd.Series(rows, name="value").to_csv(f"{args.out_dir}/coverage.csv")
    log.append("COVERAGE\n" + "\n".join(f"  {k}: {v}" for k, v in rows.items()))
    miss = e[e.n_host_questions == 0].episode_id.tolist()
    log.append(f"  episodes with no host question: {miss}")


def shares(q, args):
    for key, name in [("guest_gender", "guest"), ("host_gender", "host"), ("dyad", "dyad")]:
        if key == "dyad":
            q["dyad"] = q.host_gender.str.upper() + "->" + q.guest_gender.str.upper()
        hard = pd.crosstab(q[key], q.pred).reindex(columns=T, fill_value=0)
        soft = q.groupby(key)[[f"p_{t}" for t in T]].mean()
        soft.columns = T
        out = pd.concat({"n": hard, "hard_share": hard.div(hard.sum(1), axis=0), "soft_share": soft}, axis=1)
        out.to_csv(f"{args.out_dir}/shares_{name}.csv")


def episode_boot_ci(q, key, B=B_BOOT, seed=0):
    """Cluster bootstrap (resample episodes) CI for question-level type share per group."""
    rng = np.random.default_rng(seed)
    eps = q.groupby("episode_id").agg(g=(key, "first"))
    cnt = q.groupby(["episode_id"])[[f"y_{t}" for t in T]].sum()
    n = q.groupby("episode_id").size()
    rows = []
    for g in sorted(eps.g.unique()):
        ids = eps.index[eps.g == g].values
        est = cnt.loc[ids].sum() / n.loc[ids].sum()
        bs = []
        for _ in range(B):
            s = rng.choice(ids, len(ids))
            bs.append(cnt.loc[s].sum().values / n.loc[s].sum())
        bs = np.array(bs)
        for i, t in enumerate(T):
            rows.append(dict(group=g, type=t, share=est.iloc[i], lo=np.percentile(bs[:, i], 2.5), hi=np.percentile(bs[:, i], 97.5)))
    return pd.DataFrame(rows)


def main_models(q):
    rows = []
    for t in T:
        d = q.assign(y=q[f"y_{t}"])
        m = {
            "M1_episode_cluster": gee_fit("y ~ gf", d, "episode_id", "gf"),
            "M2_hostFE_episode_cluster": gee_fit("y ~ gf + C(host_id)", d, "episode_id", "gf"),
            "M3_host_cluster": gee_fit("y ~ gf", d, "host_id", "gf"),
            "M4_interaction_hostFE": gee_fit("y ~ gf + gf:hf + C(host_id)", d, "episode_id", "gf:hf"),
            "M5_host_gender_host_cluster": gee_fit("y ~ hf + gf", d, "host_id", "hf"),
        }
        for k, r in m.items():
            rows.append(dict(type=t, model=k, **r, bonferroni_sig=bool(r["p"] < ALPHA) if r["ok"] else False))
        # M6 episode-level mixed model
        ep = d.groupby("episode_id").agg(share=("y", "mean"), gf=("gf", "first"), host_id=("host_id", "first"), n=("y", "size")).reset_index()
        try:
            r6 = smf.mixedlm("share ~ gf", ep, groups=ep["host_id"]).fit(reml=True)
            b, se = r6.params["gf"], r6.bse["gf"]
            rows.append(dict(type=t, model="M6_episode_LMM_host_RE", coef=b, se=se, OR=np.nan, lo=b - 1.96 * se, hi=b + 1.96 * se,
                             p=r6.pvalues["gf"], n_clusters=ep.host_id.nunique(), ok=True, bonferroni_sig=bool(r6.pvalues["gf"] < ALPHA)))
        except Exception as e:
            rows.append(dict(type=t, model="M6_episode_LMM_host_RE", ok=False, err=type(e).__name__))
    return pd.DataFrame(rows)


def within_host(q, rng):
    ep = q.groupby("episode_id").agg(host_id=("host_id", "first"), gf=("gf", "first"), n=("pred", "size"),
                                      **{t: (f"y_{t}", "mean") for t in T}).reset_index()
    rows = []
    for h, g in ep.groupby("host_id"):
        f, m = g[g.gf == 1], g[g.gf == 0]
        if len(f) < MIN_EP or len(m) < MIN_EP:
            continue
        for t in T:
            d = f[t].mean() - m[t].mean()
            bs = [rng.choice(f[t].values, len(f)).mean() - rng.choice(m[t].values, len(m)).mean() for _ in range(B_BOOT)]
            u = stats.mannwhitneyu(f[t], m[t], alternative="two-sided")
            rows.append(dict(host=h, type=t, n_ep_female=len(f), n_ep_male=len(m), mean_female=f[t].mean(), mean_male=m[t].mean(),
                             diff=d, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), mwu_p=u.pvalue))
    return pd.DataFrame(rows)


def sens_threshold(q):
    rows = []
    for thr in [0.0, 0.5, 0.6, 0.7]:
        d = q[q.confidence >= thr]
        for t in T:
            r = gee_fit("y ~ gf + C(host_id)", d.assign(y=d[f"y_{t}"]), "episode_id", "gf")
            rows.append(dict(threshold=thr, n_questions=len(d), coverage=len(d) / len(q), type=t, **r))
    return pd.DataFrame(rows)


def sens_soft(q):
    rows = []
    for t in T:
        d = q.assign(y=q[f"p_{t}"])
        try:
            r = smf.glm("y ~ gf + C(host_id)", data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["episode_id"].astype("category").cat.codes})
            b, se = r.params["gf"], r.bse["gf"]
            rows.append(dict(type=t, coef=b, se=se, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=r.pvalues["gf"]))
        except Exception as e:
            rows.append(dict(type=t, err=type(e).__name__))
    return pd.DataFrame(rows)


def sens_mi(q, rng, types=("personal", "professional")):
    P = q[[f"p_{t}" for t in T]].values
    cum = P.cumsum(1)
    ests = {t: [] for t in types}
    for _ in range(N_MI):
        u = rng.random(len(q))[:, None]
        lab = (u > cum).sum(1).clip(max=len(T) - 1)
        for t in types:
            d = q.assign(y=(lab == T.index(t)).astype(int))
            r = gee_fit("y ~ gf + C(host_id)", d, "episode_id", "gf")
            if r["ok"]:
                ests[t].append((r["coef"], r["se"] ** 2))
    rows = []
    for t, v in ests.items():
        b = np.array([x[0] for x in v]); U = np.array([x[1] for x in v]); m = len(b)
        qbar, ubar, bvar = b.mean(), U.mean(), b.var(ddof=1)
        tot = ubar + (1 + 1 / m) * bvar
        se = np.sqrt(tot)
        df = (m - 1) * (1 + ubar / ((1 + 1 / m) * bvar)) ** 2 if bvar > 0 else 1e6
        p = 2 * (1 - stats.t.cdf(abs(qbar / se), df))
        rows.append(dict(type=t, n_imputations=m, coef=qbar, se=se, OR=np.exp(qbar), lo=np.exp(qbar - stats.t.ppf(.975, df) * se),
                         hi=np.exp(qbar + stats.t.ppf(.975, df) * se), p=p, between_var_share=(1 + 1 / m) * bvar / tot))
    return pd.DataFrame(rows)


def corrected_diff(q, conf_df, rng):
    """Host-standardised female-minus-male type-share difference, observed vs
    misclassification-corrected, with bootstrap over episodes and over the
    confusion-matrix sample."""
    cd = conf_df[conf_df.true.isin(T) & conf_df.pred.isin(T)]
    cnt = pd.crosstab(cd.true, cd.pred).reindex(index=T, columns=T, fill_value=0).values + 0.5   # Laplace
    M0 = cnt / cnt.sum(1, keepdims=True)                    # P(pred | true)
    hosts = [h for h, g in q.groupby("host_id") if g.gf.nunique() == 2]
    qh = q[q.host_id.isin(hosts)]
    ep = qh.groupby("episode_id").agg(host_id=("host_id", "first"), gf=("gf", "first"))
    ec = qh.groupby("episode_id")[[f"y_{t}" for t in T]].sum()
    en = qh.groupby("episode_id").size()
    by = {(h, g): ep.index[(ep.host_id == h) & (ep.gf == g)].values for h in hosts for g in (0, 1)}
    wts = {h: (qh.host_id == h).sum() for h in hosts}
    wsum = sum(wts.values())

    def est(M, resample):
        diff_obs, diff_cor = np.zeros(6), np.zeros(6)
        for h in hosts:
            o = {}
            for g in (0, 1):
                ids = by[(h, g)]
                if resample:
                    ids = rng.choice(ids, len(ids))
                o[g] = ec.loc[ids].sum().values / en.loc[ids].sum()
            c = {g: np.linalg.solve(M.T, o[g]) for g in (0, 1)}
            c = {g: np.clip(v, 0, None) / np.clip(v, 0, None).sum() for g, v in c.items()}
            w = wts[h] / wsum
            diff_obs += w * (o[1] - o[0])
            diff_cor += w * (c[1] - c[0])
        return diff_obs, diff_cor

    obs, cor = est(M0, False)
    bo, bc = [], []
    for _ in range(B_BOOT):
        Mb = np.array([rng.multinomial(int(cnt[i].sum()), M0[i]) for i in range(6)]) + 0.5
        Mb = Mb / Mb.sum(1, keepdims=True)
        a, b = est(Mb, True)
        bo.append(a); bc.append(b)
    bo, bc = np.array(bo), np.array(bc)
    rows = []
    for i, t in enumerate(T):
        rows.append(dict(type=t, observed_diff=obs[i], obs_lo=np.percentile(bo[:, i], 2.5), obs_hi=np.percentile(bo[:, i], 97.5),
                         corrected_diff=cor[i], cor_lo=np.percentile(bc[:, i], 2.5), cor_hi=np.percentile(bc[:, i], 97.5),
                         n_hosts=len(hosts), n_conf=int(len(cd))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pool", default="results/question_type_v2/pool_host_questions.csv")
    ap.add_argument("--episodes", default="results/features/all_episodes.csv")
    ap.add_argument("--transcript_dir", default="data/outputs/whisperx")
    ap.add_argument("--confusion", required=True, help="CSV with columns true,pred from held-out data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log = []

    pool = pd.read_csv(args.pool, usecols=["qid", "episode_id", "host_id"])
    allq = pool.merge(pd.read_csv(args.pred, usecols=["qid"]), on="qid")
    coverage(args, allq, log)
    q = load(args)
    log.append(f"QUESTIONS in gender tests: {len(q)} from {q.episode_id.nunique()} episodes, {q.host_id.nunique()} host ids")
    shares(q, args)
    ci = episode_boot_ci(q, "guest_gender"); ci.to_csv(f"{args.out_dir}/shares_guest_bootci.csv", index=False)
    main_df = main_models(q); main_df.to_csv(f"{args.out_dir}/gee_main.csv", index=False)
    within_host(q, rng).to_csv(f"{args.out_dir}/within_host.csv", index=False)
    sens_threshold(q).to_csv(f"{args.out_dir}/sens_threshold.csv", index=False)
    sens_soft(q).to_csv(f"{args.out_dir}/sens_soft.csv", index=False)
    sens_mi(q, rng).to_csv(f"{args.out_dir}/sens_mi.csv", index=False)
    corrected_diff(q, pd.read_csv(args.confusion), rng).to_csv(f"{args.out_dir}/sens_corrected.csv", index=False)
    open(f"{args.out_dir}/analysis_log.txt", "w").write("\n".join(log) + "\n")
    print("\n".join(log))
    print(main_df[main_df.model == "M2_hostFE_episode_cluster"][["type", "OR", "lo", "hi", "p", "bonferroni_sig"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
