#!/usr/bin/env python3
"""
Emotional dynamics across gender dyads, from the per-turn scores written by
score_corpus.py.

Per turn: valence in [-1, 1] (P(pos) - P(neg)) and arousal in [0, 1].
Adjacent turns (consecutive in the filtered sequence, so always two different
speakers) form the unit for the three behaviours:

  Mirroring    how closely the responder's tone tracks the previous turn:
               within-episode slope / correlation of responder tone on previous
               tone, and a pooled GEE (clustered by episode) with a
               previous-tone x previous-speaker-gender interaction.
  Escalation   arousal rising across turns (step = rise >= 0.5 SD; run = >= 3
               turns of strictly rising arousal with total rise >= 1 SD) and the
               valence analogue ("negative spiral").
  Recovery     after a negative turn, does the next turn lift the tone, and who
               (host / guest, male / female) delivers the first non-negative turn
               within a 3-turn window.

Inference uses episode as the cluster (GEE, robust SEs) or episode-level
summaries (Kruskal-Wallis / Mann-Whitney / Wilcoxon). Only male and female
speakers enter the tests.

Usage:
    python src/sentiment/sentiment_dynamics.py --scores results/sentiment/turn_scores.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
STATS = []
OUT = "results/sentiment"


def record(analysis, test, n, stat, p, effect=None, note="", headline=False):
    STATS.append(dict(analysis=analysis, test=test, n=n, statistic=stat, p=p, effect=effect, note=note,
                      headline=headline))


def gee(formula, data, family):
    return smf.gee(formula, groups="episode_id", data=data, family=family,
                   cov_struct=sm.cov_struct.Independence()).fit()


def record_gee(analysis, res, term, note="", headline=False, logistic=False):
    b, p = res.params[term], res.pvalues[term]
    record(analysis, "GEE " + term, int(res.nobs), res.tvalues[term], p,
           np.exp(b) if logistic else b, note + (" (odds ratio)" if logistic else " (coef)"), headline)


def build_pairs(df):
    df = df.sort_values(["episode_id", "turn_idx"]).reset_index(drop=True)
    g = df.groupby("episode_id")
    for c in ["valence", "arousal", "role", "gender", "speaker", "valence_label", "turn_idx"]:
        df["prev_" + c] = g[c].shift(1)
        df["next_" + c] = g[c].shift(-1)
    pairs = df[df.prev_turn_idx.notna() & (df.turn_idx - df.prev_turn_idx == 1) & (df.role != df.prev_role)].copy()
    pairs = pairs.rename(columns={"valence": "v_cur", "prev_valence": "v_prev", "arousal": "a_cur",
                                  "prev_arousal": "a_prev", "role": "resp_role", "gender": "resp_gender",
                                  "prev_gender": "prev_gender"})
    pairs["dv"] = pairs.v_cur - pairs.v_prev
    pairs["da"] = pairs.a_cur - pairs.a_prev
    pairs["abs_dv"] = pairs.dv.abs()
    pairs["same_polarity"] = (np.sign(pairs.v_cur.round(1)) == np.sign(pairs.v_prev.round(1))).astype(int)
    return df, pairs


def mixed_only(d, *cols):
    for c in cols:
        d = d[d[c].isin(["male", "female"])]
    return d


def describe(df):
    """Episode-level mean tone by dyad and role; host vs guest within episodes."""
    d = mixed_only(df, "gender")
    ep = (d.groupby(["episode_id", "dyad", "role"]).agg(valence=("valence", "mean"), arousal=("arousal", "mean"),
                                                     neg_share=("valence_label", lambda s: (s == "negative").mean()),
                                                     pos_share=("valence_label", lambda s: (s == "positive").mean()),
                                                     turns=("valence", "size")).reset_index())
    ep = ep[ep.turns >= 5]
    ep.to_csv(os.path.join(OUT, "episode_role_tone.csv"), index=False)
    for m in ["valence", "arousal", "neg_share", "pos_share"]:
        for role, gg in ep.groupby("role"):
            groups = [x[m].values for _, x in gg.groupby("dyad") if len(x) > 2 and x.dyad.iloc[0] != "MALE->UNKNOWN"]
            h, p = stats.kruskal(*groups)
            record("tone by dyad", f"Kruskal-Wallis {m} ({role} turns, 4 dyads)", len(gg), h, p, headline=m in ("valence", "arousal"))
        w = ep.pivot_table(index="episode_id", columns="role", values=m).dropna()
        s, p = stats.wilcoxon(w.HOST, w.GUEST)
        record("host vs guest", f"Wilcoxon signed-rank {m}, host vs guest (paired by episode)", len(w), s, p,
               (w.HOST - w.GUEST).median(), "median host - guest")
    # speaker gender main effects within role (mixed-effects: random intercept per episode)
    for m in ["valence", "arousal"]:
        r = smf.mixedlm(f"{m} ~ C(role) * C(gender)", d, groups=d["episode_id"]).fit(reml=False)
        for t in [x for x in r.params.index if "gender" in x]:
            record("tone by dyad", f"MixedLM {m}: {t}", len(d), r.tvalues[t], r.pvalues[t], r.params[t], "turn-level, episode random intercept")
    return ep


def mirroring(pairs):
    out = []
    for tone, (cur, prev) in {"valence": ("v_cur", "v_prev"), "arousal": ("a_cur", "a_prev")}.items():
        for role in ["GUEST", "HOST"]:
            d = mixed_only(pairs[pairs.resp_role == role], "prev_gender", "resp_gender")
            res = gee(f"{cur} ~ {prev} * C(prev_gender, Treatment('female')) + {prev} * C(resp_gender, Treatment('female'))",
                      d, sm.families.Gaussian())
            t = f"{prev}:C(prev_gender, Treatment('female'))[T.male]"
            record_gee(f"mirroring ({tone})", res, t,
                       f"{role} responding; extra slope on previous tone when previous speaker is male vs female",
                       headline=True)
            t2 = f"{prev}:C(resp_gender, Treatment('female'))[T.male]"
            record_gee(f"mirroring ({tone})", res, t2, f"{role} responding; extra slope when the responder is male")
            for pg in ["male", "female"]:
                x = d[d.prev_gender == pg]
                r = stats.spearmanr(x[prev], x[cur])
                out.append(dict(tone=tone, resp_role=role, prev_gender=pg, n=len(x), spearman=r.statistic,
                                mean_abs_diff=(x[cur] - x[prev]).abs().mean(),
                                slope=sm.OLS(x[cur], sm.add_constant(x[prev])).fit().params[prev]))
            # per-episode correlation, compared across previous-speaker gender
            ep = (d.groupby(["episode_id", "dyad", "prev_gender"]).apply(
                lambda x: stats.spearmanr(x[prev], x[cur]).statistic if len(x) >= 15 and x[prev].nunique() > 2 else np.nan)
                .rename("rho").reset_index().dropna())
            ep["tone"], ep["resp_role"] = tone, role
            out_ep.append(ep)
            a, b = ep[ep.prev_gender == "male"].rho, ep[ep.prev_gender == "female"].rho
            u, p = stats.mannwhitneyu(a, b)
            record(f"mirroring ({tone})", f"Mann-Whitney per-episode rho, {role} responding, prev speaker male vs female",
                   len(ep), u, p, a.median() - b.median(), f"median rho male {a.median():.3f} vs female {b.median():.3f}")
            groups = [x.rho.values for _, x in ep.groupby("dyad") if len(x) > 2]
            h, p = stats.kruskal(*groups)
            record(f"mirroring ({tone})", f"Kruskal-Wallis per-episode rho by dyad, {role} responding", len(ep), h, p)
    return pd.DataFrame(out)


out_ep = []


def escalation(df, pairs, sd_a, sd_v):
    # step level
    pairs["esc_step"] = (pairs.da >= 0.5 * sd_a).astype(int)
    pairs["neg_step"] = (pairs.dv <= -0.5 * sd_v).astype(int)
    hi = pairs.a_prev >= pairs.a_prev.quantile(0.9)
    pairs["hi_prev"] = hi.astype(int)
    pairs["hi_cur"] = (pairs.a_cur >= pairs.a_cur.quantile(0.9)).astype(int)
    for y, label in [("esc_step", "arousal escalation step"), ("neg_step", "valence drop step")]:
        d = mixed_only(pairs, "prev_gender", "resp_gender")
        res = gee(f"{y} ~ C(resp_role) + C(prev_gender, Treatment('female')) + C(resp_gender, Treatment('female'))",
                  d, sm.families.Binomial())
        for t in [x for x in res.params.index if "gender" in x or "role" in x]:
            record_gee(f"escalation ({label})", res, t, "step-level, adjacent pairs", headline="prev_gender" in t, logistic=True)
        ep = d.groupby(["episode_id", "dyad"]).agg(rate=(y, "mean"), n=(y, "size")).reset_index()
        ep = ep[ep.n >= 30]
        h, p = stats.kruskal(*[x.rate.values for _, x in ep.groupby("dyad") if len(x) > 2])
        record(f"escalation ({label})", "Kruskal-Wallis per-episode step rate by dyad", len(ep), h, p, headline=True)
    d = mixed_only(pairs[pairs.hi_prev == 1], "prev_gender", "resp_gender")
    res = gee("hi_cur ~ C(resp_role) + C(prev_gender, Treatment('female')) + C(resp_gender, Treatment('female'))",
              d, sm.families.Binomial())
    for t in [x for x in res.params.index if "gender" in x or "role" in x]:
        record_gee("escalation (high-arousal contagion)", res, t, "P(high arousal | previous turn high arousal)", logistic=True)

    # run level: >=3 consecutive turns, each rising, total rise >= 1 SD
    runs = []
    for ep_id, g in df.groupby("episode_id"):
        g = g.reset_index(drop=True)
        for col, sd, kind, sign in [("arousal", sd_a, "arousal rise", 1), ("valence", sd_v, "valence fall", -1)]:
            x = sign * g[col].values
            i = 0
            while i < len(g) - 1:
                j = i
                while j + 1 < len(g) and x[j + 1] > x[j]:
                    j += 1
                if j - i >= 2 and x[j] - x[i] >= sd:
                    runs.append(dict(episode_id=ep_id, dyad=g.dyad[0], kind=kind, start_turn=g.turn_idx[i],
                                     length=j - i + 1, total_change=sign * (x[j] - x[i]),
                                     peak_role=g.role[j], peak_gender=g.gender[j], start_role=g.role[i]))
                i = max(j, i + 1)
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(OUT, "escalation_runs.csv"), index=False)
    turns_per_ep = df.groupby(["episode_id", "dyad"]).size().rename("turns").reset_index()
    for kind, rk in runs.groupby("kind"):
        cnt = rk.groupby("episode_id").size().rename("runs")
        e = turns_per_ep.merge(cnt, on="episode_id", how="left").fillna({"runs": 0})
        e = e[(e.turns >= 30) & (e.dyad != "MALE->UNKNOWN")]
        e["rate_per_100"] = 100 * e.runs / e.turns
        h, p = stats.kruskal(*[x.rate_per_100.values for _, x in e.groupby("dyad")])
        record(f"escalation runs ({kind})", "Kruskal-Wallis runs per 100 turns by dyad", len(e), h, p,
               note="; ".join(f"{k}: {v:.2f}" for k, v in e.groupby('dyad').rate_per_100.mean().items()), headline=True)
        e.to_csv(os.path.join(OUT, f"escalation_rate_{kind.split()[0]}.csv"), index=False)
    return pairs, runs


def recovery(df, pairs, neg_thr, lift):
    d = df.copy()
    d["neg"] = (d.valence_label == "negative") & (d.valence <= neg_thr)
    ev = []
    for ep_id, g in d.groupby("episode_id"):
        g = g.reset_index(drop=True)
        for i in np.flatnonzero(g.neg.values):
            if i + 1 >= len(g):
                continue
            nxt = g.iloc[i + 1]
            win = g.iloc[i + 1:i + 4]
            rec = win[~win.neg]
            first = rec.iloc[0] if len(rec) else None
            ev.append(dict(episode_id=ep_id, dyad=g.dyad[i], neg_role=g.role[i], neg_gender=g.gender[i],
                           neg_valence=g.valence[i], resp_role=nxt.role, resp_gender=nxt.gender,
                           next_valence=nxt.valence, next_lift=nxt.valence - g.valence[i],
                           immediate=int((not nxt.neg) and nxt.valence - g.valence[i] >= lift),
                           recovered_3=int(first is not None),
                           recoverer_role=first.role if first is not None else "none",
                           recoverer_gender=first.gender if first is not None else "none",
                           turns_to_recover=(first.turn_idx - g.turn_idx[i]) if first is not None else np.nan))
    ev = pd.DataFrame(ev)
    ev.to_csv(os.path.join(OUT, "recovery_events.csv"), index=False)
    d = mixed_only(ev, "resp_gender", "neg_gender")
    res = gee("immediate ~ C(resp_role) + C(resp_gender, Treatment('female')) + C(neg_gender, Treatment('female'))",
              d, sm.families.Binomial())
    for t in [x for x in res.params.index if "gender" in x or "role" in x]:
        record_gee("recovery (next-turn lift)", res, t, f"neg turns n={len(d)}; lift >= {lift}", headline=True, logistic=True)
    res = gee("recovered_3 ~ C(resp_role) + C(resp_gender, Treatment('female')) + C(neg_gender, Treatment('female'))",
              d, sm.families.Binomial())
    for t in [x for x in res.params.index if "gender" in x or "role" in x]:
        record_gee("recovery (within 3 turns)", res, t, "", logistic=True)
    # who delivers the first non-negative turn, by dyad
    ct = pd.crosstab(ev[ev.dyad != "MALE->UNKNOWN"].dyad, ev.recoverer_role)
    chi = stats.chi2_contingency(ct)
    record("recovery (who recovers)", "chi-square dyad x first recoverer role", int(ct.values.sum()), chi.statistic, chi.pvalue, headline=True)
    # host vs guest rate within episodes (episodes with >= 2 negatives of each side)
    rate = (ev.groupby(["episode_id", "resp_role"]).immediate.agg(["mean", "size"]).reset_index())
    rate = rate[rate["size"] >= 2].pivot(index="episode_id", columns="resp_role", values="mean").dropna()
    if len(rate) > 5:
        s, p = stats.wilcoxon(rate.HOST, rate.GUEST)
        record("recovery (host vs guest)", "Wilcoxon per-episode lift rate, host vs guest responder", len(rate), s, p,
               (rate.HOST - rate.GUEST).median(), "median host - guest")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/sentiment/turn_scores.csv")
    ap.add_argument("--neg_thr", type=float, default=-0.25, help="valence at or below which an argmax-negative turn counts as negative")
    ap.add_argument("--lift", type=float, default=0.25, help="minimum valence gain on the next turn to count as recovery")
    ap.add_argument("--suffix", default="", help="e.g. '_zs' to analyse the zero-shot column instead")
    ap.add_argument("--valence_col", default="valence")
    ap.add_argument("--out_dir", default="results/sentiment")
    args = ap.parse_args()
    global OUT
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(args.scores)
    df["valence"] = df[args.valence_col]
    if args.valence_col != "valence":
        df["valence_label"] = np.where(df.valence <= -0.25, "negative", np.where(df.valence >= 0.25, "positive", "neutral"))
    sd_a, sd_v = df.arousal.std(), df.valence.std()
    print(f"turns={len(df)} episodes={df.episode_id.nunique()} SD(arousal)={sd_a:.3f} SD(valence)={sd_v:.3f}")

    describe(df)
    df, pairs = build_pairs(df)
    mir = mirroring(pairs)
    mir.to_csv(os.path.join(OUT, "mirroring_summary.csv"), index=False)
    pd.concat(out_ep).to_csv(os.path.join(OUT, "mirroring_episode_rho.csv"), index=False)
    pairs, runs = escalation(df, pairs, sd_a, sd_v)
    pairs.to_csv(os.path.join(OUT, "adjacent_pairs.csv"), index=False)
    ev = recovery(df, pairs, args.neg_thr, args.lift)

    s = pd.DataFrame(STATS)
    s["p_holm_headline"] = np.nan
    h = s.headline
    s.loc[h, "p_holm_headline"] = multipletests(s.loc[h, "p"], method="holm")[1]
    s["p_fdr_all"] = multipletests(s.p, method="fdr_bh")[1]
    s.to_csv(os.path.join(OUT, f"sentiment_dynamics_stats{args.suffix}.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 500)
    print(s[["analysis", "test", "n", "effect", "p", "p_holm_headline", "p_fdr_all"]].round(4).to_string(index=False))
    print(f"\nnegative turns={len(ev)} ; escalation runs={len(runs)}")


if __name__ == "__main__":
    main()
