#!/usr/bin/env python3
"""
Step 1 of the topic-control revision.

(a) Recomputes every topic-control number printed in Report 1 (pp. 15-19,
    Tables 10-13, Figs. 9-12) from the saved Report 1 outputs and compares
    them with the printed values  -> verification/report1_number_check.csv
(b) Measurement diagnostics for the original definition (thresholds, ties,
    empty turns, question-answer structure, turn length, embedding check,
    host composition)                -> diagnostics/*.csv

Read-only with respect to results/topic_control/. Usage:
    python src/topic_control_revision/01_verify_and_diagnose.py
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import (DIAG, MIXED, ORIG_EMB_STATS, ORIG_EMB_TURNS, ORIG_STATS,
                    ORIG_TURNS, VERIFY, DATA, build_pairs_with_text, load_original_mixed)


def coef_from_notes(notes):
    m = re.search(r"coef=(-?[\d.]+)", notes)
    return float(m.group(1))


def threshold_bootstrap_original(mixed):
    """Exact replica of make_topic_control_figures.threshold_curves (500 episode
    resamples, seed 42) -- the procedure behind Report 1 Table 11."""
    thresholds = np.round(np.arange(0.01, 0.31, 0.01), 2)
    eps = mixed["episode_id"].unique()
    idx = {e: i for i, e in enumerate(eps)}
    above = {g: np.zeros((len(eps), len(thresholds))) for g in ["female", "male"]}
    total = {g: np.zeros(len(eps)) for g in ["female", "male"]}
    for (e, g), sims in mixed.groupby(["episode_id", "initiator_gender"])["similarity"]:
        above[g][idx[e]] = (sims.values[:, None] > thresholds[None, :]).sum(axis=0)
        total[g][idx[e]] = len(sims)
    rate = {g: above[g].sum(0) / total[g].sum() for g in above}
    rng = np.random.default_rng(42)
    boot = {g: [] for g in above}
    for _ in range(500):
        s = rng.integers(0, len(eps), len(eps))
        for g in above:
            boot[g].append(above[g][s].sum(0) / total[g][s].sum())
    diff = (np.array(boot["male"]) - np.array(boot["female"])) * 100
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    return thresholds, rate, lo, hi


def verify(m, turns_all, st, est):
    rows = []

    def add(item, where, printed, value, tol):
        ok = abs(printed - value) <= tol
        rows.append({"item": item, "report1_location": where, "printed": printed,
                     "recomputed": round(value, 4), "tolerance": tol, "match": ok})

    n_ep = m["episode_id"].nunique()
    add("Mixed-gender episodes", "p.15", 131, n_ep, 0)
    add("Mixed-gender speakers", "p.15", 262, m.groupby("episode_id")["initiator_speaker"].nunique().sum(), 0)
    add("Initiations (mixed)", "p.15", 17046, len(m), 0)
    g = m.groupby("initiator_gender")["followed_up"]
    add("Male initiations", "p.15/Table 10", 8742, g.size()["male"], 0)
    add("Female initiations", "p.15/Table 10", 8304, g.size()["female"], 0)
    pooled = g.mean() * 100
    add("Pooled follow-up, female (%)", "Table 10", 37.7, pooled["female"], 0.05)
    add("Pooled follow-up, male (%)", "Table 10", 42.2, pooled["male"], 0.05)
    add("Pooled gap male-female (pp)", "Table 10", 4.5, pooled["male"] - pooled["female"], 0.05)
    for gg, lo, hi in [("female", 36.7, 38.7), ("male", 41.2, 43.2)]:
        p, n = pooled[gg] / 100, g.size()[gg]
        se = np.sqrt(p * (1 - p) / n) * 100
        add(f"Pooled CI lower, {gg} (naive binomial)", "Table 10", lo, pooled[gg] - 1.96 * se, 0.05)
        add(f"Pooled CI upper, {gg} (naive binomial)", "Table 10", hi, pooled[gg] + 1.96 * se, 0.05)
    pe = m.groupby(["episode_id", "initiator_gender"])["followed_up"].mean().unstack() * 100
    for gg, mean, lo, hi, med in [("female", 42.6, 40.6, 44.6, 43.3), ("male", 41.5, 39.8, 43.2, 40.6)]:
        x = pe[gg]
        h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        add(f"Per-episode mean, {gg} (%)", "Table 10", mean, x.mean(), 0.05)
        add(f"Per-episode CI lower, {gg}", "Table 10", lo, x.mean() - h, 0.05)
        add(f"Per-episode CI upper, {gg}", "Table 10", hi, x.mean() + h, 0.05)
        add(f"Per-episode median, {gg} (%)", "Table 10", med, x.median(), 0.05)
    d = pe["male"] - pe["female"]
    h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    add("Per-episode gap (pp)", "Table 10/13", -1.2, d.mean(), 0.05)
    add("Per-episode gap CI lower", "Table 10/13", -4.1, d.mean() - h, 0.05)
    add("Per-episode gap CI upper", "Table 10/13", 1.7, d.mean() + h, 0.05)

    thr, rate, lo, hi = threshold_bootstrap_original(m)
    printed11 = {0.02: (33.4, 37.8, 4.4, 1.9, 7.0), 0.05: (21.7, 25.3, 3.5, 1.1, 5.9),
                 0.10: (11.9, 15.2, 3.4, 1.6, 4.9), 0.15: (7.6, 10.2, 2.6, 1.3, 3.8),
                 0.20: (5.1, 7.5, 2.4, 1.4, 3.3), 0.30: (2.9, 4.1, 1.3, 0.5, 1.9)}
    for t, (f, mm, gap, l, u) in printed11.items():
        k = int(np.argmin(np.abs(thr - t)))
        add(f"Table 11 female @{t}", "Table 11", f, rate["female"][k] * 100, 0.05)
        add(f"Table 11 male @{t}", "Table 11", mm, rate["male"][k] * 100, 0.05)
        add(f"Table 11 gap @{t}", "Table 11", gap, (rate["male"][k] - rate["female"][k]) * 100, 0.06)
        add(f"Table 11 gap CI lower @{t}", "Table 11", l, lo[k], 0.05)
        add(f"Table 11 gap CI upper @{t}", "Table 11", u, hi[k], 0.05)

    def srow(test):
        return st[st["test"] == test].iloc[0]

    for term, printed, lo_, hi_ in [("is_male_initiator", 0.85, 0.76, 0.94),
                                    ("is_host_initiator", 2.20, 1.99, 2.44)]:
        r = srow(f"gee_followedup_{term}")
        c = coef_from_notes(r["notes"])
        se = float(re.search(r"se=([\d.]+)", r["notes"]).group(1))
        add(f"GEE OR {term}", "Table 12", printed, np.exp(c), 0.005)
        add(f"GEE OR CI lower {term}", "Table 12", lo_, np.exp(c - 1.96 * se), 0.005)
        add(f"GEE OR CI upper {term}", "Table 12", hi_, np.exp(c + 1.96 * se), 0.005)
    add("GEE p male", "Table 12", 0.002, srow("gee_followedup_is_male_initiator")["p_value"], 0.0005)
    r = srow("gee_followedup_initiator_word_count")
    add("GEE OR +100 words (from rounded coef)", "Table 12", 2.64, np.exp(coef_from_notes(r["notes"]) * 100), 0.005)
    add("Mixed model p male", "Table 12", 0.42, srow("mixedlm_similarity_is_male_initiator")["p_value"], 0.005)
    add("Mixed model beta host", "Table 12", 0.019, coef_from_notes(srow("mixedlm_similarity_is_host_initiator")["notes"]), 0.0006)
    add("Mixed model p words", "Table 12", 0.53, srow("mixedlm_similarity_initiator_word_count")["p_value"], 0.005)
    add("Paired t (TF-IDF)", "Table 13", -0.78, srow("paired_ttest_episode_followup_rate")["statistic"], 0.005)
    add("Paired t p (TF-IDF)", "Table 13", 0.44, srow("paired_ttest_episode_followup_rate")["p_value"], 0.005)
    add("Permutation p (TF-IDF)", "Table 13", 0.44, srow("permutation_test_episode_followup_rate")["p_value"], 0.005)
    e1 = est[est["test"] == "embeddings_paired_ttest_episode_followup_rate"].iloc[0]
    add("Paired t (embeddings)", "Table 13", -0.78, e1["statistic"], 0.005)
    add("Paired t p (embeddings)", "Table 13", 0.43, e1["p_value"], 0.005)
    e2 = est[est["test"] == "embeddings_permutation_test_episode_followup_rate"].iloc[0]
    add("Permutation p (embeddings)", "Table 13", 0.44, e2["p_value"], 0.005)
    emb = pd.read_csv(ORIG_EMB_TURNS)
    emb = emb[emb["dyad"].isin(MIXED)]
    epe = emb.groupby(["episode_id", "initiator_gender"])["followed_up"].mean().unstack() * 100
    add("Embeddings per-episode female (%)", "Table 13", 48.4, epe["female"].mean(), 0.05)
    add("Embeddings per-episode male (%)", "Table 13", 47.2, epe["male"].mean(), 0.05)
    de = epe["male"] - epe["female"]
    he = 1.96 * de.std(ddof=1) / np.sqrt(len(de))
    add("Embeddings gap CI lower", "Table 13", -4.1, de.mean() - he, 0.05)
    add("Embeddings gap CI upper", "Table 13", 1.8, de.mean() + he, 0.05)

    male = m[m["initiator_gender"] == "male"]
    female = m[m["initiator_gender"] == "female"]
    add("Male initiations from hosts (%)", "p.17", 78, (male["initiator_role"] == "HOST").mean() * 100, 0.5)
    add("Female initiations from hosts (%)", "p.17", 30, (female["initiator_role"] == "HOST").mean() * 100, 0.5)
    fr = m.groupby(["episode_id", "initiator_role", "initiator_gender"])["followed_up"].mean().reset_index()
    for role, gg, printed in [("HOST", "female", 49), ("HOST", "male", 46), ("GUEST", "female", 36), ("GUEST", "male", 37)]:
        x = fr[(fr.initiator_role == role) & (fr.initiator_gender == gg)]["followed_up"].mean() * 100
        add(f"Fig. 10 {role.lower()} {gg} (%)", "Fig. 10", printed, x, 0.5)
    add("All-dyad episodes", "p.18", 414, turns_all["episode_id"].nunique(), 0)
    add("All-dyad initiations", "p.18", 42320, len(turns_all), 0)
    dr = turns_all.groupby(["dyad", "initiator_role", "episode_id"])["followed_up"].mean().groupby(["dyad", "initiator_role"]).mean() * 100
    for dy, h_, g_ in [("MALE->MALE", 51, 41), ("MALE->FEMALE", 46, 36), ("FEMALE->MALE", 49, 37), ("FEMALE->FEMALE", 52, 38)]:
        add(f"Fig. 12 {dy} host (%)", "Fig. 12", h_, dr[(dy, "HOST")], 0.5)
        add(f"Fig. 12 {dy} guest (%)", "Fig. 12", g_, dr[(dy, "GUEST")], 0.5)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(VERIFY, "report1_number_check.csv"), index=False)
    print(f"[verify] {out['match'].sum()}/{len(out)} printed Report 1 values reproduced")
    if not out["match"].all():
        print(out[~out["match"]].to_string())
    return out


def diagnostics(m, rb, turns_all):
    rows = []

    def add(section, item, value, note=""):
        rows.append({"section": section, "item": item, "value": value, "note": note})

    ep_turns = pd.read_csv(os.path.join(DATA, "mixed_episode_turn_counts.csv"))
    n_turns = ep_turns["n_turns_after_filtering"].sum()
    ep_cut = rb.groupby("episode_id")[["followup_threshold", "init_cutoff"]].first()
    add("thresholds", "Mixed episodes", len(ep_cut))
    add("thresholds", "Mixed episodes with follow-up cutoff = 0", int((ep_cut["followup_threshold"] == 0).sum()))
    add("thresholds", "Mixed episodes with initiation cutoff = 0", int((ep_cut["init_cutoff"] == 0).sum()))
    add("thresholds", "Median follow-up cutoff (mixed)", float(ep_cut["followup_threshold"].median()))
    allc = turns_all.groupby("episode_id")["followup_threshold"].first()
    add("thresholds", "All 414 episodes with follow-up cutoff = 0", int((allc == 0).sum()), f"of {len(allc)}")
    add("thresholds", "Follow-up similarities exactly 0 (mixed, %)", round((rb["similarity"] == 0).mean() * 100, 1))
    add("thresholds", "Followed up = any shared content token, share of pairs (%)",
        round(rb["followup_threshold"].eq(0).mean() * 100, 1),
        "pairs in episodes whose cutoff is 0: followed_up == (similarity > 0)")

    add("initiation", "Turns after merging + backchannel removal (mixed)", int(n_turns))
    add("initiation", "Initiations with a response (mixed)", len(rb))
    add("initiation", "Initiations as share of turns (%)", round(len(rb) / n_turns * 100, 1),
        "a strict median split would give <= 50%")
    ties = rb["incoming_similarity"] == rb["init_cutoff"]
    add("initiation", "Initiations exactly tied with the cutoff (%)", round(ties.mean() * 100, 1),
        "these enter only because the rule is <= median")
    add("initiation", "Initiations with incoming similarity 0 (%)", round((rb["incoming_similarity"] == 0).mean() * 100, 1))
    add("initiation", "First turn of episode counted as initiation", int(rb["incoming_similarity"].isna().sum()))

    add("empty_turns", "Initiating turns with no content tokens", int((rb["initiator_content_tokens"] == 0).sum()),
        "similarity is 0 by construction: always an initiation, never followed up")
    add("empty_turns", "Responses with no content tokens", int((rb["responder_content_tokens"] == 0).sum()),
        "never count as a follow-up")
    add("empty_turns", "Initiating turns with <= 3 words", int((rb["initiator_word_count"] <= 3).sum()))
    add("empty_turns", "Responses with <= 3 words", int((rb["responder_word_count"] <= 3).sum()))
    pd.DataFrame(rows).to_csv(os.path.join(DIAG, "measurement_diagnostics.csv"), index=False)

    # Question-answer structure
    rb = rb.copy()
    rb["init_is_question"] = rb["init_text"].str.contains(r"\?", regex=True)
    rb["prev_is_question"] = rb["prev_text"].fillna("").str.contains(r"\?", regex=True)
    qa = (rb.groupby(["initiator_role", "init_is_question"])
          .agg(n=("followed_up", "size"), followup_rate=("followed_up", "mean"),
               median_resp_words=("responder_word_count", "median"),
               median_init_words=("initiator_word_count", "median"))
          .reset_index())
    qa["share_of_role"] = qa["n"] / qa.groupby("initiator_role")["n"].transform("sum")
    qa.to_csv(os.path.join(DIAG, "question_answer_structure.csv"), index=False)
    ans = (rb.groupby(["initiator_role", "prev_is_question"])
           .agg(n=("followed_up", "size"), followup_rate=("followed_up", "mean")).reset_index())
    ans["share_of_role"] = ans["n"] / ans.groupby("initiator_role")["n"].transform("sum")
    ans.to_csv(os.path.join(DIAG, "initiations_that_answer_a_question.csv"), index=False)

    # Length
    m = m.copy()
    m["resp_q"] = pd.qcut(m["responder_word_count"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    m["init_q"] = pd.qcut(m["initiator_word_count"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    lr = []
    for var, q in [("responder_word_count", "resp_q"), ("initiator_word_count", "init_q")]:
        t = (m.groupby([q, "initiator_role"], observed=True)
             .agg(n=("followed_up", "size"), followup_rate=("followed_up", "mean"),
                  min_words=(var, "min"), max_words=(var, "max")).reset_index()
             .rename(columns={q: "quintile"}))
        t.insert(0, "length_variable", var)
        lr.append(t)
    pd.concat(lr).to_csv(os.path.join(DIAG, "followup_by_length_quintile.csv"), index=False)
    sk = []
    for var in ["initiator_word_count", "responder_word_count"]:
        x = m[var]
        sk.append({"variable": var, "median": x.median(), "mean": x.mean(), "p99": x.quantile(0.99),
                   "max": x.max(), "skewness": stats.skew(x), "skewness_log1p": stats.skew(np.log1p(x)),
                   "spearman_with_followed_up": stats.spearmanr(x, m["followed_up"])[0]})
    pd.DataFrame(sk).to_csv(os.path.join(DIAG, "word_count_distribution.csv"), index=False)
    lw = (m.groupby(["dyad", "initiator_role", "initiator_gender"])
          .agg(n=("followed_up", "size"), median_init_words=("initiator_word_count", "median"),
               mean_init_words=("initiator_word_count", "mean"),
               median_resp_words=("responder_word_count", "median"),
               followup_rate_pooled=("followed_up", "mean")).reset_index())
    lw.to_csv(os.path.join(DIAG, "cell_sizes_and_lengths.csv"), index=False)

    # Embedding check comparability
    emb = pd.read_csv(ORIG_EMB_TURNS)
    em = emb[emb["dyad"].isin(MIXED)]
    comp = pd.DataFrame([
        {"measure": "TF-IDF (Report 1 main)", "all_pairs": len(turns_all), "mixed_pairs": len(m),
         "mixed_male": int((m.initiator_gender == "male").sum()), "mixed_female": int((m.initiator_gender == "female").sum()),
         "initiation_rule": "TF-IDF incoming similarity <= episode median",
         "followup_rule": "TF-IDF outgoing similarity > episode median",
         "tests_run": "naive tests, mixed model, GEE, VB-GLMM, paired t, Wilcoxon, permutation"},
        {"measure": "Embeddings (Report 1 robustness)", "all_pairs": len(emb), "mixed_pairs": len(em),
         "mixed_male": int((em.initiator_gender == "male").sum()), "mixed_female": int((em.initiator_gender == "female").sum()),
         "initiation_rule": "MiniLM incoming cosine <= episode median (different initiation set)",
         "followup_rule": "MiniLM outgoing cosine > episode median",
         "tests_run": "paired t and permutation only (role-unadjusted)"},
    ])
    comp.to_csv(os.path.join(DIAG, "embedding_check_comparability.csv"), index=False)

    # Host composition
    hc = (m[m["initiator_role"] == "HOST"].groupby(["dyad", "host_id"])
          .agg(episodes=("episode_id", "nunique"), host_initiations=("followed_up", "size"))
          .reset_index().sort_values(["dyad", "episodes"], ascending=[True, False]))
    allp = m.groupby("host_id").size().rename("all_initiations_in_host_episodes")
    hc = hc.merge(allp, left_on="host_id", right_index=True)
    hc.to_csv(os.path.join(DIAG, "host_composition.csv"), index=False)
    print("[diag] wrote measurement diagnostics to", DIAG)


def main():
    m = load_original_mixed()
    turns_all = pd.read_csv(ORIG_TURNS)
    st = pd.read_csv(ORIG_STATS)
    est = pd.read_csv(ORIG_EMB_STATS)
    verify(m, turns_all, st, est)
    rb = build_pairs_with_text()
    diagnostics(m, rb, turns_all)


if __name__ == "__main__":
    main()
