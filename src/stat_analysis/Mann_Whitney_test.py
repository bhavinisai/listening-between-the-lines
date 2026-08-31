import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("/home/sr5868/listening-between-the-lines/results/dyads/dyad_analysis_with_asymmetry.csv")

# Only the metrics that were significant in the Kruskal-Wallis omnibus test
# (directive_asymmetry excluded since p=0.39, not significant)
metrics = [
    "host_speaking_share",
    "dominance_ratio",
    "hedge_asymmetry",
    "booster_asymmetry",
    "polite_asymmetry",
]

dyads = sorted(df["dyad"].unique())
pairs = list(combinations(dyads, 2))

results = []
for m in metrics:
    pvals = []
    pair_data = []
    for a, b in pairs:
        x = df[df.dyad == a][m].dropna()
        y = df[df.dyad == b][m].dropna()
        stat, p = mannwhitneyu(x, y, alternative="two-sided")

        # effect size: rank-biserial correlation
        n1, n2 = len(x), len(y)
        rank_biserial = 1 - (2 * stat) / (n1 * n2)

        pvals.append(p)
        pair_data.append({
            "metric": m,
            "group_a": a,
            "group_b": b,
            "median_a": x.median(),
            "median_b": y.median(),
            "U_statistic": stat,
            "raw_p": p,
            "rank_biserial_effect_size": rank_biserial,
        })

    # correct for multiple comparisons within this metric's 6 pairwise tests
    corrected = multipletests(pvals, method="fdr_bh")[1]
    for row, corr_p in zip(pair_data, corrected):
        row["corrected_p"] = corr_p
        row["significant_after_correction"] = corr_p < 0.05
        results.append(row)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["metric", "corrected_p"])
results_df.to_csv("/home/sr5868/listening-between-the-lines/results/stat_analysis/pairwise_dyad_tests.csv", index=False)

print(results_df.to_string(index=False))