from scipy.stats import kruskal
import pandas as pd

df = pd.read_csv("/home/sr5868/listening-between-the-lines/results/dyads/dyad_analysis_with_asymmetry.csv")

metrics = [
    "host_speaking_share", "dominance_ratio",
    "hedge_asymmetry", "booster_asymmetry", "polite_asymmetry", "directive_asymmetry",
]

results = []
for m in metrics:
    groups = [g[m].dropna().values for _, g in df.groupby("dyad")]
    stat, p = kruskal(*groups)
    results.append({"metric": m, "H_statistic": stat, "p_value": p})
    print(f"{m:25s} H={stat:.3f}  p={p:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/sr5868/listening-between-the-lines/results/stat_analysis/kruskal_wallis_results.csv", index=False)