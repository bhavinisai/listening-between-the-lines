import pandas as pd
import matplotlib.pyplot as plt

# Read back in, respecting the multi-level column header written by median_mad_by_dyad()
df = pd.read_csv("/home/sr5868/listening-between-the-lines/results/dyad_median_mad_stats.csv", header=[0, 1], index_col=0)

# Pick which metrics you want to plot (must match METRICS/ASYMMETRY_PAIRS names from the script)
metrics_to_plot = [
    "dominance_ratio",
    "hedge_asymmetry",
    "booster_asymmetry",
    "polite_asymmetry",
    "directive_asymmetry",
]

dyad_order = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
dyad_labels = ["MM", "MF", "FM", "FF"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4), sharey=False)

for ax, metric in zip(axes, metrics_to_plot):
    medians = df.loc[dyad_order, (metric, "median")]
    mads = df.loc[dyad_order, (metric, "mad")]

    ax.bar(dyad_labels, medians, yerr=mads, capsize=4, color="#4C72B0")
    ax.set_title(metric)
    ax.axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("/home/sr5868/listening-between-the-lines/results/dyad_median_mad_barplot.png", dpi=150)
plt.show()