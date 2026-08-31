"""
dyad_level_features.py

Step: "Calculate dyad-level features"
Input : dyad_analysis.csv  (one row per episode, host/guest metrics side by side)
Output:
    - dyad_summary_stats.csv           (mean/median/std/count per dyad, per metric)
    - dyad_median_mad_stats.csv        (robust median/MAD/count per dyad, per metric)
    - dyad_analysis_with_asymmetry.csv (original data + host-vs-guest asymmetry columns)
    - dyad_boxplots.png                (visual check of key metrics across the 4 dyads)

Usage:
    python dyad_level_features.py \
        --input results/dyads/dyad_analysis.csv \
        --outdir results/dyads/ \
        --figdir results/figures/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation


METRICS = [
    "host_speaking_share", "guest_speaking_share", "dominance_ratio",
    "host_hedge_per_1k", "guest_hedge_per_1k",
    "host_booster_per_1k", "guest_booster_per_1k",
    "host_polite_per_1k", "guest_polite_per_1k",
    "host_directive_per_1k", "guest_directive_per_1k",
]

ASYMMETRY_PAIRS = [
    ("hedge_asymmetry", "host_hedge_per_1k", "guest_hedge_per_1k"),
    ("booster_asymmetry", "host_booster_per_1k", "guest_booster_per_1k"),
    ("polite_asymmetry", "host_polite_per_1k", "guest_polite_per_1k"),
    ("directive_asymmetry", "host_directive_per_1k", "guest_directive_per_1k"),
]

BOXPLOT_METRICS = [
    "dominance_ratio", "hedge_asymmetry", "booster_asymmetry",
    "polite_asymmetry", "directive_asymmetry", "host_speaking_share",
]

DYAD_ABBREV = {
    "MALE->MALE": "MM",
    "MALE->FEMALE": "MF",
    "FEMALE->MALE": "FM",
    "FEMALE->FEMALE": "FF",
}


def check_balance(df: pd.DataFrame) -> None:
    """Print dyad group sizes and warn if unbalanced."""
    counts = df["dyad"].value_counts()
    print("Dyad group sizes:")
    print(counts)
    if counts.min() != counts.max():
        print(f"WARNING: dyad groups are unbalanced (min={counts.min()}, max={counts.max()})")
    else:
        print(f"OK: all {len(counts)} dyad groups balanced at n={counts.min()}")


def add_asymmetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add host-minus-guest asymmetry columns. Positive = host does it more than guest."""
    df = df.copy()
    for new_col, host_col, guest_col in ASYMMETRY_PAIRS:
        df[new_col] = df[host_col] - df[guest_col]
    return df


def summarize_by_dyad(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Mean / median / std / count per dyad, per metric."""
    return df.groupby("dyad")[metrics].agg(["mean", "median", "std", "count"])


def median_mad_by_dyad(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Robust median + MAD summary per dyad, per metric (less sensitive to outliers than mean/std)."""
    agg = {}
    for m in metrics:
        agg[m] = df.groupby("dyad")[m].agg(
            median="median",
            mad=lambda x: median_abs_deviation(x),
            n="count",
        )
    return pd.concat(agg, axis=1)


def plot_boxplots(df: pd.DataFrame, metrics: list, out_path: str) -> None:
    """Grid of boxplots, one per metric, grouped by dyad (abbreviated labels)."""
    df = df.copy()
    df["dyad_short"] = df["dyad"].map(DYAD_ABBREV).fillna(df["dyad"])

    n = len(metrics)
    ncols = 3
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flat if n > 1 else [axes]

    for ax, m in zip(axes, metrics):
        df.boxplot(column=m, by="dyad_short", ax=ax)
        ax.set_title(m)
        ax.set_xlabel("")

    # hide any unused subplot axes
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved boxplots to {out_path}")


def main(input_csv: str, outdir: str, figdir: str = None):
    figdir = figdir or outdir
    df = pd.read_csv(input_csv)

    print(f"Loaded {len(df)} episodes from {input_csv}\n")

    # 1. Balance check
    check_balance(df)
    print()

    # 2. Asymmetry features
    df = add_asymmetry_features(df)
    asymmetry_out = f"{outdir}/dyad_analysis_with_asymmetry.csv"
    df.to_csv(asymmetry_out, index=False)
    print(f"Saved data with asymmetry columns to {asymmetry_out}\n")

    # 3. Summary stats per dyad (include asymmetry metrics too)
    all_metrics = METRICS + [pair[0] for pair in ASYMMETRY_PAIRS]
    summary = summarize_by_dyad(df, all_metrics)
    summary_out = f"{outdir}/dyad_summary_stats.csv"
    summary.to_csv(summary_out)
    print(f"Saved dyad summary stats to {summary_out}\n")
    print(summary.round(3))
    print()

    # 3b. Robust median + MAD summary (less sensitive to outliers, e.g. dominance_ratio)
    mad_summary = median_mad_by_dyad(df, all_metrics)
    mad_summary_out = f"{outdir}/dyad_median_mad_stats.csv"
    mad_summary.to_csv(mad_summary_out)
    print(f"Saved median/MAD summary to {mad_summary_out}\n")
    print(mad_summary.round(3))
    print()

    # 4. Boxplots for a sanity check before formal testing
    plot_out = f"{figdir}/dyad_boxplots.png"
    plot_boxplots(df, BOXPLOT_METRICS, plot_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/dyads/dyad_analysis.csv", help="Path to dyad_analysis.csv")
    ap.add_argument("--outdir", default="results/dyads", help="Directory to write output CSVs to")
    ap.add_argument("--figdir", default=None, help="Directory to write figures to (defaults to --outdir)")
    args = ap.parse_args()
    main(args.input, args.outdir, args.figdir)