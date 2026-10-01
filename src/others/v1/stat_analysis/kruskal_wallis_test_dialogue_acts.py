from scipy.stats import kruskal
import pandas as pd

df = pd.read_csv("/home/sr5868/listening-between-the-lines/results/dyads/dyad_dialogue_acts.csv")

# Only the higher-frequency dialogue acts (see dialogue_act_classification.py output):
# hold, reply_no, exclaim, ask_yes_no, and backchannel are all under 1% of turns and
# too rare per episode to test reliably.
ACTS = ["ask", "answer", "say", "acknowledge", "intent", "reply_yes"]

for act in ACTS:
    df[f"{act}_asymmetry"] = df[f"host_{act}"] - df[f"guest_{act}"]

# host_ask mirrors host_speaking_share in the lexicon test: an overall host-behavior
# level metric, alongside one asymmetry (host - guest) metric per dialogue act.
metrics = ["host_ask"] + [f"{act}_asymmetry" for act in ACTS]

results = []
for m in metrics:
    groups = [g[m].dropna().values for _, g in df.groupby("dyad")]
    stat, p = kruskal(*groups)
    results.append({"metric": m, "H_statistic": stat, "p_value": p})
    print(f"{m:25s} H={stat:.3f}  p={p:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/sr5868/listening-between-the-lines/results/stat_analysis/kruskal_wallis_dialogue_acts_results.csv", index=False)
