import pandas as pd

# ---------- 1. Load ----------
rates = pd.read_csv("results/dialogue_act_rates_by_episode_role.csv")
bal = pd.read_csv("results/features/balanced_200_episodes.csv")

# ---------- 2. Keep only the 200 balanced episodes ----------
rates = rates[rates["episode_id"].isin(bal["episode_id"])].copy()

# Sanity check: every episode should have exactly one HOST and one GUEST row
role_counts = rates.groupby("episode_id")["role"].apply(lambda x: sorted(x.tolist()))
bad = role_counts[role_counts.apply(lambda r: r != ["GUEST", "HOST"])]
if len(bad) > 0:
    print(f"WARNING: {len(bad)} episodes without exactly one HOST + one GUEST row:")
    print(bad)

# ---------- 3. Pivot long -> wide (one row per episode) ----------
act_cols = [c for c in rates.columns if c not in ("episode_id", "role")]

host = rates[rates["role"] == "HOST"][["episode_id"] + act_cols].copy()
guest = rates[rates["role"] == "GUEST"][["episode_id"] + act_cols].copy()

host = host.rename(columns={c: f"host_{c}" for c in act_cols})
guest = guest.rename(columns={c: f"guest_{c}" for c in act_cols})

wide = host.merge(guest, on="episode_id", how="inner")

# ---------- 4. Merge in dyad / gender-pair info ----------
df = wide.merge(bal, on="episode_id", how="inner")

# ---------- 5. Reorder columns: id + dyad info, then host_/guest_ act pairs ----------
cols = ["episode_id", "dyad", "host_gender", "guest_gender"]
for c in act_cols:
    cols += [f"host_{c}", f"guest_{c}"]
df = df[cols]

df.to_csv("results/dyads/dyad_dialogue_acts.csv", index=False)
print(f"Saved dyad_dialogue_acts.csv with {len(df)} episodes and {len(df.columns)} columns.")
print(df.head())
