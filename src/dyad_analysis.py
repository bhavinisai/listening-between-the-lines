import pandas as pd

# ---------- 1. Load ----------
sf = pd.read_csv("results/features/speaker_features.csv")
bal = pd.read_csv("results/features/balanced_200_episodes.csv")

# ---------- 2. Keep only the 200 balanced episodes ----------
sf = sf[sf["episode_id"].isin(bal["episode_id"])].copy()

# Sanity check: every episode should have exactly one HOST and one GUEST row
role_counts = sf.groupby("episode_id")["role"].apply(lambda x: sorted(x.tolist()))
bad = role_counts[role_counts.apply(lambda r: r != ["GUEST", "HOST"])]
if len(bad) > 0:
    print(f"WARNING: {len(bad)} episodes without exactly one HOST + one GUEST row:")
    print(bad)

# ---------- 3. Pivot long -> wide (one row per episode) ----------
feature_cols = [
    "word_count",
    "hedge_count", "hedge_per_1k",
    "booster_count", "booster_per_1k",
    "polite_count", "polite_per_1k",
    "directive_count", "directive_per_1k",
]

host = sf[sf["role"] == "HOST"][["episode_id", "gender"] + feature_cols].copy()
guest = sf[sf["role"] == "GUEST"][["episode_id", "gender"] + feature_cols].copy()

host = host.rename(columns={c: f"host_{c}" for c in ["gender"] + feature_cols})
guest = guest.rename(columns={c: f"guest_{c}" for c in ["gender"] + feature_cols})

wide = host.merge(guest, on="episode_id", how="inner")

# ---------- 4. Merge in dyad / gender-pair info ----------
df = wide.merge(bal, on="episode_id", how="inner", suffixes=("_sf", ""))

# Cross-check: does speaker_features gender match balanced_200_episodes.csv gender?
gender_mismatch = df[
    (df["host_gender_sf"] != df["host_gender"]) |
    (df["guest_gender_sf"] != df["guest_gender"])
]
if len(gender_mismatch) > 0:
    print(f"WARNING: {len(gender_mismatch)} episodes have mismatched gender labels "
          f"between speaker_features.csv and balanced_200_episodes.csv")
    print(gender_mismatch[["episode_id", "host_gender_sf", "host_gender",
                            "guest_gender_sf", "guest_gender"]])

df = df.drop(columns=["host_gender_sf", "guest_gender_sf"])

# ---------- 5. Derived conversational-dynamics features ----------
df["total_words"] = df["host_word_count"] + df["guest_word_count"]
df["host_speaking_share"] = df["host_word_count"] / df["total_words"]
df["guest_speaking_share"] = df["guest_word_count"] / df["total_words"]
df["dominance_ratio"] = df["host_word_count"] / df["guest_word_count"]  # >1 = host talks more

# ---------- 6. Reorder / clean columns ----------
cols = [
    "episode_id", "dyad", "host_gender", "guest_gender",
    "host_word_count", "guest_word_count", "total_words",
    "host_speaking_share", "guest_speaking_share", "dominance_ratio",
    "host_hedge_count", "guest_hedge_count", "host_hedge_per_1k", "guest_hedge_per_1k",
    "host_booster_count", "guest_booster_count", "host_booster_per_1k", "guest_booster_per_1k",
    "host_polite_count", "guest_polite_count", "host_polite_per_1k", "guest_polite_per_1k",
    "host_directive_count", "guest_directive_count", "host_directive_per_1k", "guest_directive_per_1k",
]
df = df[cols]

df.to_csv("results/dyads/dyad_analysis.csv", index=False)
print(f"Saved dyad_analysis.csv with {len(df)} episodes and {len(df.columns)} columns.")
print(df.head())