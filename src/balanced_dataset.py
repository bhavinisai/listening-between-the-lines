import pandas as pd
import os
import shutil
import random

# Load your main features file
df = pd.read_csv("results/speaker_features.csv")

# Keep only host and guest rows
df = df[df["role"].isin(["HOST", "GUEST"])].copy()

# Extract episode-level host and guest gender
host_gender = (
    df[df["role"] == "HOST"]
    .sort_values(["episode_id", "word_count"], ascending=[True, False])
    .groupby("episode_id")["gender"]
    .first()
)

guest_gender = (
    df[df["role"] == "GUEST"]
    .sort_values(["episode_id", "word_count"], ascending=[True, False])
    .groupby("episode_id")["gender"]
    .first()
)

episodes = pd.DataFrame({
    "host_gender": host_gender,
    "guest_gender": guest_gender
}).dropna()

# Create dyad label
episodes["dyad"] = episodes["host_gender"].str.upper() + "->" + episodes["guest_gender"].str.upper()

# Filter only the 4 valid dyads
valid_dyads = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]
episodes = episodes[episodes["dyad"].isin(valid_dyads)]

# Balanced sampling: 50 per dyad
balanced_ids = []

for dyad in valid_dyads:
    subset = episodes[episodes["dyad"] == dyad]
    if len(subset) >= 50:
        sampled = subset.sample(50, random_state=42)
        balanced_ids.extend(sampled.index.tolist())
    else:
        print(f"Warning: {dyad} has only {len(subset)} episodes.")

balanced_df = episodes.loc[balanced_ids]

# Save selected episode IDs
balanced_df.to_csv("results/balanced_200_episodes.csv")

print("Balanced dataset created with", len(balanced_df), "episodes.")