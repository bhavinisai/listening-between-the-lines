import pandas as pd

# Same host/guest gender + dyad labeling as balanced_dataset.py, but keeps
# every episode instead of sampling 50 per dyad. Episodes whose guest gender
# could not be detected keep their label (e.g. MALE->UNKNOWN) so the file
# covers all transcripts; downstream scripts filter to the dyads they need.

df = pd.read_csv("results/features/speaker_features.csv")
df = df[df["role"].isin(["HOST", "GUEST"])].copy()

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

episodes["dyad"] = episodes["host_gender"].str.upper() + "->" + episodes["guest_gender"].str.upper()

episodes.to_csv("results/features/all_episodes.csv")

print("All-episodes dataset created with", len(episodes), "episodes.")
print(episodes["dyad"].value_counts())
