"""
episode_dyads.py

Assigns every episode its host->guest gender dyad, using all episodes
(no balanced sampling; balanced_dataset.py keeps the 50-per-dyad sample for
robustness checks).

Host/guest gender per episode = gender of the HOST / GUEST speaker with the
most words. Episodes whose dyad is not one of the four male/female
combinations (e.g. unknown guest gender) are dropped and listed.

Output has the same columns as balanced_200_episodes.csv:
    episode_id, host_gender, guest_gender, dyad

Usage:
    python src/conversational_style/episode_dyads.py \
        --features results/features/conversational_style/speaker_features_v2.csv \
        --out results/features/episode_dyads.csv
"""

import argparse

import pandas as pd

VALID_DYADS = ["MALE->MALE", "MALE->FEMALE", "FEMALE->MALE", "FEMALE->FEMALE"]

# Manual guest-gender corrections for episodes where gender detection returned
# "unknown" for the guest (set after manual review).
GUEST_GENDER_OVERRIDES = {
    "ep_174_whisperx_diarized.gender.host_guest": "male",
    "ep_261_whisperx_diarized.gender.host_guest": "male",
}


def main():
    ap = argparse.ArgumentParser(description="Host->guest gender dyad for every episode")
    ap.add_argument("--features", default="results/features/conversational_style/speaker_features_v2.csv")
    ap.add_argument("--out", default="results/features/episode_dyads.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.features)

    # Keep only host and guest rows
    df = df[df["role"].isin(["HOST", "GUEST"])].copy()

    # Episode-level gender of the highest-word-count speaker in each role
    def top_gender(role):
        return (
            df[df["role"] == role]
            .sort_values(["episode_id", "word_count"], ascending=[True, False])
            .groupby("episode_id")["gender"]
            .first()
        )

    episodes = pd.DataFrame({
        "host_gender": top_gender("HOST"),
        "guest_gender": top_gender("GUEST"),
    })
    episodes.index.name = "episode_id"

    for ep, gender in GUEST_GENDER_OVERRIDES.items():
        if ep in episodes.index:
            print(f"Override: {ep} guest_gender {episodes.at[ep, 'guest_gender']} -> {gender}")
            episodes.at[ep, "guest_gender"] = gender
        else:
            print(f"WARNING: override for {ep} not applied (episode not found)")

    missing_role = episodes[episodes.isna().any(axis=1)]
    episodes = episodes.dropna()

    episodes["dyad"] = episodes["host_gender"].str.upper() + "->" + episodes["guest_gender"].str.upper()
    invalid = episodes[~episodes["dyad"].isin(VALID_DYADS)]
    episodes = episodes[episodes["dyad"].isin(VALID_DYADS)]

    episodes.to_csv(args.out)

    total = episodes.index.nunique() + len(invalid) + len(missing_role)
    print(f"Episodes in {args.features}: {total}")
    if len(missing_role):
        print(f"Dropped {len(missing_role)} without both a HOST and a GUEST row: "
              f"{', '.join(missing_role.index)}")
    if len(invalid):
        print(f"Dropped {len(invalid)} with an invalid dyad: "
              + ", ".join(f"{e} ({d})" for e, d in invalid["dyad"].items()))
    print(f"\nWrote {args.out} with {len(episodes)} episodes:")
    print(episodes["dyad"].value_counts().reindex(VALID_DYADS).to_string())


if __name__ == "__main__":
    main()
