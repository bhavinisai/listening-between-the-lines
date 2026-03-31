import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    file_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "outputs"
        / "whisperx"
        / "ep_001_whisperx_diarized.gender.host_guest.txt"
    )

    output_dir = Path(__file__).resolve().parent

    print(f"Reading file: {file_path}")

    pattern = re.compile(
        r"\[(.*?) - (.*?)\]\s+(SPEAKER_\d+)\s+\((.*?),\s+(.*?)\):"
    )

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                start, end, speaker, gender, role = match.groups()

                rows.append(
                    {
                        "episode_id": "ep_001",
                        "speaker_id": speaker,
                        "gender": gender.strip().lower(),
                        "role": role.strip().lower(),
                        "start_sec": time_to_seconds(start.strip()),
                        "end_sec": time_to_seconds(end.strip()),
                    }
                )

    if not rows:
        print("No rows parsed. Check the file format or regex.")
        return

    df = pd.DataFrame(rows)

    df["duration_sec"] = df["end_sec"] - df["start_sec"]
    df = df[df["duration_sec"] > 0].copy()
    df["duration_min"] = df["duration_sec"] / 60

    summary = (
        df.groupby(["episode_id", "speaker_id", "gender", "role"], as_index=False)["duration_min"]
        .sum()
        .sort_values("duration_min", ascending=False)
    )

    episode_totals = (
        summary.groupby("episode_id", as_index=False)["duration_min"]
        .sum()
        .rename(columns={"duration_min": "episode_total_min"})
    )

    summary = summary.merge(episode_totals, on="episode_id", how="left")
    summary["speaking_share"] = summary["duration_min"] / summary["episode_total_min"]
    summary["speaking_percentage"] = summary["speaking_share"] * 100

    print("\n=== Speaking Time Summary (minutes) ===")
    print(summary.to_string(index=False))

    csv_path = output_dir / "ep_001_speaking_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    plt.figure()
    plt.bar(summary["speaker_id"], summary["speaking_percentage"])
    plt.title("Speaking Share per Speaker (Episode 001)")
    plt.xlabel("Speaker")
    plt.ylabel("Speaking Percentage (%)")

    for i, v in enumerate(summary["speaking_percentage"]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")

    speaker_plot_path = output_dir / "speaker_plot.png"
    plt.savefig(speaker_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved speaker plot to: {speaker_plot_path}")

    role_summary = (
        summary.groupby("role", as_index=False)["duration_min"]
        .sum()
    )
    role_summary["speaking_percentage"] = (
        role_summary["duration_min"] / role_summary["duration_min"].sum() * 100
    )

    plt.figure()
    plt.bar(role_summary["role"], role_summary["speaking_percentage"])
    plt.title("Speaking Share by Role")
    plt.xlabel("Role")
    plt.ylabel("Speaking Percentage (%)")

    for i, v in enumerate(role_summary["speaking_percentage"]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")

    role_plot_path = output_dir / "role_plot.png"
    plt.savefig(role_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved role plot to: {role_plot_path}")


if __name__ == "__main__":
    main()
