import os
import csv

INPUT_DIR = "data/cleaned_transcripts"
OUTPUT = "outputs/tables/text_stats.csv"

os.makedirs("outputs/tables", exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "word_count", "sentence_count"])

    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as infile:
                text = infile.read()

            word_count = len(text.split())
            sentence_count = text.count(".") + text.count("?") + text.count("!")

            writer.writerow([fname, word_count, sentence_count])

print(f"Saved → {OUTPUT}")