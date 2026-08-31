import json
import re
import random
import pandas as pd
from compute_bias_features import COMPILED  # reuse your existing compiled lexicon

def sample_matches(json_paths, category, n_samples=50, context_chars=80):
    """Pull random in-context examples of a lexicon category for manual review."""
    samples = []
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            text = (seg.get("text") or "")
            for phrase, rx in COMPILED[category]:
                for m in rx.finditer(text.lower()):
                    start = max(0, m.start() - context_chars)
                    end = min(len(text), m.end() + context_chars)
                    samples.append({
                        "episode": path,
                        "phrase": phrase,
                        "context": text[start:end],
                    })
    random.shuffle(samples)
    return pd.DataFrame(samples[:n_samples])

# Example usage:
import glob
paths = glob.glob("/home/sr5868/listening-between-the-lines/data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json")
sample_df = sample_matches(paths, "directive", n_samples=50)
sample_df["is_true_positive"] = ""  # fill in manually: yes/no
sample_df.to_csv("/home/sr5868/listening-between-the-lines/results/dialogue_acts/directive_audit_sample.csv", index=False)