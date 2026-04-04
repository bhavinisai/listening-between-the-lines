import os
import json
import re
import glob
import pandas as pd
from collections import defaultdict

# -------------------------
# 1) Lexicons (start small, expand later)
# -------------------------
LEX = {
    "hedge": [
        "maybe", "perhaps", "possibly", "probably",
        "i think", "i feel", "i guess",
        "kind of", "sort of", "a bit", "a little",
        "it seems", "it looks like", "in a way"
    ],
    "booster": [
        "definitely", "clearly", "obviously", "absolutely", "certainly",
        "for sure", "no doubt", "everyone knows"
    ],
    "polite": [
        "please", "sorry", "apologies",
        "thank you", "thanks", "appreciate",
        "excuse me", "pardon me", "if you don't mind"
    ],
    "directive": [
        "hold on", "hang on", "wait", "listen",
        "let me explain", "let me finish", "let me tell you",
        "here's the thing", "actually", "look,"
    ],
}

WORD_RE = re.compile(r"[A-Za-z']+")

def compile_lex(phrases):
    # match longer phrases first
    phrases = sorted(set([p.lower().strip() for p in phrases]), key=len, reverse=True)
    return [(p, re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)) for p in phrases]

COMPILED = {k: compile_lex(v) for k, v in LEX.items()}

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text.lower()))

def count_phrases(text: str, compiled_patterns):
    t = text.lower()
    total = 0
    per_phrase = {}
    for phrase, rx in compiled_patterns:
        hits = len(rx.findall(t))
        if hits:
            per_phrase[phrase] = hits
            total += hits
    return total, per_phrase

def featurize_text(text: str):
    wc = word_count(text)
    out = {"word_count": wc}

    for cat, patterns in COMPILED.items():
        c, per_phrase = count_phrases(text, patterns)
        out[f"{cat}_count"] = c
        out[f"{cat}_per_1k"] = (c / wc * 1000.0) if wc else 0.0
        # Optional: keep per-phrase details
        # for phrase, hits in per_phrase.items():
        #     out[f"{cat}__{phrase}"] = hits
    return out

# -------------------------
# 2) Load one WhisperX JSON and aggregate by speaker
# -------------------------
def process_episode(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episode_id = os.path.splitext(os.path.basename(json_path))[0]

    # Group text by (speaker_id, role, gender)
    buckets = defaultdict(list)
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "UNKNOWN")
        role = (seg.get("speaker_role") or "UNKNOWN").upper()
        gender = (seg.get("gender") or "unknown").lower()
        buckets[(speaker, role, gender)].append(text)

    rows = []
    for (speaker, role, gender), texts in buckets.items():
        joined = " ".join(texts)
        feats = featurize_text(joined)
        rows.append({
            "episode_id": episode_id,
            "speaker_id": speaker,
            "role": role,
            "gender": gender,
            **feats
        })

    return rows

# -------------------------
# 3) Run over a folder of episodes
# -------------------------
def main(input_glob: str, out_csv: str = "speaker_features.csv"):
    all_rows = []
    for path in glob.glob(input_glob):
        all_rows.extend(process_episode(path))

    df = pd.DataFrame(all_rows)

    # Basic sanity filters (optional)
    df = df[df["word_count"] >= 50].copy()  # drop tiny speakers
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(df)} speaker-episode rows.")

if __name__ == "__main__":
    # Example: main("data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="Glob for JSON files")
    ap.add_argument("--out_csv", default="speaker_features.csv")
    args = ap.parse_args()
    main(args.input_glob, args.out_csv)
