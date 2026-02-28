import os
import json

RAW_DIR = "data/raw_transcripts"

def convert_one(json_path: str, txt_path: str):
    """Convert a single transcript JSON file to plain text."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    
    lines = []
    for seg in data:
        text = seg.get("text", "").strip()
        if text:
            lines.append(text)

    
    joined_text = "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as out:
        out.write(joined_text)

    print(f"   Saved TXT → {txt_path}")


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"Folder not found: {RAW_DIR}")
        return

    files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".json")
    )

    if not files:
        print(f"No JSON files found in {RAW_DIR}")
        return

    print(f"Found {len(files)} JSON file(s) in {RAW_DIR}")

    for fname in files:
        json_path = os.path.join(RAW_DIR, fname)
        base = os.path.splitext(fname)[0]   
        txt_path = os.path.join(RAW_DIR, f"{base}.txt")

        print(f"\nConverting {json_path}")
        convert_one(json_path, txt_path)

    print("\n All JSON transcripts converted to TXT.")

if __name__ == "__main__":
    main()