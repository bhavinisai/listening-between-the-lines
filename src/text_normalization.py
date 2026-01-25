"""
Text normalization (safe cleanup) for podcast transcripts.

What it does:
- Unicode normalize (NFKC)
- Standardize quotes/apostrophes/dashes/ellipsis
- Remove control characters (keeps \n and \t)
- Normalize newlines
- Collapse repeated spaces
- Trim whitespace per-line and overall
- Optionally create a lowercased copy

Usage:
  python normalize_transcripts.py --in_dir ./raw_txt --out_dir ./clean_txt

Outputs:
  For each input file:
    - <name>.clean.txt        (normalized text, original casing)
    - <name>.clean.lower.txt  (normalized + lowercased)
"""

from __future__ import annotations
import argparse
import re
import unicodedata
from pathlib import Path

# --- Transliteration / normalization maps ---
TRANSLATION_TABLE = str.maketrans({
    # Curly quotes/apostrophes -> straight
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u00AB": '"', "\u00BB": '"',

    # Dashes -> hyphen
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2212": "-",  # minus sign

    # Ellipsis -> "..."
    "\u2026": "...",

    # Non-breaking space and similar -> normal space
    "\u00A0": " ", "\u2007": " ", "\u202F": " ", "\u2009": " ", "\u200A": " ",
})

# Remove zero-width / BOM characters
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

# Collapse spaces/tabs (not newlines)
MULTISPACE_RE = re.compile(r"[ \t]{2,}")

# Collapse too many blank lines (3+ -> 2)
MULTIBLANK_RE = re.compile(r"\n{3,}")

def remove_control_chars(s: str) -> str:
    """
    Remove most control chars except newline and tab.
    Keeps normal printable characters.
    """
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Cc":  # control
            if ch in ("\n", "\t"):
                out.append(ch)
            # else: drop
        else:
            out.append(ch)
    return "".join(out)

def normalize_text(text: str) -> str:
    # 1) Normalize unicode compatibility forms
    text = unicodedata.normalize("NFKC", text)

    # 2) Standardize common punctuation variants
    text = text.translate(TRANSLATION_TABLE)

    # 3) Drop zero-width/BOM artifacts
    text = ZERO_WIDTH_RE.sub("", text)

    # 4) Normalize newlines to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5) Remove control characters (keeps \n and \t)
    text = remove_control_chars(text)

    # 6) Trim whitespace around each line (keeps paragraph structure)
    #    Also prevents lines that are "   " from staying non-empty.
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)

    # 7) Collapse repeated spaces/tabs within lines
    text = MULTISPACE_RE.sub(" ", text)

    # 8) Collapse excessive blank lines
    text = MULTIBLANK_RE.sub("\n\n", text)

    # 9) Strip leading/trailing whitespace
    return text.strip()

def process_file(in_path: Path, out_dir: Path) -> None:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    clean = normalize_text(raw)

    out_clean = out_dir / f"{in_path.stem}.clean.txt"
    out_lower = out_dir / f"{in_path.stem}.clean.lower.txt"

    out_clean.write_text(clean, encoding="utf-8")
    out_lower.write_text(clean.lower(), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", help="Folder containing .txt transcripts")
    ap.add_argument("out_dir", help="Folder to write normalized files")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern (default: *.txt)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {in_dir}")

    for fp in files:
        process_file(fp, out_dir)

    print(f"Done. Normalized {len(files)} file(s) -> {out_dir}")

if __name__ == "__main__":
    main()
