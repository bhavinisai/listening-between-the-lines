import os
import re
import time
import random
from typing import List
from deep_translator import GoogleTranslator

RAW_DIR = "data/raw_transcripts"
CLEAN_DIR = "data/cleaned_transcripts"
LOG_PATH = "outputs/failed_chunks.log"

# Smaller chunks = fewer connection/rate-limit failures
MAX_CHARS_PER_CHUNK = 1200

# Gentle pacing between requests (helps avoid throttling)
SLEEP_BETWEEN_CHUNKS_SEC = 5.0

# Retry settings
MAX_RETRIES = 8

# Optional: detect Devanagari (Hindi) to decide whether to translate
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def split_into_chunks(text: str, max_len: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Splits text into chunks <= max_len, trying to respect newline boundaries.
    If a single line is longer than max_len, it will be hard-split.
    """
    lines = text.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0

    for line in lines:
        line = line.rstrip()

        # Hard-split very long lines
        if len(line) > max_len:
            flush()
            start = 0
            while start < len(line):
                chunks.append(line[start:start + max_len])
                start += max_len
            continue

        # Add to buffer if it fits; else flush first
        add_len = len(line) + 1  # + newline approx
        if buf_len + add_len <= max_len:
            buf.append(line)
            buf_len += add_len
        else:
            flush()
            buf.append(line)
            buf_len = add_len

    flush()
    return chunks


def translate_with_retry(chunk: str, translator: GoogleTranslator, max_retries: int = MAX_RETRIES) -> str:
    """
    Attempts to translate a chunk multiple times.
    If it ultimately fails, returns the original chunk (so output isn't empty).
    """
    chunk = chunk.strip()
    if not chunk:
        return ""

    for attempt in range(1, max_retries + 1):
        try:
            return translator.translate(chunk)
        except Exception as e:
            # exponential backoff + jitter
            wait = (2 ** (attempt - 1)) + random.uniform(0.2, 0.9)
            print(f"      Retry {attempt}/{max_retries} failed: {e} → sleeping {wait:.1f}s")
            time.sleep(wait)

    print("      Giving up on this chunk — keeping original text to avoid empty output.")
    return chunk


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def log_fallback(filename: str, chunk_idx: int, total_chunks: int) -> None:
    ensure_parent_dir(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"{filename} chunk {chunk_idx}/{total_chunks} kept original (translation failed)\n")


def is_likely_english(text: str) -> bool:
    """
    Simple heuristic:
    - If there's no Devanagari, we assume it's already English/Hinglish enough to skip translation.
    (You can remove this if you want to translate everything.)
    """
    return DEVANAGARI_RE.search(text) is None


def translate_file(raw_path: str, clean_path: str) -> None:
    print(f"  Reading: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("  ⚠ Empty file, skipping.")
        return

    # Optional skip if already English-ish
    if is_likely_english(text):
        ensure_parent_dir(clean_path)
        with open(clean_path, "w", encoding="utf-8") as out:
            out.write(text)
        print("  Detected mostly non-Hindi text → copied without translation.")
        print(f"   Saved file → {clean_path}")
        return

    chunks = split_into_chunks(text, MAX_CHARS_PER_CHUNK)
    print(f"  Text length: {len(text)} chars → {len(chunks)} chunk(s)")

    translator = GoogleTranslator(source="auto", target="en")
    translated_chunks: List[str] = []

    ensure_parent_dir(clean_path)

    for i, chunk in enumerate(chunks, start=1):
        print(f"    Translating chunk {i}/{len(chunks)}...")

        translated = translate_with_retry(chunk, translator, MAX_RETRIES)

        # If translation failed and we fell back to original, log it
        if translated.strip() == chunk.strip() and chunk.strip():
            log_fallback(os.path.basename(raw_path), i, len(chunks))

        translated_chunks.append(translated)

        # pacing
        time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)

    full_translated = "\n".join(translated_chunks)

    with open(clean_path, "w", encoding="utf-8") as out:
        out.write(full_translated)

    print(f"   Saved translated file → {clean_path}")


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"Folder not found: {RAW_DIR}")
        return

    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith("_raw.txt"))
    if not files:
        print(f"No *_raw.txt files found in {RAW_DIR}")
        return

    print(f"Found {len(files)} raw TXT file(s) in {RAW_DIR}")

    for fname in files:
        ep_id = os.path.splitext(fname)[0]  # ep001_raw
        out_name = ep_id.replace("_raw", "_cleaned") + ".txt"

        raw_path = os.path.join(RAW_DIR, fname)
        clean_path = os.path.join(CLEAN_DIR, out_name)

        print(f"\nTranslating {fname} → {out_name}")
        translate_file(raw_path, clean_path)

        time.sleep(10)

    print("\nBatch translation complete.")


if __name__ == "__main__":
    main()