"""
Remove non-content sections from podcast transcripts (.txt).

Handles:
1) Platform artifacts like: [music], [applause], (laughter), [inaudible], etc.
2) Sponsor/ads blocks using start-cues and end-cues (heuristic).
3) Intro/outro boilerplate using phrase matching + optional "trim head/tail" windows.

USAGE (flags version):
  python remove_boilerplate.py --in_dir data/normalized_transcripts --out_dir data/content_transcripts

Optional:
  --keep_removed      writes a .removed.txt file with what got removed
  --dry_run           doesn't write output, prints a summary
  --head_lines 40     only search for intro in first N lines
  --tail_lines 60     only search for outro in last N lines
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# -----------------------------
# 1) PLATFORM ARTIFACTS
# -----------------------------
# Remove standalone bracketed artifacts: [music], [applause], (laughter), etc.
ARTIFACT_LINE_RE = re.compile(
    r"""^\s*(
        \[[^\]]{1,40}\]      # [music], [applause], [inaudible], etc.
        |
        \([^\)]{1,40}\)      # (laughter), (crosstalk), etc.
        |
        <[^>]{1,40}>         # <music> (rare)
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Remove inline artifacts like: "that's funny (laughter) anyway"
INLINE_ARTIFACT_RE = re.compile(
    r"""(\s*[\(\[]\s*
        (?:laughter|laughs?|applause|music|intro|outro|crosstalk|inaudible|silence|sighs?)
        \s*[\)\]]\s*)""",
    re.IGNORECASE | re.VERBOSE,
)


# -----------------------------
# 2) SPONSOR / ADS BLOCKS
# -----------------------------
# Start cues: when we see these, we assume we are entering an ad/sponsor segment.
SPONSOR_START_CUES = [
    r"\bthis episode is sponsored by\b",
    r"\bthis podcast is sponsored by\b",
    r"\bsponsored by\b",
    r"\bthanks to\b.+\bfor sponsoring\b",
    r"\bad break\b",
    r"\bwe'll be right back\b",
    r"\bwe'll be back after\b",
    r"\bquick break\b",
    r"\bmessage from our sponsor\b",
    r"\bpartner(?:ed)? with\b",
    r"\bpromotional consideration\b",
    r"\bpaid for by\b",
]

# End cues: when we see these after sponsor start, we assume ad is ending.
SPONSOR_END_CUES = [
    r"\bback to the show\b",
    r"\bback to our conversation\b",
    r"\bnow back to\b",
    r"\blet's get back\b",
    r"\bwelcome back\b",
    r"\bthanks for sticking around\b",
    r"\band we're back\b",
]

SPONSOR_START_RE = re.compile("|".join(SPONSOR_START_CUES), re.IGNORECASE)
SPONSOR_END_RE = re.compile("|".join(SPONSOR_END_CUES), re.IGNORECASE)

# Safety caps (avoid deleting half the episode by accident)
MAX_SPONSOR_BLOCK_LINES = 140  # if no end cue is found, stop removing after this many lines


# -----------------------------
# 3) INTRO / OUTRO BOILERPLATE
# -----------------------------
INTRO_PHRASES = [
    r"\bwelcome to\b",
    r"\bwelcome back to\b",
    r"\bthanks for (?:tuning|joining) in\b",
    r"\bmy name is\b.*\b(?:host|your host)\b",
    r"\btoday(?:'s)? episode\b",
    r"\bsubscribe\b",
    r"\brate (?:and )?review\b",
    r"\bfollow (?:us|me)\b",
]

OUTRO_PHRASES = [
    r"\bthanks for listening\b",
    r"\bthanks for tuning in\b",
    r"\bsee you (?:next|soon)\b",
    r"\bthat's (?:it|all) for today\b",
    r"\bsubscribe\b",
    r"\brate (?:and )?review\b",
    r"\bfollow (?:us|me)\b",
    r"\bcheck out\b.*\b(link|show notes)\b",
    r"\bvisit\b.*\b(dot com|\.com)\b",
]

INTRO_RE = re.compile("|".join(INTRO_PHRASES), re.IGNORECASE)
OUTRO_RE = re.compile("|".join(OUTRO_PHRASES), re.IGNORECASE)


@dataclass
class RemovalLog:
    artifact_lines_removed: int = 0
    inline_artifacts_removed: int = 0
    sponsor_blocks_removed: int = 0
    sponsor_lines_removed: int = 0
    intro_lines_removed: int = 0
    outro_lines_removed: int = 0


def remove_artifacts(lines: List[str], log: RemovalLog) -> List[str]:
    cleaned: List[str] = []
    for ln in lines:
        # drop whole line artifacts
        if ARTIFACT_LINE_RE.match(ln):
            log.artifact_lines_removed += 1
            continue

        # remove inline artifacts
        new_ln, n = INLINE_ARTIFACT_RE.subn(" ", ln)
        if n:
            log.inline_artifacts_removed += n
            new_ln = re.sub(r"[ \t]{2,}", " ", new_ln).strip()

        cleaned.append(new_ln)
    return cleaned


def remove_sponsor_blocks(lines: List[str], log: RemovalLog) -> Tuple[List[str], List[str]]:
    """
    Heuristic line-based sponsor removal:
    - If a line matches sponsor-start cue, start removing lines
    - Stop removing when we hit sponsor-end cue OR we exceed MAX_SPONSOR_BLOCK_LINES
    Returns: (kept_lines, removed_lines)
    """
    kept: List[str] = []
    removed: List[str] = []
    in_sponsor = False
    sponsor_len = 0

    for ln in lines:
        if not in_sponsor and SPONSOR_START_RE.search(ln):
            in_sponsor = True
            log.sponsor_blocks_removed += 1
            sponsor_len = 0
            removed.append(ln)
            log.sponsor_lines_removed += 1
            continue

        if in_sponsor:
            sponsor_len += 1
            removed.append(ln)
            log.sponsor_lines_removed += 1

            # end condition
            if SPONSOR_END_RE.search(ln) or sponsor_len >= MAX_SPONSOR_BLOCK_LINES:
                in_sponsor = False
            continue

        kept.append(ln)

    return kept, removed


def remove_intro_outro(lines: List[str], log: RemovalLog, head_lines: int, tail_lines: int) -> Tuple[List[str], List[str]]:
    """
    Removes intro phrases in first head_lines and outro phrases in last tail_lines.
    Conservative rule:
    - Remove leading consecutive lines that match INTRO_RE (or are very short) until we hit content.
    - Remove trailing consecutive lines that match OUTRO_RE (or are very short) until we hit content.
    Returns: (kept_lines, removed_lines)
    """
    removed: List[str] = []
    n = len(lines)

    # --- INTRO removal (front) ---
    head = lines[: min(head_lines, n)]
    rest = lines[min(head_lines, n):]

    # Remove consecutive intro-like lines at the start
    i = 0
    while i < len(head):
        ln = head[i].strip()
        # treat very short lines as skippable if we are still in intro removal mode
        if ln == "" or len(ln) <= 2 or INTRO_RE.search(ln):
            removed.append(head[i])
            log.intro_lines_removed += 1
            i += 1
        else:
            break
    head_kept = head[i:]

    # --- OUTRO removal (end) ---
    combined = head_kept + rest
    n2 = len(combined)
    tail_start = max(0, n2 - tail_lines)
    body = combined[:tail_start]
    tail = combined[tail_start:]

    j = len(tail) - 1
    while j >= 0:
        ln = tail[j].strip()
        if ln == "" or len(ln) <= 2 or OUTRO_RE.search(ln):
            removed.append(tail[j])
            log.outro_lines_removed += 1
            j -= 1
        else:
            break
    tail_kept = tail[: j + 1]

    kept = body + tail_kept
    return kept, removed


def collapse_blank_lines(lines: List[str], max_consecutive: int = 2) -> List[str]:
    out: List[str] = []
    blanks = 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks <= max_consecutive:
                out.append("")
        else:
            blanks = 0
            out.append(ln.rstrip())
    # trim leading/trailing blanks
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def clean_transcript(text: str, head_lines: int, tail_lines: int) -> Tuple[str, str, RemovalLog]:
    log = RemovalLog()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Step 1: artifacts
    lines = remove_artifacts(lines, log)

    # Step 2: sponsor blocks
    lines, removed_sponsor = remove_sponsor_blocks(lines, log)

    # Step 3: intro/outro
    lines, removed_io = remove_intro_outro(lines, log, head_lines=head_lines, tail_lines=tail_lines)

    # Step 4: final cleanup
    lines = collapse_blank_lines(lines, max_consecutive=2)

    cleaned = "\n".join(lines).strip()
    removed_all = "\n".join([*removed_sponsor, *removed_io]).strip()
    return cleaned, removed_all, log


def process_file(in_path: Path, out_dir: Path, keep_removed: bool, dry_run: bool, head_lines: int, tail_lines: int) -> None:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    cleaned, removed, log = clean_transcript(raw, head_lines=head_lines, tail_lines=tail_lines)

    if dry_run:
        print(f"\n[{in_path.name}]")
        print(f"  artifacts lines removed: {log.artifact_lines_removed}")
        print(f"  inline artifacts removed: {log.inline_artifacts_removed}")
        print(f"  sponsor blocks removed: {log.sponsor_blocks_removed}")
        print(f"  sponsor lines removed: {log.sponsor_lines_removed}")
        print(f"  intro lines removed: {log.intro_lines_removed}")
        print(f"  outro lines removed: {log.outro_lines_removed}")
        print(f"  output chars: {len(cleaned)}")
        return

    out_clean = out_dir / f"{in_path.stem}.content.txt"
    out_clean.write_text(cleaned, encoding="utf-8")

    if keep_removed:
        out_removed = out_dir / f"{in_path.stem}.removed.txt"
        out_removed.write_text(removed, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing .txt transcripts")
    ap.add_argument("--out_dir", required=True, help="Folder to write cleaned transcripts")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern (default: *.txt)")
    ap.add_argument("--keep_removed", action="store_true", help="Also write a .removed.txt file per transcript")
    ap.add_argument("--dry_run", action="store_true", help="Print removal stats, don't write files")
    ap.add_argument("--head_lines", type=int, default=40, help="Search intro only within first N lines (default 40)")
    ap.add_argument("--tail_lines", type=int, default=60, help="Search outro only within last N lines (default 60)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {in_dir}")

    for fp in files:
        process_file(fp, out_dir, args.keep_removed, args.dry_run, args.head_lines, args.tail_lines)

    if not args.dry_run:
        print(f"Done. Cleaned {len(files)} file(s) -> {out_dir}")


if __name__ == "__main__":
    main()
