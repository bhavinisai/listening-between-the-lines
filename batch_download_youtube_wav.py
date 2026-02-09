#!/usr/bin/env python3
"""
Batch download YouTube videos as WAV audio files.

Requirements:
  pip install yt-dlp

Usage:
  1. Create a text file with one YouTube URL per line (urls.txt)
  2. Run: python batch_download_youtube_wav.py --urls urls.txt --outdir data/raw_audio
"""

import argparse
import subprocess
from pathlib import Path
import sys


def download_youtube_as_wav(url: str, output_dir: Path, index: int = None) -> bool:
    """
    Download a YouTube video and convert to WAV using yt-dlp.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Sanitize filename - yt-dlp will use video title by default
        if index is not None:
            output_template = str(output_dir / f"episode_{index:03d}_%(title)s.%(ext)s")
        else:
            output_template = str(output_dir / "%(title)s.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "--extract-audio",           # Extract audio only
            "--audio-format", "wav",     # Convert to WAV
            "--audio-quality", "0",      # Best quality
            "-o", output_template,       # Output template
            url
        ]
        
        print(f"[{index if index else '?'}] Downloading: {url}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[{index if index else '?'}] ✓ Success")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[{index if index else '?'}] ✗ Failed: {url}")
        print(f"    Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"[{index if index else '?'}] ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch download YouTube videos as WAV audio files"
    )
    parser.add_argument(
        "--urls",
        required=True,
        help="Text file with one YouTube URL per line"
    )
    parser.add_argument(
        "--outdir",
        default="data/raw_audio",
        help="Output directory for WAV files (default: data/raw_audio)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start from this line number (default: 1)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End at this line number (default: process all)"
    )
    
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: yt-dlp not found. Install it with: pip install yt-dlp")
        sys.exit(1)
    
    # Read URLs from file
    urls_file = Path(args.urls)
    if not urls_file.exists():
        print(f"Error: File not found: {urls_file}")
        sys.exit(1)
    
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not urls:
        print("Error: No URLs found in file")
        sys.exit(1)
    
    # Apply start/end filters
    start_idx = args.start - 1  # Convert to 0-indexed
    end_idx = args.end if args.end else len(urls)
    urls = urls[start_idx:end_idx]
    
    print(f"Found {len(urls)} URLs to download")
    
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print("-" * 60)
    
    # Download each URL
    success_count = 0
    fail_count = 0
    
    for i, url in enumerate(urls, start=args.start):
        if download_youtube_as_wav(url, output_dir, index=i):
            success_count += 1
        else:
            fail_count += 1
        print()  # Blank line between downloads
    
    # Summary
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Total URLs:    {len(urls)}")
    print(f"  Successful:    {success_count}")
    print(f"  Failed:        {fail_count}")
    print(f"  Output dir:    {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
