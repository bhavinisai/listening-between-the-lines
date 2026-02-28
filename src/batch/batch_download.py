import os
import json
from youtube_transcript_api import YouTubeTranscriptApi

EPISODE_LIST_PATH = "src/episode_list.txt"
OUTPUT_DIR = "data/raw_transcripts"

def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from different URL formats.
    If the string is already an ID, just return it.
    """
    url = url.strip()

    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    if "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[-1].split("?")[0]
    
    return url

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print(f"Reading episode list from: {EPISODE_LIST_PATH}")
    with open(EPISODE_LIST_PATH, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URL(s)")

    api = YouTubeTranscriptApi()

    for idx, url in enumerate(urls, start=1):
        video_id = extract_video_id(url)

        print(f"\nProcessing Episode {idx}: {url}")
        print(f"  → video_id = {video_id}")

        try:
            
            fetched = api.fetch(
                video_id,
                languages=["en", "en-IN", "hi"]
            )

            raw_data = fetched.to_raw_data()

            out_path = os.path.join(OUTPUT_DIR, f"ep{idx:03d}_raw.json")
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(raw_data, out_f, ensure_ascii=False, indent=2)

            print(f"   Saved: {out_path}")

        except Exception as e:
            print(f"   Failed for {url}: {e}")

if __name__ == "__main__":
    main()