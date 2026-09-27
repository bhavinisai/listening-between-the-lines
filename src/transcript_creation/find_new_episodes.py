import argparse
from yt_dlp import YoutubeDL

'''
python src/transcript_creation/find_new_episodes.py \
    --url "https://www.youtube.com/@poojajdhingra/podcasts" \
    --min-minutes 30 \
    --output new_episodes.txt
'''


def flat_extract(url):
    with YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "no_warnings": True}) as ydl:
        return ydl.extract_info(url, download=False)


def collect_videos(source_url):
    """
    Resolve a channel tab / playlist / video URL down to individual video
    entries (id, title, duration, url). Channel tabs (e.g. .../podcasts)
    list one or more playlists rather than videos directly, so nested
    playlist entries (ie_key == "YoutubeTab") are expanded one level.
    """
    top = flat_extract(source_url)

    if "entries" not in top:
        # A single video URL was passed directly.
        return [top]

    videos = []
    for entry in top["entries"]:
        if entry is None:
            continue
        if entry.get("ie_key") == "YoutubeTab":
            sub = flat_extract(entry["url"])
            videos.extend(e for e in sub.get("entries", []) if e is not None)
        else:
            videos.append(entry)
    return videos


def main():
    ap = argparse.ArgumentParser(
        description="Find videos longer than a minimum duration in a YouTube "
                    "channel tab / playlist, and write their URLs to a file."
    )
    ap.add_argument("--url", required=True, help="Channel tab, playlist, or video URL")
    ap.add_argument("--min-minutes", type=float, default=30,
                    help="Minimum video duration in minutes (default: 30)")
    ap.add_argument("--output", default="new_episodes.txt",
                    help="Output text file, one video URL per line")
    args = ap.parse_args()

    min_seconds = args.min_minutes * 60
    videos = collect_videos(args.url)

    kept = []
    for v in videos:
        duration = v.get("duration")
        if duration is not None and duration > min_seconds:
            video_id = v.get("id")
            kept.append((video_id, v.get("title"), duration))

    with open(args.output, "w", encoding="utf-8") as f:
        for video_id, _, _ in kept:
            f.write(f"https://www.youtube.com/watch?v={video_id}\n")

    print(f"Found {len(videos)} total videos, {len(kept)} longer than {args.min_minutes:.0f} min.")
    print(f"Wrote {len(kept)} URLs to {args.output}")
    for video_id, title, duration in kept:
        print(f"  {video_id}  ({duration // 60}m {duration % 60}s)  {title}")


if __name__ == "__main__":
    main()
