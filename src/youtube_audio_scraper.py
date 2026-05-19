import os
import argparse
from googleapiclient.discovery import build
from yt_dlp import YoutubeDL

'''
python src/youtube_audio_scraper.py --query "indian podcast english language the ranveer show" --max-results 10 --output "data/raw_audio"

python src/youtube_audio_scraper.py --channel UCneyi-aYq4VIBYIAQgWmk_w --max-results 200 --output data/raw_audio

python src/youtube_audio_scraper.py --id-file new_episodes.txt --output data/raw_audio

python src/youtube_audio_scraper.py --url "https://www.youtube.com/watch?v=jPoKmb3v4sM" --output data/raw_audio --filename ep_336

ffmpeg -i data/raw_audio/ep_011.wav -ss 00:24:45 -to 00:25:30 data/raw_audio/host/ep_011_speaker00_clip.wav

'''

def get_youtube_client(api_key):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)

def search_videos(youtube, query, max_results=10):
    resp = youtube.search().list(
        part="id,snippet",
        type="video",
        q=query,
        maxResults=max_results,
        order="relevance"
    ).execute()
    return [item["id"]["videoId"] for item in resp.get("items", [])]

def videos_from_channel(youtube, channel_id, max_results=10):
    """Get video IDs from channel, filtering out shorts (videos <= 60 seconds)."""
    import re
    
    all_ids = []
    next_page_token = None
    fetched = 0
    
    # Fetch more videos than needed since we'll filter out shorts
    fetch_limit = max_results * 2
    
    while len(all_ids) < max_results and fetched < fetch_limit:
        resp = youtube.search().list(
            part="id",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=min(50, fetch_limit - fetched),
            pageToken=next_page_token
        ).execute()
        
        video_ids = [item["id"]["videoId"] for item in resp.get("items", [])]
        fetched += len(video_ids)
        
        if not video_ids:
            break
        
        # Get duration details for these videos
        details = youtube.videos().list(
            part="contentDetails",
            id=",".join(video_ids)
        ).execute()
        
        for item in details.get("items", []):
            duration_str = item["contentDetails"]["duration"]
            # Parse ISO 8601 duration (e.g., PT1H23M45S)
            duration_seconds = parse_duration(duration_str)
            
            # Filter out shorts (videos <= 60 seconds)
            if duration_seconds > 60:
                all_ids.append(item["id"])
                if len(all_ids) >= max_results:
                    break
        
        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break
    
    return all_ids[:max_results]


def parse_duration(duration_str):
    """Convert ISO 8601 duration string to seconds. E.g., PT1H23M45S -> 5025"""
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    total_seconds = int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)
    return total_seconds

def extract_video_id(url_or_id):
    """Extract video ID from a URL or return as-is if already an ID."""
    if "watch?v=" in url_or_id:
        return url_or_id.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("/")[-1].split("?")[0]
    return url_or_id

def download_audio(video_id, output_dir, index, filename=None):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_name = filename if filename else f"ep_{index:03d}"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, f"{out_name}.%(ext)s"),
        "cookiefile": "cookies.txt",   # add this line
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": False,
        "no_warnings": True,
        "continue_dl": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    parser = argparse.ArgumentParser(description="YouTube audio scraper")
    parser.add_argument("--query", help="Search query to find videos")
    parser.add_argument("--channel", help="Channel ID for latest uploads")
    parser.add_argument("--url", help="Single YouTube URL or video ID to download")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output", default="youtube_audio")
    parser.add_argument("--filename", help="Output filename without extension (only used with --url)")
    parser.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY"))
    parser.add_argument("--id-file", default=None,
                        help="Text file with one video id or URL per line")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Need YOUTUBE_API_KEY or --api-key")
    if not (args.query or args.channel or args.id_file or args.url):
        raise SystemExit("Need --query, --channel, --id-file, or --url")

    youtube = get_youtube_client(args.api_key)
    ids = []

    if args.url:
        ids.append(extract_video_id(args.url))

    if args.id_file:
        with open(args.id_file, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if not txt:
                    continue
                ids.append(extract_video_id(txt))

    if args.query:
        ids.extend(search_videos(youtube, args.query, args.max_results))

    if args.channel:
        ids.extend(videos_from_channel(youtube, args.channel, args.max_results))

    ids = list(dict.fromkeys(ids))
    if not ids:
        print("No videos found.")
        return

    print(f"Downloading {len(ids)} audio track(s) to {args.output}")

    failed = []
    for idx, vid in enumerate(ids, start=341):
        filename = args.filename if (args.url and len(ids) == 1) else None
        print(f"-> ep_{idx:03d}: {vid}")
        try:
            download_audio(vid, args.output, idx, filename=filename)
        except Exception as e:
            print(f"  FAILED: {vid} — {e}")
            failed.append((idx, vid))

    # Update audio_files.txt with successfully downloaded files
    with open("audio_files.txt", "a", encoding="utf-8") as f:
        if args.url and len(ids) == 1 and args.filename:
            f.write(f"{os.path.join(args.output, f'{args.filename}.wav')}\n")
        else:
            for idx, vid in [(i, v) for i, v in enumerate(ids, start=341)
                             if (i, v) not in [(fi, fv) for fi, fv in failed]]:
                f.write(f"{os.path.join(args.output, f'ep_{idx:03d}.wav')}\n")

    print(f"\nCompleted: {len(ids) - len(failed)}/{len(ids)}")

    if failed:
        print(f"\nFailed downloads ({len(failed)}) — download these locally and transfer via scp:")
        for idx, vid in failed:
            print(f"  ep_{idx:03d}: https://www.youtube.com/watch?v={vid}")


if __name__ == "__main__":
    main()

