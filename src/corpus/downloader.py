"""YouTube channel scraper and audio downloader for Iván Cepeda speeches."""

import json
import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)

CHANNEL_URL = "https://www.youtube.com/@IvanCepedaCastro/videos"
MANIFEST_PATH = Path("data/speech_manifest.json")
AUDIO_DIR = Path("data/audio")
RAW_DIR = Path("data/raw")


def scrape_channel_metadata(
    channel_url: str = CHANNEL_URL,
    max_videos: int | None = None,
) -> list[dict]:
    """Scrape video metadata from YouTube channel (no download).

    Returns list of dicts with: id, title, url, duration, view_count.
    Videos are returned newest-first.
    """
    ydl_opts: dict = {
        "extract_flat": True,
        "quiet": True,
    }
    if max_videos:
        ydl_opts["playlistend"] = max_videos

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = []
    for entry in info.get("entries", []):
        entries.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "url": entry.get("url"),
            "duration_seconds": int(entry.get("duration") or 0),
            "view_count": entry.get("view_count"),
        })

    logger.info("Scraped %d videos from channel", len(entries))
    return entries


def get_video_full_metadata(video_url: str) -> dict:
    """Extract full metadata for a single video (includes upload_date)."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    upload_date = info.get("upload_date")  # format: YYYYMMDD
    if upload_date and len(upload_date) == 8:
        formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    else:
        formatted_date = None

    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "url": info.get("webpage_url"),
        "duration_seconds": int(info.get("duration") or 0),
        "upload_date": formatted_date,
        "view_count": info.get("view_count"),
        "description": info.get("description", ""),
        "source": "youtube",
    }


def download_audio(video_url: str, output_dir: Path = AUDIO_DIR) -> Path:
    """Download audio from a YouTube video as MP3.

    Returns the path to the downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": output_template,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info["id"]

    audio_path = output_dir / f"{video_id}.mp3"
    if not audio_path.exists():
        # yt-dlp sometimes uses different extension
        for ext in ["m4a", "webm", "ogg", "wav"]:
            alt = output_dir / f"{video_id}.{ext}"
            if alt.exists():
                audio_path = alt
                break

    logger.info("Downloaded audio: %s", audio_path)
    return audio_path


def build_manifest(
    entries: list[dict],
    limit: int | None = None,
    manifest_path: Path = MANIFEST_PATH,
) -> list[dict]:
    """Build speech manifest from scraped channel entries.

    Fetches full metadata for each video and saves to manifest JSON.
    If manifest already exists, merges new entries (skips existing IDs).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if present
    existing = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for entry in json.load(f):
                existing[entry["id"]] = entry

    entries_to_process = entries[:limit] if limit else entries
    manifest = list(existing.values())

    for i, entry in enumerate(entries_to_process):
        video_id = entry["id"]
        if video_id in existing:
            logger.info("Skipping %s (already in manifest)", video_id)
            continue

        logger.info(
            "Fetching metadata %d/%d: %s",
            i + 1, len(entries_to_process), entry["title"],
        )
        try:
            full_meta = get_video_full_metadata(entry["url"])
            full_meta["status"] = "pending"
            manifest.append(full_meta)
            existing[video_id] = full_meta
        except Exception:
            logger.exception("Failed to fetch metadata for %s", video_id)
            continue

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Manifest saved: %d entries at %s", len(manifest), manifest_path)
    return manifest


def register_text_speech(
    speech_id: str,
    title: str,
    text_file: Path,
    speech_date: str | None = None,
    location: str | None = None,
    event: str | None = None,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Register a text-only speech (from website) into the manifest.

    The text file should already be in data/raw/{speech_id}.txt.
    """
    entry = {
        "id": speech_id,
        "title": title,
        "url": None,
        "duration_seconds": None,
        "upload_date": speech_date,
        "view_count": None,
        "description": "",
        "source": "website_text",
        "text_file": str(text_file),
        "status": "has_text",
        "location": location,
        "event": event,
    }

    # Load and update manifest
    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Check if already exists
    existing_ids = {e["id"] for e in manifest}
    if speech_id in existing_ids:
        logger.warning("Speech %s already in manifest", speech_id)
        return entry

    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Registered text speech: %s", title)
    return entry


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Scrape channel and build manifest for first 45 videos
    logger.info("Scraping channel metadata...")
    entries = scrape_channel_metadata(max_videos=45)
    logger.info("Found %d videos", len(entries))

    # Show summary
    for i, e in enumerate(entries):
        mins = e["duration_seconds"] / 60
        print(f"  {i+1:3d}. [{mins:5.1f}m] {e['title']}")
