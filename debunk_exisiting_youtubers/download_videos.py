#!/usr/bin/env python3
"""Download random videos and metadata from educational YouTube channels."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

from channels import CHANNELS

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
VIDS_DIR = OUTPUTS_DIR / "vids"
METADATA_DIR = OUTPUTS_DIR / "metadata"
LOG_PATH = METADATA_DIR / "download_log.json"


def load_download_log() -> dict:
    if not LOG_PATH.exists():
        return {}
    with LOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_download_log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def metadata_to_result(metadata: dict) -> dict:
    video_id = metadata.get("video_id") or "unknown"
    channel_name = metadata.get("source_channel_key") or "unknown"
    meta_path = METADATA_DIR / channel_name / f"{video_id}.json"
    return {
        "channel": channel_name,
        "video_id": video_id,
        "title": metadata.get("title"),
        "url": metadata.get("url"),
        "metadata_path": str(meta_path),
        "video_path": metadata.get("local_video_path"),
        "transcript_path": metadata.get("local_transcript_path"),
        "thumbnail_path": metadata.get("local_thumbnail_path"),
    }


def load_channel_completed(channel_name: str) -> list[dict]:
    channel_meta_dir = METADATA_DIR / channel_name
    if not channel_meta_dir.exists():
        return []

    completed = []
    for meta_path in channel_meta_dir.glob("*.json"):
        with meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        video_path = metadata.get("local_video_path")
        if video_path and Path(video_path).exists():
            completed.append(metadata_to_result(metadata))

    return completed


def get_channel_videos(channel_url: str, playlist_end: int) -> list[str]:
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": playlist_end,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    videos = []
    for entry in info.get("entries") or []:
        if entry and entry.get("id"):
            videos.append(f"https://www.youtube.com/watch?v={entry['id']}")

    return videos


def extract_metadata(video_url: str) -> dict | None:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except (DownloadError, ExtractorError):
        return None

    if not info:
        return None

    subtitles = {}
    for source in ("subtitles", "automatic_captions"):
        for lang, tracks in (info.get(source) or {}).items():
            if tracks:
                subtitles[lang] = [
                    {"ext": track.get("ext"), "url": track.get("url")}
                    for track in tracks
                ]

    return {
        "video_id": info.get("id"),
        "url": info.get("webpage_url") or video_url,
        "title": info.get("title"),
        "description": info.get("description"),
        "duration_seconds": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "channel_url": info.get("channel_url"),
        "tags": info.get("tags") or [],
        "categories": info.get("categories") or [],
        "thumbnail": info.get("thumbnail"),
        "thumbnails": info.get("thumbnails") or [],
        "subtitles": subtitles,
    }


def download_video(video_url: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "restrictfilenames": False,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            if not info:
                return None

            ext = info.get("ext") or "mp4"
            if ext != "mp4":
                ext = "mp4"
            filename = output_dir / f"{info['title']}.{ext}"
            if not filename.exists():
                candidates = list(output_dir.glob(f"{info['title']}.*"))
                filename = candidates[0] if candidates else None

            return filename
    except (DownloadError, ExtractorError):
        return None


def download_thumbnail(metadata: dict, output_dir: Path) -> Path | None:
    thumbnail_url = metadata.get("thumbnail")
    if not thumbnail_url:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = metadata.get("video_id") or "unknown"
    output_path = output_dir / f"{video_id}.jpg"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writethumbnail": True,
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([metadata["url"]])
    except (DownloadError, ExtractorError):
        return None

    for candidate in output_dir.glob(f"{video_id}.*"):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            if candidate != output_path and candidate.suffix.lower() != ".jpg":
                candidate.rename(output_path)
            return output_path if output_path.exists() else candidate

    return None


def download_transcript(metadata: dict, output_dir: Path) -> Path | None:
    subtitles = metadata.get("subtitles") or {}
    tracks = subtitles.get("en") or next(iter(subtitles.values()), None)
    if not tracks:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = metadata.get("video_id") or "unknown"
    output_path = output_dir / f"{video_id}.vtt"

    preferred = next(
        (track for track in tracks if track.get("ext") in {"vtt", "srv3", "ttml"}),
        tracks[0],
    )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": preferred.get("ext") or "vtt",
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([metadata["url"]])
    except (DownloadError, ExtractorError):
        return None

    for candidate in output_dir.glob(f"{video_id}.*"):
        if candidate.suffix.lower() in {".vtt", ".srt", ".srv3", ".ttml", ".json3"}:
            return candidate

    return output_path if output_path.exists() else None


def save_metadata(metadata: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_channel(
    channel_name: str,
    channel_url: str,
    videos_per_channel: int,
    playlist_end: int,
    seed: int | None,
    exclude_video_ids: set[str] | None = None,
    existing_results: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    print(f"\nProcessing {channel_name} ({channel_url})")

    existing_results = existing_results or []
    exclude_video_ids = exclude_video_ids or set()
    remaining = videos_per_channel - len(existing_results)

    if remaining <= 0:
        print(f"  Already complete ({len(existing_results)}/{videos_per_channel})")
        return existing_results, []

    videos = get_channel_videos(channel_url, playlist_end)
    if not videos:
        print("  Skipping: no videos found")
        return existing_results, []

    rng = random.Random(seed)
    candidates = [
        url
        for url in videos
        if url.rsplit("v=", 1)[-1] not in exclude_video_ids
    ]
    rng.shuffle(candidates)

    if not candidates:
        print("  Skipping: no new candidates left to try")
        return existing_results, []

    channel_vid_dir = VIDS_DIR / channel_name
    channel_meta_dir = METADATA_DIR / channel_name
    results = list(existing_results)
    skipped = []

    for video_url in candidates:
        if len(results) >= videos_per_channel:
            break

        print(f"  Fetching metadata: {video_url}")
        metadata = extract_metadata(video_url)
        if metadata is None:
            reason = "metadata unavailable (members-only, private, or removed)"
            print(f"  Skipping: {reason}")
            skipped.append({"url": video_url, "reason": reason})
            continue

        metadata["source_channel_key"] = channel_name
        metadata["source_channel_url"] = channel_url
        metadata["collected_at"] = datetime.now(timezone.utc).isoformat()

        video_id = metadata.get("video_id") or "unknown"
        meta_path = channel_meta_dir / f"{video_id}.json"

        print(f"  Downloading video: {metadata.get('title')}")
        video_path = download_video(video_url, channel_vid_dir)
        if video_path is None:
            reason = "video download failed (members-only, private, or removed)"
            print(f"  Skipping: {reason}")
            skipped.append({"url": video_url, "reason": reason, "title": metadata.get("title")})
            continue

        metadata["local_video_path"] = str(video_path)

        print("  Downloading thumbnail")
        thumb_path = download_thumbnail(metadata, channel_meta_dir / "thumbnails")
        metadata["local_thumbnail_path"] = str(thumb_path) if thumb_path else None

        print("  Downloading transcript")
        transcript_path = download_transcript(metadata, channel_meta_dir / "transcripts")
        metadata["local_transcript_path"] = (
            str(transcript_path) if transcript_path else None
        )

        save_metadata(metadata, meta_path)
        print(f"  Saved metadata: {meta_path}")

        results.append(
            {
                "channel": channel_name,
                "video_id": video_id,
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "metadata_path": str(meta_path),
                "video_path": metadata.get("local_video_path"),
                "transcript_path": metadata.get("local_transcript_path"),
                "thumbnail_path": metadata.get("local_thumbnail_path"),
            }
        )

    new_count = len(results) - len(existing_results)
    if len(results) < videos_per_channel:
        print(
            f"  Warning: only have {len(results)}/{videos_per_channel} "
            f"after trying {len(candidates)} candidate(s) ({new_count} new)"
        )

    return results, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download random videos and metadata from YouTube channels."
    )
    parser.add_argument(
        "--videos-per-channel",
        type=int,
        default=2,
        help="Number of random videos to download per channel (default: 2)",
    )
    parser.add_argument(
        "--playlist-end",
        type=int,
        default=100,
        help="How many recent channel videos to sample from (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible video selection",
    )
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        metavar="KEY",
        help="Only process this channel key (repeatable). Default: all channels.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Only fill in missing work: skip completed channels/videos and "
            "retry channels that failed or were interrupted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    VIDS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    channels = CHANNELS
    if args.channels:
        unknown = [key for key in args.channels if key not in CHANNELS]
        if unknown:
            raise SystemExit(
                f"Unknown channel key(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(CHANNELS)}"
            )
        channels = {key: CHANNELS[key] for key in args.channels}

    prior_log = load_download_log() if args.resume else {}
    seed = args.seed if args.seed is not None else prior_log.get("seed")

    all_skipped: list[dict] = list(prior_log.get("skipped", [])) if args.resume else []

    if args.resume:
        print("Resume mode: skipping channels and videos already on disk.")

    for name, url in channels.items():
        existing_results = load_channel_completed(name) if args.resume else []

        if args.resume and len(existing_results) >= args.videos_per_channel:
            print(f"\nSkipping {name}: already complete ({len(existing_results)} video(s))")
            continue

        if args.resume and existing_results:
            print(
                f"\nResuming {name}: {len(existing_results)}/{args.videos_per_channel} "
                "already downloaded"
            )

        channel_seed = None if seed is None else seed + hash(name) % 10_000
        exclude_ids = {item["video_id"] for item in existing_results} if args.resume else set()

        _, skipped = process_channel(
            name,
            url,
            videos_per_channel=args.videos_per_channel,
            playlist_end=args.playlist_end,
            seed=channel_seed,
            exclude_video_ids=exclude_ids,
            existing_results=existing_results if args.resume else None,
        )
        all_skipped.extend(skipped)

        if args.resume:
            checkpoint_results = []
            for channel_name in channels:
                checkpoint_results.extend(load_channel_completed(channel_name))
            save_download_log(
                {
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "videos_per_channel": args.videos_per_channel,
                    "playlist_end": args.playlist_end,
                    "seed": seed,
                    "total_downloaded": len(checkpoint_results),
                    "total_skipped": len(all_skipped),
                    "videos": checkpoint_results,
                    "skipped": all_skipped,
                }
            )

    all_results = []
    for name in channels:
        all_results.extend(load_channel_completed(name))

    log_payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "videos_per_channel": args.videos_per_channel,
        "playlist_end": args.playlist_end,
        "seed": seed,
        "total_downloaded": len(all_results),
        "total_skipped": len(all_skipped),
        "videos": all_results,
        "skipped": all_skipped,
    }
    save_download_log(log_payload)

    print(f"\nDone. {len(all_results)} video(s) on disk.")
    print(f"Videos:   {VIDS_DIR}")
    print(f"Metadata: {METADATA_DIR}")
    print(f"Log:      {LOG_PATH}")


if __name__ == "__main__":
    main()
