#!/usr/bin/env python3
"""Detect scene boundaries with PySceneDetect and split a video into clips."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from scenedetect import ContentDetector, FrameTimecode, SceneManager, open_video, split_video_ffmpeg

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "scenes"

TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d+))?$")


def parse_time(value: str | None) -> float | None:
    """Parse seconds (float) or HH:MM:SS[.ms] into seconds."""
    if value is None:
        return None

    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    match = TIME_RE.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: {value!r}. Use seconds or HH:MM:SS."
        )

    hours, minutes, seconds, millis = match.groups()
    total = int(minutes) * 60 + int(seconds)
    if hours:
        total += int(hours) * 3600
    if millis:
        total += float(f"0.{millis}")
    return float(total)


def format_timecode(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def detect_scenes(
    video_path: Path,
    *,
    threshold: float,
    backend: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> tuple[list[tuple[float, float]], float]:
    video = open_video(str(video_path), backend=backend)
    frame_rate = float(video.frame_rate)

    if start_seconds is not None:
        video.seek(FrameTimecode(start_seconds, fps=frame_rate))

    end_timecode = (
        FrameTimecode(end_seconds, fps=frame_rate) if end_seconds is not None else None
    )

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(
        video=video,
        show_progress=True,
        end_time=end_timecode,
    )
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    if not scene_list:
        duration = end_seconds
        if duration is None:
            duration = video.duration.seconds if video.duration else 0.0
        start = start_seconds or 0.0
        if duration and duration > start:
            return [(start, duration)], frame_rate
        return [], frame_rate

    return [
        (scene[0].seconds, scene[1].seconds)
        for scene in scene_list
    ], frame_rate


def scene_tuples_to_frametimecodes(
    scene_seconds: list[tuple[float, float]],
    frame_rate: float,
):
    return [
        (
            FrameTimecode(start, fps=frame_rate),
            FrameTimecode(end, fps=frame_rate),
        )
        for start, end in scene_seconds
    ]


def build_scene_records(
    video_path: Path,
    scene_seconds: list[tuple[float, float]],
    output_dir: Path,
    *,
    frame_rate: float,
    split_clips: bool,
) -> list[dict]:
    clip_paths: dict[int, str] = {}
    if split_clips and scene_seconds:
        scene_list = scene_tuples_to_frametimecodes(scene_seconds, frame_rate)
        split_video_ffmpeg(
            str(video_path),
            scene_list,
            output_dir=output_dir,
            show_progress=True,
        )

        for clip_path in sorted(output_dir.glob(f"{video_path.stem}-Scene-*.mp4")):
            scene_number = int(clip_path.stem.rsplit("-", 1)[-1])
            clip_paths[scene_number] = str(clip_path.resolve())

    scenes = []
    for index, (start, end) in enumerate(scene_seconds, start=1):
        scenes.append(
            {
                "scene_id": index,
                "start": format_timecode(start),
                "end": format_timecode(end),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "clip_path": clip_paths.get(index),
            }
        )
    return scenes


def run_scene_cut(args: argparse.Namespace) -> dict:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    if (
        start_seconds is not None
        and end_seconds is not None
        and start_seconds >= end_seconds
    ):
        raise ValueError("--start must be earlier than --end")

    scene_seconds, frame_rate = detect_scenes(
        video_path,
        threshold=args.threshold,
        backend=args.backend,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    scenes = build_scene_records(
        video_path,
        scene_seconds,
        output_dir,
        frame_rate=frame_rate,
        split_clips=not args.no_split,
    )

    manifest = {
        "video_path": str(video_path),
        "video_name": video_path.stem,
        "output_dir": str(output_dir),
        "scene_count": len(scenes),
        "threshold": args.threshold,
        "backend": args.backend,
        "frame_rate": frame_rate,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenes": scenes,
    }

    manifest_path = output_dir / "scenes.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Detected {len(scenes)} scene(s)")
    print(f"Manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect scene cuts and optionally split a video into clips."
    )
    parser.add_argument(
        "video",
        help="Path to the input video (e.g. outputs/vids/channel/video.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for scene clips and scenes.json (default: outputs/scenes/<video_stem>)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="ContentDetector threshold (lower = more cuts, default: 27.0)",
    )
    parser.add_argument(
        "--backend",
        choices=["pyav", "opencv", "moviepy"],
        default="pyav",
        help="Video decoder backend. Use pyav for AV1/YouTube downloads (default: pyav)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Only process from this time (seconds or HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Only process until this time (seconds or HH:MM:SS)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Detect scenes and write scenes.json without creating clip files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        video_stem = Path(args.video).expanduser().stem
        args.output_dir = DEFAULT_OUTPUT_DIR / video_stem

    run_scene_cut(args)


if __name__ == "__main__":
    main()
