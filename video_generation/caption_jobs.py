#!/usr/bin/env python3
"""Load scene caption manifests and build SANA generation jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STORYBOARD_FIELDS = (
    "scene_summary",
    "shot_type",
    "camera",
    "characters",
    "action",
    "objects",
    "environment",
    "visual_style",
    "emotion",
)

DEFAULT_CAPTIONS_ROOT = (
    Path(__file__).resolve().parent.parent
    / "scene_captioning"
    / "outputs"
    / "captions"
)


def get_storyboard(scene: dict[str, Any]) -> dict[str, str]:
    storyboard = scene.get("storyboard")
    if isinstance(storyboard, dict):
        return {key: str(storyboard.get(key, "") or "") for key in STORYBOARD_FIELDS}

    return {key: str(scene.get(key, "") or "") for key in STORYBOARD_FIELDS}


def scene_prompt(scene: dict[str, Any]) -> str:
    storyboard = get_storyboard(scene)
    parts: list[str] = []

    if storyboard["shot_type"]:
        parts.append(f"{storyboard['shot_type']}.")
    if storyboard["camera"]:
        parts.append(f"{storyboard['camera']}.")
    if storyboard["scene_summary"]:
        parts.append(storyboard["scene_summary"])
    if storyboard["action"]:
        parts.append(f"{storyboard['action']}.")
    if storyboard["characters"]:
        parts.append(f"Characters: {storyboard['characters']}.")
    if storyboard["objects"]:
        parts.append(f"Objects: {storyboard['objects']}.")
    if storyboard["environment"]:
        parts.append(f"{storyboard['environment']}.")
    if storyboard["visual_style"]:
        parts.append(f"Visual style: {storyboard['visual_style']}.")
    if storyboard["emotion"]:
        parts.append(f"Mood: {storyboard['emotion']}.")

    prompt = " ".join(part.strip() for part in parts if part.strip()).strip()
    if not prompt:
        raise ValueError(f"Scene {scene.get('scene_id')} has no usable caption text.")
    return prompt


def resolve_reference_video(scene: dict[str, Any], data: dict[str, Any]) -> Path:
    clip_path = scene.get("clip_path")
    if clip_path:
        clip = Path(clip_path)
        if clip.is_file():
            return clip

    source_video = data.get("source_video")
    if source_video and Path(source_video).is_file():
        return Path(source_video)

    raise FileError(
        f"Scene {scene.get('scene_id')} has no readable clip_path or source_video."
    )


@dataclass(frozen=True)
class CaptionJob:
    video_title: str
    scene_id: int
    prompt: str
    reference_video: Path | None
    frame_time: float
    captions_json: Path
    output_stem: str


def scene_output_stem(video_title: str, scene_id: int) -> str:
    return f"scene_{int(scene_id):03d}"


def collect_caption_jobs(
    captions_root: Path,
    *,
    video_title: str | None = None,
    scene_ids: set[int] | None = None,
) -> list[CaptionJob]:
    if not captions_root.is_dir():
        raise FileNotFoundError(f"Captions directory not found: {captions_root}")

    jobs: list[CaptionJob] = []
    for folder in sorted(captions_root.iterdir()):
        if not folder.is_dir():
            continue
        if video_title and folder.name != video_title:
            continue

        captions_json = folder / "captions.json"
        if not captions_json.is_file():
            continue

        with captions_json.open(encoding="utf-8") as handle:
            data = json.load(handle)

        for scene in data.get("scenes", []):
            if scene.get("caption_status") != "success":
                continue

            scene_id = int(scene["scene_id"])
            if scene_ids and scene_id not in scene_ids:
                continue

            prompt = scene_prompt(scene)
            clip_path = scene.get("clip_path")
            reference_video = None
            frame_time = 0.0

            if clip_path and Path(clip_path).is_file():
                reference_video = Path(clip_path)
            else:
                source_video = data.get("source_video")
                if source_video and Path(source_video).is_file():
                    reference_video = Path(source_video)
                    frame_time = float(scene.get("start_seconds") or 0.0)

            jobs.append(
                CaptionJob(
                    video_title=folder.name,
                    scene_id=scene_id,
                    prompt=prompt,
                    reference_video=reference_video,
                    frame_time=frame_time,
                    captions_json=captions_json,
                    output_stem=scene_output_stem(folder.name, scene_id),
                )
            )

    if not jobs:
        raise ValueError(f"No captioned scenes found under {captions_root}.")
    return jobs


def add_caption_batch_args(parser) -> None:
    parser.add_argument(
        "--captions-root",
        default=None,
        help=(
            "Root folder containing per-video captions.json files "
            f"(default when using run scripts: {DEFAULT_CAPTIONS_ROOT})."
        ),
    )
    parser.add_argument(
        "--video-title",
        default=None,
        help="Process only this video folder name under captions-root.",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        action="append",
        dest="scene_ids",
        help="Process only these scene ids (repeatable).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes whose output mp4 already exists.",
    )
