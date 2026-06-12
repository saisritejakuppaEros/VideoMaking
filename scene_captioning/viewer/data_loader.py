"""Load caption JSON outputs from outputs/captions/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CAPTIONS_DIR = SCRIPT_DIR / "outputs" / "captions"

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

META_CATEGORY_FIELDS = (
    "shot_type",
    "camera",
    "characters",
    "action",
    "objects",
    "environment",
    "visual_style",
    "emotion",
)

FIELD_LABELS: dict[str, str] = {
    "shot_type": "Shot types",
    "camera": "Camera",
    "characters": "Characters",
    "action": "Action",
    "objects": "Objects",
    "environment": "Environment",
    "visual_style": "Visual style",
    "emotion": "Emotion",
}


def list_caption_files() -> list[tuple[str, Path]]:
    """Return (video_title, captions.json path) pairs sorted by title."""
    if not CAPTIONS_DIR.is_dir():
        return []

    entries: list[tuple[str, Path]] = []
    for folder in sorted(CAPTIONS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        json_path = folder / "captions.json"
        if json_path.is_file():
            entries.append((folder.name, json_path))
    return entries


def load_caption_data(json_path: Path | str) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as handle:
        return json.load(handle)


def video_duration_seconds(scenes: list[dict[str, Any]]) -> float:
    if not scenes:
        return 0.0
    return max(float(scene.get("end_seconds", 0.0)) for scene in scenes)


def resolve_video_path(scene: dict[str, Any], data: dict[str, Any]) -> Path | None:
    clip = scene.get("clip_path")
    if clip and Path(clip).is_file():
        return Path(clip)

    source = data.get("source_video")
    if source and Path(source).is_file():
        return Path(source)
    return None


def scene_label(scene: dict[str, Any]) -> str:
    scene_id = scene.get("scene_id", "?")
    start = scene.get("start", "")
    end = scene.get("end", "")
    return f"Scene {scene_id} ({start} – {end})"


def get_storyboard(scene: dict[str, Any]) -> dict[str, str]:
    storyboard = scene.get("storyboard")
    if isinstance(storyboard, dict):
        return {key: str(storyboard.get(key, "") or "") for key in STORYBOARD_FIELDS}

    result: dict[str, str] = {}
    for key in STORYBOARD_FIELDS:
        result[key] = str(scene.get(key, "") or "")
    return result


def get_field_value(scene: dict[str, Any], field: str) -> str:
    storyboard = get_storyboard(scene)
    return (storyboard.get(field) or scene.get(field, "")).strip()


def aggregate_field_counts(
    scenes: list[dict[str, Any]],
    field: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for scene in scenes:
        value = get_field_value(scene, field)
        if value:
            counts[value] += 1
    return counts


def load_all_caption_entries() -> list[tuple[str, dict[str, Any], Path]]:
    """Return (video_title, caption_data, json_path) for every captioned video."""
    entries: list[tuple[str, dict[str, Any], Path]] = []
    for title, json_path in list_caption_files():
        entries.append((title, load_caption_data(json_path), json_path))
    return entries


def aggregate_corpus_field_counts(
    entries: list[tuple[str, dict[str, Any], Path]],
    field: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, data, _ in entries:
        for scene in data.get("scenes", []):
            value = get_field_value(scene, field)
            if value:
                counts[value] += 1
    return counts
