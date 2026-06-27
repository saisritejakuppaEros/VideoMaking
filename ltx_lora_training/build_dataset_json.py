#!/usr/bin/env python3
"""Build LTX trainer dataset.json from scene_captioning outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _caption_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def storyboard_to_caption(scene: dict) -> str:
    """Convert a Qwen storyboard scene into a single LTX-style caption."""
    storyboard = scene.get("storyboard")
    if isinstance(storyboard, dict):
        parts = [
            _caption_text(storyboard.get("scene_summary", "")),
            _caption_text(storyboard.get("action", "")),
            _caption_text(storyboard.get("environment", "")),
            _caption_text(storyboard.get("visual_style", "")),
            _caption_text(storyboard.get("emotion", "")),
        ]
        text = " ".join(p for p in parts if p)
        if text:
            return text

    caption = scene.get("caption", "")
    if isinstance(caption, str) and caption.strip():
        caption = caption.strip()
        if caption.startswith("{"):
            try:
                parsed = json.loads(caption)
                if isinstance(parsed, dict):
                    return storyboard_to_caption({"storyboard": parsed})
            except json.JSONDecodeError:
                pass
        return caption

    return ""


def build_dataset(
    captions_root: Path,
    output_path: Path,
    min_duration_seconds: float,
    use_symlinks: bool,
    scenes_out: Path | None,
) -> dict:
    entries: list[dict[str, str]] = []
    skipped_short = 0
    skipped_missing = 0
    skipped_empty = 0

    for captions_file in sorted(captions_root.glob("*/captions.json")):
        data = json.loads(captions_file.read_text(encoding="utf-8"))
        for scene in data.get("scenes", []):
            if scene.get("caption_status") != "success":
                continue

            duration = float(scene.get("duration_seconds") or 0.0)
            if duration < min_duration_seconds:
                skipped_short += 1
                continue

            clip_path = Path(scene["clip_path"])
            if not clip_path.is_file():
                skipped_missing += 1
                continue

            caption = storyboard_to_caption(scene)
            if not caption:
                skipped_empty += 1
                continue

            if use_symlinks and scenes_out is not None:
                rel_parent = clip_path.parent.name
                target_dir = scenes_out / rel_parent
                target_dir.mkdir(parents=True, exist_ok=True)
                target_clip = target_dir / clip_path.name
                if not target_clip.exists():
                    target_clip.symlink_to(clip_path)
                media_path = str(target_clip.relative_to(output_path.parent))
            else:
                media_path = str(clip_path)

            entries.append({"caption": caption, "media_path": media_path})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "entries": len(entries),
        "skipped_short": skipped_short,
        "skipped_missing": skipped_missing,
        "skipped_empty": skipped_empty,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions-root",
        type=Path,
        default=Path(
            "/mnt/data0/harsha/new_paper/VideoMaking/scene_captioning/outputs/captions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data0/harsha/new_paper/VideoMaking/ltx_lora_training/data/dataset.json"),
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=5.0,
        help="Skip clips shorter than this (matches split_scenes --filter-shorter-than 5s).",
    )
    parser.add_argument(
        "--symlink-clips",
        action="store_true",
        help="Symlink clips into data/scenes/ and use relative media_path entries.",
    )
    parser.add_argument(
        "--scenes-out",
        type=Path,
        default=Path("/mnt/data0/harsha/new_paper/VideoMaking/ltx_lora_training/data/scenes"),
    )
    args = parser.parse_args()

    summary = build_dataset(
        captions_root=args.captions_root,
        output_path=args.output,
        min_duration_seconds=args.min_duration_seconds,
        use_symlinks=args.symlink_clips,
        scenes_out=args.scenes_out if args.symlink_clips else None,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
