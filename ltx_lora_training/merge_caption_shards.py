#!/usr/bin/env python3
"""Merge per-GPU caption shard files into one LTX dataset.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(shard_paths: list[Path], output_path: Path) -> dict:
    merged: dict[str, str] = {}
    for shard in shard_paths:
        if not shard.is_file():
            continue
        entries = json.loads(shard.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Expected JSON list in {shard}")
        for entry in entries:
            media_path = entry["media_path"]
            merged[media_path] = entry["caption"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"caption": caption, "media_path": media_path} for media_path, caption in sorted(merged.items())]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"shards": len(shard_paths), "entries": len(payload), "output": str(output_path)}


def collect_per_folder_captions(input_root: Path, output_path: Path) -> dict:
    shard_paths = sorted(input_root.glob("*/captions.json"))
    return merge_shards(shard_paths, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path)
    parser.add_argument("--input-root", type=Path, help="Root with per-folder captions.json files")
    parser.add_argument("--pattern", default="dataset.part*.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input_root is not None:
        summary = collect_per_folder_captions(args.input_root, args.output)
    elif args.shards_dir is not None:
        shard_paths = sorted(args.shards_dir.glob(args.pattern))
        summary = merge_shards(shard_paths, args.output)
    else:
        parser.error("Provide --input-root or --shards-dir")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
