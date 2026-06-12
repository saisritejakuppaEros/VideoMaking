#!/usr/bin/env bash
set -euo pipefail

PYTHON="/mnt/data0/harsha/new_paper/VideoMaking/.env/bin/python"
SCENE_CUT="/mnt/data0/harsha/new_paper/VideoMaking/scene_captioning/scene_cut.py"
VIDS_ROOT="/mnt/data0/harsha/new_paper/VideoMaking/debunk_exisiting_youtubers/outputs"

START=0
END=120

mapfile -d '' videos < <(find "$VIDS_ROOT" -type f -name '*.mp4' -print0 | sort -z)

total=${#videos[@]}
if (( total == 0 )); then
  echo "No .mp4 files found under $VIDS_ROOT"
  exit 1
fi

echo "Found $total video(s) under $VIDS_ROOT"
echo "Processing seconds $START to $END for each video"
echo

for i in "${!videos[@]}"; do
  video="${videos[$i]}"
  n=$((i + 1))
  echo "========================================"
  echo "[$n/$total] Processing: $video"
  echo "========================================"
  "$PYTHON" "$SCENE_CUT" "$video" --start "$START" --end "$END"
  echo
done

echo "Done. Processed $total video(s)."
