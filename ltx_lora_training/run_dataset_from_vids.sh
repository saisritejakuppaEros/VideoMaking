#!/usr/bin/env bash
# Full dataset rebuild from raw videos in debunk_exisiting_youtubers/outputs/vids
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

export DEVICE="${DEVICE:-cuda}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Preprocess will use NUM_PROCESSES=${NUM_PROCESSES}"
echo

echo "=== Step 1/4: Scene-cut all videos ==="
bash "${SCRIPT_DIR}/run_scene_cut_all.sh"

echo
echo "=== Step 2/4: Caption all scene clips ==="
bash "${SCRIPT_DIR}/run_caption_all.sh"

echo
echo "=== Step 3/4: Build dataset.json ==="
bash "${SCRIPT_DIR}/run_build_dataset.sh"

echo
echo "=== Step 4/4: Preprocess latents (video + audio + text) ==="
OVERWRITE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  bash "${SCRIPT_DIR}/run_preprocess.sh"

echo
echo "Dataset ready for training."
