#!/usr/bin/env bash
# Steps 2–4 of LTX dataset prep after scene clips exist under scene_captioning/outputs/scenes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

echo "=== Step 2/3: Caption scene clips (LTX official Qwen2.5-Omni, video+audio) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash "${SCRIPT_DIR}/run_caption_multi_gpu.sh"

echo
echo "=== Collect per-folder captions into dataset.json for preprocessing ==="
"${VENV_PYTHON}" "${SCRIPT_DIR}/merge_caption_shards.py" \
  --input-root "${SCENE_CAPTION_OUTPUT}" \
  --output "${DATASET_JSON}"

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Expected merged dataset at ${DATASET_JSON}" >&2
  exit 1
fi

echo
echo "=== Step 3/3: Preprocess latents + text embeddings + audio ==="

OVERWRITE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  DATASET_JSON="${DATASET_JSON}" \
  PRECOMPUTED_ROOT="${SCENES_ROOT}/.precomputed" \
  bash "${SCRIPT_DIR}/run_preprocess.sh"

echo
echo "Done."
echo "  per-folder captions: ${SCENE_CAPTION_OUTPUT}/<video>/captions.json"
echo "  merged dataset.json: ${DATASET_JSON}"
echo "  preprocessed data:   ${PRECOMPUTED_ROOT}"
