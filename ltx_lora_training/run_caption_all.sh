#!/usr/bin/env bash
# Caption all scene clips under scene_captioning/outputs/scenes (Qwen3-VL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

CAPTION_SCRIPT="${SCENE_CAPTIONING_ROOT}/caption_scenes.py"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
DEVICE="${DEVICE:-cuda}"
PROMPT_STYLE="${PROMPT_STYLE:-storyboard}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if [[ ! -d "${SCENES_ROOT}" ]]; then
  echo "Scenes directory not found: ${SCENES_ROOT}" >&2
  echo "Run: bash run_scene_cut_all.sh" >&2
  exit 1
fi

video_count="$(find "${SCENES_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
if [[ "${video_count}" -eq 0 ]]; then
  echo "No scene folders found under ${SCENES_ROOT}" >&2
  exit 1
fi

echo "Captioning scenes for ${video_count} video(s)"
echo "  scenes root: ${SCENES_ROOT}"
echo "  model:       ${MODEL_ID}"
echo

"${VENV_PYTHON}" "${CAPTION_SCRIPT}" \
  --scenes-root "${SCENES_ROOT}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --prompt-style "${PROMPT_STYLE}" \
  --video-fps "${VIDEO_FPS}" \
  --max-tokens "${MAX_TOKENS}" \
  --video-reader "${FORCE_QWENVL_VIDEO_READER}"

echo
echo "Captions written under ${CAPTIONS_ROOT}"
