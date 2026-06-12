#!/usr/bin/env bash
# Caption all scenes for every scenes.json under outputs/scenes/.
# Loads Qwen3-VL once and processes every video in a single Python run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.env/bin/python"
CAPTION_SCRIPT="${SCRIPT_DIR}/caption_scenes.py"
SCENES_ROOT="${SCRIPT_DIR}/outputs/scenes"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
DEVICE="${DEVICE:-cuda}"
PROMPT_STYLE="${PROMPT_STYLE:-storyboard}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python venv not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${SCENES_ROOT}" ]]; then
  echo "Scenes directory not found: ${SCENES_ROOT}" >&2
  exit 1
fi

video_count="$(find "${SCENES_ROOT}" -mindepth 2 -maxdepth 2 -name 'scenes.json' | wc -l | tr -d ' ')"
if [[ "${video_count}" -eq 0 ]]; then
  echo "No scenes.json files found under ${SCENES_ROOT}" >&2
  exit 1
fi

echo "Found ${video_count} video(s) to caption"
echo "Model: ${MODEL_ID}"
echo "Device: ${DEVICE}"
echo "Loading model once, then captioning all scenes..."
echo

if "${PYTHON}" "${CAPTION_SCRIPT}" \
  --scenes-root "${SCENES_ROOT}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --prompt-style "${PROMPT_STYLE}" \
  --video-fps "${VIDEO_FPS}" \
  --max-tokens "${MAX_TOKENS}" \
  --video-reader "${FORCE_QWENVL_VIDEO_READER}"; then
  echo
  echo "Done. Caption run completed for ${video_count} video manifest(s)."
else
  exit_code=$?
  echo
  echo "Caption run exited with code ${exit_code}. Check the summary above for failed/skipped scenes." >&2
  exit "${exit_code}"
fi
