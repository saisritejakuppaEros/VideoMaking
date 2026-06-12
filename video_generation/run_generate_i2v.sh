#!/usr/bin/env bash
# Generate videos from scene captions + first frame of each scene clip (image-to-video).
# By default, reads captions from scene_captioning/outputs/captions/ and uses clip_path.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.env/bin/python"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_i2v.py"
CAPTIONS_ROOT="${CAPTIONS_ROOT:-${SCRIPT_DIR}/../scene_captioning/outputs/captions}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"

MODEL_ID="${MODEL_ID:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MOTION_SCORE="${MOTION_SCORE:-30}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAVE_REFERENCE="${SAVE_REFERENCE:-1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python venv not found: ${PYTHON}" >&2
  echo "Create it with: python -m venv ${SCRIPT_DIR}/../.env && pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
  exit 1
fi

if [[ ! -d "${CAPTIONS_ROOT}" ]]; then
  echo "Captions directory not found: ${CAPTIONS_ROOT}" >&2
  exit 1
fi

caption_count="$(find "${CAPTIONS_ROOT}" -mindepth 2 -maxdepth 2 -name 'captions.json' | wc -l | tr -d ' ')"
if [[ "${caption_count}" -eq 0 ]]; then
  echo "No captions.json files found under ${CAPTIONS_ROOT}" >&2
  exit 1
fi

extra_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  extra_args+=(--skip-existing)
fi
if [[ "${SAVE_REFERENCE}" == "1" ]]; then
  extra_args+=(--save-reference)
fi

if [[ $# -eq 0 ]]; then
  echo "Found ${caption_count} captioned video(s)"
  echo "Captions root: ${CAPTIONS_ROOT}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Model: ${MODEL_ID}"
  echo "Using scene clip first frames as reference images..."
  echo

  exec "${PYTHON}" "${GENERATE_SCRIPT}" \
    --captions-root "${CAPTIONS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --motion-score "${MOTION_SCORE}" \
    "${extra_args[@]}"
fi

exec "${PYTHON}" "${GENERATE_SCRIPT}" "$@"
