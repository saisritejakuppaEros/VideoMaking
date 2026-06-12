#!/usr/bin/env bash
# Generate videos from scene captions + first frame of each scene clip with Wan 2.1 I2V.
# Models are loaded from /mnt/data0/parth/hf_models_cache by default.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/data0/parth/ai_ed/ai_ed/bin/python}"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_i2v.py"
CAPTIONS_ROOT="${CAPTIONS_ROOT:-${SCRIPT_DIR}/../scene_captioning/outputs/captions}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"

MODEL_PATH="${MODEL_PATH:-/mnt/data0/parth/hf_models_cache/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers}"
HF_CACHE="${HF_CACHE:-/mnt/data0/parth/hf_models_cache}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAVE_REFERENCE="${SAVE_REFERENCE:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python venv not found: ${PYTHON}" >&2
  echo "Set PYTHON to a valid interpreter, e.g. /mnt/data0/parth/ai_ed/ai_ed/bin/python" >&2
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
if [[ "${CPU_OFFLOAD}" == "1" ]]; then
  extra_args+=(--cpu-offload)
fi

if [[ $# -eq 0 ]]; then
  echo "Found ${caption_count} captioned video(s)"
  echo "Captions root: ${CAPTIONS_ROOT}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Model path: ${MODEL_PATH}"
  echo "HF cache: ${HF_CACHE}"
  echo "Using scene clip first frames as reference images..."
  echo

  exec "${PYTHON}" "${GENERATE_SCRIPT}" \
    --captions-root "${CAPTIONS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --hf-cache "${HF_CACHE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${extra_args[@]}"
fi

exec "${PYTHON}" "${GENERATE_SCRIPT}" "$@"
