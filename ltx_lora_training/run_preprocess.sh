#!/usr/bin/env bash
set -euo pipefail

if [[ "${NUM_PROCESSES:-1}" -gt 1 ]]; then
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_TREE_THRESHOLD="${NCCL_TREE_THRESHOLD:-0}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
  export NCCL_ALGO="${NCCL_ALGO:-Ring}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "LTX trainer env missing. Run: ${SCRIPT_DIR}/setup_trainer.sh" >&2
  exit 1
fi

if [[ ! -f "${LTX_MODEL_PATH}" ]]; then
  echo "LTX model missing. Run: ${SCRIPT_DIR}/download_models.sh" >&2
  exit 1
fi

if [[ ! -f "${GEMMA_MODEL_DIR}/config.json" ]]; then
  echo "Gemma text encoder missing. Run: ${SCRIPT_DIR}/download_models.sh" >&2
  exit 1
fi

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "dataset.json missing: ${DATASET_JSON}" >&2
  echo "Run captioning first, e.g.: bash run_caption.sh" >&2
  exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-1}"
WITH_AUDIO="${WITH_AUDIO:-1}"
DECODE="${DECODE:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOAD_TEXT_ENCODER_8BIT="${LOAD_TEXT_ENCODER_8BIT:-1}"

extra_args=()
if [[ "${WITH_AUDIO}" == "1" ]]; then
  extra_args+=(--with-audio)
fi
if [[ "${LOAD_TEXT_ENCODER_8BIT}" == "1" ]]; then
  extra_args+=(--load-text-encoder-in-8bit)
fi
if [[ "${DECODE}" == "1" ]]; then
  extra_args+=(--decode)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ -n "${LORA_TRIGGER}" ]]; then
  extra_args+=(--lora-trigger "${LORA_TRIGGER}")
fi

cd "${LTX_TRAINER}"

echo "Preprocessing dataset for LTX-2.3 LoRA training"
echo "  dataset:     ${DATASET_JSON}"
echo "  output:      ${PRECOMPUTED_ROOT}"
echo "  model:       ${LTX_MODEL_PATH}"
echo "  text encoder:${GEMMA_MODEL_DIR}"
echo "  buckets:     ${RESOLUTION_BUCKETS}"
echo "  GPUs:        ${NUM_PROCESSES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  "${LTX_ACCELERATE}" launch --num_processes "${NUM_PROCESSES}" scripts/process_dataset.py \
    "${DATASET_JSON}" \
    --resolution-buckets "${RESOLUTION_BUCKETS}" \
    --model-path "${LTX_MODEL_PATH}" \
    --text-encoder-path "${GEMMA_MODEL_DIR}" \
    --output-dir "${PRECOMPUTED_ROOT}" \
    "${extra_args[@]}"
else
  "${LTX_PYTHON}" scripts/process_dataset.py \
    "${DATASET_JSON}" \
    --resolution-buckets "${RESOLUTION_BUCKETS}" \
    --model-path "${LTX_MODEL_PATH}" \
    --text-encoder-path "${GEMMA_MODEL_DIR}" \
    --output-dir "${PRECOMPUTED_ROOT}" \
    "${extra_args[@]}"
fi

echo
echo "Preprocessing complete: ${PRECOMPUTED_ROOT}"
