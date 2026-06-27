#!/usr/bin/env bash
# Train LTX-2.3 audio-video LoRA (compatible with TI2VidTwoStagesPipeline at inference).
set -euo pipefail

# NCCL settings for multi-GPU stability (must be set before PyTorch/accelerate start).
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_TREE_THRESHOLD="${NCCL_TREE_THRESHOLD:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

LTX_TRAINER="${LTX_TRAINER:-${ID_LORA_ROOT}/ID-LoRA-2.3/packages/ltx-trainer}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/ltx23_av_lora.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${LTX_TRAINER}/configs/accelerate/ddp.yaml}"

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "LTX trainer env missing. Run: ${SCRIPT_DIR}/setup_trainer.sh" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config not found: ${CONFIG}" >&2
  exit 1
fi

PRECOMPUTED="${PRECOMPUTED_ROOT:-${SCRIPT_DIR}/data/.precomputed}"
if [[ ! -d "${PRECOMPUTED}/latents" || ! -d "${PRECOMPUTED}/audio_latents" ]]; then
  echo "Preprocessed data missing. Run: ${SCRIPT_DIR}/run_preprocess.sh" >&2
  exit 1
fi

cd "${LTX_TRAINER}"

echo "Training LTX-2.3 audio-video LoRA"
echo "  config:      ${CONFIG}"
echo "  data:        ${PRECOMPUTED}"
echo "  processes:   ${NUM_PROCESSES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  "${LTX_ACCELERATE}" launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    scripts/train.py "${CONFIG}"
else
  "${LTX_PYTHON}" scripts/train.py "${CONFIG}"
fi

echo
echo "Training finished. LoRA weights should be under the output_dir in your config."
