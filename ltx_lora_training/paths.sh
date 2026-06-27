#!/usr/bin/env bash
# Shared paths for LTX-2.3 LoRA dataset prep and training.
# Source this file from other scripts: source "$(dirname "$0")/paths.sh"

PROJECT_ROOT="/mnt/data0/harsha/new_paper/VideoMaking"
ID_LORA_ROOT="${PROJECT_ROOT}/video_generation_ltx/ID-LoRA"
LTX_REPO="${ID_LORA_ROOT}"
LTX_TRAINER="${ID_LORA_ROOT}/ID-LoRA-2.3/packages/ltx-trainer"

# Project venv (VideoMaking/.env) — used for Qwen captioning only
VENV_PYTHON="${PROJECT_ROOT}/.env/bin/python"
VENV_HF="${PROJECT_ROOT}/.env/bin/hf"

# ID-LoRA venv (created by `uv sync` in ID_LORA_ROOT)
LTX_PYTHON="${LTX_REPO}/.venv/bin/python"
LTX_ACCELERATE="${LTX_REPO}/.venv/bin/accelerate"

export HF_HOME="${ID_LORA_ROOT}/models/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"

MODELS_ROOT="${ID_LORA_ROOT}/models"
LTX_MODEL_DIR="${MODELS_ROOT}"
LTX_MODEL_PATH="${MODELS_ROOT}/ltx-2.3-22b-dev.safetensors"
GEMMA_MODEL_DIR="${MODELS_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized"

# Raw YouTube downloads
RAW_VIDS_ROOT="${PROJECT_ROOT}/debunk_exisiting_youtubers/outputs/vids"

# Reuse scene clips + Qwen captions from scene_captioning
SCENE_CAPTIONING_ROOT="${PROJECT_ROOT}/scene_captioning"
SCENES_ROOT="${SCENE_CAPTIONING_ROOT}/outputs/scenes"
CAPTIONS_ROOT="${SCENE_CAPTIONING_ROOT}/outputs/captions"
SCENE_CAPTION_OUTPUT="${SCENE_CAPTIONING_ROOT}/outputs/scene_caption_op"
CAPTION_SHARDS_DIR="${SCENE_CAPTION_OUTPUT}/caption_shards"

# Merged dataset.json for LTX preprocess (built from per-folder captions when needed)
DATA_ROOT="${PROJECT_ROOT}/ltx_lora_training/data"
SCENES_OUT="${DATA_ROOT}/scenes"
# Per-folder LTX captions land here; merged dataset.json for preprocess uses SCENES_ROOT
DATASET_JSON="${DATASET_JSON:-${SCENES_ROOT}/dataset.json}"
PRECOMPUTED_ROOT="${PRECOMPUTED_ROOT:-${SCENES_ROOT}/.precomputed}"

# Training resolution bucket: width x height x frames (frames % 8 == 1)
RESOLUTION_BUCKETS="${RESOLUTION_BUCKETS:-832x512x49}"

# Default GPU selection for multi-GPU caption/preprocess/train jobs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export NUM_PROCESSES="${NUM_PROCESSES:-7}"

# Optional LoRA trigger token prepended during preprocessing
LORA_TRIGGER="${LORA_TRIGGER:-}"

# ID-LoRA venv ships libcudart.so.12 but torchaudio expects libcudart.so.13
CUDA_COMPAT_LIB="${ID_LORA_ROOT}/.venv/lib/cuda_compat"
CUDA_RUNTIME_LIB="${ID_LORA_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${CUDA_COMPAT_LIB}:${CUDA_RUNTIME_LIB}:${LD_LIBRARY_PATH:-}"
