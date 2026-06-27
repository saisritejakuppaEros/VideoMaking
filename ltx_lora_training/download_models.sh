#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

mkdir -p "${MODELS_ROOT}" "${LTX_MODEL_DIR}" "${GEMMA_MODEL_DIR}" "${HF_HOME}"

if [[ ! -x "${VENV_HF}" ]]; then
  echo "hf CLI not found in project venv: ${VENV_HF}" >&2
  echo "Install with: ${VENV_PYTHON} -m pip install 'huggingface_hub[cli]>=0.25.0'" >&2
  exit 1
fi

# Gemma is a gated HF repo. Prefer an authenticated download; otherwise copy a
# complete local snapshot if one exists on this machine.
LOCAL_GEMMA_SOURCE="${LOCAL_GEMMA_SOURCE:-/mnt/data0/parth/hf_models_cache/google--gemma-3-12b-it-qat-q4_0-unquantized}"

gemma_is_complete() {
  [[ -f "${GEMMA_MODEL_DIR}/config.json" && -f "${GEMMA_MODEL_DIR}/model.safetensors.index.json" ]]
}

copy_local_gemma() {
  local source_dir="$1"
  if [[ ! -f "${source_dir}/config.json" ]]; then
    echo "Local Gemma source is incomplete: ${source_dir}" >&2
    return 1
  fi

  echo "Copying Gemma text encoder from local snapshot:"
  echo "  source: ${source_dir}"
  echo "  target: ${GEMMA_MODEL_DIR}"
  rsync -a --info=progress2 \
    --exclude '.cache/' \
    --exclude '.git/' \
    "${source_dir}/" "${GEMMA_MODEL_DIR}/"
}

echo "Downloading models into ${MODELS_ROOT}"
echo "HF cache: ${HF_HOME}"
echo

if [[ -f "${LTX_MODEL_PATH}" ]]; then
  echo "LTX-2.3 dev checkpoint already present: ${LTX_MODEL_PATH}"
else
  echo "Downloading Lightricks/LTX-2.3 -> ${LTX_MODEL_DIR}"
  "${VENV_HF}" download Lightricks/LTX-2.3 \
    ltx-2.3-22b-dev.safetensors \
    --local-dir "${LTX_MODEL_DIR}"
fi

if gemma_is_complete; then
  echo "Gemma text encoder already present: ${GEMMA_MODEL_DIR}"
else
  rm -rf "${GEMMA_MODEL_DIR:?}/"*
  mkdir -p "${GEMMA_MODEL_DIR}"

  if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "Downloading google/gemma-3-12b-it-qat-q4_0-unquantized with HF token -> ${GEMMA_MODEL_DIR}"
    if "${VENV_HF}" download google/gemma-3-12b-it-qat-q4_0-unquantized \
      --local-dir "${GEMMA_MODEL_DIR}"; then
      :
    elif [[ -d "${LOCAL_GEMMA_SOURCE}" ]]; then
      echo "HF download failed; falling back to local Gemma snapshot." >&2
      copy_local_gemma "${LOCAL_GEMMA_SOURCE}"
    else
      echo "Gemma download failed and no local fallback found at ${LOCAL_GEMMA_SOURCE}." >&2
      echo "Accept the model license on Hugging Face, then either:" >&2
      echo "  export HF_TOKEN=hf_... && bash download_models.sh" >&2
      echo "  LOCAL_GEMMA_SOURCE=/path/to/gemma bash download_models.sh" >&2
      exit 1
    fi
  elif [[ -d "${LOCAL_GEMMA_SOURCE}" && -f "${LOCAL_GEMMA_SOURCE}/config.json" ]]; then
    copy_local_gemma "${LOCAL_GEMMA_SOURCE}"
  else
    echo "Gemma is gated on Hugging Face and no local snapshot was found." >&2
    echo "Either set HF_TOKEN after accepting the license, or set LOCAL_GEMMA_SOURCE." >&2
    exit 1
  fi
fi

if ! gemma_is_complete; then
  echo "Gemma install looks incomplete under ${GEMMA_MODEL_DIR}" >&2
  exit 1
fi

echo
echo "Model download complete."
echo "  LTX model:    ${LTX_MODEL_PATH}"
echo "  Text encoder: ${GEMMA_MODEL_DIR}"
