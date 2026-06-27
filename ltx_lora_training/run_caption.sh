#!/usr/bin/env bash
# Optional: caption scene clips with LTX trainer's Qwen2.5-Omni captioner (video + audio).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "LTX trainer env missing. Run: ${SCRIPT_DIR}/setup_trainer.sh" >&2
  exit 1
fi

SCENES_INPUT="${SCENES_INPUT:-${SCENES_ROOT}}"
CAPTION_OUTPUT="${CAPTION_OUTPUT:-${SCENE_CAPTION_OUTPUT}}"
USE_8BIT="${USE_8BIT:-1}"
CAPTION_FPS="${CAPTION_FPS:-1}"
RECURSIVE="${RECURSIVE:-0}"

mkdir -p "${CAPTION_OUTPUT}"
cd "${LTX_TRAINER}"

extra_args=(--fps "${CAPTION_FPS}")
if [[ "${USE_8BIT}" == "1" ]]; then
  extra_args+=(--use-8bit)
fi
if [[ "${RECURSIVE}" == "1" ]]; then
  extra_args+=(--recursive)
fi

echo "Captioning clips under ${SCENES_INPUT}"
echo "Output root: ${CAPTION_OUTPUT}/<video>/captions.json"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

mapfile -t VIDEO_DIRS < <(find "${SCENES_INPUT}" -mindepth 1 -maxdepth 1 -type d | sort)
for video_dir in "${VIDEO_DIRS[@]}"; do
  folder_name="$(basename "${video_dir}")"
  final_output="${CAPTION_OUTPUT}/${folder_name}/captions.json"
  tmp_output="${video_dir}/.ltx_caption.tmp.json"

  echo "Captioning ${folder_name}"
  rm -f "${tmp_output}"
  uv run python scripts/caption_videos.py "${video_dir}" \
    --output "${tmp_output}" \
    "${extra_args[@]}"

  "${VENV_PYTHON}" "${SCRIPT_DIR}/caption_progress.py" finalize \
    --tmp-path "${tmp_output}" \
    --scenes-root "${SCENES_INPUT}" \
    --video-dir "${video_dir}" \
    --output-path "${final_output}"
done

echo "Captions written under ${CAPTION_OUTPUT}"
