#!/usr/bin/env bash
# Split all raw YouTube videos under outputs/vids into scene clips.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

SCENE_CUT="${SCENE_CAPTIONING_ROOT}/scene_cut.py"
START="${SCENE_START:-}"
END="${SCENE_END:-}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python venv not found: ${VENV_PYTHON}" >&2
  exit 1
fi

mapfile -d '' videos < <(
  find "${RAW_VIDS_ROOT}" -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.mkv' \) -print0 | sort -z
)
total=${#videos[@]}
if (( total == 0 )); then
  echo "No videos found under ${RAW_VIDS_ROOT}" >&2
  exit 1
fi

echo "Scene-cutting ${total} video(s) from ${RAW_VIDS_ROOT}"
echo "Output: ${SCENES_ROOT}/<video_title>/"
if [[ -n "${START}" || -n "${END}" ]]; then
  echo "Time window: start=${START:-0} end=${END:-EOF}"
else
  echo "Time window: full video"
fi
echo

for i in "${!videos[@]}"; do
  video="${videos[$i]}"
  n=$((i + 1))
  echo "[$n/${total}] ${video}"

  args=("${video}")
  if [[ -n "${START}" ]]; then
    args+=(--start "${START}")
  fi
  if [[ -n "${END}" ]]; then
    args+=(--end "${END}")
  fi

  "${VENV_PYTHON}" "${SCENE_CUT}" "${args[@]}"
done

echo
echo "Done. Scene manifests are under ${SCENES_ROOT}"
