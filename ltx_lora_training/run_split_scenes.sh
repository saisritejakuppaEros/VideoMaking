#!/usr/bin/env bash
# Optional: split raw long-form YouTube videos into scenes using the official LTX trainer script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "LTX trainer env missing. Run: ${SCRIPT_DIR}/setup_trainer.sh" >&2
  exit 1
fi

FILTER_SHORTER_THAN="${FILTER_SHORTER_THAN:-5s}"
mkdir -p "${SCENES_OUT}"

mapfile -d '' videos < <(find "${RAW_VIDS_ROOT}" -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.mkv' \) -print0 | sort -z)
total=${#videos[@]}
if (( total == 0 )); then
  echo "No videos found under ${RAW_VIDS_ROOT}" >&2
  exit 1
fi

echo "Splitting ${total} raw video(s) into scenes"
cd "${LTX_TRAINER}"

for i in "${!videos[@]}"; do
  video="${videos[$i]}"
  n=$((i + 1))
  channel="$(basename "$(dirname "${video}")")"
  stem="$(basename "${video}")"
  stem="${stem%.*}"
  out_dir="${SCENES_OUT}/${channel}/${stem}"
  mkdir -p "${out_dir}"

  echo "[$n/${total}] ${video} -> ${out_dir}"
  uv run python scripts/split_scenes.py "${video}" "${out_dir}" \
    --filter-shorter-than "${FILTER_SHORTER_THAN}"
done

echo "Scene splitting complete under ${SCENES_OUT}"
