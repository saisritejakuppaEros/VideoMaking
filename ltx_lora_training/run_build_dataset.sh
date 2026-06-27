#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

MIN_DURATION="${MIN_DURATION_SECONDS:-5.0}"

echo "Building dataset.json from scene_captioning captions"
echo "  captions root: ${CAPTIONS_ROOT}"
echo "  output:        ${DATASET_JSON}"
echo "  min duration:  ${MIN_DURATION}s"
echo

"${VENV_PYTHON}" "${SCRIPT_DIR}/build_dataset_json.py" \
  --captions-root "${CAPTIONS_ROOT}" \
  --output "${DATASET_JSON}" \
  --min-duration-seconds "${MIN_DURATION}" \
  --symlink-clips \
  --scenes-out "${SCENES_OUT}"

echo
echo "Dataset manifest written to ${DATASET_JSON}"
