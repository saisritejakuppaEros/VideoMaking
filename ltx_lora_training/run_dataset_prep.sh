#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1/4: Install LTX trainer (uv sync) ==="
bash "${SCRIPT_DIR}/setup_trainer.sh"

echo
echo "=== Step 2/4: Download LTX-2.3 + Gemma models locally ==="
bash "${SCRIPT_DIR}/download_models.sh"

echo
echo "=== Step 3/4: Build dataset.json from scene captions ==="
bash "${SCRIPT_DIR}/run_build_dataset.sh"

echo
echo "=== Step 4/4: Preprocess latents + text embeddings ==="
bash "${SCRIPT_DIR}/run_preprocess.sh"

echo
echo "Dataset preparation finished."
