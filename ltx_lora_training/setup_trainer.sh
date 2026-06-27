#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "Installing LTX trainer dependencies with uv in ${LTX_REPO}"
cd "${LTX_REPO}"
uv sync

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "Expected trainer python at ${LTX_PYTHON} after uv sync" >&2
  exit 1
fi

echo
echo "Trainer environment ready:"
"${LTX_PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
import ltx_trainer
print("ltx_trainer", ltx_trainer.__file__)
PY
