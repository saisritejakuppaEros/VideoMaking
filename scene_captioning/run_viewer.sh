#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PIP="/mnt/data0/harsha/new_paper/VideoMaking/.env/bin/pip"
VENV_STREAMLIT="/mnt/data0/harsha/new_paper/VideoMaking/.env/bin/streamlit"

if [[ -x "$VENV_PIP" ]]; then
  "$VENV_PIP" install -q -r requirements-viewer.txt
  STREAMLIT_BIN="$VENV_STREAMLIT"
else
  pip install -q -r requirements-viewer.txt
  STREAMLIT_BIN="streamlit"
fi

exec "$STREAMLIT_BIN" run app.py --server.address 0.0.0.0 --server.port 8501
