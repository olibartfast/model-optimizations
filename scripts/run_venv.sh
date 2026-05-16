#!/bin/bash
# Bootstrap a Python 3.12 virtual environment for this repo.
#
# The repo's working venv is `quantization_venv/` (already installed with
# nvidia-modelopt[torch] from the NGC index). This script recreates it from
# scratch in the same directory, driven by configs/requirements.txt.
#
# Usage:
#   ./scripts/run_venv.sh
#
# After completion, invoke Python through the venv directly:
#   quantization_venv/bin/python <script>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV_DIR="quantization_venv"
REQS="configs/requirements.txt"

if [ ! -f "$REQS" ]; then
    echo "ERROR: $REQS not found." >&2
    exit 2
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 virtual environment at $VENV_DIR ..."
    python3.12 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "Upgrading pip + wheel..."
"$PIP" install --upgrade pip wheel

# nvidia-modelopt requires --no-build-isolation and the NGC extra index; the
# [torch] extra pulls in modelopt.torch.* used by the QAT pipeline.
echo "Installing dependencies from $REQS ..."
"$PIP" install --no-build-isolation \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -r "$REQS"

echo
echo "Done. Verify the install with:"
echo "  $PY -c 'import torch, onnx, pulp, huggingface_hub, modelopt.torch.quantization, modelopt.torch.export; print(\"ok\")'"
echo
echo "Run scripts with: $PY <path/to/script.py>"
