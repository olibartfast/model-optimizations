#!/bin/bash
# Bootstrap a Python 3.12 virtual environment for this repo.
#
# This script creates or refreshes your chosen Python 3.12 virtual environment
# from configs/requirements.txt, using the NGC index required for
# nvidia-modelopt[torch].
#
# Usage:
#   export QUANTIZATION_VENV="<venv-dir>"
#   ./scripts/run_venv.sh
#
# After completion, invoke Python through the venv directly:
#   "$QUANTIZATION_VENV/bin/python" <script>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

: "${QUANTIZATION_VENV:?Set QUANTIZATION_VENV to your personal venv directory, e.g. export QUANTIZATION_VENV=\"<venv-dir>\"}"
VENV_DIR="$QUANTIZATION_VENV"
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
