#!/bin/bash
# Run NVIDIA ModelOpt PTQ on Ultralytics YOLO models over COCO val/test.
#
# Usage:
#   ./scripts/run_modelopt_yolo.sh                    # default: yolo11x, yolo26x, int8
#   ./scripts/run_modelopt_yolo.sh --quant-modes int8 fp8
#
# Any args are forwarded to examples/nvidia_modelopt_yolo.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if present (matches scripts/run_venv.sh convention)
if [ -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Make sure COCO is downloaded
if [ ! -d "datasets/coco/val2017" ]; then
    echo "⚠️  COCO val2017 not found. Downloading now (requires ~25GB)..."
    ./scripts/download_coco_dataset.sh
fi

exec python examples/nvidia_modelopt_yolo.py "$@"
