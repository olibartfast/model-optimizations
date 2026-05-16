#!/bin/bash
# Run NVIDIA ModelOpt on Ultralytics YOLO over COCO val2017.
#
# Default mode is QAT (FP32 eval -> PTQ calibrate -> QAT distill -> ONNX export)
# via the canonical script at yolo_quantization/qat/nvidia_modelopt_yolo_qat.py.
# Switch to PTQ-only ONNX with the `ptq` first arg.
#
# Usage:
#   ./scripts/run_modelopt_yolo.sh                              # QAT defaults
#   ./scripts/run_modelopt_yolo.sh --models yolo26s
#   ./scripts/run_modelopt_yolo.sh ptq --quant-modes int8 fp8   # ONNX-only PTQ
#
# Any remaining args are forwarded to the underlying Python script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODE="qat"
if [ "${1:-}" = "qat" ] || [ "${1:-}" = "ptq" ]; then
    MODE="$1"
    shift
fi

# Prefer the pre-built quantization_venv (ModelOpt is installed there).
PY="quantization_venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Bootstrap a venv with scripts/run_venv.sh or" \
         "ensure quantization_venv/ exists at the repo root." >&2
    exit 2
fi

if [ ! -d "datasets/coco/images/val2017" ] && [ ! -d "datasets/coco/val2017" ]; then
    echo "COCO val2017 not found. Downloading raw zips (requires ~25GB)..."
    ./scripts/download_coco_dataset.sh
fi

case "$MODE" in
    qat) exec "$PY" yolo_quantization/qat/nvidia_modelopt_yolo_qat.py "$@" ;;
    ptq) exec "$PY" yolo_quantization/ptq/nvidia_modelopt_yolo.py "$@" ;;
    *)   echo "Unknown mode: $MODE (expected qat|ptq)" >&2; exit 2 ;;
esac
