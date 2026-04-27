#!/bin/bash
# Run NVIDIA ModelOpt on Ultralytics YOLO over COCO val/test.
#
# Default mode is QAT (quantization-aware training), matching NVIDIA's
# examples/cnn_qat workflow: PTQ calibration -> mtq.quantize -> QAT
# fine-tune -> mto.save -> ONNX export.
#
# Usage:
#   ./scripts/run_modelopt_yolo.sh                             # QAT, yolo11x + yolo26x
#   ./scripts/run_modelopt_yolo.sh --qat-epochs 3 --batch 32
#   ./scripts/run_modelopt_yolo.sh ptq --quant-modes int8 fp8  # ONNX-only PTQ variant
#
# Any remaining args are forwarded to the underlying Python script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

MODE="qat"
if [ "${1:-}" = "qat" ] || [ "${1:-}" = "ptq" ]; then
    MODE="$1"
    shift
fi

if [ -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

if [ ! -d "datasets/coco/val2017" ]; then
    echo "⚠️  COCO val2017 not found. Downloading now (requires ~25GB)..."
    ./scripts/download_coco_dataset.sh
fi

case "$MODE" in
    qat) exec python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py "$@" ;;
    ptq) exec python yolo_quantization/ptq/nvidia_modelopt_yolo.py "$@" ;;
    *)   echo "Unknown mode: $MODE (expected qat|ptq)"; exit 2 ;;
esac
