#!/bin/bash
# Validate the *exported* INT8 graphs (ONNX, optionally TRT engine) by running
# them through Ultralytics' COCO val pipeline and comparing mAP to the
# PT-checkpoint baseline.
#
# This is what catches export bugs that Q/DQ-node counts cannot: a model with
# the right number of Q/DQ pairs can still produce garbage if the calibrated
# scales were dropped, if some op fell back to a wrong precision, or if the
# I/O bindings were mis-shaped during export.
#
# Usage:
#   ./scripts/measure_accuracy.sh yolo11s
#   ./scripts/measure_accuracy.sh yolo26s [--engine]
#
# Env:
#   PY=<python>             force a Python executable
#   QUANTIZATION_VENV=<venv-dir>
#                           use <venv-dir>/bin/python when PY is unset
#
# Without --engine: validates yolo*_fp32.onnx and yolo*_qat.onnx via
# onnxruntime (CPU/GPU fallback). Validates the *symbolic* INT8 graph: Q/DQ
# nodes are executed as fake-quant (FP arithmetic with quant/dequant noise),
# matching what the QAT pipeline produces.
#
# With --engine: also validates the pre-built TRT engines if they exist in
# runs/modelopt_qat/<stem>/ (run measure_speedup.sh --keep-engines first).
# Uses Ultralytics' built-in YOLO(engine_path) loader.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

STEM="${1:-}"
INCLUDE_ENGINE=0
if [ -z "$STEM" ] || [[ "$STEM" == -* ]]; then
    echo "Usage: $0 <model_stem> [--engine]" >&2
    exit 2
fi
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --engine) INCLUDE_ENGINE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

OUT_DIR="runs/modelopt_qat/$STEM"
FP32_ONNX="$OUT_DIR/${STEM}_fp32.onnx"
QAT_ONNX="$OUT_DIR/${STEM}_qat.onnx"
PY="${PY:-}"
if [ -z "$PY" ] && [ -n "${QUANTIZATION_VENV:-}" ]; then
    PY="$QUANTIZATION_VENV/bin/python"
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: Python executable not found. Set PY=<python> or QUANTIZATION_VENV=<venv-dir>." >&2
    exit 2
fi

if [ ! -f "$FP32_ONNX" ] || [ ! -f "$QAT_ONNX" ]; then
    echo "ERROR: expected $FP32_ONNX and $QAT_ONNX" >&2
    echo "       Run ./scripts/measure_speedup.sh $STEM first to produce both ONNX files." >&2
    exit 2
fi

# Make TensorRT libs visible if --engine is requested.
if [ "$INCLUDE_ENGINE" -eq 1 ]; then
    for cand in /home/oli/dependencies/TensorRT-*/targets/x86_64-linux-gnu/bin/trtexec; do
        if [ -x "$cand" ]; then
            lib_dir="$(dirname "$(dirname "$cand")")/lib"
            [ -d "$lib_dir" ] && export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

PT_FILE="${STEM}.pt"
ENGINE_FP32="$OUT_DIR/${STEM}_fp32.plan"
ENGINE_INT8="$OUT_DIR/${STEM}_qat_int8.plan"

"$PY" - <<PY
import json
from pathlib import Path
import sys

from ultralytics import YOLO

STEM = "$STEM"
PT_FILE = "$PT_FILE"
FP32_ONNX = "$FP32_ONNX"
QAT_ONNX = "$QAT_ONNX"
ENGINE_FP32 = "$ENGINE_FP32"
ENGINE_INT8 = "$ENGINE_INT8"
INCLUDE_ENGINE = bool(int("$INCLUDE_ENGINE"))
COCO_YAML = "configs/coco.yaml"

def val(label, path):
    if not Path(path).exists():
        print(f"  SKIP {label}: {path} not found")
        return None
    print(f"\n=== {label}: {path} ===")
    try:
        model = YOLO(path)
        r = model.val(data=COCO_YAML, imgsz=640, batch=1, conf=0.001, iou=0.6,
                      device=0, plots=False, verbose=False, save_json=False)
        m50 = float(r.box.map50)
        m = float(r.box.map)
        print(f"  mAP50={m50:.4f}  mAP50-95={m:.4f}")
        return {"label": label, "path": path, "map50": m50, "map": m}
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {"label": label, "path": path, "error": repr(exc)}

results = []
if Path(PT_FILE).exists():
    results.append(val("pt_fp32",       PT_FILE))
results.append(    val("onnx_fp32",     FP32_ONNX))
results.append(    val("onnx_qat_int8", QAT_ONNX))
if INCLUDE_ENGINE:
    results.append(val("engine_fp32",       ENGINE_FP32))
    results.append(val("engine_qat_int8",   ENGINE_INT8))

print()
print("=== Accuracy comparison ===")
print(f"  {'label':<20} {'mAP50':>8} {'mAP50-95':>10}  delta_vs_pt")
pt_map = next((r["map"] for r in results if r and r.get("label")=="pt_fp32" and "map" in r), None)
for r in results:
    if r is None or "map" not in r:
        continue
    delta = ""
    if pt_map is not None and r["label"] != "pt_fp32":
        delta = f"  {r['map'] - pt_map:+.4f}"
    print(f"  {r['label']:<20} {r['map50']:>8.4f} {r['map']:>10.4f}{delta}")

out_path = Path("$OUT_DIR") / "accuracy.json"
out_path.write_text(json.dumps([r for r in results if r], indent=2))
print(f"\nWrote {out_path}")
PY
