#!/bin/bash
# End-to-end FP32-vs-INT8(QAT) TensorRT inference benchmark for a YOLO model.
#
# Pipeline:
#   1. Locate FP32 ONNX (export from <stem>.pt via Ultralytics if missing).
#   2. Locate QAT ONNX from runs/modelopt_qat/<stem>/<stem>_qat.onnx.
#   3. Build a TensorRT engine for each (FP32 with --noTF32 to force FP32 math
#      so the comparison isn't muddied by Ampere TF32 fallback, INT8 with
#      --fp16 fallback for non-quantized layers).
#   4. Run trtexec --iterations latency on each.
#   5. Print a comparison table (latency mean/median, throughput, speedup).
#
# Usage:
#   ./scripts/measure_speedup.sh yolo11s
#   ./scripts/measure_speedup.sh yolo26s --iterations 500 --warmup 2000
#
# Env / flags:
#   TRTEXEC=<path>          force a specific trtexec binary
#   IMGSZ=640 BATCH=1       (defaults)
#   --iterations N          trtexec --iterations
#   --warmup N              trtexec --warmUp (ms)
#   --keep-engines          don't delete the .plan files after measurement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

STEM="${1:-}"
if [ -z "$STEM" ] || [[ "$STEM" == -* ]]; then
    echo "Usage: $0 <model_stem> [--iterations N] [--warmup N] [--keep-engines]" >&2
    exit 2
fi
shift

IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-1}"
ITERATIONS=200
WARMUP=1000
KEEP_ENGINES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --keep-engines) KEEP_ENGINES=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- Resolve trtexec ------------------------------------------------------
TRTEXEC="${TRTEXEC:-}"
if [ -z "$TRTEXEC" ]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="trtexec"
    else
        for cand in \
            /home/oli/dependencies/TensorRT-*/targets/x86_64-linux-gnu/bin/trtexec \
            /opt/nvidia/tensorrt*/bin/trtexec \
            /usr/local/cuda*/bin/trtexec
        do
            if [ -x "$cand" ]; then
                TRTEXEC="$cand"
                # libnvinfer*.so live in ../lib relative to the trtexec bin.
                bin_dir="$(dirname "$cand")"
                lib_dir="$(dirname "$bin_dir")/lib"
                if [ -d "$lib_dir" ]; then
                    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
                fi
                break
            fi
        done
    fi
fi
if [ -z "$TRTEXEC" ] || ! "$TRTEXEC" --help >/dev/null 2>&1; then
    if [ -z "$TRTEXEC" ]; then
        echo "ERROR: trtexec not found. Install TensorRT or set TRTEXEC=<path>." >&2
    else
        echo "ERROR: trtexec found at $TRTEXEC but fails to run (likely missing libnvinfer plugins)." >&2
        echo "       Set LD_LIBRARY_PATH to the TensorRT lib directory before re-running." >&2
    fi
    exit 2
fi
echo "trtexec: $TRTEXEC"
echo "ld path: $LD_LIBRARY_PATH"

# ---- Resolve ONNX paths ---------------------------------------------------
OUT_DIR="runs/modelopt_qat/$STEM"
QAT_ONNX="$OUT_DIR/${STEM}_qat.onnx"
FP32_ONNX="$OUT_DIR/${STEM}_fp32.onnx"
PT_FILE="${STEM}.pt"

if [ ! -f "$QAT_ONNX" ]; then
    echo "ERROR: QAT ONNX not found: $QAT_ONNX" >&2
    echo "       Run the QAT pipeline first (without --no-export)." >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
if [ ! -f "$FP32_ONNX" ]; then
    if [ ! -f "$PT_FILE" ]; then
        echo "ERROR: missing both $FP32_ONNX and $PT_FILE — cannot create FP32 baseline." >&2
        exit 2
    fi
    echo "Exporting FP32 ONNX from $PT_FILE ..."
    quantization_venv/bin/python - <<PY
from pathlib import Path
from ultralytics import YOLO

model = YOLO("$PT_FILE")
exported = model.export(format="onnx", imgsz=$IMGSZ, dynamic=False, simplify=False, opset=17, int8=False, half=False)
Path(exported).replace("$FP32_ONNX")
print(f"  wrote $FP32_ONNX")
PY
fi

# ---- Bench one ONNX -------------------------------------------------------
bench_one() {
    local label="$1"
    local onnx_path="$2"
    local extra_flags="$3"
    local engine="$OUT_DIR/${STEM}_${label}.plan"
    local logfile="$engine.log"
    local shape="images:${BATCH}x3x${IMGSZ}x${IMGSZ}"

    echo
    echo "=== [$label] $onnx_path ==="
    echo "    engine : $engine"

    # Static-shape ONNX models (which Ultralytics' export=onnx dynamic=False
    # produces) reject --shapes. Try with --shapes first; if that fails,
    # rerun without.
    local cmd=(
        "$TRTEXEC"
        --onnx="$onnx_path"
        --saveEngine="$engine"
        --shapes="$shape"
        --iterations="$ITERATIONS"
        --warmUp="$WARMUP"
        --noDataTransfers
        --useSpinWait
        $extra_flags
    )
    if ! "${cmd[@]}" > "$logfile" 2>&1; then
        if grep -qE "Static model does not take explicit shapes|Cannot find input tensor with name" "$logfile"; then
            echo "    (retrying without --shapes: ONNX has static shapes or different input name)"
            cmd=(
                "$TRTEXEC"
                --onnx="$onnx_path"
                --saveEngine="$engine"
                --iterations="$ITERATIONS"
                --warmUp="$WARMUP"
                --noDataTransfers
                --useSpinWait
                $extra_flags
            )
            "${cmd[@]}" > "$logfile" 2>&1 || {
                echo "  trtexec FAILED (log: $logfile)" >&2
                tail -10 "$logfile" >&2
                return 1
            }
        else
            echo "  trtexec FAILED (log: $logfile)" >&2
            tail -10 "$logfile" >&2
            return 1
        fi
    fi

    # Parse latency / throughput from trtexec log. trtexec emits multiple
    # mean/median lines (Latency, Enqueue, H2D, GPU Compute, D2H); pin to
    # the "Latency:" line so we get end-to-end latency, not a per-stage zero.
    local mean_ms median_ms qps
    mean_ms=$(grep -E '\[I\] Latency:' "$logfile" | grep -oE 'mean = [0-9.]+' | head -1 | awk '{print $3}')
    median_ms=$(grep -E '\[I\] Latency:' "$logfile" | grep -oE 'median = [0-9.]+' | head -1 | awk '{print $3}')
    qps=$(grep -oE 'Throughput: [0-9.]+ qps' "$logfile" | head -1 | awk '{print $2}')
    printf "  mean=%s ms  median=%s ms  throughput=%s qps\n" "$mean_ms" "$median_ms" "$qps"
    echo "$label,$mean_ms,$median_ms,$qps" >> "$OUT_DIR/speedup.csv"
}

# Reset CSV
echo "engine,mean_ms,median_ms,throughput_qps" > "$OUT_DIR/speedup.csv"

# Build + bench. FP32 baseline forced via --noTF32 so any speedup we report
# is purely INT8 vs strict-FP32, not INT8 vs Ampere TF32.
bench_one "fp32"     "$FP32_ONNX" "--noTF32"
bench_one "qat_int8" "$QAT_ONNX"  "--int8 --fp16"

# ---- Speedup summary ------------------------------------------------------
echo
echo "=== Speedup summary ($STEM @ imgsz=$IMGSZ batch=$BATCH) ==="
quantization_venv/bin/python - <<PY
import csv
rows = list(csv.DictReader(open("$OUT_DIR/speedup.csv")))
data = {r["engine"]: r for r in rows}
if "fp32" in data and "qat_int8" in data:
    fp = data["fp32"]; q = data["qat_int8"]
    def f(x):
        try: return float(x)
        except: return float("nan")
    mean_fp, mean_q = f(fp["mean_ms"]), f(q["mean_ms"])
    qps_fp, qps_q = f(fp["throughput_qps"]), f(q["throughput_qps"])
    print(f"  {'engine':<10} {'mean_ms':>10} {'qps':>10}")
    print(f"  {'fp32':<10} {mean_fp:>10.3f} {qps_fp:>10.1f}")
    print(f"  {'qat_int8':<10} {mean_q:>10.3f} {qps_q:>10.1f}")
    if mean_fp and mean_q:
        print(f"  speedup (mean latency): {mean_fp / mean_q:.2f}x")
    if qps_fp and qps_q:
        print(f"  speedup (throughput):   {qps_q / qps_fp:.2f}x")
PY

if [ "$KEEP_ENGINES" -ne 1 ]; then
    rm -f "$OUT_DIR/${STEM}_fp32.plan" "$OUT_DIR/${STEM}_qat_int8.plan"
fi
echo
echo "CSV: $OUT_DIR/speedup.csv"
