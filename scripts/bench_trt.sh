#!/bin/bash
# Benchmark an exported QAT ONNX (Q/DQ) via NVIDIA's trtexec.
#
# Builds a TensorRT engine in INT8 mode (with --fp16 fallback for non-quantized
# layers) and prints throughput / latency. Optional --fp32-onnx points at a
# floating-point baseline so the two are reported side-by-side.
#
# Usage:
#   ./scripts/bench_trt.sh runs/modelopt_qat/yolo26s/yolo26s_qat.onnx
#   ./scripts/bench_trt.sh --fp32-onnx runs/modelopt/yolo26s/yolo26s.onnx \
#                          runs/modelopt_qat/yolo26s/yolo26s_qat.onnx
#
# Optional environment overrides:
#   TRTEXEC      path to trtexec (default: trtexec on PATH)
#   IMGSZ        input H/W (default 640)
#   BATCH        explicit batch size (default 1)
#   ITERATIONS   trtexec --iterations (default 200)
#   WARMUP       trtexec --warmUp ms (default 1000)
#   ENGINE_DIR   directory for *.plan engines (default alongside the ONNX)

set -e

# Resolve trtexec: prefer $TRTEXEC -> $PATH -> the locally installed TensorRT release.
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
                lib_dir="$(dirname "$(dirname "$cand")")/lib"
                if [ -d "$lib_dir" ]; then
                    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
                fi
                break
            fi
        done
    fi
fi
if [ -z "$TRTEXEC" ]; then
    TRTEXEC="trtexec"  # will fail the next check with a clear error
fi
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-1}"
ITERATIONS="${ITERATIONS:-200}"
WARMUP="${WARMUP:-1000}"

FP32_ONNX=""
QAT_ONNX=""

while [ $# -gt 0 ]; do
    case "$1" in
        --fp32-onnx) FP32_ONNX="$2"; shift 2 ;;
        --imgsz)     IMGSZ="$2"; shift 2 ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --warmup)    WARMUP="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            if [ -z "$QAT_ONNX" ]; then
                QAT_ONNX="$1"
            else
                echo "Unexpected positional arg: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

if [ -z "$QAT_ONNX" ]; then
    echo "ERROR: positional argument <qat.onnx> is required." >&2
    grep '^# ' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
if ! command -v "$TRTEXEC" >/dev/null 2>&1; then
    echo "ERROR: $TRTEXEC not found on PATH. Install TensorRT or set TRTEXEC." >&2
    exit 2
fi

bench_one() {
    local label="$1"
    local onnx_path="$2"
    local extra_flags="$3"

    if [ ! -f "$onnx_path" ]; then
        echo "[$label] SKIP: $onnx_path does not exist" >&2
        return 0
    fi
    local engine_dir
    engine_dir="${ENGINE_DIR:-$(dirname "$onnx_path")}"
    mkdir -p "$engine_dir"
    local engine="$engine_dir/$(basename "${onnx_path%.onnx}").plan"
    local shape="images:${BATCH}x3x${IMGSZ}x${IMGSZ}"

    echo
    echo "=== [$label] $onnx_path ==="
    echo "    engine: $engine"
    echo "    shape : $shape"
    "$TRTEXEC" \
        --onnx="$onnx_path" \
        --saveEngine="$engine" \
        --shapes="$shape" \
        --iterations="$ITERATIONS" \
        --warmUp="$WARMUP" \
        $extra_flags \
        2>&1 | tee "$engine.log" \
        | grep -E "Throughput|mean:|median:|Latency|GPU Compute Time" \
        || true
}

bench_one "qat-int8" "$QAT_ONNX" "--int8 --fp16"
if [ -n "$FP32_ONNX" ]; then
    bench_one "fp32"     "$FP32_ONNX" ""
fi

echo
echo "Done. Full trtexec logs alongside the *.plan files."
