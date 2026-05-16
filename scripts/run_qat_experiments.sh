#!/bin/bash
# Drive the runtime QAT experiments noted in yolo_quantization/qat/README.md:
#
#   matrix   - run the winning recipe across yolo26n/s/m/l/x + yolo11x and
#              collect mAP rows for README's results table
#   ablation - rerun yolo26s with --disable-detect-output-quant ON vs OFF
#              against the current winning recipe
#   seeds    - rerun yolo26s with --seed {0,1,2} to put an error bar on the
#              QAT-vs-PTQ gap (resumes from yolo26s_ptq.pth when present)
#
# Each subrun writes to runs/modelopt_qat_experiments/<mode>/<tag>/ so the
# canonical runs/modelopt_qat/ tree is not overwritten.
#
# Usage:
#   ./scripts/run_qat_experiments.sh matrix
#   ./scripts/run_qat_experiments.sh ablation
#   ./scripts/run_qat_experiments.sh seeds
#   DRY_RUN=1 ./scripts/run_qat_experiments.sh seeds   # print commands only
#
# Optional environment overrides:
#   PY              python interpreter (default quantization_venv/bin/python)
#   QAT_EPOCHS      override epoch count (default 10)
#   QAT_BPE         override batches per epoch (default 200)
#   CALIB_SIZE      override calib size (default 260)
#   BATCH           train+calib batch (default 10)
#   VAL_BATCH       val batch (default 8)
#   IMGSZ           image size (default 640)
#   DEVICE          ultralytics device string (default 0)
#   LOG_EVERY       --qat-log-every (default 20)
#   EVAL_EVERY      --qat-eval-every (default 0 = end only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-quantization_venv/bin/python}"
QAT_SCRIPT="yolo_quantization/qat/nvidia_modelopt_yolo_qat.py"
RUNS_ROOT="runs/modelopt_qat"
OUT_ROOT="runs/modelopt_qat_experiments"

QAT_EPOCHS="${QAT_EPOCHS:-10}"
QAT_BPE="${QAT_BPE:-200}"
CALIB_SIZE="${CALIB_SIZE:-260}"
BATCH="${BATCH:-10}"
VAL_BATCH="${VAL_BATCH:-8}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-0}"

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found." >&2
    exit 2
fi
if [ ! -f "$QAT_SCRIPT" ]; then
    echo "ERROR: $QAT_SCRIPT not found." >&2
    exit 2
fi

MODE="${1:-}"
shift || true
if [ -z "$MODE" ]; then
    echo "Usage: $0 {matrix|ablation|seeds}" >&2
    exit 2
fi

run_cmd() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    "$@"
}

base_args=(
    --qat-mode distill
    --calib-method entropy
    --calib-size "$CALIB_SIZE"
    --qat-epochs "$QAT_EPOCHS"
    --qat-batches-per-epoch "$QAT_BPE"
    --batch "$BATCH"
    --val-batch "$VAL_BATCH"
    --imgsz "$IMGSZ"
    --device "$DEVICE"
    --qat-log-every "$LOG_EVERY"
    --qat-eval-every "$EVAL_EVERY"
)

stage_for() {
    # Stage the canonical runs/modelopt_qat/<model>/ as the working dir for this
    # subrun by moving the experiment outputs out afterwards. We accomplish the
    # same thing by checking the per-model summary.json post-run.
    local tag="$1"
    local mode="$2"
    local model="$3"
    mkdir -p "$OUT_ROOT/$mode/$tag"
}

stash_outputs() {
    local mode="$1"
    local tag="$2"
    local model="$3"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  (dry-run) skipping stash of $RUNS_ROOT/$model"
        return 0
    fi
    local src="$RUNS_ROOT/$model"
    local dst="$OUT_ROOT/$mode/$tag/$model"
    if [ -d "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        rm -rf "$dst"
        cp -r "$src" "$dst"
        echo "  stashed -> $dst"
    fi
}

case "$MODE" in
    matrix)
        # Full winning recipe across the model family. PTQ is calibrated fresh
        # for each model since we don't have *_ptq.pth for n/m/l/x or yolo11x.
        for model in yolo26n yolo26s yolo26m yolo26l yolo26x yolo11x; do
            tag="$model"
            stage_for "$tag" matrix "$model"
            run_cmd "$PY" "$QAT_SCRIPT" --models "$model" "${base_args[@]}" --skip-fp32-eval
            stash_outputs matrix "$tag" "$model"
        done
        ;;

    ablation)
        # On yolo26s only: toggle --disable-detect-output-quant. Both subruns
        # calibrate from scratch because the flag affects PTQ.
        for variant in detect_quant_on detect_quant_off; do
            extra=()
            if [ "$variant" = "detect_quant_off" ]; then
                extra=(--disable-detect-output-quant)
            fi
            stage_for "$variant" ablation yolo26s
            run_cmd "$PY" "$QAT_SCRIPT" --models yolo26s \
                "${base_args[@]}" --skip-fp32-eval "${extra[@]}"
            stash_outputs ablation "$variant" yolo26s
        done
        ;;

    seeds)
        # Multi-seed variance on yolo26s. Resume from the cached PTQ checkpoint
        # if available so PTQ amax is identical across seeds and only the QAT
        # fine-tune varies.
        ptq_ckpt="$RUNS_ROOT/yolo26s/yolo26s_ptq.pth"
        resume_args=()
        if [ -f "$ptq_ckpt" ]; then
            resume_args=(--from-ptq "$ptq_ckpt")
            echo "  resuming from $ptq_ckpt"
        else
            echo "  $ptq_ckpt missing — each seed will recalibrate PTQ from scratch"
        fi
        for seed in 0 1 2; do
            tag="seed_$seed"
            stage_for "$tag" seeds yolo26s
            run_cmd "$PY" "$QAT_SCRIPT" --models yolo26s \
                "${base_args[@]}" --skip-fp32-eval --seed "$seed" \
                "${resume_args[@]}"
            stash_outputs seeds "$tag" yolo26s
        done
        ;;

    *)
        echo "Unknown mode: $MODE (expected matrix|ablation|seeds)" >&2
        exit 2
        ;;
esac

echo
echo "Done. Per-subrun outputs under $OUT_ROOT/$MODE/"
echo "Aggregate with: jq -r '.stages[] | select(.stage==\"qat\") | [\"$MODE\", input_filename, .map50, .map] | @tsv' $OUT_ROOT/$MODE/*/*/summary.json"
