#!/bin/bash
# One-shot bootstrap for Colab / RunPod / AWS GPU instances.
#
# Idempotent — re-runs are safe:
#   1. clone the repo (or `git pull` if already cloned),
#   2. create/refresh the Python 3.12 venv at quantization_venv/ via run_venv.sh,
#   3. fetch + extract COCO 2017 into datasets/coco/ via download_coco_dataset.sh,
#   4. verify the import chain that the QAT/PTQ pipeline depends on.
#
# Designed to be runnable from a single Colab cell:
#   !bash <(curl -fsSL <raw-url>/scripts/cloud_bootstrap.sh)
#
# Or after cloning manually:
#   bash scripts/cloud_bootstrap.sh
#
# Environment overrides:
#   REPO_URL        git URL to clone (default: https://github.com/olibartfast/model-optimizations.git)
#   REPO_DIR        target dir name (default: model-optimizations)
#   BRANCH          branch/tag/sha to check out (default: master)
#   SKIP_VENV=1     skip running run_venv.sh
#   SKIP_DATASET=1  skip COCO download (e.g. if already mounted)
#   SKIP_VERIFY=1   skip the final import sanity check

set -e

REPO_URL="${REPO_URL:-https://github.com/olibartfast/model-optimizations.git}"
REPO_DIR="${REPO_DIR:-model-optimizations}"
BRANCH="${BRANCH:-master}"

echo "=== Cloud bootstrap for model-optimizations ==="
echo "    repo:   $REPO_URL"
echo "    branch: $BRANCH"
echo "    dir:    $REPO_DIR (absolute: $(pwd)/$REPO_DIR)"
echo

# ---- 1. Clone or update ----------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Repo already present; pulling latest on $BRANCH ..."
    git -C "$REPO_DIR" fetch --quiet origin "$BRANCH"
    git -C "$REPO_DIR" checkout --quiet "$BRANCH"
    git -C "$REPO_DIR" reset --hard --quiet "origin/$BRANCH"
elif [ -d "$REPO_DIR" ] && [ -f "$REPO_DIR/yolo_quantization/qat/nvidia_modelopt_yolo_qat.py" ]; then
    echo "[1/4] Existing checkout (no .git); skipping clone."
else
    echo "[1/4] Cloning $REPO_URL ..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "    -> at commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo

# ---- 2. Python 3.12 check --------------------------------------------------
if ! command -v python3.12 >/dev/null 2>&1; then
    cat <<'EOF' >&2
ERROR: python3.12 not found on PATH.

Options:
  - Colab: switch runtime to a Python 3.12 image (Runtime > Change runtime type),
    or `apt install -y python3.12 python3.12-venv python3.12-dev`.
  - RunPod / AWS: use a Deadsnakes PPA install
    (see docs/readme_python3.12_on_old_ubuntu_version.md).
EOF
    exit 2
fi

# ---- 3. Bootstrap venv -----------------------------------------------------
if [ "${SKIP_VENV:-0}" = "1" ]; then
    echo "[2/4] SKIP_VENV=1, skipping run_venv.sh"
else
    echo "[2/4] Bootstrapping quantization_venv/ via scripts/run_venv.sh ..."
    bash scripts/run_venv.sh
fi
echo

# ---- 4. COCO dataset -------------------------------------------------------
if [ "${SKIP_DATASET:-0}" = "1" ]; then
    echo "[3/4] SKIP_DATASET=1, skipping COCO download"
elif [ -d "datasets/coco/images/val2017" ] || [ -d "datasets/coco/val2017" ]; then
    echo "[3/4] datasets/coco/val2017 already present; skipping download"
else
    echo "[3/4] Fetching COCO 2017 (~25GB) via scripts/download_coco_dataset.sh ..."
    bash scripts/download_coco_dataset.sh
fi
echo

# ---- 5. Sanity check imports ----------------------------------------------
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    echo "[4/4] SKIP_VERIFY=1, skipping import check"
else
    echo "[4/4] Verifying the QAT/PTQ import chain ..."
    if [ ! -x "quantization_venv/bin/python" ]; then
        echo "WARN: quantization_venv/bin/python not found; skipping verify." >&2
    else
        quantization_venv/bin/python - <<'PY'
import sys
mods = [
    "torch", "numpy", "onnx", "ultralytics",
    "pulp", "huggingface_hub",
    "modelopt.torch.quantization", "modelopt.torch.export",
]
fail = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        fail.append((m, repr(e)))
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
if fail:
    for m, e in fail:
        print(f"  FAIL: import {m}: {e}")
    sys.exit(1)
print("import chain ok")
PY
    fi
fi

cat <<EOF

=== Bootstrap complete ===
Next:
  # Resume the existing yolo26s baseline (if you copied yolo26s_ptq.pth):
  quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \\
    --models yolo26s \\
    --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \\
    --qat-epochs 10 --qat-batches-per-epoch 200 \\
    --qat-log-every 20 --qat-eval-every 5 --seed 0 \\
    --batch 16 --val-batch 8 --skip-fp32-eval

  # Or run a fresh model (e.g. yolo26x on a 24GB+ GPU):
  quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \\
    --models yolo26x --qat-epochs 10 --qat-batches-per-epoch 200 \\
    --qat-log-every 20 --qat-eval-every 5 --seed 0 \\
    --batch 16 --val-batch 8 --device 0

  # Run helper tests (no GPU required):
  quantization_venv/bin/python -m pytest tests/ -q
EOF
