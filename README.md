# Model Optimizations

This repository contains scripts and tools for optimizing deep learning models, currently with a focus on YOLO object detection models.

## YOLO26 quantization

INT8 quantization of Ultralytics YOLO26 detectors via NVIDIA ModelOpt:
PTQ histogram calibration on COCO images, followed by QAT fine-tuning that
combines a co-aligned `one2one`-head teacher–student distillation with the
COCO supervised detection loss. Full pipeline, recovery history, and
root-cause investigation: [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

### Results — INT8 PTQ + QAT on COCO `val2017`

Pipeline: NVIDIA ModelOpt PTQ calibration → QAT fine-tune (co-aligned `one2one`
distillation + COCO supervised loss) → ONNX export. Validated with
`conf=0.001, iou=0.6, imgsz=640` on RTX 3060 Laptop GPU.

| Model | Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---:|---:|---:|
| **yolo26s** | FP32 | 0.6384 | 0.4718 | baseline |
|             | PTQ INT8 | 0.6368 | 0.4706 | -0.0012 (≈0.25%) |
|             | **QAT INT8** | **0.6370** | **0.4697** | **-0.0021 (≈0.44%)** |
| yolo26n | — | — | — | TBD |
| yolo26m | — | — | — | TBD |
| yolo26l | — | — | — | TBD |
| yolo26x | — | — | — | TBD |
| yolo11x | — | — | — | TBD |

QAT closes the regression vs PTQ to within statistical noise of a 5000-image
val set (Δ = -0.0009 mAP50-95).

## Project Structure

- `yolo_quantization/` - Canonical quantization pipelines
  - `ptq/nvidia_modelopt_yolo.py` - Post-Training Quantization (ONNX-based, INT8/FP8/INT4)
  - `qat/nvidia_modelopt_yolo_qat.py` - Quantization-Aware Training (torch-based, INT8)
  - `qat/README.md` - Active YOLO26 QAT log: metrics, recipe, resume commands
- `examples/` - Older copies of the same pipelines; lag behind `yolo_quantization/` (kept for reference)
- `configs/` - `coco.yaml` and `requirements.txt`
- `scripts/` - Bootstrap shell scripts (`run_modelopt_yolo.sh`, `run_venv.sh`, `download_coco_dataset.sh`)
- `docs/`
  - `yolo26_int8_qat_paper.md` - Write-up of the YOLO26 INT8 QAT recovery recipe
  - `LINEAR_QUANTIZATION_THEORY.md` - Linear quantization theory background
  - `readme_python3.12_on_old_ubuntu_version.md` - Python 3.12 via Deadsnakes PPA
- `datasets/` - COCO data (gitignored)
- `quantization_venv/` - Pre-built Python 3.12 venv with ModelOpt installed

## Getting Started

Canonical setup and run commands live in [`AGENTS.md`](AGENTS.md) and
[`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

Quick examples (run from repo root):

```bash
# Full YOLO26-small QAT pipeline (FP32 eval -> PTQ -> QAT -> ONNX)
quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --qat-epochs 3 --calib-size 512 --imgsz 640 --batch 16 \
  --val-batch 8 --device 0

# PTQ-only ONNX (INT8/FP8/INT4)
quantization_venv/bin/python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x --quant-modes int8 fp8 --calib-size 256
```

## Requirements

- Python 3.12+
- PyTorch (CUDA build for GPU runs)
- `nvidia-modelopt[torch]` (installed with `--no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com`)
- Remaining dependencies pinned in `configs/requirements.txt`
