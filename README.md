# Model Optimizations

This repository contains scripts and tools for optimizing deep learning models, with a focus on YOLO (You Only Look Once) object detection models.

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

- `yolo_quantization/` - Core quantization implementations for YOLO models
  - `ptq/` - Post-Training Quantization scripts
  - `qat/` - Quantization-Aware Training scripts
  - `configs/` - Configuration files for quantization
  - `models/` - Optimized model outputs
  - `scripts/` - Utility scripts for quantization processes

- `docs/` - Documentation files
  - `NVIDIA_MODELOPT_YOLO.md` - Details on NVIDIA ModelOpt for YOLO
  - `LINEAR_QUANTIZATION_THEORY.md` - Theoretical background on quantization
  - `COCO_DATASET.md` - Information about COCO dataset usage
  - `INSTALLATION_FIX.md` - Installation troubleshooting
  - `readme_python3.12_on_old_ubuntu_version.md` - Specific installation instructions

- `datasets/` - Dataset storage (gitignored)
- `models/` - Model storage (may contain large files)
- `quantization_venv/` - Python virtual environment for quantization

## Getting Started

See the documentation in the `docs/` directory for detailed instructions on:
- Installation procedures
- How to run post-training quantization (PTQ)
- How to run quantization-aware training (QAT)
- Model conversion and optimization techniques

## Requirements

- Python 3.12+
- PyTorch
- NVIDIA ModelOpt (for specific quantization techniques)
- Other dependencies listed in requirements files

## Usage

Navigate to the `yolo_quantization/` directory and explore the `ptq/` and `qat/` subdirectories for specific quantization implementations.

Example usage of the NVIDIA ModelOpt QAT script (runs on yolo11x by default):
```bash
python examples/nvidia_modelopt_yolo_qat.py
```