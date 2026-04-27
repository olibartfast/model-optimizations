# Model Optimizations

This repository contains scripts and tools for optimizing deep learning models, with a focus on YOLO (You Only Look Once) object detection models.

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