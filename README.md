# Model Optimizations

This repository contains scripts and tools for optimizing deep learning models, currently with a focus on YOLO object detection models.

## YOLO INT8 QAT

INT8 quantization-aware training of Ultralytics YOLO detectors via NVIDIA
ModelOpt. Two recipes ship in
[`yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`](yolo_quantization/qat/nvidia_modelopt_yolo_qat.py),
auto-selected per model family by `--qat-recipe auto`:

- **`yolo26-distill`** — for E2E dual-head models (yolo26*). PTQ histogram
  calibration on COCO images, then QAT fine-tune with a co-aligned `one2one`
  teacher–student distillation combined with the COCO supervised detection
  loss. Low/high/low LR ladder at `1e-5`.
- **`yolo11-distill`** — for single-head models (YOLOv8 / YOLO11 / YOLO12 …).
  Same distill+supervised loss schedule, plus two **PTQ layer exclusions**
  (modules kept in FP during INT8 quantization): the DFL head (its 16-bin
  probability distribution doesn't survive INT8) and the Detect head output
  quantizers (wide dynamic range). Uses `max` calibration. Required for
  single-head models because their PTQ is already near-lossless and any QAT
  movement on a fully-quantized graph regresses below PTQ.

Full experiment log, resume commands, and the goal-tracking diary live in
[`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

### Results — INT8 PTQ + QAT on COCO `val2017`

Validated with `conf=0.001, iou=0.6, imgsz=640` on RTX 3060 Laptop GPU.

| Model | Recipe | Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---|---:|---:|---:|
| **yolo26s** | `yolo26-distill` | FP32 | 0.6384 | 0.4718 | baseline |
|             |                  | PTQ INT8 | 0.6368 | 0.4706 | -0.0012 (≈0.25%) |
|             |                  | **QAT INT8** | **0.6370** | **0.4701** | **-0.0017 (≈0.36%)** |
| **yolo11s** | `yolo11-distill` | FP32 | 0.6376 | 0.4622 | baseline |
|             |                  | PTQ INT8 | 0.6370 | 0.4614 | -0.0008 (≈0.17%) |
|             |                  | **QAT INT8** (best @ ep 2) | **0.6358** | **0.4602** | **-0.0020 (≈0.43%)** |

### TensorRT inference speedup — batch=1, imgsz=640, RTX 3060 Laptop GPU

Measured via `scripts/measure_speedup.sh <stem>` (trtexec 10.13.3,
`--noTF32` on the FP32 engine for a strict-FP32 baseline).

| Model | Engine | Mean latency | Throughput | Speedup |
|---|---|---:|---:|---:|
| **yolo11s** | FP32 | 6.04 ms | 165.5 qps | 1.00× |
|             | INT8 QAT | 2.48 ms | 403.5 qps | **2.44×** |
| **yolo26s** | FP32 | 6.24 ms | 160.3 qps | 1.00× |
|             | INT8 QAT | 2.60 ms | 385.0 qps | **2.40×** |

### Export accuracy verification — Ultralytics val on COCO `val2017`

Q/DQ node counts only verify graph topology, not numerical correctness. To
confirm calibrated scales and FP fallback work end-to-end, the exported ONNX
files are validated against the PT-checkpoint baseline via Ultralytics'
onnxruntime backend (`scripts/measure_accuracy.sh <stem>`):

| Model | Format | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 ONNX |
|---|---|---:|---:|---:|
| **yolo11s** | ONNX FP32 (ort baseline) | 0.6336 | 0.4592 | baseline |
|             | ONNX INT8 QAT | 0.6325 | 0.4572 | −0.0020 (≈0.4%) |
| **yolo26s** | ONNX FP32 (ort baseline) | 0.6362 | 0.4717 | baseline |
|             | ONNX INT8 QAT | 0.6354 | 0.4700 | −0.0017 (≈0.4%) |

The PT-vs-ONNX-FP32 delta (≈0.003 for yolo11s, ≈0.0001 for yolo26s) is
onnxruntime postprocessing roundoff vs Ultralytics' native PyTorch path
— independent of quantization. The ONNX-FP32-vs-ONNX-INT8 delta is the
relevant "did INT8 quantization break the export" measurement: both models
are within statistical noise of their FP32 ONNX baselines.

## Project Structure

- `yolo_quantization/` - Canonical quantization pipelines
  - `ptq/nvidia_modelopt_yolo.py` - Post-Training Quantization (ONNX-based, INT8/FP8/INT4)
  - `qat/nvidia_modelopt_yolo_qat.py` - Quantization-Aware Training (torch-based, INT8)
  - `qat/README.md` - Active YOLO26 QAT log: metrics, recipe, resume commands
- `examples/` - Older copies of the same pipelines; lag behind `yolo_quantization/` (kept for reference)
- `configs/` - `coco.yaml` and `requirements.txt`
- `scripts/` - Bootstrap shell scripts (`run_modelopt_yolo.sh`, `run_venv.sh`, `download_coco_dataset.sh`)
- `docs/`
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
  --models yolo26s --qat-epochs 10 --qat-batches-per-epoch 200 \
  --calib-size 260 --imgsz 640 --batch 10 --val-batch 8 --device 0

# PTQ-only ONNX (INT8/FP8/INT4)
quantization_venv/bin/python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x --quant-modes int8 fp8 --calib-size 256

# Custom Ultralytics dataset YAML; calibration can use train, val, or an image dir/list
quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --data configs/my_dataset.yaml --calib-source train
```

Both canonical entry points accept `--data <dataset.yaml>`. The YAML should use
the normal Ultralytics detection layout (`path`, `train`, `val`, `names`).
`--calib-source` accepts `train`, `val`, the legacy COCO aliases
`train2017`/`val2017`, or a direct path to an image directory or newline-delimited
image list. The default remains COCO `configs/coco.yaml` with `val2017`
calibration. See
[`docs/CUSTOM_DATASET_QUANTIZATION.md`](docs/CUSTOM_DATASET_QUANTIZATION.md)
for the full custom-dataset runbook.

## Requirements

- Python 3.12+
- PyTorch (CUDA build for GPU runs)
- `nvidia-modelopt[torch]` (installed with `--no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com`)
- Remaining dependencies pinned in `configs/requirements.txt`

## References

### Code in this repository

Pipelines:
- [`yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`](yolo_quantization/qat/nvidia_modelopt_yolo_qat.py) — canonical QAT entry point (FP32 eval → PTQ → distill QAT → ONNX), recipe registry, sensitivity subcommand
- [`yolo_quantization/ptq/nvidia_modelopt_yolo.py`](yolo_quantization/ptq/nvidia_modelopt_yolo.py) — PTQ-only ONNX pipeline (INT8/FP8/INT4)
- [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md) — active YOLO26 QAT experiment log, recipe table, design decisions
- [`AGENTS.md`](AGENTS.md) — canonical agent guide for this repo (env, common commands, architecture notes)

Helper scripts (`scripts/`):
- [`run_modelopt_yolo.sh`](scripts/run_modelopt_yolo.sh) — thin wrapper that picks `qat` or `ptq` and forwards args
- [`run_venv.sh`](scripts/run_venv.sh) — recreate `quantization_venv/` from `configs/requirements.txt` with the NGC extra index
- [`cloud_bootstrap.sh`](scripts/cloud_bootstrap.sh) — one-shot clone + venv + COCO setup for Colab / RunPod / AWS
- [`download_coco_dataset.sh`](scripts/download_coco_dataset.sh) — fetch + unpack COCO 2017 val/test/annotations
- [`run_qat_experiments.sh`](scripts/run_qat_experiments.sh) — drive `matrix` / `ablation` / `seeds` experiment runs into `runs/modelopt_qat_experiments/`
- [`bench_trt.sh`](scripts/bench_trt.sh) — single-ONNX `trtexec` wrapper (INT8 + FP16 fallback)
- [`measure_speedup.sh`](scripts/measure_speedup.sh) — FP32-vs-INT8 TensorRT speedup table; auto-exports FP32 ONNX if missing
- [`measure_accuracy.sh`](scripts/measure_accuracy.sh) — validate exported ONNX (and optionally TRT engines) against the PT-checkpoint mAP

Configs and data:
- [`configs/coco.yaml`](configs/coco.yaml) — Ultralytics-format COCO config pointing at `./datasets/coco`
- [`configs/requirements.txt`](configs/requirements.txt) — pinned dependency list (ModelOpt + ONNX export extras)

Tests:
- [`tests/test_qat_helpers.py`](tests/test_qat_helpers.py) — unit tests for QAT helpers (recipe resolution, amax-drift, DFL discovery, seed-pin, …)
- [`tests/test_ptq_helpers.py`](tests/test_ptq_helpers.py) — unit tests for PTQ helpers

Documentation:
- [`docs/CUSTOM_DATASET_QUANTIZATION.md`](docs/CUSTOM_DATASET_QUANTIZATION.md) — run QAT/PTQ quantization on a custom Ultralytics dataset
- [`docs/LINEAR_QUANTIZATION_THEORY.md`](docs/LINEAR_QUANTIZATION_THEORY.md) — background on linear quantization math
- [`docs/readme_python3.12_on_old_ubuntu_version.md`](docs/readme_python3.12_on_old_ubuntu_version.md) — Python 3.12 via Deadsnakes PPA

### External references

NVIDIA ModelOpt (the underlying quantization library):
- [Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [PyTorch quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html) — `mtq.quantize`, calibrators, quantizer-state-frozen-during-QAT note
- [`cnn_qat` example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat) — the calibrate-closure + `mtq.quantize` + fine-tune pattern this pipeline adapts
- [Model Optimizer docs landing page](https://docs.nvidia.com/deeplearning/modelopt/)

YOLO QAT recipes referenced when designing the loss and exclusion strategy:
- [yolov7_qat (NVIDIA-AI-IOT)](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat) — origin of the histogram-calibration + teacher-student supervision pattern used by `yolo26-distill`
- [ubonpartners/ultralytics `UBON_QAT.md`](https://github.com/ubonpartners/ultralytics/blob/94046a6b2ee10c00281fd11519c576ebf5b3895e/UBON_QAT.md) — documents the DFL + Detect-output exclusion + `max`-calibration pattern that informs `yolo11-distill` / `yolo11-supervised`

Ultralytics (training harness and ONNX/Engine backends used everywhere):
- [Ultralytics repo](https://github.com/ultralytics/ultralytics)
- [Ultralytics docs](https://docs.ultralytics.com/)
- [Training mode reference](https://docs.ultralytics.com/modes/train/)

Datasets and deployment:
- [COCO dataset (cocodataset.org)](https://cocodataset.org/) — `val2017` is the eval set; `train2017` is the QAT fine-tune source pool
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — INT8 engine builder used by `scripts/measure_speedup.sh` and `scripts/bench_trt.sh`
