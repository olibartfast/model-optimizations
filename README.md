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

Result tables below compare two backends side-by-side:

- **NVIDIA ModelOpt** — canonical, deployable INT8 pipeline.
- **PyTorch AO** — fake-quant path
  ([`yolo_quantization/qat/pytorch_ao_yolo_qat.py`](yolo_quantization/qat/pytorch_ao_yolo_qat.py))
  using `torch.ao.quantization` observers. Its activation fake-quant is
  **symmetric int8** (zero-point 0), so the exported Q/DQ ONNX builds a
  TensorRT INT8 engine — the AO rows below are now measured end-to-end, not
  TBD. ModelOpt remains the canonical path; AO is kept for recipe comparison.

### Results (ModelOpt vs PyTorch AO) — INT8 PTQ + QAT on COCO `val2017`

All numbers validated with `conf=0.001, iou=0.6, imgsz=640` on RTX 3060
Laptop GPU.

| Model | Backend | Recipe | Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---|---|---:|---:|---:|
| **yolo26s** | ModelOpt | `yolo26-distill` | FP32 | 0.6384 | 0.4718 | baseline |
|             | ModelOpt |                  | PTQ INT8 | 0.6368 | 0.4706 | -0.0012 (≈0.25%) |
|             | ModelOpt |                  | **QAT INT8** | **0.6370** | **0.4701** | **-0.0017 (≈0.36%)** |
|             | PyTorch AO | `yolo26-ao-distill` | PTQ INT8 | 0.6257 | 0.4525 | -0.0194 (≈4.1%) |
|             | PyTorch AO |                     | **QAT INT8** | **0.6263** | **0.4551** | **-0.0168 (≈3.6%)** |
| **yolo11s** | ModelOpt | `yolo11-distill` | FP32 | 0.6376 | 0.4622 | baseline |
|             | ModelOpt |                  | PTQ INT8 | 0.6370 | 0.4614 | -0.0008 (≈0.17%) |
|             | ModelOpt |                  | **QAT INT8** (best @ ep 2) | **0.6358** | **0.4602** | **-0.0020 (≈0.43%)** |
|             | PyTorch AO | `yolo11-ao-distill` | PTQ INT8 | 0.6240 | 0.4494 | -0.0128 (≈2.8%) |
|             | PyTorch AO |                     | **QAT INT8** | **0.6292** | **0.4511** | **-0.0111 (≈2.4%)** |

For both models the PyTorch AO QAT step recovers mAP over its PTQ baseline
(+0.0026 yolo26s, +0.0017 yolo11s) but the AO path trails ModelOpt: AO's
symmetric-int8 activation quantizers are stricter than ModelOpt's calibrated
per-tensor quantizers, so its PTQ baseline starts ≈3-4% lower and QAT does not
fully close the gap. ModelOpt stays the recommended INT8 pipeline.

### TensorRT inference speedup (ModelOpt vs PyTorch AO) — batch=1, imgsz=640, RTX 3060 Laptop GPU

Measured via `scripts/measure_speedup.sh <stem>` for ModelOpt and
`scripts/measure_speedup.sh <stem> --ao` for PyTorch AO (trtexec 10.13.3,
`--noTF32` on the FP32 engine for a strict-FP32 baseline). The FP32 engine is
backend-independent; each `--ao` run re-measures it and the AO INT8 speedup is
computed against that same-run FP32 number.

| Model | Backend | Engine | Mean latency | Throughput | Speedup |
|---|---|---|---:|---:|---:|
| **yolo11s** | ModelOpt   | FP32     | 6.04 ms | 165.5 qps | 1.00× |
|             | ModelOpt   | INT8 QAT | 2.48 ms | 403.5 qps | **2.44×** |
|             | PyTorch AO | INT8 QAT | 1.98 ms | 503.5 qps | **3.05×** |
| **yolo26s** | ModelOpt   | FP32     | 6.24 ms | 160.3 qps | 1.00× |
|             | ModelOpt   | INT8 QAT | 2.60 ms | 385.0 qps | **2.40×** |
|             | PyTorch AO | INT8 QAT | 2.04 ms | 490.6 qps | **3.08×** |

The AO INT8 engine is faster than the ModelOpt INT8 engine because the
`*-ao-distill` recipes disable Conv *output* fake-quant, so the exported graph
carries fewer Q/DQ nodes and TensorRT fuses longer INT8 runs. That speed comes
with the lower mAP shown above — a latency/accuracy trade-off, not a free win.

### Export accuracy verification (ModelOpt vs PyTorch AO) — Ultralytics val on COCO `val2017`

Q/DQ node counts only verify graph topology, not numerical correctness. To
confirm calibrated scales and FP fallback work end-to-end, the exported ONNX
files are validated against the PT-checkpoint baseline via Ultralytics'
onnxruntime backend (`scripts/measure_accuracy.sh <stem>`, add `--ao` for the
PyTorch AO export). This executes the Q/DQ graph TensorRT ingests, so it is
the relevant "did INT8 export break accuracy" measurement.

| Model | Backend | Format | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 ONNX |
|---|---|---|---:|---:|---:|
| **yolo11s** | ModelOpt   | ONNX FP32 (ort baseline) | 0.6336 | 0.4592 | baseline |
|             | ModelOpt   | ONNX INT8 QAT            | 0.6325 | 0.4572 | −0.0020 (≈0.4%) |
|             | PyTorch AO | ONNX INT8 QAT            | 0.6265 | 0.4491 | −0.0101 (≈2.2%) |
| **yolo26s** | ModelOpt   | ONNX FP32 (ort baseline) | 0.6362 | 0.4717 | baseline |
|             | ModelOpt   | ONNX INT8 QAT            | 0.6354 | 0.4700 | −0.0017 (≈0.4%) |
|             | PyTorch AO | ONNX INT8 QAT            | 0.6252 | 0.4564 | −0.0153 (≈3.2%) |

For the ModelOpt rows, the PT-vs-ONNX-FP32 delta (≈0.003 for yolo11s,
≈0.0001 for yolo26s) is onnxruntime postprocessing roundoff vs Ultralytics'
native PyTorch path — independent of quantization. The ONNX-FP32-vs-ONNX-INT8
delta is the relevant "did INT8 quantization break the export" measurement:
both ModelOpt models are within statistical noise of their FP32 ONNX
baselines, while the PyTorch AO INT8 export tracks its lower QAT-INT8 mAP
(0.4564 vs the 0.4551 PT checkpoint for yolo26s; 0.4491 vs 0.4511 for
yolo11s) — i.e. the export is faithful, the accuracy loss is the AO recipe's
symmetric-int8 quantization itself, not an export bug. The literal TensorRT
engine was not validated through Ultralytics because its `tensorrt` Python
package (10.16) and the `trtexec` that builds the engine (10.13) disagree on
the serialized engine format; the onnxruntime Q/DQ measurement above is the
backend-independent stand-in.

### PyTorch AO QAT

The PyTorch AO rows in the tables above are populated by
[`yolo_quantization/qat/pytorch_ao_yolo_qat.py`](yolo_quantization/qat/pytorch_ao_yolo_qat.py).
It uses `torch.ao.quantization` observers and `FakeQuantize` modules wrapped
around YOLO `Conv2d` layers, then reuses the same teacher-student +
supervised detection fine-tune loop as the ModelOpt recipe.

The activation fake-quant is **symmetric int8** (`qint8`, `per_tensor_symmetric`,
zero-point 0) and weights are per-channel symmetric int8. TensorRT rejects
asymmetric (non-zero zero-point) `QuantizeLinear` nodes on GPU, so this
symmetric scheme is what makes the exported Q/DQ ONNX build an INT8 engine —
verified above with `measure_speedup.sh --ao` and `measure_accuracy.sh --ao`.
Checkpoints and the ONNX export are saved under `runs/pytorch_ao_qat/`. This
path is kept for recipe research; ModelOpt is still the recommended INT8
pipeline because it lands a higher mAP at this quantization budget.

```bash
python yolo_quantization/qat/pytorch_ao_yolo_qat.py \
  --models yolo26s --ao-recipe auto \
  --calib-source train2017 --calib-size 260 \
  --qat-epochs 10 --qat-batches-per-epoch 200 \
  --batch 10 --val-batch 8 --device 0 \
  --skip-fp32-eval

# TensorRT speedup + export-accuracy of the AO INT8 ONNX:
./scripts/measure_speedup.sh  yolo26s --ao
./scripts/measure_accuracy.sh yolo26s --ao
```

## Project Structure

- `yolo_quantization/` - Canonical quantization pipelines
  - `ptq/nvidia_modelopt_yolo.py` - Post-Training Quantization (ONNX-based, INT8/FP8/INT4)
  - `qat/nvidia_modelopt_yolo_qat.py` - Quantization-Aware Training (torch-based, INT8)
  - `qat/pytorch_ao_yolo_qat.py` - PyTorch AO fake-quant QAT recipe (symmetric-int8, TensorRT-deployable)
  - `qat/README.md` - Active YOLO26 QAT log: metrics, recipe, resume commands
- `examples/` - Tutorials and small environment examples
- `configs/` - `coco.yaml` and `requirements.txt`
- `scripts/` - Bootstrap shell scripts (`run_modelopt_yolo.sh`, `run_venv.sh`, `download_coco_dataset.sh`)
- `docs/`
  - `LINEAR_QUANTIZATION_THEORY.md` - Linear quantization theory background
  - `readme_python3.12_on_old_ubuntu_version.md` - Python 3.12 via Deadsnakes PPA
- `datasets/` - COCO data (gitignored)
- `<venv-dir>/` - Repo-local Python 3.12 venv created by `scripts/run_venv.sh`

## Getting Started

Canonical setup and run commands live in [`AGENTS.md`](AGENTS.md) and
[`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

Quick examples (run from repo root):

```bash
# Create a Python 3.12 virtual environment
python3.12 -m venv <venv-dir>
source <venv-dir>/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt

# Equivalent convenience wrapper:
export QUANTIZATION_VENV=<venv-dir>
./scripts/run_venv.sh

# Full YOLO26-small QAT pipeline (FP32 eval -> PTQ -> QAT -> ONNX)
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --qat-epochs 10 --qat-batches-per-epoch 200 \
  --calib-size 260 --imgsz 640 --batch 10 --val-batch 8 --device 0

# PTQ-only ONNX (INT8/FP8/INT4)
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x --quant-modes int8 fp8 --calib-size 256

# Custom Ultralytics dataset YAML; calibration can use train, val, or an image dir/list
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
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
- [`yolo_quantization/qat/pytorch_ao_yolo_qat.py`](yolo_quantization/qat/pytorch_ao_yolo_qat.py) — PyTorch AO fake-quant QAT path for comparing non-ModelOpt recipes
- [`yolo_quantization/ptq/nvidia_modelopt_yolo.py`](yolo_quantization/ptq/nvidia_modelopt_yolo.py) — PTQ-only ONNX pipeline (INT8/FP8/INT4)
- [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md) — active YOLO26 QAT experiment log, recipe table, design decisions
- [`AGENTS.md`](AGENTS.md) — canonical agent guide for this repo (env, common commands, architecture notes)

Helper scripts (`scripts/`):
- [`run_modelopt_yolo.sh`](scripts/run_modelopt_yolo.sh) — thin wrapper that picks `qat` or `ptq` and forwards args
- [`run_venv.sh`](scripts/run_venv.sh) — create `$QUANTIZATION_VENV` from `configs/requirements.txt` with the NGC extra index
- [`cloud_bootstrap.sh`](scripts/cloud_bootstrap.sh) — one-shot clone + venv + COCO setup for Colab / RunPod / AWS
- [`download_coco_dataset.sh`](scripts/download_coco_dataset.sh) — fetch + unpack COCO 2017 val/test/annotations
- [`run_qat_experiments.sh`](scripts/run_qat_experiments.sh) — drive `matrix` / `ablation` / `seeds` experiment runs into `runs/modelopt_qat_experiments/`
- [`bench_trt.sh`](scripts/bench_trt.sh) — single-ONNX `trtexec` wrapper (INT8 + FP16 fallback)
- [`measure_speedup.sh`](scripts/measure_speedup.sh) — FP32-vs-INT8 TensorRT speedup table; auto-exports FP32 ONNX if missing; `--ao` targets the PyTorch AO export
- [`measure_accuracy.sh`](scripts/measure_accuracy.sh) — validate exported ONNX (and optionally TRT engines) against the PT-checkpoint mAP; `--ao` targets the PyTorch AO export

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
