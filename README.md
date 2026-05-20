# Model Optimizations

A workspace for deep-learning model optimization experiments, tooling, and deployment-oriented recipes.

The repository is organized around practical optimization workflows: quantization, export validation, benchmarking, helper scripts, tests, and documentation. The most developed workflow today is the **YOLO INT8 quantization** stack, which includes NVIDIA ModelOpt and PyTorch AO pipelines for PTQ/QAT, ONNX Q/DQ export, and TensorRT benchmarking.

Current YOLO validation covers **`yolo11s`** and **`yolo26s`** on COCO `val2017`. Other model families or YOLO variants may work with the same infrastructure, but they should be treated as unvalidated until tested.

## Repository scope

This repository is intended to grow as a collection of model-optimization utilities and experiments, not as a YOLO-only project.

Current areas:

* quantization pipelines;
* ONNX export and validation;
* TensorRT latency benchmarking;
* reproducible setup scripts;
* experiment helpers;
* quantization theory notes;
* custom-dataset quantization documentation;
* tests for PTQ/QAT helper logic.

Currently mature area:

* INT8 PTQ/QAT for Ultralytics YOLO detectors.

## Current highlighted workflow: YOLO INT8 QAT

The main implemented workflow is INT8 quantization-aware training of Ultralytics YOLO detectors.

Two YOLO quantization paths are compared:

* **NVIDIA ModelOpt** — the canonical, deployable INT8 pipeline.
* **PyTorch AO** — a fake-quant reference path using `torch.ao.quantization`, kept for recipe comparison and research.

The main QAT entry point is:

[`yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`](yolo_quantization/qat/nvidia_modelopt_yolo_qat.py)

The PyTorch AO comparison path is:

[`yolo_quantization/qat/pytorch_ao_yolo_qat.py`](yolo_quantization/qat/pytorch_ao_yolo_qat.py)

The full YOLO QAT experiment log, resume commands, and design notes live in:

[`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md)

## YOLO results at a glance

INT8 QAT was evaluated on COCO `val2017` with `conf=0.001`, `iou=0.6`, and `imgsz=640` on an RTX 3060 Laptop GPU. TensorRT latency was measured at batch 1. Δ is the relative mAP50-95 change versus the FP32 baseline.

| Model       | Pipeline            | mAP50-95 | Δ vs FP32 | TRT latency |   Speedup |
| ----------- | ------------------- | -------: | --------: | ----------: | --------: |
| **yolo26s** | FP32 baseline       |   0.4718 |         — |     6.24 ms |     1.00× |
|             | ModelOpt INT8 QAT   |   0.4701 |     −0.4% |     2.60 ms | **2.40×** |
|             | PyTorch AO INT8 QAT |   0.4661 |     −1.2% |     2.23 ms | **2.81×** |
| **yolo11s** | FP32 baseline       |   0.4622 |         — |     6.04 ms |     1.00× |
|             | ModelOpt INT8 QAT   |   0.4602 |     −0.4% |     2.48 ms | **2.44×** |
|             | PyTorch AO INT8 QAT |   0.4560 |     −1.3% |     2.23 ms | **2.68×** |

**Current recommendation for YOLO:** use **ModelOpt INT8 QAT** for production-oriented INT8 deployment. On the tested models, it gives about **0.4% relative mAP50-95 loss** with roughly **2.4× TensorRT speedup**. The PyTorch AO path is now close — within **≈1.3% relative mAP50-95** at a **2.7–2.8× speedup** — but remains a recipe-research path; ModelOpt is fully INT8 and slightly more accurate.

## Quick start

Run these commands from the repository root.

```bash
# Create a Python 3.12 virtual environment
python3.12 -m venv <venv-dir>
source <venv-dir>/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt
```

Or use the convenience wrapper:

```bash
export QUANTIZATION_VENV=<venv-dir>
./scripts/run_venv.sh
```

Run the full YOLO26-small QAT pipeline:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --qat-epochs 10 \
  --qat-batches-per-epoch 200 \
  --calib-size 260 \
  --imgsz 640 \
  --batch 10 \
  --val-batch 8 \
  --device 0
```

Run PTQ-only ONNX export:

```bash
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x \
  --quant-modes int8 fp8 \
  --calib-size 256
```

Run on a custom Ultralytics dataset:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --calib-source train
```

Both canonical YOLO entry points accept `--data <dataset.yaml>`. The YAML should follow the standard Ultralytics detection layout with `path`, `train`, `val`, and `names` fields. `--calib-source` accepts `train`, `val`, the legacy COCO aliases `train2017` / `val2017`, or a direct path to an image directory or newline-delimited image list.

For the full custom-dataset runbook, see [`docs/CUSTOM_DATASET_QUANTIZATION.md`](docs/CUSTOM_DATASET_QUANTIZATION.md).

## YOLO QAT recipes

The ModelOpt QAT entry point provides two recipes, auto-selected by `--qat-recipe auto`.

### `yolo26-distill`

Tested on YOLO26 E2E dual-head models.

The recipe performs PTQ histogram calibration on COCO images, then fine-tunes with QAT using a co-aligned `one2one` teacher-student distillation loss combined with the standard COCO supervised detection loss.

### `yolo11-distill`

Tested on YOLO11 single-head models.

The recipe is designed for the tested YOLO11 setup and uses the same distillation plus supervised fine-tuning schedule as the YOLO26 path. It also applies two PTQ layer exclusions:

* the DFL head, because its 16-bin probability distribution is fragile under INT8 quantization;
* the Detect head output quantizers, because of their wide dynamic range.

In the tested YOLO11 setup, these exclusions are required because PTQ is already near-lossless. QAT movement on the fully quantized graph regresses below the PTQ baseline.

## Accuracy by stage: PTQ vs QAT

All numbers were validated on COCO `val2017` with `conf=0.001`, `iou=0.6`, and `imgsz=640` on an RTX 3060 Laptop GPU. Δ shows absolute mAP50-95 loss and relative percentage loss versus FP32.

| Model       | Backend    | Recipe              | Stage                    |      mAP50 |   mAP50-95 | Δ mAP50-95 vs FP32 |
| ----------- | ---------- | ------------------- | ------------------------ | ---------: | ---------: | -----------------: |
| **yolo26s** | ModelOpt   | `yolo26-distill`    | FP32                     |     0.6384 |     0.4718 |           baseline |
|             | ModelOpt   |                     | PTQ INT8                 |     0.6368 |     0.4706 |     −0.0012 ≈0.25% |
|             | ModelOpt   |                     | **QAT INT8**             | **0.6370** | **0.4701** | **−0.0017 ≈0.36%** |
|             | PyTorch AO | `yolo26-ao-distill` | PTQ INT8                 |     0.6321 |     0.4666 |      −0.0053 ≈1.1% |
|             | PyTorch AO |                     | **QAT INT8**             | **0.6326** | **0.4661** |  **−0.0058 ≈1.2%** |
| **yolo11s** | ModelOpt   | `yolo11-distill`    | FP32                     |     0.6376 |     0.4622 |           baseline |
|             | ModelOpt   |                     | PTQ INT8                 |     0.6370 |     0.4614 |     −0.0008 ≈0.17% |
|             | ModelOpt   |                     | **QAT INT8** best @ ep 2 | **0.6358** | **0.4602** | **−0.0020 ≈0.43%** |
|             | PyTorch AO | `yolo11-ao-distill` | PTQ INT8                 |     0.6321 |     0.4582 |     −0.0040 ≈0.87% |
|             | PyTorch AO |                     | **QAT INT8**             | **0.6337** | **0.4560** |  **−0.0062 ≈1.3%** |

Both AO recipes keep the Detect head in FP32 and calibrate with histogram observers (see [PyTorch AO quantization details](#pytorch-ao-quantization-details)); this brought AO PTQ to within 0.003–0.005 mAP50-95 of ModelOpt PTQ. Once PTQ is that strong the QAT step no longer helps — for AO (as for ModelOpt single-head models) QAT lands at or just below PTQ, so the deployable model is whichever stage scored higher.

## TensorRT speedup

Measured at batch 1 and `imgsz=640` on an RTX 3060 Laptop GPU.

The benchmark uses `scripts/measure_speedup.sh <stem>`, with `--ao` for the PyTorch AO export. `trtexec` 10.13.3 was used. The FP32 engine was measured with `--noTF32` for a strict-FP32 baseline.

| Model       | Backend    | Engine   | Mean latency | Throughput |   Speedup |
| ----------- | ---------- | -------- | -----------: | ---------: | --------: |
| **yolo26s** | ModelOpt   | FP32     |      6.24 ms |  160.3 qps |     1.00× |
|             | ModelOpt   | INT8 QAT |      2.60 ms |  385.0 qps | **2.40×** |
|             | PyTorch AO | INT8 QAT |      2.23 ms |  448.5 qps | **2.81×** |
| **yolo11s** | ModelOpt   | FP32     |      6.04 ms |  165.5 qps |     1.00× |
|             | ModelOpt   | INT8 QAT |      2.48 ms |  403.5 qps | **2.44×** |
|             | PyTorch AO | INT8 QAT |      2.23 ms |  448.4 qps | **2.68×** |

The AO INT8 engine is still slightly faster than the ModelOpt engine because the `*-ao-distill` recipes disable Conv output fake-quant, so the graph has fewer Q/DQ nodes and TensorRT fuses longer INT8 regions. Keeping the Detect head in FP32 — the change that closed most of the accuracy gap — gives some speed back: AO measured 3.0–3.1× when the head was quantized, 2.7–2.8× with it in FP. Net, AO buys ≈0.011 mAP50-95 for ≈0.3× of speedup.

## Export accuracy: ONNX vs FP32 baseline

Q/DQ node counts verify graph topology, but not numerical fidelity. Exported ONNX models are validated against the PyTorch checkpoint baseline through Ultralytics' ONNX Runtime backend using `scripts/measure_accuracy.sh <stem>`, with `--ao` for the AO export.

| Model       | Backend    | Format                 |  mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 ONNX |
| ----------- | ---------- | ---------------------- | -----: | -------: | ----------------------: |
| **yolo26s** | ModelOpt   | ONNX FP32 ORT baseline | 0.6362 |   0.4717 |                baseline |
|             | ModelOpt   | ONNX INT8 QAT          | 0.6354 |   0.4700 |           −0.0017 ≈0.4% |
|             | PyTorch AO | ONNX INT8 QAT          | 0.6304 |   0.4659 |           −0.0058 ≈1.2% |
| **yolo11s** | ModelOpt   | ONNX FP32 ORT baseline | 0.6336 |   0.4592 |                baseline |
|             | ModelOpt   | ONNX INT8 QAT          | 0.6325 |   0.4572 |           −0.0020 ≈0.4% |
|             | PyTorch AO | ONNX INT8 QAT          | 0.6313 |   0.4544 |           −0.0048 ≈1.0% |

The PyTorch-checkpoint vs ONNX-FP32 gap is about 0.003 for `yolo11s` and about 0.0001 for `yolo26s`. This appears to come from ONNX Runtime postprocessing roundoff and is independent of quantization.

The ONNX-FP32 to ONNX-INT8 delta is the main export-fidelity check. ModelOpt INT8 stays within noise of its FP32 ONNX baseline. PyTorch AO INT8 matches its own QAT checkpoint closely, so the AO loss comes from the quantization recipe rather than an export bug.

> **Note:** TensorRT engines are benchmarked for speed, but not validated for accuracy through Ultralytics in this setup. The installed `tensorrt` Python package version 10.16 and the `trtexec` version 10.13 used to build the engine disagree on the serialized engine format. The ONNX Runtime Q/DQ validation above is used as the backend-independent accuracy stand-in.

## PyTorch AO quantization details

The PyTorch AO path uses symmetric INT8 quantization:

* activations: `qint8`, `per_tensor_symmetric`, zero-point 0;
* weights: per-channel symmetric INT8.

TensorRT rejects asymmetric `QuantizeLinear` nodes with non-zero zero-points on GPU, so the symmetric scheme is required for the exported Q/DQ ONNX graph to build as an INT8 engine.

Two recipe choices close most of the accuracy gap to ModelOpt, both found by PTQ-only ablations on `yolo26s`:

* **Detect head kept FP32** (`--keep-detect-head-fp32`, the default for both AO recipes). AO's symmetric-int8 quantizers on the detect-head prediction towers cost ≈0.013 mAP50-95; keeping the whole `Detect` module FP — backbone and neck stay INT8 — recovers it. This is the dominant fix: it took `yolo26s` PTQ from 0.4523 to 0.4655.
* **Histogram activation observers** for both recipes. `HistogramObserver` clips activation outliers far better than `MovingAverageMinMaxObserver` (+0.0075 mAP50-95 at PTQ for `yolo11s`; histogram already won for `yolo26s`).

With both, AO PTQ lands within 0.003–0.005 mAP50-95 of ModelOpt PTQ. The cost is speed: the FP Detect head drops the TensorRT speedup from ≈3.0× to ≈2.7×.

Checkpoints and ONNX exports are saved under:

```text
runs/pytorch_ao_qat/
```

## Project structure

```text
.github/workflows/                     # CI workflows
configs/                               # Dataset and dependency configuration
docs/                                  # Documentation and quantization notes
examples/                              # Tutorials and small environment examples
scripts/                               # Setup, benchmark, export, and experiment helpers
tests/                                 # Unit tests for PTQ/QAT helper logic
yolo_quantization/                     # Current mature quantization workflow for YOLO
AGENTS.md                              # Agent/developer guide for this repo
CLAUDE.md                              # Claude-specific project guide
README.md                              # This file
```

YOLO-specific layout:

```text
yolo_quantization/
  ptq/
    nvidia_modelopt_yolo.py            # PTQ-only ONNX pipeline: INT8 / FP8 / INT4
  qat/
    nvidia_modelopt_yolo_qat.py        # Canonical ModelOpt QAT pipeline
    pytorch_ao_yolo_qat.py             # PyTorch AO fake-quant QAT path
    README.md                          # Active QAT experiment log and design notes
```

Supporting files:

```text
configs/
  coco.yaml                            # COCO config for Ultralytics
  requirements.txt                     # Pinned dependencies

docs/
  CUSTOM_DATASET_QUANTIZATION.md       # Custom dataset runbook
  LINEAR_QUANTIZATION_THEORY.md        # Quantization theory background
  readme_python3.12_on_old_ubuntu_version.md

scripts/
  run_modelopt_yolo.sh                 # PTQ/QAT wrapper
  run_venv.sh                          # Virtualenv setup wrapper
  cloud_bootstrap.sh                   # Cloud setup helper
  download_coco_dataset.sh             # COCO download helper
  run_qat_experiments.sh               # Matrix / ablation / seed experiments
  bench_trt.sh                         # Single-ONNX trtexec wrapper
  measure_speedup.sh                   # FP32 vs INT8 TensorRT benchmark
  measure_accuracy.sh                  # ONNX / optional engine accuracy validation

tests/
  test_qat_helpers.py                  # QAT helper tests
  test_ptq_helpers.py                  # PTQ helper tests
```

## Requirements

* Python 3.12+
* PyTorch with CUDA support for GPU runs
* `nvidia-modelopt[torch]`
* Dependencies pinned in [`configs/requirements.txt`](configs/requirements.txt)

Install dependencies with the NVIDIA PyPI extra index:

```bash
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt
```

## Repository references

### Current optimization pipelines

* [`yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`](yolo_quantization/qat/nvidia_modelopt_yolo_qat.py) — canonical YOLO QAT entry point: FP32 eval, PTQ, distillation QAT, ONNX export, recipe registry, and sensitivity subcommand.
* [`yolo_quantization/qat/pytorch_ao_yolo_qat.py`](yolo_quantization/qat/pytorch_ao_yolo_qat.py) — PyTorch AO fake-quant QAT path for comparing non-ModelOpt recipes.
* [`yolo_quantization/ptq/nvidia_modelopt_yolo.py`](yolo_quantization/ptq/nvidia_modelopt_yolo.py) — PTQ-only ONNX pipeline for INT8, FP8, and INT4.
* [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md) — active YOLO QAT experiment log, recipe table, and design decisions.
* [`AGENTS.md`](AGENTS.md) — agent guide for environment setup, common commands, and architecture notes.

### Helper scripts

* [`scripts/run_modelopt_yolo.sh`](scripts/run_modelopt_yolo.sh) — thin wrapper that picks `qat` or `ptq` and forwards arguments.
* [`scripts/run_venv.sh`](scripts/run_venv.sh) — create `$QUANTIZATION_VENV` from `configs/requirements.txt` with the NGC extra index.
* [`scripts/cloud_bootstrap.sh`](scripts/cloud_bootstrap.sh) — one-shot clone, virtualenv, and COCO setup for Colab, RunPod, or AWS.
* [`scripts/download_coco_dataset.sh`](scripts/download_coco_dataset.sh) — fetch and unpack COCO 2017 val/test/annotations.
* [`scripts/run_qat_experiments.sh`](scripts/run_qat_experiments.sh) — run matrix, ablation, and seed experiments into `runs/modelopt_qat_experiments/`.
* [`scripts/bench_trt.sh`](scripts/bench_trt.sh) — single-ONNX `trtexec` wrapper with INT8 and FP16 fallback.
* [`scripts/measure_speedup.sh`](scripts/measure_speedup.sh) — FP32-vs-INT8 TensorRT speedup table. Auto-exports FP32 ONNX if missing. Use `--ao` for the PyTorch AO export.
* [`scripts/measure_accuracy.sh`](scripts/measure_accuracy.sh) — validate exported ONNX, and optionally TensorRT engines, against the PyTorch-checkpoint mAP. Use `--ao` for the PyTorch AO export.

### Configs and data

* [`configs/coco.yaml`](configs/coco.yaml) — Ultralytics-format COCO config pointing at `./datasets/coco`.
* [`configs/requirements.txt`](configs/requirements.txt) — pinned dependency list with ModelOpt and ONNX export extras.

### Tests

* [`tests/test_qat_helpers.py`](tests/test_qat_helpers.py) — unit tests for QAT helpers, including recipe resolution, amax drift, DFL discovery, and seed pinning.
* [`tests/test_ptq_helpers.py`](tests/test_ptq_helpers.py) — unit tests for PTQ helpers.

### Documentation

* [`docs/CUSTOM_DATASET_QUANTIZATION.md`](docs/CUSTOM_DATASET_QUANTIZATION.md) — QAT/PTQ quantization on custom Ultralytics datasets.
* [`docs/LINEAR_QUANTIZATION_THEORY.md`](docs/LINEAR_QUANTIZATION_THEORY.md) — background on linear quantization math.
* [`docs/readme_python3.12_on_old_ubuntu_version.md`](docs/readme_python3.12_on_old_ubuntu_version.md) — Python 3.12 via Deadsnakes PPA.

## External references

### NVIDIA ModelOpt

* [Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
* [PyTorch quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
* [`cnn_qat` example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat)
* [Model Optimizer docs](https://docs.nvidia.com/deeplearning/modelopt/)

### YOLO QAT references

* [NVIDIA-AI-IOT YOLO DeepStream QAT examples](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat)
* [UBON QAT notes for Ultralytics](https://github.com/ubonpartners/ultralytics/blob/94046a6b2ee10c00281fd11519c576ebf5b3895e/UBON_QAT.md)

### Ultralytics

* [Ultralytics repository](https://github.com/ultralytics/ultralytics)
* [Ultralytics documentation](https://docs.ultralytics.com/)
* [Training mode reference](https://docs.ultralytics.com/modes/train/)

### Datasets and deployment

* [COCO dataset](https://cocodataset.org/)
* [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
