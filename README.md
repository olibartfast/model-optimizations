# Model Optimizations

A workspace for deep-learning model optimization workflows focused on reproducible INT8 PTQ and INT8 QAT for Ultralytics YOLO.

The repository is organized around practical optimization workflows: quantization, export validation, benchmarking, helper scripts, tests, and documentation. The most developed workflow today is the **YOLO INT8 quantization** stack, which includes NVIDIA ModelOpt and PyTorch AO pipelines for PTQ/QAT, ONNX Q/DQ export, and TensorRT benchmarking.

Current YOLO validation covers **`yolo11s`** and **`yolo26s`** on COCO `val2017`. Other model families or YOLO variants may work with the same infrastructure, but they should be treated as unvalidated until tested.

## Repository scope

This repository is intended to grow as a collection of model-optimization utilities and experiments, not as a YOLO-only project.

Current areas:

* quantization pipelines;
* ONNX export and validation;
* TensorRT latency benchmarking;
* reproducible setup scripts;

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

ModelOpt QAT exports include real Conv Q/DQ coverage. The current metrics and commands live in [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

The freshest end-to-end fixed-export results are on COCO `val2017` with `conf=0.001`, `iou=0.6`, and `imgsz=640` on an RTX 3060 Laptop GPU. TensorRT latency was measured at batch 1 with TensorRT 10.13.3.

| Model       | Pipeline                | Conv Q/DQ | mAP50-95 | Delta vs PyTorch FP32 | TRT latency | vs FP16 |
| ----------- | ----------------------- | --------: | -------: | --------------------: | ----------: | ------: |
| **yolo11s** | PyTorch FP32 baseline   |         - |   0.4622 |              baseline |     6.05 ms |   0.28x |
|             | FP16 baseline           |         - | ~0.4592 ONNX |           -0.0030 |     2.16 ms |   1.00x |
|             | ModelOpt INT8 QAT       |     87/88 |   0.4470 |               -0.0153 |     1.83 ms |   1.18x |
| **yolo26s** | PyTorch FP32 baseline   |         - |   0.4718 |              baseline |     6.24 ms |   0.36x |
|             | FP16 baseline           |         - | ~0.4717 ONNX |           -0.0001 |     2.23 ms |   1.00x |
|             | ModelOpt INT8 QAT       |   102/102 |   0.4700 |               -0.0018 |     2.60 ms |   0.86x |
|             | ModelOpt INT8 PTQ       |   102/102 |   0.4706 |               -0.0012 |     2.60 ms |   0.86x |

Current interpretation:
* **yolo11s**: The fixed ModelOpt export gives a real INT8 throughput gain (1.18x) over FP16 on this GPU, but the `yolo11-distill` QAT recipe loses about 0.012 mAP50-95 versus the ONNX FP32 baseline.
* **yolo26s**: The fixed ModelOpt QAT export achieves near-lossless INT8 accuracy, matching within **-0.0017 mAP50-95** of the ONNX FP32 baseline and **-0.0018 mAP50-95** of the PyTorch FP32 baseline. At batch size 1, the fully quantized network (102/102 convs + residual-adds) runs slower than clean FP16 on this GPU due to TensorRT scheduling/reformat overhead.

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
  --models yolo26s --qat-recipe auto \
  --qat-log-every 20 --qat-eval-every 5 --seed 0
```

Run PTQ-only ONNX export:

```bash
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x \
  --quant-modes int8 \
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

The ModelOpt QAT entry point provides several named recipes. The default `--qat-recipe auto` selects the accuracy recipe by model family.

### `yolo26-distill`

Tested on YOLO26 E2E dual-head models.

The recipe performs PTQ histogram calibration on COCO images, then fine-tunes with QAT using a co-aligned `one2one` teacher-student distillation loss combined with the standard COCO supervised detection loss.

### `yolo11-distill`

Tested on YOLO11 single-head models.

The recipe is designed for the tested YOLO11 setup and uses the same distillation plus supervised fine-tuning schedule as the YOLO26 path. It also applies two PTQ layer exclusions:

* the DFL head, because its 16-bin probability distribution is fragile under INT8 quantization;
* the Detect head output quantizers, because of their wide dynamic range.

In the tested YOLO11 setup, these exclusions raise the PTQ baseline. Without them, single-head QAT on the fully quantized graph regresses below PTQ.

## Accuracy by stage: PTQ vs QAT

Current fixed-export ModelOpt metrics are tracked in [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md).

### `yolo11s` (with `yolo11-distill` recipe)

| Stage | mAP50 | mAP50-95 | Notes |
| ----- | ----: | -------: | ----- |
| PyTorch FP32 | 0.6374 | 0.4622 | Baseline checkpoint validation. |
| ONNX FP32 | 0.6336 | 0.4592 | Export baseline used for ONNX delta. |
| PTQ fake quant | 0.6220 | 0.4475 | 1024-image `max` calibration. |
| QAT fake quant | 0.6263 | 0.4493 | 10 epochs x 200 batches, batch 8. |
| QAT INT8 ONNX | 0.6242 | 0.4470 | `Conv Q/DQ=87/88`. |

### `yolo26s` (with `yolo26-distill` recipe)

| Stage | mAP50 | mAP50-95 | Notes |
| ----- | ----: | -------: | ----- |
| PyTorch FP32 | 0.6385 | 0.4718 | Baseline checkpoint validation. |
| ONNX FP32 | 0.6362 | 0.4717 | Export baseline used for ONNX delta. |
| PTQ fake quant | 0.6368 | 0.4706 | 260-image `entropy` calibration. |
| QAT fake quant | 0.6370 | 0.4706 | 1 epoch distill QAT check. |
| QAT INT8 ONNX | 0.6354 | 0.4700 | `Conv Q/DQ=102/102`. |

Pre-fix ModelOpt tables that reported near-lossless QAT with slower-than-FP16 TensorRT are intentionally omitted here because those ONNX exports had no quantized convolutions.

## TensorRT speedup

Measured at batch 1 and `imgsz=640` on an RTX 3060 Laptop GPU with `scripts/measure_speedup.sh <model> --iterations 200 --warmup 1000`:

### `yolo11s`

| Engine | Mean latency | Throughput | vs FP16 |
| ------ | -----------: | ---------: | ------: |
| FP32 | 6.0473 ms | 165.3 FPS | 0.28x |
| FP16 baseline | 2.1614 ms | 462.3 FPS | 1.00x |
| Fixed ModelOpt QAT INT8 | 1.8337 ms | 544.8 FPS | 1.18x |

### `yolo26s`

| Engine | Mean latency | Throughput | vs FP16 |
| ------ | -----------: | ---------: | ------: |
| FP32 | 6.2365 ms | 160.3 FPS | 0.36x |
| FP16 baseline | 2.2317 ms | 447.8 FPS | 1.00x |
| Fixed ModelOpt QAT INT8 | 2.5955 ms | 385.0 FPS | 0.86x |

## Export accuracy: ONNX vs FP32 baseline

Q/DQ node counts verify graph topology, but not numerical fidelity. Exported ONNX models are validated against the PyTorch checkpoint baseline through Ultralytics ONNX Runtime backend using `scripts/measure_accuracy.sh <stem>`, with `--ao` for the AO export. For ModelOpt results, also check Conv Q/DQ coverage; Q/DQ node count alone is not enough.

## PyTorch AO quantization details

The PyTorch AO path uses symmetric INT8 quantization:

* activations: `qint8`, `per_tensor_symmetric`, zero-point 0;
* weights: per-channel symmetric INT8.

TensorRT rejects asymmetric `QuantizeLinear` nodes with non-zero zero-points on GPU, so the symmetric scheme is required for the exported Q/DQ ONNX graph to build as an INT8 engine.

Two recipe choices materially improve AO accuracy, both found by PTQ-only ablations on `yolo26s`:

* **Detect head kept FP32** (`--keep-detect-head-fp32`, the default for both AO recipes). AO's symmetric-int8 quantizers on the detect-head prediction towers cost ≈0.013 mAP50-95; keeping the whole `Detect` module FP — backbone and neck stay INT8 — recovers it. This is the dominant fix: it took `yolo26s` AO PTQ from 0.4523 to 0.4655.
* **Histogram activation observers** for both recipes. `HistogramObserver` clips activation outliers far better than `MovingAverageMinMaxObserver` (+0.0075 mAP50-95 at PTQ for `yolo11s`; histogram already won for `yolo26s`).

AO exports carry real Conv Q/DQ coverage, but on the tested RTX 3060 Laptop GPU the AO INT8 engines matched or trailed FP16 batch-1 latency while losing about 1% mAP50-95. Keeping the Detect head in FP32 also caps AO TensorRT speedup below the fixed ModelOpt INT8 path. Use AO mainly as a non-ModelOpt recipe comparison, not as the deployable default here.

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
    nvidia_modelopt_yolo.py            # PTQ-only ONNX pipeline: INT8
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
* [`yolo_quantization/ptq/nvidia_modelopt_yolo.py`](yolo_quantization/ptq/nvidia_modelopt_yolo.py) — PTQ-only ONNX pipeline for INT8.
* [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md) — active YOLO QAT experiment log, recipe table, and design decisions.
* [`AGENTS.md`](AGENTS.md) — agent guide for environment setup, common commands, and architecture notes.

### Helper scripts

* [`scripts/run_modelopt_yolo.sh`](scripts/run_modelopt_yolo.sh) — thin wrapper that picks `qat` or `ptq` and forwards arguments.
* [`scripts/run_venv.sh`](scripts/run_venv.sh) — create `$QUANTIZATION_VENV` from `configs/requirements.txt` with the NGC extra index.
* [`scripts/cloud_bootstrap.sh`](scripts/cloud_bootstrap.sh) — one-shot clone, virtualenv, and COCO setup for Colab, RunPod, or AWS.
* [`scripts/download_coco_dataset.sh`](scripts/download_coco_dataset.sh) — fetch and unpack COCO 2017 val/test/annotations.
* [`scripts/bench_trt.sh`](scripts/bench_trt.sh) — single-ONNX `trtexec` wrapper with INT8 and FP16 fallback.
* [`scripts/measure_speedup.sh`](scripts/measure_speedup.sh) — FP32-vs-INT8 TensorRT speedup table. Auto-exports FP32 ONNX if missing. Use `--ao` for the PyTorch AO export and `--run-name` for alternate run directories.
* [`scripts/measure_accuracy.sh`](scripts/measure_accuracy.sh) — validate exported ONNX, and optionally TensorRT engines, against the PyTorch-checkpoint mAP. Use `--ao` for the PyTorch AO export and `--run-name` for alternate run directories.

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
