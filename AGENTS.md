# AGENTS.md

This file provides guidance to coding agents (Claude Code, etc.) when working with code in this repository.

> **Read this first:** [`yolo_quantization/qat/README.md`](yolo_quantization/qat/README.md) — current state of the YOLO26 INT8 QAT work, latest FP32/PTQ/QAT metrics, and copy-paste resume commands. Re-read it before changing the QAT pipeline or reporting new numbers, and update it when those change.

## What this repo is

Research scripts for **INT8 / FP8 / INT4 quantization of Ultralytics YOLO**
detectors (YOLO11x, YOLO26x, YOLO26s, etc.) via **NVIDIA ModelOpt**, evaluated
against **COCO `val2017`** by default. The canonical pipelines also accept
custom Ultralytics detection dataset YAMLs via `--data`. Two pipelines:

- **PTQ** (post-training, ONNX-based) — `yolo_quantization/ptq/nvidia_modelopt_yolo.py`
- **QAT** (post-training calibration + fine-tune, torch-based) — `yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`

The latter is the active development surface (YOLO26 INT8 QAT).

## Environment

Create a Python 3.12 virtual environment before running quantization:

```bash
python3.12 -m venv <venv-dir>
source <venv-dir>/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt
```

Equivalent convenience wrapper:

```bash
export QUANTIZATION_VENV=<venv-dir>
./scripts/run_venv.sh
```

The required details are `--no-build-isolation` and NVIDIA's NGC extra index
for `nvidia-modelopt[torch]`. The working install includes `pulp` and
`huggingface_hub`, both required by `modelopt.torch.export`.

After activation, use `python` for repo commands:

```bash
python <script>
```

Dependency list: `configs/requirements.txt`. ModelOpt must be installed with
`--no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com 'nvidia-modelopt[torch]'`.

## Dataset layout (Ultralytics)

`configs/coco.yaml` points at `./datasets/coco` with:

```
datasets/coco/
├── images/{train2017,val2017}/*.jpg     # train2017 = 118287; val2017 = 5000
├── labels/{train2017,val2017}/*.txt
├── annotations/instances_val2017.json
└── train2017.txt val2017.txt test-dev2017.txt
```

`scripts/download_coco_dataset.sh` fetches the raw zips into `datasets/coco/`.
The current workspace already has the full `train2017` and `val2017` image
layout in place; see `yolo_quantization/qat/README.md` for the active QAT
data budget.

Custom datasets use the normal Ultralytics detection YAML format:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class_name
```

Both canonical entry points accept `--data <dataset.yaml>`. `--calib-source`
accepts `train`, `val`, legacy COCO aliases `train2017`/`val2017`, or a direct
path to an image directory or newline-delimited image list.

## Common commands

Run from repo root.

**Tests** (helper tests for the QAT/PTQ scripts; no GPU required):
```bash
python -m pytest tests/ -q
python -m pytest tests/test_qat_helpers.py::test_distill_epoch_lr_uses_low_high_low_schedule -q
```

**Run QAT (full pipeline: FP32 eval → PTQ calibrate → QAT distill → eval → ONNX)**:
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --qat-recipe auto \
  --qat-log-every 20 --qat-eval-every 5 --seed 0
```

`--qat-recipe auto` (default) picks `yolo26-distill` for E2E yolo26 models and
`yolo11-distill` for single-head models (YOLOv8 / YOLO11 / etc.). The
single-head recipe **excludes DFL and Detect-head output quantizers** and
uses `max` calibration; without those, single-head QAT regresses below PTQ.
See `yolo_quantization/qat/README.md` for the full recipe table.

`--qat-log-every N` emits a `[qat/distill] step k/B` heartbeat every N batches.
`--qat-eval-every N` runs an Ultralytics validation pass every N epochs (and on
the final one), saving `*_qat_best.pth` whenever mAP50-95 improves. `--seed N`
pins the RNGs for reproducibility. Per-epoch training metrics — `lr, sup, sup_box, sup_cls,
sup_dfl, mse, secs, amax_rel_drift, map50, map` — are persisted to
`runs/modelopt_qat/<model>/qat_train.csv` and mirrored under
`summary.json['qat_train']`.

**Resume QAT from a saved PTQ checkpoint** (skips re-calibration):
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --skip-fp32-eval --qat-mode distill --qat-epochs 10 \
  --qat-batches-per-epoch 200 --batch 10
```

**Per-module PTQ sensitivity sweep** (subcommand of the same script):
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  sensitivity --models yolo26s --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth
```

**Selective dequantization** — disable quantizers on specific modules
identified by the sensitivity sweep:
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --keep-fp32-modules model.16.m.0.m.0 model.19.m.0.m.0 --skip-fp32-eval
```

**Calibration source** — use a held-out train2017 slice instead of val2017 to
avoid the calibration/eval leak:
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --calib-source train2017 --calib-size 260
```

**Custom dataset** — train/evaluate with a non-COCO Ultralytics dataset YAML:
```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --data configs/my_dataset.yaml --calib-source train
```

**Custom dataset PTQ-only ONNX**:
```bash
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x --data configs/my_dataset.yaml --calib-source train \
  --quant-modes int8
```

**Experiment driver** (`scripts/run_qat_experiments.sh`) runs the recipe across
multiple variants and stashes outputs under `runs/modelopt_qat_experiments/`:

```bash
DRY_RUN=1 ./scripts/run_qat_experiments.sh seeds      # preview the 3-seed plan
./scripts/run_qat_experiments.sh matrix               # yolo26{n,s,m,l,x} + yolo11x
./scripts/run_qat_experiments.sh ablation             # --disable-detect-output-quant on/off
./scripts/run_qat_experiments.sh seeds                # 3-seed variance for yolo26s
```

**TensorRT runtime benchmark** of an exported `*_qat.onnx`:

```bash
./scripts/bench_trt.sh runs/modelopt_qat/yolo26s/yolo26s_qat.onnx
```

**Cloud bootstrap** (Colab / RunPod / AWS) — one-shot clone + venv + COCO:

```bash
bash scripts/cloud_bootstrap.sh
# Or remotely without cloning first:
# bash <(curl -fsSL <repo-raw>/scripts/cloud_bootstrap.sh)
```

**Run PTQ-only ONNX pipeline**:
```bash
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x --quant-modes int8 fp8 --calib-size 256
```

Outputs land in `runs/modelopt_qat/<model>/` (QAT: `*_ptq.pth`, `*_qat.pth`,
`*_qat.onnx`, `summary.json`) and `runs/modelopt/` (PTQ).

## Architecture you need to know before editing

### Single pipeline entry points

There is only one supported copy of each quantization pipeline:

- `yolo_quantization/qat/nvidia_modelopt_yolo_qat.py` — torch-based INT8 QAT.
- `yolo_quantization/ptq/nvidia_modelopt_yolo.py` — ONNX-based PTQ.

Do not add duplicate runnable copies under `examples/` or elsewhere. If a new
workflow needs a wrapper, keep it thin and delegate to the canonical script.

### QAT pipeline shape (`nvidia_modelopt_yolo_qat.py`)

The flow is FP32 eval → PTQ calibrate (with `mtq.quantize` or the manual
`manual_quantize_model` path for `max`/`entropy`/`percentile`) → save with
`mto.save` → QAT fine-tune → `mto.save` → ONNX export. Three quirks that bite
when editing:

1. **Residual adds are patched** (`patch_residual_add_quantizers`) — raw
   tensor adds in Ultralytics `Bottleneck`, `GhostBottleneck`, `CIB`,
   `HGBlock`, `ResNetBlock`, `Residual` blocks are wrapped with
   `add_lhs_quantizer` / `add_rhs_quantizer`. This is necessary for clean
   Q/DQ ONNX export. Restoring a ModelOpt checkpoint with these patched
   modules requires `_restore_modelopt_checkpoint`, which re-creates the
   local `_amax` buffers that ModelOpt's `restore_from_modelopt_state` does
   not know about.

2. **Inference-mode buffers must be cloned before autograd**
   (`_clone_inference_buffers` + `_clone_quantizer_amax_state`). ModelOpt
   leaves some tensors in `torch.inference_mode`, which breaks backprop.
   Both PTQ-save and QAT-start call these.

3. **Detect-head output quantizers are NOT skipped by default.**
   `--disable-detect-output-quant` flips them off via
   `apply_detect_output_ignore_policy` (which uses ModelOpt's
   `set_quantizer_attribute` policy, not ad-hoc `.disable()` calls). YOLO mAP
   is sensitive to this; it is the first knob to try when PTQ collapses.

### QAT distillation loop (`run_distillation_qat`)

Default `--qat-mode distill` combines a **COCO supervised detection loss**
(`E2ELoss.one2one.loss(...)`) with a **per-tensor-normalized teacher-student
MSE** on the `one2one` head. Both terms are weighted independently via
`--qat-supervised-weight` and `--qat-distill-weight` (default 1.0 each).

The loop uses an Ultralytics `DetectionTrainer` only for the
dataloader/preprocessing. Teacher is intentionally `.train()` (not `.eval()`)
so YOLO26 emits its `one2many`/`one2one` dict outputs; both teacher and
student are co-aligned to `one2one` by `_coalign_distill_outputs` because the
restored quantized student has an empty `one2many` dict (some Detect-head
submodule is bypassed after PTQ).

Empirically (yolo26s, 10 epochs x 200 batches, batch 10, Adam @ 1e-5 with
low/high/low LR ladder), this recipe lands within ~0.001 mAP50-95 of PTQ.
See `yolo_quantization/qat/README.md` for the current metrics.

**Avoid `--qat-mode ultralytics`** — `YOLO.train()` rebuilds the model from
yaml and reloads weights, dropping every quantizer module. The resulting
"QAT" model collapses to mAP≈0. Kept as a code path only to preserve the
flag.

### Where to look for current metrics and known issues

`yolo_quantization/qat/README.md` tracks:
- per-stage mAP (FP32 / PTQ / QAT) for `yolo26s`,
- the active full-source QAT data and training budget,
- copy-paste resume commands tied to the on-disk checkpoint layout.

**Always update that doc when changing the QAT recipe or recording new metrics.**

## Conventions

- Always run scripts from the repo root; `PROJECT_ROOT` is computed from
  `__file__` and the scripts use `runs/`, `datasets/`, `configs/` paths
  relative to it.
- Each pipeline stage is **idempotent and resumable**: PTQ writes
  `*_ptq.pth`, QAT writes `*_qat.pth`, both via `mto.save`. Use `--from-ptq`
  rather than re-calibrating.
- Validation uses `conf=0.001 iou=0.6` consistently across stages so mAP is
  comparable.

## Keeping `docs/` and `scripts/` in sync

When you change a pipeline's entry point, CLI flags, dataset layout, default
recipe, or install procedure, update `docs/` and `scripts/` in the same
change — do not leave stale guidance behind. Specifically:

- **Entry-point or CLI changes** (anything in `yolo_quantization/qat/` or
  `yolo_quantization/ptq/`): refresh the top-level `README.md` snippets and
  `yolo_quantization/qat/README.md`'s "Winning command" / resume blocks. If
  `scripts/run_modelopt_yolo.sh` invokes the changed script, verify it still
  points at the canonical path and forwards the new flags correctly.
- **Install / venv / dependency changes**: update `configs/requirements.txt`
  *and* `scripts/run_venv.sh` together. Do not hardcode a local venv name;
  use the `QUANTIZATION_VENV=<venv-dir>` placeholder pattern in docs and
  scripts.
- **Dataset layout changes** (`configs/coco.yaml`, label/image directory
  shape): update `scripts/download_coco_dataset.sh` if the layout it produces
  no longer matches, and the layout block at the bottom of
  `yolo_quantization/qat/README.md`.
- **Deprecating a doc**: delete it rather than letting it drift. Update the
  `docs/` listing in `README.md` in the same commit so there are no dangling
  references. Do **not** add new tutorial-style docs that duplicate
  information already in `README.md`, `AGENTS.md`, or
  `yolo_quantization/qat/README.md`.
- **No duplicate pipeline scripts**: keep QAT/PTQ implementation changes in
  `yolo_quantization/{qat,ptq}/`. Wrappers may forward arguments to those
  scripts, but should not copy their implementation.
