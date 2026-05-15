# YOLO26-small INT8 QAT — Work In Progress

Procedure adapted from [NVIDIA-AI-IOT/yolo_deepstream/yolov7_qat](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat):

1. PTQ histogram calibration on COCO images.
2. QAT fine-tune with teacher–student supervision (MSE on raw detect outputs).
3. Evaluate mAP at three stages — **FP32**, **PTQ-INT8**, **QAT-INT8** — on COCO `val2017`.
4. Export ONNX with explicit Q/DQ nodes.

Target model: `yolo26s` (Ultralytics YOLO26, released 2026-01-14).
Hardware: NVIDIA GeForce RTX 3060 Laptop GPU.

---

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Eval set | `val2017` | COCO `test2017` has no public labels — local mAP impossible. |
| QAT training data | 4000-image subsample of `train2017` | Avoid downloading full 19 GB `train2017.zip`; avoid using `val2017` for both train and eval (leakage). |
| Calibration data | `val2017` (existing pipeline default) | Matches NVIDIA cnn_qat reference and current script. PTQ amax estimation is not gradient-based, so this is acceptable. |
| Calibration algorithm | `entropy` (histogram-based) | yolov7_qat uses Histogram(MSE); ModelOpt's CLI choices are `max / entropy / percentile / smoothquant` — no MSE built-in. `entropy` is the closest histogram option. |
| QAT mode | `distill` (teacher-student MSE on raw detect outputs) | Mirrors yolov7_qat's "Supervision" hook. |
| QAT budget | 2 epochs × 200 batches/epoch | Script default; ≈30–60 min on 3060 laptop. |
| Image size, batch | 640, 16 | Standard COCO YOLO settings. |

---

## State of the workspace

Last updated: 2026-05-15 18:04:41 CEST.

### Completed
1. **Ultralytics upgraded to 8.4.50** in `quantization_venv/` — confirms `yolo26s.pt` (and all `yolo26{n,s,m,l,x}` detect/seg/pose/obb/cls variants) are present in `GITHUB_ASSETS_NAMES`.
2. **COCO labels + manifests downloaded and extracted** into the Ultralytics layout:
   ```
   datasets/coco/
     LICENSE  README.txt
     annotations/instances_val2017.json
     labels/{train2017,val2017}/*.txt        # 117266 + 4952 label files
     train2017.txt  val2017.txt  test-dev2017.txt   # full path manifests
     images/val2017/*.jpg                    # 5000 images (moved from datasets/coco/val2017/)
     images/train2017/*.jpg                  # 4000-image deterministic subsample
   ```
3. **Train subsample selected and downloaded** (deterministic, seed=0) — 4000 train images are present under `datasets/coco/images/train2017`.
4. **`configs/coco.yaml` points at the Ultralytics layout**:
   - `path: ./datasets/coco`
   - `train: images/train2017`
   - `val: images/val2017`
5. **PTQ completed for `yolo26s`** and saved `runs/modelopt_qat/yolo26s/yolo26s_ptq.pth`.
   - Latest restored PTQ validation smoke: `mAP50=0.6370`, `mAP50-95=0.4706`, `114.8s` with `--val-batch 2`.
   - Cached summary before the latest fixes recorded `mAP50=0.6368497861948106`, `mAP50-95=0.4706139006701033`.
6. **QAT one-batch restore-and-distill smoke passed** from the existing PTQ checkpoint:
   ```
   [qat] cloned 26 quantizer amax state tensor(s) before training
   [qat/distill] epoch 1/1 lr=1.00e-05 mse=15.544082
   restore_distill_smoke_ok
   ```
7. **Code fixes verified** in `yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`:
   - clone inference-mode / quantizer `_amax` state before autograd sees it;
   - support `--from-ptq` resume for the main QAT command;
   - restore local residual-add quantizer `_amax` buffers from ModelOpt checkpoints;
   - handle YOLO26 train-mode dict outputs and fall back from empty `one2many` to `one2one`.
8. **Helper tests pass**:
   ```
   quantization_venv/bin/python -m pytest tests/test_qat_helpers.py -q
   # 11 passed
   ```
9. **Full resumed QAT completed** from `runs/modelopt_qat/yolo26s/yolo26s_ptq.pth`.
   - PTQ validation: `mAP50=0.6368497861948106`, `mAP50-95=0.4706139006701033`, `54.2s`.
   - QAT validation after 2 epochs x 200 batches: `mAP50=0.5910232507423059`, `mAP50-95=0.3826682713718743`, `57.4s`.
   - QAT training losses:
     ```
     [qat/distill] epoch 1/2 lr=1.00e-05 mse=6.601666
     [qat/distill] epoch 2/2 lr=1.00e-05 mse=5.253283
     ```
   - Saved checkpoint: `runs/modelopt_qat/yolo26s/yolo26s_qat.pth`.
   - ONNX export failed before dependency update because `modelopt.torch.export` could not import optional dependencies.
10. **Missing export dependencies added to the venv and requirements**:
    - `pulp` was already installed and listed.
    - Installed and listed `huggingface_hub`; after that, `import modelopt.torch.export` succeeds.

### Pending
- Re-run ONNX export now that `huggingface_hub` is installed.
- Decide whether to tune or replace the current distillation objective because QAT regressed below PTQ.
- Aggregate and report FP32 vs PTQ-INT8 vs QAT-INT8 mAP. FP32 was skipped in the resumed run.
- Mirror the latest package-script fixes into `examples/nvidia_modelopt_yolo_qat.py` if that entry point will remain supported for YOLO26 resumed QAT.
- Optional cleanup: old archives may still be present at `datasets/coco/{val2017.zip,test2017.zip,annotations_trainval2017.zip}`.

## Current Metrics

| Stage | mAP50 | mAP50-95 | Delta vs PTQ |
|---|---:|---:|---:|
| FP32 | not measured in resumed run | not measured in resumed run | n/a |
| PTQ INT8 | 0.6368 | 0.4706 | baseline |
| QAT INT8 | 0.5910 | 0.3827 | -0.0458 mAP50 / -0.0879 mAP50-95 |

The current `distill` recipe hurts accuracy. The QAT checkpoint is useful for debugging the training path, but it is not a better model than PTQ.

## QAT Drop Investigation

Observed behavior:

- The original autograd failure was fixed by cloning inference-mode / quantizer `_amax` state before training.
- YOLO26 returns train-mode dictionaries with `one2many` and `one2one` branches.
- The FP32 teacher has non-empty `one2many` and `one2one` outputs.
- The restored quantized student has an empty `one2many` dictionary and non-empty `one2one` outputs.
- The current normalizer falls back to `one2one` only when `one2many` has no tensors. That means the current distillation can compare teacher `one2many` tensors to student `one2one` tensors.

Most likely contributors to the drop:

1. **Branch mismatch in distillation target**: teacher `one2many` vs student `one2one` is probably not the intended pairing for YOLO26. A controlled test should force both teacher and student to `one2one` and compare against the current result.
2. **Raw MSE objective is unbalanced**: boxes, class scores, and feature maps have very different scales and counts. The feature maps dominate the loss unless weighted.
3. **No detection loss is mixed in**: this is pure teacher-student MSE, so it can optimize internal similarity while harming COCO mAP.
4. **Learning rate / schedule may be too aggressive for the quantized graph**: `1e-5` for 400 batches moved the model enough to lose ~0.088 mAP50-95.
5. **The train subset is small**: 4000 images are enough for a smoke/tuning pass, but not enough to conclude the QAT recipe is generally bad.

Recommended next experiments:

1. Force both teacher and student to distill `one2one` outputs only.
2. Distill only `scores` and `boxes` first, or add explicit weights such as low feature-map weight.
3. Lower LR to `1e-6` or `3e-6` and run 1 epoch x 200 batches.
4. Try `--qat-mode ultralytics` as a baseline using YOLO's detection loss.
5. Run a short sensitivity sweep on the QAT checkpoint to see whether a small set of quantizers is responsible for most of the loss.

---

## Commands to resume — copy/paste in order

Run all from the repo root: `cd /home/oli/repos/model-optimizations`.

### Step 1 — verify local dataset and PTQ checkpoint
```bash
test -f yolo26s.pt
test -f runs/modelopt_qat/yolo26s/yolo26s_ptq.pth
find datasets/coco/images/train2017 -maxdepth 1 -type f | wc -l   # expect 4000
find datasets/coco/images/val2017 -maxdepth 1 -type f | wc -l     # expect 5000
```

### Step 2 — quick QAT smoke from the existing PTQ checkpoint

This avoids another PTQ calibration pass and validates the exact path that previously failed.

```bash
quantization_venv/bin/python -c "from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from ultralytics import YOLO
import yolo_quantization.qat.nvidia_modelopt_yolo_qat as q
mto, mtq = q._import_modelopt_qat_modules()
yolo = YOLO('yolo26s.pt')
device = q._resolve_device('0')
yolo.model.to(device)
yolo.fuse()
yolo.model.to(device)
quant_cfg = q.build_quant_cfg(mtq, 'entropy', 99.99)
q.patch_residual_add_quantizers(yolo.model, quant_cfg)
yolo.model = q._restore_modelopt_checkpoint(mto, yolo.model, Path('runs/modelopt_qat/yolo26s/yolo26s_ptq.pth'), device)
q.run_distillation_qat(yolo.model, 'yolo26s.pt', imgsz=640, batch=2, device='0', epochs=1, peak_lr=1e-5, low_lr=1e-6, batches_per_epoch=1, workers=0)
print('restore_distill_smoke_ok')"
```

### Step 3 — run the full resumed QAT pipeline

Use the package script for now; it has the latest `--from-ptq` and YOLO26 dict-output fixes.

```bash
quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --calib-method entropy \
  --qat-mode distill \
  --qat-epochs 2 \
  --qat-batches-per-epoch 200 \
  --imgsz 640 \
  --batch 16 \
  --val-batch 8 \
  --workers 4 \
  --device 0 \
  --skip-fp32-eval
```

Outputs land under `runs/modelopt_qat/yolo26s/`:
- `yolo26s_ptq.pth`, `yolo26s_qat.pth` — ModelOpt checkpoints (reloadable via `mto.restore`).
- `yolo26s_qat.onnx` — deployable ONNX with explicit Q/DQ nodes (verified via `count_qdq_nodes`).
- `summary.json` — per-stage `map50`, `map50-95`, seconds, checkpoint paths.

Console will print a final report table with the three rows (`fp32 / ptq / qat`) and their mAP.

### Step 4 — read the final comparison
```bash
cat runs/modelopt_qat/yolo26s/summary.json
cat runs/modelopt_qat/report.json
```

### Step 5 — re-run ONNX export after installing dependencies

The full QAT run saved `yolo26s_qat.pth`, but export failed before `huggingface_hub` was installed. The import check now passes:

```bash
quantization_venv/bin/python -c "import pulp, huggingface_hub, modelopt.torch.export; print('modelopt export import ok')"
```

Re-run export through the QAT script or add an export-only helper before relying on the ONNX artifact.

---

## Mapping yolov7_qat → this pipeline

| yolov7_qat step | This pipeline equivalent |
|---|---|
| `bash scripts/get_coco.sh` | Ultralytics `coco2017labels.zip` + train-subsample fetch (above). |
| `python scripts/qat.py quantize yolov7.pt --ptq=ptq.pt --qat=qat.pt --eval-ptq --eval-origin` | Single run of `examples/nvidia_modelopt_yolo_qat.py` (does FP32 eval → PTQ calibrate → eval → QAT distill → eval). |
| Histogram(MSE) calibration | `--calib-method entropy` (closest available — ModelOpt has no MSE; `entropy` shares the same histogram backend). |
| Supervision (teacher-student) | `--qat-mode distill`: MSE on raw multi-scale detect outputs of teacher (FP32 frozen) vs student (quantized, trainable). |
| `bash scripts/eval-trt.sh qat.pt` | Built into the same script — uses Ultralytics' `model.val()` with `data=coco.yaml, conf=0.001, iou=0.6`. |
| `python scripts/qat.py export qat.pt --size=640 --save=qat.onnx --dynamic` | ModelOpt `get_onnx_bytes()` (or `yolo.export(format="onnx")`) — invoked automatically unless `--no-export` is passed. |

---

## Known caveats / things to revisit

1. **Calibration leak**: `val2017` is used for calibration *and* mAP eval. PTQ amax estimation is statistics-only (no gradients), so the typical leakage concern is mild — but for a rigorous benchmark, swap to a held-out slice of `train2017`.
2. **Calibrator mismatch with yolov7_qat**: yolov7_qat uses `mse` histogram; this run uses `entropy`. If results diverge from the yolov7_qat baseline, try `--calib-method percentile --calib-percentile 99.99` as a third reference point, or add an `mse` option by passing `module.load_calib_amax("mse")` in `manual_histogram_quantize`.
3. **Train subsample size**: 4000 images cover 2 epochs × 200 batches × 16 = 6400 sample-steps with 1.25× cycling. If `--qat-batches-per-epoch` or `--qat-epochs` is bumped, also bump the subsample.
4. **Old archive cleanup**: `datasets/coco/{val2017.zip, test2017.zip, annotations_trainval2017.zip}` (~1.8 GB) can be deleted; data is already extracted.
5. **Detect-head output quantization**: defaults to *enabled*. If PTQ mAP collapses, retry with `--disable-detect-output-quant` (the script handles this cleanly).
6. **Residual-add quantizer patching** is on by default; necessary for clean Q/DQ ONNX export on YOLO bottleneck blocks. Disable with `--no-residual-add-quant` only when debugging.
7. **Resumed PTQ validation is not optional yet**: `--from-ptq` currently restores and validates PTQ before QAT. With `--val-batch 8` the PTQ validation took ~54s on the RTX 3060 Laptop GPU.
8. **Entry-point divergence**: `yolo_quantization/qat/nvidia_modelopt_yolo_qat.py` has the latest resume and YOLO26 output fixes. `examples/nvidia_modelopt_yolo_qat.py` still needs a mirror pass before using it for resumed QAT.

---

## Quick reference — directory layout after Step 2

```
model-optimizations/
├── configs/coco.yaml                    ← updated in Step 3
├── examples/nvidia_modelopt_yolo_qat.py ← entry point
├── datasets/coco/
│   ├── images/
│   │   ├── train2017/   (4000 jpg, ~600 MB)
│   │   └── val2017/     (5000 jpg, ~780 MB)
│   ├── labels/
│   │   ├── train2017/   (117266 txt)   ← unused entries safely ignored
│   │   └── val2017/     (4952 txt)
│   ├── annotations/instances_val2017.json
│   ├── train2017.txt  val2017.txt  test-dev2017.txt
│   └── LICENSE  README.txt
└── runs/modelopt_qat/yolo26s/           ← created by Step 4
    ├── yolo26s_ptq.pth
    ├── yolo26s_qat.pth
    ├── yolo26s_qat.onnx
    └── summary.json
```
