# YOLO26-small INT8 QAT Work Log

Procedure adapted from [NVIDIA-AI-IOT/yolo_deepstream/yolov7_qat](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat):

1. PTQ histogram calibration on COCO images.
2. QAT fine-tune with combined COCO supervised detection loss and teacher-student MSE on raw detect outputs.
3. Evaluate mAP at FP32, PTQ-INT8, and QAT-INT8 stages on COCO `val2017`.
4. Export ONNX with explicit Q/DQ nodes when `--no-export` is not used.

Target model: `yolo26s` (Ultralytics YOLO26, released 2026-01-14).
Hardware: NVIDIA GeForce RTX 3060 Laptop GPU.

---

## Current Baseline

The active baseline uses the full COCO `train2017` image pool so the QAT dataloader can sample from the same distribution scale as the NVIDIA examples.

| Item | Current value |
|---|---:|
| Train image pool | 118,287 images |
| Validation set | 5,000 images |
| Train labels present | 117,266 label files |
| Validation labels present | 4,952 label files |
| PTQ calibration budget | 260 images = 26 batches x batch 10 |
| QAT budget | 20,000 train presentations = 10 epochs x 200 batches/epoch x batch 10 |
| Image size | 640 |
| Calibration method | `entropy` |
| QAT mode | `distill` |
| Export for metric runs | disabled with `--no-export` |

Why 260 and 20,000: the NVIDIA yolov7_qat/cuDLA QAT scripts use batch 10, `num_batch=25` in a loop that processes batches 0..25, and 10 fine-tune epochs of 200 iterations. That means 260 calibration images and 20,000 train image presentations, not 118,287 train presentations.

---

## Current Run

Completed command:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --calib-size 260 \
  --calib-method entropy \
  --qat-mode distill \
  --qat-epochs 10 \
  --qat-batches-per-epoch 200 \
  --batch 10 \
  --val-batch 8 \
  --workers 4 \
  --device 0 \
  --skip-fp32-eval \
  --no-export
```

Expected output files:

```text
runs/modelopt_qat/yolo26s/
  yolo26s_ptq.pth
  yolo26s_qat.pth
  summary.json
```

`yolo26s_qat.onnx` is intentionally not produced by this metric run because `--no-export` is set.

---

## Current Metrics

Metrics are from `runs/modelopt_qat/yolo26s/summary.json` for the full-source rerun, except FP32, which was not rerun in the metric command.

| Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 | Δ mAP50-95 vs PTQ | Notes |
|---|---:|---:|---:|---:|---|
| FP32 | 0.6384 | 0.4718 | baseline | - | Previously measured with the same val params. |
| PTQ INT8 | 0.6368 | 0.4706 | -0.0012 | baseline | Fresh PTQ from 260-image entropy calibration. |
| QAT INT8 | 0.6370 | 0.4701 | -0.0017 | -0.0005 | 10 x 200 x batch-10 full-source QAT. |

Raw saved values:

```text
PTQ: mAP50=0.6368257036873418, mAP50-95=0.47062866593006675, val=64.5s
QAT: mAP50=0.6369914336897926, mAP50-95=0.47010458539764405, val=66.1s
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Eval set | `val2017` | COCO `test2017` has no public labels, so local mAP is not possible. |
| QAT training data | Full `train2017` image pool | Uses the full source distribution while keeping the same 20,000-presentation NVIDIA fine-tune budget. |
| Calibration data | `val2017` | Matches the existing script and NVIDIA-style reference shape. PTQ amax estimation is statistics-only, but a train-held-out calibration split would be cleaner for a final benchmark. |
| Calibration algorithm | `entropy` | yolov7_qat uses Histogram(MSE); ModelOpt CLI choices are `max`, `entropy`, `percentile`, and `smoothquant`, so `entropy` is the closest built-in histogram path. |
| QAT mode | `distill` | Combines COCO supervised loss with teacher-student MSE, matching the intended NVIDIA supervision pattern while preserving ModelOpt quantizers. |
| QAT budget | 10 epochs x 200 batches/epoch x batch 10 | Matches the effective NVIDIA script budget: 20,000 training presentations. |

---

## Implementation Notes

`yolo_quantization/qat/nvidia_modelopt_yolo_qat.py` is the canonical entry point. The `examples/` copy is older and should not be used for resumed YOLO26 QAT unless fixes are mirrored.

Important QAT fixes already in the canonical script:

1. Residual-add quantizers are patched into Ultralytics residual blocks for clean Q/DQ export.
2. ModelOpt inference-mode `_amax` buffers are cloned before autograd.
3. `--from-ptq` restores ModelOpt checkpoints with local residual-add `_amax` buffers.
4. YOLO26 train-mode dict outputs are co-aligned to the `one2one` head for distillation.
5. Supervised COCO loss uses `E2ELoss.one2one.loss(...)` to avoid the empty `one2many` branch on restored PTQ students.

Avoid `--qat-mode ultralytics`: `YOLO.train()` rebuilds the model from YAML and drops ModelOpt quantizer modules.

## Recipes

`--qat-recipe` selects a bundle of hyperparameters. The default `auto` picks
per model family (yolo26* → `yolo26-distill`; everything else →
`yolo11-distill`). Explicit CLI flags always win over recipe defaults.

| Recipe | Targets | LR ladder | Loss | Calibration | Detect output Q | DFL Q | Epochs |
|---|---|---|---|---|---|---|---|
| `yolo26-distill` | E2E dual-head (yolo26*) | 1e-5 peak / 1e-6 low | distill (1.0) + supervised (1.0) | entropy, 260 imgs | enabled | enabled | 10 |
| `yolo11-distill` | Single-head (v8/v11/…) | 1e-5 peak / 1e-6 low | distill (1.0) + supervised (1.0) | **max, 1024 imgs** | **disabled** | **disabled** | 10 |
| `yolo11-supervised` | Single-head, larger PTQ-FP32 gap (rare) | 1e-4 peak / 1e-5 low | supervised-only (1.0) | max, 1024 imgs | disabled | disabled | 5 |

Why the single-head recipe differs — what "PTQ layer exclusions" actually means:

- **DFL (Distribution-Focal-Loss) head kept FP** — the DFL head learns a
  16-bin probability distribution per box coordinate and integrates it back
  to a continuous regression value. INT8 quantization of those bin
  probabilities introduces ~1% noise per bin, and the integration amplifies
  it across the 16 bins, producing wrong box centers and sizes. Excluding
  the DFL conv keeps box regression accurate while the rest of the network
  stays INT8. yolo26 uses an E2E head where `model.23.dfl` is an `Identity`
  no-op, so this flag is a no-op there — `--qat-recipe auto` only enables
  it for single-head models.
- **Detect-head output quantizers disabled** — the final Detect head
  outputs (raw class logits and decoded box tensors) have a much wider
  dynamic range than mid-network features. Per-tensor INT8 calibration
  clamps the tails, suppressing both rare-class confidence and large-box
  coordinates. Disabling these quantizers leaves the head in FP without
  affecting the preceding backbone/neck INT8.
- **`max` calibration** — empirically gives a higher PTQ baseline for
  single-head models (yolo11s PTQ: 0.4614 with max vs 0.4603 with entropy).
- **`yolo11-supervised` was tried first** and *collapsed* yolo11s mAP at
  lr=1e-4: epoch 2 dropped to mAP50-95=0.3807. Kept as an option for cases
  where the PTQ-FP32 gap is large enough to justify supervised-only
  training, but `yolo11-distill` is recommended.

The DFL-exclusion technique is documented in the
[`UBON_QAT.md`](https://github.com/ubonpartners/ultralytics/blob/94046a6b2ee10c00281fd11519c576ebf5b3895e/UBON_QAT.md)
reference in the ubonpartners/ultralytics fork, but the underlying observation
is independent of that fork — DFL's bin-probability semantics are why INT8
hurts it.

---

## Future Roadmap

- INT4 QAT exploration: keep the production baseline on INT8 QAT until INT4 has its own smoke-tested path. Start with ModelOpt torch `INT4_AWQ_CFG` or `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` as weight-only INT4 fine-tuning, then consider activation INT4 only after the weight-only path is stable.
- Required smoke checks before any full COCO run: PTQ/restore succeeds, one-batch backward works, mAP is sane on a small validation slice, and ONNX export/deployability is verified. Do not treat activation+weight INT4 as ready or expected to work until those checks pass.

---

## Commands

Choose, create, and activate a personal venv directory before running the
commands below:

```bash
python3.12 -m venv <venv-dir>
source <venv-dir>/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt
```

Verify the full dataset:

```bash
test -f yolo26s.pt
find datasets/coco/images/train2017 -maxdepth 1 -name '*.jpg' | wc -l   # expect 118287
find datasets/coco/images/val2017 -maxdepth 1 -name '*.jpg' | wc -l     # expect 5000
wc -l datasets/coco/train2017.txt datasets/coco/val2017.txt
```

Run the full-source metric experiment:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --calib-size 260 \
  --calib-method entropy \
  --qat-mode distill \
  --qat-epochs 10 \
  --qat-batches-per-epoch 200 \
  --batch 10 \
  --val-batch 8 \
  --workers 4 \
  --device 0 \
  --skip-fp32-eval \
  --no-export
```

Resume QAT from a saved PTQ checkpoint:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --qat-mode distill \
  --qat-epochs 10 \
  --qat-batches-per-epoch 200 \
  --batch 10 \
  --val-batch 8 \
  --workers 4 \
  --device 0 \
  --skip-fp32-eval \
  --no-export
```

Add structured training logs and intermediate eval to either command above:

```bash
  --qat-log-every 20    # heartbeat every 20 batches: [qat/distill] step k/B ...
  --qat-eval-every 5    # COCO val every 5 epochs; saves yolo*_qat_best.pth on improvement
  --seed 0              # pin Python/Numpy/Torch RNG for reproducibility
```

Each run now writes `runs/modelopt_qat/<model>/qat_train.csv` with per-epoch
`lr, sup, sup_box, sup_cls, sup_dfl, mse, total, secs, amax_count, amax_mean,
amax_rel_drift, amax_max_rel_drift` (and `map50/map/eval_secs` whenever
`--qat-eval-every` fires). The same rows land in `summary.json['qat_train']`
alongside `best_checkpoint`, `best_map`, `best_epoch`, and a `config` block.

Read final metrics:

```bash
cat runs/modelopt_qat/yolo26s/summary.json
cat runs/modelopt_qat/yolo26s/qat_train.csv
cat runs/modelopt_qat/report.json
```

Selective dequantization (disable quantizers on modules surfaced by the
sensitivity sweep):

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --keep-fp32-modules model.16.m.0.m.0 model.19.m.0.m.0 \
  --skip-fp32-eval --no-export
```

Calibrate on a held-out `train2017` slice to avoid the val/eval leak called
out in caveat #1:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s --calib-source train2017 --calib-size 260
```

Run against a custom Ultralytics detection dataset:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --calib-source train \
  --calib-size 260
```

`--data` is passed to Ultralytics for validation and QAT training. The YAML
should provide the usual `path`, `train`, `val`, and `names` fields.
`--calib-source` accepts `train`, `val`, legacy COCO aliases
`train2017`/`val2017`, or a direct path to an image directory or image-list
text file. For custom datasets, using `train` for calibration avoids calibrating
on the validation split that is later used for mAP.

Run the runtime experiment matrices (each subrun copies its outputs into
`runs/modelopt_qat_experiments/<mode>/<tag>/` so the canonical tree is not
overwritten):

```bash
./scripts/run_qat_experiments.sh matrix       # yolo26{n,s,m,l,x} + yolo11x
./scripts/run_qat_experiments.sh ablation     # --disable-detect-output-quant on/off
./scripts/run_qat_experiments.sh seeds        # multi-seed variance on yolo26s
DRY_RUN=1 ./scripts/run_qat_experiments.sh seeds   # preview the commands
```

Benchmark an exported `*_qat.onnx` through TensorRT (`trtexec` must be on PATH):

```bash
./scripts/bench_trt.sh runs/modelopt_qat/yolo26s/yolo26s_qat.onnx
```

Run helper tests:

```bash
python -m pytest tests/ -q
```

---

## Known Caveats

1. Calibration still uses `val2017`, which is also the eval set. For a final paper-quality benchmark, use a held-out `train2017` calibration slice.
2. The script uses `entropy`, not NVIDIA's Histogram(MSE), because ModelOpt does not expose an MSE calibrator through the current CLI path.
3. Export is separate from the metric run when `--no-export` is set. Re-run without `--no-export` or add an export-only helper before relying on ONNX artifacts.
4. Detect-head output quantization defaults to enabled. If PTQ collapses, retry with `--disable-detect-output-quant`.
5. The corrupt partial archive was moved aside as `datasets/coco/train2017.zip.bad`; delete it only after confirming no further recovery is needed.

---

## Dataset Layout

```text
datasets/coco/
  annotations/instances_val2017.json
  images/
    train2017/   118287 jpg
    val2017/       5000 jpg
  labels/
    train2017/   117266 txt
    val2017/       4952 txt
  train2017.txt
  val2017.txt
  test-dev2017.txt
```
