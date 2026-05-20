# PyTorch AO QAT — Benchmark Roadmap (WIP)

Status: **yolo26s AO benchmark complete; AO remains below ModelOpt** (2026-05-20, branch `feature/torch-ao`).

The experimental fake-quant recipe at
[`pytorch_ao_yolo_qat.py`](pytorch_ao_yolo_qat.py) now has a machine-written
`yolo26s` FP32/PTQ/QAT result plus output-quant ablations. The current best AO
recipe disables Conv output fake-quant and uses histogram activation observers
for `yolo26s`: QAT reaches 0.4654 mAP50-95, still behind the same-split
ModelOpt PTQ result of 0.4708.

---

## Goal

Produce three result tables for the AO path that mirror the ModelOpt tables
in the root README, on the same hardware, dataset, and budget:

1. mAP on COCO `val2017` at FP32 → PTQ INT8 → QAT INT8.
2. TensorRT batch-1 latency vs FP32 baseline (only if AO ONNX export is
   TRT-loadable — see Risks).
3. ONNX export sanity (ort vs PT mAP delta) to detect calibration / Q-DQ
   topology breakage.

Apples-to-apples means: same models (`yolo26s`, `yolo11s`), same
`imgsz=640`, same calibration budget (260 images, batch 10), same QAT budget
(10 epochs × 200 batches × batch 10 = 20,000 train presentations), same
RTX 3060 Laptop GPU, same `conf=0.001 iou=0.6` eval thresholds.

---

## Milestones

### M0 — Environment sanity
- [x] Confirm GPU free memory ≥ ~6 GB before launching (QAT + teacher copy).
- [x] Confirm `torch.ao.quantization` import path works inside the repo venv.
- [x] Run the recipe end-to-end on **1 epoch × 20 batches** for `yolo26s`
      as a smoke test (catches API drift before committing to a 10-epoch run).

### M1 — yolo26s AO benchmark
- [x] Full PTQ + QAT run with `--ao-recipe auto` (resolves to `yolo26-ao-distill`).
- [x] Re-measure FP32 / PTQ-INT8 / QAT-INT8 mAP from machine-written artifacts after the `requires_grad` fix.
- [x] Save PTQ and QAT checkpoints under `runs/pytorch_ao_qat/yolo26s/`.

### M2 — yolo11s AO benchmark
- [ ] Full PTQ + QAT run with `--ao-recipe auto` (resolves to `yolo11-ao-distill`).
- [ ] Verify AO recipe handles the DFL + Detect-output exclusions correctly
      (the ModelOpt path requires these for single-head models — confirm the
      AO equivalent is wired up via `--exclude-dfl-quant` /
      `--disable-detect-output-quant`).

### M3 — ONNX export attempt
- [x] Drop `--no-export` on a re-run (or post-process the saved `.pth`).
- [ ] Inspect produced ONNX in Netron: count Q/DQ pairs, confirm they wrap
      the same Conv2d layers as the ModelOpt export.
- [x] Document the initial export failure mode and the rerun that produced
      `runs/pytorch_ao_qat/yolo26s/yolo26s_ao_qat.onnx`.

### M4 — TensorRT engine + latency
- [ ] Build INT8 engine via `scripts/bench_trt.sh <ao-stem>`.
- [ ] If build succeeds: `scripts/measure_speedup.sh <ao-stem>` to fill the
      latency table. If TRT rejects the Q/DQ graph: document and stop.

### M5 — Export-accuracy verification
- [ ] `scripts/measure_accuracy.sh <ao-stem>` to compare ONNX-INT8 vs ONNX-FP32
      ort mAP, same as the ModelOpt table.

### M6 — Publish
- [ ] Add the three AO result tables to the root README, directly below the
      ModelOpt tables, with a clear "PyTorch AO" attribution heading.
- [ ] Update [`README.md`](../../README.md) experimental subsection to drop
      the "no published numbers yet" caveat.

---

## Planned commands

Smoke test (M0):

```bash
python yolo_quantization/qat/pytorch_ao_yolo_qat.py \
  --models yolo26s --ao-recipe auto \
  --calib-source train2017 --calib-size 40 \
  --qat-epochs 1 --qat-batches-per-epoch 20 \
  --batch 10 --val-batch 8 --device 0 \
  --skip-fp32-eval --no-export
```

Full yolo26s run (M1):

```bash
python yolo_quantization/qat/pytorch_ao_yolo_qat.py \
  --models yolo26s --ao-recipe auto \
  --calib-source train2017 --calib-size 260 \
  --qat-epochs 10 --qat-batches-per-epoch 200 \
  --batch 10 --val-batch 8 --workers 4 --device 0 \
  --no-export
```

Full yolo11s run (M2) — same flags as yolo26s but with the single-head
exclusions; verify the AO recipe applies them automatically or pass
explicitly:

```bash
python yolo_quantization/qat/pytorch_ao_yolo_qat.py \
  --models yolo11s --ao-recipe auto \
  --calib-source train2017 --calib-size 260 \
  --qat-epochs 10 --qat-batches-per-epoch 200 \
  --batch 10 --val-batch 8 --device 0 \
  --exclude-dfl-quant --disable-detect-output-quant \
  --no-export
```

---

## Results so far

Filled in as runs complete. Δ columns are vs the FP32 row of the same model.

### mAP on COCO `val2017`

| Model | Recipe | Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---|---:|---:|---:|
| `yolo26s` | `yolo26-ao-distill` all Conv output Q | FP32 | 0.6385 | 0.4719 | baseline |
|           |                              | PTQ INT8 | 0.6200 | 0.4493 | -0.0226 |
|           |                              | QAT INT8 | 0.6200 | 0.4495 | -0.0224 |
| `yolo26s` | `yolo26-ao-distill` no Conv output Q | FP32 | 0.6385 | 0.4719 | baseline |
|           |                                      | PTQ INT8 | 0.6317 | 0.4619 | -0.0100 |
|           |                                      | QAT INT8 | 0.6324 | 0.4637 | -0.0082 |
| `yolo26s` | `yolo26-ao-distill` no Conv output Q + histogram (current default) | FP32 | 0.6385 | 0.4719 | baseline |
|           |                                                          | PTQ INT8 | 0.6316 | 0.4631 | -0.0088 |
|           |                                                          | QAT INT8 | 0.6335 | 0.4654 | -0.0065 |
| `yolo11s` | `yolo11-ao-distill` | FP32 | — | — | baseline |
|           |                     | PTQ INT8 | — | — | — |
|           |                     | QAT INT8 | — | — | — |

Raw saved all-Conv-output-quant `yolo26s` values from `runs/pytorch_ao_qat/yolo26s/summary.json`:

```text
FP32: mAP50=0.6384812606617837, mAP50-95=0.4718507598844442, val=46.0s
PTQ:  mAP50=0.6200075185205597, mAP50-95=0.44925470598981365, val=87.1s
QAT:  mAP50=0.6199956692953007, mAP50-95=0.44949609835523346, val=100.6s
```

Raw saved no-Conv-output-quant values from `/tmp/pytorch_ao_qat_no_conv_output_full/yolo26s/summary.json`:

```text
PTQ:  mAP50=0.6316523156505134, mAP50-95=0.4618733983330739, val=78.2s
QAT:  mAP50=0.632447310565266, mAP50-95=0.46369452440923326, val=85.7s
```

Raw saved no-Conv-output-quant + histogram values from `/tmp/pytorch_ao_qat_histogram_ptq/yolo26s/summary.json` and `/tmp/pytorch_ao_qat_histogram_full/yolo26s/summary.json`:

```text
PTQ:  mAP50=0.6316158283061234, mAP50-95=0.4630594795232257, val=78.4s
QAT:  mAP50=0.6334846163298964, mAP50-95=0.46538059020686426, val=85.4s
```

### yolo26s comparison vs ModelOpt

ModelOpt baseline values are from `yolo_quantization/qat/README.md`. A
separate PTQ-only same-split check also measured ModelOpt with `train2017`
entropy calibration under `/tmp/modelopt_qat_train2017_ptq`.

| Path | Stage | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 |
|---|---|---:|---:|---:|
| ModelOpt | FP32 | 0.6384 | 0.4718 | baseline |
| ModelOpt | PTQ INT8, README baseline | 0.6368 | 0.4706 | -0.0012 |
| ModelOpt | QAT INT8, README baseline | 0.6370 | 0.4701 | -0.0017 |
| ModelOpt | PTQ INT8, train2017 calibration | 0.6370 | 0.4708 | -0.0011 |
| PyTorch AO | FP32 | 0.6385 | 0.4719 | baseline |
| PyTorch AO | PTQ INT8, all Conv output Q | 0.6200 | 0.4493 | -0.0226 |
| PyTorch AO | QAT INT8, all Conv output Q | 0.6200 | 0.4495 | -0.0224 |
| PyTorch AO | PTQ INT8, no Conv output Q | 0.6317 | 0.4619 | -0.0100 |
| PyTorch AO | QAT INT8, no Conv output Q | 0.6324 | 0.4637 | -0.0082 |
| PyTorch AO | PTQ INT8, no Conv output Q + histogram | 0.6316 | 0.4631 | -0.0088 |
| PyTorch AO | QAT INT8, no Conv output Q + histogram | 0.6335 | 0.4654 | -0.0065 |

Disabling Conv output fake-quant plus switching activation observers to histogram
cuts the AO-vs-FP32 QAT drop from ~0.0224 to ~0.0065 mAP50-95. It still trails
same-split ModelOpt PTQ by ~0.0054, so the remaining gap is likely residual-add
scale handling, qscheme details, or other ModelOpt quantizer behavior rather
than calibration-source leakage.

### AO PTQ output-quant sensitivity

These PTQ-only ablations were run under `/tmp` so they did not overwrite the
canonical AO artifacts.

| Variant | Fake-quant modules | mAP50 | mAP50-95 | Δ vs AO baseline PTQ |
|---|---:|---:|---:|---:|
| Baseline AO, Conv input/weight/output fake quant | 306 | 0.6200 | 0.4493 | baseline |
| Disable Detect-head Conv output fake quant | 282 | 0.6263 | 0.4566 | +0.0074 |
| Disable all Conv output fake quant | 204 | 0.6317 | 0.4619 | +0.0126 |
| Disable all Conv output fake quant + 10-epoch QAT | 204 | 0.6324 | 0.4637 | +0.0144 |
| Disable all Conv output fake quant + histogram PTQ | 204 | 0.6316 | 0.4631 | +0.0138 |
| Disable all Conv output fake quant + histogram + 10-epoch QAT | 204 | 0.6335 | 0.4654 | +0.0161 |

This points to redundant Conv output fake quant as a major contributor to the
low AO PTQ score. Disabling all Conv output fake quant recovers over half of
the AO-vs-FP32 drop; histogram observers and QAT add another +0.0035 mAP50-95
on top of that. `yolo26-ao-distill` now defaults to this no-Conv-output-quant
plus histogram policy.

### TensorRT batch-1 latency (RTX 3060 Laptop GPU)

| Model | Engine | Mean latency | Throughput | Speedup |
|---|---|---:|---:|---:|
| `yolo11s` | FP32 | — | — | 1.00× |
|           | INT8 QAT (AO) | — | — | — |
| `yolo26s` | FP32 | — | — | 1.00× |
|           | INT8 QAT (AO) | — | — | — |

### Superseded yolo26s numbers

The previous yolo26s table listed FP32=0.4719, PTQ INT8=0.4493, and QAT
INT8=0.4490 mAP50-95 without a surviving machine-written summary because an
export-only rerun had overwritten `summary.json`. That result is superseded by
the 2026-05-20 full rerun above. The current `summary.json` and
`ao_qat_train.csv` both come from the same 10-epoch run.

### ONNX export accuracy

| Model | Format | mAP50 | mAP50-95 | Δ mAP50-95 vs FP32 ONNX |
|---|---|---:|---:|---:|
| `yolo11s` | ONNX FP32 (ort baseline) | — | — | baseline |
|           | ONNX INT8 QAT (AO) | — | — | — |
| `yolo26s` | ONNX FP32 (ort baseline) | — | — | baseline |
|           | ONNX INT8 QAT (AO) | exported | 296 Q / 296 DQ | mAP not measured yet |

Run notes (2026-05-20):

- Full source command used `--calib-source train2017 --calib-size 260`,
  `--qat-epochs 10 --qat-batches-per-epoch 200`, batch 10, val batch 8,
  workers 4, seed 0, and RTX 3060 Laptop GPU. The improved AO run added
  `--disable-conv-output-quant` and `--ao-activation-observer histogram`,
  both now yolo26 AO recipe defaults.
- `runs/pytorch_ao_qat/yolo26s/ao_qat_train.csv` now has all 10 epochs. Total
  loss moved from 25.0714 to 24.2159, while distill MSE stayed near 0.40.
- AO quant-state logging recorded `amax_count=306`. `amax_rel_drift` stayed
  0.0 because AO observer/fake-quant scale snapshots did not change during QAT.
- ONNX export initially failed because `YOLO.export()` expects
  `model.args` to be a dict after trainer code had replaced it with an
  `IterableSimpleNamespace`. The AO script now normalizes `model.args` before
  export. The existing exported file
  `runs/pytorch_ao_qat/yolo26s/yolo26s_ao_qat.onnx` is from the earlier
  export-only rerun, not from the 2026-05-20 metric run, which used
  `--no-export`.

---

## Risks / open questions

- **TRT compatibility of AO Q/DQ graph.** ModelOpt emits Q/DQ nodes in the
  exact pattern TensorRT's INT8 builder expects. `torch.ao.quantization`
  `convert_pt2e` / FX paths can emit a topologically different graph; TRT
  may fall back to FP or reject it. Mitigation: verify Netron diff against
  a ModelOpt export before promising INT8 engine numbers.
- **Single-head exclusions.** The ModelOpt `yolo11-distill` recipe gates QAT
  regressions by excluding the DFL head and Detect output quantizers. The
  AO equivalents are CLI flags (`--exclude-dfl-quant`,
  `--disable-detect-output-quant`); the Detect flag disables Conv output fake
  quant under Detect heads while leaving their input/weight fake quant enabled.
  Confirm `--ao-recipe auto` applies both flags for `yolo11*`.
  If not, pass them explicitly in M2.
- **AO quant-state logging.** Previous AO runs wrote `amax_count=0` because the shared QAT loop only inspected ModelOpt `_amax` buffers. The AO path now passes AO fake-quant scale snapshots into the loop; the full yolo26s rerun recorded `amax_count=306` (102 Conv wrappers x input/weight/output fake quant).
- **Residual-add / quantizer parity.** AO now uses histogram activation observers for yolo26 and disables Conv output fake quant, but the remaining ~0.0054 mAP50-95 gap versus same-split ModelOpt PTQ suggests residual-add scale handling, qscheme details, or other ModelOpt quantizer behavior still matters.
- **Teacher / student BN handling.** ModelOpt freezes teacher BN; verify
  `--freeze-teacher-bn` defaults match across recipes so the loss curves
  are comparable.
- **Cost.** Two full runs × ~10 epochs each ≈ a few hours of GPU time
  per model on RTX 3060 Laptop. Smoke test first (M0) before committing.

---

## Out of scope (for this roadmap)

- Tuning AO-specific hyperparams (per-channel vs per-tensor, observer
  variants, alternative qschemes). First pass uses recipe defaults so the
  comparison isolates "ModelOpt vs AO with the same training schedule."
- INT4 / FP8 — ModelOpt-only for now.
- Non-YOLO models (`rfdetr_nano` checkpoint present in tree but unrelated).
