# RF-DETR INT8 Quantization — Work In Progress

Branch: `feature/rfdetr-quant`. Tracks the goal *"INT8 QAT quantization on
RF-DETR using NVIDIA ModelOpt"* started 2026-05-18.

---

## Status snapshot

| Stage | State | Notes |
|---|---|---|
| Architecture scouting | ✅ done | LWDETR: backbone (DINOv2 ViT) + transformer + heads (`class_embed` Linear + `bbox_embed` MLP); 155 Linear, 54 LayerNorm, 9 Conv2d, 2 MultiheadAttention. |
| Dependency conflict | ✅ resolved | `rfdetr` requires transformers≥5; modelopt's HF plugin used the removed `transformers.modeling_utils.Conv1D`. Patched the plugin to guard that registration (`quantization_venv/.../plugins/huggingface.py:209`). |
| Calibration pipeline | ✅ working | Bypass rfdetr's val transforms (brittle on Albumentations 2.x); hand-built loader does cv2 square resize → /255 → ImageNet normalize → `NestedTensor`. RF-DETR resizes to a square by design, so this is the *correct* preprocessing. |
| PTQ | ✅ working | `mtq.quantize` inserts ~498 `TensorQuantizer` modules; `*class_embed*` / `*bbox_embed*` heads excluded. |
| mAP evaluator | ✅ done | `evaluate_map` runs COCO `val2017` through the quantized model + `PostProcess` and scores via rfdetr's `CocoEvaluator` (raw COCO category ids, no remap). |
| QAT distillation loop | ✅ done | FP32 teacher / INT8 student, **backbone feature-map distillation** (normalized MSE); Adam low/high/low ladder; per-epoch eval + best-checkpoint rollback. |
| e2e pipeline | ✅ done | `main()` runs FP32 eval → PTQ → PTQ eval → QAT → QAT eval → ONNX, with stage skip/resume flags. |
| ONNX export | ✅ validated | `rfdetr_nano_qat.onnx` exports at opset 17 with 192 Q/DQ pairs; `rfdetr_nano_fp32.onnx` exports as the FP16 TensorRT baseline input. |
| TRT speedup measurement | ✅ done | `rfdetr_trt.py`: INT8 Q/DQ engine 2.304 ms / 433.9 fps / mAP 0.4929; FP16 baseline 2.324 ms / 430.3 fps / mAP 0.4999; speedup 1.01x. |
| TRT profiling | ✅ grouped pass | Python TensorRT `IProfiler` support added to `rfdetr_trt.py`; profile-only runs write grouped totals plus top layers to `trt_profile_summary.json`, `rfdetr_nano_int8_profile.csv`, and `rfdetr_nano_fp16_profile.csv`. INT8 profile: MLP/linear 45.4%, MHA 10.6%, conv/projection 9.0%; FP16 profile: MLP/linear 36.8%, MHA 25.5%, conv/projection 7.2%. |

---

## The QAT objective — why output-level distillation fails on DETR

Two QAT attempts with **output-level** distillation failed:

1. *Naive per-query MSE on `(pred_logits, pred_boxes)`* — collapsed mAP from
   ~0.49 (PTQ) to ~0.003. Root cause: **DETR predictions are an unordered
   set.** Of the 300 queries only a handful predict real objects; the rest are
   "no-object" queries whose `pred_boxes` are arbitrary, so a plain per-query
   box MSE is dominated by ~270 junk boxes and trains the student to fit noise.
2. *Confidence-weighted box MSE* (box term weighted by teacher
   `sigmoid(logits).amax(-1)`) — better, but still wrong: best epoch 0.4647 vs
   PTQ 0.4917, and the higher LR rungs still collapsed it to ~0.01. Even the
   gentlest epoch regressed. Decoder query slots are simply not stably aligned
   between two slightly different models, so *any* per-slot loss on the final
   predictions is a miscalibrated objective.

**Working objective — backbone feature distillation** (`_feature_distill_loss`):
distill the *spatial backbone feature maps* `[B,C,H,W]`, which are perfectly
position-aligned, so a plain normalized MSE is a correct and stable target.
The DINOv2 ViT backbone is 23.3M of RF-DETR-nano's 30.5M params and the
dominant INT8-error source; QAT trains only the backbone quantized weights to
pull their features back onto the FP32 manifold. The transformer/head
quantizers (small PTQ loss) are left as-is.

QAT also: runs only `model.backbone` per step (skips transformer + heads —
much faster, less memory), clips grad-norm (default 1.0), uses one Adam
instance across epochs (momentum carries; only LR steps via the low/high/low
ladder), evaluates every `--qat-eval-every` epochs, and **rolls back to the
best-mAP epoch**.

---

## Pipeline module

`rfdetr_quantization/nvidia_modelopt_rfdetr_qat.py` — the canonical e2e entry
point. Stages (each skippable / resumable):

1. **FP32 eval** — COCO `val2017` mAP of the unquantized model.
2. **PTQ calibrate** — `mtq.quantize` over `--calib-size` val2017 images, or
   `--from-ptq <ckpt>` to restore and skip calibration.
3. **PTQ eval** — mAP of the calibrated INT8 model.
4. **QAT distill** — `run_distillation_qat` (see above).
5. **QAT eval** — mAP of the best QAT epoch.
6. **ONNX export** — `_RFDETRTracingWrapper` + ModelOpt deploy path, with
   `_patch_interpolate_no_antialias()` around tracing to avoid the DINOv2
   `aten::_upsample_bicubic2d_aa` exporter blocker.

Outputs land in `runs/modelopt_qat/rfdetr_nano/`: `*_ptq.pth`, `*_qat.pth`,
`summary.json` (per-stage mAP), `qat_train.csv` (per-epoch loss + mAP).

Full run::

    quantization_venv/bin/python rfdetr_quantization/nvidia_modelopt_rfdetr_qat.py \
      --model rfdetr-nano --calib-size 256 --qat-epochs 4 --qat-images 960

Resume QAT from a saved PTQ checkpoint (skips calibration)::

    quantization_venv/bin/python rfdetr_quantization/nvidia_modelopt_rfdetr_qat.py \
      --model rfdetr-nano --skip-fp32-eval \
      --from-ptq runs/modelopt_qat/rfdetr_nano/rfdetr_nano_ptq.pth \
      --qat-epochs 4 --qat-images 960

Useful flags: `--eval-limit N` (cap eval images), `--skip-qat` (PTQ-only),
`--qat-source {train,val}` (QAT image pool; `train` avoids the eval leak),
`--qat-lr / --qat-low-lr`, `--qat-eval-every N`, `--no-export`,
`--onnx-opset N` (default 17).

---

## Latest metrics (rfdetr-nano, COCO val2017)

Eval: first 1500 `val2017` images, `PostProcess` num_select=300, COCO `bbox`
AP. PTQ calibrated on 256 `val2017` images; QAT = backbone feature
distillation, 6 epochs × 960 `train2017` images, batch 2, Adam low/high/low
ladder (1e-6 / 1e-5), best-epoch rollback.

| Stage | mAP50-95 | mAP50 | Δ vs FP32 |
|---|---|---|---|
| FP32   | 0.5001 | 0.6875 | — |
| INT8 PTQ | 0.4917 | 0.6779 | −0.0084 |
| INT8 QAT | **0.4925** | 0.6778 | −0.0076 (**+0.0008 over PTQ**) |

QAT recovers ~10% of the PTQ gap and, crucially, is **stable** — no collapse,
QAT ≥ PTQ. The low-LR epochs (1e-6) hold/recover; the 1e-5 rung perturbs
slightly (epoch 4 dipped to 0.4854), and best-epoch rollback keeps the epoch-6
0.4925 checkpoint. This matches the YOLO QAT bar ("within ~0.001 mAP of PTQ").
Per-epoch trace in `runs/modelopt_qat/rfdetr_nano/qat_train.csv`.

Headroom for a follow-up: QAT only fine-tunes the backbone; the transformer +
head quantizers are left at their PTQ state. Extending feature distillation to
the transformer encoder memory (also position-aligned) is the next lever.

Export/deployment status: ONNX export and TensorRT benchmarking are complete.
The QAT checkpoint exports to `rfdetr_nano_qat.onnx` with 192 Q/DQ pairs, and
TensorRT builds a 36.0 MB INT8 engine. On RTX 3060 Laptop GPU, INT8 latency is
2.304 ms mean / 433.9 fps with deployed-engine mAP50-95 0.4929. The FP32 ONNX
baseline built as FP16 is 2.324 ms / 430.3 fps with mAP50-95 0.4999, so RF-DETR
is accuracy-correct but only 1.01x faster in this configuration.

---

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| QAT objective | Backbone feature-map distillation | Avoids RF-DETR's brittle Albumentations target transforms *and* the DETR set-prediction mismatch — feature maps are position-aligned so MSE is correct (see above). |
| QAT image pool | COCO `train2017` (default `--qat-source train`) | No labels are used, but training on `train2017` keeps the QAT data disjoint from the `val2017` eval set. |
| Calibration data | COCO `val2017` | Reuses the existing dataset; calibration is statistics-only so leakage is mild. |
| Preprocessing | cv2 square resize + ImageNet normalize | RF-DETR resizes inputs to a square by design; `PostProcess` rescales boxes by the original image size, undoing the distortion. |
| Head exclusions | `*class_embed*`, `*bbox_embed*` | DETR analog of YOLO's DFL + Detect-output exclusion: the final-prediction Linears are accuracy-critical and kept FP32. |
| Best-epoch rollback | Checkpoint the highest-mAP epoch | QAT can overshoot; the lowest-loss epoch is not always the highest-mAP one. |
| Variant for first runs | `rfdetr-nano` (30.5M params, 384px) | Smallest, lowest VRAM, fastest iteration. |

---

## Open questions / next steps

1. **Selective transformer quantization.** Initial TRT profiling shows broad INT8
   is dominated by DINOv2 backbone MLP `fc1` MatMuls and fused MHA kernels, and
   the INT8 MLP MatMuls are slower than the FP16 fused-FC path. Try policies
   that leave attention/MLP blocks FP16 while quantizing only the patch/projector
   convs, then compare latency and mAP.
2. **Selective quantization policy sweep.** Now that grouped TRT profiling is in
   place, implement ModelOpt exclusion policies for backbone MLP/attention blocks
   versus patch/projector convs and compare latency, grouped profile, and mAP.
3. **Supervised QAT.** Distillation recovers most of the PTQ gap; a supervised
   `SetCriterion` fine-tune (with Hungarian matching) is the next lever if more
   accuracy is needed — it would need COCO targets in DETR format.
4. **Broader AO comparison.** The PyTorch AO RF-DETR script exists; run the same
   1500-image eval and TensorRT export path once an AO ONNX export is proven.
5. **Repo layout.** If RF-DETR support grows, rename the top-level package from
   `yolo_quantization/` to `model_quantization/` and split the workstreams.

---

## Bugs / gotchas discovered along the way

- **DETR set-prediction vs. per-slot distillation.** See the QAT objective
  section — the single most important gotcha. Distill *features*, not the
  set-valued `pred_*` outputs.
- **modelopt + transformers 5.x.** One-line guard patch in
  `modelopt/torch/quantization/plugins/huggingface.py` for the missing
  `Conv1D` attribute. Generic (`getattr(..., None)`); should be upstreamed.
- **rfdetr Albumentations transforms.** `make_coco_transforms('val', ...)`
  silently falls back to no-resize on Albumentations 2.x — sidestepped for
  both calibration and eval.
- **`PROJECT_ROOT` was off by one.** After the move to a top-level package the
  path computation pointed one directory above the repo root; fixed to
  `Path(__file__).resolve().parent.parent`.
- **`transformers.use_return_dict` deprecation warning** prints on every
  forward in transformers 5.x; cosmetic.
