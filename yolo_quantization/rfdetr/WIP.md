# RF-DETR INT8 Quantization — Work In Progress

Branch: `rfdetr-quant`. Tracks the in-flight goal *"INT8 QAT quantization on
RF-DETR using NVIDIA ModelOpt"* started 2026-05-18.

---

## Status snapshot

| Stage | State | Notes |
|---|---|---|
| Architecture scouting | ✅ done | LWDETR: backbone (DINOv2 ViT) + transformer + heads (`class_embed` Linear + `bbox_embed` MLP); 155 Linear, 54 LayerNorm, 9 Conv2d, 2 MultiheadAttention. |
| Dependency conflict | ✅ resolved | `rfdetr` requires transformers≥5; modelopt's HF plugin used the removed `transformers.modeling_utils.Conv1D`. Patched the plugin to guard that registration (`quantization_venv/.../plugins/huggingface.py:209`). |
| Calibration pipeline | ✅ working | Bypass rfdetr's val transforms (brittle on Albumentations 2.x); hand-built loader does cv2 resize → /255 → ImageNet normalize → `NestedTensor`. |
| PTQ smoke (32 imgs) | ✅ passes | `mtq.quantize` inserts 498 `TensorQuantizer` modules; post-PTQ forward returns valid `pred_logits (B,300,91)` + `pred_boxes (B,300,4)`. |
| `mto.save` checkpoint | ✅ works | 122.6 MB for rfdetr-nano @ 256 calib images. |
| ONNX export | ⚠️ partial | Hits `aten::_upsample_bicubic2d_aa` not supported in opset 17 (used by DINOv2 backbone for positional-encoding resample). Defer until QAT recipe is locked; retry with opset 20+. |
| mAP evaluator | ❌ not yet | Need to wire up `rfdetr.evaluation.coco_eval.CocoEvaluator` against the quantized model so we can measure PTQ accuracy. |
| QAT loop | ❌ not yet | LWDETR.forward(samples, targets) emits loss when targets are passed; criterion is internal. Plan: mirror `run_distillation_qat` pattern — teacher FP32 + student INT8 + MSE on (pred_logits, pred_boxes) + supervised criterion. |
| TRT speedup measurement | ❌ blocked on ONNX export | |

---

## Pipeline module

`yolo_quantization/rfdetr/nvidia_modelopt_rfdetr_qat.py` is the canonical entry
point for now. Currently ships:

- `load_rfdetr(variant, device)` — variants `rfdetr-{nano,small,medium,base,large}`.
- `build_calib_batches(image_dir, size, batch, num_samples, device, seed)` —
  COCO val2017 calibration set as `NestedTensor` batches at the model's native
  resolution (384 for nano).
- `build_quant_cfg(mtq, head_excludes=("*class_embed*", "*bbox_embed*"))` —
  INT8 cfg with DETR-head exclusions analogous to YOLO's DFL/Detect-output.
- `run_ptq(model, calib_batches, ...)` — `mtq.quantize` + quantizer count.
- `export_onnx(model, imgsz, device, dst)` — wraps the LWDETR in
  `_RFDETRTracingWrapper` (synthesises the pad mask) and routes through
  ModelOpt's deploy ONNX bytes path. Currently fails on opset-17 due to
  bicubic-AA upsample in the backbone — known issue.
- `main()` — CLI entry: `--model rfdetr-nano --calib-size 256 --imgsz 0
  --batch 4 --device 0 --seed 0`.

Run::

    quantization_venv/bin/python yolo_quantization/rfdetr/nvidia_modelopt_rfdetr_qat.py \
      --model rfdetr-nano --calib-size 256 --batch 4 --seed 0

Output: `runs/modelopt_qat/rfdetr_nano/rfdetr_nano_ptq.pth` + `summary.json`.

---

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Calibration data | COCO `val2017` (same as yolo pipeline) | Reuses existing dataset; calibration is statistics-only so leakage is mild. |
| Calibration loader | Hand-built `cv2`+ImageNet normalize | RF-DETR's `make_coco_transforms` relies on Albumentations 1.x transform names that fail silently on 2.x. Sidestepping for calibration only. |
| Head exclusions | `*class_embed*`, `*bbox_embed*` | DETR analog of YOLO single-head DFL+Detect-output exclusion: the final-prediction Linears (91-class logits and 4-coord box MLP) are accuracy-critical and worth keeping FP32. |
| LayerNorm / Embedding | Not explicitly excluded | ModelOpt's `INT8_DEFAULT_CFG` already skips these by op type; no extra config needed. |
| Calibration size | 256 images (CLI default) | Matches the ~260 used in the YOLO `yolo11-distill` recipe — proven sufficient for histogram-style calibration on 384px inputs. |
| Variant for first smoke | `rfdetr-nano` (30.5M params, 384px) | Smallest, lowest VRAM, fastest iteration. Other variants flagged in `RFDETR_VARIANTS`. |
| Repo location | `yolo_quantization/rfdetr/` | Sits next to `yolo_quantization/{ptq,qat}/`. Top-level dir name is now misleading; rename to `model_quantization/` is a follow-up if the rfdetr work matures. |

---

## Open questions / next steps

1. **mAP evaluator wiring.** Need a small `rfdetr_eval.py` helper that runs the
   quantized model through COCO val2017 and computes mAP via
   `rfdetr.evaluation.coco_eval.CocoEvaluator`. Blocks accuracy verification.
2. **QAT distillation loop.** Pattern to copy from
   `yolo_quantization/qat/nvidia_modelopt_yolo_qat.py:run_distillation_qat`:
   - frozen FP32 teacher + INT8 student on same batch
   - supervised criterion = `model(samples, targets)["loss"]` (LWDETR built-in)
   - distill MSE on `(pred_logits, pred_boxes)` outputs
   - per-tensor-normalized MSE (same shape concern as YOLO)
   - Adam @ 1e-5 with low/high/low LR ladder
3. **ONNX export.** Retry with `onnx_opset=20` (supports
   `_upsample_bicubic2d_aa`), or pre-process the DINOv2 positional-encoding
   resample to a static interpolation before tracing.
4. **TRT speedup.** Once ONNX exports, reuse `scripts/measure_speedup.sh` —
   it's model-agnostic. Add an `rfdetr-nano` row to the README table.
5. **Repo layout.** If RF-DETR support grows, rename the top-level package
   from `yolo_quantization/` to `model_quantization/` and split YOLO and
   RF-DETR into sibling subdirs.

---

## Bugs / gotchas discovered along the way

- **modelopt + transformers 5.x.** Required a one-line patch to
  `modelopt/torch/quantization/plugins/huggingface.py` to guard the missing
  `Conv1D` attribute. The fix is generic (uses `getattr(..., None)`), not
  RF-DETR-specific. **Should be upstreamed to NVIDIA/Model-Optimizer.**
- **rfdetr Albumentations transforms.** `make_coco_transforms('val', ...)`
  registers `SmallestMaxSize` / `LongestMaxSize` via `getattr(A, name)` but
  silently falls back to no resize when Albumentations 2.x changes the
  builder interaction. The dataset then yields full-resolution images,
  causing the DINOv2 backbone to assert on shape divisibility.
- **`transformers.use_return_dict` deprecation warning** prints on every
  forward in transformers 5.x; cosmetic, not a functional issue.

---

## Reproducing the current state

```bash
# 1. Branch
git checkout rfdetr-quant

# 2. Install (transformers gets pinned to 5.x by rfdetr; modelopt plugin needs the
#    Conv1D guard patch — see the file edit in the diff for this branch)
quantization_venv/bin/pip install rfdetr pycocotools albumentations

# 3. Symlink classic-COCO layout from our Ultralytics layout
mkdir -p datasets/coco_rfdetr/annotations
ln -s ../coco/images/val2017 datasets/coco_rfdetr/val2017
ln -s ../../coco/annotations/instances_val2017.json datasets/coco_rfdetr/annotations/

# 4. Run PTQ smoke (saves *_ptq.pth)
quantization_venv/bin/python yolo_quantization/rfdetr/nvidia_modelopt_rfdetr_qat.py \
  --model rfdetr-nano --calib-size 32 --batch 4 --seed 0
```
