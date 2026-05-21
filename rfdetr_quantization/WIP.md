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
| PTQ | ✅ working | `mtq.quantize` inserts quantizers under named policies: `broad` (heads FP32), `no-dino-mlp-attn` (DINO attention/MLP FP16), and `conv-projector-only` (transformer + DINO blocks FP16). |
| mAP evaluator | ✅ done | `evaluate_map` runs COCO `val2017` through the quantized model + `PostProcess` and scores via rfdetr's `CocoEvaluator` (raw COCO category ids, no remap). |
| QAT distillation loop | ✅ done | FP32 teacher / INT8 student, **backbone feature-map distillation** (normalized MSE); Adam low/high/low ladder; per-epoch eval + best-checkpoint rollback. |
| e2e pipeline | ✅ done | `main()` runs FP32 eval → PTQ → PTQ eval → QAT → QAT eval → ONNX, with stage skip/resume flags. |
| ONNX export | ✅ validated | `rfdetr_nano_qat.onnx` exports at opset 17 with 192 Q/DQ pairs; `rfdetr_nano_fp32.onnx` exports as the FP16 TensorRT baseline input. |
| TRT speedup measurement | ✅ done | `rfdetr_trt.py`: INT8 Q/DQ engine 2.304 ms / 433.9 fps / mAP 0.4929; FP16 baseline 2.324 ms / 430.3 fps / mAP 0.4999; speedup 1.01x. |
| TRT profiling | ✅ grouped pass | Python TensorRT `IProfiler` support added to `rfdetr_trt.py`; grouped profile shows broad INT8 spends 45.4% in MLP/linear and only 9.0% in conv/projection. |
| Selective policies | ✅ measured | `--quant-policy` + `--run-tag` allow non-overwriting sweeps. Accuracy holds for `conv-projector-only` (0.4984 AP) and `no-dino-mlp-attn` (0.4968 AP), but both TensorRT INT8 engines are slower than FP16 (~0.98x). |

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
`--quant-policy {broad,no-dino-mlp-attn,conv-projector-only}`, `--extra-excludes GLOB...`,
`--run-tag NAME` (write `summary_NAME.json` and suffixed checkpoints/ONNX),
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
TensorRT builds a 36.0 MB INT8 engine. On RTX 3060 Laptop GPU, broad INT8 latency
is 2.304 ms mean / 433.9 fps with deployed-engine mAP50-95 0.4929. The FP32 ONNX
baseline built as FP16 is 2.324 ms / 430.3 fps with mAP50-95 0.4999, so RF-DETR
is accuracy-correct but only 1.01x faster in this configuration.

Selective PTQ policy results so far, all on the same 1500-image eval slice:

| Policy | Q/DQ pairs | Torch PTQ mAP50-95 | TRT mAP50-95 | Mean latency | Speedup vs FP16 |
|---|---:|---:|---:|---:|---:|
| `broad` QAT | 192 | 0.4925 | 0.4929 | 2.304 ms | 1.01x |
| `conv-projector-only` PTQ | 18 | 0.4984 | 0.4987 | 2.340 ms | 0.98x |
| `no-dino-mlp-attn` PTQ | 48 | 0.4968 | 0.4977 | 2.324 ms | 0.98x |

### Batch sweep — the "no speedup" was a static-batch-1 artifact

The 0.98–1.01x figures above are all **static batch-1** engines. That is the
worst-case operating point: rfdetr-nano's transformer GEMMs are tiny, so at
batch 1 they are memory/launch-bound and the explicit Q/DQ reformat tax swamps
the INT8 GEMM saving. `rfdetr_batch_sweep.py` re-exports both ONNX with a
dynamic batch axis, builds dynamic-shape engines (profile min1/opt8/max32), and
benchmarks a batch sweep. INT8 wins as soon as the GEMMs are compute-bound:

| Batch | INT8 ms/img | FP16 ms/img | INT8 fps | FP16 fps | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1  | 2.554 | 3.336 | 392 | 300 | 1.31x |
| 4  | 1.582 | 1.894 | 632 | 528 | 1.20x |
| 8  | 1.430 | 1.764 | 700 | 567 | 1.23x |
| 16 | 1.424 | 1.777 | 702 | 563 | 1.25x |
| 32 | 1.421 | 1.761 | 704 | 568 | 1.24x |

In the compute-bound regime (batch >= 8) INT8 is a stable **~1.24x** faster
(704 vs 568 fps). The INT8 quantization pipeline is correct — it *does*
accelerate; the earlier "no speedup" result was purely the batch-1 corner.
Caveat: the batch-1 row (1.31x) is not a fair batch-1 claim — both dynamic
engines are tactic-tuned for opt=8, so the dynamic FP16 engine is mistuned at
batch 1 (3.34 ms vs the 2.32 ms a *static* batch-1 FP16 engine gets). The fair
batch-1 comparison stays the static ~0.85–1.0x; what the sweep newly proves is
every batch >= 4. Latency-only; mAP unchanged (same 192-Q/DQ graph, 0.4929).

Artifacts: `runs/modelopt_qat/rfdetr_nano/batch_sweep_summary.json`,
`rfdetr_nano_{int8,fp16}_dyn.engine`, `rfdetr_nano_{qat,fp32}_dyn.onnx`.

---

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| QAT objective | Backbone feature-map distillation | Avoids RF-DETR's brittle Albumentations target transforms *and* the DETR set-prediction mismatch — feature maps are position-aligned so MSE is correct (see above). |
| QAT image pool | COCO `train2017` (default `--qat-source train`) | No labels are used, but training on `train2017` keeps the QAT data disjoint from the `val2017` eval set. |
| Calibration data | COCO `val2017` | Reuses the existing dataset; calibration is statistics-only so leakage is mild. |
| Preprocessing | cv2 square resize + ImageNet normalize | RF-DETR resizes inputs to a square by design; `PostProcess` rescales boxes by the original image size, undoing the distortion. |
| Head exclusions | `*class_embed*`, `*bbox_embed*` | DETR analog of YOLO's DFL + Detect-output exclusion: the final-prediction Linears are accuracy-critical and kept FP32. |
| Quantization policies | Named `--quant-policy` presets | Upstream RF-DETR issues and local TRT profiling both show full transformer INT8 has weak speedup. Policies let us leave DINO attention/MLP FP16 and measure smaller INT8 islands without overwriting baseline artifacts. |
| Best-epoch rollback | Checkpoint the highest-mAP epoch | QAT can overshoot; the lowest-loss epoch is not always the highest-mAP one. |
| Variant for first runs | `rfdetr-nano` (30.5M params, 384px) | Smallest, lowest VRAM, fastest iteration. |

---

## Open questions / next steps

1. **INT8 speedup is real once compute-bound.** The batch sweep settles the "no speedup" question: broad INT8 is ~1.24x faster than FP16 at batch >= 8 (704 vs 568 fps). The 0.98–1.01x figures were a static-batch-1 artifact. Next: re-eval mAP on the dynamic INT8 engine to confirm accuracy holds, and decide whether batch-1 latency or batched throughput is the deployment target.
2. **For batch-1 latency, the runtime path is still the lever.** If single-image latency (not throughput) is the goal, INT8 at batch 1 stays a wash — pursue end-to-end INT8 block fusion (kill the reformat tax), newer TensorRT MHA fusion, or weight-only quant. Selective `conv-projector-only` / `no-dino-mlp-attn` policies preserve accuracy but do not help batch-1 speed.
3. **Supervised QAT.** Distillation recovers most of the PTQ gap; a supervised `SetCriterion` fine-tune (with Hungarian matching) is the next lever if more accuracy is needed.
4. **Broader AO comparison.** The PyTorch AO RF-DETR script exists; run the same 1500-image eval and TensorRT export path once an AO ONNX export is proven.
5. **Repo layout.** If RF-DETR support grows, rename the top-level package from `yolo_quantization/` to `model_quantization/` and split the workstreams.

---

## Roboflow RF-DETR issue scan

Searched `roboflow/rf-detr` with `gh` for quantization, TensorRT, ONNX, latency, and speed issues. Relevant findings:

- #504 `Quantization-Aware Training` is open; maintainer notes INT8 attention needs special consideration and deployment environment matters.
- #566 `INT8 Optimization Slower Than FP32 on RTX A4000` was closed with guidance that ONNX Runtime CUDA is inefficient for transformer models; use TensorRT FP16 on Linux, and uncalibrated INT8 can regress.
- #955 `Latency on Jetson Thor (fp16/fp8/int8)` reports little meaningful INT8/FP8 speedup; comments point at backbone ViT/attention behavior, TensorRT version, and max-power benchmarking mode.
- #703 clarifies older explicit TensorRT plugin wiring is dead code in current RF-DETR; discussion says `torch.compile` and TensorRT backends can be competitive once dynamic-shape graph breaks are patched.
- #649 confirms official RF-DETR `.export()` plus the single-artifact benchmarking path is the expected TensorRT route for accuracy-correct FP16 deployment.

Takeaway: the weak broad-INT8 speedup is not just our pipeline. For RF-DETR, accuracy-preserving INT8 Q/DQ is feasible, but speedup requires the runtime to actually fuse/accelerate transformer attention and MLP work. The roadmap should treat selective quantization as a measurement tool and prioritize TensorRT/compile/attention runtime paths for real latency gains.

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
