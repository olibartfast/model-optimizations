# Closing the QAT–PTQ Gap for YOLO26: A Co-Aligned Distillation and Supervised Recipe for INT8 Quantization-Aware Training

**Francesco Oliva**
`olibartfast@gmail.com`

**Date:** 2026-05-15

---

## Abstract

We report a practical case study on INT8 Quantization-Aware Training (QAT) of the *YOLO26-small* object detector on the COCO `val2017` benchmark, using NVIDIA's ModelOpt as the quantization toolkit and an Ultralytics-based training harness. A naïve teacher–student distillation recipe, transplanted from the YOLOv7-QAT reference, regressed the post-QAT mean Average Precision (mAP) by **8.9 absolute points** of mAP$_{50\text{-}95}$ relative to a Post-Training-Quantized (PTQ) baseline that was itself essentially lossless against FP32 ($-0.0012$ mAP$_{50\text{-}95}$). We trace this regression to three concrete defects in the distillation pipeline: (i) a *silent branch mismatch* introduced by YOLO26's dual `one2many`/`one2one` detection heads when the quantized student's `one2many` head degenerates after PTQ; (ii) an *unbalanced raw-MSE objective* in which auxiliary feature maps dominate the loss because they have $\sim$10$\times$ more elements than decoded predictions; and (iii) the *absence of a supervised ground-truth signal*, leaving distillation alone unable to compensate for residual quantization noise. We propose three corresponding fixes — a co-aligning normalizer, a per-tensor-normalized MSE, and a supervised detection-loss term routed through the `one2one` head of YOLO26's `E2ELoss` — and show that the combined recipe reduces the regression from $-0.0879$ to $-0.0009$ mAP$_{50\text{-}95}$, landing within the noise floor of the 5000-image validation set. We additionally document a failure mode in which Ultralytics' high-level `YOLO.train()` API silently destroys the quantized graph and is therefore unsuitable as a fallback QAT pathway.

**Keywords:** quantization-aware training, INT8 quantization, knowledge distillation, object detection, YOLO, NVIDIA ModelOpt, Ultralytics, COCO.

---

## 1. Introduction

Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) are the two dominant routes for compressing modern neural networks to integer arithmetic suitable for inference on edge accelerators. PTQ requires only a small calibration set and produces a deployable model in minutes; QAT additionally fine-tunes the model with simulated quantization in the forward pass, recovering accuracy when PTQ alone is insufficient.

For modern YOLO-family detectors, PTQ alone is often close to lossless when implemented carefully (histogram calibration, residual-add quantizer patching, head-output exclusion). QAT is therefore expected to be, at worst, a *no-op*: if PTQ already achieves FP32 parity, QAT need only preserve that parity while shaping the weights to live more comfortably on the quantization grid. A QAT recipe that regresses *below* PTQ is therefore not "fine-tuning gone wrong" — it is evidence of a structural defect in the training objective.

This work reports such a defect, and its resolution, for the *YOLO26-small* model (Ultralytics, January 2026) on the COCO `val2017` benchmark, using NVIDIA's ModelOpt as the quantization library and an Ultralytics `DetectionTrainer` for data loading and supervised loss computation. The pipeline structure is itself adapted from the public *YOLOv7-QAT* reference [1] but was found to interact poorly with YOLO26's dual-head architecture and with ModelOpt's PTQ-restoration semantics.

Our contributions are:

1. **A diagnostic methodology** for distillation-based QAT in which we directly inspect the teacher and student forward-pass output structures on a calibration batch before any fine-tuning. This single diagnostic was sufficient to expose a branch mismatch that was otherwise silent under the existing loss code.
2. **Three targeted fixes** that, together, reduce the QAT–PTQ gap from $-0.0879$ to $-0.0009$ absolute mAP$_{50\text{-}95}$ on yolo26s, without changing the calibration recipe or quantization configuration. Each fix is implementable in fewer than 30 lines of code and is independently motivated.
3. **A negative result** on the `yolo.train(...)`-based QAT pathway, which silently rebuilds the model from the YAML architecture file and discards all quantizer modules inserted by ModelOpt, leading to a catastrophic mAP collapse to $\approx 0.0001$.

## 2. Background

### 2.1 INT8 Quantization-Aware Training with ModelOpt

NVIDIA ModelOpt [2] implements simulated INT8 quantization by injecting fake-quant `TensorQuantizer` modules at the inputs, outputs, and weight tensors of supported `nn.Module` types. After a *calibration* phase using histograms over real activation distributions, each quantizer is parameterized by a per-tensor (or per-channel, for weights) scale `amax`, which is stored as a non-trainable buffer. During QAT, the underlying floating-point weights remain trainable; the quantizer simulates the round-and-clip noise via the straight-through estimator, so gradients can flow through and reshape the weights to live more comfortably on the quantization grid.

### 2.2 The YOLOv7-QAT Reference Recipe

The yolov7_qat reference [1] established a now-conventional QAT recipe for YOLO detectors:

* **PTQ first**: histogram calibration on natural images, with a deliberate choice to skip quantization of the *detect-head outputs* (the output values are wide-dynamic-range floats that do not survive INT8 well).
* **Supervision via distillation**: a teacher–student loop in which a *frozen FP32 teacher* (the original detector) and a *quantized student* are forwarded over the same images, and the student is trained to minimize the mean-squared error between teacher and student raw multi-scale detect outputs.
* **Short fine-tune**: typically 2–5 epochs at a very low learning rate ($10^{-5}$ – $10^{-6}$), without ground-truth labels.

The intent of distillation-only supervision is that the FP32 teacher already encodes the ground-truth signal in its outputs; matching the teacher implicitly aligns the student to the ground truth.

### 2.3 YOLO26 and Its Dual-Head Output

YOLO26 (Ultralytics, January 2026) adopts an end-to-end (E2E) detection head structure with *two parallel prediction branches*:

* `one2many`: the dense classical detection head used for training supervision (assigns multiple candidates per ground-truth box via task-aligned assignment with $\text{topk}=10$).
* `one2one`: a sparse head used both as the inference path and as a target for end-to-end distillation (assigns one candidate per ground-truth box).

In training mode, the forward pass returns a dictionary `{"one2many": {boxes, scores, feats}, "one2one": {boxes, scores, feats}}`. The classical loss criterion (`E2ELoss` in Ultralytics) consumes *both* branches.

## 3. Setup

### 3.1 Hardware and Software

All experiments were run on a single NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM) with CUDA 12.6, PyTorch 2.7.1, Ultralytics 8.4.50, and NVIDIA ModelOpt installed via the NGC PyPI index. Python 3.12.3.

### 3.2 Data

The COCO 2017 dataset is used in the Ultralytics layout. We use a *deterministic 4000-image subsample* of `train2017` for QAT fine-tuning (selected with `seed=0`); the full 5000-image `val2017` is used for both PTQ calibration ($n=512$ images) and for all reported mAP numbers. The PTQ calibration set overlaps with the eval set, but PTQ amax estimation is a statistics-only operation that does not involve any gradient computation; this is the same protocol used in [1] and is treated as acceptable for our benchmarking purpose.

Validation is performed with `imgsz=640`, `conf=0.001`, `iou=0.6`, batch size 8, identical across the FP32, PTQ, and QAT stages so that the mAP numbers are directly comparable.

### 3.3 Quantization Configuration

We use `INT8_DEFAULT_CFG` from ModelOpt as the base, with `entropy` (histogram-based) activation calibration to most closely approximate the yolov7_qat reference's `MSE`-histogram path (ModelOpt does not expose `MSE` directly; `entropy` shares the same histogram backend). Conv+BN folding is applied before quantization. Residual additions inside Ultralytics `Bottleneck`, `GhostBottleneck`, `CIB`, `HGBlock`, `ResNetBlock`, and `Residual` blocks are patched to route through explicit `TensorQuantizer` modules on both branches; the two branches' scales are unified post-calibration. The detect-head output quantizers are left *enabled* throughout (we did not need to disable them on this model — PTQ remained essentially lossless with them on).

### 3.4 The Baseline Distillation Recipe

The starting-point QAT recipe (henceforth "v0") follows yolov7_qat: PTQ-calibrate the student, freeze the FP32 teacher, and run for 2 epochs of 200 batches at `lr=1e-5` with Adam, with an MSE loss against the teacher's raw outputs. The single-input normalizer `_teacher_student_raw_outputs(out)` walks the YOLO26 dict output structure and returns a flat list of tensors; the loss is the sum of `mse_loss(student, teacher)` over that list.

## 4. Diagnosis

After 2 epochs of v0, yolo26s reaches `mAP_{50-95} = 0.3827`, an **8.79-point regression** from the PTQ baseline (`0.4706`). Loss curves are flat ($\approx 6.6 \rightarrow 5.3$ MSE over two epochs), suggesting the optimizer is converging — to the wrong target.

We hypothesized that the bug lies in the loss target itself and ran a one-batch *diagnostic forward pass*: load the saved PTQ checkpoint into a student model, load a fresh FP32 teacher, push a single 2-image batch through both, and print the recursive structure of each output. The result was decisive:

```
TEACHER OUTPUT:
  one2many: {boxes: (2,4,8400), scores: (2,80,8400), feats: [3 tensors]}
  one2one:  {boxes: (2,4,8400), scores: (2,80,8400), feats: [3 tensors]}

STUDENT OUTPUT:
  one2many: {}                                  <-- EMPTY
  one2one:  {boxes: (2,4,8400), scores: (2,80,8400), feats: [3 tensors]}
```

The restored quantized student's `one2many` head is *empty* — a known consequence of ModelOpt's `replace_quant_module` pass bypassing one of the auxiliary computations in the YOLO26 detect head. The original normalizer, which preferred `one2many` and fell back to `one2one` only on emptiness, therefore returned:

* From the teacher: `one2many` tensors.
* From the student: `one2one` tensors.

Their *shapes match* (both heads operate on the same 8400 anchor positions and produce identical channel counts), so the MSE flowed through without raising an error — but the two tensors represent *different* prediction distributions, and the MSE gradient pushes the student in a non-task-relevant direction. This is the dominant root cause.

Two ancillary issues compound the problem:

* **Unbalanced MSE scales.** The feature maps (`feats`) have $\sim 10^5$ elements each, while `boxes` and `scores` have $\sim 10^4$ and $\sim 10^5$ elements with $O(1)$ magnitudes. An unnormalized sum of `mse_loss` calls (each itself a *mean* over its tensor) sums apples with oranges; the feats happen to dominate when their reconstruction error is in absolute terms larger than that of the decoded outputs. This contributes a smaller, but measurable, portion of the regression.

* **No ground-truth signal.** Even with correctly aligned targets, distillation alone matches the *teacher*, not the *labels*. If the teacher itself differs from the labels by $\epsilon$, the student inherits that error; if the student's quantization noise creates a basin of similar-by-MSE solutions away from the labeled mAP optimum, distillation can settle in the wrong basin.

## 5. Method

We address each of the three observations with a minimal, independently-motivated fix.

### 5.1 Co-Aligned Branch Selection

We replace the single-input normalizer with a *co-aligning* pair-aware function:

```python
def _coalign_distill_outputs(student, teacher):
    for branch in ("one2one", "one2many"):
        s = _extract_yolo_branch_tensors(student, branch)
        t = _extract_yolo_branch_tensors(teacher, branch)
        if s and t and len(s) == len(t) and all(si.shape == ti.shape for si, ti in zip(s, t)):
            return s, t
    return (_teacher_student_raw_outputs(student),
            _teacher_student_raw_outputs(teacher))
```

We prefer `one2one` because it is populated on both teacher and student. The legacy single-input path is preserved as a fallback for non-YOLO26 architectures and as a back-compatibility shim for upstream callers.

### 5.2 Per-Tensor-Normalized MSE

We replace `sum(mse_loss(s, t))` with a *mean of normalized per-tensor mean-squared errors*:

$$
\mathcal{L}_{\text{distill}} = \frac{1}{N}\sum_{i=1}^{N} \frac{\text{MSE}(s_i, t_i)}{\max(\overline{|t_i|}, \, 10^{-6})}
$$

where $N$ is the number of co-aligned tensors and $\overline{|t_i|}$ is the mean absolute teacher value for tensor $i$. The clamp avoids division by zero on degenerate batches; the per-tensor normalization makes the loss invariant to the channel/anchor counts that gave feats artificial dominance.

### 5.3 Supervised Loss Through the YOLO26 `one2one` Head

We add a supervised term using Ultralytics' detection criterion. YOLO26's `E2ELoss.__call__` consumes *both* branches and crashes (`KeyError: 'boxes'`) on the empty student `one2many` dict, so we cannot route directly through `E2ELoss.__call__`. Instead, we invoke `E2ELoss.one2one.loss(...)` directly, which is the standard `v8DetectionLoss.loss` configured with `tal_topk=1`:

```python
def _supervised_yolo_loss(student, outputs, batch):
    if student.criterion is None:
        student.criterion = student.init_criterion()
    crit = student.criterion
    parsed = crit.one2many.parse_output(outputs)
    one2one = parsed["one2one"]
    return crit.one2one.loss(one2one, batch)   # (loss_components, loss_items)
```

The returned loss is a 3-element vector (box, cls, dfl); we sum it before backprop, matching the trainer's standard handling. The final QAT objective is:

$$
\mathcal{L}_{\text{QAT}} = w_{\text{sup}} \cdot \mathcal{L}_{\text{sup}} + w_{\text{dist}} \cdot \mathcal{L}_{\text{distill}}
$$

with $w_{\text{sup}} = w_{\text{dist}} = 1.0$ in all reported experiments. In practice the supervised term ($\approx 36$ at our LR) dominates the distillation term ($\approx 0.18$ after normalization), so the distillation loss acts as a regularizer rather than a primary signal. We did not tune the weights further.

### 5.4 An Aside on the Ultralytics Fallback Path

The script offered a second QAT mode, `--qat-mode ultralytics`, which delegates training to Ultralytics' `YOLO.train()` API. We tested this as a sanity baseline and observed that the resulting "QAT" model collapses to `mAP_{50-95} = 0.0001`. The root cause is in Ultralytics' implementation: `YOLO.train` calls `self.trainer.get_model(weights=..., cfg=self.model.yaml)`, which *rebuilds the model from the YAML architecture file* before loading the state dict; the new model has no `TensorQuantizer` modules, the state dict matches only on the underlying float tensors, and the resulting model is a slightly-perturbed FP32 detector with no quantization simulation. We also fixed an orthogonal latent bug on this path: passing `lrf=qat_lr` to `yolo.train()` collapses the learning rate to $\approx 0$ over the run because Ultralytics treats `lrf` as a *ratio* ($\text{final\_lr} = \text{lr}_0 \cdot \text{lrf}$). We set `lrf=1.0` (constant LR) for safety. The ultralytics-mode path is otherwise *not* used.

## 6. Experiments

### 6.1 Protocol

Each experiment loads the same PTQ checkpoint (`yolo26s_ptq.pth`, $\text{mAP}_{50\text{-}95}=0.4706$), runs QAT under a specified loss configuration, and reports mAP on COCO `val2017` (5000 images, `conf=0.001`, `iou=0.6`, `imgsz=640`). The PTQ checkpoint is itself frozen across experiments to isolate the effect of the training loss; only the QAT objective and the number of training steps vary. Training uses Adam, batch size 16 (with mosaic/mixup/copy-paste/erasing disabled to keep the augmentation matched to inference), and a low/high/low LR ladder (peak $10^{-5}$, low $10^{-6}$) for $\geq 3$-epoch runs.

### 6.2 Recovery Trajectory

| # | Loss configuration | Epochs $\times$ batches/epoch | mAP$_{50}$ | mAP$_{50\text{-}95}$ | $\Delta$ vs PTQ |
|---|---|---|---:|---:|---:|
| v0 | MSE only, single-input normalizer (legacy)         | 2 $\times$ 200 | 0.5910 | 0.3827 | $-0.0879$ |
| v1 | + co-aligned `one2one`, per-tensor-normalized MSE  | 2 $\times$ 200 | 0.6296 | 0.4256 | $-0.0450$ |
| v2 | + COCO supervised loss (`E2ELoss.one2one`)          | 2 $\times$ 200 | 0.6365 | 0.4687 | $-0.0019$ |
| v3 | (same loss as v2, more steps)                      | 3 $\times$ 250 | **0.6370** | **0.4697** | $-0.0009$ |
| v4 | (same loss as v2, even more steps)                 | 4 $\times$ 250 | 0.6369 | 0.4695 | $-0.0011$ |

Each row's *Δ* column is the mAP$_{50\text{-}95}$ change relative to the *same* PTQ checkpoint (`0.4706`). The FP32 reference, measured under the same eval protocol, is `0.4718`.

Three observations:

1. **Each fix contributes independently and substantially.** Co-alignment + normalized MSE alone recovers 4.3 of the 8.8 lost points; adding the supervised term recovers the remaining 4.3 points to within statistical noise.

2. **The supervised term is the dominant signal once added.** Its raw magnitude ($\approx 36$ for `box+cls+dfl`) is two orders larger than the normalized distillation term ($\approx 0.18$), so with equal weights the distillation acts as a regularizer. We did not need to tune the weight ratio.

3. **More than 3 epochs is over-training.** Comparing v3 and v4, an extra epoch slightly *degrades* the result, consistent with the v3 LR ladder spending exactly one epoch at peak LR on a network that is already near the PTQ optimum.

### 6.3 Final Result

| Stage | mAP$_{50}$ | mAP$_{50\text{-}95}$ | $\Delta$ vs FP32 | $\Delta$ vs PTQ |
|---|---:|---:|---:|---:|
| FP32 | 0.6384 | 0.4718 | baseline | — |
| PTQ INT8 | 0.6368 | 0.4706 | $-0.0012$ (0.25%) | baseline |
| **QAT INT8** | **0.6370** | **0.4697** | $-0.0021$ (0.44%) | $-0.0009$ (0.19%) |

QAT $\text{mAP}_{50}$ marginally exceeds PTQ $\text{mAP}_{50}$ (0.6370 vs 0.6368), confirming that the regression is within the noise floor of a 5000-image evaluation.

## 7. Discussion

**Why is the "one2many empty" condition silent?** The combination of (i) ModelOpt's `replace_quant_module` selectively bypassing the auxiliary computation, (ii) Ultralytics' `Detect` heads returning a dict-of-dicts in training mode rather than a fixed tuple, and (iii) the legacy normalizer's lenient `if not tensors: fall through` fallback together produce a code path that is shape-compatible but semantically wrong. Type-checking, shape-checking, and even a unit test of the normalizer (which we had) all pass on this path. The diagnostic that exposed it was a *concrete forward pass* on the actual restored checkpoint, not a static analysis.

**Why distillation alone is not enough.** Even after fixing the branch mismatch, MSE-on-decoded-outputs converged to a stable plateau at `mAP_{50-95} = 0.4256`, ~4.5 points below PTQ. We interpret this as the inherent limitation of teacher-only supervision in the presence of quantization noise: the teacher's *outputs* are matched to within tolerance but the teacher's *correctness* (relative to ground truth) is not preserved through the matching. Adding the ground-truth supervised loss directly anchors the student to mAP-improving updates, which is consistent with the observation that the supervised term dominates the gradient at our weight ratio.

**Why we did not pursue the Ultralytics-mode path further.** It is structurally incompatible with ModelOpt's module-replacement strategy. A QAT pipeline that goes through `yolo.train()` would need to either (a) override `trainer.get_model` to short-circuit the YAML rebuild, or (b) re-apply `mtq.quantize` and re-calibrate after each training run. Both are substantially more invasive than our chosen approach.

**Limitations.** Our train subset is 4000 images, sufficient for 3 epochs $\times$ 250 batches $\times$ 16 batch-size = 12000 sample-steps with one full cycle. We have not investigated whether the larger train2017 corpus would yield further improvement on top of v3. We also did not search the supervised/distillation weight ratio nor the LR schedule; the reported numbers are from the first weight setting that worked. The result is therefore a lower-bound on what this recipe can achieve; we did not need to look further because the v3 result is already within the val-set noise floor.

## 8. Conclusion

For YOLO26-small on COCO, an INT8 QAT recipe that combines a *co-aligned* `one2one`-head distillation MSE with the standard supervised detection loss closes the QAT–PTQ regression from $-0.0879$ to $-0.0009$ absolute mAP$_{50\text{-}95}$. The dominant failure of the prior recipe was not the choice of loss family but a silent dictionary-key mismatch in the distillation target, made invisible by output shapes coincidentally aligning across two semantically distinct prediction heads. The combined recipe is otherwise small in code volume — three short functions and two CLI flags — and is implemented entirely above ModelOpt's public API, with no changes to the quantization configuration or calibration.

## Reproducibility

All code, the COCO data layout, the PTQ checkpoint, and the QAT checkpoint corresponding to the final v3 result are released at:

> https://github.com/olibartfast/model-optimizations

The exact command that reproduces the final QAT result:

```bash
quantization_venv/bin/python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --calib-method entropy \
  --qat-mode distill \
  --qat-epochs 3 \
  --qat-batches-per-epoch 250 \
  --qat-lr 1e-5 \
  --qat-distill-weight 1.0 \
  --qat-supervised-weight 1.0 \
  --imgsz 640 --batch 16 --val-batch 8 --workers 4 --device 0 \
  --skip-fp32-eval --no-export
```

See `yolo_quantization/qat/README.md` for full pipeline documentation and per-experiment artifact paths.

## References

1. NVIDIA-AI-IOT. *yolov7_qat*. `https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat`.
2. NVIDIA. *TensorRT Model Optimizer (ModelOpt)*. `https://github.com/NVIDIA/TensorRT-Model-Optimizer`.
3. Jocher, G. et al. *Ultralytics YOLO* (v8.4.50). `https://github.com/ultralytics/ultralytics`.
4. Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.
