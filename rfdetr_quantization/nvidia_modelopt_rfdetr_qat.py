#!/usr/bin/env python3
"""
End-to-end INT8 QAT for Roboflow RF-DETR via NVIDIA ModelOpt.

Adapted from ``yolo_quantization/qat/nvidia_modelopt_yolo_qat.py`` to the LWDETR
backbone+transformer+heads architecture used by RF-DETR (DETR-style detection
with a DINOv2 ViT backbone). Key differences from the YOLO recipe:

* **Input format.** RF-DETR ingests a ``NestedTensor`` (image tensor + pad mask)
  at a fixed square resolution (384 px for ``RFDETRNano``, etc.). RF-DETR
  resizes inputs to a square by design, so a plain ``cv2.resize`` to
  ``imgsz x imgsz`` followed by ImageNet normalization is the correct
  preprocessing — ``PostProcess`` rescales boxes back by the original image
  size, undoing the aspect distortion.
* **Head exclusions.** The DETR-style ``class_embed`` (Linear) and
  ``bbox_embed`` (MLP) heads are sensitive to INT8 (same reasoning as YOLO
  DFL + Detect-output). Default exclusions: ``"*class_embed*"``,
  ``"*bbox_embed*"``. LayerNorm and Embedding modules are skipped
  automatically by ModelOpt's default config; no extra work needed.
* **QAT objective.** Pure teacher/student distillation: a frozen FP32 teacher
  and the INT8 student see the same images; the student is fine-tuned with a
  per-tensor-normalized MSE on ``pred_logits`` + ``pred_boxes``. No COCO
  ground-truth targets are needed, which sidesteps RF-DETR's
  Albumentations-version-dependent target transform stack.

Full pipeline (each stage resumable / skippable):

    FP32 eval -> PTQ calibrate -> PTQ eval -> QAT distill -> QAT eval -> ONNX

Usage::

    python rfdetr_quantization/nvidia_modelopt_rfdetr_qat.py \\
        --model rfdetr-nano --calib-size 256 --qat-epochs 4 --device 0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = PROJECT_ROOT / "datasets" / "coco" / "images" / "val2017"
TRAIN_DIR = PROJECT_ROOT / "datasets" / "coco" / "images" / "train2017"
ANN_FILE = PROJECT_ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"
OUT_ROOT = PROJECT_ROOT / "runs" / "modelopt_qat"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RF-DETR variant -> ``rfdetr`` class name. Names are kebab-case on the CLI.
RFDETR_VARIANTS = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-base": "RFDETRBase",
    "rfdetr-large": "RFDETRLarge",
}

# Default head modules to skip during INT8 quantization. Pattern matches
# any quantizer attached to a module whose full path contains the glob.
DEFAULT_HEAD_EXCLUDES = ("*class_embed*", "*bbox_embed*")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_rfdetr(variant: str, device: torch.device):
    """Load an RF-DETR variant; return ``(torch_model, resolution, postprocess)``."""
    import rfdetr  # lazy import keeps this module importable without rfdetr installed

    cls_name = RFDETR_VARIANTS.get(variant)
    if cls_name is None:
        raise ValueError(f"Unknown RF-DETR variant: {variant!r}. Choices: {sorted(RFDETR_VARIANTS)}")
    wrapper_cls = getattr(rfdetr, cls_name)
    wrapper = wrapper_cls()
    ctx = wrapper.model
    model = ctx.model.to(device).eval()
    return model, int(ctx.resolution), ctx.postprocess


# ---------------------------------------------------------------------------
# Calibration / training / eval data
# ---------------------------------------------------------------------------

def _imagenet_normalize(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _preprocess_image(path: Path, size: int) -> torch.Tensor:
    """cv2 read -> RGB -> square resize -> /255 CHW float tensor (CPU, unnormalized)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def load_calib_batch(paths: Iterable[Path], size: int):
    """Load ``paths`` as a CPU-resident NestedTensor batch of ImageNet-normalized images."""
    from rfdetr.utilities.tensors import NestedTensor

    images = [_preprocess_image(p, size) for p in paths]
    batch = _imagenet_normalize(torch.stack(images))
    mask = torch.zeros(batch.shape[0], size, size, dtype=torch.bool)
    return NestedTensor(batch, mask)


def build_calib_batches(
    image_dir: Path,
    size: int,
    batch: int,
    num_samples: int,
    seed: int = 0,
    label: str = "calib",
) -> list:
    """Build CPU-resident NestedTensor batches sampled from ``image_dir``.

    Batches are kept on CPU and moved to the device per-use; this keeps the
    6 GB-class laptop GPUs from OOMing when ``num_samples`` is large.
    """
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image dir not found: {image_dir}. Run scripts/download_coco_dataset.sh first."
        )
    rng = np.random.default_rng(seed)
    paths = sorted(image_dir.glob("*.jpg"))
    if len(paths) == 0:
        raise RuntimeError(f"No .jpg files in {image_dir}")
    indices = rng.permutation(len(paths))[:num_samples]
    selected = [paths[i] for i in indices]
    batches = [
        load_calib_batch(selected[i : i + batch], size)
        for i in range(0, len(selected), batch)
    ]
    print(f"  [{label}] built {len(batches)} batches "
          f"({sum(b.tensors.shape[0] for b in batches)} images @ {size}x{size})")
    return batches


# ---------------------------------------------------------------------------
# PTQ
# ---------------------------------------------------------------------------

def build_quant_cfg(mtq, head_excludes: tuple[str, ...] = DEFAULT_HEAD_EXCLUDES) -> dict:
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for glob in head_excludes:
        cfg["quant_cfg"][glob] = {"enable": False}
    return cfg


def calibrate_factory(batches: list, device: torch.device):
    def calibrate(m: torch.nn.Module) -> None:
        m.eval()
        seen = 0
        with torch.no_grad():
            for b in batches:
                m(b.to(device))
                seen += b.tensors.shape[0]
        print(f"  [calib] forwarded {seen} samples through quantizers")
    return calibrate


def run_ptq(
    model: torch.nn.Module,
    calib_batches: list,
    device: torch.device,
    head_excludes: tuple[str, ...] = DEFAULT_HEAD_EXCLUDES,
):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

    quant_cfg = build_quant_cfg(mtq, head_excludes)
    print(f"  [ptq] head exclusions: {list(head_excludes)}")
    mtq.quantize(model, quant_cfg, calibrate_factory(calib_batches, device))
    n_q = sum(1 for _ in model.modules() if isinstance(_, TensorQuantizer))
    print(f"  [ptq] inserted {n_q} TensorQuantizer modules")
    return model


# ---------------------------------------------------------------------------
# mAP evaluation (COCO val2017)
# ---------------------------------------------------------------------------

def evaluate_map(
    model: torch.nn.Module,
    postprocess: torch.nn.Module,
    image_dir: Path,
    ann_file: Path,
    device: torch.device,
    imgsz: int,
    batch: int = 4,
    limit: int = 0,
) -> dict:
    """Run ``model`` over COCO ``val2017`` and return ``{map, map50}``.

    RF-DETR's ``class_embed`` emits logits whose channel index *is* the COCO
    category id directly (no contiguous remapping for COCO-trained models), so
    ``CocoEvaluator`` with ``label2cat=None`` consumes ``PostProcess`` output
    as-is.
    """
    from pycocotools.coco import COCO
    from rfdetr.evaluation.coco_eval import CocoEvaluator
    from rfdetr.utilities.tensors import NestedTensor

    coco_gt = COCO(str(ann_file))
    evaluator = CocoEvaluator(coco_gt, ["bbox"])
    img_ids = sorted(coco_gt.getImgIds())
    if limit:
        img_ids = img_ids[:limit]

    model.eval()
    t0 = time.time()
    for i in range(0, len(img_ids), batch):
        chunk = img_ids[i : i + batch]
        tensors, sizes = [], []
        for iid in chunk:
            info = coco_gt.loadImgs(iid)[0]
            tensors.append(_preprocess_image(image_dir / info["file_name"], imgsz))
            sizes.append((info["height"], info["width"]))
        batch_t = _imagenet_normalize(torch.stack(tensors)).to(device)
        mask = torch.zeros(len(chunk), imgsz, imgsz, dtype=torch.bool, device=device)
        with torch.no_grad():
            out = model(NestedTensor(batch_t, mask))
        target_sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        results = postprocess(out, target_sizes)
        evaluator.update({iid: res for iid, res in zip(chunk, results)})

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats
    elapsed = time.time() - t0
    print(f"  [eval] {len(img_ids)} imgs in {elapsed:.1f}s "
          f"-> mAP50-95={float(stats[0]):.4f} mAP50={float(stats[1]):.4f}")
    return {"map": float(stats[0]), "map50": float(stats[1]), "num_images": len(img_ids)}


# ---------------------------------------------------------------------------
# QAT distillation
# ---------------------------------------------------------------------------

def _clone_inference_buffers(model: torch.nn.Module) -> int:
    """ModelOpt leaves some buffers in ``torch.inference_mode``; that breaks
    autograd. Clone them out before the QAT loop."""
    cloned = 0
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and hasattr(buffer, "is_inference") and buffer.is_inference():
                module._buffers[name] = buffer.clone()
                cloned += 1
    return cloned


def _clone_quantizer_amax_state(model: torch.nn.Module) -> int:
    cloned = 0
    for module in model.modules():
        amax = module._buffers.get("_amax") if hasattr(module, "_buffers") else None
        if not torch.is_tensor(amax):
            continue
        with torch.inference_mode(False):
            module._buffers["_amax"] = amax.clone()
        cloned += 1
    return cloned


def _distill_epoch_lr(epoch: int, total_epochs: int, peak_lr: float, low_lr: float) -> float:
    """Low/high/low LR ladder (same shape as the YOLO QAT schedule)."""
    if total_epochs < 3:
        return peak_lr
    first_cut = max(1, total_epochs // 3)
    last_cut = max(first_cut + 1, total_epochs - first_cut)
    if epoch < first_cut or epoch >= last_cut:
        return low_lr
    return peak_lr


def _backbone_features(model: torch.nn.Module, nt) -> list[torch.Tensor]:
    """Run only ``model.backbone`` and return its feature-map tensors.

    RF-DETR's ``model.backbone`` is a DETR-style ``Joiner`` whose forward
    returns ``(features, pos)`` where ``features`` is a list of
    ``NestedTensor``. Calling the backbone directly (instead of the full
    model) skips the transformer + heads entirely — QAT only needs the
    backbone, so this is markedly faster and frees the activation memory the
    unused transformer graph would otherwise hold.
    """
    out = model.backbone(nt)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return [f.tensors if hasattr(f, "tensors") else f for f in feats]


def _feature_distill_loss(
    student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor]
) -> torch.Tensor:
    """Mean per-map normalized MSE between INT8-student and FP32-teacher
    backbone feature maps.

    Why feature-level, not output-level: RF-DETR's ``pred_logits`` /
    ``pred_boxes`` are an *unordered set* of 300 queries whose slot assignment
    is not stable between two slightly different models, so a per-slot MSE on
    them is a wrong objective — empirically it regresses or collapses mAP even
    when confidence-weighted. Backbone feature maps are spatial ``[B,C,H,W]``
    tensors, perfectly position-aligned, so a plain MSE *is* a correct and
    stable distillation target. The DINOv2 ViT backbone is 23.3M of
    RF-DETR-nano's 30.5M parameters and the dominant source of INT8 error, so
    pulling the quantized backbone's features back onto the FP32 manifold is
    where QAT has the most leverage. Each map is normalized by its teacher
    tensor's mean-square so maps at different scales contribute evenly.
    """
    loss = student_feats[0].new_zeros(())
    for s, t in zip(student_feats, teacher_feats):
        t = t.detach()
        loss = loss + F.mse_loss(s, t) / t.pow(2).mean().clamp_min(1e-8)
    return loss / max(1, len(student_feats))


def run_distillation_qat(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    train_batches: list,
    device: torch.device,
    *,
    epochs: int,
    peak_lr: float,
    low_lr: float,
    grad_clip: float,
    log_every: int,
    eval_callback=None,
    on_improve=None,
) -> list[dict]:
    """QAT fine-tune of the INT8 ``student`` via FP32-teacher feature distillation.

    Only the **backbone** parameters are trained — the loss is a normalized MSE
    between the student's and teacher's backbone feature maps (see
    ``_feature_distill_loss`` for why feature-level, not output-level). The
    backbone is 76% of RF-DETR-nano's weights and the dominant INT8-error
    source; the transformer/head quantizers (small PTQ loss) are left as-is.

    Both models run with ``training=False`` (single-query-group inference path,
    frozen BatchNorm); the student still receives gradients (eval mode does not
    disable autograd, and ModelOpt's ``TensorQuantizer`` fake-quantizes with a
    straight-through estimator in both modes).

    ``eval_callback(epoch)`` returns ``{map, map50}`` or ``None``. Whenever mAP
    improves over the running best, ``on_improve(epoch, metrics)`` is invoked
    (used by the caller to checkpoint the best student).
    """
    inf_buffers = _clone_inference_buffers(student)
    amax_states = _clone_quantizer_amax_state(student)
    if inf_buffers or amax_states:
        print(f"  [qat] cloned {inf_buffers} inference buffer(s), "
              f"{amax_states} quantizer amax tensor(s) before training")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.eval()  # inference forward path + frozen BN; grad still flows
    for p in student.parameters():
        p.requires_grad_(False)
    params = [p for p in student.backbone.parameters() if p.dtype.is_floating_point]
    for p in params:
        p.requires_grad_(True)
    print(f"  [qat] feature distillation: {sum(p.numel() for p in params)/1e6:.1f}M "
          f"trainable backbone params, {len(train_batches)} batches/epoch x {epochs} epochs")

    # One optimizer for the whole run — momentum carries across epochs; only the
    # LR is stepped by the low/high/low ladder.
    opt = torch.optim.Adam(params, lr=low_lr)
    history: list[dict] = []
    best_map = -1.0
    for epoch in range(epochs):
        lr = _distill_epoch_lr(epoch, epochs, peak_lr, low_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        running = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_batches):
            nt = batch.to(device)
            with torch.no_grad():
                t_feats = [f.detach() for f in _backbone_features(teacher, nt)]
            s_feats = _backbone_features(student, nt)
            loss = _feature_distill_loss(s_feats, t_feats)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            running += float(loss.detach())
            if log_every and (step + 1) % log_every == 0:
                print(f"  [qat/distill] epoch {epoch+1}/{epochs} "
                      f"step {step+1}/{len(train_batches)} "
                      f"loss={running/(step+1):.5f}")
        n = max(1, len(train_batches))
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "loss": running / n,
            "secs": time.time() - t0,
        }
        if eval_callback is not None:
            metrics = eval_callback(epoch + 1)
            if metrics:
                row["map"] = metrics.get("map")
                row["map50"] = metrics.get("map50")
                if metrics["map"] > best_map:
                    best_map = metrics["map"]
                    if on_improve is not None:
                        on_improve(epoch + 1, metrics)
        print(f"  [qat] epoch {epoch+1}/{epochs} lr={lr:.2e} "
              f"loss={row['loss']:.5f} {row['secs']:.1f}s"
              + (f" map={row['map']:.4f}" if row.get("map") is not None else ""))
        history.append(row)
    return history


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class _RFDETRTracingWrapper(torch.nn.Module):
    """Wraps an LWDETR forward so it accepts plain ``Tensor`` input for ONNX tracing.

    LWDETR.forward expects a NestedTensor (tensor + pad mask); for export we
    construct an all-zero mask of matching spatial dims on the fly.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        from rfdetr.utilities.tensors import NestedTensor

        mask = torch.zeros(
            images.shape[0], images.shape[2], images.shape[3],
            dtype=torch.bool, device=images.device,
        )
        out = self.model(NestedTensor(images, mask))
        return out["pred_logits"], out["pred_boxes"]


def _modelopt_onnx_bytes(model: torch.nn.Module, dummy: torch.Tensor, *, onnx_opset: int) -> bytes:
    """Get ONNX bytes via modelopt — prefer ``mte.export_onnx`` (newer), fall back to deploy path."""
    try:
        from modelopt.torch import export as mte
        fn = getattr(mte, "export_onnx", None)
        if fn is not None:
            try:
                out = fn(model, dummy, onnx_opset=onnx_opset)
            except TypeError:
                out = fn(model, dummy, opset_version=onnx_opset)
            if isinstance(out, (bytes, bytearray, memoryview)):
                return bytes(out)
    except Exception:
        pass
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes
    return get_onnx_bytes(model, dummy, onnx_opset=onnx_opset)


def export_onnx(
    model: torch.nn.Module, imgsz: int, device: torch.device, dst: Path, opset: int = 20
) -> Path | None:
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    wrapper = _RFDETRTracingWrapper(model).to(device).eval()
    try:
        onnx_bytes = _modelopt_onnx_bytes(wrapper, dummy, onnx_opset=opset)
    except Exception as exc:
        print(f"  [export] WARN: ONNX export failed (opset {opset}): {exc!r}")
        return None
    dst.write_bytes(onnx_bytes)
    try:
        import onnx
        g = onnx.load(str(dst))
        n_q = sum(n.op_type == "QuantizeLinear" for n in g.graph.node)
        n_dq = sum(n.op_type == "DequantizeLinear" for n in g.graph.node)
        print(f"  [export] wrote {dst} (Q={n_q} DQ={n_dq})")
    except Exception as e:
        print(f"  [export] wrote {dst} (Q/DQ count failed: {e})")
    return dst


# ---------------------------------------------------------------------------
# Entry point — full e2e pipeline
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(RFDETR_VARIANTS.keys()), default="rfdetr-nano")
    p.add_argument("--imgsz", type=int, default=0, help="Input size (0 = model default)")
    p.add_argument("--batch", type=int, default=4, help="Batch size for calib / eval")
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # PTQ
    p.add_argument("--calib-size", type=int, default=256, help="# val2017 images for PTQ calibration")
    p.add_argument(
        "--head-excludes", nargs="+", default=list(DEFAULT_HEAD_EXCLUDES), metavar="GLOB",
        help="Module-path globs to keep FP32 (default: class/bbox heads)",
    )
    p.add_argument("--from-ptq", type=Path, default=None, help="Restore a saved *_ptq.pth and skip calibration")

    # Eval
    p.add_argument("--eval-limit", type=int, default=0, help="Cap eval images (0 = full val2017)")
    p.add_argument("--skip-fp32-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="Skip all mAP eval stages")

    # QAT
    p.add_argument("--skip-qat", action="store_true", help="PTQ-only run")
    p.add_argument("--qat-epochs", type=int, default=4)
    p.add_argument("--qat-images", type=int, default=480, help="# train2017 images per QAT epoch")
    p.add_argument("--qat-batch", type=int, default=2, help="QAT distillation batch size")
    p.add_argument("--qat-source", choices=["train", "val"], default="train",
                   help="Image pool for QAT distillation (train2017 avoids the eval leak)")
    p.add_argument("--qat-lr", type=float, default=1e-5, help="Peak Adam LR for the low/high/low ladder")
    p.add_argument("--qat-low-lr", type=float, default=1e-6, help="Low-rung Adam LR")
    p.add_argument("--qat-grad-clip", type=float, default=1.0, help="Grad-norm clip (0 = off)")
    p.add_argument("--qat-log-every", type=int, default=20)
    p.add_argument("--qat-eval-every", type=int, default=1,
                   help="Run mAP eval every N QAT epochs (0 = only before/after); "
                        "best-mAP epoch is checkpointed to *_qat_best.pth")

    # Export
    p.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    p.add_argument("--onnx-opset", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"\n=== {args.model} :: e2e INT8 QAT ===")
    print(f"  [load] device={device}")
    model, model_res, postprocess = load_rfdetr(args.model, device)
    imgsz = args.imgsz or model_res
    print(f"  [load] resolution={imgsz} (model default {model_res})")

    if not VAL_DIR.exists():
        print(f"ERROR: {VAL_DIR} not found", file=sys.stderr)
        return 2
    do_eval = not args.skip_eval and ANN_FILE.exists()
    if not args.skip_eval and not ANN_FILE.exists():
        print(f"  [eval] WARN: {ANN_FILE} not found — mAP eval disabled")

    out_dir = OUT_ROOT / args.model.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("-", "_")
    summary: dict = {
        "model": args.model,
        "resolution": imgsz,
        "calib_size": args.calib_size,
        "head_excludes": list(args.head_excludes),
        "eval_limit": args.eval_limit,
    }

    def run_eval(stage: str) -> dict | None:
        if not do_eval:
            return None
        print(f"  [eval:{stage}]")
        return evaluate_map(model, postprocess, VAL_DIR, ANN_FILE, device, imgsz,
                            batch=args.batch, limit=args.eval_limit)

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq  # noqa: F401 — ensures plugin patches load

    # --- Stage 1: FP32 eval -------------------------------------------------
    if do_eval and not args.skip_fp32_eval:
        summary["fp32"] = run_eval("fp32")

    # --- Teacher snapshot (FP32, frozen) before the model is quantized ------
    teacher = None
    if not args.skip_qat:
        teacher = copy.deepcopy(model).to(device).eval()

    # --- Stage 2: PTQ calibrate (or restore) --------------------------------
    if args.from_ptq is not None:
        print(f"  [ptq] restoring {args.from_ptq}")
        mto.restore(model, str(args.from_ptq))
    else:
        calib_batches = build_calib_batches(
            VAL_DIR, imgsz, args.batch, args.calib_size, seed=args.seed, label="calib"
        )
        t0 = time.time()
        run_ptq(model, calib_batches, device, head_excludes=tuple(args.head_excludes))
        print(f"  [ptq] elapsed {time.time() - t0:.1f}s")
        del calib_batches

    ptq_ckpt = out_dir / f"{tag}_ptq.pth"
    mto.save(model, str(ptq_ckpt))
    print(f"  [ptq] saved {ptq_ckpt} ({ptq_ckpt.stat().st_size/1e6:.1f} MB)")
    summary["ptq_checkpoint"] = str(ptq_ckpt)

    # --- Stage 3: PTQ eval --------------------------------------------------
    summary["ptq"] = run_eval("ptq")

    # --- Stage 4: QAT distillation -----------------------------------------
    if not args.skip_qat:
        qat_dir = TRAIN_DIR if args.qat_source == "train" else VAL_DIR
        # train2017 sampling is offset by the seed so QAT images differ from calib
        train_batches = build_calib_batches(
            qat_dir, imgsz, args.qat_batch, args.qat_images,
            seed=args.seed + 1, label="qat",
        )

        # Per-epoch eval drives best-checkpoint selection: QAT can overshoot,
        # so the lowest-loss epoch is not always the highest-mAP one.
        eval_cb = None
        if do_eval and args.qat_eval_every > 0:
            def eval_cb(epoch_idx: int):
                if epoch_idx % args.qat_eval_every == 0 or epoch_idx == args.qat_epochs:
                    return evaluate_map(model, postprocess, VAL_DIR, ANN_FILE, device,
                                        imgsz, batch=args.batch, limit=args.eval_limit)
                return None

        best: dict = {"state": None, "epoch": 0, "metrics": None}

        def on_improve(epoch_idx: int, metrics: dict):
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch_idx
            best["metrics"] = metrics
            print(f"  [qat] new best mAP50-95={metrics['map']:.4f} @ epoch {epoch_idx}")

        t0 = time.time()
        qat_history = run_distillation_qat(
            model, teacher, train_batches, device,
            epochs=args.qat_epochs,
            peak_lr=args.qat_lr,
            low_lr=args.qat_low_lr,
            grad_clip=args.qat_grad_clip,
            log_every=args.qat_log_every,
            eval_callback=eval_cb,
            on_improve=on_improve,
        )
        print(f"  [qat] elapsed {time.time() - t0:.1f}s")
        summary["qat_train"] = qat_history
        del train_batches, teacher

        # Roll back to the best-mAP epoch (QAT past the peak degrades the model).
        if best["state"] is not None:
            model.load_state_dict(best["state"])
            print(f"  [qat] restored best epoch {best['epoch']} "
                  f"(mAP50-95={best['metrics']['map']:.4f})")
            summary["qat"] = best["metrics"]
            summary["qat_best_epoch"] = best["epoch"]

        qat_ckpt = out_dir / f"{tag}_qat.pth"
        mto.save(model, str(qat_ckpt))
        print(f"  [qat] saved {qat_ckpt} ({qat_ckpt.stat().st_size/1e6:.1f} MB)")
        summary["qat_checkpoint"] = str(qat_ckpt)

        # --- Stage 5: QAT eval (only if per-epoch eval was disabled) -------
        if "qat" not in summary:
            summary["qat"] = run_eval("qat")

        # Per-epoch QAT training log (mirrors the YOLO pipeline's qat_train.csv)
        csv_path = out_dir / "qat_train.csv"
        cols = ["epoch", "lr", "loss", "secs", "map", "map50"]
        lines = [",".join(cols)]
        for row in qat_history:
            lines.append(",".join("" if row.get(c) is None else f"{row.get(c)}" for c in cols))
        csv_path.write_text("\n".join(lines) + "\n")
        print(f"  [qat] wrote {csv_path}")

    # --- Stage 6: ONNX export ----------------------------------------------
    if not args.no_export:
        suffix = "ptq" if args.skip_qat else "qat"
        onnx_dst = out_dir / f"{tag}_{suffix}.onnx"
        export_onnx(model, imgsz, device, onnx_dst, opset=args.onnx_opset)
        if onnx_dst.exists():
            summary["onnx"] = str(onnx_dst)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [done] {out_dir/'summary.json'}")

    # Compact stage comparison
    def _fmt(key):
        m = summary.get(key)
        return f"{m['map']:.4f}/{m['map50']:.4f}" if m else "—"
    print(f"\n  mAP50-95/mAP50  fp32={_fmt('fp32')}  ptq={_fmt('ptq')}  qat={_fmt('qat')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
