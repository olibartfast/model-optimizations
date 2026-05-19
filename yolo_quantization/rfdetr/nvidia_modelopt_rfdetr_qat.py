#!/usr/bin/env python3
"""
INT8 Quantization for Roboflow RF-DETR via NVIDIA ModelOpt.

Adapted from yolo_quantization/qat/nvidia_modelopt_yolo_qat.py to the LWDETR
backbone+transformer+heads architecture used by RF-DETR (DETR-style detection
with a DINOv2 ViT backbone). Key differences from the YOLO recipe:

* **Input format.** RF-DETR ingests a ``NestedTensor`` (image tensor + pad mask)
  at a fixed square resolution (384 px for ``RFDETRNano``, etc.). ImageNet
  normalization is applied before tensor packing.
* **Head exclusions.** The DETR-style ``class_embed`` (Linear) and
  ``bbox_embed`` (MLP) heads are sensitive to INT8 (same reasoning as YOLO
  DFL + Detect-output). Default exclusions: ``"*class_embed*"``,
  ``"*bbox_embed*"``. LayerNorm and Embedding modules are skipped
  automatically by ModelOpt's default config; no extra work needed.
* **Calibration.** Plain val2017 jpgs preprocessed in-script (resize → /255 →
  ImageNet mean/std). RF-DETR's own dataset pipeline has an
  Albumentations-version-dependent transform stack that proved brittle, so
  we sidestep it for calibration only.

This module ships PTQ + ONNX export. QAT loop and mAP eval are the next steps
(see TODO at the bottom).

Usage::

    python yolo_quantization/rfdetr/nvidia_modelopt_rfdetr_qat.py \\
        --model rfdetr-nano --calib-size 256 --imgsz 384 --device 0
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_DIR = PROJECT_ROOT / "datasets" / "coco" / "images" / "val2017"
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
    """Load an RF-DETR variant and return ``(torch_model, resolution)``."""
    import rfdetr  # lazy import keeps this module importable without rfdetr installed

    cls_name = RFDETR_VARIANTS.get(variant)
    if cls_name is None:
        raise ValueError(f"Unknown RF-DETR variant: {variant!r}. Choices: {sorted(RFDETR_VARIANTS)}")
    wrapper_cls = getattr(rfdetr, cls_name)
    wrapper = wrapper_cls()
    ctx = wrapper.model
    model = ctx.model.to(device).eval()
    return model, int(ctx.resolution)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _imagenet_normalize(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def load_calib_batch(paths: Iterable[Path], size: int, device: torch.device):
    """Load ``paths`` as a NestedTensor batch of size×size ImageNet-normalized images."""
    from rfdetr.utilities.tensors import NestedTensor

    images = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    batch = torch.stack(images).to(device, non_blocking=True)
    batch = _imagenet_normalize(batch, device)
    mask = torch.zeros(batch.shape[0], size, size, dtype=torch.bool, device=device)
    return NestedTensor(batch, mask)


def build_calib_batches(
    image_dir: Path,
    size: int,
    batch: int,
    num_samples: int,
    device: torch.device,
    seed: int = 0,
) -> list:
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Calibration image dir not found: {image_dir}. "
            f"Run scripts/download_coco_dataset.sh first."
        )
    rng = np.random.default_rng(seed)
    paths = sorted(image_dir.glob("*.jpg"))
    if len(paths) == 0:
        raise RuntimeError(f"No .jpg files in {image_dir}")
    indices = rng.permutation(len(paths))[:num_samples]
    selected = [paths[i] for i in indices]
    batches = []
    for i in range(0, len(selected), batch):
        batches.append(load_calib_batch(selected[i : i + batch], size, device))
    print(f"  [calib] built {len(batches)} batches "
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
                m(b)
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
# ONNX export
# ---------------------------------------------------------------------------

class _RFDETRTracingWrapper(torch.nn.Module):
    """Wraps an LWDETR forward so it accepts plain ``Tensor`` input for ONNX tracing.

    LWDETR.forward expects a NestedTensor (tensor + pad mask); for export we
    construct an all-zero mask of matching spatial dims on the fly. This mirrors
    rfdetr's own ONNX export wrapper.
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
        # Return only the inference outputs; aux_outputs / enc_outputs are
        # training-time artifacts and confuse downstream TRT.
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


def export_onnx(model: torch.nn.Module, imgsz: int, device: torch.device, dst: Path) -> Path | None:
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    wrapper = _RFDETRTracingWrapper(model).to(device).eval()
    try:
        onnx_bytes = _modelopt_onnx_bytes(wrapper, dummy, onnx_opset=17)
    except Exception as exc:
        print(f"  [export] WARN: ONNX export failed: {exc!r}")
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
# Entry point — PTQ only (QAT loop is a follow-up; see module docstring)
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(RFDETR_VARIANTS.keys()), default="rfdetr-nano")
    p.add_argument("--calib-size", type=int, default=256, help="# val2017 images for PTQ")
    p.add_argument("--imgsz", type=int, default=0, help="Input size (0 = model default)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--head-excludes",
        nargs="+",
        default=list(DEFAULT_HEAD_EXCLUDES),
        metavar="GLOB",
        help="Module-path globs to keep FP32 (default: class/bbox heads)",
    )
    p.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch as _torch
    _torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"\n=== {args.model} ===")
    print(f"  [load] device={device}")
    model, model_res = load_rfdetr(args.model, device)
    imgsz = args.imgsz or model_res
    print(f"  [load] resolution={imgsz} (model default {model_res})")

    if not VAL_DIR.exists():
        print(f"ERROR: {VAL_DIR} not found", file=sys.stderr)
        return 2

    calib_batches = build_calib_batches(
        VAL_DIR, imgsz, args.batch, args.calib_size, device, seed=args.seed
    )

    t0 = time.time()
    run_ptq(model, calib_batches, device, head_excludes=tuple(args.head_excludes))
    print(f"  [ptq] elapsed {time.time() - t0:.1f}s")

    import modelopt.torch.opt as mto

    out_dir = OUT_ROOT / args.model.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.model.replace('-', '_')}_ptq.pth"
    mto.save(model, str(ckpt))
    print(f"  [ptq] saved {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)")

    if not args.no_export:
        onnx_dst = out_dir / f"{args.model.replace('-', '_')}_ptq.onnx"
        export_onnx(model, imgsz, device, onnx_dst)

    summary = {
        "model": args.model,
        "resolution": imgsz,
        "calib_size": args.calib_size,
        "head_excludes": list(args.head_excludes),
        "ptq_checkpoint": str(ckpt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [done] {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
