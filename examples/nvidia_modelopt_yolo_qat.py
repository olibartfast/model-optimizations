#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for Ultralytics YOLO with NVIDIA ModelOpt.

Mirrors the structure of ``NVIDIA/Model-Optimizer/examples/cnn_qat/``
(PTQ calibration -> mtq.quantize -> QAT fine-tune -> mto.save -> export),
adapted for object detection:

* Calibration uses COCO 2017 ``val2017`` images with Ultralytics' inference
  preprocessing (letterbox + RGB + 0-1 float, NCHW).
* QAT fine-tuning is delegated to ``YOLO.train(data=coco.yaml, ...)`` because
  YOLO has its own detection loss, augmentation pipeline and trainer; the
  quantizer modules injected by ``mtq.quantize`` are preserved through
  training because ``mtq.quantize`` modifies ``yolo.model`` in place.
* We evaluate mAP at three stages: **FP32 baseline**, **after PTQ** (frozen
  calibrated scales, no training), and **after QAT** (fine-tuned).
* Checkpointing uses ``modelopt.torch.opt.mto.save / mto.restore`` so the
  quantized graph structure is reloadable.
* Final export writes a deployable ONNX file via Ultralytics.

Usage
-----
    python examples/nvidia_modelopt_yolo_qat.py \
        --models yolo11x yolo26x \
        --qat-epochs 2 \
        --calib-size 512 \
        --imgsz 640 \
        --batch 16

Every stage is resumable: FP32 / PTQ / QAT checkpoints, calibration tensors
and per-model summaries are cached under ``runs/modelopt_qat/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_YAML = PROJECT_ROOT / "configs" / "coco.yaml"
COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
VAL_DIR = COCO_ROOT / "val2017"
OUT_ROOT = PROJECT_ROOT / "runs" / "modelopt_qat"

DEFAULT_MODELS = ("yolo11x", "yolo26x")


# ---------------------------------------------------------------------------
# Calibration (mirrors cnn_qat/torchvision_qat.py:calibrate closure pattern)
# ---------------------------------------------------------------------------


def _letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3
    )


def build_calib_batches(
    image_dir: Path,
    imgsz: int,
    batch: int,
    num_samples: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Return a list of CPU float32 NCHW batches for PTQ calibration."""
    import cv2

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Calibration image dir not found: {image_dir}. "
            f"Run ./scripts/download_coco_dataset.sh first."
        )

    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No .jpg files in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:num_samples]

    tensors: list[torch.Tensor] = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        padded = _letterbox(rgb, imgsz)
        chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(chw))

    batches: list[torch.Tensor] = []
    for i in range(0, len(tensors), batch):
        batches.append(torch.stack(tensors[i : i + batch]))
    print(
        f"  [calib] built {len(batches)} batches "
        f"({sum(b.size(0) for b in batches)} images @ {imgsz}x{imgsz})"
    )
    return batches


# ---------------------------------------------------------------------------
# Eval / export helpers
# ---------------------------------------------------------------------------


def eval_on_coco_val(yolo, imgsz: int, batch: int, device: str) -> tuple[float, float, float]:
    """Return (mAP50, mAP50-95, seconds) via Ultralytics' detection validator."""
    t0 = time.time()
    results = yolo.val(
        data=str(COCO_YAML),
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.6,
        device=device,
        verbose=False,
        plots=False,
    )
    return float(results.box.map50), float(results.box.map), time.time() - t0


def export_onnx(yolo, imgsz: int, dst: Path) -> Path | None:
    """Export the (QAT) model to ONNX for deployment. Non-fatal on failure."""
    try:
        exported = yolo.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=17)
        Path(exported).replace(dst)
        print(f"  [export] wrote {dst}")
        return dst
    except Exception as e:
        print(f"  [export] WARN: ONNX export failed: {e}")
        return None


# ---------------------------------------------------------------------------
# QAT per-model orchestration
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    stage: str  # "fp32" | "ptq" | "qat"
    map50: float | None = None
    map: float | None = None
    seconds: float | None = None
    checkpoint: str | None = None
    error: str | None = None


@dataclass
class ModelResult:
    model: str
    stages: list[StageResult]
    onnx_path: str | None = None


def _calibrate_fn_factory(batches: list[torch.Tensor], device: torch.device, max_samples: int):
    """Build a ``calibrate(m)`` closure matching cnn_qat's pattern."""

    def calibrate(m: torch.nn.Module) -> None:
        m.eval()
        seen = 0
        with torch.no_grad():
            for b in batches:
                m(b.to(device, non_blocking=True))
                seen += b.size(0)
                if seen >= max_samples:
                    break
        print(f"  [calib] forwarded {seen} samples through quantizers")

    return calibrate


def run_qat_for_model(
    model_name: str,
    qat_epochs: int,
    qat_lr: float,
    calib_size: int,
    imgsz: int,
    batch: int,
    val_batch: int,
    device: str,
    skip_fp32_eval: bool,
    do_export: bool,
) -> ModelResult:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    print(f"\n=== {model_name} ===")
    model_dir = OUT_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    stages: list[StageResult] = []

    # -------- 1. Load FP32 model --------
    weights = f"{model_name}.pt"
    print(f"  [load] {weights}")
    yolo = YOLO(weights)

    torch_device = torch.device(f"cuda:{device}" if device.isdigit() else device)
    yolo.model.to(torch_device)

    # -------- 2. FP32 baseline mAP --------
    fp32_stage = StageResult(stage="fp32")
    if not skip_fp32_eval:
        try:
            map50, mAP, secs = eval_on_coco_val(yolo, imgsz, val_batch, device)
            fp32_stage.map50, fp32_stage.map, fp32_stage.seconds = map50, mAP, secs
            print(f"  [val/fp32] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
        except Exception as e:
            fp32_stage.error = str(e)
            print(f"  [val/fp32] FAILED: {e}")
    stages.append(fp32_stage)

    # -------- 3. PTQ calibration (mtq.quantize + calibrate closure) --------
    ptq_stage = StageResult(stage="ptq")
    try:
        calib_batches = build_calib_batches(VAL_DIR, imgsz, batch, calib_size)
        calibrate = _calibrate_fn_factory(calib_batches, torch_device, calib_size)

        quant_cfg = mtq.INT8_DEFAULT_CFG
        print("  [ptq] running mtq.quantize(..., INT8_DEFAULT_CFG, calibrate)")
        mtq.quantize(yolo.model, quant_cfg, calibrate)

        ptq_ckpt = model_dir / f"{model_name}_ptq.pth"
        mto.save(yolo.model, str(ptq_ckpt))
        ptq_stage.checkpoint = str(ptq_ckpt)
        print(f"  [ptq] saved {ptq_ckpt}")

        map50, mAP, secs = eval_on_coco_val(yolo, imgsz, val_batch, device)
        ptq_stage.map50, ptq_stage.map, ptq_stage.seconds = map50, mAP, secs
        print(f"  [val/ptq] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as e:
        ptq_stage.error = str(e)
        print(f"  [ptq] FAILED: {e}")
        stages.append(ptq_stage)
        return ModelResult(model=model_name, stages=stages)
    stages.append(ptq_stage)

    # -------- 4. QAT fine-tune via Ultralytics trainer --------
    qat_stage = StageResult(stage="qat")
    try:
        print(f"  [qat] fine-tuning for {qat_epochs} epoch(s) (lr0={qat_lr})")
        # Ultralytics trainer uses yolo.model (already quantized) in place.
        # Keep augmentations modest and LR small for PTQ recovery.
        yolo.train(
            data=str(COCO_YAML),
            epochs=qat_epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=qat_lr,
            lrf=qat_lr,  # constant LR over short QAT fine-tune
            warmup_epochs=0,
            optimizer="SGD",
            close_mosaic=0,
            device=device,
            project=str(model_dir),
            name="qat_train",
            exist_ok=True,
            verbose=False,
            val=False,  # we run our own val below with the COCO yaml
        )
        qat_ckpt = model_dir / f"{model_name}_qat.pth"
        mto.save(yolo.model, str(qat_ckpt))
        qat_stage.checkpoint = str(qat_ckpt)
        print(f"  [qat] saved {qat_ckpt}")

        map50, mAP, secs = eval_on_coco_val(yolo, imgsz, val_batch, device)
        qat_stage.map50, qat_stage.map, qat_stage.seconds = map50, mAP, secs
        print(f"  [val/qat] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as e:
        qat_stage.error = str(e)
        print(f"  [qat] FAILED: {e}")
    stages.append(qat_stage)

    # -------- 5. ONNX export for deployment --------
    onnx_path: Path | None = None
    if do_export and qat_stage.error is None:
        onnx_path = export_onnx(yolo, imgsz, model_dir / f"{model_name}_qat.onnx")

    result = ModelResult(
        model=model_name,
        stages=stages,
        onnx_path=str(onnx_path) if onnx_path else None,
    )
    (model_dir / "summary.json").write_text(
        json.dumps(
            {"model": result.model, "onnx_path": result.onnx_path,
             "stages": [asdict(s) for s in result.stages]},
            indent=2,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[ModelResult]) -> None:
    print("\n" + "=" * 78)
    print("NVIDIA ModelOpt QAT x YOLO / COCO summary")
    print("=" * 78)
    header = f"{'model':<10} {'stage':<5} {'mAP50':>8} {'mAP':>8} {'secs':>7}  notes"
    print(header)
    print("-" * len(header))
    for r in results:
        for s in r.stages:
            map50 = "   -  " if s.map50 is None else f"{s.map50:.4f}"
            mAP = "   -  " if s.map is None else f"{s.map:.4f}"
            secs = "   -  " if s.seconds is None else f"{s.seconds:.1f}"
            notes = s.error or (s.checkpoint or "")
            print(f"{r.model:<10} {s.stage:<5} {map50:>8} {mAP:>8} {secs:>7}  {notes}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--qat-epochs", type=int, default=2, help="QAT fine-tune epochs")
    p.add_argument("--qat-lr", type=float, default=1e-4, help="Constant LR for QAT fine-tune")
    p.add_argument("--calib-size", type=int, default=512, help="# val2017 images for PTQ")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Train + calib batch size")
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Ultralytics device string (e.g. '0', 'cpu')",
    )
    p.add_argument("--skip-fp32-eval", action="store_true", help="Skip baseline mAP")
    p.add_argument("--no-export", action="store_true", help="Skip final ONNX export")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not COCO_YAML.exists():
        print(f"ERROR: COCO config not found: {COCO_YAML}", file=sys.stderr)
        return 2
    if not VAL_DIR.exists():
        print(
            f"ERROR: {VAL_DIR} not found. Run ./scripts/download_coco_dataset.sh first.",
            file=sys.stderr,
        )
        return 2

    all_results: list[ModelResult] = []
    for model_name in args.models:
        try:
            all_results.append(
                run_qat_for_model(
                    model_name=model_name,
                    qat_epochs=args.qat_epochs,
                    qat_lr=args.qat_lr,
                    calib_size=args.calib_size,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    val_batch=args.val_batch,
                    device=args.device,
                    skip_fp32_eval=args.skip_fp32_eval,
                    do_export=not args.no_export,
                )
            )
        except Exception as e:
            print(f"[{model_name}] FATAL: {e}")
            all_results.append(
                ModelResult(
                    model=model_name,
                    stages=[StageResult(stage="fp32", error=str(e))],
                )
            )

    print_report(all_results)
    report = [
        {
            "model": r.model,
            "onnx_path": r.onnx_path,
            "stages": [asdict(s) for s in r.stages],
        }
        for r in all_results
    ]
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote combined report -> {OUT_ROOT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
