#!/usr/bin/env python3
"""
Apply NVIDIA Model Optimizer (modelopt) to Ultralytics YOLO models.

Workflow
--------
1. Load an Ultralytics YOLO checkpoint (e.g. ``yolo11x.pt``, ``yolo26x.pt``).
2. Export the model to FP32 ONNX via Ultralytics.
3. Build a calibration dataset from the Ultralytics dataset YAML (COCO
   ``val2017`` by default) using the same preprocessing as Ultralytics.
4. Run NVIDIA ``modelopt.onnx.quantization`` post-training quantization
   (INT8 / FP8 / INT4) with the calibration data.
5. Validate the quantized ONNX model with Ultralytics
   (``model.val(data=<dataset.yaml>)``) to get mAP50 / mAP50-95.
6. Optionally run inference on a sample image source and save
   visualisations.

Usage
-----
    python examples/nvidia_modelopt_yolo.py \
        --models yolo11x yolo26x \
        --quant-modes int8 fp8 \
        --calib-size 256 \
        --imgsz 640

The script is intentionally self-contained: every step can be re-run
independently because the intermediate artifacts (FP32 ONNX, calibration
tensor, quantized ONNX) are cached on disk under ``./runs/modelopt``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

# Ultralytics / ONNX deps are required. Import lazily inside functions so the
# CLI ``--help`` works even if the environment is incomplete.


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COCO_YAML = PROJECT_ROOT / "configs" / "coco.yaml"
DEFAULT_DATA_YAML = COCO_YAML
COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
VAL_DIR = COCO_ROOT / "images" / "val2017"
TEST_DIR = COCO_ROOT / "images" / "test2017"
OUT_ROOT = PROJECT_ROOT / "runs" / "modelopt"

DEFAULT_MODELS = ("yolo11x", "yolo26x")
DEFAULT_MODES = ("int8",)
SUPPORTED_MODES = ("int8", "fp8", "int4")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_dataset_yaml(data_yaml: Path) -> dict[str, Any]:
    import yaml

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")
    return data


def _resolve_existing_path(path: Path, data_yaml: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path),
        (PROJECT_ROOT / path),
        (data_yaml.parent / path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _first_split_value(value: Any, split: str) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Dataset split {split!r} is empty")
        value = value[0]
    if value is None:
        raise KeyError(f"Dataset YAML has no {split!r} split")
    return str(value)


def resolve_dataset_split(data_yaml: Path, split: str) -> Path:
    cfg = _load_dataset_yaml(data_yaml)
    root_value = cfg.get("path", "")
    root = _resolve_existing_path(Path(str(root_value)).expanduser(), data_yaml) if root_value else data_yaml.parent
    split_value = Path(_first_split_value(cfg.get(split), split)).expanduser()
    if split_value.is_absolute():
        return split_value
    candidates = [root / split_value, PROJECT_ROOT / split_value, data_yaml.parent / split_value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_calib_source(source: str, data_yaml: Path) -> Path:
    if source in ("train", "val"):
        return resolve_dataset_split(data_yaml, source)
    if source in ("val2017", "train2017") and data_yaml.resolve() == DEFAULT_DATA_YAML.resolve():
        return VAL_DIR if source == "val2017" else COCO_ROOT / "images" / "train2017"
    candidate = Path(source).expanduser()
    if candidate.is_absolute() or candidate.exists() or any(sep in source for sep in ("/", "\\")):
        return _resolve_existing_path(candidate, data_yaml)
    if source.endswith("2017") and data_yaml.resolve() != DEFAULT_DATA_YAML.resolve():
        split = "train" if source.startswith("train") else "val" if source.startswith("val") else None
        if split:
            return resolve_dataset_split(data_yaml, split)
    raise ValueError("Unknown --calib-source {!r}; expected train, val, val2017, train2017, or a path".format(source))


def _image_paths_from_source(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Image source not found: {source}")
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    paths: list[Path] = []
    for raw in source.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        if not p.is_absolute():
            candidates = [source.parent / p, PROJECT_ROOT / p, Path.cwd() / p]
            p = next((c for c in candidates if c.exists()), candidates[0])
        paths.append(p.resolve())
    return [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]


# ---------------------------------------------------------------------------
# Calibration data loader
# ---------------------------------------------------------------------------


def _letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114) -> np.ndarray:
    """Resize an image to ``new_shape`` keeping aspect ratio, then pad.

    Matches Ultralytics' inference preprocessing (letterbox + RGB + 0-1 float).
    """
    import cv2  # imported here to keep ``--help`` usable without OpenCV.

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3
    )
    return padded


def build_calibration_array(
    image_source: Path,
    num_samples: int,
    imgsz: int,
    cache_path: Path,
    seed: int = 0,
) -> np.ndarray:
    """Build a ``(N, 3, H, W)`` float32 NCHW tensor for calibration.

    Caches the result to ``cache_path`` so subsequent runs skip decoding.
    """
    import cv2

    if cache_path.exists():
        print(f"  [calib] loading cached calibration tensor: {cache_path}")
        return np.load(cache_path)

    paths = _image_paths_from_source(image_source)
    if not paths:
        raise RuntimeError(f"No calibration images found in {image_source}")

    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:num_samples]

    print(f"  [calib] preprocessing {len(paths)} images at {imgsz}x{imgsz}")
    out = np.empty((len(paths), 3, imgsz, imgsz), dtype=np.float32)
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        padded = _letterbox(rgb, new_shape=imgsz)
        chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = chw

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    print(f"  [calib] cached calibration tensor -> {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Model export + quantization
# ---------------------------------------------------------------------------


def export_fp32_onnx(model_name: str, imgsz: int, out_dir: Path) -> Path:
    """Export an Ultralytics YOLO model to FP32 ONNX. Returns the ONNX path."""
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{model_name}.onnx"
    if onnx_path.exists():
        print(f"  [export] reusing existing ONNX: {onnx_path}")
        return onnx_path

    weights = f"{model_name}.pt"
    print(f"  [export] loading {weights}")
    model = YOLO(weights)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=17,
        half=False,
    )
    exported_path = Path(exported)
    exported_path.replace(onnx_path)
    print(f"  [export] wrote {onnx_path}")
    return onnx_path


def quantize_onnx(
    fp32_path: Path,
    quant_mode: str,
    calib_array: np.ndarray,
    out_path: Path,
) -> Path:
    """Apply modelopt PTQ to an ONNX file. Returns the quantized path."""
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    if out_path.exists():
        print(f"  [quant/{quant_mode}] reusing existing quantized ONNX: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [quant/{quant_mode}] running modelopt PTQ -> {out_path}")
    t0 = time.time()

    # modelopt expects a dict mapping input-name -> numpy array for calibration
    # data. Discover the input name from the ONNX graph so we don't hard-code
    # "images".
    import onnx

    graph_inputs = [i.name for i in onnx.load(str(fp32_path)).graph.input]
    if not graph_inputs:
        raise RuntimeError(f"ONNX model {fp32_path} has no inputs")
    input_name = graph_inputs[0]
    calibration_data = {input_name: calib_array}

    modelopt_quantize(
        onnx_path=str(fp32_path),
        quantize_mode=quant_mode,
        calibration_data=calibration_data,
        calibration_method="entropy" if quant_mode == "int8" else "max",
        output_path=str(out_path),
    )
    print(f"  [quant/{quant_mode}] done in {time.time() - t0:.1f}s")
    return out_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    model: str
    variant: str  # "fp32" | "int8" | "fp8" | "int4"
    onnx_path: str
    size_mb: float
    map50: float | None = None
    map: float | None = None
    val_seconds: float | None = None
    error: str | None = None
    test_samples: list[dict] = field(default_factory=list)


def evaluate_on_coco_val(
    onnx_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> tuple[float, float, float]:
    """Run ``model.val(data=coco.yaml)`` on an ONNX file. Returns (map50, map, seconds)."""
    from ultralytics import YOLO

    model = YOLO(str(onnx_path), task="detect")
    t0 = time.time()
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.6,
        device=device,
        verbose=False,
        plots=False,
    )
    return float(results.box.map50), float(results.box.map), time.time() - t0


def run_inference_samples(
    onnx_path: Path,
    image_source: Path,
    num_samples: int,
    imgsz: int,
    device: str,
    out_dir: Path,
) -> list[dict]:
    """Run inference on ``num_samples`` random COCO test2017 images, save vis."""
    from ultralytics import YOLO

    if not image_source.exists():
        print(f"  [test] {image_source} not found; skipping inference samples.")
        return []

    test_images = _image_paths_from_source(image_source)
    if not test_images:
        print(f"  [test] no images in {image_source}; skipping")
        return []

    rng = random.Random(0)
    rng.shuffle(test_images)
    test_images = test_images[:num_samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(onnx_path), task="detect")

    results_summary: list[dict] = []
    for img_path in test_images:
        res = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
        )[0]
        save_path = out_dir / f"{img_path.stem}.jpg"
        res.save(filename=str(save_path))
        boxes = res.boxes
        results_summary.append(
            {
                "image": img_path.name,
                "num_detections": 0 if boxes is None else int(len(boxes)),
                "annotated": save_path.name,
            }
        )
    print(f"  [test] wrote {len(results_summary)} annotated samples to {out_dir}")
    return results_summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def process_model(
    model_name: str,
    quant_modes: Iterable[str],
    data_yaml: Path,
    calib_source: Path,
    imgsz: int,
    calib_size: int,
    val_batch: int,
    test_samples: int,
    test_source: Path,
    device: str,
    skip_val: bool,
) -> list[EvalResult]:
    print(f"\n=== {model_name} ===")
    model_dir = OUT_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. FP32 ONNX export (shared by all quant variants + serves as baseline)
    fp32_onnx = export_fp32_onnx(model_name, imgsz, model_dir)

    # 2. Calibration tensor (shared across quant modes; imgsz-specific cache)
    safe_source = "".join(c if c.isalnum() else "_" for c in str(calib_source))[-80:]
    calib_cache = OUT_ROOT / f"calib_{safe_source}_n{calib_size}_imgsz{imgsz}.npy"
    calib_array = build_calibration_array(calib_source, calib_size, imgsz, calib_cache)

    results: list[EvalResult] = []

    # 3. Baseline FP32 evaluation
    fp32_result = EvalResult(
        model=model_name,
        variant="fp32",
        onnx_path=str(fp32_onnx),
        size_mb=_size_mb(fp32_onnx),
    )
    if not skip_val:
        try:
            map50, mAP, secs = evaluate_on_coco_val(fp32_onnx, data_yaml, imgsz, val_batch, device)
            fp32_result.map50 = map50
            fp32_result.map = mAP
            fp32_result.val_seconds = secs
            print(
                f"  [val/fp32] mAP50={map50:.4f} mAP50-95={mAP:.4f} "
                f"({secs:.1f}s, {fp32_result.size_mb:.1f} MB)"
            )
        except Exception as e:
            fp32_result.error = f"val failed: {e}"
            print(f"  [val/fp32] failed: {e}")

    if test_samples > 0:
        fp32_result.test_samples = run_inference_samples(
            fp32_onnx, test_source, test_samples, imgsz, device, model_dir / "test_fp32"
        )
    results.append(fp32_result)

    # 4. For each quant mode: quantize + eval
    for mode in quant_modes:
        q_out = model_dir / f"{model_name}_{mode}.onnx"
        result = EvalResult(
            model=model_name, variant=mode, onnx_path=str(q_out), size_mb=0.0
        )
        try:
            quantize_onnx(fp32_onnx, mode, calib_array, q_out)
            result.size_mb = _size_mb(q_out)
            if not skip_val:
                map50, mAP, secs = evaluate_on_coco_val(q_out, data_yaml, imgsz, val_batch, device)
                result.map50 = map50
                result.map = mAP
                result.val_seconds = secs
                print(
                    f"  [val/{mode}] mAP50={map50:.4f} mAP50-95={mAP:.4f} "
                    f"({secs:.1f}s, {result.size_mb:.1f} MB)"
                )
            if test_samples > 0:
                result.test_samples = run_inference_samples(
                    q_out, test_source, test_samples, imgsz, device, model_dir / f"test_{mode}"
                )
        except Exception as e:
            result.error = str(e)
            print(f"  [{mode}] FAILED: {e}")
        results.append(result)

    # 5. Persist per-model summary
    summary_path = model_dir / "summary.json"
    summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"  [out] summary -> {summary_path}")
    return results


def print_report(all_results: list[EvalResult]) -> None:
    print("\n" + "=" * 78)
    print("NVIDIA ModelOpt x YOLO / COCO summary")
    print("=" * 78)
    header = f"{'model':<10} {'variant':<6} {'size MB':>8} {'mAP50':>7} {'mAP':>7}  notes"
    print(header)
    print("-" * len(header))
    for r in all_results:
        map50 = "  -  " if r.map50 is None else f"{r.map50:.4f}"
        mAP = "  -  " if r.map is None else f"{r.map:.4f}"
        notes = r.error or ""
        print(
            f"{r.model:<10} {r.variant:<6} {r.size_mb:>8.1f} {map50:>7} {mAP:>7}  {notes}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Ultralytics model stems (default: {' '.join(DEFAULT_MODELS)})",
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Ultralytics dataset YAML for validation")
    p.add_argument(
        "--quant-modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=SUPPORTED_MODES,
        help=f"Quantization modes to run (default: {' '.join(DEFAULT_MODES)})",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument(
        "--calib-size",
        type=int,
        default=256,
        help="Number of images used for PTQ calibration",
    )
    p.add_argument(
        "--calib-source",
        default="val2017",
        help="PTQ calibration source: train, val, val2017, train2017, or an explicit image directory/list path",
    )
    p.add_argument("--val-batch", type=int, default=8, help="Batch size for val()")
    p.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="If > 0, run inference on this many random test2017 images and save vis.",
    )
    p.add_argument(
        "--test-source",
        type=Path,
        default=TEST_DIR,
        help="Image directory/list used by --test-samples",
    )
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device passed to Ultralytics (e.g. '0', 'cpu')",
    )
    p.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip COCO mAP evaluation (useful for a quick quantization smoke test)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data_yaml = args.data
    if not data_yaml.exists():
        print(f"ERROR: dataset config not found: {data_yaml}", file=sys.stderr)
        return 2
    try:
        calib_source = resolve_calib_source(args.calib_source, data_yaml)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not calib_source.exists():
        print(f"ERROR: calibration source not found: {calib_source}", file=sys.stderr)
        return 2
    test_source = _resolve_existing_path(args.test_source.expanduser(), data_yaml)

    all_results: list[EvalResult] = []
    for model_name in args.models:
        try:
            all_results.extend(
                process_model(
                    model_name=model_name,
                    quant_modes=args.quant_modes,
                    data_yaml=data_yaml,
                    calib_source=calib_source,
                    imgsz=args.imgsz,
                    calib_size=args.calib_size,
                    val_batch=args.val_batch,
                    test_samples=args.test_samples,
                    test_source=test_source,
                    device=args.device,
                    skip_val=args.skip_val,
                )
            )
        except Exception as e:
            print(f"[{model_name}] FATAL: {e}")
            all_results.append(
                EvalResult(model=model_name, variant="fp32", onnx_path="", size_mb=0.0, error=str(e))
            )

    print_report(all_results)
    report_path = OUT_ROOT / "report.json"
    report_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    print(f"\nWrote combined report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
