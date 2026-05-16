#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for Ultralytics YOLO with NVIDIA ModelOpt.

Mirrors the structure of ``NVIDIA/Model-Optimizer/examples/cnn_qat/``
(PTQ calibration -> mtq.quantize -> QAT fine-tune -> mto.save -> export),
adapted for object detection:

* Calibration uses images from the Ultralytics dataset YAML (COCO ``val2017``
  by default) with Ultralytics' inference preprocessing (letterbox + RGB +
  0-1 float, NCHW).
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
    python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
        --models yolo11x yolo26x \
        --qat-epochs 10 \
        --qat-batches-per-epoch 200 \
        --calib-size 260 \
        --calib-method entropy \
        --imgsz 640 \
        --batch 10

    # PTQ-only sensitivity sweep (optionally ``--from-ptq`` to load a prior ``*_ptq.pth``)::

        python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py sensitivity --models yolo11n

Every stage is resumable: FP32 / PTQ / QAT checkpoints, calibration tensors
and per-model summaries are cached under ``runs/modelopt_qat/``.
Detect-head output int8 is skipped via :func:`apply_detect_output_ignore_policy`
(``--disable-detect-output-quant``), not ad-hoc ``.disable()`` on modules.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COCO_YAML = PROJECT_ROOT / "configs" / "coco.yaml"
DEFAULT_DATA_YAML = COCO_YAML
COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
VAL_DIR = COCO_ROOT / "images" / "val2017"
TRAIN_DIR = COCO_ROOT / "images" / "train2017"
OUT_ROOT = PROJECT_ROOT / "runs" / "modelopt_qat"

DEFAULT_MODELS = ("yolo11x", "yolo26x")
CALIB_SOURCES = {"val": VAL_DIR, "val2017": VAL_DIR, "train": TRAIN_DIR, "train2017": TRAIN_DIR}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# QAT recipe registry.
#
# Each entry holds the *default overrides* a recipe applies on top of the
# argparse defaults. Only keys the user did not explicitly pass on the CLI
# are overwritten (see :func:`_apply_recipe`). The `auto` recipe picks one
# per model family: E2E YOLO26 stays on the distillation recipe that earned
# the recorded yolo26s baseline; single-head models (YOLOv8/v11/v12) switch
# to a variant that *keeps the DFL head and Detect output quantizers in
# floating-point* during INT8 quantization (those layers are particularly
# noise-sensitive on single-head architectures).
#
# References:
#   - NVIDIA ModelOpt PyTorch quantization guide: QAT for ~10% of original
#     training epochs; quantizer scales frozen during QAT.
#   - ubonpartners/ultralytics UBON_QAT.md (commit 94046a6): documents the
#     DFL + Detect-output exclusion + max calibration pattern. The underlying
#     observation (DFL bin probabilities don't survive INT8) is independent
#     of that fork.
QAT_RECIPES: dict[str, dict] = {
    "yolo26-distill": {
        # Preserves the recorded yolo26s baseline exactly. Do not modify
        # without re-running the full yolo26s metrics table.
        "qat_epochs": 10,
        "qat_batches_per_epoch": 200,
        "qat_lr": 1e-5,
        "qat_low_lr": 1e-6,
        "qat_distill_weight": 1.0,
        "qat_supervised_weight": 1.0,
        "calib_method": "entropy",
        "calib_size": 260,
        "disable_detect_output_quant": False,
        "exclude_dfl_quant": False,
    },
    "yolo11-supervised": {
        # Supervised-only fine-tune, max calibration, DFL + Detect-output
        # quantizers excluded. Pattern lifted from the ubonpartners
        # UBON_QAT.md reference. Targets single-head v8/v11/v12-class models
        # where PTQ is already near-lossless and the distillation recipe
        # over-fits.
        #
        # Empirical note (2026-05-16): on yolo11s this recipe's lr=1e-4 was
        # catastrophic (epoch 2 collapsed to mAP50-95=0.3807). Kept as an
        # explicit option for cases where supervised-only with high LR is
        # appropriate (e.g. larger PTQ-FP32 gap), but `yolo11-distill` is
        # the recommended default for single-head models.
        "qat_epochs": 5,
        "qat_batches_per_epoch": 200,
        "qat_lr": 1e-4,
        "qat_low_lr": 1e-5,
        "qat_distill_weight": 0.0,
        "qat_supervised_weight": 1.0,
        "calib_method": "max",
        "calib_size": 1024,
        "disable_detect_output_quant": True,
        "exclude_dfl_quant": True,
    },
    "yolo11-distill": {
        # Hybrid: keep yolo26-distill's loss schedule (both teacher MSE and
        # COCO supervised at lr=1e-5 with low/high/low ladder), but **keep
        # DFL and Detect output quantizers in floating-point** (the same
        # exclusions that landed for yolo11-supervised) and switch to `max`
        # calibration. Targets single-head v8/v11/v12 where PTQ alone is
        # already near-lossless: the exclusions lift PTQ by ~+0.001 mAP50-95,
        # and the conservative distillation schedule lets QAT match-or-exceed
        # that baseline instead of collapsing it.
        "qat_epochs": 10,
        "qat_batches_per_epoch": 200,
        "qat_lr": 1e-5,
        "qat_low_lr": 1e-6,
        "qat_distill_weight": 1.0,
        "qat_supervised_weight": 1.0,
        "calib_method": "max",
        "calib_size": 1024,
        "disable_detect_output_quant": True,
        "exclude_dfl_quant": True,
    },
}


def _resolve_recipe(name: str, model_name: str | None, is_e2e: bool | None) -> str:
    """Map ``name`` (possibly ``'auto'``) to a concrete recipe key.

    Resolution order:
      * explicit recipe name -> use it.
      * ``auto`` + caller-provided ``is_e2e`` flag -> distill for E2E, supervised otherwise.
      * ``auto`` + ``model_name`` only -> use a name heuristic (``yolo26*`` is E2E).
      * fallback -> ``yolo26-distill`` (preserves existing default behaviour).
    """
    if name != "auto":
        if name not in QAT_RECIPES:
            raise ValueError(f"Unknown QAT recipe: {name!r}. Choices: {sorted(QAT_RECIPES)}")
        return name
    # Single-head models default to the safer distill variant: empirically
    # supervised-only at lr=1e-4 collapsed yolo11s mAP, so we pick the loss
    # schedule that anchored yolo26s and combine it with single-head PTQ
    # exclusions. yolo26-supervised stays on the recorded distill recipe.
    if is_e2e is not None:
        return "yolo26-distill" if is_e2e else "yolo11-distill"
    if model_name is not None:
        return "yolo26-distill" if model_name.lower().startswith("yolo26") else "yolo11-distill"
    return "yolo26-distill"


def _apply_recipe(args, explicit_flags: set[str]) -> str:
    """Apply ``args.qat_recipe`` defaults in-place for any flag the user did not pass.

    Returns the resolved recipe name (``yolo26-distill`` or ``yolo11-supervised``).
    ``explicit_flags`` is the set of long flag names (``"--qat-lr"`` etc.) that
    appeared on the command line — we only fill in values for flags absent from
    that set, so explicit CLI flags always win.
    """
    # For --qat-recipe=auto we need the model name (the first --models entry)
    # to pick between the two named recipes.
    model_hint = args.models[0] if getattr(args, "models", None) else None
    resolved = _resolve_recipe(args.qat_recipe, model_hint, is_e2e=None)
    overrides = QAT_RECIPES[resolved]
    flag_map = {
        "qat_epochs": "--qat-epochs",
        "qat_batches_per_epoch": "--qat-batches-per-epoch",
        "qat_lr": "--qat-lr",
        "qat_low_lr": "--qat-low-lr",
        "qat_distill_weight": "--qat-distill-weight",
        "qat_supervised_weight": "--qat-supervised-weight",
        "calib_method": "--calib-method",
        "calib_size": "--calib-size",
        "disable_detect_output_quant": "--disable-detect-output-quant",
        "exclude_dfl_quant": "--exclude-dfl-quant",
    }
    applied: list[str] = []
    for key, value in overrides.items():
        if flag_map[key] in explicit_flags:
            continue
        setattr(args, key, value)
        applied.append(f"{key}={value}")
    if applied:
        print(
            f"[recipe={resolved}] applied defaults: " + ", ".join(applied)
        )
    return resolved


def _pin_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


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
    """Resolve an Ultralytics dataset split (``train``/``val``) to a dir or image list."""
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
    """Resolve calibration source aliases or an explicit image dir/list path."""
    if source in ("train", "val"):
        return resolve_dataset_split(data_yaml, source)
    if source in CALIB_SOURCES and data_yaml.resolve() == DEFAULT_DATA_YAML.resolve():
        return CALIB_SOURCES[source]
    candidate = Path(source).expanduser()
    if candidate.is_absolute() or candidate.exists() or any(sep in source for sep in ("/", "\\")):
        return _resolve_existing_path(candidate, data_yaml)
    if source.endswith("2017") and data_yaml.resolve() != DEFAULT_DATA_YAML.resolve():
        split = "train" if source.startswith("train") else "val" if source.startswith("val") else None
        if split:
            return resolve_dataset_split(data_yaml, split)
    valid = "train, val, val2017, train2017, or a path to an image directory/list"
    raise ValueError(f"Unknown --calib-source {source!r}; expected {valid}")


def _image_paths_from_source(source: Path) -> list[Path]:
    """Collect calibration images from a directory or a newline-delimited image list."""
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
    image_source: Path,
    imgsz: int,
    batch: int,
    num_samples: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Return a list of CPU float32 NCHW batches for PTQ calibration."""
    import cv2

    paths = _image_paths_from_source(image_source)
    if not paths:
        raise RuntimeError(f"No calibration images found in {image_source}")

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


def eval_on_dataset(yolo, data_yaml: Path, imgsz: int, batch: int, device: str) -> tuple[float, float, float]:
    """Return (mAP50, mAP50-95, seconds) via Ultralytics' detection validator."""
    t0 = time.time()
    results = yolo.val(
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


def export_onnx(yolo, imgsz: int, dst: Path, device: str, use_modelopt_export: bool) -> Path | None:
    """Export the (QAT) model to ONNX for deployment. Non-fatal on failure."""
    try:
        if use_modelopt_export:
            export_onnx_modelopt(yolo.model, imgsz, device, dst)
        else:
            exported = yolo.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=17)
            Path(exported).replace(dst)
        print(f"  [export] wrote {dst}")
        return dst
    except Exception as e:
        print(f"  [export] WARN: ONNX export failed: {e}")
        return None


def count_qdq_nodes(onnx_path: Path) -> tuple[int, int]:
    """Return (#QuantizeLinear, #DequantizeLinear) nodes in an ONNX graph."""
    import onnx

    model = onnx.load(str(onnx_path))
    quantize = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dequantize = sum(node.op_type == "DequantizeLinear" for node in model.graph.node)
    return quantize, dequantize


def export_onnx_modelopt(model: torch.nn.Module, imgsz: int, device: str, dst: Path) -> None:
    """Export the quantized model through ModelOpt (``export_onnx`` if available, else deploy ``get_onnx_bytes``)."""
    from ultralytics.nn.modules.head import Detect

    dummy = torch.zeros(
        1,
        3,
        imgsz,
        imgsz,
        device=torch.device(f"cuda:{device}" if device.isdigit() else device),
    )
    detect_state: list[tuple[torch.nn.Module, bool, bool, str | None, int | None, bool]] = []
    try:
        for module in model.modules():
            if isinstance(module, Detect):
                detect_state.append(
                    (
                        module,
                        getattr(module, "dynamic", False),
                        getattr(module, "export", False),
                        getattr(module, "format", None),
                        getattr(module, "max_det", None),
                        getattr(module, "xyxy", False),
                    )
                )
                module.dynamic = False
                module.export = True
                module.format = "onnx"
        model.eval()
        onnx_bytes = _get_modelopt_onnx_bytes(model, dummy, onnx_opset=17)
        dst.write_bytes(onnx_bytes)
    finally:
        for module, dynamic, export, fmt, max_det, xyxy in detect_state:
            module.dynamic = dynamic
            module.export = export
            module.format = fmt
            if max_det is not None:
                module.max_det = max_det
            module.xyxy = xyxy
    # ModelOpt names the graph IO 'x' / 'out'; Ultralytics' ONNX backend binds
    # tensors by name and expects 'images' / 'output0'. Rename so the exported
    # ONNX drops straight into `yolo val model=<path>.onnx`.
    _rename_onnx_io(dst, {"x": "images"}, {"out": "output0"})


def _rename_onnx_io(onnx_path: Path, input_map: dict[str, str], output_map: dict[str, str]) -> None:
    """Rewrite graph input/output names so the file matches Ultralytics conventions."""
    import onnx

    model = onnx.load(str(onnx_path))
    for graph_input in model.graph.input:
        if graph_input.name in input_map:
            graph_input.name = input_map[graph_input.name]
    for graph_output in model.graph.output:
        if graph_output.name in output_map:
            graph_output.name = output_map[graph_output.name]
    rename = {**input_map, **output_map}
    for node in model.graph.node:
        node.input[:] = [rename.get(n, n) for n in node.input]
        node.output[:] = [rename.get(n, n) for n in node.output]
    onnx.save(model, str(onnx_path))


def build_quant_cfg(
    mtq,
    calib_method: str,
    calib_percentile: float,
) -> dict:
    """Build a YOLO-friendly ModelOpt ``quant_cfg`` (wildcards for calibrators / smoothquant).

    Skipping ``Detect`` output int8 is not part of this dict: apply
    :func:`apply_detect_output_ignore_policy` so it is expressed as ModelOpt
    policy (``set_quantizer_attribute``) rather than mutating submodules.
    """
    if calib_method == "smoothquant":
        quant_cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    else:
        quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    if calib_method in {"entropy", "percentile"}:
        quant_cfg["algorithm"] = "max"
        quant_cfg["quant_cfg"]["*input_quantizer"] = {
            "num_bits": 8,
            "axis": None,
            "calibrator": "histogram",
        }
        quant_cfg["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": 8,
            "axis": 0,
            "calibrator": "max",
        }
        quant_cfg["calib_amax_method"] = calib_method
        if calib_method == "percentile":
            quant_cfg["calib_percentile"] = calib_percentile

    return quant_cfg


def build_detect_output_ignore_policy_filter(model: torch.nn.Module):
    """Return a name filter for ``set_quantizer_attribute`` (ModelOpt's config-level ignore policy).

    Disables only ``*output_quantizer`` submodules whose parent is ``ultralytics...Detect``,
    matching the intent of skipping head activations from INT8 Q/DQ without poking at modules by hand.
    """
    from ultralytics.nn.modules.head import Detect

    def _detect_output_q(quant_name: str) -> bool:
        if not quant_name.endswith("output_quantizer"):
            return False
        parent_name = quant_name.rsplit(".", 1)[0]
        if not parent_name:
            return False
        try:
            parent = model.get_submodule(parent_name)
        except (AttributeError, KeyError, ValueError):
            return False
        return isinstance(parent, Detect)

    return _detect_output_q


def apply_detect_output_ignore_policy(mtq, model: torch.nn.Module) -> int:
    """Apply ignore policy for YOLO ``Detect`` outputs using ``mtq.set_quantizer_attribute`` (not ad-hoc ``.disable()``)."""
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer

    flt = build_detect_output_ignore_policy_filter(model)
    n_match = 0
    for name, m in model.named_modules():
        if not isinstance(m, (TensorQuantizer, SequentialQuantizer)):
            continue
        if flt(name):
            n_match += 1
    if n_match:
        mtq.set_quantizer_attribute(model, flt, {"enable": False})
    return n_match


def manual_histogram_quantize(
    model: torch.nn.Module,
    calibrate,
    calib_method: str,
    calib_percentile: float,
) -> None:
    """Quantize a model and finish histogram calibration explicitly."""
    from modelopt.torch.quantization.model_calib import enable_stats_collection
    from modelopt.torch.quantization.nn import TensorQuantizer

    enable_stats_collection(model)
    calibrate(model)

    for module in model.modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue
        if module._calibrator is not None and not module._dynamic:
            if calib_method == "entropy":
                module.load_calib_amax("entropy")
            elif calib_method == "percentile":
                module.load_calib_amax("percentile", percentile=calib_percentile)
        if module.bias_calibrator is not None and module.bias_type == "static":
            module.load_calib_bias()
        module.enable()
        module.enable_quant()
        module.disable_calib()


def _residual_quant_attr(quant_cfg: dict) -> dict:
    """Return the activation quantizer attributes to reuse for residual adds."""
    attr = copy.deepcopy(quant_cfg["quant_cfg"]["*input_quantizer"])
    attr["enable"] = True
    return attr


def patch_residual_add_quantizers(model: torch.nn.Module, quant_cfg: dict) -> int:
    """Patch common Ultralytics raw tensor adds to pass through TensorQuantizers."""
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.utils.network import bind_forward_method
    from ultralytics.nn.modules.block import Bottleneck, CIB, GhostBottleneck, HGBlock, Residual, ResNetBlock

    residual_attr = _residual_quant_attr(quant_cfg)
    patched = 0

    def _make_quantizer() -> TensorQuantizer:
        quantizer = TensorQuantizer()
        quantizer.set_from_attribute_config(residual_attr)
        return quantizer

    def _patch_single_add_module(module: torch.nn.Module, forward_fn) -> bool:
        nonlocal patched
        if getattr(module, "_mopt_residual_add_patched", False):
            return False
        module.add_lhs_quantizer = _make_quantizer()
        module.add_rhs_quantizer = _make_quantizer()
        bind_forward_method(module, forward_fn, "_forward_no_residual_quant")
        module._mopt_residual_add_patched = True
        patched += 1
        return True

    def _quant_input(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(False):
            return x.clone()

    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        if not self.add:
            return y
        return self.add_lhs_quantizer(_quant_input(x)) + self.add_rhs_quantizer(_quant_input(y))

    def ghost_bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        shortcut = self.shortcut(x)
        return self.add_lhs_quantizer(_quant_input(y)) + self.add_rhs_quantizer(_quant_input(shortcut))

    def resnet_block_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv3(self.cv2(self.cv1(x)))
        shortcut = self.shortcut(x)
        return torch.nn.functional.relu(
            self.add_lhs_quantizer(_quant_input(y)) + self.add_rhs_quantizer(_quant_input(shortcut))
        )

    def hg_block_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        if not self.add:
            return y
        return self.add_lhs_quantizer(_quant_input(y)) + self.add_rhs_quantizer(_quant_input(x))

    def cib_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        if not self.add:
            return y
        return self.add_lhs_quantizer(_quant_input(x)) + self.add_rhs_quantizer(_quant_input(y))

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.m(x)
        return self.add_lhs_quantizer(_quant_input(x)) + self.add_rhs_quantizer(_quant_input(y))

    for module in model.modules():
        if isinstance(module, Bottleneck) and getattr(module, "add", False):
            _patch_single_add_module(module, bottleneck_forward)
        elif isinstance(module, GhostBottleneck):
            _patch_single_add_module(module, ghost_bottleneck_forward)
        elif isinstance(module, ResNetBlock):
            _patch_single_add_module(module, resnet_block_forward)
        elif isinstance(module, HGBlock) and getattr(module, "add", False):
            _patch_single_add_module(module, hg_block_forward)
        elif isinstance(module, CIB) and getattr(module, "add", False):
            _patch_single_add_module(module, cib_forward)
        elif isinstance(module, Residual):
            _patch_single_add_module(module, residual_forward)

    return patched


def unify_residual_add_scales(model: torch.nn.Module) -> int:
    """Force both branches of each patched residual add to share the same amax."""
    unified = 0
    for module in model.modules():
        if not getattr(module, "_mopt_residual_add_patched", False):
            continue
        lhs = getattr(module, "add_lhs_quantizer", None)
        rhs = getattr(module, "add_rhs_quantizer", None)
        if lhs is None or rhs is None or lhs.amax is None or rhs.amax is None:
            continue
        shared_amax = torch.maximum(lhs.amax, rhs.amax)
        lhs.amax = shared_amax
        rhs.amax = shared_amax
        unified += 1
    return unified


def manual_quantize_model(
    mtq,
    model: torch.nn.Module,
    quant_cfg: dict,
    calibrate,
    calib_method: str,
    calib_percentile: float,
    patch_residual_adds: bool,
    share_residual_add_scales: bool,
    disable_detect_output_quant: bool,
) -> tuple[int, int, int]:
    """Quantize a model manually so extra YOLO-specific hooks can run before calibration."""
    from modelopt.torch.quantization.model_calib import max_calibrate

    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, quant_cfg["quant_cfg"])

    patched_residual_adds = 0
    if patch_residual_adds:
        patched_residual_adds = patch_residual_add_quantizers(model, quant_cfg)

    disabled_detect_outputs = 0
    if disable_detect_output_quant:
        disabled_detect_outputs = apply_detect_output_ignore_policy(mtq, model)

    if calib_method == "max":
        max_calibrate(model, calibrate)
    elif calib_method in {"entropy", "percentile"}:
        manual_histogram_quantize(model, calibrate, calib_method, calib_percentile)
    else:
        raise ValueError(f"Unsupported manual calibration method: {calib_method}")

    unified_scales = 0
    if share_residual_add_scales:
        unified_scales = unify_residual_add_scales(model)

    return patched_residual_adds, unified_scales, disabled_detect_outputs


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
    sensitivity: list[dict] | None = None
    qat_train: list[dict] | None = None
    best_checkpoint: str | None = None
    best_map: float | None = None
    config: dict | None = None


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


def _resolve_device(device: str) -> torch.device:
    return torch.device(f"cuda:{device}" if device.isdigit() else device)


def _raise_modelopt_import_error(exc: Exception) -> None:
    raise RuntimeError(
        f"modelopt.torch sub-module failed to import: {exc!r}. "
        "The current venv is missing one of ModelOpt's optional torch extras. "
        "Known dependencies the export path pulls in: 'pulp', 'huggingface_hub', "
        "'onnxconverter_common'. Install whichever module is named in the error "
        "above; pip install --no-build-isolation --extra-index-url "
        "https://pypi.ngc.nvidia.com 'nvidia-modelopt[torch,onnx]' is the catch-all."
    ) from exc


def _import_modelopt_qat_modules():
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except Exception as exc:
        _raise_modelopt_import_error(exc)
    return mto, mtq


def _get_modelopt_onnx_bytes(
    model: torch.nn.Module, dummy: torch.Tensor, *, onnx_opset: int
) -> bytes:
    """Resolve ONNX bytes from ModelOpt: use ``modelopt.torch.export.export_onnx`` when present, else deploy ``get_onnx_bytes``."""
    try:
        from modelopt.torch import export as mte

        export_fn = getattr(mte, "export_onnx", None)
        if export_fn is not None:
            try:
                out = export_fn(model, dummy, onnx_opset=onnx_opset)
            except TypeError:
                out = export_fn(model, dummy, opset_version=onnx_opset)
            if isinstance(out, (bytes, bytearray, memoryview)):
                return bytes(out)
    except Exception:
        pass
    try:
        from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes
    except Exception as exc:
        _raise_modelopt_import_error(exc)
    return get_onnx_bytes(model, dummy, onnx_opset=onnx_opset)


def _teacher_student_raw_outputs(outputs) -> list[torch.Tensor]:
    """Normalize YOLO detect outputs to raw multi-scale tensors for distillation.

    Used as a single-input helper for the legacy/back-compat path. The distillation
    loss path uses :func:`_coalign_distill_outputs` instead so that teacher and student
    are forced to the same YOLO26 head (``one2one`` vs ``one2many``).
    """
    if isinstance(outputs, dict):
        if "one2many" in outputs:
            tensors = _teacher_student_raw_outputs(outputs["one2many"])
            if tensors:
                return tensors
        if "one2one" in outputs:
            tensors = _teacher_student_raw_outputs(outputs["one2one"])
            if tensors:
                return tensors
        tensors: list[torch.Tensor] = []
        for value in outputs.values():
            tensors.extend(_teacher_student_raw_outputs(value))
        return tensors
    if isinstance(outputs, list) and all(torch.is_tensor(x) for x in outputs):
        return outputs
    if isinstance(outputs, tuple):
        for item in reversed(outputs):
            if isinstance(item, list) and all(torch.is_tensor(x) for x in item):
                return item
    if isinstance(outputs, list):
        tensors = []
        for item in outputs:
            tensors.extend(_teacher_student_raw_outputs(item))
        return tensors
    if torch.is_tensor(outputs):
        return [outputs]
    raise TypeError(f"Unsupported YOLO output type for distillation: {type(outputs)!r}")


def _extract_yolo_branch_tensors(outputs, branch: str) -> list[torch.Tensor]:
    """Pull tensors from a specific YOLO26 head (``one2one`` or ``one2many``).

    Returns an empty list if the branch is absent or empty (the quantized student
    can present an empty ``one2many`` dict).
    """
    if not isinstance(outputs, dict) or branch not in outputs:
        return []
    branch_out = outputs[branch]
    if isinstance(branch_out, dict) and not branch_out:
        return []
    return _teacher_student_raw_outputs(branch_out)


def _coalign_distill_outputs(
    student_outputs, teacher_outputs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Co-align teacher and student to the same YOLO26 head.

    ``one2one`` is preferred because both the FP32 teacher and the quantized student
    populate it; the quantized student's ``one2many`` is empty, which previously caused
    a silent branch mismatch (teacher one2many vs student one2one) that flowed through
    MSE and degraded mAP.
    """
    for branch in ("one2one", "one2many"):
        s = _extract_yolo_branch_tensors(student_outputs, branch)
        t = _extract_yolo_branch_tensors(teacher_outputs, branch)
        if s and t and len(s) == len(t) and all(si.shape == ti.shape for si, ti in zip(s, t)):
            return s, t
    s_fallback = _teacher_student_raw_outputs(student_outputs)
    t_fallback = _teacher_student_raw_outputs(teacher_outputs)
    return s_fallback, t_fallback


def _mse_distill_loss(student_outputs, teacher_outputs) -> torch.Tensor:
    """Per-tensor-normalized MSE so feature maps don't dominate boxes/scores.

    The previous implementation summed un-normalized ``mse_loss`` across tensors;
    ``feats`` maps have ~10^5 elements while ``boxes``/``scores`` are ~10^4-10^5
    but with very different magnitudes, so feature reconstruction dominated and the
    decoded prediction signal was effectively ignored.
    """
    student_raw, teacher_raw = _coalign_distill_outputs(student_outputs, teacher_outputs)
    if len(student_raw) != len(teacher_raw):
        raise RuntimeError(
            f"Teacher/student raw output count mismatch: {len(student_raw)} vs {len(teacher_raw)}"
        )
    if not student_raw:
        raise RuntimeError("Distillation got empty output lists from both teacher and student")
    losses = []
    for s, t in zip(student_raw, teacher_raw):
        t_detached = t.detach()
        denom = t_detached.abs().mean().clamp(min=1e-6)
        losses.append(torch.nn.functional.mse_loss(s, t_detached) / denom)
    return sum(losses) / len(losses)


def _distill_epoch_lr(epoch: int, total_epochs: int, peak_lr: float, low_lr: float) -> float:
    """Use a simple low/high/low ladder similar to the YOLOv7 QAT schedule."""
    if total_epochs < 3:
        return peak_lr
    first_cut = max(1, total_epochs // 3)
    last_cut = max(first_cut + 1, total_epochs - first_cut)
    if epoch < first_cut or epoch >= last_cut:
        return low_lr
    return peak_lr


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def _freeze_batch_norm_stats(model: torch.nn.Module) -> int:
    """Keep BatchNorm running stats fixed while leaving affine params trainable."""
    frozen = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen += 1
    return frozen


def _clone_inference_buffers(model: torch.nn.Module) -> int:
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
        amax = module._buffers.get("_amax")
        if not torch.is_tensor(amax):
            continue
        with torch.inference_mode(False):
            module._buffers["_amax"] = amax.clone()
        cloned += 1
    return cloned


def _prepare_quantized_model_for_training(model: torch.nn.Module) -> tuple[int, int]:
    """Clone any inference-mode quantizer state before autograd sees it."""
    inference_buffers = _clone_inference_buffers(model)
    amax_states = _clone_quantizer_amax_state(model)
    return inference_buffers, amax_states


def _snapshot_amax_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot every quantizer's ``_amax`` buffer keyed by module path (detached, CPU)."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        amax = module._buffers.get("_amax") if hasattr(module, "_buffers") else None
        if torch.is_tensor(amax):
            snapshot[name] = amax.detach().float().cpu().clone()
    return snapshot


def _amax_drift_stats(
    initial: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Aggregate scale drift across the quantizer population.

    ``mean_amax``: mean(|amax|) over the current snapshot.
    ``rel_drift``: ||amax_now - amax_init||_2 / max(||amax_init||_2, eps), averaged per quantizer.
    ``max_rel_drift``: worst single-quantizer relative drift.
    """
    if not current:
        return {"count": 0, "mean_amax": 0.0, "rel_drift": 0.0, "max_rel_drift": 0.0}
    rel_drifts: list[float] = []
    abs_values: list[float] = []
    for name, amax_now in current.items():
        abs_values.append(float(amax_now.abs().mean()))
        amax_init = initial.get(name)
        if amax_init is None or amax_init.shape != amax_now.shape:
            continue
        denom = float(amax_init.norm().clamp(min=1e-12))
        rel_drifts.append(float((amax_now - amax_init).norm()) / denom)
    return {
        "count": len(current),
        "mean_amax": sum(abs_values) / max(1, len(abs_values)),
        "rel_drift": sum(rel_drifts) / max(1, len(rel_drifts)),
        "max_rel_drift": max(rel_drifts) if rel_drifts else 0.0,
    }


def find_dfl_module_paths(model: torch.nn.Module) -> list[str]:
    """Return ``named_modules()`` paths for every Ultralytics DFL module.

    DFL is the Distribution-Focal-Loss head used by YOLOv8 / YOLO11 / YOLO26 to
    convert distribution-over-bins outputs into continuous box-regression
    coordinates. Per the DFL-exclusion practice (and consistent with NVIDIA's
    sensitivity-to-quantization observations for similar heads), DFL weights
    and inputs should be excluded from INT8 quantization — quantizing the
    discrete bin probabilities causes outsized mAP regressions because tiny
    numerical errors compound across the cumulative-distribution decode.
    """
    paths: list[str] = []
    for name, module in model.named_modules():
        if type(module).__name__ == "DFL":
            paths.append(name)
        elif name.endswith(".dfl") or ".dfl." in name:
            # Defensive: catch DFL submodules even if class lookup misses.
            paths.append(name)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def disable_module_quantizers(model: torch.nn.Module, prefixes: list[str]) -> tuple[int, int]:
    """Disable every quantizer attached to modules whose ``named_modules()`` path matches a prefix.

    Returns ``(modules_matched, quantizers_disabled)``. A prefix matches the module path
    exactly *or* matches the prefix of a dotted path (so ``"model.16"`` disables
    ``"model.16.m.0.m.0"``'s quantizers too).
    """
    if not prefixes:
        return 0, 0
    matched_modules = 0
    disabled = 0
    for name, module in model.named_modules():
        if not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        module_disabled = 0
        for quantizer in _iter_module_quantizers(module):
            quantizer.disable()
            module_disabled += 1
        if module_disabled:
            matched_modules += 1
            disabled += module_disabled
    return matched_modules, disabled


def _restore_modelopt_checkpoint(mto, model: torch.nn.Module, path: Path, torch_device: torch.device) -> torch.nn.Module:
    """Restore a ModelOpt checkpoint, including local residual-add quantizer buffers."""
    objs = torch.load(path, map_location=str(torch_device), weights_only=False)
    restored = mto.restore_from_modelopt_state(model, objs["modelopt_state"])
    state_dict = objs["model_state_dict"]
    current_keys = set(restored.state_dict().keys())
    for key, value in state_dict.items():
        if key in current_keys or not key.endswith("._amax"):
            continue
        module_name = key[: -len("._amax")]
        try:
            module = restored.get_submodule(module_name)
        except (AttributeError, ValueError):
            continue
        if "_amax" not in module._buffers:
            module.register_buffer("_amax", value.detach().clone())
    restored.load_state_dict(state_dict)
    return restored


def _build_detection_trainer(
    weights: str,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = {
        "model": weights,
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "epochs": 1,
        "amp": True,
        "multi_scale": False,
        "rect": False,
        "plots": False,
        "save": False,
        "cache": False,
        "optimizer": "Adam",
        "close_mosaic": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }
    return DetectionTrainer(overrides=overrides)


def _is_e2e_yolo_model(model: torch.nn.Module) -> bool:
    """Detect whether ``model`` uses YOLO26's end-to-end dual-head structure.

    Single-head detectors (YOLOv8 / YOLO11 / YOLO12 / YOLO13) use ``v8DetectionLoss``;
    YOLO26 uses ``E2ELoss`` with ``.one2one`` and ``.one2many`` sub-criteria.
    The check initializes the criterion if needed but does not run a forward pass.
    Errors fall through to ``False`` so this can never block a legitimate run.
    """
    if getattr(model, "criterion", None) is None:
        try:
            model.criterion = model.init_criterion()
        except Exception:
            return False
    crit = model.criterion
    return hasattr(crit, "one2one") and hasattr(crit, "one2many")


def _supervised_yolo_loss(student_model, student_outputs, batch_data):
    """Compute YOLO supervised detection loss; for YOLO26 use the one2one head only.

    The quantized student's ``one2many`` dict is empty after PTQ restore (some
    Detect-head submodule got bypassed by ModelOpt's module replacement). The
    default ``E2ELoss`` criterion crashes on the empty branch, so we drive the
    supervised signal through ``one2one`` exclusively. The teacher MSE term in
    :func:`_mse_distill_loss` already regularizes the student toward FP32 features
    across the network, so dropping the ``one2many`` *training* loss is fine.
    """
    if getattr(student_model, "criterion", None) is None:
        student_model.criterion = student_model.init_criterion()
    crit = student_model.criterion
    if hasattr(crit, "one2one") and hasattr(crit, "one2many"):
        try:
            parsed = crit.one2many.parse_output(student_outputs)
        except Exception:
            parsed = student_outputs
        one2one = parsed.get("one2one") if isinstance(parsed, dict) else None
        if one2one is None or (isinstance(one2one, dict) and not one2one):
            return crit(student_outputs, batch_data)
        return crit.one2one.loss(one2one, batch_data)
    return crit(student_outputs, batch_data)


def run_distillation_qat(
    student_model: torch.nn.Module,
    teacher_weights: str,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    epochs: int,
    peak_lr: float,
    low_lr: float,
    batches_per_epoch: int,
    workers: int,
    distill_weight: float = 1.0,
    supervised_weight: float = 1.0,
    freeze_teacher_bn: bool | None = None,
    freeze_student_bn: bool = False,
    log_every: int = 0,
    eval_callback=None,
    eval_every: int = 0,
) -> list[dict]:
    """QAT loop combining COCO supervised detection loss with teacher-student MSE.

    Distillation alone (MSE on raw outputs) was empirically insufficient to bring
    the quantized student back to PTQ-level mAP — it converged to a plateau that
    sits below PTQ. Adding the supervised YOLO detection loss against the COCO
    labels provides the ground-truth anchor that pulls the student weights toward
    correctness, while the teacher MSE term regularizes them toward FP32 features.

    Returns a per-epoch log: ``[{epoch, lr, sup, sup_box, sup_cls, sup_dfl, mse,
    total, secs, amax_*, map50?, map?}]``. ``eval_callback(epoch_idx_1based)``,
    if provided, is invoked every ``eval_every`` epochs *and* on the final epoch,
    and may return ``{"map50": float, "map": float, "secs": float}`` which is
    folded into the corresponding epoch row.
    """
    from ultralytics import YOLO

    cloned_inference_buffers, cloned_amax_states = _prepare_quantized_model_for_training(student_model)
    if cloned_inference_buffers:
        print(f"  [qat] cloned {cloned_inference_buffers} inference buffer(s) before training")
    if cloned_amax_states:
        print(f"  [qat] cloned {cloned_amax_states} quantizer amax state tensor(s) before training")

    trainer = _build_detection_trainer(teacher_weights, data_yaml, imgsz, batch, device, workers)
    trainer.model = student_model
    trainer.device = _resolve_device(device)
    trainer.stride = max(int(student_model.stride.max() if hasattr(student_model, "stride") else 32), 32)
    trainer.set_model_attributes()
    dataloader = trainer.get_dataloader(trainer.data["train"], batch_size=batch, rank=-1, mode="train")

    teacher = YOLO(teacher_weights).model.to(trainer.device)
    teacher.train()  # required by both E2E and single-head YOLOs to emit raw multi-scale outputs

    # freeze_teacher_bn=None means "auto": single-head YOLOs (v8/v11/etc.) need
    # BN running stats frozen so the teacher reference is stable across batches;
    # YOLO26's BN drift is measured as +0.00003 mAP (noise) so we leave it alone
    # by default to preserve the recorded yolo26s baseline exactly.
    is_e2e = _is_e2e_yolo_model(student_model)
    resolved_freeze_teacher_bn = (
        freeze_teacher_bn if freeze_teacher_bn is not None else (not is_e2e)
    )
    print(f"  [qat] head type: {'E2E (one2one/one2many)' if is_e2e else 'single-head'}")
    frozen_teacher_bn = (
        _freeze_batch_norm_stats(teacher) if resolved_freeze_teacher_bn else 0
    )
    if frozen_teacher_bn:
        print(f"  [qat] froze {frozen_teacher_bn} teacher BatchNorm module(s)")
    elif resolved_freeze_teacher_bn:
        print("  [qat] requested teacher BatchNorm freeze, but found no BatchNorm modules")
    for p in teacher.parameters():
        p.requires_grad = False

    amp_enabled = trainer.device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if amp_enabled
        else torch.amp.GradScaler("cpu", enabled=False)
    )
    params = _trainable_parameters(student_model)
    if not params:
        raise RuntimeError("QAT student model has no trainable parameters")
    optimizer = torch.optim.Adam(params, lr=peak_lr)

    use_supervised = supervised_weight > 0.0
    use_distill = distill_weight > 0.0
    initial_amax = _snapshot_amax_state(student_model)
    if initial_amax:
        print(f"  [qat] tracking amax drift across {len(initial_amax)} quantizer(s)")

    training_log: list[dict] = []

    for epoch in range(epochs):
        student_model.train()
        frozen_student_bn = _freeze_batch_norm_stats(student_model) if freeze_student_bn else 0
        if epoch == 0 and frozen_student_bn:
            print(f"  [qat] froze {frozen_student_bn} student BatchNorm module(s)")
        elif epoch == 0 and freeze_student_bn:
            print("  [qat] requested student BatchNorm freeze, but found no BatchNorm modules")
        epoch_lr = _distill_epoch_lr(epoch, epochs, peak_lr, low_lr)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr

        running_distill = 0.0
        running_supervised = 0.0
        running_box = 0.0
        running_cls = 0.0
        running_dfl = 0.0
        seen = 0
        epoch_t0 = time.time()
        for step, batch_data in enumerate(dataloader):
            if step >= batches_per_epoch:
                break
            batch_data = trainer.preprocess_batch(batch_data)
            imgs = batch_data["img"].clone()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_outputs = teacher(imgs) if use_distill else None
            with torch.autocast(device_type=amp_device, enabled=amp_enabled):
                student_outputs = student_model(imgs)
                distill_loss = (
                    _mse_distill_loss(student_outputs, teacher_outputs)
                    if use_distill
                    else imgs.new_zeros(())
                )
                step_box = step_cls = step_dfl = 0.0
                if use_supervised:
                    sup_components, _ = _supervised_yolo_loss(student_model, student_outputs, batch_data)
                    # v8DetectionLoss.loss returns a per-component vector (box/cls/dfl);
                    # the trainer normally sums it before backward.
                    if sup_components.dim() > 0:
                        sup_loss = sup_components.sum()
                        if sup_components.numel() >= 3:
                            step_box = float(sup_components[0].detach().cpu())
                            step_cls = float(sup_components[1].detach().cpu())
                            step_dfl = float(sup_components[2].detach().cpu())
                    else:
                        sup_loss = sup_components
                else:
                    sup_loss = imgs.new_zeros(())
                loss = supervised_weight * sup_loss + distill_weight * distill_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            d = float(distill_loss.detach().cpu())
            s = float(sup_loss.detach().cpu())
            running_distill += d
            running_supervised += s
            running_box += step_box
            running_cls += step_cls
            running_dfl += step_dfl
            seen += 1

            if log_every and seen % log_every == 0:
                print(
                    f"  [qat/distill] epoch {epoch + 1}/{epochs} "
                    f"step {seen}/{batches_per_epoch} lr={epoch_lr:.2e} "
                    f"sup={s:.4f} mse={d:.4f}"
                )

        denom = max(1, seen)
        avg_d = running_distill / denom
        avg_s = running_supervised / denom
        avg_box = running_box / denom
        avg_cls = running_cls / denom
        avg_dfl = running_dfl / denom
        epoch_secs = time.time() - epoch_t0
        drift = _amax_drift_stats(initial_amax, _snapshot_amax_state(student_model))
        total = supervised_weight * avg_s + distill_weight * avg_d
        row: dict = {
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "sup": avg_s,
            "sup_box": avg_box,
            "sup_cls": avg_cls,
            "sup_dfl": avg_dfl,
            "mse": avg_d,
            "total": total,
            "secs": epoch_secs,
            "amax_count": drift["count"],
            "amax_mean": drift["mean_amax"],
            "amax_rel_drift": drift["rel_drift"],
            "amax_max_rel_drift": drift["max_rel_drift"],
        }
        print(
            f"  [qat/distill] epoch {epoch + 1}/{epochs} lr={epoch_lr:.2e} "
            f"sup={avg_s:.4f}(box={avg_box:.4f} cls={avg_cls:.4f} dfl={avg_dfl:.4f}) "
            f"mse={avg_d:.4f} secs={epoch_secs:.1f} "
            f"amax_drift={drift['rel_drift']:.4f}"
        )

        run_eval = eval_callback is not None and eval_every > 0 and (
            (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        )
        if run_eval:
            eval_result = eval_callback(epoch + 1) or {}
            for k in ("map50", "map", "secs"):
                if k in eval_result:
                    row[f"eval_{k}" if k == "secs" else k] = eval_result[k]

        training_log.append(row)
    return training_log


def _iter_module_quantizers(module: torch.nn.Module):
    for attr in (
        "input_quantizer",
        "output_quantizer",
        "weight_quantizer",
        "add_lhs_quantizer",
        "add_rhs_quantizer",
    ):
        quantizer = getattr(module, attr, None)
        if quantizer is not None:
            yield quantizer


def _select_sensitivity_candidates(
    candidates: list[tuple[str, torch.nn.Module]], max_layers: int
) -> list[tuple[str, torch.nn.Module]]:
    """Limit sensitivity candidates by sampling across model depth instead of truncating early layers."""
    if max_layers <= 0 or len(candidates) <= max_layers:
        return candidates
    if max_layers == 1:
        return [candidates[-1]]

    last_idx = len(candidates) - 1
    selected_indices = {
        round(i * last_idx / (max_layers - 1))
        for i in range(max_layers)
    }
    return [candidate for idx, candidate in enumerate(candidates) if idx in selected_indices]


def run_sensitivity_sweep(
    yolo,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    max_layers: int,
    baseline_map: float | None,
    baseline_map50: float | None,
) -> list[dict]:
    """Disable one quantized module at a time and rank modules by recovered mAP."""
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in yolo.model.named_modules():
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            continue
        quantizers = list(_iter_module_quantizers(module))
        if quantizers:
            candidates.append((name, module))

    candidates = _select_sensitivity_candidates(candidates, max_layers)

    results: list[dict] = []
    for name, module in candidates:
        quantizers = list(_iter_module_quantizers(module))
        for quantizer in quantizers:
            quantizer.disable()
        try:
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, batch, device)
            result = {
                "name": name,
                "map50": map50,
                "map": mAP,
                "seconds": secs,
                "delta_map50": None if baseline_map50 is None else map50 - baseline_map50,
                "delta_map": None if baseline_map is None else mAP - baseline_map,
            }
            print(
                f"  [sensitivity] {name} mAP50={map50:.4f} mAP50-95={mAP:.4f} "
                f"delta={result['delta_map'] if result['delta_map'] is not None else float('nan'):.4f}"
            )
            results.append(result)
        finally:
            for quantizer in quantizers:
                quantizer.enable()

    results.sort(key=lambda item: item["delta_map"] if item["delta_map"] is not None else -math.inf, reverse=True)
    return results


def perform_ptq_on_yolo(
    yolo,
    model_name: str,
    model_dir: Path,
    mto,
    mtq,
    torch_device: torch.device,
    calib_size: int,
    calib_method: str,
    calib_percentile: float,
    imgsz: int,
    batch: int,
    val_batch: int,
    device: str,
    patch_residual_adds: bool,
    share_residual_add_scales: bool,
    disable_detect_output_quant: bool,
    calib_dir: Path = VAL_DIR,
    data_yaml: Path = DEFAULT_DATA_YAML,
    keep_fp32_modules: list[str] | None = None,
    exclude_dfl_quant: bool = False,
) -> tuple[StageResult, int, int, int]:
    """Run PTQ (manual max/entropy/percentile or ``mtq.quantize`` for smoothquant) and return the PTQ stage + stats."""
    ptq_stage = StageResult(stage="ptq")
    print(f"  [calib] source={calib_dir.relative_to(PROJECT_ROOT) if calib_dir.is_relative_to(PROJECT_ROOT) else calib_dir}")
    calib_batches = build_calib_batches(calib_dir, imgsz, batch, calib_size)
    calibrate = _calibrate_fn_factory(calib_batches, torch_device, calib_size)

    quant_cfg = build_quant_cfg(mtq, calib_method, calib_percentile)
    patched_adds = 0
    unified_add_scales = 0
    disabled_detect_outputs = 0
    if calib_method in {"max", "entropy", "percentile"}:
        print(f"  [ptq] running manual quantization path ({calib_method})")
        patched_adds, unified_add_scales, disabled_detect_outputs = manual_quantize_model(
            mtq,
            yolo.model,
            quant_cfg,
            calibrate,
            calib_method,
            calib_percentile,
            patch_residual_adds=patch_residual_adds,
            share_residual_add_scales=share_residual_add_scales,
            disable_detect_output_quant=disable_detect_output_quant,
        )
    else:
        print(f"  [ptq] running mtq.quantize(..., {calib_method}, calibrate)")
        mtq.quantize(yolo.model, quant_cfg, calibrate)
        if patch_residual_adds:
            print("  [ptq] WARN: residual-add patching is only applied on the manual max/entropy/percentile paths")
        if disable_detect_output_quant:
            disabled_detect_outputs = apply_detect_output_ignore_policy(mtq, yolo.model)

    if patched_adds:
        print(f"  [ptq] patched {patched_adds} residual add module(s)")
    if unified_add_scales:
        print(f"  [ptq] unified scales across {unified_add_scales} residual add(s)")
    if disable_detect_output_quant:
        print(f"  [ptq] policy-disabled {disabled_detect_outputs} detect-head output quantizer(s)")

    if exclude_dfl_quant:
        dfl_paths = find_dfl_module_paths(yolo.model)
        dfl_mods, dfl_q = disable_module_quantizers(yolo.model, dfl_paths)
        print(
            f"  [ptq] dfl-exclude: disabled {dfl_q} quantizer(s) on "
            f"{dfl_mods} module(s) under {dfl_paths or '[]'}"
        )
    if keep_fp32_modules:
        kept_modules, kept_quantizers = disable_module_quantizers(yolo.model, keep_fp32_modules)
        print(
            f"  [ptq] kept-fp32: disabled {kept_quantizers} quantizer(s) on "
            f"{kept_modules} module(s) matching {keep_fp32_modules}"
        )

    ptq_ckpt = model_dir / f"{model_name}_ptq.pth"
    cloned_inference_buffers = _clone_inference_buffers(yolo.model)
    if cloned_inference_buffers:
        print(f"  [ptq] cloned {cloned_inference_buffers} inference buffer(s)")
    cloned_amax_states = _clone_quantizer_amax_state(yolo.model)
    if cloned_amax_states:
        print(f"  [ptq] cloned {cloned_amax_states} quantizer amax state tensor(s)")
    mto.save(yolo.model, str(ptq_ckpt))
    ptq_stage.checkpoint = str(ptq_ckpt)
    print(f"  [ptq] saved {ptq_ckpt}")

    map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
    ptq_stage.map50, ptq_stage.map, ptq_stage.seconds = map50, mAP, secs
    print(f"  [val/ptq] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")

    return ptq_stage, patched_adds, unified_add_scales, disabled_detect_outputs


def run_qat_for_model(
    model_name: str,
    qat_epochs: int,
    qat_lr: float,
    optimizer: str,
    qat_mode: str,
    qat_low_lr: float,
    qat_batches_per_epoch: int,
    qat_distill_weight: float,
    qat_supervised_weight: float,
    calib_size: int,
    calib_method: str,
    calib_percentile: float,
    imgsz: int,
    batch: int,
    val_batch: int,
    device: str,
    dataloader_workers: int,
    skip_fp32_eval: bool,
    do_export: bool,
    use_modelopt_export: bool,
    disable_detect_output_quant: bool,
    fuse_before_quant: bool,
    patch_residual_adds: bool,
    share_residual_add_scales: bool,
    freeze_teacher_bn: bool | None,
    freeze_student_bn: bool,
    sensitivity: bool,
    sensitivity_stage: str,
    sensitivity_max_layers: int,
    qat_eval_every: int = 0,
    qat_log_every: int = 0,
    calib_dir: Path = VAL_DIR,
    data_yaml: Path = DEFAULT_DATA_YAML,
    keep_fp32_modules: list[str] | None = None,
    exclude_dfl_quant: bool = False,
    seed: int | None = None,
    recipe: str | None = None,
    from_ptq: Path | None = None,
) -> ModelResult:
    from ultralytics import YOLO

    mto, mtq = _import_modelopt_qat_modules()

    print(f"\n=== {model_name} ===")
    model_dir = OUT_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    stages: list[StageResult] = []
    sensitivity_results: list[dict] | None = None

    # -------- 1. Load FP32 model --------
    weights = f"{model_name}.pt"
    print(f"  [load] {weights}")
    yolo = YOLO(weights)

    torch_device = _resolve_device(device)
    yolo.model.to(torch_device)
    if fuse_before_quant:
        print("  [prep] fusing Conv+BN before quantization")
        yolo.fuse()
        yolo.model.to(torch_device)

    # -------- 2. FP32 baseline mAP --------
    fp32_stage = StageResult(stage="fp32")
    if not skip_fp32_eval:
        try:
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
            fp32_stage.map50, fp32_stage.map, fp32_stage.seconds = map50, mAP, secs
            print(f"  [val/fp32] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
        except Exception as e:
            fp32_stage.error = str(e)
            print(f"  [val/fp32] FAILED: {e}")
    stages.append(fp32_stage)

    # -------- 3. PTQ calibration (mtq.quantize + calibrate closure) --------
    try:
        if from_ptq is not None:
            if not from_ptq.is_file():
                raise FileNotFoundError(f"--from-ptq is not a file: {from_ptq}")
            if patch_residual_adds:
                quant_cfg = build_quant_cfg(mtq, calib_method, calib_percentile)
                patched_adds = patch_residual_add_quantizers(yolo.model, quant_cfg)
                if patched_adds:
                    print(f"  [ptq] patched {patched_adds} residual add module(s) before restore")
            print(f"  [ptq] restoring {from_ptq}")
            yolo.model = _restore_modelopt_checkpoint(mto, yolo.model, from_ptq, torch_device)
            if exclude_dfl_quant:
                dfl_paths = find_dfl_module_paths(yolo.model)
                dfl_mods, dfl_q = disable_module_quantizers(yolo.model, dfl_paths)
                print(
                    f"  [ptq] dfl-exclude: disabled {dfl_q} quantizer(s) on "
                    f"{dfl_mods} module(s) under {dfl_paths or '[]'}"
                )
            if keep_fp32_modules:
                kept_modules, kept_quantizers = disable_module_quantizers(
                    yolo.model, keep_fp32_modules
                )
                print(
                    f"  [ptq] kept-fp32: disabled {kept_quantizers} quantizer(s) on "
                    f"{kept_modules} module(s) matching {keep_fp32_modules}"
                )
            ptq_stage = StageResult(stage="ptq", checkpoint=str(from_ptq))
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
            ptq_stage.map50, ptq_stage.map, ptq_stage.seconds = map50, mAP, secs
            print(f"  [val/ptq] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
        else:
            ptq_stage, _patched_adds, _unified_add_scales, _disabled_d = perform_ptq_on_yolo(
                yolo,
                model_name,
                model_dir,
                mto,
                mtq,
                torch_device,
                calib_size,
                calib_method,
                calib_percentile,
                imgsz,
                batch,
                val_batch,
                device,
                patch_residual_adds,
                share_residual_add_scales,
                disable_detect_output_quant,
                calib_dir=calib_dir,
                data_yaml=data_yaml,
                keep_fp32_modules=keep_fp32_modules,
                exclude_dfl_quant=exclude_dfl_quant,
            )
    except Exception as e:
        ptq_stage = StageResult(stage="ptq", error=str(e))
        print(f"  [ptq] FAILED: {e}")
        stages.append(ptq_stage)
        return ModelResult(model=model_name, stages=stages)
    stages.append(ptq_stage)

    if sensitivity and sensitivity_stage == "ptq":
        print(f"  [sensitivity] sweeping ptq modules (limit={sensitivity_max_layers})")
        sensitivity_results = run_sensitivity_sweep(
            yolo,
            data_yaml,
            imgsz,
            val_batch,
            device,
            max_layers=sensitivity_max_layers,
            baseline_map=ptq_stage.map,
            baseline_map50=ptq_stage.map50,
        )

    # -------- 4. QAT fine-tune via Ultralytics trainer --------
    qat_stage = StageResult(stage="qat")
    training_log: list[dict] = []
    best_state = {"map": None, "epoch": None, "checkpoint": None, "map50": None}
    best_ckpt_path = model_dir / f"{model_name}_qat_best.pth"

    def _qat_eval_callback(epoch_1based: int) -> dict | None:
        try:
            yolo.model.eval()
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
        except Exception as exc:  # eval is best-effort; don't kill the run
            print(f"  [qat/eval] epoch {epoch_1based} FAILED: {exc}")
            return None
        finally:
            # Restore training state after Ultralytics' val pipeline. Three
            # things can be left in a non-trainable state by val():
            #   1. module.training flags (need .train()),
            #   2. ModelOpt quantizer buffers marked inference-mode
            #      (need _prepare_quantized_model_for_training),
            #   3. parameter requires_grad flags (Ultralytics' validator path
            #      can flip them via AutoBackend / fuse interactions).
            # Restoring all three keeps the next training backward valid.
            yolo.model.train()
            _prepare_quantized_model_for_training(yolo.model)
            _trainable_parameters(yolo.model)
        print(
            f"  [qat/eval] epoch {epoch_1based} mAP50={map50:.4f} "
            f"mAP50-95={mAP:.4f} ({secs:.1f}s)"
        )
        if best_state["map"] is None or mAP > best_state["map"]:
            best_state.update(
                {"map": mAP, "map50": map50, "epoch": epoch_1based, "checkpoint": str(best_ckpt_path)}
            )
            try:
                mto.save(yolo.model, str(best_ckpt_path))
                print(f"  [qat/eval] new best @ epoch {epoch_1based}: saved {best_ckpt_path}")
            except Exception as exc:
                print(f"  [qat/eval] WARN: best checkpoint save failed: {exc}")
        return {"map50": map50, "map": mAP, "secs": secs}

    try:
        print(f"  [qat] fine-tuning for {qat_epochs} epoch(s) (mode={qat_mode}, lr0={qat_lr}, optimizer={optimizer})")
        if qat_mode == "distill":
            training_log = run_distillation_qat(
                student_model=yolo.model,
                teacher_weights=weights,
                data_yaml=data_yaml,
                imgsz=imgsz,
                batch=batch,
                device=device,
                epochs=qat_epochs,
                peak_lr=qat_lr,
                low_lr=qat_low_lr,
                batches_per_epoch=qat_batches_per_epoch,
                workers=dataloader_workers,
                distill_weight=qat_distill_weight,
                supervised_weight=qat_supervised_weight,
                freeze_teacher_bn=freeze_teacher_bn,
                freeze_student_bn=freeze_student_bn,
                log_every=qat_log_every,
                eval_callback=_qat_eval_callback if qat_eval_every > 0 else None,
                eval_every=qat_eval_every,
            )
        else:
            yolo.train(
                data=str(data_yaml),
                epochs=qat_epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=qat_lr,
                lrf=1.0,  # constant LR (Ultralytics treats lrf as ratio: final_lr = lr0 * lrf)
                warmup_epochs=0,
                optimizer=optimizer,
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

        map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
        qat_stage.map50, qat_stage.map, qat_stage.seconds = map50, mAP, secs
        print(f"  [val/qat] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as e:
        qat_stage.error = str(e)
        print(f"  [qat] FAILED: {e}")
        traceback.print_exc()
    stages.append(qat_stage)

    # -------- 5. ONNX export for deployment --------
    onnx_path: Path | None = None
    if do_export and qat_stage.error is None:
        onnx_path = export_onnx(
            yolo,
            imgsz,
            model_dir / f"{model_name}_qat.onnx",
            device=device,
            use_modelopt_export=use_modelopt_export,
        )
        if onnx_path is not None:
            try:
                q_nodes, dq_nodes = count_qdq_nodes(onnx_path)
                print(f"  [export] ONNX Q/DQ nodes: QuantizeLinear={q_nodes}, DequantizeLinear={dq_nodes}")
                if q_nodes == 0 or dq_nodes == 0:
                    print("  [export] WARN: exported ONNX does not appear to preserve explicit Q/DQ")
            except Exception as e:
                print(f"  [export] WARN: Q/DQ verification failed: {e}")

    if sensitivity and sensitivity_stage == "qat":
        if qat_stage.error is not None:
            print("  [sensitivity] WARN: QAT stage failed, skipping requested qat sensitivity sweep")
        else:
            print(f"  [sensitivity] sweeping qat modules (limit={sensitivity_max_layers})")
            sensitivity_results = run_sensitivity_sweep(
                yolo,
                data_yaml,
                imgsz,
                val_batch,
                device,
                max_layers=sensitivity_max_layers,
                baseline_map=qat_stage.map,
                baseline_map50=qat_stage.map50,
            )

    if training_log:
        _write_qat_train_csv(model_dir / "qat_train.csv", training_log)

    config_snapshot = {
        "seed": seed,
        "data": str(data_yaml),
        "calib_source": str(calib_dir),
        "calib_size": calib_size,
        "calib_method": calib_method,
        "calib_percentile": calib_percentile,
        "qat_epochs": qat_epochs,
        "qat_batches_per_epoch": qat_batches_per_epoch,
        "qat_lr": qat_lr,
        "qat_low_lr": qat_low_lr,
        "qat_distill_weight": qat_distill_weight,
        "qat_supervised_weight": qat_supervised_weight,
        "qat_eval_every": qat_eval_every,
        "qat_log_every": qat_log_every,
        "imgsz": imgsz,
        "batch": batch,
        "val_batch": val_batch,
        "disable_detect_output_quant": disable_detect_output_quant,
        "keep_fp32_modules": keep_fp32_modules or [],
        "exclude_dfl_quant": exclude_dfl_quant,
        "recipe": recipe,
        "freeze_teacher_bn": freeze_teacher_bn,
        "freeze_student_bn": freeze_student_bn,
    }
    result = ModelResult(
        model=model_name,
        stages=stages,
        onnx_path=str(onnx_path) if onnx_path else None,
        sensitivity=sensitivity_results,
        qat_train=training_log or None,
        best_checkpoint=best_state["checkpoint"],
        best_map=best_state["map"],
        config=config_snapshot,
    )
    (model_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": result.model,
                "onnx_path": result.onnx_path,
                "stages": [asdict(s) for s in result.stages],
                "sensitivity": result.sensitivity,
                "qat_train": result.qat_train,
                "best_checkpoint": result.best_checkpoint,
                "best_map": result.best_map,
                "best_map50": best_state["map50"],
                "best_epoch": best_state["epoch"],
                "config": result.config,
            },
            indent=2,
        )
    )
    return result


def _write_qat_train_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def parse_sensitivity_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-module PTQ sensitivity sweep: quantiles mAP when each quantized module is toggled off. "
        "Reuses the same COCO val setup as the main QAT script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Ultralytics checkpoint stem(s) (e.g. yolo11n)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Ultralytics dataset YAML for validation/training")
    p.add_argument("--calib-size", type=int, default=260, help="# images for PTQ calibration (unless --from-ptq)")
    p.add_argument(
        "--calib-method",
        choices=("max", "entropy", "percentile", "smoothquant"),
        default="entropy",
        help="PTQ path used before the sweep (ignored when --from-ptq is set)",
    )
    p.add_argument("--calib-percentile", type=float, default=99.99, help="For --calib-method=percentile")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--no-fuse-before-quant",
        action="store_true",
    )
    p.add_argument(
        "--disable-detect-output-quant",
        action="store_true",
    )
    p.add_argument(
        "--no-residual-add-quant",
        action="store_true",
    )
    p.add_argument(
        "--no-share-residual-add-scales",
        action="store_true",
    )
    p.add_argument(
        "--sensitivity-max-layers",
        type=int,
        default=24,
        help="Max quantized parent modules to evaluate, sampled across model depth (same as the main --sensitivity)",
    )
    p.add_argument(
        "--from-ptq",
        type=Path,
        default=None,
        help="Use an existing mto.save() PTQ checkpoint; requires a single --models entry and skips PTQ",
    )
    p.add_argument(
        "--calib-source",
        default="val2017",
        help="PTQ calibration source: train, val, val2017, train2017, or an explicit image directory/list path",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT / "sensitivity_report.json",
        help="JSON report path (merged for all --models)",
    )
    return p.parse_args(argv)


def run_sensitivity_subcommand(argv: list[str] | None = None) -> int:
    from ultralytics import YOLO

    args = parse_sensitivity_args(argv)
    data_yaml = args.data
    if not data_yaml.exists():
        print(f"ERROR: dataset config not found: {data_yaml}", file=sys.stderr)
        return 2
    try:
        calib_dir = resolve_calib_source(args.calib_source, data_yaml)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not calib_dir.exists():
        print(f"ERROR: calibration source not found: {calib_dir}", file=sys.stderr)
        return 2
    if args.from_ptq is not None and len(args.models) != 1:
        print("ERROR: --from-ptq requires exactly one --models name", file=sys.stderr)
        return 2

    mto, mtq = _import_modelopt_qat_modules()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for model_name in args.models:
        print(f"\n=== sensitivity: {model_name} ===")
        model_dir = OUT_ROOT / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = f"{model_name}.pt"
        yolo = YOLO(weights)
        torch_device = _resolve_device(args.device)
        yolo.model.to(torch_device)
        if not args.no_fuse_before_quant:
            yolo.fuse()
            yolo.model.to(torch_device)

        if args.from_ptq is not None:
            if not args.from_ptq.is_file():
                print(f"ERROR: --from-ptq is not a file: {args.from_ptq}", file=sys.stderr)
                return 2
            print(f"  [sensitivity] restoring PTQ from {args.from_ptq}")
            yolo.model = mto.restore(yolo.model, str(args.from_ptq), map_location=str(torch_device))
        else:
            try:
                perform_ptq_on_yolo(
                    yolo,
                    model_name,
                    model_dir,
                    mto,
                    mtq,
                    torch_device,
                    args.calib_size,
                    args.calib_method,
                    args.calib_percentile,
                    args.imgsz,
                    args.batch,
                    args.val_batch,
                    args.device,
                    patch_residual_adds=not args.no_residual_add_quant,
                    share_residual_add_scales=not args.no_share_residual_add_scales,
                    disable_detect_output_quant=args.disable_detect_output_quant,
                    calib_dir=calib_dir,
                    data_yaml=data_yaml,
                )
            except Exception as e:
                print(f"  [sensitivity] PTQ failed: {e}", file=sys.stderr)
                return 1

        map50, mAP, _ = eval_on_dataset(yolo, data_yaml, args.imgsz, args.val_batch, args.device)
        print(f"  [sensitivity] PTQ baseline mAP50={map50:.4f} mAP50-95={mAP:.4f}")
        sens = run_sensitivity_sweep(
            yolo,
            data_yaml,
            args.imgsz,
            args.val_batch,
            args.device,
            max_layers=args.sensitivity_max_layers,
            baseline_map=mAP,
            baseline_map50=map50,
        )
        results.append({"model": model_name, "ptq_map50": map50, "ptq_map": mAP, "sensitivity": sens})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote sensitivity report -> {args.out}")
    return 0


def parse_qat_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Ultralytics dataset YAML for validation and QAT training")
    p.add_argument("--qat-epochs", type=int, default=10, help="QAT fine-tune epochs")
    p.add_argument("--qat-lr", type=float, default=1e-5, help="Constant LR for QAT fine-tune")
    p.add_argument("--qat-low-lr", type=float, default=1e-6, help="Low LR used on the outer legs of the distillation schedule")
    p.add_argument(
        "--qat-mode",
        choices=("distill", "ultralytics"),
        default="distill",
        help="Use a short teacher-distillation loop or fall back to Ultralytics training",
    )
    p.add_argument(
        "--qat-batches-per-epoch",
        type=int,
        default=200,
        help="Maximum training batches per QAT epoch for the distillation loop",
    )
    p.add_argument(
        "--qat-distill-weight",
        type=float,
        default=1.0,
        help="Weight on teacher-student MSE in distill mode (0 disables it)",
    )
    p.add_argument(
        "--qat-supervised-weight",
        type=float,
        default=1.0,
        help="Weight on COCO supervised detection loss in distill mode (0 disables it)",
    )
    p.add_argument("--optimizer", default="Adam", help="Ultralytics optimizer for QAT fine-tune")
    p.add_argument("--calib-size", type=int, default=260, help="# images for PTQ calibration")
    p.add_argument(
        "--calib-method",
        choices=("max", "entropy", "percentile", "smoothquant"),
        default="entropy",
        help="Activation calibration method for PTQ/QAT bootstrap",
    )
    p.add_argument(
        "--calib-percentile",
        type=float,
        default=99.99,
        help="Percentile used when --calib-method=percentile",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=10, help="Train + calib batch size")
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers for custom QAT/distillation")
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Ultralytics device string (e.g. '0', 'cpu')",
    )
    p.add_argument("--skip-fp32-eval", action="store_true", help="Skip baseline mAP")
    p.add_argument(
        "--from-ptq",
        type=Path,
        default=None,
        help="Restore an existing mto.save() PTQ checkpoint and continue with QAT; requires one --models entry",
    )
    p.add_argument("--no-export", action="store_true", help="Skip final ONNX export")
    p.add_argument(
        "--no-modelopt-export",
        action="store_true",
        help="Use Ultralytics' exporter instead of ModelOpt's Torch ONNX export utility",
    )
    p.add_argument(
        "--qat-recipe",
        choices=("auto", *QAT_RECIPES.keys()),
        default="auto",
        help=(
            "Named QAT hyperparameter recipe. 'auto' picks 'yolo26-distill' for E2E "
            "models (yolo26*) and 'yolo11-supervised' for single-head models "
            "(YOLOv8/v11/etc.). Recipe values fill in only flags the user didn't pass; "
            "explicit CLI flags always override."
        ),
    )
    p.add_argument(
        "--disable-detect-output-quant",
        action="store_true",
        help="Disable output quantizers on Ultralytics Detect heads after quantization",
    )
    p.add_argument(
        "--exclude-dfl-quant",
        action="store_true",
        help=(
            "Disable quantizers on every Ultralytics DFL module. Default off (preserves "
            "the yolo26s baseline) but auto-enabled by --qat-recipe yolo11-supervised."
        ),
    )
    p.add_argument(
        "--no-fuse-before-quant",
        action="store_true",
        help="Skip Conv+BN fusion before quantization",
    )
    p.add_argument(
        "--no-residual-add-quant",
        action="store_true",
        help="Skip patching raw residual tensor adds with explicit branch quantizers",
    )
    p.add_argument(
        "--no-share-residual-add-scales",
        action="store_true",
        help="Skip forcing patched residual add branches to share the same post-calibration amax",
    )
    p.add_argument(
        "--freeze-teacher-bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force teacher BatchNorm running stats fixed (or --no-freeze-teacher-bn to force-off). "
            "Default: auto — freeze for single-head models (YOLOv8/v11/etc.), "
            "leave open for E2E YOLO26 to preserve the recorded yolo26s baseline."
        ),
    )
    p.add_argument(
        "--freeze-student-bn",
        action="store_true",
        help="Keep student BatchNorm running stats fixed during QAT while still training affine weights",
    )
    p.add_argument("--sensitivity", action="store_true", help="Run a per-module quantization sensitivity sweep")
    p.add_argument(
        "--sensitivity-stage",
        choices=("ptq", "qat"),
        default="qat",
        help="Which stage to sweep when --sensitivity is enabled",
    )
    p.add_argument(
        "--sensitivity-max-layers",
        type=int,
        default=24,
        help="Maximum number of quantized modules to evaluate, sampled across model depth",
    )
    p.add_argument(
        "--calib-source",
        default="val2017",
        help="PTQ calibration source: train, val, val2017, train2017, or an explicit image directory/list path",
    )
    p.add_argument(
        "--keep-fp32-modules",
        nargs="+",
        default=[],
        metavar="MODULE",
        help="Disable quantizers on these module paths (and their submodules) right after PTQ / restore. "
             "Useful for sensitivity-driven selective dequantization.",
    )
    p.add_argument(
        "--qat-log-every",
        type=int,
        default=0,
        help="Print a [qat/distill] step heartbeat every N batches within an epoch (0 = epoch summary only)",
    )
    p.add_argument(
        "--qat-eval-every",
        type=int,
        default=0,
        help="Run a COCO val pass every N QAT epochs (and on the final epoch); saves yolo*_qat_best.pth on improvement",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Pin Python/Numpy/Torch RNG to this seed for reproducibility",
    )
    p.epilog = (
        "For a dedicated PTQ sensitivity subcommand, run: "
        "`python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py sensitivity --help`"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 1 and argv[0] == "sensitivity":
        return run_sensitivity_subcommand(argv[1:])

    args = parse_qat_args(argv)
    # Capture explicit CLI flags so the recipe layer doesn't overwrite them.
    explicit_flags = {tok for tok in argv if tok.startswith("--")}
    resolved_recipe = _apply_recipe(args, explicit_flags)
    args.resolved_recipe = resolved_recipe
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _pin_seed(args.seed)

    data_yaml = args.data
    if not data_yaml.exists():
        print(f"ERROR: dataset config not found: {data_yaml}", file=sys.stderr)
        return 2
    try:
        calib_dir = resolve_calib_source(args.calib_source, data_yaml)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not calib_dir.exists():
        print(f"ERROR: calibration source not found: {calib_dir}", file=sys.stderr)
        return 2
    if args.from_ptq is not None and len(args.models) != 1:
        print("ERROR: --from-ptq requires exactly one --models entry", file=sys.stderr)
        return 2

    all_results: list[ModelResult] = []
    for model_name in args.models:
        try:
            all_results.append(
                run_qat_for_model(
                    model_name=model_name,
                    qat_epochs=args.qat_epochs,
                    qat_lr=args.qat_lr,
                    optimizer=args.optimizer,
                    qat_mode=args.qat_mode,
                    qat_low_lr=args.qat_low_lr,
                    qat_batches_per_epoch=args.qat_batches_per_epoch,
                    qat_distill_weight=args.qat_distill_weight,
                    qat_supervised_weight=args.qat_supervised_weight,
                    calib_size=args.calib_size,
                    calib_method=args.calib_method,
                    calib_percentile=args.calib_percentile,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    val_batch=args.val_batch,
                    device=args.device,
                    dataloader_workers=args.workers,
                    skip_fp32_eval=args.skip_fp32_eval,
                    do_export=not args.no_export,
                    use_modelopt_export=not args.no_modelopt_export,
                    disable_detect_output_quant=args.disable_detect_output_quant,
                    fuse_before_quant=not args.no_fuse_before_quant,
                    patch_residual_adds=not args.no_residual_add_quant,
                    share_residual_add_scales=not args.no_share_residual_add_scales,
                    freeze_teacher_bn=args.freeze_teacher_bn,
                    freeze_student_bn=args.freeze_student_bn,
                    sensitivity=args.sensitivity,
                    sensitivity_stage=args.sensitivity_stage,
                    sensitivity_max_layers=args.sensitivity_max_layers,
                    qat_eval_every=args.qat_eval_every,
                    qat_log_every=args.qat_log_every,
                    calib_dir=calib_dir,
                    data_yaml=data_yaml,
                    keep_fp32_modules=args.keep_fp32_modules or None,
                    exclude_dfl_quant=args.exclude_dfl_quant,
                    seed=args.seed,
                    recipe=resolved_recipe,
                    from_ptq=args.from_ptq,
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
            "sensitivity": r.sensitivity,
            "best_checkpoint": r.best_checkpoint,
            "best_map": r.best_map,
            "config": r.config,
        }
        for r in all_results
    ]
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote combined report -> {OUT_ROOT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
