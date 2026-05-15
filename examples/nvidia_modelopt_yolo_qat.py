#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for Ultralytics YOLO with NVIDIA ModelOpt.

Mirrors the structure of ``NVIDIA/Model-Optimizer/examples/cnn_qat/``
(PTQ calibration -> mtq.quantize -> QAT fine-tune -> mto.save -> export),
adapted for object detection:

* Calibration uses COCO 2017 ``images/val2017`` images with Ultralytics' inference
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
        [--models yolo11x] \
        --qat-epochs 2 \
        --calib-size 512 \
        --imgsz 640 \
        --batch 16

Every stage is resumable: FP32 / PTQ / QAT checkpoints, calibration tensors
and per-model summaries are cached under ``runs/modelopt_qat/``.
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

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_YAML = PROJECT_ROOT / "configs" / "coco.yaml"
COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
VAL_DIR = COCO_ROOT / "images" / "val2017"
OUT_ROOT = PROJECT_ROOT / "runs" / "modelopt_qat"

DEFAULT_MODELS = ("yolo11x",)


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
    """Export the quantized model through ModelOpt's Torch ONNX utility."""
    from ultralytics.nn.modules.head import Detect

    get_onnx_bytes = _import_modelopt_onnx_export()

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
        onnx_bytes = get_onnx_bytes(model, dummy, onnx_opset=17)
        dst.write_bytes(onnx_bytes)
    finally:
        for module, dynamic, export, fmt, max_det, xyxy in detect_state:
            module.dynamic = dynamic
            module.export = export
            module.format = fmt
            if max_det is not None:
                module.max_det = max_det
            module.xyxy = xyxy


def build_quant_cfg(
    mtq,
    calib_method: str,
    calib_percentile: float,
) -> dict:
    """Build a YOLO-friendly ModelOpt quantization config."""
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


def apply_detect_head_output_exclusion(model: torch.nn.Module) -> int:
    """Disable output quantizers on Ultralytics detection heads."""
    from ultralytics.nn.modules.head import Detect

    disabled = 0
    for module in model.modules():
        if isinstance(module, Detect) and hasattr(module, "output_quantizer"):
            module.output_quantizer.disable()
            disabled += 1
    return disabled


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
        disabled_detect_outputs = apply_detect_head_output_exclusion(model)

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
        "modelopt.torch is not importable in this environment. "
        "The current venv is missing optional torch dependencies "
        "(observed missing module: 'pulp'). Install the full ModelOpt torch extras "
        "or add 'pulp' to the environment before running the QAT/PTQ/export pipeline."
    ) from exc


def _import_modelopt_qat_modules():
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except Exception as exc:
        _raise_modelopt_import_error(exc)
    return mto, mtq


def _import_modelopt_onnx_export():
    try:
        from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes
    except Exception as exc:
        _raise_modelopt_import_error(exc)
    return get_onnx_bytes


def _teacher_student_raw_outputs(outputs) -> list[torch.Tensor]:
    """Normalize YOLO detect outputs to raw multi-scale tensors for distillation."""
    if isinstance(outputs, list):
        return outputs
    if isinstance(outputs, tuple):
        for item in reversed(outputs):
            if isinstance(item, list) and all(torch.is_tensor(x) for x in item):
                return item
    if torch.is_tensor(outputs):
        return [outputs]
    raise TypeError(f"Unsupported YOLO output type for distillation: {type(outputs)!r}")


def _mse_distill_loss(student_outputs, teacher_outputs) -> torch.Tensor:
    student_raw = _teacher_student_raw_outputs(student_outputs)
    teacher_raw = _teacher_student_raw_outputs(teacher_outputs)
    if len(student_raw) != len(teacher_raw):
        raise RuntimeError(
            f"Teacher/student raw output count mismatch: {len(student_raw)} vs {len(teacher_raw)}"
        )
    return sum(
        torch.nn.functional.mse_loss(s.clone(), t.detach().clone())
        for s, t in zip(student_raw, teacher_raw)
    )


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


def _build_detection_trainer(
    weights: str,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = {
        "model": weights,
        "data": str(COCO_YAML),
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


def run_distillation_qat(
    student_model: torch.nn.Module,
    teacher_weights: str,
    imgsz: int,
    batch: int,
    device: str,
    epochs: int,
    peak_lr: float,
    low_lr: float,
    batches_per_epoch: int,
    workers: int,
) -> None:
    """Short QAT loop that matches raw teacher outputs on Ultralytics detection batches."""
    from ultralytics import YOLO

    cloned_inference_buffers, cloned_amax_states = _prepare_quantized_model_for_training(student_model)
    if cloned_inference_buffers:
        print(f"  [qat] cloned {cloned_inference_buffers} inference buffer(s) before training")
    if cloned_amax_states:
        print(f"  [qat] cloned {cloned_amax_states} quantizer amax state tensor(s) before training")

    trainer = _build_detection_trainer(teacher_weights, imgsz, batch, device, workers)
    trainer.model = student_model
    trainer.device = _resolve_device(device)
    trainer.stride = max(int(student_model.stride.max() if hasattr(student_model, "stride") else 32), 32)
    trainer.set_model_attributes()
    dataloader = trainer.get_dataloader(trainer.data["train"], batch_size=batch, rank=-1, mode="train")

    teacher = YOLO(teacher_weights).model.to(trainer.device)
    teacher.eval()
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

    for epoch in range(epochs):
        student_model.train()
        epoch_lr = _distill_epoch_lr(epoch, epochs, peak_lr, low_lr)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr

        running_loss = 0.0
        for step, batch_data in enumerate(dataloader):
            if step >= batches_per_epoch:
                break
            batch_data = trainer.preprocess_batch(batch_data)
            imgs = batch_data["img"].clone()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_outputs = teacher(imgs)
            with torch.autocast(device_type=amp_device, enabled=amp_enabled):
                student_outputs = student_model(imgs)
                loss = _mse_distill_loss(student_outputs, teacher_outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(1, min(len(dataloader), batches_per_epoch))
        print(f"  [qat/distill] epoch {epoch + 1}/{epochs} lr={epoch_lr:.2e} mse={avg_loss:.6f}")


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


def run_sensitivity_sweep(
    yolo,
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

    if max_layers > 0:
        candidates = candidates[:max_layers]

    results: list[dict] = []
    for name, module in candidates:
        quantizers = list(_iter_module_quantizers(module))
        for quantizer in quantizers:
            quantizer.disable()
        try:
            map50, mAP, secs = eval_on_coco_val(yolo, imgsz, batch, device)
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


def run_qat_for_model(
    model_name: str,
    qat_epochs: int,
    qat_lr: float,
    optimizer: str,
    qat_mode: str,
    qat_low_lr: float,
    qat_batches_per_epoch: int,
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
    sensitivity: bool,
    sensitivity_stage: str,
    sensitivity_max_layers: int,
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
                disabled_detect_outputs = apply_detect_head_output_exclusion(yolo.model)

        if patched_adds:
            print(f"  [ptq] patched {patched_adds} residual add module(s)")
        if unified_add_scales:
            print(f"  [ptq] unified scales across {unified_add_scales} residual add(s)")
        if disable_detect_output_quant:
            print(f"  [ptq] disabled {disabled_detect_outputs} detect-head output quantizer(s)")

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

        map50, mAP, secs = eval_on_coco_val(yolo, imgsz, val_batch, device)
        ptq_stage.map50, ptq_stage.map, ptq_stage.seconds = map50, mAP, secs
        print(f"  [val/ptq] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as e:
        ptq_stage.error = str(e)
        print(f"  [ptq] FAILED: {e}")
        stages.append(ptq_stage)
        return ModelResult(model=model_name, stages=stages)
    stages.append(ptq_stage)

    if sensitivity and sensitivity_stage == "ptq":
        print(f"  [sensitivity] sweeping ptq modules (limit={sensitivity_max_layers})")
        sensitivity_results = run_sensitivity_sweep(
            yolo,
            imgsz,
            val_batch,
            device,
            max_layers=sensitivity_max_layers,
            baseline_map=ptq_stage.map,
            baseline_map50=ptq_stage.map50,
        )

    # -------- 4. QAT fine-tune via Ultralytics trainer --------
    qat_stage = StageResult(stage="qat")
    try:
        print(f"  [qat] fine-tuning for {qat_epochs} epoch(s) (mode={qat_mode}, lr0={qat_lr}, optimizer={optimizer})")
        if qat_mode == "distill":
            run_distillation_qat(
                student_model=yolo.model,
                teacher_weights=weights,
                imgsz=imgsz,
                batch=batch,
                device=device,
                epochs=qat_epochs,
                peak_lr=qat_lr,
                low_lr=qat_low_lr,
                batches_per_epoch=qat_batches_per_epoch,
                workers=dataloader_workers,
            )
        else:
            yolo.train(
                data=str(COCO_YAML),
                epochs=qat_epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=qat_lr,
                lrf=qat_lr,  # constant LR over short QAT fine-tune
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

        map50, mAP, secs = eval_on_coco_val(yolo, imgsz, val_batch, device)
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
                imgsz,
                val_batch,
                device,
                max_layers=sensitivity_max_layers,
                baseline_map=qat_stage.map,
                baseline_map50=qat_stage.map50,
            )

    result = ModelResult(
        model=model_name,
        stages=stages,
        onnx_path=str(onnx_path) if onnx_path else None,
        sensitivity=sensitivity_results,
    )
    (model_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": result.model,
                "onnx_path": result.onnx_path,
                "stages": [asdict(s) for s in result.stages],
                "sensitivity": result.sensitivity,
            },
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
    p.add_argument("--optimizer", default="Adam", help="Ultralytics optimizer for QAT fine-tune")
    p.add_argument("--calib-size", type=int, default=512, help="# val2017 images for PTQ")
    p.add_argument(
        "--calib-method",
        choices=("max", "entropy", "percentile", "smoothquant"),
        default="max",
        help="Activation calibration method for PTQ/QAT bootstrap",
    )
    p.add_argument(
        "--calib-percentile",
        type=float,
        default=99.99,
        help="Percentile used when --calib-method=percentile",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Train + calib batch size")
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers for custom QAT/distillation")
    p.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Ultralytics device string (e.g. '0', 'cpu')",
    )
    p.add_argument("--skip-fp32-eval", action="store_true", help="Skip baseline mAP")
    p.add_argument("--no-export", action="store_true", help="Skip final ONNX export")
    p.add_argument(
        "--no-modelopt-export",
        action="store_true",
        help="Use Ultralytics' exporter instead of ModelOpt's Torch ONNX export utility",
    )
    p.add_argument(
        "--disable-detect-output-quant",
        action="store_true",
        help="Disable output quantizers on Ultralytics Detect heads after quantization",
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
        help="Maximum number of quantized modules to evaluate in the sensitivity sweep",
    )
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
                    optimizer=args.optimizer,
                    qat_mode=args.qat_mode,
                    qat_low_lr=args.qat_low_lr,
                    qat_batches_per_epoch=args.qat_batches_per_epoch,
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
                    sensitivity=args.sensitivity,
                    sensitivity_stage=args.sensitivity_stage,
                    sensitivity_max_layers=args.sensitivity_max_layers,
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
        }
        for r in all_results
    ]
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote combined report -> {OUT_ROOT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
