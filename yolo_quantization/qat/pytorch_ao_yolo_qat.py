#!/usr/bin/env python3
"""
Experimental PyTorch AO fake-quant QAT recipe for Ultralytics YOLO.

This is an alternative research path to the NVIDIA ModelOpt pipeline in
``nvidia_modelopt_yolo_qat.py``. It uses ``torch.ao.quantization`` observers and
``FakeQuantize`` modules directly around YOLO ``nn.Conv2d`` layer inputs,
weights, and outputs:

1. Load an FP32 Ultralytics checkpoint.
2. Optionally evaluate FP32 mAP.
3. Replace selected Conv2d modules with trainable AO fake-quant wrappers.
4. Calibrate observers on dataset images with fake quant disabled.
5. Freeze observers, enable fake quant, evaluate the calibrated fake-quant PTQ
   proxy, then fine-tune with the same distill+supervised loop as ModelOpt.
6. Save PyTorch checkpoints under ``runs/pytorch_ao_qat/<model>/``.

The output is a trainable fake-quant model, not a ModelOpt checkpoint and not a
backend-converted CPU int8 graph. Treat ONNX export as best-effort until a
deployment-specific AO export path is proven for YOLO Detect heads.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nvidia_modelopt_yolo_qat import (
    DEFAULT_DATA_YAML,
    DEFAULT_MODELS,
    PROJECT_ROOT,
    StageResult,
    _pin_seed,
    _resolve_device,
    build_calib_batches,
    eval_on_dataset,
    export_onnx,
    find_dfl_module_paths,
    resolve_calib_source,
    run_distillation_qat,
    _write_qat_train_csv,
)


OUT_ROOT = PROJECT_ROOT / "runs" / "pytorch_ao_qat"


AO_QAT_RECIPES: dict[str, dict[str, Any]] = {
    "yolo26-ao-distill": {
        "qat_epochs": 10,
        "qat_batches_per_epoch": 200,
        "qat_lr": 1e-5,
        "qat_low_lr": 1e-6,
        "qat_distill_weight": 1.0,
        "qat_supervised_weight": 1.0,
        "calib_size": 260,
        "ao_activation_observer": "histogram",
        "exclude_dfl_quant": False,
        "disable_detect_output_quant": False,
        "disable_conv_output_quant": True,
    },
    "yolo11-ao-distill": {
        "qat_epochs": 10,
        "qat_batches_per_epoch": 200,
        "qat_lr": 1e-5,
        "qat_low_lr": 1e-6,
        "qat_distill_weight": 1.0,
        "qat_supervised_weight": 1.0,
        "calib_size": 1024,
        "ao_activation_observer": "moving_average_minmax",
        "exclude_dfl_quant": True,
        "disable_detect_output_quant": True,
        "disable_conv_output_quant": False,
    },
}


def _resolve_ao_recipe(name: str, model_name: str | None) -> str:
    if name != "auto":
        if name not in AO_QAT_RECIPES:
            raise ValueError(f"Unknown AO recipe: {name!r}. Choices: {sorted(AO_QAT_RECIPES)}")
        return name
    if model_name and model_name.lower().startswith("yolo26"):
        return "yolo26-ao-distill"
    return "yolo11-ao-distill"


def _apply_ao_recipe(args, explicit_flags: set[str]) -> str:
    resolved = _resolve_ao_recipe(args.ao_recipe, args.models[0] if args.models else None)
    flag_map = {
        "qat_epochs": "--qat-epochs",
        "qat_batches_per_epoch": "--qat-batches-per-epoch",
        "qat_lr": "--qat-lr",
        "qat_low_lr": "--qat-low-lr",
        "qat_distill_weight": "--qat-distill-weight",
        "qat_supervised_weight": "--qat-supervised-weight",
        "calib_size": "--calib-size",
        "ao_activation_observer": "--ao-activation-observer",
        "exclude_dfl_quant": "--exclude-dfl-quant",
        "disable_detect_output_quant": "--disable-detect-output-quant",
        "disable_conv_output_quant": "--disable-conv-output-quant",
    }
    applied: list[str] = []
    for key, value in AO_QAT_RECIPES[resolved].items():
        if flag_map[key] in explicit_flags:
            continue
        setattr(args, key, value)
        applied.append(f"{key}={value}")
    if applied:
        print(f"[ao-recipe={resolved}] applied defaults: " + ", ".join(applied))
    return resolved


class AOConv2dQATWrapper(torch.nn.Module):
    """Conv2d wrapper using PyTorch AO fake quant for activation, weight, and output."""

    def __init__(
        self,
        conv: torch.nn.Conv2d,
        *,
        activation_quant: bool = True,
        weight_quant: bool = True,
        output_quant: bool = True,
        activation_observer: str = "moving_average_minmax",
    ) -> None:
        super().__init__()
        from torch.ao.quantization import (
            FakeQuantize,
            HistogramObserver,
            MovingAverageMinMaxObserver,
            MovingAveragePerChannelMinMaxObserver,
        )

        activation_observer_cls = {
            "moving_average_minmax": MovingAverageMinMaxObserver,
            "histogram": HistogramObserver,
        }[activation_observer]

        self.conv = conv
        self.activation_observer = activation_observer
        self.activation_fake_quant = (
            FakeQuantize(
                observer=activation_observer_cls,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            )
            if activation_quant
            else torch.nn.Identity()
        )
        self.weight_fake_quant = (
            FakeQuantize(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            )
            if weight_quant
            else torch.nn.Identity()
        )
        self.output_fake_quant = (
            FakeQuantize(
                observer=activation_observer_cls,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            )
            if output_quant
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        y = torch.nn.functional.conv2d(
            x,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return self.output_fake_quant(y)


def _is_under_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _replace_child_module(root: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    parent._modules[child_name] = replacement


def prepare_pytorch_ao_fake_quant(
    model: torch.nn.Module,
    *,
    exclude_prefixes: list[str] | None = None,
    disable_output_quant_prefixes: list[str] | None = None,
    activation_observer: str = "moving_average_minmax",
) -> int:
    """Wrap Conv2d modules with AO fake quant, excluding requested module prefixes."""
    exclude_prefixes = exclude_prefixes or []
    disable_output_quant_prefixes = disable_output_quant_prefixes or []
    replacements: list[tuple[str, AOConv2dQATWrapper]] = []
    for name, module in model.named_modules():
        if not name or _is_under_prefix(name, exclude_prefixes):
            continue
        if isinstance(module, torch.nn.Conv2d):
            replacements.append(
                (
                    name,
                    AOConv2dQATWrapper(
                        module,
                        output_quant=not _is_under_prefix(name, disable_output_quant_prefixes),
                        activation_observer=activation_observer,
                    ),
                )
            )
    for name, wrapper in replacements:
        _replace_child_module(model, name, wrapper)
    return len(replacements)


def iter_ao_fake_quant_modules(model: torch.nn.Module):
    from torch.ao.quantization import FakeQuantize

    for module in model.modules():
        if isinstance(module, FakeQuantize):
            yield module


def set_ao_observer_enabled(model: torch.nn.Module, enabled: bool) -> int:
    count = 0
    for module in iter_ao_fake_quant_modules(model):
        if enabled:
            module.enable_observer()
        else:
            module.disable_observer()
        count += 1
    return count


def set_ao_fake_quant_enabled(model: torch.nn.Module, enabled: bool) -> int:
    count = 0
    for module in iter_ao_fake_quant_modules(model):
        if enabled:
            module.enable_fake_quant()
        else:
            module.disable_fake_quant()
        count += 1
    return count


def snapshot_ao_qparams(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU copies of AO fake-quant scale tensors for debugging/reporting."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        scale = getattr(module, "scale", None)
        if torch.is_tensor(scale):
            snapshot[name] = scale.detach().float().cpu().clone()
    return snapshot


def restore_student_requires_grad(model: torch.nn.Module) -> int:
    """Undo Ultralytics validation side effects that mark student params non-trainable."""
    restored = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad_(True)
            restored += 1
    return restored


def normalize_ultralytics_model_args(model: torch.nn.Module, imgsz: int) -> None:
    """Keep ``YOLO.export()`` compatible after trainer code replaces args with a namespace."""
    args = getattr(model, "args", None)
    if args is None:
        model.args = {"imgsz": imgsz}
    elif not isinstance(args, dict):
        model.args = vars(args).copy()
    model.args.setdefault("imgsz", imgsz)


def calibrate_ao_fake_quant_model(
    model: torch.nn.Module,
    calib_dir: Path,
    imgsz: int,
    batch: int,
    calib_size: int,
    device: torch.device,
) -> None:
    batches = build_calib_batches(calib_dir, imgsz, batch, calib_size)
    observer_count = set_ao_observer_enabled(model, True)
    set_ao_fake_quant_enabled(model, False)
    print(f"  [ao/calib] collecting observers for {observer_count} fake-quant module(s)")
    model.eval()
    seen = 0
    with torch.no_grad():
        for b in batches:
            model(b.to(device, non_blocking=True))
            seen += b.size(0)
            if seen >= calib_size:
                break
    set_ao_observer_enabled(model, False)
    set_ao_fake_quant_enabled(model, True)
    print(f"  [ao/calib] frozen observers after {seen} sample(s); fake quant enabled")


@dataclass
class AOModelResult:
    model: str
    stages: list[StageResult]
    onnx_path: str | None = None
    qat_train: list[dict] | None = None
    best_checkpoint: str | None = None
    best_map: float | None = None
    config: dict | None = None


def _save_ao_checkpoint(path: Path, model: torch.nn.Module, config: dict[str, Any]) -> None:
    torch.save(
        {
            "format": "pytorch_ao_fake_quant_yolo_qat_v2",
            "model_state_dict": model.state_dict(),
            "config": copy.deepcopy(config),
        },
        path,
    )


def _load_ao_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_missing = all(".output_fake_quant." in key for key in incompatible.missing_keys)
    if allowed_missing and not incompatible.unexpected_keys:
        print(
            "  [ao/ptq] restored legacy checkpoint without Conv output fake-quant state; "
            "new output observers will recalibrate from defaults"
        )
        return

    missing = ", ".join(incompatible.missing_keys[:8])
    unexpected = ", ".join(incompatible.unexpected_keys[:8])
    raise RuntimeError(
        "AO checkpoint is incompatible with the prepared wrapper structure "
        f"(missing=[{missing}], unexpected=[{unexpected}])"
    )


def _load_ao_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=str(device), weights_only=False)
    state_dict = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    _load_ao_state_dict(model, state_dict)
    return obj if isinstance(obj, dict) else {}


def print_ao_report(results: list[AOModelResult]) -> None:
    print("\n" + "=" * 78)
    print("PyTorch AO fake-quant QAT x YOLO / COCO summary")
    print("=" * 78)
    header = f"{'model':<10} {'stage':<5} {'mAP50':>8} {'mAP':>8} {'secs':>7}  notes"
    print(header)
    print("-" * len(header))
    for result in results:
        for stage in result.stages:
            map50 = "   -  " if stage.map50 is None else f"{stage.map50:.4f}"
            mAP = "   -  " if stage.map is None else f"{stage.map:.4f}"
            secs = "   -  " if stage.seconds is None else f"{stage.seconds:.1f}"
            notes = stage.error or (stage.checkpoint or "")
            print(f"{result.model:<10} {stage.stage:<5} {map50:>8} {mAP:>8} {secs:>7}  {notes}")


def run_ao_qat_for_model(
    *,
    model_name: str,
    data_yaml: Path,
    calib_dir: Path,
    imgsz: int,
    batch: int,
    val_batch: int,
    device: str,
    workers: int,
    calib_size: int,
    qat_epochs: int,
    qat_lr: float,
    qat_low_lr: float,
    qat_batches_per_epoch: int,
    qat_distill_weight: float,
    qat_supervised_weight: float,
    qat_eval_every: int,
    qat_log_every: int,
    skip_fp32_eval: bool,
    skip_ptq_eval: bool,
    skip_qat_eval: bool,
    do_export: bool,
    freeze_teacher_bn: bool | None,
    freeze_student_bn: bool,
    exclude_dfl_quant: bool,
    disable_detect_output_quant: bool,
    disable_conv_output_quant: bool,
    ao_activation_observer: str,
    keep_fp32_modules: list[str] | None,
    recipe: str,
    seed: int | None,
    from_ao_ptq: Path | None,
) -> AOModelResult:
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    print(f"\n=== {model_name} / PyTorch AO ===")
    weights = f"{model_name}.pt"
    model_dir = OUT_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(weights)
    torch_device = _resolve_device(device)
    yolo.model.to(torch_device)
    print("  [prep] fusing Conv+BN before AO fake quant")
    yolo.fuse()
    yolo.model.to(torch_device)

    stages: list[StageResult] = []
    fp32_stage = StageResult(stage="fp32")
    if not skip_fp32_eval:
        try:
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
            fp32_stage.map50, fp32_stage.map, fp32_stage.seconds = map50, mAP, secs
            print(f"  [val/fp32] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
        except Exception as exc:
            fp32_stage.error = str(exc)
            print(f"  [val/fp32] FAILED: {exc}")
    stages.append(fp32_stage)

    exclude_prefixes = list(keep_fp32_modules or [])
    if exclude_dfl_quant:
        exclude_prefixes.extend(find_dfl_module_paths(yolo.model))
    if disable_conv_output_quant:
        disable_output_quant_prefixes = [
            name for name, module in yolo.model.named_modules() if isinstance(module, torch.nn.Conv2d)
        ]
    elif disable_detect_output_quant:
        disable_output_quant_prefixes = [
            name for name, module in yolo.model.named_modules() if isinstance(module, Detect)
        ]
    else:
        disable_output_quant_prefixes = []

    wrapped = prepare_pytorch_ao_fake_quant(
        yolo.model,
        exclude_prefixes=exclude_prefixes,
        disable_output_quant_prefixes=disable_output_quant_prefixes,
        activation_observer=ao_activation_observer,
    )
    yolo.model.to(torch_device)
    print(f"  [ao/prepare] wrapped {wrapped} Conv2d module(s)")
    if exclude_prefixes:
        print(f"  [ao/prepare] excluded prefixes: {exclude_prefixes}")
    if disable_output_quant_prefixes:
        print(f"  [ao/prepare] disabled Conv output fake quant under: {disable_output_quant_prefixes}")

    config_snapshot = {
        "seed": seed,
        "data": str(data_yaml),
        "calib_source": str(calib_dir),
        "calib_size": calib_size,
        "qat_epochs": qat_epochs,
        "qat_batches_per_epoch": qat_batches_per_epoch,
        "qat_lr": qat_lr,
        "qat_low_lr": qat_low_lr,
        "qat_distill_weight": qat_distill_weight,
        "qat_supervised_weight": qat_supervised_weight,
        "qat_eval_every": qat_eval_every,
        "qat_log_every": qat_log_every,
        "skip_fp32_eval": skip_fp32_eval,
        "skip_ptq_eval": skip_ptq_eval,
        "skip_qat_eval": skip_qat_eval,
        "imgsz": imgsz,
        "batch": batch,
        "val_batch": val_batch,
        "exclude_dfl_quant": exclude_dfl_quant,
        "disable_detect_output_quant": disable_detect_output_quant,
        "disable_conv_output_quant": disable_conv_output_quant,
        "ao_activation_observer": ao_activation_observer,
        "keep_fp32_modules": keep_fp32_modules or [],
        "disable_output_quant_prefixes": disable_output_quant_prefixes,
        "ao_wrapped_convs": wrapped,
        "recipe": recipe,
        "freeze_teacher_bn": freeze_teacher_bn,
        "freeze_student_bn": freeze_student_bn,
    }

    ptq_stage = StageResult(stage="ptq")
    try:
        if from_ao_ptq is not None:
            print(f"  [ao/ptq] restoring {from_ao_ptq}")
            _load_ao_checkpoint(from_ao_ptq, yolo.model, torch_device)
            set_ao_observer_enabled(yolo.model, False)
            set_ao_fake_quant_enabled(yolo.model, True)
            ptq_stage.checkpoint = str(from_ao_ptq)
        else:
            calibrate_ao_fake_quant_model(
                yolo.model, calib_dir, imgsz, batch, calib_size, torch_device
            )
            ptq_ckpt = model_dir / f"{model_name}_ao_ptq.pth"
            _save_ao_checkpoint(ptq_ckpt, yolo.model, config_snapshot)
            ptq_stage.checkpoint = str(ptq_ckpt)
            print(f"  [ao/ptq] saved {ptq_ckpt}")
        if skip_ptq_eval:
            print("  [val/ao-ptq] skipped")
        else:
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
            ptq_stage.map50, ptq_stage.map, ptq_stage.seconds = map50, mAP, secs
            print(f"  [val/ao-ptq] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as exc:
        ptq_stage.error = str(exc)
        print(f"  [ao/ptq] FAILED: {exc}")
        stages.append(ptq_stage)
        return AOModelResult(model=model_name, stages=stages, config=config_snapshot)
    stages.append(ptq_stage)

    qat_stage = StageResult(stage="qat")
    training_log: list[dict] = []
    best_state = {"map": None, "map50": None, "epoch": None, "checkpoint": None}
    best_ckpt_path = model_dir / f"{model_name}_ao_qat_best.pth"

    def _qat_eval_callback(epoch_1based: int) -> dict | None:
        try:
            yolo.model.eval()
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
        except Exception as exc:
            print(f"  [ao/qat/eval] epoch {epoch_1based} FAILED: {exc}")
            return None
        finally:
            yolo.model.train()
            restored = restore_student_requires_grad(yolo.model)
            if restored:
                print(f"  [ao/qat/eval] restored requires_grad on {restored} student parameter(s)")
        print(
            f"  [ao/qat/eval] epoch {epoch_1based} mAP50={map50:.4f} "
            f"mAP50-95={mAP:.4f} ({secs:.1f}s)"
        )
        if best_state["map"] is None or mAP > best_state["map"]:
            best_state.update(
                {"map": mAP, "map50": map50, "epoch": epoch_1based, "checkpoint": str(best_ckpt_path)}
            )
            _save_ao_checkpoint(best_ckpt_path, yolo.model, config_snapshot)
            print(f"  [ao/qat/eval] new best @ epoch {epoch_1based}: saved {best_ckpt_path}")
        return {"map50": map50, "map": mAP, "secs": secs}

    try:
        print(
            f"  [ao/qat] fine-tuning for {qat_epochs} epoch(s) "
            f"(lr0={qat_lr}, low_lr={qat_low_lr})"
        )
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
            workers=workers,
            distill_weight=qat_distill_weight,
            supervised_weight=qat_supervised_weight,
            freeze_teacher_bn=freeze_teacher_bn,
            freeze_student_bn=freeze_student_bn,
            log_every=qat_log_every,
            eval_callback=_qat_eval_callback if qat_eval_every > 0 else None,
            eval_every=qat_eval_every,
            quant_state_snapshot_fn=snapshot_ao_qparams,
        )
        qat_ckpt = model_dir / f"{model_name}_ao_qat.pth"
        _save_ao_checkpoint(qat_ckpt, yolo.model, config_snapshot)
        qat_stage.checkpoint = str(qat_ckpt)
        print(f"  [ao/qat] saved {qat_ckpt}")

        if skip_qat_eval:
            print("  [val/ao-qat] skipped")
        else:
            map50, mAP, secs = eval_on_dataset(yolo, data_yaml, imgsz, val_batch, device)
            qat_stage.map50, qat_stage.map, qat_stage.seconds = map50, mAP, secs
            print(f"  [val/ao-qat] mAP50={map50:.4f} mAP50-95={mAP:.4f} ({secs:.1f}s)")
    except Exception as exc:
        qat_stage.error = str(exc)
        print(f"  [ao/qat] FAILED: {exc}")
    stages.append(qat_stage)

    onnx_path: Path | None = None
    if do_export and qat_stage.error is None:
        normalize_ultralytics_model_args(yolo.model, imgsz)
        onnx_path = export_onnx(
            yolo,
            imgsz,
            model_dir / f"{model_name}_ao_qat.onnx",
            device=device,
            use_modelopt_export=False,
        )

    if training_log:
        _write_qat_train_csv(model_dir / "ao_qat_train.csv", training_log)

    result = AOModelResult(
        model=model_name,
        stages=stages,
        onnx_path=str(onnx_path) if onnx_path else None,
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
                "stages": [asdict(stage) for stage in result.stages],
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    p.add_argument(
        "--ao-recipe",
        choices=("auto", *AO_QAT_RECIPES.keys()),
        default="auto",
        help="Named PyTorch AO recipe. auto picks yolo26-ao-distill for yolo26*, else yolo11-ao-distill.",
    )
    p.add_argument("--calib-size", type=int, default=260)
    p.add_argument(
        "--calib-source",
        default="val2017",
        help="Calibration source: train, val, val2017, train2017, or an image directory/list path.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--qat-epochs", type=int, default=10)
    p.add_argument("--qat-batches-per-epoch", type=int, default=200)
    p.add_argument("--qat-lr", type=float, default=1e-5)
    p.add_argument("--qat-low-lr", type=float, default=1e-6)
    p.add_argument("--qat-distill-weight", type=float, default=1.0)
    p.add_argument("--qat-supervised-weight", type=float, default=1.0)
    p.add_argument("--qat-log-every", type=int, default=0)
    p.add_argument("--qat-eval-every", type=int, default=0)
    p.add_argument(
        "--ao-activation-observer",
        choices=("moving_average_minmax", "histogram"),
        default="moving_average_minmax",
        help="Observer used for AO Conv input/output fake quant. Weight quant stays per-channel moving-average min/max.",
    )
    p.add_argument("--skip-fp32-eval", action="store_true")
    p.add_argument("--skip-ptq-eval", action="store_true")
    p.add_argument("--skip-qat-eval", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--exclude-dfl-quant", action="store_true")
    p.add_argument(
        "--disable-detect-output-quant",
        action="store_true",
        help="Disable AO Conv output fake quant under Ultralytics Detect heads while keeping their input/weight fake quant enabled.",
    )
    p.add_argument(
        "--disable-conv-output-quant",
        action="store_true",
        help="Disable AO Conv output fake quant on every Conv2d (input/weight fake quant stay enabled). Ablation switch.",
    )
    p.add_argument("--keep-fp32-modules", nargs="+", default=[], metavar="MODULE")
    p.add_argument("--freeze-teacher-bn", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--freeze-student-bn", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--from-ao-ptq",
        type=Path,
        default=None,
        help="Restore a prior *_ao_ptq.pth after preparing the same AO wrapper structure.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    explicit_flags = {tok for tok in argv if tok.startswith("--")}
    resolved_recipe = _apply_ao_recipe(args, explicit_flags)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _pin_seed(args.seed)

    if not args.data.exists():
        print(f"ERROR: dataset config not found: {args.data}", file=sys.stderr)
        return 2
    try:
        calib_dir = resolve_calib_source(args.calib_source, args.data)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not calib_dir.exists():
        print(f"ERROR: calibration source not found: {calib_dir}", file=sys.stderr)
        return 2
    if args.from_ao_ptq is not None and len(args.models) != 1:
        print("ERROR: --from-ao-ptq requires exactly one --models entry", file=sys.stderr)
        return 2

    results: list[AOModelResult] = []
    for model_name in args.models:
        try:
            results.append(
                run_ao_qat_for_model(
                    model_name=model_name,
                    data_yaml=args.data,
                    calib_dir=calib_dir,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    val_batch=args.val_batch,
                    device=args.device,
                    workers=args.workers,
                    calib_size=args.calib_size,
                    qat_epochs=args.qat_epochs,
                    qat_lr=args.qat_lr,
                    qat_low_lr=args.qat_low_lr,
                    qat_batches_per_epoch=args.qat_batches_per_epoch,
                    qat_distill_weight=args.qat_distill_weight,
                    qat_supervised_weight=args.qat_supervised_weight,
                    qat_eval_every=args.qat_eval_every,
                    qat_log_every=args.qat_log_every,
                    skip_fp32_eval=args.skip_fp32_eval,
                    skip_ptq_eval=args.skip_ptq_eval,
                    skip_qat_eval=args.skip_qat_eval,
                    do_export=not args.no_export,
                    freeze_teacher_bn=args.freeze_teacher_bn,
                    freeze_student_bn=args.freeze_student_bn,
                    exclude_dfl_quant=args.exclude_dfl_quant,
                    disable_detect_output_quant=args.disable_detect_output_quant,
                    disable_conv_output_quant=args.disable_conv_output_quant,
                    ao_activation_observer=args.ao_activation_observer,
                    keep_fp32_modules=args.keep_fp32_modules or None,
                    recipe=resolved_recipe,
                    seed=args.seed,
                    from_ao_ptq=args.from_ao_ptq,
                )
            )
        except Exception as exc:
            print(f"[{model_name}] FATAL: {exc}")
            results.append(AOModelResult(model=model_name, stages=[StageResult(stage="fp32", error=str(exc))]))

    print_ao_report(results)
    report = [
        {
            "model": result.model,
            "onnx_path": result.onnx_path,
            "stages": [asdict(stage) for stage in result.stages],
            "best_checkpoint": result.best_checkpoint,
            "best_map": result.best_map,
            "config": result.config,
        }
        for result in results
    ]
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote combined report -> {OUT_ROOT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
