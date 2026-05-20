#!/usr/bin/env python3
"""
Experimental PyTorch AO fake-quant PTQ/QAT for RF-DETR.

This is the PyTorch AO counterpart to ``nvidia_modelopt_rfdetr_qat.py``. It
uses ``torch.ao.quantization.FakeQuantize`` wrappers around selected ``Conv2d``
and ``Linear`` modules, then reuses the RF-DETR calibration, COCO mAP, and
backbone feature-distillation utilities from the ModelOpt pipeline.

Default policy:

* wrap Conv2d + Linear modules with symmetric int8 activation/output fake quant;
* use per-channel symmetric int8 fake quant for Conv/Linear weights;
* keep DETR prediction heads (``class_embed`` and ``bbox_embed``) FP32;
* calibrate observers on COCO val2017 (or a requested source), freeze observers,
  enable fake quant, evaluate PTQ, then optionally QAT-distill the backbone.

The result is a trainable fake-quant PyTorch checkpoint under
``runs/pytorch_ao_qat/rfdetr_nano/``. ONNX export is best-effort because AO fake
quant export support is backend/version dependent.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfdetr_quantization.nvidia_modelopt_rfdetr_qat import (  # noqa: E402
    ANN_FILE,
    TRAIN_DIR,
    VAL_DIR,
    build_calib_batches,
    evaluate_map,
    export_onnx,
    load_rfdetr,
    run_distillation_qat,
)

OUT_ROOT = PROJECT_ROOT / "runs" / "pytorch_ao_qat"
DEFAULT_HEAD_EXCLUDES = ("*class_embed*", "*bbox_embed*")


def make_activation_fake_quant(observer: str) -> torch.nn.Module:
    """Build symmetric int8 activation fake quant for TensorRT-friendly Q/DQ."""
    from torch.ao.quantization import FakeQuantize, HistogramObserver, MovingAverageMinMaxObserver

    observer_cls = {
        "moving_average_minmax": MovingAverageMinMaxObserver,
        "histogram": HistogramObserver,
    }[observer]
    return FakeQuantize(
        observer=observer_cls,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


def make_weight_fake_quant(ch_axis: int = 0) -> torch.nn.Module:
    from torch.ao.quantization import FakeQuantize, MovingAveragePerChannelMinMaxObserver

    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
    )


class AOConv2dQATWrapper(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, *, activation_observer: str) -> None:
        super().__init__()
        self.conv = conv
        self.input_fake_quant = make_activation_fake_quant(activation_observer)
        self.weight_fake_quant = make_weight_fake_quant(ch_axis=0)
        self.output_fake_quant = make_activation_fake_quant(activation_observer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        y = F.conv2d(x, weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return self.output_fake_quant(y)


class AOLinearQATWrapper(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, *, activation_observer: str) -> None:
        super().__init__()
        self.linear = linear
        self.input_fake_quant = make_activation_fake_quant(activation_observer)
        self.weight_fake_quant = make_weight_fake_quant(ch_axis=0)
        self.output_fake_quant = make_activation_fake_quant(activation_observer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fake_quant(x)
        weight = self.weight_fake_quant(self.linear.weight)
        y = F.linear(x, weight, self.linear.bias)
        return self.output_fake_quant(y)


def _matches_any(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def _replace_child(root: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    parent._modules[child_name] = replacement


def prepare_ao_fake_quant(model: torch.nn.Module, *, head_excludes: tuple[str, ...], quantize_linears: bool, activation_observer: str) -> dict[str, Any]:
    replacements: list[tuple[str, torch.nn.Module]] = []
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if _matches_any(name, head_excludes):
            skipped.append(name)
            continue
        if isinstance(module, torch.nn.Conv2d):
            replacements.append((name, AOConv2dQATWrapper(module, activation_observer=activation_observer)))
        elif quantize_linears and isinstance(module, torch.nn.Linear):
            replacements.append((name, AOLinearQATWrapper(module, activation_observer=activation_observer)))
    for name, replacement in replacements:
        _replace_child(model, name, replacement)
    return {
        "wrapped_convs": sum(isinstance(m, AOConv2dQATWrapper) for m in model.modules()),
        "wrapped_linears": sum(isinstance(m, AOLinearQATWrapper) for m in model.modules()),
        "skipped_heads": sorted(set(skipped)),
    }


def iter_fake_quant_modules(model: torch.nn.Module):
    from torch.ao.quantization import FakeQuantize

    for module in model.modules():
        if isinstance(module, FakeQuantize):
            yield module


def set_observer_enabled(model: torch.nn.Module, enabled: bool) -> int:
    count = 0
    for module in iter_fake_quant_modules(model):
        module.enable_observer() if enabled else module.disable_observer()
        count += 1
    return count


def set_fake_quant_enabled(model: torch.nn.Module, enabled: bool) -> int:
    count = 0
    for module in iter_fake_quant_modules(model):
        module.enable_fake_quant() if enabled else module.disable_fake_quant()
        count += 1
    return count


def calibrate_ao_model(model: torch.nn.Module, batches: list, device: torch.device) -> int:
    observers = set_observer_enabled(model, True)
    set_fake_quant_enabled(model, False)
    model.eval()
    seen = 0
    with torch.no_grad():
        for batch in batches:
            nt = batch.to(device)
            model(nt)
            seen += int(nt.tensors.shape[0])
    set_observer_enabled(model, False)
    set_fake_quant_enabled(model, True)
    print(f"  [ao/calib] collected {observers} observer(s) over {seen} image(s)")
    return seen


def save_ao_checkpoint(path: Path, model: torch.nn.Module, config: dict[str, Any]) -> None:
    torch.save({"format": "pytorch_ao_fake_quant_rfdetr_qat_v1", "model_state_dict": model.state_dict(), "config": copy.deepcopy(config)}, path)


def load_ao_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=str(device), weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    set_observer_enabled(model, False)
    set_fake_quant_enabled(model, True)
    return obj if isinstance(obj, dict) else {}


def write_qat_csv(path: Path, rows: list[dict]) -> None:
    cols = ["epoch", "lr", "loss", "secs", "map", "map50"]
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join("" if row.get(c) is None else str(row.get(c)) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def resolve_image_source(name: str) -> Path:
    aliases = {"val": VAL_DIR, "val2017": VAL_DIR, "train": TRAIN_DIR, "train2017": TRAIN_DIR}
    path = aliases.get(name, Path(name))
    if not path.exists():
        raise FileNotFoundError(f"image source not found: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="rfdetr-nano")
    p.add_argument("--imgsz", type=int, default=0, help="Input size (0 = model default)")
    p.add_argument("--batch", type=int, default=4, help="Batch size for calibration/eval")
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib-source", default="val2017")
    p.add_argument("--calib-size", type=int, default=256)
    p.add_argument("--from-ao-ptq", type=Path, default=None)
    p.add_argument("--head-excludes", nargs="+", default=list(DEFAULT_HEAD_EXCLUDES))
    p.add_argument("--no-linear-quant", action="store_true", help="Only wrap Conv2d modules")
    p.add_argument("--ao-activation-observer", choices=("moving_average_minmax", "histogram"), default="histogram")
    p.add_argument("--eval-limit", type=int, default=0, help="Cap eval images (0 = full val2017)")
    p.add_argument("--skip-fp32-eval", action="store_true")
    p.add_argument("--skip-ptq-eval", action="store_true")
    p.add_argument("--skip-qat-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-qat", action="store_true")
    p.add_argument("--qat-source", choices=("train", "val"), default="train")
    p.add_argument("--qat-epochs", type=int, default=4)
    p.add_argument("--qat-images", type=int, default=480)
    p.add_argument("--qat-batch", type=int, default=2)
    p.add_argument("--qat-lr", type=float, default=1e-5)
    p.add_argument("--qat-low-lr", type=float, default=1e-6)
    p.add_argument("--qat-grad-clip", type=float, default=1.0)
    p.add_argument("--qat-log-every", type=int, default=20)
    p.add_argument("--qat-eval-every", type=int, default=1)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--onnx-opset", type=int, default=17)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    print(f"\n=== {args.model} :: PyTorch AO fake-quant QAT ===")
    model, model_res, postprocess = load_rfdetr(args.model, device)
    imgsz = args.imgsz or model_res
    do_eval = not args.skip_eval and ANN_FILE.exists()
    tag = args.model.replace("-", "_")
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model": args.model, "resolution": imgsz, "calib_size": args.calib_size, "head_excludes": list(args.head_excludes), "eval_limit": args.eval_limit, "ao_activation_observer": args.ao_activation_observer, "quantize_linears": not args.no_linear_quant}

    def run_eval(stage: str) -> dict | None:
        if not do_eval:
            return None
        print(f"  [eval:{stage}]")
        return evaluate_map(model, postprocess, VAL_DIR, ANN_FILE, device, imgsz, batch=args.batch, limit=args.eval_limit)

    if do_eval and not args.skip_fp32_eval:
        summary["fp32"] = run_eval("fp32")
    teacher = None if args.skip_qat else copy.deepcopy(model).to(device).eval()
    prep = prepare_ao_fake_quant(model, head_excludes=tuple(args.head_excludes), quantize_linears=not args.no_linear_quant, activation_observer=args.ao_activation_observer)
    model.to(device)
    summary["ao_prepare"] = prep
    print(f"  [ao/prepare] wrapped {prep['wrapped_convs']} Conv2d and {prep['wrapped_linears']} Linear module(s)")
    if prep["skipped_heads"]:
        print(f"  [ao/prepare] kept FP32 heads: {prep['skipped_heads'][:12]}")

    if args.from_ao_ptq is not None:
        print(f"  [ao/ptq] restoring {args.from_ao_ptq}")
        load_ao_checkpoint(args.from_ao_ptq, model, device)
    else:
        calib_dir = resolve_image_source(args.calib_source)
        calib_batches = build_calib_batches(calib_dir, imgsz, args.batch, args.calib_size, seed=args.seed, label="ao/calib")
        calibrate_ao_model(model, calib_batches, device)
        del calib_batches

    config = copy.deepcopy(summary)
    ptq_ckpt = out_dir / f"{tag}_ao_ptq.pth"
    save_ao_checkpoint(ptq_ckpt, model, config)
    summary["ptq_checkpoint"] = str(ptq_ckpt)
    print(f"  [ao/ptq] saved {ptq_ckpt} ({ptq_ckpt.stat().st_size/1e6:.1f} MB)")
    if do_eval and not args.skip_ptq_eval:
        summary["ptq"] = run_eval("ao-ptq")

    if not args.skip_qat:
        qat_dir = TRAIN_DIR if args.qat_source == "train" else VAL_DIR
        train_batches = build_calib_batches(qat_dir, imgsz, args.qat_batch, args.qat_images, seed=args.seed + 1, label="ao/qat")
        best: dict[str, Any] = {"state": None, "epoch": 0, "metrics": None}

        def eval_cb(epoch_idx: int):
            if not do_eval or args.skip_qat_eval or args.qat_eval_every <= 0:
                return None
            if epoch_idx % args.qat_eval_every == 0 or epoch_idx == args.qat_epochs:
                return evaluate_map(model, postprocess, VAL_DIR, ANN_FILE, device, imgsz, batch=args.batch, limit=args.eval_limit)
            return None

        def on_improve(epoch_idx: int, metrics: dict):
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch_idx
            best["metrics"] = metrics
            print(f"  [ao/qat] new best mAP50-95={metrics['map']:.4f} @ epoch {epoch_idx}")

        t0 = time.time()
        history = run_distillation_qat(model, teacher, train_batches, device, epochs=args.qat_epochs, peak_lr=args.qat_lr, low_lr=args.qat_low_lr, grad_clip=args.qat_grad_clip, log_every=args.qat_log_every, eval_callback=eval_cb, on_improve=on_improve)
        print(f"  [ao/qat] elapsed {time.time() - t0:.1f}s")
        summary["qat_train"] = history
        del train_batches, teacher
        if best["state"] is not None:
            model.load_state_dict(best["state"])
            summary["qat"] = best["metrics"]
            summary["qat_best_epoch"] = best["epoch"]
            print(f"  [ao/qat] restored best epoch {best['epoch']}")
        elif do_eval and not args.skip_qat_eval:
            summary["qat"] = run_eval("ao-qat")
        qat_ckpt = out_dir / f"{tag}_ao_qat.pth"
        save_ao_checkpoint(qat_ckpt, model, config)
        summary["qat_checkpoint"] = str(qat_ckpt)
        print(f"  [ao/qat] saved {qat_ckpt} ({qat_ckpt.stat().st_size/1e6:.1f} MB)")
        write_qat_csv(out_dir / "ao_qat_train.csv", history)

    if not args.no_export:
        suffix = "ptq" if args.skip_qat else "qat"
        onnx_dst = out_dir / f"{tag}_ao_{suffix}.onnx"
        export_onnx(model, imgsz, device, onnx_dst, opset=args.onnx_opset)
        if onnx_dst.exists():
            summary["onnx"] = str(onnx_dst)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [done] {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
