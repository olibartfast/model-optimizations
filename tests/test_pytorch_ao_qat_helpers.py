from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _load_ao_module():
    module_dir = Path(__file__).resolve().parents[1] / "yolo_quantization" / "qat"
    module_path = module_dir / "pytorch_ao_yolo_qat.py"
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location("pytorch_ao_qat_module", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(module_dir))


def test_resolve_ao_recipe_auto_by_model_family():
    module = _load_ao_module()

    assert module._resolve_ao_recipe("auto", "yolo26s") == "yolo26-ao-distill"
    assert module._resolve_ao_recipe("auto", "yolo11s") == "yolo11-ao-distill"
    assert module._resolve_ao_recipe("yolo26-ao-distill", "yolo11s") == "yolo26-ao-distill"


def test_resolve_ao_recipe_rejects_unknown():
    module = _load_ao_module()

    with pytest.raises(ValueError):
        module._resolve_ao_recipe("typo", "yolo11s")


def test_apply_ao_recipe_respects_explicit_flags():
    module = _load_ao_module()

    class A:
        pass

    args = A()
    args.ao_recipe = "yolo11-ao-distill"
    args.models = ["yolo11s"]
    args.qat_epochs = 10
    args.qat_batches_per_epoch = 200
    args.qat_lr = 1e-5
    args.qat_low_lr = 1e-6
    args.qat_distill_weight = 1.0
    args.qat_supervised_weight = 1.0
    args.calib_size = 260
    args.exclude_dfl_quant = False
    args.disable_detect_output_quant = False

    resolved = module._apply_ao_recipe(args, {"--calib-size"})

    assert resolved == "yolo11-ao-distill"
    assert args.calib_size == 260
    assert args.exclude_dfl_quant is True
    assert args.disable_detect_output_quant is True


def test_prepare_pytorch_ao_fake_quant_wraps_conv2d_and_preserves_forward_shape():
    module = _load_ao_module()
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, 1),
    )

    wrapped = module.prepare_pytorch_ao_fake_quant(model)

    assert wrapped == 2
    assert isinstance(model[0], module.AOConv2dQATWrapper)
    assert isinstance(model[2], module.AOConv2dQATWrapper)
    y = model(torch.randn(1, 3, 8, 8))
    assert y.shape == (1, 2, 8, 8)


def test_prepare_pytorch_ao_fake_quant_respects_exclude_prefixes():
    module = _load_ao_module()
    model = torch.nn.Module()
    model.add_module("backbone", torch.nn.Conv2d(3, 4, 1))
    model.add_module("head", torch.nn.Module())
    model.head.add_module("dfl", torch.nn.Conv2d(4, 4, 1))

    wrapped = module.prepare_pytorch_ao_fake_quant(model, exclude_prefixes=["head"])

    assert wrapped == 1
    assert isinstance(model.backbone, module.AOConv2dQATWrapper)
    assert isinstance(model.head.dfl, torch.nn.Conv2d)


def test_ao_fake_quant_state_toggles_all_fake_quant_modules():
    module = _load_ao_module()
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
    module.prepare_pytorch_ao_fake_quant(model)

    assert module.set_ao_observer_enabled(model, False) == 2
    assert module.set_ao_fake_quant_enabled(model, False) == 2
    fake_quant_modules = list(module.iter_ao_fake_quant_modules(model))

    assert len(fake_quant_modules) == 2
    assert all(int(fq.observer_enabled[0]) == 0 for fq in fake_quant_modules)
    assert all(int(fq.fake_quant_enabled[0]) == 0 for fq in fake_quant_modules)

    module.set_ao_observer_enabled(model, True)
    module.set_ao_fake_quant_enabled(model, True)

    assert all(int(fq.observer_enabled[0]) == 1 for fq in fake_quant_modules)
    assert all(int(fq.fake_quant_enabled[0]) == 1 for fq in fake_quant_modules)


def test_snapshot_ao_qparams_records_scale_after_forward():
    module = _load_ao_module()
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
    module.prepare_pytorch_ao_fake_quant(model)

    model(torch.randn(2, 3, 8, 8))
    snapshot = module.snapshot_ao_qparams(model)

    assert snapshot
    assert all(torch.is_tensor(scale) for scale in snapshot.values())
