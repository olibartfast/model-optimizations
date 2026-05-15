from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _load_qat_module():
    module_path = Path(__file__).resolve().parents[1] / "yolo_quantization" / "qat" / "nvidia_modelopt_yolo_qat.py"
    spec = importlib.util.spec_from_file_location("qat_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distill_epoch_lr_uses_low_high_low_schedule():
    module = _load_qat_module()
    peak = 1e-4
    low = 1e-6

    assert module._distill_epoch_lr(0, 2, peak, low) == peak
    assert module._distill_epoch_lr(0, 6, peak, low) == low
    assert module._distill_epoch_lr(2, 6, peak, low) == peak
    assert module._distill_epoch_lr(5, 6, peak, low) == low


def test_teacher_student_raw_outputs_supports_list_tuple_and_tensor():
    module = _load_qat_module()
    t1 = torch.randn(1, 2, 3)
    t2 = torch.randn(1, 2, 3)

    assert module._teacher_student_raw_outputs([t1, t2]) == [t1, t2]
    assert module._teacher_student_raw_outputs(("ignored", [t1, t2])) == [t1, t2]
    assert module._teacher_student_raw_outputs(t1) == [t1]
    assert module._teacher_student_raw_outputs({"one2many": [t1, t2], "one2one": [t2]}) == [t1, t2]
    assert module._teacher_student_raw_outputs({"one2many": {}, "one2one": {"feats": [t1, t2]}}) == [t1, t2]


def test_teacher_student_raw_outputs_raises_for_unsupported_type():
    module = _load_qat_module()
    with pytest.raises(TypeError):
        module._teacher_student_raw_outputs({"unexpected": "mapping"})


def test_mse_distill_loss_matches_expected_manual_sum():
    module = _load_qat_module()
    student = [torch.zeros(2, 2, requires_grad=True), torch.ones(2, 2, requires_grad=True)]
    teacher = [torch.ones(2, 2, requires_grad=True), torch.zeros(2, 2, requires_grad=True)]

    got = module._mse_distill_loss(student, teacher)
    expected = torch.nn.functional.mse_loss(student[0], teacher[0]) + torch.nn.functional.mse_loss(
        student[1], teacher[1]
    )
    assert torch.isclose(got, expected)
    got.backward()
    assert all(t.grad is None for t in teacher)
    assert all(s.grad is not None for s in student)


def test_mse_distill_loss_raises_when_output_lengths_differ():
    module = _load_qat_module()
    with pytest.raises(RuntimeError):
        module._mse_distill_loss([torch.zeros(1)], [torch.zeros(1), torch.zeros(1)])


def test_trainable_parameters_unfreezes_student_model():
    module = _load_qat_module()
    model = torch.nn.Linear(2, 1)
    for param in model.parameters():
        param.requires_grad = False

    params = module._trainable_parameters(model)

    assert params
    assert all(param.requires_grad for param in params)


def test_clone_inference_buffers_ignores_normal_buffers():
    module = _load_qat_module()
    model = torch.nn.BatchNorm2d(2)

    assert module._clone_inference_buffers(model) == 0


def test_clone_quantizer_amax_state_clones_tensor_property():
    module = _load_qat_module()

    class DummyQuantizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.ones(1))

        @property
        def amax(self):
            return self._amax

        @amax.setter
        def amax(self, value):
            self._amax = value

    quantizer = DummyQuantizer()
    original = quantizer._buffers["_amax"]

    assert module._clone_quantizer_amax_state(quantizer) == 1
    assert torch.equal(quantizer.amax, original)
    assert quantizer._buffers["_amax"] is not original


def test_prepare_quantized_model_for_training_clones_amax_state():
    module = _load_qat_module()

    class DummyQuantizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_amax", torch.ones(1))

    quantizer = DummyQuantizer()
    original = quantizer._buffers["_amax"]

    inference_buffers, amax_states = module._prepare_quantized_model_for_training(quantizer)

    assert inference_buffers == 0
    assert amax_states == 1
    assert torch.equal(quantizer._buffers["_amax"], original)
    assert quantizer._buffers["_amax"] is not original


def test_prepare_quantized_model_for_training_clones_inference_buffer_to_normal_tensor():
    module = _load_qat_module()
    model = torch.nn.Module()
    with torch.inference_mode():
        model.register_buffer("scale", torch.ones(1))
    original = model._buffers["scale"]

    inference_buffers, amax_states = module._prepare_quantized_model_for_training(model)

    assert inference_buffers == 1
    assert amax_states == 0
    assert original.is_inference()
    assert not model._buffers["scale"].is_inference()
    assert torch.equal(model._buffers["scale"], original)


def test_restore_modelopt_checkpoint_registers_extra_amax_buffers(monkeypatch, tmp_path):
    module = _load_qat_module()
    model = torch.nn.Module()
    model.add_lhs_quantizer = torch.nn.Module()
    ckpt = {
        "modelopt_state": {},
        "model_state_dict": {"add_lhs_quantizer._amax": torch.ones(1)},
    }
    ckpt_path = tmp_path / "modelopt.pth"
    torch.save(ckpt, ckpt_path)

    class DummyMto:
        @staticmethod
        def restore_from_modelopt_state(model_arg, modelopt_state):
            assert modelopt_state == {}
            return model_arg

    restored = module._restore_modelopt_checkpoint(DummyMto, model, ckpt_path, torch.device("cpu"))

    assert torch.equal(restored.add_lhs_quantizer._buffers["_amax"], torch.ones(1))
