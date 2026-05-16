from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_ptq_module():
    module_path = Path(__file__).resolve().parents[1] / "yolo_quantization" / "ptq" / "nvidia_modelopt_yolo.py"
    spec = importlib.util.spec_from_file_location("ptq_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_size_mb_returns_expected_value(tmp_path):
    module = _load_ptq_module()
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"x" * (2 * 1024 * 1024))

    assert module._size_mb(file_path) == pytest.approx(2.0, rel=1e-6)


def test_parse_args_defaults_and_quant_mode_validation():
    module = _load_ptq_module()
    args = module.parse_args([])
    assert args.models == list(module.DEFAULT_MODELS)
    assert args.quant_modes == list(module.DEFAULT_MODES)

    with pytest.raises(SystemExit):
        module.parse_args(["--quant-modes", "not-a-mode"])


def test_parse_args_accepts_multiple_quant_modes():
    module = _load_ptq_module()
    args = module.parse_args(["--quant-modes", "int8", "fp8", "--imgsz", "320"])

    assert args.quant_modes == ["int8", "fp8"]
    assert args.imgsz == 320


def test_resolve_dataset_split_for_custom_yaml(tmp_path):
    module = _load_ptq_module()
    root = tmp_path / "dataset"
    val_dir = root / "images" / "val"
    val_dir.mkdir(parents=True)
    data_yaml = tmp_path / "custom.yaml"
    data_yaml.write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: {0: object}\n"
    )

    assert module.resolve_dataset_split(data_yaml, "val") == val_dir.resolve()


def test_resolve_calib_source_accepts_custom_path(tmp_path):
    module = _load_ptq_module()
    data_yaml = tmp_path / "custom.yaml"
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    data_yaml.write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: {0: object}\n"
    )

    assert module.resolve_calib_source(str(calib_dir), data_yaml) == calib_dir.resolve()
