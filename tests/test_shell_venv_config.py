from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_without_python_overrides() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PY", None)
    env.pop("QUANTIZATION_VENV", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def test_run_venv_requires_personal_venv_placeholder():
    result = subprocess.run(
        ["bash", "scripts/run_venv.sh"],
        cwd=REPO_ROOT,
        env=_env_without_python_overrides(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "QUANTIZATION_VENV" in result.stderr
    assert "<venv-dir>" in result.stderr


def test_top_level_wrapper_requires_configured_python():
    result = subprocess.run(
        ["bash", "scripts/run_modelopt_yolo.sh", "--help"],
        cwd=REPO_ROOT,
        env=_env_without_python_overrides(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "Set PY=<python> or" in result.stderr
    assert "QUANTIZATION_VENV=<venv-dir>" in result.stderr


def test_legacy_wrapper_requires_configured_python():
    result = subprocess.run(
        ["bash", "yolo_quantization/scripts/run_modelopt_yolo.sh", "--help"],
        cwd=REPO_ROOT,
        env=_env_without_python_overrides(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "Set PY=<python> or" in result.stderr
    assert "QUANTIZATION_VENV=<venv-dir>" in result.stderr
