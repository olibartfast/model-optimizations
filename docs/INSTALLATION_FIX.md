# NVIDIA PyIndex Installation

## Problem
The `nvidia-pyindex` package was failing to build during installation with the following error:
```
Building wheel for nvidia-pyindex (pyproject.toml) ... error
RuntimeError: Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'pip'
```

## Root Cause
The `nvidia-pyindex` package has build issues in Python 3.12 environments, specifically related to accessing pip configuration during the wheel building process.

## Solution
Instead of using `nvidia-pyindex`, install NVIDIA packages directly from the NVIDIA PyPI index:

### Method 1: Direct installation (Recommended)
```bash
# Install basic packages first
pip install wheel ultralytics onnx

# Install NVIDIA packages directly with no build isolation
pip install --no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com nvidia-modelopt
```

### Method 2: Alternative approach
```bash
# Add the NVIDIA index permanently to pip configuration
pip config set global.extra-index-url https://pypi.ngc.nvidia.com

# Then install normally
pip install nvidia-modelopt
```

## Updated Files

### requirements.txt
```
wheel

# Install the core libraries
ultralytics
onnx

# Install the NVIDIA model optimizer directly (nvidia-pyindex has build issues)  
# Use: pip install --no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com nvidia-modelopt
nvidia-modelopt
```

### run_venv.sh
```bash
#!/bin/bash
# source run_venv.sh 

# python3.12 -m venv venv
source ~/workspace/quantization_playground/venv/bin/activate

# Install basic packages first
pip install wheel ultralytics onnx

# Install NVIDIA packages with specific approach to avoid build issues
pip install --no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com nvidia-modelopt
```

## Testing
To verify the installation works:

```python
import torch
import onnx
import modelopt  # Note: import as 'modelopt', not 'nvidia_modelopt'

print("PyTorch version:", torch.__version__)
print("ONNX version:", onnx.__version__)
print("All packages imported successfully!")
```

## Key Points
1. **No nvidia-pyindex needed**: Skip the problematic nvidia-pyindex package entirely
2. **Direct installation**: Use `--extra-index-url https://pypi.ngc.nvidia.com`
3. **No build isolation**: Use `--no-build-isolation` flag to avoid build issues
4. **Import name**: Import as `modelopt`, not `nvidia_modelopt`
5. **Python 3.12 compatible**: This solution works with Python 3.12 on Ubuntu

## Environment Details
- Ubuntu version: Older version with Python 3.12 from Deadsnakes PPA
- Python version: 3.12
- Virtual environment: venv
- PyTorch: 2.7.1+cu126 (CUDA-enabled)
