#!/bin/bash
# Quantization Environment Setup Script
# Run from the model-optimizations root directory

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3.12 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic packages first
echo "Installing basic packages..."
pip install wheel ultralytics onnx

# Install NVIDIA packages with specific approach to avoid build issues
echo "Installing NVIDIA ModelOpt..."
pip install --no-build-isolation --extra-index-url https://pypi.ngc.nvidia.com nvidia-modelopt

echo "✅ Environment setup complete!"
echo "Activate with: source venv/bin/activate"
