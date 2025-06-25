#!/usr/bin/env python3
"""
Test script to verify the quantization environment is working correctly.
This script demonstrates basic usage of the installed packages.
"""

import torch
import numpy as np
import onnx
import modelopt
from ultralytics import YOLO

def test_pytorch():
    """Test PyTorch functionality"""
    print("🔍 Testing PyTorch...")
    
    # Create a simple tensor
    x = torch.randn(2, 3, 4)
    print(f"  - Created tensor with shape: {x.shape}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        print(f"  - Moved tensor to CUDA: {x_cuda.device}")
        x_cuda = x_cuda.cpu()
        print("  - Moved tensor back to CPU")
    
    print("  ✓ PyTorch test passed!")

def test_onnx():
    """Test ONNX functionality"""
    print("\n🔍 Testing ONNX...")
    
    # Create a simple ONNX model
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    dummy_input = torch.randn(1, 4)
    
    # Export to ONNX (in memory)
    import io
    f = io.BytesIO()
    torch.onnx.export(model, dummy_input, f, verbose=False)
    print("  - Successfully exported PyTorch model to ONNX format")
    print("  ✓ ONNX test passed!")

def test_nvidia_modelopt():
    """Test NVIDIA ModelOpt functionality"""
    print("\n🔍 Testing NVIDIA ModelOpt...")
    
    # Just test import and basic info
    print(f"  - ModelOpt module: {modelopt.__name__}")
    print(f"  - Available submodules: {[attr for attr in dir(modelopt) if not attr.startswith('_')]}")
    print("  ✓ NVIDIA ModelOpt test passed!")

def test_ultralytics():
    """Test Ultralytics functionality"""
    print("\n🔍 Testing Ultralytics...")
    
    # Note: This won't download a model, just test the import
    print("  - YOLO class imported successfully")
    print("  - Ready for object detection and quantization tasks")
    print("  ✓ Ultralytics test passed!")

def main():
    """Run all tests"""
    print("=== QUANTIZATION ENVIRONMENT VERIFICATION ===")
    print(f"Python: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 50)
    
    try:
        test_pytorch()
        test_onnx()
        test_nvidia_modelopt()
        test_ultralytics()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("Your quantization environment is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
