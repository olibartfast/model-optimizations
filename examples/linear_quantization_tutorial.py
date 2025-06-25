"""
Basic Linear Quantization Tutorial
==================================

This module demonstrates fundamental quantization concepts from DeepLearning.AI,
integrated into our quantization playground.

Key Concepts:
- Linear quantization formula: q = round((r / s) + z)
- Scale factor (s) and zero point (z) calculation
- Quantization and dequantization process
- Quantization error analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def linear_quantize_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """
    Quantizes a tensor using linear quantization scheme.
    
    Formula: q = round((r / s) + z)
    
    Args:
        tensor (torch.Tensor): Input tensor to quantize
        scale (float): Scale factor for quantization
        zero_point (int): Zero point offset
        dtype (torch.dtype): Target quantized data type
        
    Returns:
        torch.Tensor: Quantized tensor
    """
    # Scale and shift the tensor
    scaled_and_shifted = tensor / scale + zero_point
    
    # Round to nearest integer
    rounded = torch.round(scaled_and_shifted)
    
    # Clamp to valid range for the dtype
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    quantized = rounded.clamp(q_min, q_max).to(dtype)
    
    return quantized


def linear_dequantize(quantized_tensor, scale, zero_point):
    """
    Dequantizes a quantized tensor back to floating point.
    
    Formula: r = s * (q - z)
    
    Args:
        quantized_tensor (torch.Tensor): Quantized tensor
        scale (float): Scale factor used in quantization
        zero_point (int): Zero point used in quantization
        
    Returns:
        torch.Tensor: Dequantized tensor
    """
    return scale * (quantized_tensor.float() - zero_point)


def calculate_scale_and_zero_point(tensor, dtype=torch.int8):
    """
    Calculate optimal scale and zero point for a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor
        dtype (torch.dtype): Target quantized data type
        
    Returns:
        tuple: (scale, zero_point)
    """
    # Get the range of the quantized data type
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    
    # Get the range of the input tensor
    r_min = tensor.min().item()
    r_max = tensor.max().item()
    
    # Calculate scale
    scale = (r_max - r_min) / (q_max - q_min)
    
    # Calculate zero point
    zero_point = q_min - r_min / scale
    zero_point = int(round(zero_point))
    
    # Clamp zero point to valid range
    zero_point = max(q_min, min(q_max, zero_point))
    
    return scale, zero_point


def quantization_demo():
    """
    Comprehensive quantization demonstration with error analysis.
    """
    print("=" * 50)
    print("Linear Quantization Tutorial")
    print("=" * 50)
    
    # Create a test tensor with a range of values
    test_tensor = torch.tensor([
        [191.6, -13.5, 728.6],
        [92.14, 295.5, -184.0],
        [0.0, 684.6, 245.5]
    ])
    
    print(f"Original tensor:\n{test_tensor}")
    print(f"Tensor range: [{test_tensor.min():.2f}, {test_tensor.max():.2f}]")
    
    # Method 1: Manual scale and zero point
    print("\n" + "-" * 30)
    print("Method 1: Manual scale and zero point")
    print("-" * 30)
    
    scale_manual = 3.5
    zero_point_manual = -70
    
    quantized_manual = linear_quantize_with_scale_and_zero_point(
        test_tensor, scale_manual, zero_point_manual
    )
    dequantized_manual = linear_dequantize(quantized_manual, scale_manual, zero_point_manual)
    
    print(f"Manual scale: {scale_manual}, zero_point: {zero_point_manual}")
    print(f"Quantized tensor:\n{quantized_manual}")
    print(f"Dequantized tensor:\n{dequantized_manual}")
    
    error_manual = (dequantized_manual - test_tensor).square().mean()
    print(f"Quantization error (MSE): {error_manual:.4f}")
    
    # Method 2: Calculated optimal scale and zero point
    print("\n" + "-" * 30)
    print("Method 2: Calculated optimal scale and zero point")
    print("-" * 30)
    
    scale_calc, zero_point_calc = calculate_scale_and_zero_point(test_tensor)
    
    quantized_calc = linear_quantize_with_scale_and_zero_point(
        test_tensor, scale_calc, zero_point_calc
    )
    dequantized_calc = linear_dequantize(quantized_calc, scale_calc, zero_point_calc)
    
    print(f"Calculated scale: {scale_calc:.4f}, zero_point: {zero_point_calc}")
    print(f"Quantized tensor:\n{quantized_calc}")
    print(f"Dequantized tensor:\n{dequantized_calc}")
    
    error_calc = (dequantized_calc - test_tensor).square().mean()
    print(f"Quantization error (MSE): {error_calc:.4f}")
    
    # Compare methods
    print("\n" + "=" * 30)
    print("Comparison")
    print("=" * 30)
    print(f"Manual method error: {error_manual:.4f}")
    print(f"Calculated method error: {error_calc:.4f}")
    print(f"Improvement: {((error_manual - error_calc) / error_manual * 100):.1f}%")
    
    return {
        'original': test_tensor,
        'quantized_manual': quantized_manual,
        'dequantized_manual': dequantized_manual,
        'quantized_calc': quantized_calc,
        'dequantized_calc': dequantized_calc,
        'error_manual': error_manual,
        'error_calc': error_calc
    }


def visualize_quantization_error(tensor, scale, zero_point):
    """
    Visualize quantization effects on a larger tensor.
    """
    # Create a smooth signal
    x = torch.linspace(-10, 10, 1000)
    signal = torch.sin(x) * 5 + torch.cos(x * 2) * 2
    
    # Quantize and dequantize
    quantized = linear_quantize_with_scale_and_zero_point(signal, scale, zero_point)
    dequantized = linear_dequantize(quantized, scale, zero_point)
    
    # Calculate error
    error = (dequantized - signal).abs()
    
    print(f"\nSignal quantization analysis:")
    print(f"Original signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    print(f"Mean absolute error: {error.mean():.4f}")
    print(f"Max absolute error: {error.max():.4f}")
    
    return signal.numpy(), dequantized.numpy(), error.numpy()


if __name__ == "__main__":
    # Run the demonstration
    results = quantization_demo()
    
    # Additional analysis with a signal
    print("\n" + "=" * 50)
    print("Signal Quantization Analysis")
    print("=" * 50)
    
    # Use calculated optimal parameters
    scale, zero_point = calculate_scale_and_zero_point(
        torch.tensor([-7.0, 7.0])  # Expected signal range
    )
    
    original, quantized_signal, error = visualize_quantization_error(
        None, scale, zero_point
    )
    
    print("\n✅ Linear quantization tutorial completed!")
    print("Key takeaways:")
    print("1. Optimal scale and zero point minimize quantization error")
    print("2. Quantization introduces controlled precision loss")
    print("3. INT8 quantization can preserve most information with proper calibration")
