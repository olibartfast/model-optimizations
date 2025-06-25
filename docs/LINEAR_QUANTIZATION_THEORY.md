# Linear Quantization Theory

This document contains the theoretical foundation for linear quantization, originally from DeepLearning.AI's quantization course.

## Quantization Formula

The core formula for linear quantization is:

```
q = round((r / s) + z)
```

Where:
- `r`: The real (floating-point) value tensor
- `s`: The scaling factor
- `z`: The zero-point or offset
- `q`: The quantized (integer) value

## Quantization Process

### Step 1: Quantize the Tensor
```python
q = round((r / s) + z)
```

### Step 2: Ensure Valid Range
After calculating `q`, clip it to ensure it falls within the valid quantized range `[q_min, q_max]`:

```python
q = max(q_min, min(q, q_max))
```

## Dequantization

To convert back to floating-point:

```python
r_reconstructed = s * (q - z)
```

## Scale and Zero Point Calculation

For optimal quantization, calculate scale and zero point based on the tensor's range:

```python
# Get tensor range
r_min, r_max = tensor.min(), tensor.max()

# Get quantized type range  
q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

# Calculate scale
scale = (r_max - r_min) / (q_max - q_min)

# Calculate zero point
zero_point = q_min - r_min / scale
zero_point = int(round(zero_point))
zero_point = max(q_min, min(q_max, zero_point))
```

## Implementation

See `examples/linear_quantization_tutorial.py` for a complete implementation with:
- Linear quantization functions
- Automatic scale/zero-point calculation
- Error analysis and comparison
- Visualization examples

## References

- [Coursera: Quantization Fundamentals](https://www.coursera.org/projects/quantization-fundamentals)
- [DeepLearning.AI: Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
