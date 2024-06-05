# DLAI quantization playground



## How to Get a Quantized Tensor `q`

Having: 

- `r`: The real value tensor.
- `s`: The scaling factor.
- `z`: The zero-point or offset.


### Quantize the Tensor `q`
```
q = round((r / s) + z)
```

### Ensure `q` is within the Quantized Range `[q_min, q_max]`

After calculating `q`, clip it to ensure it falls within the valid range, 
the quantized tensor is calculated using the formula:

```
q = max(q_min, min((r / s) + z, q_max))
```

This process is demonstrated in the script ``linear_quantize_dequantize_tensor.py``.



# References

* https://www.coursera.org/projects/quantization-fundamentals
* https://www.deeplearning.ai/short-courses/quantization-in-depth/
