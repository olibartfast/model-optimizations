import torch

# Quantization and Dequantization with Random Scale and Zero Point

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):
    """
    Quantizes a given tensor using the linear quantization scheme with specified scale and zero point.

    Parameters:
        tensor (torch.Tensor): The input tensor to be quantized.
        scale (float): The scale factor used for quantization.
        zero_point (int): The zero point value used for quantization.
        dtype (torch.dtype, optional): The data type of the quantized tensor. Defaults to torch.int8.

    Returns:
        torch.Tensor: The quantized tensor.

    """
    # It scales and shifts the input tensor by dividing it by the scale and adding the zero point.
    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    
    return q_tensor


def linear_dequantization(quantized_tensor, scale, zero_point):
    """
    Dequantizes a given quantized tensor using the linear dequantization scheme with specified scale and zero point.

    Parameters:
        quantized_tensor (torch.Tensor): The input quantized tensor to be dequantized.
        scale (float): The scale factor used for dequantization.
        zero_point (int): The zero point value used for dequantization.

    Returns:
        torch.Tensor: The dequantized tensor.

    """    
    return scale * (quantized_tensor.float() - zero_point)




### a dummy tensor to test the implementation
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

### these are random values for "scale" and "zero_point"
### to test the implementation
scale = 3.5
zero_point = -70

quantized_tensor = linear_q_with_scale_and_zero_point(
    test_tensor, scale, zero_point)

print(quantized_tensor)
#tensor([[ -15,  -74,  127],
#        [ -44,   14, -123],
#        [ -70,  126,    0]], dtype=torch.int8)



dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
print(dequantized_tensor)
# tensor([[ 192.5000,  -14.0000,  689.5000],
#         [  91.0000,  294.0000, -185.5000],
#         [   0.0000,  686.0000,  245.0000]])

print(dequantized_tensor - test_tensor)

print(dequantized_tensor - test_tensor).square().mean()