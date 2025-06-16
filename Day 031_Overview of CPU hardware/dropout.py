# https://isamu-website.medium.com/understanding-triton-tutorials-part-2-f6839ce50ae7

# -----------------------------------------------------------------------------
# Dropout Implementation using Triton
# -----------------------------------------------------------------------------
# This example demonstrates a simple implementation of Dropout using Triton.
# Dropout is a common regularization technique used in neural networks.
# -----------------------------------------------------------------------------

import torch
import tabulate
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Device Kernel: Dropout
# -----------------------------------------------------------------------------
@triton.jit
def dropout_kernel(
    x_ptr,           # * Input tensor
    mask_ptr,        # * Dropout mask (0s and 1s)
    out_ptr,         # * Output tensor
    n_elements,      # * Total number of elements
    p,               # * Dropout probability
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    keep_mask = tl.load(mask_ptr + offsets, mask=mask)

    # Apply dropout: scale retained elements, zero out others
    y = tl.where(keep_mask, x / (1 - p), 0.0)

    # Store output
    tl.store(out_ptr + offsets, y, mask=mask)

# -----------------------------------------------------------------------------
# Host Function: Dropout Wrapper
# -----------------------------------------------------------------------------
def dropout(x: torch.Tensor, keep_mask: torch.Tensor, p: float) -> torch.Tensor:
    """
    Applies dropout to tensor `x` using mask `keep_mask`.

    Args:
        x (Tensor): Input tensor.
        keep_mask (Tensor): Tensor of 0s and 1s indicating which elements to keep.
        p (float): Dropout probability.

    Returns:
        Tensor: Output tensor after applying dropout.
    """
    assert x.is_cuda and keep_mask.is_cuda, "Inputs must be on CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    dropout_kernel[grid](x, keep_mask, output, n_elements, p, BLOCK_SIZE=1024)
    return output

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    p = 0.5
    x = torch.randn(10, device='cuda')
    keep_mask = (torch.rand_like(x) > p).to(torch.int32)

    out = dropout(x, keep_mask, p)

    # Display inputs and outputs
    print(tabulate.tabulate([
        ["Input      "] + x.tolist(),
        ["Keep Mask  "] + keep_mask.tolist(),
        ["Dropout Out"] + out.tolist(),
    ]))
