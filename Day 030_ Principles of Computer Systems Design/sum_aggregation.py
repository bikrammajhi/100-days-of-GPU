"""
===============================================================================
Title  : Sum Aggregation Over First Dimension using Triton and PyTorch
Original Author : https://github.com/Aalanli/MusicGeneration/tree/7d268322d692013d8ac6e70be31741cea519fa28
Date   : 2025-06-12
Purpose:
    - Demonstrates how to sum over the first dimension of a tensor using Triton.
    - Compares Triton performance to PyTorch's built-in sum.
===============================================================================
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import torch
import triton
import triton.language as tl
import torch.nn as nn
import time

# -----------------------------------------------------------------------------
# Triton Kernel: Sum over first dimension (dim=0)
# -----------------------------------------------------------------------------

@triton.jit
def sum_over_first_dim_kernel(in_ptr, out_ptr, N, D, XBLOCK: tl.constexpr):
    """
    Parameters:
    - in_ptr  : Pointer to input data (flattened)
    - out_ptr : Pointer to output buffer
    - N       : Number of rows (first dimension)
    - D       : Number of elements per row (flattened second+ dims)
    - XBLOCK  : Number of elements processed per program instance (warp)
    """

    # Program ID (each program instance handles XBLOCK elements)
    pid = tl.program_id(0)

    # Compute offsets for this program
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)

    # Mask for valid indices (to avoid out-of-bounds)
    mask = offsets < D

    # Accumulator initialized to 0
    acc = tl.zeros([XBLOCK], dtype=tl.float32)

    # Loop over all rows (N) and accumulate
    for i in range(0, N):
        acc += tl.load(in_ptr + i * D + offsets, mask=mask)

    # Write result to global memory
    tl.store(out_ptr + offsets, acc, mask=mask)

# -----------------------------------------------------------------------------
# PyTorch Wrapper Module
# -----------------------------------------------------------------------------

class SumAggregatorGeneral(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Input:  x of shape (N, ...)
        Output: sum over dim=0, shape (...)
        """
        N = x.shape[0]
        D = x[0].numel()  # Flatten all dims except dim=0

        # Flatten input tensor to shape (N * D,)
        x_flat = x.contiguous().view(N * D)

        # Allocate output buffer
        out_flat = torch.empty(D, device=x.device, dtype=torch.float32)

        # Launch Triton kernel
        grid = lambda meta: (triton.cdiv(D, meta['XBLOCK']),)
        sum_over_first_dim_kernel[grid](x_flat, out_flat, N, D, XBLOCK=1024)

        return out_flat.view(*x.shape[1:])

# -----------------------------------------------------------------------------
# Correctness Check (Small Tensor)
# -----------------------------------------------------------------------------

x = torch.randn(4, 4, 4, 4, device='cuda', dtype=torch.float32)
aggregator = SumAggregatorGeneral()

out_triton = aggregator(x)
out_torch = x.sum(dim=0)

print("Equal outputs:", torch.allclose(out_triton, out_torch, atol=1e-4))  # ✅ Should be True

# -----------------------------------------------------------------------------
# Benchmark (Larger Tensor)
# -----------------------------------------------------------------------------

x = torch.randn(512, 64, 64, device='cuda', dtype=torch.float32)

# Warm-up runs
aggregator(x)
x.sum(dim=0)

# Time Triton
start = time.time()
for _ in range(10):
    out1 = aggregator(x)
torch.cuda.synchronize()
print("Triton time per run: {:.6f} sec".format((time.time() - start) / 10))

# Time PyTorch
start = time.time()
for _ in range(10):
    out2 = x.sum(dim=0)
torch.cuda.synchronize()
print("PyTorch time per run: {:.6f} sec".format((time.time() - start) / 10))

# Verify correctness
print("Equal outputs (large tensor):", torch.allclose(out1, out2, atol=1e-4))
