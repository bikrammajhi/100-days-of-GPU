import torch
import triton
import triton.language as tl

# Define the matrix multiplication operation using Triton
@triton.jit
def _matmul_kernel(A, B, C, M, N, K, **meta):
    # Define the tile size for the operation
    TILE_M = meta['BLOCK_M']
    TILE_N = meta['BLOCK_N']
    TILE_K = 128

    # Get the indices for the current tile
    m = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    n = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)

    # Compute the accumulation for the current tile
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, K, TILE_K):
        # Load a tile of A and B into shared memory
        a = tl.load(A + m[:, None] * K + k, mask=[m[:, None] < M, None], other=0.0)
        b = tl.load(B + k * N + n, mask=[None, n < N], other=0.0)
        # Perform the matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    tl.store(C + m[:, None] * N + n, acc, mask=[m[:, None] < M, n < N])

def matmul(A, B, BLOCK_M=128, BLOCK_N=128):
    M, K = A.shape
    K, N = B.shape

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid dimensions
    grid = lambda meta: [triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N'])]

    # Launch kernel
    _matmul_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return C

# Define the size of the matrices
size = 1024

# Create random matrices
A = torch.randn(size, size, device='cuda', dtype=torch.float32)
B = torch.randn(size, size, device='cuda', dtype=torch.float32)

# Benchmark the matrix multiplication
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = matmul(A, B)
end.record()
torch.cuda.synchronize()

# Calculate elapsed time
elapsed_time_ms = start.elapsed_time(end)
elapsed_time_ms