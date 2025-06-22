#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __forceinline__ void tileMemcpy(
    float *src,
    float *dst, 
    const unsigned int src_stride,
    const unsigned int tile_rows, 
    const unsigned int tile_cols
){
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;


    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;

    for(unsigned int r = thread_row; r < tile_rows; r += row_step){
        dst[r * tile_cols + thread_col] = src[r * src_stride + thread_col];
    }

}

// GEMM Kernel: C = alpha * A * B + beta * C
template<
    unsigned int BM_dim, 
    unsigned int BN_dim, 
    unsigned int BK_dim, 
    unsigned int NUM_THREADS
>
__global__ void
sgemm_shared_kernel(
    float* A,
    float* B, 
    float* C, 
    const float alpha, 
    const float beta,
    const unsigned int M, 
    const unsigned int N, 
    const unsigned int K  
){
    // leading dimensions
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int C_stride = N;

    // block index
    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;

    // thread index
    const unsigned int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int thread_row = thread_id / BN_dim;
    const unsigned int thread_col = thread_id % BN_dim;

    // Number of tiles along K dimensions
    const unsigned int num_blocks_tiles_k = K / BK_dim;

    // Shared memory allocation
    extern __shared__ float shmem[];
    float* A_block_smem = shmem;
    float* B_block_smem = &shmem[BM_dim * BK_dim];

    // Output tile pointer
    float* C_tile = &C[block_m * BM_dim * C_stride + block_n * BN_dim];

    float acc_register = 0.0f;

    for(unsigned int block_k = 0; block_k < num_blocks_tiles_k; ++block_k){
        // Global memory tiles
        float* A_block_gemm = &A[block_m * BM_dim * A_stride + block_k * BK_dim];
        float* B_block_gemm = &B[block_k * BK_dim * B_stride + block_n * BN_dim];

        // Global to Shared memory copy
        tileMemcpy(A_block_gemm, A_block_smem, A_stride, BM_dim, BK_dim);
        tileMemcpy(B_block_gemm, A_block_smem, B_stride, BK_dim, BN_dim);

        __syncthreads();

        // Inner product accumulation
        for(unsigned int k_inner=0; k_inner < BK_dim; ++k_inner){
            float a = A_block_smem[thread_row * BK_dim + k_inner];
            float b = B_block_smem[thread_row * k_inner + thread_col];
            acc_register += a * b;
        }

        __syncthreads();
    }

    C_tile[thread_row * C_stride + thread_col] = 
    alpha * acc_register + beta * C_tile[thread_row * C_stride + thread_col];


}