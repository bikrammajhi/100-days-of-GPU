#include "solve.h"
#include <cuda_runtime.h>

// CUDA kernel for performing matrix multiplication C = A * B       ⟶ Launches on GPU
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;      // Compute global column index
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;      // Compute global row index

    float sum = 0.0f;                                               // Accumulator for the dot product
    if(row < M && col < K){                                        // Bounds check to avoid out-of-range access
        for(unsigned int i = 0; i < N; i++){                        // Iterate over shared dimension
            sum += A[row * N + i] * B[i * K + col];                // Dot product: A[row][i] * B[i][col]
        }
        C[row * K + col] = sum;                                    // Store result in C[row][col]
    } 
}

// Host function to configure and launch the kernel
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);                                  // Define a 16x16 thread block
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Grid width covers K columns
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y); // Grid height covers M rows
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);  // Launch kernel
    cudaDeviceSynchronize();                                       // Ensure kernel execution completes before returning
}
