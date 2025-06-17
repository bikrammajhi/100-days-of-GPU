#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void wmma_tiled_gemm_kernel(half *A, half *B, float *C, int M, int N, int K) {
    // Calculate tile indices
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;

    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);

    // Compute matrix multiplication in tiles
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Calculate matrix indices
        int a_index = (tile_row * TILE_M) * K + tile_k;
        int b_index = tile_k * N + (tile_col * TILE_N);

        // Load matrix fragments
        wmma::load_matrix_sync(a_frag, A + a_index, K);
        wmma::load_matrix_sync(b_frag, B + b_index, N);

        // Perform matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int c_index = (tile_row * TILE_M) * N + (tile_col * TILE_N);
    wmma::store_matrix_sync(C + c_index, c_frag, N, wmma::mem_row_major);
}

int main() {
    // Check CUDA device capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 7) {
        std::cerr << "WMMA requires compute capability 7.0 or higher. Current: " 
                  << prop.major << "." << prop.minor << std::endl;
        return -1;
    }
    
    std::cout << "Using device: " << prop.name << " (Compute " 
              << prop.major << "." << prop.minor << ")" << std::endl;

    int M = 128, N = 128, K = 128;

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(32, 1);
    dim3 blocks(N / TILE_N, M / TILE_M);

    wmma_tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify result (for 1x1 matrices, result should be K=128)
    std::cout << "C[0] = " << h_C[0] << " (expected: " << K << ")" << std::endl;
    
    // Verify a few more elements
    std::cout << "C[1] = " << h_C[1] << std::endl;
    std::cout << "C[N] = " << h_C[N] << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}