#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// Warp-level GEMM kernel
__global__ void warp_gemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    // Calculate which tile this warp processes
    int warps_per_row = (N + TILE_N - 1) / TILE_N;
    int warp_row = warp_id / warps_per_row;
    int warp_col = warp_id % warps_per_row;
    
    if (warp_row * TILE_M >= M || warp_col * TILE_N >= N) return;
    
    float accum[TILE_M][TILE_N] = {0};
    
    int thread_row = lane_id / 4; // 0–7
    int thread_col = lane_id % 4; // 0–3
    
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        float a_frag[TILE_M];
        float b_frag[TILE_N];
        
        // Load A - each thread loads different elements
        for (int i = 0; i < TILE_M; i++) {
            int row = warp_row * TILE_M + i;
            int col = tile_k + thread_col;
            a_frag[i] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        
        // Load B - each thread loads different elements
        for (int j = 0; j < TILE_N; j++) {
            int row = tile_k + thread_row;
            int col = warp_col * TILE_N + j;
            b_frag[j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        
        // Synchronize warp before computation
        warp.sync();
        
        // Compute using warp shuffles for better data sharing (T4 compatible)
        for (int k = 0; k < TILE_K; k++) {
            // Simplified computation - each thread works on its own data
            // Use shuffle to share A values across threads
            for (int i = 0; i < TILE_M; i++) {
                // Broadcast A value from thread that loaded it
                int a_source_lane = (k % 4); // thread that has this k value
                float a_val = warp.shfl(a_frag[i], a_source_lane);
                
                for (int j = 0; j < TILE_N; j++) {
                    // Broadcast B value from thread that loaded it  
                    int b_source_lane = (k / 4) * 4; // thread group that has this k value
                    float b_val = warp.shfl(b_frag[j], b_source_lane);
                    
                    accum[i][j] += a_val * b_val;
                }
            }
        }
        
        // Synchronize after computation
        warp.sync();
    }
    
    // Store C with manual warp-level reduction for T4 compatibility
    for (int i = 0; i < TILE_M; i++) {
        int row = warp_row * TILE_M + i;
        for (int j = 0; j < TILE_N; j++) {
            int col = warp_col * TILE_N + j;
            if (row < M && col < N) {
                // Manual warp reduction using shuffle
                float partial_sum = accum[i][j];
                
                // Reduce across warp using shuffle down
                for (int offset = 16; offset > 0; offset /= 2) {
                    partial_sum += warp.shfl_down(partial_sum, offset);
                }
                
                // Only the first thread in warp writes the result
                if (lane_id == 0) {
                    C[row * N + col] = partial_sum;
                }
            }
        }
    }
}

// Host code to run GEMM
int main() {
    const int M = 64, N = 64, K = 64;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    // Launch warp-GEMM
    dim3 blockDim(128);
    int warps_per_block = blockDim.x / 32;
    int total_warps = ((M + TILE_M - 1) / TILE_M) * ((N + TILE_N - 1) / TILE_N);
    dim3 gridDim((total_warps + warps_per_block - 1) / warps_per_block);
    
    warp_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    
    // Print a small portion of output
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[M-1][N-1] = %f\n", h_C[(M - 1) * N + (N - 1)]);
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
