#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Optimized version using warp-level tiling
template<int TILE_SIZE>
__global__ void matrix_multiply_warp_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    auto block = cg::this_thread_block();
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int warp_id = (threadIdx.y * TILE_SIZE + threadIdx.x) / 32;
    int lane_id = (threadIdx.y * TILE_SIZE + threadIdx.x) % 32;
    
    float sum = 0.0f;
    
    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        
        // 🚀 Warp-cooperative loading with better memory access patterns
        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
        
        // Load tile_A - coalesced access
        if (tid < TILE_SIZE * TILE_SIZE) {
            int load_row = tid / TILE_SIZE;
            int load_col = tid % TILE_SIZE;
            int a_row = blockIdx.y * TILE_SIZE + load_row;
            int a_col = tile_idx * TILE_SIZE + load_col;
            
            tile_A[load_row][load_col] = (a_row < M && a_col < K) ? 
                A[a_row * K + a_col] : 0.0f;
        }
        
        // Load tile_B - coalesced access  
        if (tid < TILE_SIZE * TILE_SIZE) {
            int load_row = tid / TILE_SIZE;
            int load_col = tid % TILE_SIZE;
            int b_row = tile_idx * TILE_SIZE + load_row;
            int b_col = blockIdx.x * TILE_SIZE + load_col;
            
            tile_B[load_row][load_col] = (b_row < K && b_col < N) ? 
                B[b_row * N + b_col] : 0.0f;
        }
        
        __syncthreads();
        
        // ⚡ Compute with warp-level optimizations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Use register blocking for better ILP
            float a_reg = tile_A[threadIdx.y][k];
            float b_reg = tile_B[k][threadIdx.x];
            
            // Warp shuffle for additional data sharing (optional)
            float shared_a = __shfl_sync(0xFFFFFFFF, a_reg, lane_id);
            sum += a_reg * b_reg;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, 
                        size_t m, size_t n, size_t k) {
    const int TILE_SIZE = 16;
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (m + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Use the warp-cooperative tiled version
    matrix_multiply_warp_tiled<TILE_SIZE><<<gridSize, blockSize>>>(
        const_cast<float*>(input_a),
        const_cast<float*>(input_b),
        output_c,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k)
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
