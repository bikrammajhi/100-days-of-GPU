#include <cuda_runtime.h>
#include <cooperative_groups.h>

template<int TILE_SIZE>
__global__ void matrix_multiply_coop(float* A, float* B, float* C, int M, int N, int K) {
    // 🏗️ Shared memory tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // 🤝 Create cooperative groups
    auto block = cooperative_groups::this_thread_block();
    
    // 📍 Thread coordinates in output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Row in C (0 to M-1)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Col in C (0 to N-1)
    
    float sum = 0.0f;
    
    // 🔄 Iterate through tiles along the K dimension
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 🚚 Collaborative loading using warps
        // For matrix A: row stays same, col moves along K dimension
        int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        
        // For matrix B: row moves along K dimension, col stays same
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;
        
        // 🔍 Bounds checking and loading
        bool valid_a = (a_row < M && a_col < K);
        bool valid_b = (b_row < K && b_col < N);
        
        // Load tiles with bounds checking
        tile_A[threadIdx.y][threadIdx.x] = valid_a ? A[a_row * K + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = valid_b ? B[b_row * N + b_col] : 0.0f;
        
        // 🛑 Block-level synchronization
        block.sync();
        
        // ⚙️ Compute partial result
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        // 🛑 Sync before loading next tile
        block.sync();
    }
    
    // 📤 Write final result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    // Define tile size (must match template parameter)
    const int TILE_SIZE = 16;  // Common choice for good occupancy
    
    // Calculate grid dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(
        (n + TILE_SIZE - 1) / TILE_SIZE,  // Number of blocks along N (columns of C)
        (m + TILE_SIZE - 1) / TILE_SIZE   // Number of blocks along M (rows of C)
    );
    
    // Launch kernel with cooperative groups support
    matrix_multiply_coop<TILE_SIZE><<<gridSize, blockSize>>>(
        const_cast<float*>(input_a),  // A is M×K
        const_cast<float*>(input_b),  // B is K×N  
        output_c,                     // C is M×N
        static_cast<int>(m),          // M: rows of A, rows of C
        static_cast<int>(n),          // N: cols of B, cols of C
        static_cast<int>(k)           // K: cols of A, rows of B
    );
    
    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}