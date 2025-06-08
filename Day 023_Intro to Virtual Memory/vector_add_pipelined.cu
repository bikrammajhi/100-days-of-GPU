#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/barrier>

template<int block_dim, int num_stages>
__global__ void vector_add_pipelined(float* a, float* b, float* c, int n) {
    // 📦 Multi-stage buffers
    __shared__ float shared_a[num_stages][block_dim];
    __shared__ float shared_b[num_stages][block_dim];
    
    auto pipeline = cuda::make_pipeline();
    const int stride = gridDim.x * block_dim;
    const int offset = blockIdx.x * block_dim;
    int stage = 0;
    
    // 🚀 Phase 1: Fill pipeline
    for (int s = 0; s < num_stages; ++s) {
        pipeline.producer_acquire();
        
        int global_start = offset + s * stride;
        if (global_start < n) {
            int copy_size = min(block_dim, n - global_start);
            cuda::memcpy_async(&shared_a[s][0], &a[global_start], 
                              sizeof(float) * copy_size, pipeline);
            cuda::memcpy_async(&shared_b[s][0], &b[global_start], 
                              sizeof(float) * copy_size, pipeline);
        }
        
        pipeline.producer_commit();
    }
    
    // ⚡ Phase 2: Steady-state processing
    for (int block_start = offset; block_start < n; block_start += stride) {
        // ⏳ Wait for current stage to be ready
        cuda::pipeline_consumer_wait_prior<num_stages - 1>(pipeline);
        __syncthreads();
        
        // ⚙️ Compute on current stage
        float result = shared_a[stage][threadIdx.x] + shared_b[stage][threadIdx.x];
        
        // 📤 Write result
        int global_idx = block_start + threadIdx.x;
        if (global_idx < n) {
            c[global_idx] = result;
        }
        
        __syncthreads();
        
        // 🔄 Pipeline management
        pipeline.consumer_release();
        pipeline.producer_acquire();
        
        // 🔮 Prefetch next data
        int next_start = block_start + num_stages * stride;
        if (next_start < n) {
            int copy_size = min(block_dim, n - next_start);
            cuda::memcpy_async(&shared_a[stage][0], &a[next_start], 
                              sizeof(float) * copy_size, pipeline);
            cuda::memcpy_async(&shared_b[stage][0], &b[next_start], 
                              sizeof(float) * copy_size, pipeline);
        }
        
        pipeline.producer_commit();
        stage = (stage + 1) % num_stages;
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    // Define template parameters
    constexpr int BLOCK_DIM = 256;
    constexpr int NUM_STAGES = 4;
    
    // Calculate grid dimensions
    const int num_blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    const int grid_dim = min(num_blocks, 65535); // Max grid dimension limit
    
    // Launch kernel with template parameters
    vector_add_pipelined<BLOCK_DIM, NUM_STAGES><<<grid_dim, BLOCK_DIM>>>(
        const_cast<float*>(d_input1), 
        const_cast<float*>(d_input2), 
        d_output, 
        static_cast<int>(n)
    );
    
    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately for your use case
        return;
    }
}