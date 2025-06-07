#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ __forceinline__ float warpReduceSum(thread_block_tile<32> warp, float val) {
    // Use cooperative groups for more explicit and portable warp reduction
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    // Get thread block group
    thread_block block = this_thread_block();
    
    // Get warp-level group
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    // Warp-level reduction
    val = warpReduceSum(warp, val);
    
    // Shared memory for storing warp results
    __shared__ float warp_results[32]; // Max 32 warps per block (1024 threads)
    
    // First thread in each warp writes result to shared memory
    if (warp.thread_rank() == 0) {
        warp_results[warp.meta_group_rank()] = val;
    }
    
    block.sync();
    
    // Final reduction using first warp
    if (warp.meta_group_rank() == 0) {
        val = (warp.thread_rank() < block.group_dim().x / warpSize) ? 
              warp_results[warp.thread_rank()] : 0.0f;
        val = warpReduceSum(warp, val);
    }
    
    return val;
}

__global__ void cooperative_group_sum(float *input, float *output, int n) {
    thread_block block = this_thread_block();
    
    unsigned int tid = block.thread_rank();
    unsigned int i = block.group_index().x * block.group_dim().x + tid;
    
    // Load data with bounds checking
    float val = (i < n) ? input[i] : 0.0f;
    
    // Perform block-level reduction
    val = blockReduceSum(val);
    
    // Write result of this block to output
    if (tid == 0) {
        output[block.group_index().x] = val;
    }
}

// Alternative version using grid-level cooperative groups for single-pass reduction
__global__ void cooperative_grid_sum(float *input, float *output, int n) {
    grid_group grid = this_grid();
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    __shared__ float block_sum;
    
    unsigned int tid = block.thread_rank();
    unsigned int global_tid = grid.thread_rank();
    
    // Initialize shared memory
    if (tid == 0) {
        block_sum = 0.0f;
    }
    block.sync();
    
    float val = 0.0f;
    
    // Grid-stride loop for processing multiple elements per thread
    for (unsigned int i = global_tid; i < n; i += grid.size()) {
        val += input[i];
    }
    
    // Warp-level reduction
    val = warpReduceSum(warp, val);
    
    // Store warp result in shared memory using atomic add
    if (warp.thread_rank() == 0) {
        atomicAdd(&block_sum, val);
    }
    
    block.sync();
    
    // Grid-level reduction using atomics (only first thread of each block)
    if (tid == 0) {
        atomicAdd(output, block_sum);
    }
}

int main() {
    const int N = 1024 * 1024; // Larger dataset
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Initialize input data
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Method 1: Block-level cooperative groups
    std::cout << "=== Method 1: Block-level Cooperative Groups ===" << std::endl;
    
    // Reset output array
    cudaMemset(d_output, 0, blocks * sizeof(float));
    
    cooperative_group_sum<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Copy partial results and perform final reduction on host
    float *h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final_sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        final_sum += h_partial[i];
    }
    
    std::cout << "Sum (block-level) = " << final_sum << std::endl;
    
    // Method 2: Grid-level cooperative groups (single-pass)
    std::cout << "\n=== Method 2: Grid-level Cooperative Groups ===" << std::endl;
    
    float *d_grid_output;
    cudaMalloc(&d_grid_output, sizeof(float));
    cudaMemset(d_grid_output, 0, sizeof(float));
    
    // Check if device supports cooperative launch
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    void *kernelArgs[] = {(void*)&d_input, (void*)&d_grid_output, (void*)&N};
    
    if (deviceProp.cooperativeLaunch) {
        // Query maximum blocks for cooperative launch
        int maxBlocksPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, 
                                                      cooperative_grid_sum, 
                                                      threadsPerBlock, 0);
        int maxBlocks = maxBlocksPerSM * deviceProp.multiProcessorCount;
        int coopBlocks = min(blocks, maxBlocks);
        
        std::cout << "Using " << coopBlocks << " blocks (max: " << maxBlocks << ") for cooperative launch" << std::endl;
        
        // Launch cooperative kernel with limited blocks
        cudaError_t result = cudaLaunchCooperativeKernel((void*)cooperative_grid_sum, 
                                   dim3(coopBlocks), dim3(threadsPerBlock), 
                                   kernelArgs, 0, 0);
        
        if (result != cudaSuccess) {
            std::cout << "Cooperative kernel launch failed: " << cudaGetErrorString(result) << std::endl;
            // Fallback to regular kernel
            cooperative_grid_sum<<<blocks, threadsPerBlock>>>(d_input, d_grid_output, N);
        }
    } else {
        std::cout << "Device does not support cooperative launch, using regular launch..." << std::endl;
        cooperative_grid_sum<<<blocks, threadsPerBlock>>>(d_input, d_grid_output, N);
    }
    cudaDeviceSynchronize();
    
    float grid_sum;
    cudaMemcpy(&grid_sum, d_grid_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Sum (grid-level) = " << grid_sum << std::endl;
    
    // Performance timing example
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Time block-level method
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        cooperative_group_sum<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float block_time;
    cudaEventElapsedTime(&block_time, start, stop);
    
    // Time grid-level method
    cudaMemset(d_grid_output, 0, sizeof(float));
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        if (deviceProp.cooperativeLaunch) {
            cudaLaunchCooperativeKernel((void*)cooperative_grid_sum, 
                                       dim3(blocks), dim3(threadsPerBlock), 
                                       kernelArgs, 0, 0);
        } else {
            cooperative_grid_sum<<<blocks, threadsPerBlock>>>(d_input, d_grid_output, N);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float grid_time;
    cudaEventElapsedTime(&grid_time, start, stop);
    
    std::cout << "Block-level method: " << block_time / 100.0f << " ms/iteration" << std::endl;
    std::cout << "Grid-level method:  " << grid_time / 100.0f << " ms/iteration" << std::endl;
    
    // Cleanup
    delete[] h_input;
    delete[] h_partial;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grid_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}