#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel that performs vector addition in chunks of 8 floats per thread
// This is a high-performance implementation that uses vectorized memory access (float4)
__global__ void vectorAddKernel(const float* __restrict__ d_input1, 
                                const float* __restrict__ d_input2, 
                                float* __restrict__ d_output, 
                                size_t n) {
    
    // Compute global thread ID using thread and block indices
    uint32_t tid = threadIdx.x;
    uint32_t ctaid = blockIdx.x;
    uint32_t ntid = blockDim.x;
    uint32_t global_tid = ctaid * ntid + tid;

    // Each thread is responsible for 8 float elements
    // Multiply global thread ID by 8 to get the base index of elements this thread will process
    uint64_t base_idx = static_cast<uint64_t>(global_tid) << 3;

    // Check if this thread's 8-element range goes beyond the array size
    uint64_t check_addr = base_idx + 7;
    bool out_of_bounds = (check_addr >= n);

    if (!out_of_bounds) {
        // Fast vectorized path using float4 (4 floats per load/store)
        // Load 8 elements from each input array, process them, and store 8 results

        // Compute base addresses
        const float* addr1 = d_input1 + base_idx;
        const float* addr2 = d_input2 + base_idx;
        float* addr_out = d_output + base_idx;

        // Load first 4 elements (vectorized as float4)
        float4 vec1_first = *reinterpret_cast<const float4*>(addr1);
        float4 vec2_first = *reinterpret_cast<const float4*>(addr2);

        // Load next 4 elements (offset by +4)
        float4 vec1_second = *reinterpret_cast<const float4*>(addr1 + 4);
        float4 vec2_second = *reinterpret_cast<const float4*>(addr2 + 4);

        // Perform element-wise addition for all 8 values
        float s1 = vec1_first.x + vec2_first.x;
        float s2 = vec1_first.y + vec2_first.y;
        float s3 = vec1_first.z + vec2_first.z;
        float s4 = vec1_first.w + vec2_first.w;
        float s5 = vec1_second.x + vec2_second.x;
        float s6 = vec1_second.y + vec2_second.y;
        float s7 = vec1_second.z + vec2_second.z;
        float s8 = vec1_second.w + vec2_second.w;

        // Store the first 4 results
        float4 result_first = make_float4(s1, s2, s3, s4);
        *reinterpret_cast<float4*>(addr_out) = result_first;

        // Store the next 4 results
        float4 result_second = make_float4(s5, s6, s7, s8);
        *reinterpret_cast<float4*>(addr_out + 4) = result_second;

        // Done with vectorized path
        return;
    }

    // Fallback for handling remaining tail elements (n not divisible by 8)
    // Process elements one at a time if we can't load a full chunk

    // If base index is already out of bounds, nothing to do
    if (base_idx >= n) {
        return;
    }

    // Compute scalar pointers for the fallback path
    const float* addr1 = d_input1 + base_idx;
    const float* addr2 = d_input2 + base_idx;
    float* addr_out = d_output + base_idx;

    // Perform element-wise addition for this remaining element
    float s1 = *addr1;
    float r1 = *addr2;
    float result = s1 + r1;
    *addr_out = result;
}

// Host function that launches the CUDA kernel
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    // Each thread processes 8 elements, so we need ceil(n / 8) threads
    int totalThreadsNeeded = (n + 7) / 8;

    // Use 512 threads per block, which generally gives good occupancy on most GPUs
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);
}
