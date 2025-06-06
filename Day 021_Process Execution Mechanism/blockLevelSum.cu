#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ __forceinline__ float warpReduceSum(thread_block_tile<32> warp, float val) {
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    val = warpReduceSum(warp, val);

    __shared__ float warp_results[32];
    if (warp.thread_rank() == 0) {
        warp_results[warp.meta_group_rank()] = val;
    }

    block.sync();

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

    float val = (i < n) ? input[i] : 0.0f;
    val = blockReduceSum(val);

    if (tid == 0) {
        output[block.group_index().x] = val;
    }
}

int main() {
    const int N = 1024 * 1024;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "=== Block-level Cooperative Group Reduction ===" << std::endl;

    cudaMemset(d_output, 0, blocks * sizeof(float));
    cooperative_group_sum<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    float *h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float final_sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        final_sum += h_partial[i];
    }

    std::cout << "Sum = " << final_sum << std::endl;

    delete[] h_input;
    delete[] h_partial;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
