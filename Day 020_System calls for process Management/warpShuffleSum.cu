#include <iostream>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    // Perform warp-level reduction using shuffle down
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_shuffle_sum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    // Load data
    float val = (i < n) ? input[i] : 0.0f;
    
    // Warp-level reduction
    val = warpReduceSum(val);
    
    // Write warp results to shared memory
    if ((tid & (warpSize - 1)) == 0) {
        sdata[tid / warpSize] = val;
    }
    
    __syncthreads();
    
    // Final reduction of warp results
    if (tid < (blockDim.x / warpSize)) {
        val = (tid < (blockDim.x / warpSize)) ? sdata[tid] : 0.0f;
        val = warpReduceSum(val);
    }
    
    // Write result of this block to output
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Shared memory size reduced - only need space for warp results
    int sharedMemSize = (threadsPerBlock / 32) * sizeof(float);
    warp_shuffle_sum<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    
    float *h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final_sum = 0.0f;
    for (int i = 0; i < blocks; ++i) final_sum += h_partial[i];
    
    std::cout << "Sum = " << final_sum << std::endl;
    
    delete[] h_input;
    delete[] h_partial;
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
