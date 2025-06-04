#include <iostream>
#include <cuda_runtime.h>

__global__ void simple_interleaved_sum(float *input, float *output, int n) {
    extern __shared__ float partialSum[];

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + t;

    // Load data into shared memory
    partialSum[t] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    /*
    // ITERATION 0:Interleaved addressing reduction 
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0 && (t + stride) < blockDim.x) {
            partialSum[t] += partialSum[t + stride];
        }
    }
    */

    // ITERATION 1: Replace modulo with bitwise AND
    // Previous: Expensive modulo operation
    // Improvement: Use bitwise operations for power-of-2 checks
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if ((t & (2 * stride - 1)) == 0 && (t + stride) < blockDim.x) {
            partialSum[t] += partialSum[t + stride];
        }
    }

    // Write result of this block to output
    if (t == 0) {
        output[blockIdx.x] = partialSum[0];
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

    simple_interleaved_sum<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);

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