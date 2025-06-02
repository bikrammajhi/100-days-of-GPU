// Filename: conv1d.cu

#include <cuda_runtime.h>
#include <iostream>

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
constexpr int kInputSize = 16;
constexpr int kKernelSize = 3;  // Must be odd for center alignment

// ----------------------------------------------------------------------------
// CUDA 1D Convolution Kernel
// ----------------------------------------------------------------------------
__global__ void Convolve1D(const float* __restrict__ input,
                           const float* __restrict__ kernel,
                           float* output,
                           int input_size,
                           int kernel_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = kernel_size / 2;

    if (tid < input_size) {
        float sum = 0.0f;

        // Apply kernel
        for (int k = -radius; k <= radius; ++k) {
            int index = tid + k;
            if (index >= 0 && index < input_size) {
                sum += input[index] * kernel[k + radius];
            }
        }

        output[tid] = sum;
    }
}

// ----------------------------------------------------------------------------
// Host Utility
// ----------------------------------------------------------------------------
void CheckCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(result) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    // Host memory allocation
    float h_input[kInputSize], h_kernel[kKernelSize], h_output[kInputSize];

    // Initialize host input and kernel
    for (int i = 0; i < kInputSize; ++i) {
        h_input[i] = static_cast<float>(i + 1);  // 1, 2, ..., 16
    }

    h_kernel[0] = 0.2f;
    h_kernel[1] = 0.5f;
    h_kernel[2] = 0.3f;

    // Device memory allocation
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    CheckCudaError(cudaMalloc((void**)&d_input, kInputSize * sizeof(float)), "cudaMalloc d_input");
    CheckCudaError(cudaMalloc((void**)&d_kernel, kKernelSize * sizeof(float)), "cudaMalloc d_kernel");
    CheckCudaError(cudaMalloc((void**)&d_output, kInputSize * sizeof(float)), "cudaMalloc d_output");

    // Copy data from host to device
    CheckCudaError(cudaMemcpy(d_input, h_input, kInputSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_input");
    CheckCudaError(cudaMemcpy(d_kernel, h_kernel, kKernelSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_kernel");

    // Launch kernel
    const int threads_per_block = 128;
    const int num_blocks = (kInputSize + threads_per_block - 1) / threads_per_block;

    Convolve1D<<<num_blocks, threads_per_block>>>(d_input, d_kernel, d_output, kInputSize, kKernelSize);
    CheckCudaError(cudaGetLastError(), "Kernel launch");

    // Copy results back to host
    CheckCudaError(cudaMemcpy(h_output, d_output, kInputSize * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_output");

    // Print result
    std::cout << "Convolution Output:\n";
    for (int i = 0; i < kInputSize; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
