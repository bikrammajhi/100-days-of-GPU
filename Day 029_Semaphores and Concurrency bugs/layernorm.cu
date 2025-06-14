#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

// Constants 
constexpr float kEpsilon = 1e-6f;
constexpr int kMaxThreadsPerBlock = 1024;

/**
 * @brief Naive LayerNorm kernel - processes one row per thread
 * @param input_tensor Input tensor [batch_size x hidden_dim]
 * @param output_tensor Output normalized tensor [batch_size x hidden_dim]  
 * @param batch_size Number of rows (batch dimension)
 * @param hidden_dim Number of columns (feature dimension)
 */
__global__ void LayerNormKernel(
    const float* __restrict__ input_tensor,
    float* __restrict__ output_tensor,
    const int batch_size,
    const int hidden_dim
) {
    // Calculate global thread index
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row_idx >= batch_size) return;
    
    // Calculate row offset
    const int row_offset = row_idx * hidden_dim;
    
    // Step 1: Calculate mean
    float row_mean = 0.0f;
    for (int col_idx = 0; col_idx < hidden_dim; ++col_idx) {
        row_mean += input_tensor[row_offset + col_idx];
    }
    row_mean /= static_cast<float>(hidden_dim);
    
    // Step 2: Calculate variance
    float row_variance = 0.0f;
    for (int col_idx = 0; col_idx < hidden_dim; ++col_idx) {
        const float diff = input_tensor[row_offset + col_idx] - row_mean;
        row_variance += diff * diff;
    }
    row_variance /= static_cast<float>(hidden_dim);
    
    // Step 3: Calculate inverse standard deviation
    const float inv_std = rsqrtf(row_variance + kEpsilon);
    
    // Step 4: Normalize and write output
    for (int col_idx = 0; col_idx < hidden_dim; ++col_idx) {
        const float normalized_val = (input_tensor[row_offset + col_idx] - row_mean) * inv_std;
        output_tensor[row_offset + col_idx] = normalized_val;
    }
}

/**
 * @brief Host function to launch LayerNorm kernel
 */
void LaunchLayerNorm(
    const float* d_input,
    float* d_output,
    const int batch_size,
    const int hidden_dim,
    cudaStream_t stream = 0
) {
    // Calculate grid and block dimensions
    const dim3 threads_per_block(kMaxThreadsPerBlock);
    const dim3 blocks_per_grid((batch_size + threads_per_block.x - 1) / threads_per_block.x);
    
    // Launch kernel
    LayerNormKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_input, d_output, batch_size, hidden_dim
    );
}

/**
 * @brief Initialize input tensor with test data
 */
void InitializeInputTensor(float* host_input, const int batch_size, const int hidden_dim) {
    const int total_elements = batch_size * hidden_dim;
    for (int i = 0; i < total_elements; ++i) {
        host_input[i] = static_cast<float>(i % 100);
    }
}

/**
 * @brief Main function demonstrating LayerNorm usage
 */
int main() {
    // Problem dimensions
    const int batch_size = 1024;      // Number of sequences/rows
    const int hidden_dim = 512;       // Feature dimension
    const size_t tensor_size_bytes = batch_size * hidden_dim * sizeof(float);
    
    std::cout << "LayerNorm Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Hidden dim: " << hidden_dim << "\n";
    std::cout << "  Tensor size: " << tensor_size_bytes / (1024 * 1024) << " MB\n\n";
    
    // Allocate host memory
    float* h_input = static_cast<float*>(malloc(tensor_size_bytes));
    float* h_output = static_cast<float*>(malloc(tensor_size_bytes));
    
    if (!h_input || !h_output) {
        std::cerr << "Failed to allocate host memory\n";
        return -1;
    }
    
    // Initialize input data
    InitializeInputTensor(h_input, batch_size, hidden_dim);
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaError_t cuda_status = cudaMalloc(&d_input, tensor_size_bytes);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_input: " << cudaGetErrorString(cuda_status) << "\n";
        free(h_input); free(h_output);
        return -1;
    }
    
    cuda_status = cudaMalloc(&d_output, tensor_size_bytes);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_output: " << cudaGetErrorString(cuda_status) << "\n";
        cudaFree(d_input); free(h_input); free(h_output);
        return -1;
    }
    
    // Copy input data to device
    cuda_status = cudaMemcpy(d_input, h_input, tensor_size_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(cuda_status) << "\n";
        cudaFree(d_input); cudaFree(d_output); free(h_input); free(h_output);
        return -1;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // Record start time and launch kernel
    cudaEventRecord(start_event);
    LaunchLayerNorm(d_input, d_output, batch_size, hidden_dim);
    cudaEventRecord(stop_event);
    
    // Wait for kernel completion
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(cuda_status) << "\n";
        cudaFree(d_input); cudaFree(d_output); free(h_input); free(h_output);
        return -1;
    }
    
    // Calculate execution time
    cudaEventSynchronize(stop_event);
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    
    std::cout << "Performance Results:\n";
    std::cout << "  Kernel execution time: " << kernel_time_ms << " ms\n";
    
    // Calculate throughput
    const float throughput_gb_s = (tensor_size_bytes * 2.0f) / (kernel_time_ms * 1e-3f) / (1024.0f * 1024.0f * 1024.0f);
    std::cout << "  Memory throughput: " << throughput_gb_s << " GB/s\n\n";
    
    // Copy result back to host
    cuda_status = cudaMemcpy(h_output, d_output, tensor_size_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(cuda_status) << "\n";
    } else {
        std::cout << "LayerNorm completed successfully!\n";
        
        // Print sample results
        std::cout << "Sample output (first 5 elements): ";
        for (int i = 0; i < 5; ++i) {
            std::cout << h_output[i] << " ";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return 0;
}