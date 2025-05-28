#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * Naive CUDA Softmax Implementation
 * - Each thread processes one row of the M x N matrix
 * - Uses numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * - Memory layout: row-major order
 * - Parallelize over rows of matrix
 * - One thread for each row
 * - Assuming M 1024 rows, --> 1024 threds
 */

#define M 1024      // Number of rows
#define N 32768     // Number of columns (features per row)

__global__ void naiveSoftmax_kernel(float *X_d, float *O_d, int M, int N){
    // Calculate global thread ID (one thread per row)
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if(row < M){
        float x_max = -INFINITY;    // Track maximum value for numerical stability
        float norm = 0.0f;          // Normalization factor (sum of exponentials)
        int index;                  // Linear index for row-major access
        
        // Pass 1: Find maximum value in current row
        // This prevents overflow in exponential calculations
        for(int c = 0; c < N; ++c){
            index = row * N + c;
            x_max = fmaxf(X_d[index], x_max);
        }
        
        // Pass 2: Calculate sum of exp(x_i - x_max) for normalization
        for(int c = 0; c < N; ++c){
            index = row * N + c;
            norm += expf(X_d[index] - x_max);
        }
        
        // Pass 3: Compute final softmax values
        // softmax_i = exp(x_i - x_max) / norm
        for(int c = 0; c < N; ++c){
            index = row * N + c;
            O_d[index] = expf(X_d[index] - x_max) / norm;
        }
    }
}


int main(){
    size_t size = M * N * sizeof(float);

    float *X, *O;           // Host pointers
    float *X_d, *O_d;       // Device pointers

    // Allocate memory on host and device
    X = (float *)malloc(size);
    O = (float *)malloc(size);

    cudaMalloc((void **)&X_d, size); 
    cudaMalloc((void **)&O_d, size);  

    // Initialize input data with random values
    for(int i = 0; i < M * N; ++i){ 
        X[i] = static_cast<float>(rand()) / RAND_MAX;  
    }

    // Transfer data from host to device memory
    cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);  

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);     
    float ms = 0.f;
    cudaEventRecord(start);

    // Launch Kernel with correct grid/block configuration
    dim3 threadsPerBlock = 32;  
    dim3 blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;  

    naiveSoftmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(X_d, O_d, M, N);
    cudaDeviceSynchronize();

    // Record timing and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << "Kernel execution time: " << ms << "ms" << std::endl;  
    
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result data from GPU to CPU
    cudaMemcpy(O, O_d, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(X_d);
    cudaFree(O_d);

    // Free CPU memory
    free(X);  
    free(O);  

    return 0;
}