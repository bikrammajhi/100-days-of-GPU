/* 
This code is adapted from: https://github.com/HMUNACHI/cuda-tutorials/blob/main/kernels/03.vectorDot.cu
*/

// Include the necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 
Dot Product of Two Vectors: 
--------------------------
Given two vectors 'a' and 'b' of the same dimension 'n', their dot product is defined as: 

a · b = a1 * b1 + a2 * b2 + ... + an * bn

Geometrically, the dot product measures the cosine of the angle between the vectors scaled by their magnitudes.  
It's useful for tasks like determining if vectors are orthogonal (dot product is zero), projecting one vector onto another, calculating work done by a force.
Dot products are commutative, distributive, and linear. 


CUDA Tweaks:
--------------------
In CUDA, we can exploit parallelism to accelerate the calculation of dot products, especially for large vectors. 
However, naive parallelization can lead to race conditions when multiple threads attempt to update a shared sum simultaneously. 
So we at such structure our code in the following way:

1. Thread Assignment: Each thread is assigned to compute the product of one pair of elements from the vectors (e.g., thread 1 computes a1 * b1).
2. Partial Products: Each thread stores its partial product in a local variable.
3. Atomic Addition: Instead of directly adding their partial products to a shared sum, each thread uses atomicAdd() to perform an atomic update. This guarantees that only one thread updates the shared sum at a time, avoiding race conditions. 
4. Synchronization (Optional): A synchronization barrier (e.g., __syncthreads()) may be needed to ensure all threads have completed their computations before the final sum is used.

NOTE:
- z_d must be zero-initialized to avoid garbage accumulation.
*/

__global__ void vectorDot_kernel(const float* x_d, const float* y_d, float* z_d, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < N) {
        atomicAdd(z_d, x_d[index] * y_d[index]);  // thread-safe accumulation
    }
}

// ======================================
// Function to generate random floats
// ======================================
float randomFloat(int randMax = 1000) {
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

// ======================================
// Main function
// ======================================
int main() {
    // -----------------------
    // Problem size
    // -----------------------
    int N = 1000;
    size_t size = N * sizeof(float);

    // -----------------------
    // Host memory allocation
    // -----------------------
    float *x, *y, *z;
    x = (float*)malloc(size);        // Input vector x
    y = (float*)malloc(size);        // Input vector y
    z = (float*)malloc(sizeof(float)); // Output scalar dot product

    // -----------------------
    // Initialize input vectors
    // -----------------------
    for (int i = 0; i < N; ++i) {
        x[i] = randomFloat();        // Random float for x[i]
        y[i] = randomFloat();        // Random float for y[i]
    }
    z[0] = 0.0f;                     // Initialize result to 0

    // -----------------------
    // Device memory allocation
    // -----------------------
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, size);               // Device vector x
    cudaMalloc((void**)&y_d, size);               // Device vector y
    cudaMalloc((void**)&z_d, sizeof(float));      // Device scalar result

    // -----------------------
    // Copy data to device
    // -----------------------
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(z_d, z, sizeof(float), cudaMemcpyHostToDevice); // copy z = 0

    // -----------------------
    // Launch CUDA kernel
    // -----------------------
    int ThreadsPerBlock = 32;
    int blocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    vectorDot_kernel<<<blocksPerGrid, ThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();

    // -----------------------
    // Copy result back to host
    // -----------------------
    cudaMemcpy(z, z_d, sizeof(float), cudaMemcpyDeviceToHost);

    // -----------------------
    // Output the result
    // -----------------------
    std::cout << "Dot Product: " << z[0] << std::endl;

    // -----------------------
    // Free device memory
    // -----------------------
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    // -----------------------
    // Free host memory
    // -----------------------
    free(x);
    free(y);
    free(z);

    return 0;
}
