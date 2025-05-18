#include <stdio.h>  // Required for printf and standard I/O

// ==============================
// CPU Vector Addition Function
// ==============================

// Performs element-wise addition of vectors x and y into vector z on the CPU
void vecadd_cpu(float *x, float *y, float *z, unsigned int N){
    for (int i = 0; i < N; ++i){
        z[i] = x[i] + y[i];  // Simple element-wise addition
    }
}

// ==============================
// CUDA Kernel for Vector Addition
// ==============================

// This function runs on the GPU and is executed in parallel by multiple threads
__global__ void vecadd_kernel(float *x, float *y, float *z, unsigned int N){
    // Each thread calculates its global index (position in the array)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread does not access out-of-bounds memory
    if (i < N){
        z[i] = x[i] + y[i];  // Perform the vector addition for this thread's index
    }
}

// ==============================
// GPU Wrapper Function for Vector Addition
// ==============================

// Manages GPU memory allocation, data transfer, kernel execution, and cleanup
void vecadd_gpu(float *x, float *y, float *z, unsigned int N){
    // 1. Allocate memory on GPU for input and output arrays
    float *x_d, *y_d, *z_d;  // Pointers for device memory
    cudaMalloc((void**)&x_d, N*sizeof(float));  // Allocate memory for x
    cudaMalloc((void**)&y_d, N*sizeof(float));  // Allocate memory for y
    cudaMalloc((void**)&z_d, N*sizeof(float));  // Allocate memory for z

    // 2. Copy input data from host (CPU) to device (GPU)
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);  // Copy x
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);  // Copy y

    // 3. Define GPU kernel launch configuration
    const unsigned int numThreadsPerBlock = 256;  // 256 threads per block (typical size)
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    // This ensures all elements are covered, even if N is not divisible by 256

    // 4. Launch the kernel on the GPU
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);

    // 5. Copy result from device (GPU) to host (CPU)
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

// ==============================
// Main Program
// ==============================

int main(int argc, char** argv){
    cudaDeviceSynchronize();  // Ensure the GPU is initialized and ready

    // --------------------------
    // Memory Allocation and Initialization
    // --------------------------
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);  // Default N = 2^25 (~33 million floats)
    float *x = (float*) malloc(N*sizeof(float));  // Allocate memory for input vector x
    float *y = (float*) malloc(N*sizeof(float));  // Allocate memory for input vector y
    float *z = (float*) malloc(N*sizeof(float));  // Allocate memory for output vector z

    // Initialize input vectors with random values
    for (unsigned int i = 0; i < N; ++i){
        x[i] = rand();  // Random float (note: `rand()` returns int; use float cast if needed)
        y[i] = rand();
    }

    // --------------------------
    // CUDA Event Timing Setup
    // --------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  // Create CUDA event for timing start
    cudaEventCreate(&stop);   // Create CUDA event for timing stop

    // ======================
    // Vector Addition on CPU
    // ======================
    cudaEventRecord(start);             // Start timing
    vecadd_cpu(x, y, z, N);             // Perform addition on CPU
    cudaEventRecord(stop);              // Stop timing
    cudaEventSynchronize(stop);         // Wait for the event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  // Get elapsed time
    printf("CPU time: %f ms\n", milliseconds);

    // ======================
    // Vector Addition on GPU
    // ======================
    cudaEventRecord(start);             // Start timing
    vecadd_gpu(x, y, z, N);             // Perform addition on GPU
    cudaEventRecord(stop);              // Stop timing
    cudaEventSynchronize(stop);         // Wait for the event to complete
    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);  // Get elapsed time
    printf("GPU time: %f ms\n", msec);

    // --------------------------
    // Cleanup
    // --------------------------
    free(x);  // Free host memory for x
    free(y);  // Free host memory for y
    free(z);  // Free host memory for z

    return 0;  // Exit the program
}
