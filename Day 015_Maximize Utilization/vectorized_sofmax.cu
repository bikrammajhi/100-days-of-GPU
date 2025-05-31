#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

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

__global__ void sharedSoftmax_kernel(float *X_d, float *O_d, int M, int N) {
    // Calculate global thread ID (one thread per row)
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if(row < M) {
        // Allocate shared memory for caching input data
        extern __shared__ float s_data[];
        
        float x_max = -INFINITY;    // Track maximum value for numerical stability
        float norm = 0.0f;          // Normalization factor (sum of exponentials)
        int index;                  // Linear index for row-major access
        
        // Pass 1: Load data into shared memory and find maximum value
        for(int c = 0; c < N; c += blockDim.x) {
            // Load data into shared memory
            if(c + threadIdx.x < N) {
                index = row * N + c + threadIdx.x;
                s_data[threadIdx.x] = X_d[index];
            }
            __syncthreads();
            
            // Find maximum in shared memory
            for(int i = 0; i < blockDim.x && (c + i) < N; i++) {
                x_max = fmaxf(s_data[i], x_max);
            }
            __syncthreads();
        }
        
        // Pass 2: Calculate sum of exp(x_i - x_max) using shared memory
        for(int c = 0; c < N; c += blockDim.x) {
            // Load data into shared memory
            if(c + threadIdx.x < N) {
                index = row * N + c + threadIdx.x;
                s_data[threadIdx.x] = expf(X_d[index] - x_max);
            }
            __syncthreads();
            
            // Sum exponentials in shared memory
            for(int i = 0; i < blockDim.x && (c + i) < N; i++) {
                norm += s_data[i];
            }
            __syncthreads();
        }
        
        // Pass 3: Compute final softmax values using shared memory
        for(int c = 0; c < N; c += blockDim.x) {
            // Load data into shared memory
            if(c + threadIdx.x < N) {
                index = row * N + c + threadIdx.x;
                s_data[threadIdx.x] = expf(X_d[index] - x_max) / norm;
            }
            __syncthreads();
            
            // Write results back to global memory
            if(c + threadIdx.x < N) {
                index = row * N + c + threadIdx.x;
                O_d[index] = s_data[threadIdx.x];
            }
            __syncthreads();
        }
    }
}


// Enhanced version with vectorized memory access
__global__ void vectorizedSoftmax_kernel(float *X_d, float *O_d, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    if (row >= M) return;
    
    extern __shared__ float sdata[];
    
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // Use float4 for vectorized access when possible
    int vectorized_N = (N / 4) * 4;
    float4 *X_vec = reinterpret_cast<float4*>(X_d + row * N);
    float4 *O_vec = reinterpret_cast<float4*>(O_d + row * N);
    
    // Phase 1: Vectorized maximum finding
    for (int i = tid; i < vectorized_N / 4; i += blockSize) {
        float4 val = X_vec[i];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }
    
    // Handle remaining elements
    for (int i = vectorized_N + tid; i < N; i += blockSize) {
        thread_max = fmaxf(thread_max, X_d[row * N + i]);
    }
    
    // Parallel reduction for maximum (using shared memory)
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    
    float row_max = sdata[0];
    __syncthreads();
    
    // Phase 2: Vectorized sum calculation
    for (int i = tid; i < vectorized_N / 4; i += blockSize) {
        float4 val = X_vec[i];
        thread_sum += expf(val.x - row_max) + expf(val.y - row_max) + 
                     expf(val.z - row_max) + expf(val.w - row_max);
    }
    
    // Handle remaining elements
    for (int i = vectorized_N + tid; i < N; i += blockSize) {
        thread_sum += expf(X_d[row * N + i] - row_max);
    }
    
    // Parallel reduction for sum
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    float row_sum = sdata[0];
    __syncthreads();
    
    // Phase 3: Vectorized output computation
    for (int i = tid; i < vectorized_N / 4; i += blockSize) {
        float4 val = X_vec[i];
        float4 result;
        result.x = expf(val.x - row_max) / row_sum;
        result.y = expf(val.y - row_max) / row_sum;
        result.z = expf(val.z - row_max) / row_sum;
        result.w = expf(val.w - row_max) / row_sum;
        O_vec[i] = result;
    }
    
    // Handle remaining elements
    for (int i = vectorized_N + tid; i < N; i += blockSize) {
        O_d[row * N + i] = expf(X_d[row * N + i] - row_max) / row_sum;
    }
}

#define M 1024      // Number of rows
#define N 32768     // Number of columns (features per row)

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

    // Launch Kernel with correct grid/block configuration
    int threadsPerBlock = 32;  
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;  

    // Run naive version
    cudaEventRecord(start);
    naiveSoftmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(X_d, O_d, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << "Naive kernel execution time: " << ms << "ms" << std::endl;  

    // Calculate shared memory size (per block)
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    
    // Run shared memory version
    cudaEventRecord(start);
    sharedSoftmax_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(X_d, O_d, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << "Shared memory kernel execution time: " << ms << "ms" << std::endl;  

    // Run Vectorized version
    cudaEventRecord(start);
    vectorizedSoftmax_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(X_d, O_d, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << "Vectorized kernel: " << ms << "ms" << std::endl;  
    
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