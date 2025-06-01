#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define NUM_ROWS 1024           // Number of rows in input (e.g., batch size)
#define FEATURE_SIZE 32768      // Number of features per row (e.g., vocabulary size)

// =======================================================================================
// Kernel 1: Naive Softmax Implementation (Global Memory Only)
// Each thread computes the softmax for one row of the input matrix.
// =======================================================================================
__global__ void SoftmaxNaiveKernel(const float *d_input, float *d_output, int numRows, int featureDim) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= numRows) return;

    float maxVal = -INFINITY;
    float sumExp = 0.0f;

    // Step 1: Find the max value in the row for numerical stability
    for (int col = 0; col < featureDim; ++col) {
        int idx = rowIdx * featureDim + col;
        maxVal = fmaxf(d_input[idx], maxVal);
    }

    // Step 2: Compute sum of exponentials
    for (int col = 0; col < featureDim; ++col) {
        int idx = rowIdx * featureDim + col;
        sumExp += expf(d_input[idx] - maxVal);
    }

    // Step 3: Compute softmax values
    for (int col = 0; col < featureDim; ++col) {
        int idx = rowIdx * featureDim + col;
        d_output[idx] = expf(d_input[idx] - maxVal) / sumExp;
    }
}

// =======================================================================================
// Kernel 2: Shared Memory Optimized Softmax
// Uses shared memory to reduce global memory accesses during max and sum computations.
// =======================================================================================
__global__ void SoftmaxSharedKernel(const float *d_input, float *d_output, int numRows, int featureDim) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= numRows) return;

    extern __shared__ float s_temp[]; // Shared memory buffer

    float maxVal = -INFINITY;
    float sumExp = 0.0f;

    // Step 1: Compute max using tiled loading into shared memory
    for (int col = 0; col < featureDim; col += blockDim.x) {
        int globalIdx = rowIdx * featureDim + col + threadIdx.x;
        if (col + threadIdx.x < featureDim) {
            s_temp[threadIdx.x] = d_input[globalIdx];
        }
        __syncthreads();

        for (int i = 0; i < blockDim.x && (col + i) < featureDim; ++i) {
            maxVal = fmaxf(maxVal, s_temp[i]);
        }
        __syncthreads();
    }

    // Step 2: Compute sum of exponentials
    for (int col = 0; col < featureDim; col += blockDim.x) {
        int globalIdx = rowIdx * featureDim + col + threadIdx.x;
        if (col + threadIdx.x < featureDim) {
            s_temp[threadIdx.x] = expf(d_input[globalIdx] - maxVal);
        }
        __syncthreads();

        for (int i = 0; i < blockDim.x && (col + i) < featureDim; ++i) {
            sumExp += s_temp[i];
        }
        __syncthreads();
    }

    // Step 3: Compute normalized softmax values
    for (int col = 0; col < featureDim; col += blockDim.x) {
        int globalIdx = rowIdx * featureDim + col + threadIdx.x;
        if (col + threadIdx.x < featureDim) {
            s_temp[threadIdx.x] = expf(d_input[globalIdx] - maxVal) / sumExp;
        }
        __syncthreads();

        if (col + threadIdx.x < featureDim) {
            d_output[globalIdx] = s_temp[threadIdx.x];
        }
        __syncthreads();
    }
}

// =======================================================================================
// Kernel 3: Vectorized Softmax using float4
// Processes 4 elements per thread to reduce instruction count and memory transactions.
// Uses shared memory for reduction steps.
// =======================================================================================
__global__ void SoftmaxVectorizedKernel(const float *d_input, float *d_output, int numRows, int featureDim) {
    int rowIdx = blockIdx.x;
    int threadIdxX = threadIdx.x;
    int blockSize = blockDim.x;

    if (rowIdx >= numRows) return;

    extern __shared__ float s_shared[];  // Shared memory buffer for reduction

    float localMax = -INFINITY;
    float localSum = 0.0f;

    int vecLimit = (featureDim / 4) * 4;
    const float4 *inputVec = reinterpret_cast<const float4*>(d_input + rowIdx * featureDim);
    float4 *outputVec = reinterpret_cast<float4*>(d_output + rowIdx * featureDim);

    // Step 1: Compute max value per row using float4
    for (int i = threadIdxX; i < vecLimit / 4; i += blockSize) {
        float4 val = inputVec[i];
        localMax = fmaxf(localMax, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }

    // Handle tail (non-divisible by 4)
    for (int i = vecLimit + threadIdxX; i < featureDim; i += blockSize) {
        localMax = fmaxf(localMax, d_input[rowIdx * featureDim + i]);
    }

    // Reduce max across threads
    s_shared[threadIdxX] = localMax;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdxX < stride) {
            s_shared[threadIdxX] = fmaxf(s_shared[threadIdxX], s_shared[threadIdxX + stride]);
        }
        __syncthreads();
    }
    float rowMax = s_shared[0];
    __syncthreads();

    // Step 2: Compute exponential sum
    for (int i = threadIdxX; i < vecLimit / 4; i += blockSize) {
        float4 val = inputVec[i];
        localSum += expf(val.x - rowMax) + expf(val.y - rowMax) +
                    expf(val.z - rowMax) + expf(val.w - rowMax);
    }

    for (int i = vecLimit + threadIdxX; i < featureDim; i += blockSize) {
        localSum += expf(d_input[rowIdx * featureDim + i] - rowMax);
    }

    // Reduce sum across threads
    s_shared[threadIdxX] = localSum;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdxX < stride) {
            s_shared[threadIdxX] += s_shared[threadIdxX + stride];
        }
        __syncthreads();
    }
    float rowSum = s_shared[0];
    __syncthreads();

    // Step 3: Normalize and write results
    for (int i = threadIdxX; i < vecLimit / 4; i += blockSize) {
        float4 val = inputVec[i];
        float4 result;
        result.x = expf(val.x - rowMax) / rowSum;
        result.y = expf(val.y - rowMax) / rowSum;
        result.z = expf(val.z - rowMax) / rowSum;
        result.w = expf(val.w - rowMax) / rowSum;
        outputVec[i] = result;
    }

    for (int i = vecLimit + threadIdxX; i < featureDim; i += blockSize) {
        d_output[rowIdx * featureDim + i] =
            expf(d_input[rowIdx * featureDim + i] - rowMax) / rowSum;
    }
}


// =======================================================================================
// Kernel 4: Coalesced Memory Access Softmax (Block-wise Processing)
// Each block processes multiple rows, with threads accessing consecutive memory addresses
// =======================================================================================
__global__ void SoftmaxCoalescedKernel(const float *d_input, float *d_output, int numRows, int featureDim) {
    extern __shared__ float s_data[];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int rowIdx = blockIdx.x;
    
    if (rowIdx >= numRows) return;
    
    // Shared memory layout: [row_data | max_vals | sum_vals]
    float *s_row = s_data;
    float *s_max = s_data + featureDim;
    float *s_sum = s_data + featureDim + blockSize;
    
    // Step 1: Coalesced load of entire row into shared memory
    int offset = rowIdx * featureDim;
    for (int i = tid; i < featureDim; i += blockSize) {
        s_row[i] = d_input[offset + i];
    }
    __syncthreads();
    
    // Step 2: Find maximum value using parallel reduction
    float threadMax = -INFINITY;
    for (int i = tid; i < featureDim; i += blockSize) {
        threadMax = fmaxf(threadMax, s_row[i]);
    }
    s_max[tid] = threadMax;
    __syncthreads();
    
    // Parallel reduction for max
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    float maxVal = s_max[0];
    __syncthreads();
    
    // Step 3: Compute sum of exponentials using parallel reduction
    float threadSum = 0.0f;
    for (int i = tid; i < featureDim; i += blockSize) {
        threadSum += expf(s_row[i] - maxVal);
    }
    s_sum[tid] = threadSum;
    __syncthreads();
    
    // Parallel reduction for sum
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float sumExp = s_sum[0];
    __syncthreads();
    
    // Step 4: Compute and store softmax values with coalesced writes
    for (int i = tid; i < featureDim; i += blockSize) {
        d_output[offset + i] = expf(s_row[i] - maxVal) / sumExp;
    }
}


// =======================================================================================
// Main function: Allocates memory, launches kernels, and benchmarks them.
// =======================================================================================
int main() {
    size_t totalBytes = NUM_ROWS * FEATURE_SIZE * sizeof(float);

    // Host memory allocation
    float *h_input = (float*)malloc(totalBytes);
    float *h_output = (float*)malloc(totalBytes);

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, totalBytes);
    cudaMalloc((void**)&d_output, totalBytes);

    // Initialize host input with random data
    for (int i = 0; i < NUM_ROWS * FEATURE_SIZE; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    int threadsPerBlock = 256;
    int numBlocks = (NUM_ROWS + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedBytes = threadsPerBlock * sizeof(float);
    float elapsedMs;

    // Run naive kernel
    cudaEventRecord(startEvent);
    SoftmaxNaiveKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, NUM_ROWS, FEATURE_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    std::cout << "Naive Kernel Time: " << elapsedMs << " ms\n";

    // Run shared memory kernel
    cudaEventRecord(startEvent);
    SoftmaxSharedKernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(d_input, d_output, NUM_ROWS, FEATURE_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    std::cout << "Shared Kernel Time: " << elapsedMs << " ms\n";

    // Run vectorized kernel
    cudaEventRecord(startEvent);
    SoftmaxVectorizedKernel<<<NUM_ROWS, threadsPerBlock, sharedBytes>>>(d_input, d_output, NUM_ROWS, FEATURE_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    std::cout << "Vectorized Kernel Time: " << elapsedMs << " ms\n";

    // Run shared coalsced memory kernel
    cudaEventRecord(startEvent);
    //SoftmaxCoalescedKernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(d_input, d_output, NUM_ROWS, FEATURE_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    std::cout << "Coalsced Shared Kernel Time: " << elapsedMs << " ms\n";


    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

