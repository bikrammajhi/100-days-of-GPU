#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 32
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Create CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));
    
    // Start total timing
    CUDA_CHECK(cudaEventRecord(start_total, 0));
    // Start Host-to-Device transfer timing
    CUDA_CHECK(cudaEventRecord(start_h2d, 0));
    
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A.elements, size));
    CUDA_CHECK(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));
    
    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_B.elements, size));
    CUDA_CHECK(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_C.elements, size));
    
    // Stop Host-to-Device transfer timing
    CUDA_CHECK(cudaEventRecord(stop_h2d, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));
    
    // Invoke kernel with reduced grid size
    // Each thread block will process multiple tiles sequentially
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int tilesX = (B.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int tilesY = (A.height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int totalTiles = tilesX * tilesY;
    
    // Use fewer blocks - each block will process multiple tiles
    // Adjust this factor based on your GPU's SM count and desired occupancy
    int reductionFactor = 4; // Each block processes ~4 tiles on average
    int targetBlocks = (totalTiles + reductionFactor - 1) / reductionFactor;
    
    // Calculate grid dimensions (try to keep it roughly square)
    int gridX = (int)sqrt(targetBlocks);
    int gridY = (targetBlocks + gridX - 1) / gridX;
    
    dim3 dimGrid(gridX, gridY);
    printf("Launching %dx%d blocks to process %d tiles (reduction factor: %d)\n", 
           gridX, gridY, totalTiles, reductionFactor);
    
    // Start kernel timing
    CUDA_CHECK(cudaEventRecord(start_kernel, 0));
    
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    // Stop kernel timing
    CUDA_CHECK(cudaEventRecord(stop_kernel, 0));
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    // Start Device-to-Host transfer timing
    CUDA_CHECK(cudaEventRecord(start_d2h, 0));
    
    // Read C from device memory
    CUDA_CHECK(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));
    
    // Stop Device-to-Host transfer timing
    CUDA_CHECK(cudaEventRecord(stop_d2h, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));
    
    // Stop total timing
    CUDA_CHECK(cudaEventRecord(stop_total, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    
    // Calculate and print timing results
    float time_h2d, time_kernel, time_d2h, time_total;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start_total, stop_total));
    
    // Calculate performance metrics
    size_t total_ops = 2LL * A.height * A.width * B.width; // 2 ops per multiply-add
    double gflops = (double)total_ops / (time_kernel * 1e6); // Convert ms to s, then to GFLOPS
 
    double bandwidth_h2d = (double)(A.width * A.height + B.width * B.height) * sizeof(float) / (time_h2d * 1e6); // GB/s
    double bandwidth_d2h = (double)(C.width * C.height) * sizeof(float) / (time_d2h * 1e6); // GB/s
    
    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("Matrix size: %dx%d x %dx%d -> %dx%d\n", A.height, A.width, B.height, B.width, C.height, C.width);
    printf("Host-to-Device transfer: %.3f ms (%.2f GB/s)\n", time_h2d, bandwidth_h2d);
    printf("Kernel execution:        %.3f ms (%.2f GFLOPS)\n", time_kernel, gflops);
    printf("Device-to-Host transfer: %.3f ms (%.2f GB/s)\n", time_d2h, bandwidth_d2h);
    printf("Total time:              %.3f ms\n", time_total);
    printf("Compute efficiency:      %.1f%% (kernel/total)\n", (time_kernel/time_total)*100);
    printf("===========================\n\n");
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A.elements));
    CUDA_CHECK(cudaFree(d_B.elements));
    CUDA_CHECK(cudaFree(d_C.elements));
    
    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));
}

// Matrix multiplication kernel called by MatMul()
// Each thread block processes multiple output tiles sequentially
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Shared memory used to store sub-matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Calculate total number of tiles in each dimension
    int tilesX = (C.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int tilesY = (C.height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int totalTiles = tilesX * tilesY;
    int tilesPerBlock = (totalTiles + gridDim.x * gridDim.y - 1) / (gridDim.x * gridDim.y);
    
    // Each thread block processes multiple tiles sequentially
    for (int tileIdx = 0; tileIdx < tilesPerBlock; ++tileIdx) {
        // Calculate which tile this thread block is processing
        int linearTileId = blockIdx.y * gridDim.x + blockIdx.x + tileIdx * gridDim.x * gridDim.y;
        
        // Break if we've processed all tiles
        if (linearTileId >= totalTiles) break;
        
        // Convert linear tile ID back to 2D coordinates
        int blockRow = linearTileId / tilesX;
        int blockCol = linearTileId % tilesX;
        
        // Check if this tile is within bounds
        if (blockRow >= tilesY || blockCol >= tilesX) continue;
        
        // Each thread block computes one sub-matrix Csub of C
        Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
        
        // Each thread computes one element of Csub
        // by accumulating results into Cvalue
        float Cvalue = 0;
        
        // Loop over all the sub-matrices of A and B that are
        // required to compute Csub
        // Multiply each pair of sub-matrices together
        // and accumulate the results
        for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
            // Get sub-matrix Asub of A
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            // Get sub-matrix Bsub of B
            Matrix Bsub = GetSubMatrix(B, m, blockCol);
            
            // Load Asub and Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            // Check bounds to handle matrices not perfectly divisible by BLOCK_SIZE
            int globalRow = blockRow * BLOCK_SIZE + row;
            int globalColA = m * BLOCK_SIZE + col;
            int globalRowB = m * BLOCK_SIZE + row;
            int globalColB = blockCol * BLOCK_SIZE + col;
            
            As[row][col] = (globalRow < A.height && globalColA < A.width) ? 
                          GetElement(Asub, row, col) : 0.0f;
            Bs[row][col] = (globalRowB < B.height && globalColB < B.width) ? 
                          GetElement(Bsub, row, col) : 0.0f;
            
            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            __syncthreads();
            
            // Multiply Asub and Bsub together
            for (int e = 0; e < BLOCK_SIZE; ++e)
                Cvalue += As[row][e] * Bs[e][col];
            
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }
        
        // Write Csub to device memory
        // Each thread writes one element, check bounds
        int globalOutRow = blockRow * BLOCK_SIZE + row;
        int globalOutCol = blockCol * BLOCK_SIZE + col;
        if (globalOutRow < C.height && globalOutCol < C.width) {
            SetElement(Csub, row, col, Cvalue);
        }
    }
}

int main() {
    // Print GPU information
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessor Count: %d\n\n", prop.multiProcessorCount);
    
    // Dimensions must be multiples of BLOCK_SIZE
    int width = 1024;  // Increased size for better performance measurement
    int height = 1024;

    printf("Matrix multiplication: %dx%d x %dx%d\n", height, width, width, height);
    printf("Total elements per matrix: %d\n", width * height);
    printf("Memory per matrix: %.1f MB\n\n", (width * height * sizeof(float)) / (1024.0*1024.0));
    // Allocate and initialize host matrices A and B
    Matrix A, B, C;
    A.width = B.width = C.width = width;
    A.height = B.height = C.height = height;
    A.stride = B.stride = C.stride = width;

    size_t size = width * height * sizeof(float);
    A.elements = (float*)malloc(size);
    B.elements = (float*)malloc(size);
    C.elements = (float*)malloc(size);

    if (!A.elements || !B.elements || !C.elements) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Fill A and B with sample data
    for (int i = 0; i < width * height; ++i) {
        A.elements[i] = 1.0f;
        B.elements[i] = 2.0f;
    }

    // Perform matrix multiplication
    MatMul(A, B, C);

    // Print a portion of result matrix C (first 8x8 for readability)
    printf("Result Matrix C (first 8x8 portion):\n");
    int print_size = (width < 8) ? width : 8;
    for (int i = 0; i < print_size; ++i) {
        for (int j = 0; j < print_size; ++j) {
            printf("%6.1f ", C.elements[i * width + j]);
        }
        printf("\n");
    }

    // Verify correctness - each element should be width*2 (64*2=128)
    printf("\nExpected value per element: %.1f\n", (float)(width * 2));
    printf("Actual value at C[0][0]: %.1f\n", C.elements[0]);

    // Free host memory
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}

