# CUDA Matrix Multiplication with Shared Memory Implementation Guide

## Overview

This document explains an optimized CUDA implementation of matrix multiplication that leverages **shared memory** for significant performance improvements. Unlike the basic implementation, this version uses a tiling strategy where each thread block collaboratively computes a sub-matrix of the result, dramatically reducing global memory access and improving cache efficiency.

![Matrix Multiplication with Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png)

## Key Optimization Strategy

### Tiled Matrix Multiplication Concept
- Each thread block computes one **square sub-matrix** (Csub) of the result matrix C
- Each thread within a block computes **one element** of that sub-matrix
- The computation is broken down into smaller tiles that fit in shared memory
- Data is reused across multiple threads within the same block

### Memory Access Improvements
- **Matrix A** is read only **(B.width / BLOCK_SIZE)** times from global memory
- **Matrix B** is read only **(A.height / BLOCK_SIZE)** times from global memory
- **Shared memory** provides fast access to frequently used data
- **Significant bandwidth reduction** compared to the naive implementation

## Code Implementation

### Enhanced Matrix Structure

```c
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;      // Number of columns in the matrix
    int height;     // Number of rows in the matrix
    int stride;     // Row stride (allows for sub-matrix representation)
    float* elements; // Pointer to matrix data
} Matrix;
```

**Key Enhancement:**
- **stride field** added to support efficient sub-matrix operations
- Allows representing sub-matrices without copying data
- Stride typically equals width for full matrices

### Device Utility Functions

#### Element Access Function
```c
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}
```

**Explanation:**
- `__device__` keyword: Function runs on GPU, callable only from GPU code
- Uses stride for proper indexing (handles both full matrices and sub-matrices)
- Encapsulates the row-major access pattern

#### Element Assignment Function
```c
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}
```

**Explanation:**
- Provides safe element assignment with stride consideration
- Maintains consistency with GetElement function
- Essential for writing results back to sub-matrices

#### Sub-Matrix Extraction Function
```c
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
```

**Detailed Explanation:**
- **Purpose:** Creates a view of a BLOCK_SIZE×BLOCK_SIZE sub-matrix within a larger matrix
- **row, col parameters:** Sub-matrix coordinates (not element coordinates)
- **Stride preservation:** Maintains original matrix stride for correct indexing
- **Pointer arithmetic:** 
  - `A.stride * BLOCK_SIZE * row`: Moves down `row` blocks vertically
  - `BLOCK_SIZE * col`: Moves right `col` blocks horizontally
- **No data copying:** Creates a view, not a copy

### Host Function: MatMul

```c
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
```

#### Memory Transfer Setup (Enhanced)
```c
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;  // Set both width and stride
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
```

**Key Changes:**
- **stride = width:** For full matrices, stride equals width
- Same memory allocation and transfer process as basic version
- Both matrices A and B follow identical setup pattern

#### Kernel Launch (Identical to Basic Version)
```c
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
```

### GPU Kernel: MatMulKernel (The Core Optimization)

```c
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
```

#### Block-Level Coordination
```c
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
```

**Explanation:**
- **blockRow, blockCol:** Current thread block's position in the grid
- **Csub:** The sub-matrix this entire thread block will compute
- Each block is responsible for a BLOCK_SIZE×BLOCK_SIZE region of the result

#### Thread-Level Setup
```c
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
```

**Explanation:**
- **Cvalue:** Accumulator for the final result of this thread
- **row, col:** This thread's position within the current sub-matrix (0 to BLOCK_SIZE-1)

#### Main Computation Loop (Tiling Strategy)
```c
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
```

**Critical Understanding:**
- **m:** Iterates through tile positions along the shared dimension
- **Number of iterations:** A.width / BLOCK_SIZE (number of tiles needed)
- Each iteration processes one pair of sub-matrices

#### Sub-Matrix Loading
```c
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B  
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
```

**Tiling Logic:**
- **Asub:** Row `blockRow`, column `m` of matrix A's tile grid
- **Bsub:** Row `m`, column `blockCol` of matrix B's tile grid
- These sub-matrices will be multiplied together in shared memory

#### Shared Memory Declaration
```c
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
```

**Shared Memory Benefits:**
- **__shared__ keyword:** Memory shared among all threads in a block
- **Fast access:** Much faster than global memory access
- **Data reuse:** Each element loaded once, used BLOCK_SIZE times
- **Size:** Fixed at compile time (BLOCK_SIZE×BLOCK_SIZE)

#### Cooperative Data Loading
```c
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
```

**Cooperative Loading Strategy:**
- **One element per thread:** Each of the 256 threads loads one element
- **Parallel loading:** All elements loaded simultaneously
- **Perfect mapping:** Thread (row,col) loads position (row,col) of sub-matrix

#### Critical Synchronization
```c
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
```

**Why Synchronization is Essential:**
- **Thread coordination:** Ensures all threads complete loading before any start computing
- **Data consistency:** Prevents reading uninitialized shared memory
- **Block-wide barrier:** All 256 threads must reach this point before any continue

#### Sub-Matrix Multiplication
```c
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
```

**Computation Details:**
- **Inner loop:** Computes dot product for one element of result sub-matrix
- **As[row][e]:** Elements from current thread's row in As
- **Bs[e][col]:** Elements from current thread's column in Bs
- **Fast access:** All data comes from fast shared memory

#### Post-Computation Synchronization
```c
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
```

**Purpose:**
- **Prevents data races:** Ensures computation completes before shared memory reuse
- **Safe iteration:** Allows shared memory to be safely reused in next iteration

#### Result Writing
```c
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

**Final Step:**
- Each thread writes its computed result to global memory
- Uses SetElement for proper sub-matrix indexing

## Performance Analysis

### Memory Access Optimization

#### Compared to Basic Implementation:
- **Basic version:**
  - A read A.height × A.width × B.width times
  - B read B.height × B.width × A.height times

- **Shared memory version:**
  - A read (B.width / BLOCK_SIZE) times
  - B read (A.height / BLOCK_SIZE) times
  - **Reduction factor:** BLOCK_SIZE (typically 16x improvement)

### Shared Memory Benefits

1. **Data Reuse:**
   - Each element in shared memory used BLOCK_SIZE times
   - Amortizes the cost of global memory access

2. **Memory Bandwidth:**
   - Dramatically reduced global memory traffic
   - Higher effective memory bandwidth utilization

3. **Cache Efficiency:**
   - Better spatial and temporal locality
   - Reduced memory latency impact

### Thread Block Efficiency

- **Cooperative Loading:** All threads participate in data loading
- **Parallel Computation:** All threads compute simultaneously
- **Synchronization Overhead:** Minimal compared to memory savings
- **Resource Utilization:** Better GPU resource utilization

## Algorithm Visualization

The tiling strategy works as follows:

1. **Grid Layout:** Thread blocks cover the entire result matrix
2. **Block Responsibility:** Each block computes one sub-matrix of result
3. **Tiled Computation:** Each sub-matrix computed as sum of tile products
4. **Shared Memory Usage:** Each tile pair loaded into shared memory once
5. **Parallel Execution:** All threads in block participate in every step

## Usage Requirements

1. **Matrix Dimensions:** Must be multiples of BLOCK_SIZE
2. **Memory Layout:** Row-major order with proper stride setup
3. **CUDA Capability:** Requires shared memory support
4. **Block Size:** Currently fixed at 16×16 (can be tuned)

## Performance Characteristics

### Advantages
- **Significant speedup:** Typically 5-15x faster than basic version
- **Better scaling:** Performance scales well with matrix size
- **Memory efficient:** Reduced global memory bandwidth usage
- **Cache friendly:** Excellent data locality

### Considerations
- **Shared memory limits:** Block size limited by shared memory capacity
- **Synchronization overhead:** __syncthreads() calls add small overhead
- **Complexity:** More complex code structure
- **Fixed tile size:** Less flexible than some advanced implementations

## Potential Further Optimizations

1. **Memory Coalescing:** Optimize global memory access patterns
2. **Bank Conflicts:** Avoid shared memory bank conflicts
3. **Prefetching:** Overlap computation with memory transfers  
4. **Multiple Data Types:** Template-based implementation
5. **Rectangular Tiles:** Support non-square tile sizes
6. **Warp-Level Optimizations:** Utilize warp-level primitives

This shared memory implementation represents a significant step toward high-performance GPU matrix multiplication, demonstrating key optimization principles that apply to many GPU computing problems.

## Complete Code

Here is the complete CUDA matrix multiplication implementation with shared memory optimization:

```c
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

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        
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
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```
