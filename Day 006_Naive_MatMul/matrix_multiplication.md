# CUDA Matrix Multiplication Implementation Guide

## Overview

This document explains a straightforward CUDA implementation of matrix multiplication that demonstrates the basic concepts of GPU programming. This implementation does **not** use shared memory optimization, making it easier to understand but less efficient than optimized versions.

## Key Concepts

### Memory Access Pattern
- Matrix A is read **B.width** times from global memory
- Matrix B is read **A.height** times from global memory
- Each thread computes exactly one element of the result matrix C
- Uses row-major storage order for all matrices
  
![Matrix Multiplication without Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png)


### Thread Organization
- Each thread block is 16×16 threads (256 threads total)
- Grid dimensions are calculated to cover the entire result matrix
- Thread mapping: one thread per output matrix element
  
## Code Implementation

### Matrix Structure Definition

```cpp
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;      // Number of columns in the matrix
    int height;     // Number of rows in the matrix
    float* elements; // Pointer to matrix data (1D array representation)
} Matrix;
```

**Explanation:**
- The Matrix structure encapsulates both dimensions and data pointer
- Row-major order means elements are stored row by row in memory
- Access formula: `M(row, col) = M.elements[row * M.width + col]`

### Thread Block Configuration

```cpp
// Thread block size
#define BLOCK_SIZE 16
```

**Explanation:**
- Defines a 16×16 thread block (256 threads per block)
- This is a common choice that balances occupancy and resource usage
- Matrix dimensions must be multiples of BLOCK_SIZE for this implementation

### Kernel Declaration

```cpp
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
```

**Explanation:**
- `__global__` keyword indicates this function runs on GPU and is called from CPU
- Forward declaration allows the host function to reference the kernel

### Host Function: MatMul

```cpp
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
```

This function orchestrates the entire matrix multiplication process on the GPU.

#### Step 1: Transfer Matrix A to GPU

```cpp
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
```

**Explanation:**
- Creates device copy of matrix A (`d_A`)
- Copies dimensions from host matrix
- Calculates memory size needed (width × height × sizeof(float))
- `cudaMalloc`: Allocates GPU memory
- `cudaMemcpy`: Copies data from CPU (host) to GPU (device)

#### Step 2: Transfer Matrix B to GPU

```cpp
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
```

**Explanation:**
- Same process as matrix A
- Creates device copy of matrix B (`d_B`)
- Allocates memory and transfers data to GPU

#### Step 3: Allocate Result Matrix on GPU

```cpp
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
```

**Explanation:**
- Allocates memory for result matrix C on GPU
- Only allocates memory (no data transfer needed since C will be computed)

#### Step 4: Configure and Launch Kernel

```cpp
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
```

**Explanation:**
- `dimBlock`: Defines thread block dimensions (16×16 = 256 threads per block)
- `dimGrid`: Calculates grid dimensions to cover entire result matrix
  - Grid width: `B.width / BLOCK_SIZE` (number of column blocks needed)
  - Grid height: `A.height / BLOCK_SIZE` (number of row blocks needed)
- `<<<dimGrid, dimBlock>>>`: CUDA kernel launch syntax

#### Step 5: Transfer Result Back and Cleanup

```cpp
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
```

**Explanation:**
- `cudaMemcpy`: Copies computed result from GPU back to CPU
- `cudaFree`: Releases GPU memory for all three matrices
- Essential for preventing memory leaks

### GPU Kernel: MatMulKernel

```cpp
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    
    C.elements[row * C.width + col] = Cvalue;
}
```

**Detailed Kernel Explanation:**

#### Thread Index Calculation
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx`: Block coordinates in the grid
- `blockDim`: Block dimensions (16×16 in this case)
- `threadIdx`: Thread coordinates within the block
- Formula maps each thread to a unique (row, col) position in result matrix

#### Matrix Multiplication Loop
```cpp
for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
```
- Implements dot product: C[row][col] = Σ(A[row][e] × B[e][col])
- `e` iterates through elements for dot product calculation
- `A.elements[row * A.width + e]`: Accesses element (row, e) of matrix A
- `B.elements[e * B.width + col]`: Accesses element (e, col) of matrix B

#### Result Storage
```cpp
C.elements[row * C.width + col] = Cvalue;
```
- Stores computed dot product in corresponding position of result matrix

## Memory Access Analysis

### Inefficiencies in This Implementation

1. **Matrix A Access Pattern:**
   - Each row of A is read B.width times (once per column of result)
   - Total reads: A.height × A.width × B.width

2. **Matrix B Access Pattern:**
   - Each column of B is read A.height times (once per row of result)
   - Total reads: B.height × B.width × A.height

3. **No Data Reuse:**
   - No shared memory usage means no data reuse between threads
   - Same data is read multiple times from slow global memory

### Thread Execution Model

- **Total threads launched:** (A.height × B.width)
- **Threads per block:** 256 (16×16)
- **Number of blocks:** (A.height × B.width) / 256
- Each thread is completely independent and computes one output element

## Performance Considerations

### Advantages
- Simple and easy to understand
- Straightforward mapping of threads to output elements
- Good for educational purposes

### Disadvantages
- No memory access optimization
- High global memory bandwidth usage
- No data reuse between threads in the same block
- Not suitable for large matrices due to inefficient memory access

## Usage Requirements

1. **Matrix Dimensions:** Must be multiples of BLOCK_SIZE (16)
2. **Memory Layout:** All matrices must use row-major order
3. **Data Type:** Currently supports only float data type
4. **CUDA Capability:** Requires CUDA-enabled GPU

## Potential Improvements

1. **Shared Memory:** Use shared memory to cache frequently accessed data
2. **Tiling:** Implement tiled matrix multiplication for better data locality
3. **Memory Coalescing:** Optimize memory access patterns
4. **Multiple Data Types:** Template-based implementation for different data types
5. **Error Checking:** Add CUDA error checking for robustness

## Complete Code

Here is the complete CUDA matrix multiplication implementation:

```cpp
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
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
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    
    C.elements[row * C.width + col] = Cvalue;
}
```
