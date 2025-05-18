# 🚀 Day 003: Vector Addition with CUDA 🚀
## 100 Days of GPU Challenge

Welcome to Day 3 of my 100 Days of GPU Challenge! Today we're exploring **Vector Addition** - the "Hello World" of parallel programming on GPUs.

## 📚 What You'll Learn

- Types of parallelism
- CUDA memory management
- Launching parallel kernels
- Handling boundary conditions
- Error checking in CUDA

## 🔄 Types of Parallelism

### 🧩 Task Parallelism
- Different operations performed on same or different data
- Usually involves a modest number of tasks with modest parallelism
- Example: Running different algorithms simultaneously on different processors

### 📊 Data Parallelism
- Same operation performed on different data
- Unleashes **massive amounts of parallelism**
- **Perfect for GPUs** with their thousands of cores
- Example: Our vector addition problem!

## 🧮 Vector Addition Example

Vector addition is the perfect first example for data parallelism:

```
Input Vector x: [x₀, x₁, x₂, ..., xₙ]
Input Vector y: [y₀, y₁, y₂, ..., yₙ]
Output Vector z: [x₀+y₀, x₁+y₁, x₂+y₂, ..., xₙ+yₙ]
```

### ⚙️ Sequential Implementation

```c
void vecadd(float* x, float* y, float* z, int N) {
    for(unsigned int i = 0; i < N; ++i) {
        z[i] = x[i] + y[i];
    }
}
```

## 🧠 CUDA Memory Management 

Before we run operations on the GPU, we need to manage memory carefully. CPU and GPU have separate memory spaces!

### 🔹 Memory Allocation
```c
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

### 🔹 Memory Deallocation
```c
cudaError_t cudaFree(void *devPtr);
```

### 🔹 Memory Copy
```c
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```

Where `kind` specifies the direction:
- `cudaMemcpyHostToHost`
- `cudaMemcpyHostToDevice` ← CPU to GPU
- `cudaMemcpyDeviceToHost` ← GPU to CPU
- `cudaMemcpyDeviceToDevice`

### 📝 Memory Management Code Example

```c
void vecadd(float* x, float* y, float* z, int N) {
    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, N*sizeof(float));
    cudaMalloc((void**) &y_d, N*sizeof(float));
    cudaMalloc((void**) &z_d, N*sizeof(float));

    // Copy data to GPU memory
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform computation on GPU
    // ...

    // Copy data from GPU memory
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}
```

## 🌐 CUDA's Parallel Execution Model

In CUDA, we organize threads in a hierarchical structure:

### 🧱 Thread Organization
- **Threads**: Individual execution units
- **Blocks**: Groups of threads that can collaborate
- **Grid**: Collection of blocks

![CUDA Thread Hierarchy](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

### 🚀 Launching a Kernel

A kernel is launched with a configuration specifying the grid and block dimensions:

```c
const unsigned int numThreadsPerBlock = 512;
const unsigned int numBlocks = N/numThreadsPerBlock;
vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
```

### 💻 Parallel Vector Addition Kernel

```c
__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    z[i] = x[i] + y[i];
}
```

The magic formula: `i = blockDim.x*blockIdx.x + threadIdx.x` calculates a unique index for each thread!

## 🛡️ Handling Boundary Conditions

What if N isn't a multiple of our block size? We need to handle the boundary!

### ✅ Better Launch Configuration
```c
const unsigned int numBlocks = (N + numThreadsPerBlock - 1)/numThreadsPerBlock;
```
This is a clever way to calculate the ceiling of N/numThreadsPerBlock.

### ✅ Safer Kernel with Boundary Check
```c
__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N) {
        z[i] = x[i] + y[i];
    }
}
```

## 🏷️ Function Qualifiers in CUDA

CUDA uses special keywords to indicate where a function can run:

- `__global__`: Runs on the device (GPU), called from host (CPU)
- `__device__`: Runs on the device, called from device
- `__host__`: Runs on the host, called from host (default)

You can even make functions that run on both:

```c
__host__ __device__ float f(float a, float b) {
    return a + b;
}
```

## ⚡ Asynchronous Execution

CUDA kernel launches are asynchronous by default! This enables overlapping computations.

To wait for completion:
```c
cudaError_t cudaDeviceSynchronize();
```

## 🔍 Error Checking

Always check for errors in your CUDA code:

```c
cudaError_t err = cudaMalloc((void**) &d_a, size);
if(err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

For kernel calls:
```c
vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
cudaError_t err = cudaGetLastError();
if(err != cudaSuccess) {
    printf("Kernel Error: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

## 📈 Complete Example

Here's a complete example of vector addition in CUDA:

```c
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel definition
__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N) {
        z[i] = x[i] + y[i];
    }
}

void vecadd(float* x, float* y, float* z, int N) {
    // Device memory pointers
    float *x_d, *y_d, *z_d;
    
    // Allocate device memory
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));
    
    // Copy inputs to device
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Copy result back to host
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main() {
    const int N = 1000000;
    
    // Allocate host memory
    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));
    float *z = (float*)malloc(N*sizeof(float));
    
    // Initialize input vectors
    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // Perform vector addition
    vecadd(x, y, z, N);
    
    // Verify result
    bool correct = true;
    for(int i = 0; i < N; i++) {
        if(z[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    
    printf("Vector addition %s\n", correct ? "PASSED" : "FAILED");
    
    // Free host memory
    free(x);
    free(y);
    free(z);
    
    return 0;
}
```

## 📚 References

- 📘 Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. "Programming Massively Parallel Processors: A Hands-on Approach." Morgan Kaufmann, 2022.
- 🔗 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
