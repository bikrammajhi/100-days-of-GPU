# 🚀 Day 006 of 100 Days of GPU
---

## 🌟 **Scalable Programming Model**  
CUDA's magic lies in its **scalable parallelism**! It uses three key abstractions to simplify GPU programming:  
1. **🧵 Thread Groups** (Hierarchy of threads, blocks, and grids)  
2. **💾 Shared Memories** (Fast, low-latency memory for cooperative threads)  
3. **⏳ Barrier Synchronization** (`__syncthreads()` for thread coordination)  

### Key Idea:  
- Decompose problems into **coarse sub-problems** (handled by **thread blocks**) and **finer pieces** (solved by threads within a block).  
- **Automatic Scalability**: Blocks run on any Streaming Multiprocessor (SM), so code scales across GPUs!  

![Scalability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/automatic-scalability.png)  
*Figure 3: Scalability across SMs (Source: CUDA C++ Programming Guide)*  

---

## 🧑‍💻 **Kernels: The Heart of CUDA**  
Kernels are functions that run **N times in parallel** across **N threads**.  

### Example: Vector Addition  
```cpp
// Kernel definition (runs on GPU)
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;  // Built-in thread index 🆔
    C[i] = A[i] + B[i];
}

int main() {
    // Launch kernel with 1 block of N threads 🚀
    VecAdd<<<1, N>>>(A, B, C);
}
```
- `__global__`: Marks a kernel (GPU function).  
- `<<<1, N>>>`: **Execution configuration** (1 block, N threads).  

---

## 🧵 **Thread Hierarchy**  
CUDA threads are organized in **3D grids** of **3D blocks**:  
- **Thread Block**: Up to 1024 threads (e.g., 16x16 for matrices).  
- **Grid**: Multiple blocks (scales with data size).  

### Example: Matrix Addition (2D Block)  
```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global row index 🌍
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Global column index 🌍
    if (i < N && j < N) 
        C[i][j] = A[i][j] + B[i][j];
}

int main() {
    dim3 threadsPerBlock(16, 16);  // 256 threads/block 🧊
    dim3 numBlocks(N/16, N/16);    // Grid size 📦
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```
- `blockIdx`: Block index in the grid.  
- `blockDim`: Block dimensions (e.g., 16x16).  

![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)  
*Figure 4: Grid of Thread Blocks (Source: CUDA C++ Programming Guide)*  

---

## 🌀 **Thread Block Clusters** (Compute Capability 9.0+)

With NVIDIA Compute Capability **9.0**, CUDA introduces an **optional hierarchical level**: **Thread Block Clusters**. These clusters group multiple thread blocks and allow new forms of **shared memory access** and **synchronization** between them.

### ✅ Key Benefits:

* **Guaranteed Co-Scheduling**: All blocks in a cluster are co-located on the same **GPU Processing Cluster (GPC)**.
* **Distributed Shared Memory (DSM)**: Thread blocks in a cluster can **access and modify each other's shared memory**.
* **Hardware Synchronization**: Supports `cluster.sync()` and other cooperative primitives.
* **Flexible Grid Configuration**: Can define clusters at **compile-time** or **runtime**.

### 🔧 **1. Runtime Cluster Launch using `cudaLaunchKernelEx`**

This method allows you to define the cluster size dynamically at runtime.

```cpp
// Host code
cudaLaunchConfig_t config = {0};
// The grid dimension is not affected by cluster launch, and is still enumerated
// using number of blocks.
// The grid dimension should be a multiple of cluster size.
config.gridDim = numBlocks;
config.blockDim = threadsPerBlock;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 2;  // 2 blocks per cluster
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;

config.attrs = attribute;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_kernel, input, output);
```

```cpp
// Device code
__global__ void cluster_kernel(float* input, float* output) {
    // cluster.sync();         // Barrier for blocks in the same cluster
    // num_blocks(), num_threads() to query size
    // dim_blocks(), dim_threads() for thread/block rank
}
```

### 🧵 **2. Compile-Time Cluster Declaration using `__cluster_dims__`**

You can also specify the cluster configuration at **compile time** using a special kernel attribute.

```cpp
// Device code with cluster attribute
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float* input, float* output) {
    // DSM access and sync possible here
}
```

```cpp
// Host code
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

// Make sure numBlocks is a multiple of cluster size
cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
```

> ❗Note: When using `__cluster_dims__`, the cluster size is fixed at compile time and **cannot** be changed at runtime.

### 💡 Additional Notes:

* **Max cluster size** (portable): **8 thread blocks**
  * May be smaller depending on GPU hardware or MIG (Multi-Instance GPU) config.
* Use `cudaOccupancyMaxPotentialClusterSize()` to query max cluster size.
* Distributed Shared Memory allows **atomic operations**, **read/write access**, and **shared state** across blocks in a cluster.

### 📊 Cluster Layout Illustration:

![Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)
*Figure: Grid of Clusters (Source: CUDA C++ Programming Guide)*

---

## 💾 **Memory Hierarchy**  
CUDA threads access multiple memory spaces:  
1. **Local Memory**: Private per-thread.  
2. **Shared Memory**: Fast memory per block.  
3. **Global Memory**: Accessible by all threads.  
4. **Constant/Texture Memory**: Read-only, optimized for specific patterns.  

![Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)  
*Figure 6: Memory Hierarchy (Source: CUDA C++ Programming Guide)*  

---

## 🌐 **Heterogeneous Programming**  
CUDA bridges **CPU (Host)** and **GPU (Device)**:  
- **Host Memory**: RAM on the CPU.  
- **Device Memory**: GPU's global memory.  
- **Unified Memory**: Single coherent memory space (simplifies data management).  

![Heterogeneous Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png)  
*Figure 7: Host-Device Model (Source: CUDA C++ Programming Guide)*  

---

## ⚡ **Asynchronous SIMT Programming**

With **NVIDIA Ampere and newer architectures**, CUDA introduces **asynchronous operations**, allowing threads to initiate tasks that execute independently and **synchronize at a later point** using well-defined thread scopes and synchronization objects.

### 🔄 **Asynchronous Operations**

An *asynchronous operation* is executed **as if by another thread**, freeing the initiating thread to continue execution. Completion is coordinated using **synchronization objects**, which may be:

* `cuda::barrier` – fine-grained thread synchronization
* `cuda::pipeline` – asynchronous data copy and execution management

These objects operate across various **thread scopes**, enabling flexible synchronization:

| Thread Scope                              | Description                                              |
| ----------------------------------------- | -------------------------------------------------------- |
| `cuda::thread_scope::thread_scope_thread` | Only the initiating thread synchronizes                  |
| `cuda::thread_scope::thread_scope_block`  | Any or all threads in the **same block** may synchronize |
| `cuda::thread_scope::thread_scope_device` | Any or all threads in the **same device**                |
| `cuda::thread_scope::thread_scope_system` | Any CUDA or **CPU threads** in the system                |

### 🧠 **Use Case: Overlap Data Transfer and Compute**

```cpp
// Copy memory asynchronously across threads
cuda::memcpy_async(dst, src, size, barrier);
```

### ⏳ **Example: Asynchronous Barrier Within a Thread Block**

```cpp
#include <cuda/barrier>

__global__ void async_kernel() {
  __shared__ cuda::barrier<cuda::thread_scope_block> block_barrier;
  
  // Initialize the barrier with the number of participating threads
  if (threadIdx.x == 0) {
    new(&block_barrier) cuda::barrier<cuda::thread_scope_block>(blockDim.x);
  }
  
  __syncthreads(); // Ensure barrier is initialized before use
  
  // All threads arrive at the barrier independently
  block_barrier.arrive_and_wait();  // 🛑 Sync point
}
```

### ✅ **Why It Matters**

* Efficient **SIMT parallelism**
* Maximize **GPU utilization** via task/data overlap
* Fine-grained control over **when and how threads wait**

> 💡 These capabilities enable **high-performance GPU programs** that better exploit modern GPU architectures by minimizing idle time and maximizing concurrency.

---

## 📊 **Compute Capability**  
Each GPU has a version (e.g., **9.0 for Hopper**). Key features:  
| Compute Capability | Architecture     | Key Features                          |
|---------------------|------------------|---------------------------------------|
| 9.0                 | Hopper           | Thread Block Clusters, DSM            |
| 8.0                 | Ampere           | Async SIMT, Tensor Cores              |
| 7.5                 | Turing           | Ray Tracing, Mixed-Precision          |

---

## 📦 **CUDA Runtime**

The **CUDA Runtime API** is implemented in the `cudart` library (`cudart.lib` / `libcudart.so`) and provides all runtime-level functions prefixed with `cuda`. This API is your go-to interface for interacting with NVIDIA GPUs at a high level.

> ☝️ Pro Tip: Stick to one runtime instance! Passing pointers between components linked to different instances of cudart = ❌ Undefined Behavior.

### ⚙️ **Runtime Initialization**

Since **CUDA 12.0**, `cudaSetDevice()` will explicitly initialize the runtime and primary context for the selected GPU.

### 🧵 **Context Creation:**
- Automatically happens on first use.
- Compiles and loads device code with JIT if needed.
- Shared among all host threads.

```cpp
cudaSetDevice(0);   // 🚨 Initializes runtime!
cudaFree(0);        // 🪄 Old trick to trigger context creation
```

---

## 💾 **Device Memory Management**

Kernels run on **device memory**, not host memory. You'll need to:

* Allocate GPU memory
* Copy data from Host → Device
* Launch the kernel
* Copy results back

### 🧮 **Linear Memory**

```cpp
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```

🌟 Linear memory uses unified address space — great for pointer-based structures like linked lists.

🧠 On newer GPUs (Pascal+), address space goes up to **47–49 bits**!

![Linear Memory](https://raw.githubusercontent.com/HappyPenguinCode/cuda-images/main/device-memory-layout.png)

### 📐 **Pitched Memory**

For 2D arrays, use pitched memory for optimal performance:

```cpp
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;

// Allocate pitched memory on the device
cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height);

// Launch the kernel
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height) {
    for (int r = 0; r < height; ++r) {
        // Compute the address of the beginning of row `r`
        float* row = (float*)((char*)devPtr + r * pitch);
        
        for (int c = 0; c < width; ++c) {
            float element = row[c];
            // Perform computations with 'element' here
        }
    }
}
```

### 🧊 **3D Memory**

For 3D data structures:

```cpp
// Host code
int width = 64, height = 64, depth = 64;

// Define a 3D extent (width in bytes, height, depth)
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);

// Allocate 3D pitched memory
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);

// Launch the kernel
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);


// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth) {
    // Get the raw pointer and pitch information
    char* devPtr = (char*)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;

    // Traverse 3D memory layout
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;

        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);

            for (int x = 0; x < width; ++x) {
                float element = row[x];
                // Use 'element' as needed
            }
        }
    }
}
```

---

## 🚀 **Kernel Launching**

```cpp
// Define kernel
__global__ void addVectors(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Launch kernel
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
addVectors<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```

---

## 🚦 **Error Handling**

Always check return values from CUDA runtime calls:

```cpp
cudaError_t err = cudaMemcpy(...);
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

Also useful to check errors after kernel launch:

```cpp
// Check for kernel launch errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

---

## **Global Variables and Symbol Access**

```cpp
// Constant memory example
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));  // Copy to constant memory
cudaMemcpyFromSymbol(data, constData, sizeof(data)); // Copy from constant memory

// Device variable example
__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));  // Copy to device variable

// Device pointer example
__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));  // Allocate device memory
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));  // Copy pointer to device

// Get symbol address and size
float* d_ptr;
size_t size;
cudaGetSymbolAddress((void**)&d_ptr, devData);  // Get address of device variable
cudaGetSymbolSize(&size, devData);              // Get size of device variable
```

---

## 📚 **Summary Table**

| Function                  | Description                           |
| ------------------------- | ------------------------------------- |
| `cudaSetDevice()`         | Selects the active GPU device         |
| `cudaDeviceReset()`       | Resets the GPU device                 |
| `cudaMalloc()`            | Allocates device memory               |
| `cudaFree()`              | Frees device memory                   |
| `cudaMemcpy()`            | Copies memory between host and device |
| `cudaMallocPitch()`       | Allocates pitched 2D device memory    |
| `cudaMalloc3D()`          | Allocates 3D device memory            |
| `__global__`              | Defines a kernel function             |
| `cudaGetLastError()`      | Retrieves the last error              |
| `cudaGetErrorString()`    | Returns error string                  |

---

## 📘 **Further Reading**

For more detailed information, refer to the [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).

---

Happy Coding! 🚀
