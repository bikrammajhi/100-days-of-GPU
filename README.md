# 🚀 100 Days of GPU Challenge

Welcome to my **100 Days of GPU** journey! This repository will serve as a public log of my learning, experiments, and projects as I dive deep into the world of GPU architecture, CUDA programming, memory hierarchies, parallelism, and acceleration for deep learning and scientific computing.

The goal is to gain both theoretical and hands-on understanding of how GPUs work and how to fully leverage their power for high-performance computing.

---

## 📅 Progress Log


### ✅ Day 001: CPU vs. GPU Architectures & Parallelism [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20001_%20GPU%20vs%20CPU%20architecture)

- Processor Trends and Evolution
- Reviewed Moore’s Law and its impact on transistor scaling.
- Compared Latency-Oriented Design (CPU) vs. Throughput-Oriented Design (GPU).
- Deep dive into CPU and GPU design:
- History of GPUs: From graphics to general-purpose computing.
- Explored the limitations of parallelization.

### ✅ Day 002:  GPU Architecture Fundamentals [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20002_Hello_GPU)

- Wrote first 'Hello World' [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bikrammajhi/ddd45d3d27cd7a05c0cdd3174dc0f578/hello-world-in-cuda.ipynb) programme in CUDA

- Studied the evolution of GPU architectures: from Fermi → Kepler → Pascal → Volta → Turing → Ampere.
- Learned about:
  - SMs (Streaming Multiprocessors)
  - Warp execution and scheduling
  - CUDA threads, blocks, and grids
  - Shared memory, L1, L2 caches
  - Bank conflicts and memory coalescing
  - Tensor Cores and their matrix-multiplication acceleration

### ✅ Day 003: Vector Addition [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20003_Vector_Addition)

* Implemented vector addition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bikrammajhi/0e86a95d1a010056c70ee1decdb2275e/vector-addition-in-cuda.ipynb) - the "Hello World!" of data parallel programming
* Studied different types of parallelism:
  * Task Parallelism vs Data Parallelism
  * Why GPUs excel at data-parallel tasks
* Learned about CUDA memory management:
  * `cudaMalloc` for device memory allocation
  * `cudaMemcpy` for host-device data transfers
  * `cudaFree` for releasing GPU memory
* Learned CUDA kernel fundamentals:
  * Thread/block organization in grids
  * Using thread indices with `blockDim.x*blockIdx.x + threadIdx.x`
  * Handling boundary conditions with ceiling division
* Implemented proper error checking with `cudaGetLastError()` and `cudaGetErrorString()`
* Explored function qualifiers: `__global__`, `__device__`, and `__host__`
* Learned about asynchronous kernel execution and synchronization

### ✅ Day 004: Multidimensional Grids and Data [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20004_Multidimensional_Grids_and_Data)
* Implemented RGB to Grayscale conversion [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bikrammajhi/b28ce2e01b465c6e1dcf5124a540ac04/rgb2gray-in-cuda.ipynb) using 2D thread organization
* Explored CUDA's multidimensional grid capabilities (up to 3D)
* Learned key multidimensional indexing techniques:
  * Using `blockIdx.{x,y}` and `threadIdx.{x,y}` for 2D addressing
  * Converting between 2D coordinates and linear memory
* Handled boundary conditions in multidimensional kernels
* Fixed CUDA version mismatch on Google Colab

### ✅ Day 005: Image Blur Processing & Performance Analysis [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20005_Image_Blur)
* Implemented 3×3 average filter blur operation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bikrammajhi/e4363e9116a909dcde94013aac3d6bcd/image-blur-with-cuda.ipynb) using CUDA
* Analyzed memory transfer vs. computation performance (H2D: 0.26ms, kernel: 0.30ms, D2H: 0.66ms)
* Explored optimization strategies including shared memory and memory access patterns

### ✅ Day 006: CUDA Programming Model & Runtime API [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20006_Naive_MatMul)
* Explored CUDA's scalable parallelism with thread hierarchy (threads, blocks, grids)
* Implemented Naive Matrix Multiplication [![View on GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20006_Naive_MatMul/matmul.cu)
* Learned advanced features including Grid Block Clusters, Thread Block Clusters (9.0) and Asynchronous SIMT (8.0+)
* Compute Capability (SM version): Features supported by the GPU hardware
* Memory management techniques (linear, pitched, 3D) and their access patterns
* Explored ways of accessing global variables

### ✅ Day 007: L2 Cache and Shared memory [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20007_L2%20and%20Shared%20Memory)
* Implemented tiled-Matrix Multiplication using Shared memory[![View on GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20007_L2%20and%20Shared%20Memory/matmul.cu)
*  L2 Cache Control
*  Memory Access Patterns
*  L2 Cache Architecture
*  Device Property Insights
*  Hit Ratio Strategy
*  L2 Cache Reset Options
*  L2 Set-Aside Layout

### ✅ Day 008: CUDA Memory Transfer Performance
* Implemented CUDA memory transfer optimizations using Pinned and Unified Memory
* Understood Memory Types (Pageable, Pinned, Device, Unified)
* Analyzed Transfer Bandwidths and PCIe Efficiency
* Applied Async Transfer Techniques
* Optimized with Batching and Memory Pools
* Visualized Performance Bottlenecks

### ✅ Day 009: Page-Locked Host Memory and Thread Coarsening
* Impliemented Thread Coarsening Matrix Mulitplication [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20009_Thread%20Coarsening)
* Page-Locked Host Memory
* Portable Memory
* Write-Combining Memory
* Mapped Memory




 




