# 🚀 100 Days of GPU Challenge

Welcome to my **100 Days of GPU** journey! This repository will serve as a public log of my learning, experiments, and projects as I dive deep into the world of GPU architecture, CUDA programming, memory hierarchies, parallelism, and acceleration for deep learning and scientific computing.

The goal is to gain both theoretical and hands-on understanding of how GPUs work and how to fully leverage their power for high-performance computing.

---

## 📅 Progress Log

### ✅ Day 001: CPU vs. GPU Architectures & Parallelism 
- Processor Trends and Evolution
- Reviewed Moore’s Law and its impact on transistor scaling.
- Compared Latency-Oriented Design (CPU) vs. Throughput-Oriented Design (GPU).
- Deep dive into CPU and GPU design:
- History of GPUs: From graphics to general-purpose computing.
- Explored the limitations of parallelization.

### ✅ Day 002:  GPU Architecture Fundamentals 

- Wrote first 'Hello World' [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18CxHw1sahD4hhum3XNF4ANDAdwnr5jva?usp=sharing) programme in CUDA

- Studied the evolution of GPU architectures: from Fermi → Kepler → Pascal → Volta → Turing → Ampere.
- Learned about:
  - SMs (Streaming Multiprocessors)
  - Warp execution and scheduling
  - CUDA threads, blocks, and grids
  - Shared memory, L1, L2 caches
  - Bank conflicts and memory coalescing
  - Tensor Cores and their matrix-multiplication acceleration

### ✅ Day 003: Vector Addition

* Implemented vector addition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14A39l716HPIUHQhJLtd2xsYLI60JZH90?usp=sharing) - the "Hello World!" of data parallel programming
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

### ✅ Day 004: Multidimensional Grids and Data
* Implemented RGB to Grayscale conversion [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O_sxGy79ORW3GVuvi3BCr-jsdP9sDVgs?usp=sharing) using 2D thread organization
* Explored CUDA's multidimensional grid capabilities (up to 3D)
* Learned key multidimensional indexing techniques:
  * Using `blockIdx.{x,y}` and `threadIdx.{x,y}` for 2D addressing
  * Converting between 2D coordinates and linear memory
* Handled boundary conditions in multidimensional kernels
* Fixed CUDA version mismatch on Google Colab

### ✅ Day 005: Image Blur Processing & Performance Analysis
* Implemented 3×3 average filter blur operation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k3Rpsv2z2v0jgKRaZBomvPup_4vtZnZ4?usp=sharing) using CUDA
* Analyzed memory transfer vs. computation performance (H2D: 0.26ms, kernel: 0.30ms, D2H: 0.66ms)
* Explored optimization strategies including shared memory and memory access patterns
