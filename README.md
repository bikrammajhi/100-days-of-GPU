# ⚡🚀 100 Days of GPU Challenge 🎉🖥️

Welcome to my **100 Days of GPU** journey!  
This repository is a public log of my learning, experiments, and projects as I dive deep into the world of:

- GPU architecture  
- CUDA programming  
- Memory hierarchies  
- Parallelism  
- Acceleration for deep learning & scientific computing

🎯 The goal: Develop a strong theoretical and practical understanding of GPU-based high-performance computing.

---

## 🧑‍🏫 Mentor & Inspiration  
- 👨‍🔬 **Mentor**: [@hkproj](https://github.com/hkproj)  
- 📘 **Reference Repo**: [100-days-of-gpu](https://github.com/hkproj/100-days-of-gpu)

## 🌐 GPU Programming Platforms  

- [<img src="https://leetgpu.com/favicon.ico" width="18"/> **LeetGPU**](https://leetgpu.com/)  
- [<img src="https://tensara.org/favicon.ico" width="18"/> **Tensara**](https://tensara.org/)  
- [<img src="https://colab.research.google.com/favicon.ico" width="18"/> **Google Colab**](https://colab.research.google.com/gist/bikrammajhi/59ee47f2dc6a04fb79f8fa15d498a4bf/hello-world-in-cuda.ipynb)

## 📓 My Kaggle Notebooks

- [<img src="https://www.kaggle.com/static/images/favicon.ico" width="18"/> **Hello World – Starter Notebook**](https://www.kaggle.com/code/bikrammajhi22/hello-world)  
- [<img src="https://www.kaggle.com/static/images/favicon.ico" width="18"/> **NVIDIA CUTLASS – Deep Dive**](https://www.kaggle.com/code/bikrammajhi22/nvidia-cutlas)

---

# Progress Table   
[![YouTube](https://img.shields.io/badge/YouTube-CS149-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp) [![YouTube](https://img.shields.io/badge/YouTube-PMPP-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) [![CUDA](https://img.shields.io/badge/CUDA-C++%20Guide-76B900?logo=nvidia)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) [![PMPP](https://img.shields.io/badge/Book-PMPP-blue?logo=bookstack)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/materials/Wen-mei%20W.%20Hwu%2C%20David%20B.%20Kirk%2C%20Izzat%20El%20Hajj%20-%20Programming%20Massively%20Parallel%20Processors.%20A%20Hands-on%20Approach-Elsevier%20(2023).pdf)  [![GitHub](https://img.shields.io/badge/GitHub-LeetCUDA-black?logo=github)](https://github.com/xlite-dev/LeetCUDA) [![Website](https://img.shields.io/badge/Website-Operating%20Systems-0A66C2?logo=Google%20Chrome&logoColor=white)](https://www.cse.iitb.ac.in/~mythili/os/) [![YouTube](https://img.shields.io/badge/YouTube-Operating%20Systems-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PLDW872573QAb4bj0URobvQTD41IV6gRkx) [![Website](https://img.shields.io/badge/Website-DECS-0A66C2?logo=Google%20Chrome&logoColor=white)](https://www.cse.iitb.ac.in/~mythili/decs/) [![YouTube](https://img.shields.io/badge/YouTube-DECS-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PLOzRYVm0a65dAAfy0d4aRtj5v0OCAvoCY) [![YouTube](https://img.shields.io/badge/YouTube-CUDA_Training_Series-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj)
[![GitHub](https://img.shields.io/badge/GitHub-CUDA_Training_Series-000000?logo=github)](https://github.com/olcf/cuda-training-series/tree/master)
[![YouTube](https://img.shields.io/badge/YouTube-Heterogeneous_Parallel_Programming-FF0000?logo=youtube)](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bikrammajhi/59ee47f2dc6a04fb79f8fa15d498a4bf/hello-world-in-cuda.ipynb)

| Day | 📋 Topic | 🎯 Key Learning Areas | 💻 Implementation |
|:----|:---------|:----------------------|:------------------|
| **001** | 🖥️ **CPU vs. GPU Architectures & Parallelism** | • Processor trends and Moore's Law<br/>• Latency vs. Throughput-oriented design<br/>• GPU evolution from graphics to general computing<br/>• Parallelization limitations | [![View on GitHub](https://img.shields.io/badge/GitHub-Theoretical_study-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20001_%20GPU%20vs%20CPU%20architecture) |
| **002** | 🏗️ **GPU Architecture Fundamentals** | • GPU architecture evolution (Fermi → Ampere)<br/>• Streaming Multiprocessors (SMs)<br/>• Warp execution and scheduling<br/>• Memory hierarchy (Shared, L1, L2)<br/>• Tensor Cores for matrix acceleration | [![View on GitHub](https://img.shields.io/badge/GitHub-Hello_World_CUDA-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20002_Hello_GPU) |
| **003** | ➕ **Vector Addition** | • Data vs. Task parallelism<br/>• CUDA memory management (`cudaMalloc`, `cudaMemcpy`)<br/>• Kernel fundamentals and thread indexing<br/>• Error checking and synchronization<br/>• Function qualifiers (`__global__`, `__device__`) | [![View on GitHub](https://img.shields.io/badge/GitHub-Vector_addition_kernel-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20003_Vector_Addition) |
| **004** | 🌈 **Multidimensional Grids and Data** | • 2D/3D thread organization<br/>• Multidimensional indexing techniques<br/>• Converting 2D coordinates to linear memory<br/>• Boundary condition handling | [![View on GitHub](https://img.shields.io/badge/GitHub-RGB_to_Grayscale-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20004_Multidimensional_Grids_and_Data) |
| **005** | 🖼️ **Image Blur Processing & Performance** | • 3×3 average filter implementation<br/>• Memory transfer vs. computation analysis<br/>• Performance optimization strategies<br/>• Shared memory patterns | [![View on GitHub](https://img.shields.io/badge/GitHub-Image_blur_analysis-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20005_Image_Blur) |
| **006** | 🔢 **CUDA Programming Model & Matrix Multiplication** | • Scalable parallelism hierarchy<br/>• Grid Block Clusters and Thread Block Clusters<br/>• Asynchronous SIMT programming<br/>• Compute Capability features<br/>• Memory management techniques | [![View on GitHub](https://img.shields.io/badge/GitHub-Naive_Matrix_Multiplication-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20006_Naive_MatMul) |
| **007** | 🧠 **L2 Cache and Shared Memory** | • L2 Cache control and architecture<br/>• Memory access pattern optimization<br/>• Hit ratio strategies<br/>• L2 cache reset options<br/>• Set-aside memory layout | [![View on GitHub](https://img.shields.io/badge/GitHub-Tiled_Matrix_Multiplication-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20007_L2%20and%20Shared%20Memory) |
| **008** | 🚄 **Memory Transfer Performance** | • Memory types (Pageable, Pinned, Unified)<br/>• Transfer bandwidth analysis<br/>• PCIe efficiency optimization<br/>• Async transfer techniques<br/>• Batching and memory pools | [![View on GitHub](https://img.shields.io/badge/GitHub-Memory_transfer_benchmarking-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20008_Data_Transfer%20_Benchmark) |
| **009** | 🔄 **Page-Locked Memory & Thread Coarsening** | • Page-locked host memory benefits<br/>• Portable and write-combining memory<br/>• Mapped memory techniques<br/>• Thread coarsening optimization strategies | [![View on GitHub](https://img.shields.io/badge/GitHub-Thread_Coarsening_MatMul-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20009_Thread%20Coarsening) |
| **010** | 🔄 **Memory Synchronization Domains** | • Memory fence interference handling<br/>• Traffic isolation with domains<br/>• Domain usage in CUDA<br/>• Introduction to Triton programming | [![View on GitHub](https://img.shields.io/badge/GitHub-Vector_Addition_Triton-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20010_Memory%20Synchronization%20Domains) |
| **011** | ⚙️ **Asynchronous & Concurrent Execution** | • Vector Hadamard Product<br/>• Concurrent execution between host and device<br/>• Concurrent kernel execution<br/>• Overlap of data transfer and kernel execution<br/>• Concurrent data transfers<br/>• CUDA streams<br/>• Stream synchronization<br/>• Host functions (callbacks)<br/>• Stream priorities<br/>• Programmatic dependent launch | [![View on GitHub](https://img.shields.io/badge/GitHub-Vector_Hadamard_Product-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20011_Asynchronous%20Concurrent%20Execution/vectorHadamard.cu) |
| **012** | 🖥️ **Multi-Device System** | • Device enumeration and selection<br/>• Stream and event behavior<br/>• Peer-to-peer memory access and copy<br/>• Unified Virtual Address Space<br/>• Interprocess communication (IPC)<br/>• Error checking in CUDA | [![View on GitHub](https://img.shields.io/badge/GitHub-Vector_Dot_Product_Atomic-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20012_Multi-Device%20System) |
| **013** | 🔄 **CUDA Versioning & Compatibility** | • CUDA version compatibility rules<br/>• Mix-and-match versioning between driver and runtime<br/>• Compute mode settings and switching<br/>• Understanding compatibility modes<br/>• Naive Softmax implementation | [![View on GitHub](https://img.shields.io/badge/GitHub-Naive_Softmax-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20013_CUDA%20Versioning_Compatibility%20and%20Modes) |
| **014** | ⚙️ **Shared Memory Softmax Implementation** | • Implemented Softmax using shared memory in CUDA<br/>• Hardware Implementation <br/>• SIMT Architecture<br/>• Hardware Multithreading | [![View on GitHub](https://img.shields.io/badge/GitHub-SoftMax_(Shared)-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20014_Hardware%20Implimentation/shared_softmax.cu) |
| **015** | 🚀 **Vectorized Softmax with Shared Memory & Optimizations** | • Implemented Softmax using vectorized memory access (`float4`) and shared memory<br/>• Thread Synchronization Strategy<br/>• CUDA Occupancy APIs & Concurrent Kernel Execution<br/>• Latency Hiding & Resource Impact on Occupancy | [![View on GitHub](https://img.shields.io/badge/GitHub-SoftMax_(Vectorized)-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20015_Maximize%20Utilization) |
| **016** | 💾 **Maximize Memory Throughput** | • Implemented Softmax with coalesced memory access<br/>• Minimized Host–Device data transfers<br/>• Optimized global vs shared memory usage<br/>• Shared memory access patterns | [![View on GitHub](https://img.shields.io/badge/GitHub-SoftMax_(Coalesced)-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20016_Maximize%20Memory%20Throughput) |
| **017** | 🚀 **Maximize Instruction Throughput** | • Implemented 1D convolution<br/>• Minimized low-throughput instructions<br/>• Reduced divergent warps<br/>• Lowered total instruction count | [![View on GitHub](https://img.shields.io/badge/GitHub-1D_Convolution-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20017_Maximize%20Instruction%20Throughput/1D_convolution.cu) |
| **018** | 💡 **Performance Consideration** | • Implemented Partial Sum with Reduction<br/>• Improved global memory bandwidth usage<br/>• Applied dynamic SM resource partitioning<br/>• Added data prefetching, instruction mix tuning<br/>• Optimized thread granularity | [![View on GitHub](https://img.shields.io/badge/GitHub-Partial_Sum_with_Reduction-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20018_Minimize%20Memory%20Thrashing%20AND%20Performance%20Consideration) |
| **019** | 🧠 **OS Introduction & Process Abstraction** | • Implemented improved partial sum using bitwise operations<br/>• Studied OS abstractions | [![View on GitHub](https://img.shields.io/badge/GitHub-Process_Abstraction-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20019_OS%20Introduction%20and%20Process%20Abstraction) |
| **020** | 🔧 **Warp Shuffling & System Calls** | • Improved partial sum using warp shuffling<br/>• Integrated atomic reduction<br/>• Explored system calls for process management | [![View on GitHub](https://img.shields.io/badge/GitHub-Warp_Shuffle_Sum-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20020_System%20calls%20for%20process%20Management/warpShuffleSum.cu) |
| **021** | 🤝 **Cooperative Groups** | • Partial sum using thread-level cooperative groups<br/>• Leveraged coalesced groups and atomic aggregation | [![View on GitHub](https://img.shields.io/badge/GitHub-Coop_Group_Sum-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/tree/main/Day%20021_Process%20Execution%20Mechanism) |
| **022** | 📚 **Scheduling & IPC** | • Grid-level cooperative group for partial sum<br/>• Studied scheduling policies and IPC mechanisms | [![View on GitHub](https://img.shields.io/badge/GitHub-Grid_Coop_Sum-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20022_Scheduling%20Polices%20and%20IPC/gridLevelSum.cu) |
| **023** | 🔄 **Pipelining & Memory Patterns** | • Implemented pipelined vector addition<br/>• Optimized memory access patterns<br/>• Transitioned from synchronous to asynchronous design<br/>• Debugged memory management issues | [![View on GitHub](https://img.shields.io/badge/GitHub-Pipelined_Vector_Add-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20023_Intro%20to%20Virtual%20Memory/vector_add_pipelined.cu) |
| **024** | 🧮 **Matrix Multiplication (Cooperative Groups)** | • Matrix multiplication using block-level cooperative groups | [![View on GitHub](https://img.shields.io/badge/GitHub-MatMul_Coop_Groups-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20024_Mechanism%20of%20Adress%20Translation/cooperative_matmul.cu) |
| **025** | ⚡ **Async Data Movement & Paging** | • Matrix multiplication with asynchronous data movement<br/>• Explored paging and demand paging | [![View on GitHub](https://img.shields.io/badge/GitHub-Async_MatMul-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20025_Paging%20and%20Demand%20Paging/asynch_matmul.cu) |
| **026** | 🧹 **Memory Management** | • Memory allocation and free space handling<br/>• Optimized vector addition implementation | [![View on GitHub](https://img.shields.io/badge/GitHub-VecAdd_Optimized-black?logo=github&style=flat-square)](https://github.com/bikrammajhi/100-days-of-GPU/blob/main/Day%20026_Memory%20Allocation%20and%20Free%20space%20management/VecAdd.cu) |



# References:
[![GitHub](https://img.shields.io/badge/GitHub-LeetCUDA-181717?logo=github)](https://github.com/xlite-dev/LeetCUDA) [![GitHub](https://img.shields.io/badge/GitHub-CUDA%20Optimizations-181717?logo=github)](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) [![Awesome](https://img.shields.io/badge/Awesome-CUDA%20&%20HPC-ff6b6b?logo=awesome-lists)](https://github.com/coderonion/awesome-cuda-and-hpc) [![Awesome](https://img.shields.io/badge/Awesome-DiT%20Inference-4ecdc4?logo=awesome-lists)](https://github.com/xlite-dev/Awesome-DiT-Inference) [![GitHub](https://img.shields.io/badge/GitHub-Stable%20Diffusion%20C++-181717?logo=github)](https://github.com/leejet/stable-diffusion.cpp) 

[![Blog](https://img.shields.io/badge/Blog-Tensor%20Core%20MatMul-e91e63?logo=hashnode)](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) 
[![Blog](https://img.shields.io/badge/Blog-CUDA%20Basics-2196f3?logo=medium)](https://tinkerd.net/blog/machine-learning/cuda-basics/#wrapping-up)
[![Blog](https://img.shields.io/badge/Blog-MMA%20MatMul-2196f3?logo=medium)](https://www.spatters.ca/mma-matmul) 
[![Blog](https://img.shields.io/badge/Blog-CUDA%20MatMul%20Optimization-00d4aa?logo=dev.to)](https://siboehm.com/articles/22/CUDA-MMM) 
[![Substack](https://img.shields.io/badge/Substack-Outperforming%20cuBLAS-ff6b35?logo=substack)](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) 
[![YouTube](https://img.shields.io/badge/YouTube-Outperforming%20cuBLAS-FF0000?logo=youtube)](https://www.youtube.com/watch?v=ErTmTCRP1_U) 
[![GitHub](https://img.shields.io/badge/GitHub-Fast%20MatMuls%20%7C%20651%20TFLOPS%20on%20H100-181717?logo=github)](https://github.com/alexarmbr/fast-matmuls)

[![Blog](https://img.shields.io/badge/Blog-Optimizing%20LayerNorm-2196f3?logo=medium)](https://aryagxr.com/blogs/cuda-optimizing-layernorm)
[![Blog](https://img.shields.io/badge/Fast%20GPU%20Matrix%20Multiplication-2196f3?logo=medium)](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html)

[![GitHub](https://img.shields.io/badge/GitHub-CUDA%20HGEMM-181717?logo=github)](https://github.com/Bruce-Lee-LY/cuda_hgemm)
[![Blog](https://img.shields.io/badge/Blog-Fabian%20Schuetze's%20GPU%20Articles-2196f3?logo=medium)](https://fabianschuetze.github.io/category/articles.html)


---

 




