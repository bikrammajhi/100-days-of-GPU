# Mathematical Questions for GPU/Parallel Computing Systems Engineer Interview

## Section 1: Memory Bandwidth and Throughput (Questions 1-10)

| Question | Description |
| - | - |
| 1 | A GPU has 12 memory channels, each operating at 2000 MHz with a 32-bit bus width. Calculate the theoretical peak memory bandwidth in GB/s. |
| 2 | Your kernel transfers 4GB of data in 25ms. What is the achieved memory bandwidth? If the theoretical peak is 900 GB/s, what is your bandwidth utilization percentage? |
| 3 | A matrix multiplication kernel reads two NxN float32 matrices and writes one NxN result matrix. For N=4096, calculate the total memory traffic in GB and the arithmetic intensity (FLOPS per byte). |
| 4 | Given a GPU with 80 SMs, each with 128 KB shared memory, and global memory bandwidth of 2 TB/s. If your kernel achieves 70% of peak bandwidth, how long does it take to transfer 16 GB of data? |
| 5 | Calculate the memory bandwidth required for a convolution operation with input size 224x224x3, filter size 7x7, stride 2, padding 3, and 64 output channels, processing 32 images per batch. |
| 6 | A GPU kernel processes 1M elements, each requiring 3 memory accesses (2 reads, 1 write) of 4 bytes each. If the kernel runs in 0.5ms, what memory bandwidth is achieved? |
| 7 | Given memory latency of 400 cycles and memory bandwidth of 1 TB/s, calculate the number of concurrent memory requests needed to hide latency for a 1.5 GHz GPU. |
| 8 | A reduction operation sums 64M float32 values. Calculate the minimum memory bandwidth required to achieve 100 GFLOPS throughput. |
| 9 | Your kernel achieves 650 GB/s on a GPU with theoretical peak of 900 GB/s. The kernel has arithmetic intensity of 0.5 FLOPS/byte. What is the achieved compute throughput in TFLOPS? |
| 10 | Calculate the bank conflicts for accessing a shared memory array of 1024 float32 elements when 32 threads access elements at indices: 0, 32, 64, 96, ..., 992. |

## Section 2: Compute Performance and FLOPS (Questions 11-20)

| Question | Description |
| - | - |
| 11 | A GPU has 10,240 CUDA cores running at 1.4 GHz. Calculate the theoretical peak single-precision FLOPS assuming one FMA operation per core per clock. |
| 12 | Your matrix multiplication kernel achieves 85% of peak performance on a GPU with 125 TFLOPS peak. The matrices are 8192x8192. Calculate the actual GFLOPS achieved and execution time. |
| 13 | A convolution layer processes 64 images of size 112x112x256 with 512 filters of size 3x3. Calculate the total FLOPS required. |
| 14 | Given a GPU with 432 Tensor Cores, each capable of 256 FLOPS per clock at 1.35 GHz for mixed precision (FP16). Calculate peak Tensor FLOPS. |
| 15 | Your kernel performs 2.5 trillion operations in 12ms. Calculate the achieved TFLOPS. If this represents 78% efficiency, what is the theoretical peak? |
| 16 | A transformer attention mechanism processes sequence length 2048 with hidden dimension 1024 and 16 attention heads. Calculate FLOPS for the attention computation (Q×K^T, softmax, ×V). |
| 17 | Calculate the roofline model intersection point for a GPU with 150 TFLOPS peak compute and 2 TB/s memory bandwidth. |
| 18 | A GPU kernel achieves 45 TFLOPS with arithmetic intensity of 12 FLOPS/byte. What memory bandwidth is being utilized? |
| 19 | Your neural network training step requires 847 TFLOPS and completes in 28ms. Calculate the average compute utilization if peak GPU performance is 125 TFLOPS. |
| 20 | A sparse matrix-vector multiplication has 5% non-zero elements in a 50,000×50,000 matrix. Calculate the FLOPS and compare with dense matrix-vector multiplication. |

## Section 3: Parallelization and Thread Management (Questions 21-30)

| Question | Description |
| - | - |
| 21 | A GPU has 108 SMs, each with 2048 maximum threads. Your kernel uses 256 threads per block. Calculate maximum theoretical occupancy and number of active blocks. |
| 22 | Your kernel uses 48 KB shared memory per block. The GPU has 96 KB shared memory per SM. How many blocks can run concurrently per SM? |
| 23 | Calculate the number of warps needed to hide a memory latency of 300 cycles, assuming arithmetic instructions take 4 cycles and memory instructions comprise 20% of the workload. |
| 24 | A parallel reduction of 16M elements uses 1024 threads per block. Calculate the number of reduction steps needed and total thread-block launches required. |
| 25 | Your kernel launches 65,536 thread blocks with 512 threads each on a GPU with 80 SMs. Calculate average blocks per SM and total active threads. |
| 26 | A stencil computation on a 4096×4096 grid uses 16×16 thread blocks with 2-element halo. Calculate the shared memory requirement per block in bytes (float32). |
| 27 | Calculate the load imbalance when processing an irregular graph with vertex degrees following power-law distribution: 80% vertices have degree 1-10, 20% have degree 100-1000. |
| 28 | A parallel prefix sum (scan) of 1M elements uses the work-efficient algorithm. Calculate total work complexity and step complexity. |
| 29 | Your CUDA kernel has register usage of 32 registers per thread. The GPU has 65,536 registers per SM. Calculate maximum threads per SM due to register pressure. |
| 30 | A 2D FFT of size 2048×2048 is decomposed into row-wise and column-wise 1D FFTs. Calculate the total number of 1D FFT operations and memory transposes required. |

## Section 4: Memory Hierarchy and Cache Performance (Questions 31-40)

| Question | Description |
| - | - |
| 31 | An L2 cache has 6MB capacity with 128-byte cache lines. Calculate the number of cache lines and address bits needed for direct-mapped cache. |
| 32 | Your kernel has 90% L1 cache hit rate with 128-cycle miss penalty and 4-cycle hit latency. Calculate average memory access time. |
| 33 | A texture cache has 48KB capacity per SM with 32-byte cache lines. Calculate cache associativity if there are 6 sets per way. |
| 34 | Calculate the memory coalescing efficiency when 32 threads access a structure of arrays with 12-byte elements (3 float32 values per element). |
| 35 | Your application has working set of 2GB on a GPU with 48MB L2 cache. Estimate cache miss rate assuming random access pattern. |
| 36 | A kernel accesses a 2D array with stride pattern. Array size is 4096×4096 float32, accessed with stride (1, 4096). Calculate TLB misses assuming 4KB pages and 512-entry TLB. |
| 37 | Calculate the bandwidth amplification factor when accessing a row-major matrix column-wise, given 128-byte cache lines and 4-byte elements. |
| 38 | Shared memory has 32 banks, each 4 bytes wide. Calculate conflicts when 32 threads access a 32×32 float32 matrix in column-major order. |
| 39 | Your kernel achieves 75% of peak bandwidth with 95% cache hit rate. If cache miss penalty is 400 cycles, what would be the bandwidth with 85% hit rate? |
| 40 | A constant memory access pattern has 30% temporal locality within a 64KB window. Calculate expected cache performance with 16KB L1 constant cache. |

## Section 5: Energy Efficiency and Power (Questions 41-50)

| Question | Description |
| - | - |
| 41 | A GPU consumes 300W at 80% utilization running at 1.4 GHz. Calculate power per active core assuming 5,120 cores. |
| 42 | Your kernel achieves 50 TFLOPS while consuming 250W. Calculate energy efficiency in GFLOPS/W and compare with CPU at 2 TFLOPS and 150W. |
| 43 | Dynamic voltage and frequency scaling (DVFS) reduces frequency from 1.5 GHz to 1.2 GHz, reducing power by 30%. Calculate the energy consumption change for a fixed workload. |
| 44 | A training job requires 1000 TFLOP-hours. Calculate energy consumption on a GPU cluster with average efficiency 45 TFLOPS at 400W per GPU. |
| 45 | Memory accesses consume 50% of total GPU power. If memory bandwidth utilization increases from 60% to 80%, estimate the power increase. |
| 46 | Calculate the performance-per-watt improvement when migrating from FP32 to FP16 operations, assuming 2x throughput increase and 15% power reduction. |
| 47 | A GPU's power consumption follows P = P_base + α × f³, where P_base = 50W, α = 2×10⁻⁹, and f is frequency in Hz. Calculate power at 1.2 GHz and 1.6 GHz. |
| 48 | Your distributed training setup uses 64 GPUs, each consuming 350W at 70% average utilization over 8 hours. Calculate total energy consumption in kWh. |
| 49 | Tensor Core operations consume 0.3 pJ per operation while regular cores consume 1.2 pJ per operation. Calculate energy savings when 60% of operations use Tensor Cores. |
| 50 | A GPU cluster has PUE (Power Usage Effectiveness) of 1.4. If GPU power consumption is 12 MW, calculate total facility power consumption and cooling overhead. |

## Answer Key and Calculation Methods

### Sample Solutions for Key Questions:

**Question 1 Solution:**
- 12 channels × 2000 MHz × 32 bits × 2 (DDR) = 1,536,000 Mb/s
- Convert to GB/s: 1,536,000 ÷ 8 ÷ 1000 = 192 GB/s

**Question 11 Solution:**
- 10,240 cores × 1.4 GHz × 2 (FMA = 2 operations) = 28.672 TFLOPS

**Question 13 Solution:**
- Output size: 64 × 112 × 112 × 512 = 410,550,272 elements
- FLOPS per output: 256 × 3 × 3 = 2,304
- Total FLOPS: 410,550,272 × 2,304 = 946,307,026,944 ≈ 946 GFLOPS

**Question 21 Solution:**
- Total threads: 108 × 2048 = 221,184
- Blocks of 256 threads: 221,184 ÷ 256 = 864 blocks maximum
- Theoretical occupancy: 100% (if no other limitations)

**Question 31 Solution:**
- Cache lines: 6MB ÷ 128B = 49,152 lines
- Address bits for direct-mapped: log₂(49,152) = 16 bits (approximately)

## Section 6: CUDA Thread Hierarchy and Indexing (Questions 1-10)

| Question | Description |
| - | - |
| 1 | A CUDA kernel is launched with grid dimensions (128, 64, 1) and block dimensions (16, 8, 4). Calculate the total number of threads and the global thread ID for a thread at block (10, 5, 0) and local thread (3, 2, 1). |
| 2 | For a 2D matrix of size 4096×2048, design block and grid dimensions for optimal memory coalescing. Calculate the total number of blocks needed if each block processes a 16×16 tile. |
| 3 | A 3D convolution kernel processes a volume of size 256×256×128. Using block size (8, 8, 8), calculate grid dimensions and verify that all volume elements are covered. |
| 4 | Calculate the warp ID and lane ID for thread with threadIdx = (5, 3, 2) in a block with blockDim = (16, 8, 4). |
| 5 | A reduction kernel processes 1,048,576 elements using 1024 threads per block. Calculate the number of blocks needed and the reduction steps within each block. |
| 6 | For a tiled matrix multiplication with tile size 32×32, calculate the number of shared memory loads per thread and total shared memory accesses for multiplying 2048×2048 matrices. |
| 7 | A stencil computation uses block size (16, 16) with halo width 2. Calculate the shared memory size needed and the number of global memory loads per block for a 2D 5-point stencil. |
| 8 | Calculate the stride pattern for accessing elements in a 3D array A[D][H][W] when threads are organized in a 2D grid processing the H×W dimensions. |
| 9 | A CUDA kernel uses dynamic parallelism to launch child kernels. Parent grid has 256 blocks, each launching 4 child kernels with 128 threads each. Calculate total thread count across all kernel launches. |
| 10 | For a scan (prefix sum) operation on 65,536 elements using Hillis-Steele algorithm with 512 threads per block, calculate the number of synchronization steps and total memory accesses. |

## Section 7: CUDA Memory Management and Bandwidth (Questions 11-20)

| Question | Description |
| - | - |
| 11 | A CUDA kernel transfers 2GB of data from global to shared memory across 1024 blocks, each with 48KB shared memory. Calculate the memory bandwidth if the transfer takes 15ms. |
| 12 | Calculate the memory coalescing efficiency when 32 threads in a warp access float4 elements starting at addresses: 0, 16, 32, 48, ..., 496 bytes. |
| 13 | A texture memory access pattern has 85% cache hit rate with 4-cycle hit latency and 200-cycle miss penalty. Calculate the average memory access time and compare with global memory at 400 cycles. |
| 14 | For a CUDA kernel with 65,536 threads accessing a 1GB array randomly, estimate the L2 cache miss rate assuming 6MB L2 cache and 128-byte cache lines. |
| 15 | Calculate the bank conflicts in shared memory when 32 threads access elements at indices: 0, 1, 2, ..., 31 in a float array vs. accessing every 32nd element. |
| 16 | A cudaMemcpy operation transfers 4GB from host to device. The PCIe 3.0 x16 link has 16 GB/s theoretical bandwidth. Calculate transfer time assuming 80% efficiency. |
| 17 | Unified Memory usage shows 70% GPU access and 30% CPU access for a 8GB dataset. Calculate the memory migration overhead if each page fault costs 10μs and page size is 2MB. |
| 18 | A CUDA kernel uses constant memory for a 32KB lookup table accessed by all threads. Calculate the broadcast efficiency when 80% of warps access the same constant value simultaneously. |
| 19 | Calculate the memory throughput for a matrix transpose kernel that processes 8192×8192 float32 matrix in 12ms, including both read and write operations. |
| 20 | A CUDA application allocates 16GB GPU memory with 2GB for constant data, 4GB for textures, and 10GB for global arrays. Calculate memory utilization percentage on a GPU with 24GB memory. |

## Section 8: CUDA Kernel Performance Analysis (Questions 21-30)

| Question | Description |
| - | - |
| 21 | A CUDA kernel achieves 45% occupancy with 256 threads per block on a GPU with 2048 max threads per SM. Calculate the number of active blocks per SM and identify the limiting factor. |
| 22 | Your matrix multiplication kernel processes 4096×4096×4096 matrices in 25ms. Calculate the achieved GFLOPS and compare with theoretical peak of 125 TFLOPS. |
| 23 | A CUDA kernel uses 32 registers per thread. The GPU has 65,536 registers per SM. Calculate maximum theoretical occupancy and compare with shared memory limitation of 96KB per block with 48KB usage. |
| 24 | Calculate the warp execution efficiency when a kernel has 30% divergent branches, assuming divergent warps execute both paths. |
| 25 | A reduction kernel reduces 16M float32 values to a single sum. Using tree reduction with 1024 threads per block, calculate the number of kernel launches needed and total execution steps. |
| 26 | Your CUDA kernel launches 8192 blocks with 512 threads each on a GPU with 108 SMs. Calculate the average number of blocks per SM and estimate load balancing efficiency. |
| 27 | A convolution kernel achieves 2.5 TFLOPS on input size 224×224×256 with 512 filters of size 3×3. Calculate the kernel execution time and memory bandwidth utilization if peak bandwidth is 900 GB/s. |
| 28 | Calculate the instruction throughput for a kernel with 40% arithmetic instructions, 35% memory instructions, and 25% control instructions, running at 65% occupancy on a 1.4 GHz GPU. |
| 29 | A CUDA kernel processes sparse matrices with 5% density. Compare the arithmetic intensity with dense matrix operations for a 10,000×10,000 matrix multiplication. |
| 30 | Your kernel shows 15% warp stall due to memory dependencies. Calculate the performance impact and potential speedup if memory latency is reduced by 30%. |

## Section 9: CUDA Optimization Calculations (Questions 31-40)

| Question | Description |
| - | - |
| 31 | A tiled matrix multiplication uses shared memory tiles of size 32×32. Calculate the shared memory bank conflicts when accessing tiles in column-major order vs. row-major order. |
| 32 | Calculate the optimal block size for a kernel that processes 1D arrays, given register usage of 24 per thread, shared memory usage of 12KB per block, and target occupancy of 75%. |
| 33 | A CUDA kernel performs loop unrolling with factor 4. If the original kernel had 80% arithmetic intensity, calculate the new arithmetic intensity and expected performance improvement. |
| 34 | Calculate the memory access pattern efficiency for a 2D convolution with input size 512×512, filter size 5×5, stride 2, and padding 2, using 16×16 thread blocks. |
| 35 | A vector addition kernel achieves 600 GB/s memory bandwidth. Calculate the performance improvement when using float2 vs. float4 vector loads, assuming memory latency remains constant. |
| 36 | Your CUDA kernel uses atomic operations with 25% contention rate. Calculate the serialization overhead and potential speedup using reduction patterns instead. |
| 37 | Calculate the shared memory padding needed to avoid bank conflicts when storing a 33×33 float32 tile in shared memory with 32 banks. |
| 38 | A CUDA kernel uses warp shuffle operations for reduction within warps. Calculate the number of shuffle operations needed to reduce 32 values to 1, and compare with shared memory approach. |
| 39 | Calculate the instruction-level parallelism (ILP) improvement when unrolling a loop 8 times in a kernel with 60% independent instructions. |
| 40 | A CUDA application uses streams with 4 concurrent kernels. Each kernel requires 25% of GPU resources. Calculate the theoretical speedup and identify potential bottlenecks. |

## Section 10: Advanced CUDA Mathematics (Questions 41-50)

| Question | Description |
| - | - |
| 41 | A CUDA cooperative groups implementation uses 4 thread blocks working together. Calculate the synchronization overhead when each block waits for others 15% of execution time. |
| 42 | Calculate the memory divergence penalty when threads in a warp access memory locations with stride pattern 1, 3, 5, 7, ..., 63 (odd numbers only). |
| 43 | A CUDA kernel uses half-precision (FP16) operations achieving 2x throughput over FP32. Calculate the effective performance improvement considering 10% accuracy loss requiring 5% additional computation. |
| 44 | Your multi-GPU setup uses 8 GPUs with NVLink bandwidth of 600 GB/s between pairs. Calculate the all-reduce communication time for a 2GB gradient tensor using ring algorithm. |
| 45 | A CUDA kernel implements double buffering with 2MB buffers. Calculate the overlap efficiency when computation takes 8ms and memory transfer takes 3ms per buffer. |
| 46 | Calculate the performance impact of using CUDA dynamic parallelism when parent kernels have 20% overhead and child kernels achieve 85% of static kernel performance. |
| 47 | A CUDA graph execution reduces kernel launch overhead from 20μs to 2μs per kernel. For a workload with 1000 small kernels, calculate the performance improvement percentage. |
| 48 | Your CUDA application uses tensor cores with mixed precision. Calculate the effective throughput when 70% of operations use tensor cores (256 ops/cycle) and 30% use regular cores (1 op/cycle) at 1.4 GHz. |
| 49 | A CUDA kernel uses persistent threads processing a queue of 100,000 work items. With 1024 persistent threads, calculate the load balancing efficiency when work items have 20% variance in execution time. |
| 50 | Calculate the memory bandwidth utilization for a CUDA kernel that performs sparse matrix-vector multiplication with CSR format, processing a 50,000×50,000 matrix with 2% sparsity in 8ms. |

## Detailed Solution Examples

### Question 1 Solution:
```
Total threads = gridDim.x × gridDim.y × gridDim.z × blockDim.x × blockDim.y × blockDim.z
= 128 × 64 × 1 × 16 × 8 × 4 = 524,288 threads

Global thread ID calculation:
blockId = blockIdx.z × (gridDim.x × gridDim.y) + blockIdx.y × gridDim.x + blockIdx.x
blockId = 0 × (128 × 64) + 5 × 128 + 10 = 650

threadId = threadIdx.z × (blockDim.x × blockDim.y) + threadIdx.y × blockDim.x + threadIdx.x
threadId = 1 × (16 × 8) + 2 × 16 + 3 = 163

globalThreadId = blockId × (blockDim.x × blockDim.y × blockDim.z) + threadId
globalThreadId = 650 × 512 + 163 = 332,963
```

### Question 22 Solution:
```
Matrix multiplication FLOPS = 2 × N³ = 2 × 4096³ = 137,438,953,472 FLOPS
Achieved GFLOPS = 137.44 × 10⁹ / 0.025s = 5,497.6 GFLOPS = 5.5 TFLOPS
Efficiency = 5.5 TFLOPS / 125 TFLOPS = 4.4%
```

### Question 31 Solution:
```
Row-major access (no conflicts): threads 0-31 access consecutive elements
Column-major access: thread i accesses element (i, 0), (i, 1), etc.
Bank conflicts = 32 (all threads access same bank initially)
Conflict factor = 32-way bank conflict = 32× slower than conflict-free
```

### Question 48 Solution:
```
Tensor core throughput = 0.7 × 256 × 1.4 GHz = 251.2 GOPS
Regular core throughput = 0.3 × 1 × 1.4 GHz = 0.42 GOPS
Total effective throughput = 251.2 + 0.42 = 251.62 GOPS
```

# Advanced CUDA Programming Mathematics - Memory Types & Optimization Patterns

## Section 11: Constant Memory Mathematics (Questions 1-12)

| Question | Description |
| - | - |
| 1 | A CUDA kernel accesses a 48KB lookup table stored in constant memory. With 32 threads per warp and 80% probability that all threads access the same constant value, calculate the effective memory bandwidth if constant cache delivers 1 value per cycle at 1.4 GHz. |
| 2 | Calculate the broadcast efficiency for constant memory access when 12 out of 32 threads in a warp access the same constant value, while others access different values. Compare with the case where all 32 threads access unique values. |
| 3 | A neural network uses constant memory for weight matrices totaling 64KB across 4 layers. If each layer processes 128 feature maps with 85% cache hit rate, calculate the constant memory traffic reduction compared to global memory access. |
| 4 | Design the optimal constant memory layout for a 3D convolution kernel with filters of size 3×3×3×64. Calculate the memory access pattern efficiency when processing input volumes of 128×128×32. |
| 5 | A CUDA kernel performs matrix-vector multiplication where the vector (size 4096) is stored in constant memory. Calculate the memory bandwidth utilization when 1024 warps access this vector with 90% temporal locality. |
| 6 | Calculate the constant memory bank conflict equivalent when 16 threads in a half-warp access constant memory addresses: 0x1000, 0x1004, 0x1008, ..., 0x103C with a 8KB constant cache having 8-way set associativity. |
| 7 | A image processing kernel uses constant memory for a 5×5 convolution mask. With 65,536 threads processing a 2048×2048 image, calculate the constant memory access efficiency and cache hit rate assuming perfect spatial locality. |
| 8 | Compare the energy consumption when accessing a 16KB parameter array from constant memory vs. global memory, given constant memory consumes 0.3pJ per access and global memory consumes 1.2pJ per access, with 2M accesses total. |
| 9 | A CUDA kernel uses constant memory for storing sine/cosine lookup tables (8KB each). Calculate the performance improvement when 70% of trigonometric operations use table lookup vs. computing sin/cos directly (assuming 25 cycles for computation, 4 cycles for constant memory access). |
| 10 | Calculate the optimal constant memory allocation strategy for a multi-kernel application where Kernel A needs 32KB constants, Kernel B needs 48KB, and Kernel C needs 16KB, given 64KB total constant memory. |
| 11 | A CUDA application uses constant memory for storing transformation matrices (4×4 float matrices). With 1000 objects being transformed and 80% matrix reuse rate, calculate the memory traffic reduction compared to storing matrices in global memory. |
| 12 | Calculate the constant memory access serialization overhead when threads in a warp access 8 different constant memory locations simultaneously, assuming each serialized access takes 200 cycles. |

## Section 12: Texture Memory Mathematics (Questions 13-24)

| Question | Description |
| - | - |
| 13 | A texture memory access pattern shows 75% spatial locality and 60% temporal locality for a 2D texture of size 2048×2048×4 bytes. Calculate the effective texture cache hit rate with 48KB texture cache per SM and 128-byte cache lines. |
| 14 | Calculate the texture memory bandwidth utilization for a bilinear interpolation kernel processing 4M texture coordinates, where each access requires 4 texture fetches and the kernel completes in 8ms. |
| 15 | A CUDA kernel performs 3D texture sampling with trilinear interpolation. For a 256×256×128 volume, calculate the total number of texture fetches required and the memory bandwidth if processing takes 15ms. |
| 16 | Compare the memory access efficiency between texture memory and global memory for a stencil computation with irregular access patterns, given texture cache hit rate of 85% vs. L2 cache hit rate of 45%. |
| 17 | A image convolution kernel uses texture memory for input images (1920×1080×3 bytes) with wrap addressing mode. Calculate the effective memory access pattern when applying a 7×7 filter with 2-pixel border handling. |
| 18 | Calculate the texture memory overhead for storing a sparse 3D volume (10% occupancy) in texture memory vs. using a compressed sparse representation in global memory. |
| 19 | A CUDA kernel performs nearest-neighbor interpolation on a 4096×4096 texture using normalized coordinates. Calculate the coordinate quantization error and its impact on cache performance. |
| 20 | Compare the energy efficiency of texture memory vs. global memory for accessing a 3D lookup table, given texture memory: 0.4pJ per access, global memory: 1.1pJ per access, with texture providing 2.3x cache hit rate improvement. |
| 21 | A volume rendering kernel accesses a 512×512×512 3D texture with ray-casting algorithm. Calculate the texture memory bandwidth requirements for rendering a 1024×1024 image with average ray length of 200 samples. |
| 22 | Calculate the texture memory bank utilization when 32 threads access 2D texture coordinates with bilinear filtering, where coordinates follow a Morton (Z-order) curve pattern. |
| 23 | A CUDA application uses layered textures with 16 layers of 1024×1024×4 bytes each. Calculate the texture memory allocation overhead and access pattern efficiency when processing all layers simultaneously. |
| 24 | Calculate the performance impact of texture memory address translation overhead when accessing a 3D texture with non-unit stride patterns, assuming 5% additional latency per address translation. |

## Section 13: Shared Memory Optimization Patterns (Questions 25-36)

| Question | Description |
| - | - |
| 25 | A matrix transpose kernel uses shared memory tiles of 32×33 (padded) to avoid bank conflicts. Calculate the memory efficiency and overhead of padding for transposing a 4096×4096 matrix. |
| 26 | Calculate the optimal shared memory allocation for a 2D convolution kernel with 16×16 thread blocks, 5×5 filters, and input tiles requiring 2-element halo. Compare memory usage with and without double buffering. |
| 27 | A reduction kernel uses shared memory for tree reduction within thread blocks. For 1024 threads per block reducing float32 values, calculate the number of shared memory banks utilized and potential bank conflicts during each reduction step. |
| 28 | Calculate the shared memory broadcast efficiency for a prefix scan algorithm when threads access shared memory in powers-of-2 stride patterns: 1, 2, 4, 8, 16, 32. |
| 29 | A CUDA kernel implements cooperative groups using shared memory for cross-block communication. With 8 blocks cooperating and 48KB shared memory per block, calculate the communication overhead for synchronizing 2KB of data per block. |
| 30 | Calculate the shared memory utilization efficiency for a sparse matrix-vector multiplication kernel that uses shared memory to cache vector elements, with 20% reuse rate and 75% cache hit probability. |
| 31 | A tiled matrix multiplication kernel uses double buffering in shared memory. Calculate the memory bandwidth hidden by computation overlap when computation takes 12μs and memory transfer takes 8μs per tile. |
| 32 | Calculate the bank conflict resolution overhead for a 2D FFT kernel that accesses shared memory in bit-reversed order patterns, with 32 banks and 1024-point FFT per thread block. |
| 33 | A CUDA kernel uses shared memory for histogram computation with atomic operations. Calculate the contention rate and serialization overhead when 256 threads update a 64-bin histogram with uniform distribution. |
| 34 | Calculate the shared memory allocation strategy for a multi-stage pipeline kernel with 3 stages requiring 16KB, 24KB, and 12KB respectively, given 96KB shared memory per SM. |
| 35 | A shared memory optimization uses memory coalescing patterns to improve bank utilization. Calculate the effective bandwidth improvement when changing from strided access (stride=32) to sequential access patterns. |
| 36 | Calculate the shared memory fragmentation overhead for a kernel that dynamically allocates shared memory arrays of varying sizes: 25% use 2KB, 50% use 4KB, 25% use 8KB. |

## Section 14: Warp-Level Optimization Mathematics (Questions 37-48)

| Question | Description |
| - | - |
| 37 | Calculate the warp execution efficiency for a kernel with 40% divergent branches, where divergent paths have execution time ratio of 3:1, and 60% of warps take the longer path. |
| 38 | A warp shuffle reduction performs sum reduction of 32 float values. Calculate the total number of shuffle operations and compare the performance with shared memory reduction assuming shuffle: 1 cycle, shared memory: 4 cycles per access. |
| 39 | Calculate the warp occupancy impact when a kernel has 15% warp stalls due to memory dependencies, 25% due to execution dependencies, and 10% due to synchronization barriers. |
| 40 | A CUDA kernel uses warp voting functions (__ballot_sync) for conditional execution optimization. Calculate the performance improvement when 75% of warps can skip expensive computation blocks based on voting results. |
| 41 | Calculate the warp-level parallelism (WLP) for a kernel running 1536 warps on a GPU with 108 SMs, given average warp execution time of 120 cycles and memory latency of 400 cycles. |
| 42 | A cooperative groups implementation uses warp-level primitives for cross-warp communication. Calculate the synchronization overhead when 4 warps synchronize every 50 instructions with 5-cycle sync cost. |
| 43 | Calculate the instruction throughput improvement using warp intrinsics (__shfl_xor_sync) for butterfly reduction patterns vs. traditional shared memory approaches, given 32-element reduction per warp. |
| 44 | A CUDA kernel optimizes predication using warp mask operations. Calculate the performance gain when eliminating 30% of conditional branches through predicated execution, assuming branch penalty of 15 cycles. |
| 45 | Calculate the warp scheduling efficiency for a kernel with mixed instruction types: 40% arithmetic (4 cycles), 35% memory (variable latency), 25% control (2 cycles), running at 85% occupancy. |
| 46 | A warp-specialized optimization assigns different warps to different computation tasks. Calculate the load balancing efficiency when Task A (60% warps) takes 80 cycles and Task B (40% warps) takes 120 cycles. |
| 47 | Calculate the warp divergence penalty for a ray-tracing kernel where 25% of warps process primary rays (50 cycles), 45% process shadow rays (30 cycles), and 30% process reflection rays (80 cycles). |
| 48 | A CUDA kernel uses warp-synchronous programming patterns to eliminate explicit synchronization. Calculate the performance improvement when removing 8 __syncthreads() calls per thread block, each costing 25 cycles overhead. |

## Section 15: Advanced Memory Pattern Optimizations (Questions 49-60)

| Question | Description |
| - | - |
| 49 | Calculate the memory access pattern efficiency for a 3D stencil computation using space-filling curves (Morton order) vs. linear addressing, given L2 cache size of 6MB and 128-byte cache lines. |
| 50 | A CUDA kernel implements software-managed cache using shared memory for irregular memory access patterns. Calculate the effective hit rate and performance improvement with 32KB cache, 75% hit rate, vs. global memory with 400-cycle latency. |
| 51 | Calculate the memory coalescing efficiency for a sparse matrix operation using compressed sparse row (CSR) format, where row lengths vary from 1 to 1024 elements with power-law distribution. |
| 52 | A memory access optimization uses prefetching with 2-level hierarchy: L1 prefetch (32KB) and L2 prefetch (512KB). Calculate the hit rate improvement when prefetch accuracy is 80% and memory access has 60% spatial locality. |
| 53 | Calculate the memory bandwidth utilization for a GPU sort algorithm (radix sort) processing 64M 32-bit integers, considering multiple passes with different memory access patterns per pass. |
| 54 | A CUDA application uses memory pools to reduce allocation overhead. Calculate the memory utilization efficiency with 8 pools of sizes: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, given allocation pattern: 40% small (≤4KB), 35% medium (≤256KB), 25% large (>256KB). |
| 55 | Calculate the effective memory bandwidth for a graph algorithm with irregular access patterns, where 30% accesses are sequential, 45% are random within 4KB pages, and 25% are completely random. |
| 56 | A memory compression optimization achieves 2.5:1 compression ratio with 15% decompression overhead. Calculate the effective memory bandwidth improvement for a memory-bound kernel achieving 70% of peak bandwidth. |
| 57 | Calculate the memory access optimization benefit using register blocking technique for matrix operations, where register usage increases from 32 to 56 per thread but reduces memory traffic by 40%. |
| 58 | A CUDA kernel uses memory access pattern transformation from structure-of-arrays (SoA) to array-of-structures (AoS). Calculate the performance impact given 128-byte cache lines and 12-byte structure size. |
| 59 | Calculate the memory hierarchy optimization for a multi-level cache system: L1 (128KB, 85% hit), L2 (6MB, 75% hit), Memory (2TB/s bandwidth), given access latencies: L1=4 cycles, L2=50 cycles, Memory=400 cycles. |
| 60 | A CUDA application implements memory access scheduling to overlap computation and communication. Calculate the effective bandwidth utilization when computation hides 80% of memory latency through double buffering and prefetching techniques. |

## Detailed Solution Examples

### Question 1 - Constant Memory Broadcast:
```
Scenario: 80% same access, 20% different access
Same access bandwidth = 0.8 × 32 threads × 1 access × 1.4 GHz = 35.84 GB/s (broadcast)
Different access bandwidth = 0.2 × 32 threads × 1 access × 1.4 GHz = 8.96 GB/s (serialized)
Effective bandwidth = 35.84 + 8.96 = 44.8 GB/s
```

### Question 25 - Shared Memory Padding:
```
Original tile: 32×32 = 1024 elements = 4KB
Padded tile: 32×33 = 1056 elements = 4.224KB
Overhead = (4.224 - 4.0) / 4.0 = 5.6%
Bank conflict elimination benefit >> 5.6% overhead
```

### Question 38 - Warp Shuffle vs Shared Memory:
```
Shuffle reduction: log₂(32) = 5 steps × 1 cycle = 5 cycles
Shared memory reduction: 32 writes + 5 reduction steps × 4 cycles = 32 + 20 = 52 cycles
Speedup = 52/5 = 10.4× faster
```

### Question 51 - CSR Memory Coalescing:
```
Power-law row lengths: 80% rows have 1-10 elements, 20% have 100-1024
Coalescing efficiency ≈ 15% (due to irregular row lengths)
Effective bandwidth = 0.15 × theoretical peak
```

# Expert CUDA Mathematics - Streams, Multi-GPU, Tensor Cores & Memory Pools

## Section 16: CUDA Streams and Concurrency Mathematics (Questions 1-15)

| Question | Description |
| - | - |
| 1 | A CUDA application uses 4 streams with kernels taking 12ms, 8ms, 15ms, and 10ms respectively. Calculate the total execution time with and without streams, assuming kernels can overlap perfectly and memory transfers take 3ms each. |
| 2 | Calculate the optimal number of streams for hiding memory transfer latency when kernel execution time is 25ms and memory transfer time is 8ms per stream, given GPU has 4 copy engines. |
| 3 | A streaming application processes video frames using 8 concurrent streams. Each frame requires 150MB transfer (H2D), 45ms processing, and 75MB transfer (D2H). Calculate the sustained throughput in fps assuming perfect pipeline overlap. |
| 4 | Calculate the stream synchronization overhead when using cudaEventSynchronize() across 16 streams, where each synchronization point adds 5μs latency and streams synchronize every 100ms. |
| 5 | A multi-stream CUDA application achieves 85% overlap efficiency between compute and memory transfers. If compute takes 40ms and memory takes 12ms per stream with 6 streams, calculate the actual execution time. |
| 6 | Calculate the memory bandwidth utilization when 12 streams perform concurrent H2D transfers, each transferring 512MB, on a system with PCIe 4.0 x16 (64 GB/s theoretical) achieving 75% efficiency. |
| 7 | A CUDA stream implementation uses priority-based scheduling with 3 priority levels. High priority (2 streams): 20ms, Medium (4 streams): 35ms, Low (6 streams): 50ms. Calculate the weighted average completion time. |
| 8 | Calculate the stream dependency resolution overhead for a DAG with 8 streams where each stream waits for 2.5 predecessor streams on average, with 10μs wait overhead per dependency. |
| 9 | A concurrent kernel execution scenario has 3 kernels: K1 uses 40% GPU resources, K2 uses 35%, K3 uses 45%. Calculate if they can run concurrently and the resource utilization efficiency. |
| 10 | Calculate the optimal stream depth for a pipeline where Stage 1 takes 8ms, Stage 2 takes 15ms, Stage 3 takes 12ms, to achieve maximum throughput with minimum latency. |
| 11 | A CUDA application uses asynchronous memory operations across 6 streams. Each stream transfers 256MB with 90% overlap with computation. Calculate the effective memory bandwidth if theoretical peak is 900 GB/s. |
| 12 | Calculate the stream load balancing efficiency when work is distributed as: Stream 1: 1000 tasks, Stream 2: 850 tasks, Stream 3: 1200 tasks, Stream 4: 950 tasks, with each task taking 0.5ms ± 20% variance. |
| 13 | A multi-stream reduction operation uses tree reduction across 8 streams. Each stream reduces 2M elements locally, then participates in inter-stream reduction. Calculate total reduction steps and synchronization points. |
| 14 | Calculate the concurrent memory allocation overhead when 10 streams simultaneously allocate GPU memory: 5 streams need 512MB each, 3 streams need 1GB each, 2 streams need 256MB each. |
| 15 | A CUDA stream implementation uses double buffering across 4 streams. If buffer size is 128MB per stream and swap time is 2ms, calculate the memory overhead and pipeline efficiency. |

## Section 17: Multi-GPU Programming Calculations (Questions 16-30)

| Question | Description |
| - | - |
| 16 | Calculate the all-reduce communication time for 8 GPUs using ring algorithm, where each GPU contributes 1GB of data and inter-GPU bandwidth is 600 GB/s (NVLink) with 95% efficiency. |
| 17 | A data parallel training setup uses 16 GPUs with model size 4GB. Calculate the parameter server communication overhead when using hierarchical reduction with 4 GPUs per node and 25 GB/s inter-node bandwidth. |
| 18 | Calculate the load balancing efficiency for irregular workload distribution across 12 GPUs: 4 GPUs process 1M elements each, 6 GPUs process 800K elements each, 2 GPUs process 1.2M elements each. |
| 19 | A multi-GPU application uses 2D domain decomposition across 8 GPUs arranged in 4×2 grid. Each GPU processes 2048×1024 elements with 2-element halo exchange. Calculate communication volume and overlap efficiency. |
| 20 | Calculate the scaling efficiency for strong scaling when going from 1 GPU (100s execution) to 8 GPUs with communication overhead of 12% and load imbalance of 8%. |
| 21 | A peer-to-peer memory transfer between 4 GPUs follows the pattern: GPU0→GPU1 (2GB), GPU1→GPU2 (1.5GB), GPU2→GPU3 (2.5GB), GPU3→GPU0 (1GB). Calculate total transfer time with NVLink bandwidth 600 GB/s. |
| 22 | Calculate the memory pooling efficiency across 6 GPUs sharing a total memory pool of 48GB, where workload requires: 3 GPUs need 6GB each, 2 GPUs need 9GB each, 1 GPU needs 12GB. |
| 23 | A multi-GPU reduction tree uses binary tree topology across 16 GPUs. Each reduction step processes 512MB of data. Calculate the total reduction time and network utilization. |
| 24 | Calculate the multi-GPU synchronization overhead using barrier synchronization across 32 GPUs, where barrier implementation takes log₂(N) communication rounds with 50μs per round. |
| 25 | A distributed training scenario uses gradient compression achieving 4:1 compression ratio with 5% decompression overhead. Calculate the communication time improvement for 2GB gradients across 8 GPUs. |
| 26 | Calculate the multi-GPU memory coherence overhead when 6 GPUs share a 8GB dataset with 30% write operations requiring coherence protocol with 15μs coherence latency per write. |
| 27 | A multi-GPU sorting algorithm distributes 1 billion integers across 8 GPUs using sample sort. Calculate the load balancing efficiency when sample accuracy results in ±12% deviation from perfect distribution. |
| 28 | Calculate the fault tolerance overhead for a 16-GPU system using checkpoint/restart every 10 minutes, where checkpoint takes 30s and probability of failure is 0.1% per GPU per hour. |
| 29 | A multi-GPU graph algorithm partitions a graph with 10M vertices and 50M edges across 12 GPUs. With 15% cross-partition edges, calculate the communication volume and processing efficiency. |
| 30 | Calculate the energy efficiency comparison between single GPU (400W, 50 TFLOPS) vs. 4-GPU setup (4×350W, 4×45 TFLOPS) including 200W interconnect overhead for the same workload. |

## Section 18: Tensor Core Optimization Mathematics (Questions 31-45)

| Question | Description |
| - | - |
| 31 | A Tensor Core performs mixed-precision matrix multiplication: A(FP16) × B(FP16) + C(FP32) → D(FP32). For 4×4×4 operation at 1.4 GHz, calculate peak TOPS (Tera Operations Per Second). |
| 32 | Calculate the Tensor Core utilization efficiency when processing matrices of size 4096×4096×4096 with tile size 16×16×16, achieving 312 TOPS on hardware with peak 400 TOPS. |
| 33 | A transformer model uses Tensor Cores for attention computation with sequence length 2048, hidden dimension 1024, 16 heads. Calculate FLOPS breakdown between Tensor Core operations and regular core operations. |
| 34 | Calculate the memory bandwidth requirement for sustaining peak Tensor Core performance (400 TOPS) when processing BF16 data with arithmetic intensity of 62.5 FLOPS/byte. |
| 35 | A CUDA kernel uses Tensor Cores for 70% of operations (achieving 350 TOPS) and CUDA cores for 30% (achieving 15 TFLOPS). Calculate the effective application throughput and compare with CUDA-only implementation. |
| 36 | Calculate the precision conversion overhead when switching between FP32 and FP16 operations for Tensor Core utilization, given conversion rate of 2TB/s and workload requiring 40% conversions. |
| 37 | A mixed-precision training uses Tensor Cores with loss scaling factor 1024. Calculate the probability of gradient overflow when gradient norm follows log-normal distribution with σ = 2.5. |
| 38 | Calculate the Tensor Core memory hierarchy optimization when using 3-level tiling: L1 (16×16), L2 (64×64), L3 (256×256) for matrix multiplication, considering shared memory bandwidth of 8 TB/s. |
| 39 | A BERT model inference uses Tensor Cores for matrix multiplications representing 85% of total FLOPS. Calculate speedup over FP32 implementation assuming 8× Tensor Core advantage and 15% overhead for precision management. |
| 40 | Calculate the optimal batch size for maximizing Tensor Core utilization when processing variable-length sequences with lengths: 25% ≤ 128, 50% ≤ 512, 25% ≤ 1024 tokens, using dynamic batching. |
| 41 | A Tensor Core optimization uses block-sparse patterns with 50% sparsity. Calculate the effective TOPS considering 2× theoretical speedup with 25% sparsity handling overhead. |
| 42 | Calculate the Tensor Core scheduling efficiency when interleaving 4 different GEMM operations with dimensions: 1024×512×2048, 2048×1024×512, 512×2048×1024, 1024×1024×1024. |
| 43 | A multi-precision algorithm uses INT8 Tensor Cores for inference (1600 TOPS) with 5% accuracy loss requiring 10% additional computation for error correction. Calculate net performance improvement over FP16 (400 TOPS). |
| 44 | Calculate the Tensor Core occupancy when processing batch size 32 with sequence length 512, using thread blocks of 256 threads and requiring 84 registers per thread on a GPU with 65,536 registers per SM. |
| 45 | A distributed training setup uses Tensor Cores across 8 GPUs for gradient computation. Calculate the communication-to-computation ratio when Tensor Core operations achieve 280 TOPS per GPU and gradient synchronization requires 12 GB at 600 GB/s per link. |

## Section 19: Memory Pool and Allocation Mathematics (Questions 46-60)

| Question | Description |
| - | - |
| 46 | A CUDA memory pool manages allocations with sizes: 40% ≤ 1KB, 30% ≤ 16KB, 20% ≤ 256KB, 10% ≤ 4MB. Design optimal pool configuration to minimize fragmentation with 8GB total memory. |
| 47 | Calculate the memory allocation overhead reduction when using memory pools vs. cudaMalloc for 100,000 allocations per second, given cudaMalloc overhead of 50μs and pool allocation overhead of 0.5μs. |
| 48 | A memory pool implementation uses buddy allocation algorithm with minimum block size 1KB and maximum 1GB. Calculate internal fragmentation for allocation requests: 1.5KB, 10KB, 100KB, 1.5MB. |
| 49 | Calculate the memory pool cache efficiency when using 4 pools (1KB, 16KB, 256KB, 4MB blocks) with access pattern: 60% hits pool 1, 25% hits pool 2, 10% hits pool 3, 5% hits pool 4. |
| 50 | A CUDA application uses memory pools across 6 streams, each requiring peak allocation of 2GB with 70% temporal overlap. Calculate minimum total pool size and allocation scheduling efficiency. |
| 51 | Calculate the garbage collection overhead for a memory pool with 80% allocation rate, 15% deallocation rate, and 5% memory compaction rate, given compaction takes 10ms per GB. |
| 52 | A dynamic memory pool grows from 1GB to 8GB during execution. Calculate the reallocation overhead when growth factor is 2× and reallocation cost is 5ms per GB copied. |
| 53 | Calculate the memory pool thread safety overhead when 32 threads concurrently access pools, using lock-free algorithms with 85% success rate and 12μs retry penalty. |
| 54 | A memory pool uses slab allocation with slab sizes: 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB. Calculate utilization efficiency for workload with exponential size distribution (λ = 0.001). |
| 55 | Calculate the memory pool alignment overhead when requiring 256-byte alignment for all allocations, given allocation size distribution: 50% < 256B, 30% < 1KB, 20% ≥ 1KB. |
| 56 | A CUDA memory pool implements delayed deallocation with 100ms delay window. Calculate peak memory usage when allocation rate is 50MB/s and deallocation rate is 45MB/s with 10% variance. |
| 57 | Calculate the memory pool defragmentation efficiency when 40% of allocations create fragmentation and defragmentation recovers 75% of fragmented space with 8% CPU overhead. |
| 58 | A multi-GPU memory pool shares memory across 4 GPUs with NUMA topology. Calculate access penalty when 25% of allocations access remote GPU memory with 3× latency penalty. |
| 59 | Calculate the memory pool preallocation strategy for a workload with phases: Phase 1 needs 2GB (30% time), Phase 2 needs 6GB (50% time), Phase 3 needs 4GB (20% time). |
| 60 | A memory pool uses reference counting for automatic deallocation. Calculate the memory leak probability when reference count overflow occurs at 2^16 with average 100 references per object and 10^6 objects. |

## Detailed Solution Examples

### Question 1 - CUDA Streams Overlap:
```
Without streams: 12 + 8 + 15 + 10 + 4×3 = 57ms (sequential)
With streams: max(12+3, 8+3, 15+3, 10+3) = 18ms (parallel)
Speedup = 57/18 = 3.17×
```

### Question 16 - Multi-GPU All-Reduce:
```
Ring algorithm: (N-1) steps for reduce-scatter + (N-1) steps for all-gather
Total steps = 2×(8-1) = 14 steps
Data per step = 1GB ÷ 8 = 128MB
Time per step = 128MB ÷ (600 GB/s × 0.95) = 0.224ms
Total time = 14 × 0.224ms = 3.14ms
```

### Question 31 - Tensor Core TOPS:
```
4×4×4 mixed precision operation = 4×4×4×2 = 128 operations (FMA counted as 2)
At 1.4 GHz: 128 ops × 1.4×10⁹ Hz = 179.2 GOPS
For multiple Tensor Cores: TOPS = Tensor_Cores × 179.2 GOPS
```

### Question 46 - Memory Pool Design:
```
Pool 1 (1KB blocks): 40% × 8GB = 3.2GB → 3,276,800 blocks
Pool 2 (16KB blocks): 30% × 8GB = 2.4GB → 153,600 blocks
Pool 3 (256KB blocks): 20% × 8GB = 1.6GB → 6,400 blocks
Pool 4 (4MB blocks): 10% × 8GB = 0.8GB → 200 blocks
```
```
