# 🚀 Day 005 of 100 Days of GPU: Image Blur Operation 🚀

## 🌟 Overview
This is day 5 of my 100 Days of GPU learning journey, focusing on a **image blur operation** implemented with CUDA GPU acceleration. The goal is to understand GPU performance characteristics, memory transfers, and kernel execution for a basic image processing task.

## 🖥️ Hardware Specifications
### Detected CUDA Device:
- **GPU**: Tesla T4 💪
- **Compute capability**: 7.5
- **Global memory**: 14.74 GB 📊
- **Memory bandwidth**: 320.1 GB/s 🔄
- **Max threads per block**: 1024
- **Max grid dimensions**: (2147483647, 65535, 65535)

## 🖼️ Image Details
- **Dimensions**: 1024 × 768 pixels
- **Initialization**: Gradient pattern
- **Format**: RGBA (4 bytes per pixel)

## ⏱️ Performance Breakdown

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Memory allocation | 0.17 | 📝 GPU & host memory setup |
| Host → Device transfer | 0.26 | 📤 Transferring image data to GPU |
| **Kernel execution** | **0.30** | ⚡ Actual blur computation |
| Device → Host transfer | 0.66 | 📥 Retrieving results |
| **Total time** | **4.57** | ⏰ Complete operation |

## 🔍 Image Processing Example

### Before Blur (Sample):
```
0   1   2   3   4
1   2   3   4   5
2   3   4   5   6
3   4   5   6   7
4   5   6   7   8
```

### After Blur (Sample):
```
0   1   1   2   3
1   2   3   4   5
1   3   4   5   6
2   4   5   6   7
3   5   6   7   8
```

## 🧐 Key Insights & Performance Analysis

### ❓ Performance Question
**Q**: Is it concerning that kernel execution time (0.30 ms) is greater than host-to-device transfer time (0.26 ms)?

**A**: 👍 **Nothing is wrong!** This is completely normal, especially for:

1. **Small workloads** - Our image is only ~3MB (1024×768×4 bytes)
   - Calculation: $1024 \times 768 \times 4 \text{ bytes} = 3,145,728 \text{ bytes} \approx 3 \text{ MB}$
   - This is a trivial amount of data for modern GPUs

2. **Kernel launch overhead** becomes significant for fast kernels
   - Typical kernel launch overhead: ~10-20 μs
   - For short kernels, this overhead can be a substantial percentage
   - Formula: $\text{Overhead percentage} = \frac{\text{Launch overhead}}{\text{Total kernel time}} \times 100\%$
   - Example: $\frac{0.01\text{ms}}{0.30\text{ms}} \times 100\% = 3.3\%$ of execution time

3. **Memory bandwidth vs. Compute requirements**
   - Tesla T4 memory bandwidth: 320.1 GB/s
   - Theoretical time to process 3MB at full bandwidth: 
     - $\frac{3\text{MB}}{320.1\text{GB/s}} \approx 0.009\text{ms}$ (just for memory operations)
   - Actual computation adds additional time

4. **Sanity check calculation** (estimated transfer time):
   - PCIe 3.0 x16 theoretical bandwidth: ~12 GB/s
   - Expected transfer time: $\frac{3\text{MB}}{12\text{GB/s}} \approx 0.25\text{ms}$
   - Observed: 0.26ms ✓ Matches expected transfer time!

5. **D2H being slower (0.66 ms) detailed analysis**:
   - D2H/H2D ratio: $\frac{0.66\text{ms}}{0.26\text{ms}} = 2.54\times$ slower
   - Common causes for this asymmetry:
     - GPU memory controller optimized for coalesced reads over writes
     - PCIe bus asymmetric performance characteristics
     - Memory synchronization overhead during readback operations
     - Memory access patterns from global memory to host

### 🔄 Transfer Speeds Explained (Detailed Calculations)
- **H2D (Host → Device)**: 0.26ms for ~3MB
  - Calculation: $\frac{3\text{MB}}{0.26\text{ms}} = \text{~11.5 GB/s}$
  - 💡 Expected PCIe 3.0 x16 theoretical bandwidth: ~12 GB/s
  - Formula: $\frac{\text{Transfer size (bytes)}}{\text{Transfer time (seconds)}}$
  - $3\text{MB} = 1024 \times 768 \text{ pixels} \times 4 \text{ bytes/pixel} \approx 3,145,728 \text{ bytes}$
  - $\frac{3,145,728 \text{ bytes}}{0.00026 \text{ seconds}} \approx 12.1 \text{ GB/s}$

- **D2H (Device → Host)**: 0.66ms (slower than H2D)
  - Calculation: $\frac{3\text{MB}}{0.66\text{ms}} = \text{~4.5 GB/s}$
  - 🔎 Normal asymmetry factors in detail:
    - Memory controller optimizations favor write operations to GPU
    - PCIe bus has asymmetric read/write performance characteristics
    - Context switching overhead from kernel completion to transfer initiation
    - Memory access patterns and alignment issues during readback

## 💻 Implementation Details

### CUDA Kernel Configuration:
- **Block size**: 16×16 (256 threads per block)
  - Calculation: $16 \times 16 = 256 \text{ threads}$
  - Well below Tesla T4 max threads per block (1024)
  - Good occupancy for compute capability 7.5

- **Grid size**: Calculated to cover entire image
  - Formula: $\lceil\frac{\text{width}}{\text{blockDim.x}}\rceil \times \lceil\frac{\text{height}}{\text{blockDim.y}}\rceil$
  - Example: $\lceil\frac{1024}{16}\rceil \times \lceil\frac{768}{16}\rceil = 64 \times 48 = 3,072 \text{ blocks}$
  - Total threads launched: $3,072 \text{ blocks} \times 256 \text{ threads} = 786,432 \text{ threads}$

### Blur Algorithm Detailed Breakdown:
- **Algorithm**: Simple 3×3 average filter (9-point average)
  - For each pixel (x,y), output value = average of inputs (x±1, y±1)
  - Formula: $\text{output}(x,y) = \frac{1}{9}\sum_{i=-1}^{1}\sum_{j=-1}^{1} \text{input}(x+i, y+j)$
  
- **Memory Access Pattern**:
  - Each thread requires 9 reads from global memory (unoptimized)
  - Total memory reads: $9 \times \text{width} \times \text{height} = 9 \times 1024 \times 768 = 7,077,888 \text{ reads}$
  - Global memory throughput: $\frac{7,077,888 \text{ reads} \times 4 \text{ bytes}}{0.30\text{ms}} \approx 94.4 \text{ GB/s}$
  
- **Edge Handling**:
  - Clamping to valid coordinates: $\max(0, \min(x, \text{width}-1))$
  - Ensures border pixels have valid blur values
  - Prevents out-of-bounds memory access

- **Work Distribution**:
  - One thread per output pixel (1:1 mapping)
  - Perfect load balancing (except for edge cases)
  - Thread (x,y) writes to output position (x,y)

## 📈 Optimization Opportunities

Want to improve performance further? Consider:

1. **🔒 Use pinned memory** for faster transfers
   ```cuda
   cudaMallocHost(&h_image, size);  // Instead of malloc
   ```
   - Eliminates extra memory copy by OS during transfers
   - Can improve transfer speeds by 2-3×
   - Trade-off: Consumes physical RAM, not swappable

2. **🧠 Utilize shared memory** to reduce global memory accesses
   ```cuda
   __shared__ float shared_data[BLOCK_SIZE+2][BLOCK_SIZE+2];
   ```
   - Calculation: For 16×16 block, shared memory needed = $(16+2) \times (16+2) \times 4 \text{ bytes} = 1,296 \text{ bytes}$
   - Reduces global reads from $9 \times \text{width} \times \text{height}$ to $1 \times \text{width} \times \text{height} + \text{boundaries}$
   - Potential speedup: Up to 9× for memory-bound kernel

3. **⚡ Coalesce memory accesses** for better throughput
   ```cuda
   // Access sequential memory addresses within a warp
   ```
   - Tesla T4 memory transaction size: 32 bytes
   - Coalesced access enables 32-byte transactions instead of multiple smaller ones
   - Potential memory throughput increase: 2-4×

4. **📏 Tune block dimensions** for your specific GPU
   ```cuda
   dim3 blockSize(32, 8, 1);  // Try different configurations
   ```
   - Tesla T4 has 40 SMs with 64 warps each
   - Optimal occupancy calculation: $\frac{\text{active warps}}{\text{max warps per SM}}$
   - Different block sizes affect register usage and shared memory pressure

5. **🔄 Overlap computation and transfers** with CUDA streams
   ```cuda
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   ```
   - Enables concurrent execution of kernel and memory transfers
   - Can hide latency of operations
   - Ideal for processing image in tiles

## 📚 Learning Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Optimizing CUDA Applications](https://developer.nvidia.com/blog/tag/optimization/)
- [Nsight Systems for Performance Analysis](https://developer.nvidia.com/nsight-systems)

---

Happy GPU Computing! 🎉