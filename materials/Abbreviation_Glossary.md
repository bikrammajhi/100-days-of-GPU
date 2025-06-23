# 📘 CUDA Kernel Abbreviation Glossary

A comprehensive breakdown of commonly used abbreviations and terms used throughout high-performance WMMA-based CUDA GEMM kernels. This glossary is structured by **functionality and usage context**, inspired by NVIDIA-style repo documentation.

> ✨ *We will use this as a quick reference while reading kernels, debugging, or maintaining naming consistency across our HPC CUDA projects.*

---

## 🧠 Memory Types & Addressing

| Abbreviation                     | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| `shmem`                          | 💣 Shared memory workspace                                 |
| `shmem_stride_bytes`             | 🧮 Stride in shared memory in bytes                        |
| `gmem`                           | 🔵 Global memory                                           |
| `cvta_to_shared_u32`             | 🔀 Convert virtual address to 32-bit shared memory address |
| `A_block_smem`, `B_block_smem`   | 🧱 Block-level tiles of A/B in shared memory               |
| `A_block_gmem`, `B_block_gmem`   | 📀 Block-level tiles of A/B in global memory               |
| `C_block_gmem`, `D_block_gmem`   | 📀 Global memory tiles for output C and D                  |
| `C_warp_gmem`, `D_warp_gmem`     | 🌀 Warp-level output tiles in global memory                |
| `src`, `dst`                     | 🔀 Source and Destination pointers                         |
| `src_stride`, `dst_stride_bytes` | 🔽 Strides in source or destination memory                 |
| `thread_offset_bytes`            | 🔂 Per-thread byte offset in memory                        |

---

## 🧮 Compute: MMA & Matrix Ops

| Abbreviation               | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `mma`                      | ⚙️ Matrix Multiply-Accumulate                       |
| `mma.sync.aligned.*`       | 🔁 PTX instruction for FP16 MMA                     |
| `ldmatrix.sync.aligned.*`  | 📅 Warp-wide load from shared memory                |
| `stmatrix.*`               | 📤 Warp-wide store to global memory                 |
| `ldmatrix_a`, `ldmatrix_b` | 🧩 Specialized shared memory loaders for A/B        |
| `acc_register`             | 🧲 Accumulator register array (C/D partial results) |
| `A_register`, `B_register` | 📌 Per-warp A/B tiles in registers                  |
| `C_register`               | 📂 C matrix read from global memory                 |
| `C_mma_tile`, `D_mma_tile` | 📌 Individual MMA tile pointers in C/D matrices     |

---

## 🧱 Tiling Dimensions

| Abbreviation                                                           | Description                                   |
| ---------------------------------------------------------------------- | --------------------------------------------- |
| `BM_dim`, `BN_dim`, `BK_dim`                                           | 🧱 Block-level tiling dimensions (BlockM/N/K) |
| `WM_dim`, `WN_dim`, `WK_dim`                                           | 🎯 Warp-level tiling dimensions (WarpM/N/K)   |
| `MMA_M_dim`, `MMA_N_dim`, `MMA_K_dim`                                  | 💪 MMA intrinsic tile sizes                   |
| `mma_tiles_per_warp_m`, `mma_tiles_per_warp_n`, `mma_tiles_per_warp_k` | 📊 MMA tiles per warp along M/N/K             |

---

## 🧵 Thread & Warp Layout

| Abbreviation                                                  | Description                              |
| ------------------------------------------------------------- | ---------------------------------------- |
| `warp_m`, `warp_n`                                            | 🧶 Warp indices within a block           |
| `block_m`, `block_n`                                          | 📦 Block indices in grid                 |
| `threadIdx.x`, `threadIdx.y`                                  | 🧵 Thread index within warp/block        |
| `WARP_SIZE`                                                   | ⚙️ Fixed as 32 threads                   |
| `WARPS_PER_BLOCK_M`, `WARPS_PER_BLOCK_N`, `WARPS_PER_BLOCK_K` | 📉 Number of warps per block along M/N/K |
| `ThreadsM`, `ThreadsN`, `NumThreads`                          | 🔦 Block thread configuration            |
| `BlocksM`, `BlocksN`                                          | 🧱 Grid size in M/N dimensions           |

---

## 🚀 Kernel Execution Parameters

| Abbreviation                                   | Description                                      |
| ---------------------------------------------- | ------------------------------------------------ |
| `alpha`, `beta`                                | ✴️ Scaling factors in GEMM epilogue              |
| `alpha_`, `beta_`                              | 🔀 Casted half-precision versions                |
| `tileMemcpy*`                                  | 🚛 Tile memory copy utilities                    |
| `tileMemcpySwizzle*`                           | 🔄 Memory swizzling copy functions               |
| `int_log2`, `SWIZZLE_BITS_*`, `SWIZZLE_MASK_*` | 🧠 Bitwise swizzling utilities                   |
| `offset_direction`                             | ↔️ Used for ping-pong double buffering           |
| `BUFFER_SIZE`                                  | 📦 Shared memory buffer size per tile            |
| `shmem_bytes`                                  | 📦 Total shared memory size allocated per kernel |

---

## 🧪 Verification, Logging & Utility

| Abbreviation          | Description                                  |
| --------------------- | -------------------------------------------- |
| `sgemm_params`        | ⚙️ Struct holding all GEMM launch parameters |
| `KernelLogger`        | 📊 Measures and logs kernel GFLOPS           |
| `sgemm_verify`        | 🧪 Verifies correctness of results           |
| `elementwise_isclose` | 🧮 Compares float values element-wise        |
| `CUDA_CHECK()`        | 🛑 Error checking macro                      |

---

## 🔬 Advanced Register Swizzling

| Abbreviation                           | Description                                            |
| -------------------------------------- | ------------------------------------------------------ |
| `A_gmem_cache_reg`, `B_gmem_cache_reg` | 📀 float4-based prefetch registers                     |
| `src_reg`, `dst_reg`                   | 🔀 Used in looped copy operations                      |
| `swizzled_offset`, `logical_offset`    | 🌀 Bit-manipulated offsets for avoiding bank conflicts |

---

## 🛠️ Code Utility Macros & Constants

| Abbreviation                | Description                        |
| --------------------------- | ---------------------------------- |
| `RAND_HALF`                 | 🎲 Random value in half precision  |
| `cvta_to_shared_u32()`      | 🔀 Cast to 32-bit shared pointer   |
| `cudaFuncSetAttribute(...)` | 🧰 Set dynamic shared memory limit |

---

## 📆 Launch Config & Epilogue Details

| Abbreviation                 | Description                         |
| ---------------------------- | ----------------------------------- |
| `gridDim`, `blockDim`        | 🚀 CUDA kernel launch configuration |
| `num_block_tiles_k`          | 🔢 Number of k-tiles per GEMM block |
| `C_warp_tile`, `D_warp_tile` | 🧱 Warp tile pointer in C and D     |
| `kernel_6_launch(...)`       | 💥 Entry point to launch `kernel_6` |
| `num_runs`                   | 🔁 Iterations for benchmarking      |

---

## 📦 Epilogue Fusion

| Abbreviation               | Description                             |
| -------------------------- | --------------------------------------- |
| `ldmatrix_m16n8_gmem`      | 📅 Load C tiles for beta scaling        |
| `stmatrix_m16n8`           | 📤 Store final scaled output to D       |
| `acc_register_`            | 🔀 Half-view of accumulator registers   |
| `C_register`, `C_mma_tile` | 📌 Intermediate C tile used in epilogue |
| `D_mma_tile`               | 📌 Final output tile in D               |

---

## 🌟 Final Words

> This glossary grows with our kernel. We'll update as we introduce new optimizations (e.g., FP8, TMA, async pipelines). 🚀
