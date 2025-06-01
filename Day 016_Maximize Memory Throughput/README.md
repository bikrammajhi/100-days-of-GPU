# 📈 5.3 Maximize Memory Throughput

Optimizing memory throughput is crucial for high-performance CUDA applications. This section outlines best practices for reducing low-bandwidth transfers and improving memory access patterns.

---

## 🔁 5.3.1 Minimize Host–Device Transfers

### Key Recommendations

* **Avoid frequent host-device transfers** — they have significantly lower bandwidth.
* **Keep computation on device** whenever possible, even if parallelism is low.
* Use **intermediate device-only data structures**.
* **Batch small transfers** into large ones to reduce overhead.

### Optimization Techniques

* Use **page-locked host memory** for higher PCIe bandwidth.
* Use **mapped page-locked memory** to avoid explicit memory copies.
* On integrated systems (shared memory), **use mapped memory** directly.
* Use `cudaGetDeviceProperties()` to detect integrated devices.

---

## 🧠 5.3.2 Optimize Device Memory Accesses

### Memory Types

* **Global Memory** (slow, large): Optimize with coalesced access.
* **Shared Memory** (fast, limited): Manually managed, best for thread collaboration.
* **Local Memory** (slow, private): Used when registers overflow or for dynamic arrays.
* **Constant & Texture Memory**: Specialized caches for specific use-cases.

---

## 📦 Global Memory

### Coalescing Accesses

* Accesses must align to **32/64/128-byte boundaries**.
* Threads in a warp should access **contiguous memory locations**.
* Poor coalescing leads to **wasted bandwidth and reduced throughput**.

### Best Practices

* Use **data types** of 1, 2, 4, 8, or 16 bytes.
* Ensure **natural alignment**: Address must be a multiple of the type's size.
* Use `__align__` specifier for structures:

  ```cpp
  struct __align__(8) { float x; float y; };
  ```

### Pitfalls

* **Manual memory partitioning** may cause misalignment.
* **Misaligned 8/16-byte reads** may return incorrect data.

---

## 📐 Two-Dimensional Arrays

* Use `cudaMallocPitch()` or `cuMemAllocPitch()` to pad rows for alignment.
* Ensure **array width and block width** are multiples of warp size.
* Formula for indexing:

  ```cpp
  BaseAddress + width * ty + tx
  ```

---

## 📍 Local Memory

### Characteristics

* Allocated for:

  * Arrays with unknown index patterns.
  * Large data that can't fit in registers.
  * Register spills when register usage is exceeded.
* Resides in **global device memory**, so it's **slow**.

### Detection

* PTX code shows `.local` if local memory is used.
* Use `--ptxas-options=-v` to view `lmem` usage.
* Use `cuobjdump` to inspect actual memory layout.

### Coalescing

* Fully coalesced if all threads in a warp access the **same relative address**.

---

## 🧮 Shared Memory Usage Pattern

1. Load from global memory to shared memory.
2. Synchronize (`__syncthreads()`).
3. Process shared data.
4. Synchronize again if needed.
5. Write back to global memory.

---

## ⚙️ On-Chip Memory Configuration

* Devices of **Compute Capability 7.x and higher** allow configuring **L1 cache vs. shared memory** usage per kernel.
* Choose based on data access patterns and reuse needs.

---

## 🧠 Tips

* Maximize **on-chip memory** usage (shared, L1, L2).
* **Align and coalesce** global memory accesses.
* Prefer **device-local computation** to minimize host-device overhead.
* Use **tools** like `nvcc`, PTX, and `cuobjdump` to inspect memory behavior.

---
