
# Basic Concepts in GPU Computing

*Author: Hao Gao*  
*Original post on [Medium](https://medium.com/@smallfishbigsea/basic-concepts-in-gpu-computing-3388710e9239)*  
*Published: October 10, 2017*

---

This post summarizes the white paper on NVIDIA's **Fermi architecture** to explain fundamental GPU computing concepts.

---

## GPU Architecture

A GPU card consists of:
- **Memory dies**: Compose global memory in CUDA.
- **GPU unit**: Contains streaming multiprocessors (SMs).

Each **Fermi GPU** has:
- **16 SMs**
- Each SM has:
  - 32 CUDA cores (each with one float and one int processor)
  - 2 warp schedulers
  - Configurable shared memory (up to 48 KB)
  - L1 cache (up to 48 KB)
  - L2 cache
  - Registers
  - Warp size: 32

**Warp**: Group of 32 threads executed in parallel.

![GPU Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*AFdG_VBrn7U52LuiYx6wyA.png)

---

## Memory Hierarchy

| Type          | Characteristics                                  |
|---------------|--------------------------------------------------|
| Registers     | Fastest, per-thread                              |
| Shared Memory | Fast, per-block, configurable with L1            |
| L1 Cache      | Caches local/global memory (or just local)       |
| L2 Cache      | Caches local and global memory                   |
| Global Memory | Large but slow (RAM equivalent)                  |

---

## Example: GRID K520 (Kepler Architecture)

Extracted specs:
```
Device name: GRID K520
Compute capability: 3.0
Clock Rate: 797 MHz
Total SMs: 8
Shared Memory per SM: 48 KB
Registers per SM: 65536 (32-bit)
Global Memory: ~4 GB
L2 Cache: 512 KB
Warp Size: 32
Max Threads per Block: 1024
```

---

## Execution Hierarchy

### Computation
- Threads → Blocks → Grids
- Each block runs on a single SM
- Threads are split into **warps**

### Memory
- Shared memory is per-block, limited by SM capacity
- Overuse reduces occupancy

![Execution Hierarchy](https://miro.medium.com/v2/resize:fit:720/format:webp/1*wJG7hOPnEN0H_7GZLHwdLA.png)

---

## Shared Memory Bank Conflicts

- Shared memory is split into **banks**
- Each bank serves one 32-bit value per cycle
- **Bank conflict**: Multiple threads access same bank → serialization

**No Conflict:** If threads access different banks or same address (broadcast)
  
Example bank mapping:
```
Bank    | 1       | 2       | 3       | ...
Address | 0-3     | 4-7     | 8-11    | ...
```

- GT200 GPUs: 16 banks
- Fermi GPUs: 32 banks

![Bank Conflicts](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Wj6gB_MhhnmGu3OuToAjJg.jpeg)

---

## Modern Enhancements

- **Volta architecture** introduced **tensor cores** to accelerate neural network workloads.

![Tensor Cores](https://miro.medium.com/v2/resize:fit:720/format:webp/1*nJ8vU3WJE9IGoDbxI2JdMg.png)

---

## References

1. [NVIDIA Fermi Whitepaper](http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)  
2. [Caltech Lecture Slides](http://courses.cms.caltech.edu/cs179/2015_lectures/cs179_2015_lec05.pdf)  
3. [StackOverflow: Bank Conflicts](https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming)  
4. [NVIDIA Volta Whitepaper](http://www.nvidia.com/object/volta-architecture-whitepaper.html)

---

> 📌 *Understanding GPU architecture and memory hierarchy is essential for writing optimized CUDA programs, avoiding bottlenecks like shared memory bank conflicts, and achieving high parallel efficiency.*

![Fermi Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*xCKvtJ0-VqNHkUTryHK7VA.png)

![Logical Hierarchy](https://miro.medium.com/v2/resize:fit:720/format:webp/1*rSVR6Hzlr0BiS6U4ohz14g.png)
