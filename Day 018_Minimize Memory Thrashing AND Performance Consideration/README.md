# GPU Optimization Strategies and Trade-offs

## Optimization Strategies Overview

| Optimization | Benefit to Compute Cores | Benefit to Memory | Strategies |
|--------------|--------------------------|-------------------|------------|
| **Maximizing occupancy** | More work to hide pipeline latency | More parallel memory accesses to hide DRAM latency | Tuning usage of SM resources such as threads per block, shared memory per block, and registers per thread |
| **Enabling coalesced global memory accesses** | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic and better utilization of bursts/cache-lines | Transfer between global memory and shared memory in a coalesced manner and performing un-coalesced accesses in shared memory (e.g., corner turning)<br><br>Rearranging the mapping of threads to data<br><br>Rearranging the layout of the data |
| **Minimizing control divergence** | High SIMD efficiency (minimizing idle cores during SIMD execution) | - | Rearranging the mapping of threads to work and/or data<br><br>Rearranging the layout of the data |
| **Tiling of reused data** | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic | Placing data that is reused within a block in shared memory or registers so that it is transferred between global memory and the SM only once |
| **Privatization (covered later)** | Fewer pipeline stalls waiting for atomic updates | Less contention and serialization of atomic updates | Applying partial updates to a private copy of the data then updating the universal copy when done |
| **Thread coarsening** | Less redundant work, divergence, or synchronization | Less redundant global memory traffic | Assigning multiple units of parallelism to each thread in order to reduce the "price of parallelism" when it is incurred unnecessarily |

## Tensions Between Optimizations

### Maximizing occupancy
Maximizing occupancy hides pipeline latency, but too many threads may compete for the cache, evicting each others' data (*thrashing* the cache)

### Shared memory tiling
Using more shared memory enables more data reuse, but may limit occupancy

### Thread coarsening
Coarsening reduces parallelization overhead, but requires more resources per thread which may limit occupancy

---

***Need to find the sweet spot that achieves the best compromise***