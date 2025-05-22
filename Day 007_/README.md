# CUDA L2 Cache Memory Management 

## 🎯 Quick Overview

**What is this about?** Starting with CUDA 11.0 and compute capability 8.0+, you can control how your GPU's L2 cache handles different types of memory accesses to boost performance.

**Key Concept:** Not all memory accesses are equal - some data is used repeatedly (persisting), while other data is used once (streaming).

---

## 📊 Visual Memory Access Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ACCESS TYPES                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔄 PERSISTING ACCESS                                      │
│  ├─ Data used repeatedly (matrix multiply, iterative algos) │
│  ├─ Should stay in L2 cache longer                          │
│  └─ Higher priority for cache retention                     │
│                                                             │
│  📥 STREAMING ACCESS                                        │
│  ├─ Data used once (file I/O, initial data load)            │
│  ├─ Should be evicted quickly                               │
│  └─ Lower priority for cache retention                      │
│                                                             │
│  ⚖️ NORMAL ACCESS                                           │
│  ├─ Default behavior                                        │
│  └─ Used to reset persisting status                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ L2 Cache Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      L2 CACHE LAYOUT                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌───────────────────────────────────┐ │
│  │   SET-ASIDE     │  │        NORMAL L2 CACHE            │ │
│  │   PORTION       │  │                                   │ │
│  │                 │  │  • Used by streaming accesses     │ │
│  │ • Persisting    │  │  • Used by normal accesses        │ │
│  │   accesses get  │  │  • Can use set-aside when empty   │ │
│  │   priority      │  │                                   │ │
│  │ • Up to 75% of  │  │                                   │ │
│  │   total L2      │  │                                   │ │
│  └─────────────────┘  └───────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started - Basic Setup

### Step 1: Check Device Capabilities

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);

// Required: Compute capability 8.0+
if (prop.major < 8) {
    printf("L2 cache management not supported\n");
    return;
}

printf("L2 Cache Size: %zu bytes\n", prop.l2CacheSize);
printf("Max Persisting L2: %zu bytes\n", prop.persistingL2CacheMaxSize);
printf("Max Window Size: %zu bytes\n", prop.accessPolicyMaxWindowSize);
```

### Step 2: Configure Set-Aside Cache

```cpp
// Set aside 75% of L2 cache for persisting accesses
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
```

---

## 🎛️ Access Policy Configuration

### The Access Policy Window Structure

```cpp
cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow.base_ptr  = ptr;           // Start address
stream_attribute.accessPolicyWindow.num_bytes = window_size;   // Size of region
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;          // 60% get hitProp
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
```

### 📈 Hit Ratio Visualization

```
hitRatio = 0.6 (60% persisting, 40% streaming)

Memory Window [32KB]:
┌────────────────────────────────────────            ┐
│ 🔄🔄🔄🔄🔄🔄📥📥📥📥📥📥🔄🔄🔄🔄🔄🔄📥📥 │
│                                                    │
│ 🔄 = Persisting (60% - random selection)           │
│ 📥 = Streaming (40% - random selection)            │
└────────────────────────────────────────            ┘

L2 Set-aside [16KB]:
┌──────────────────────┐
│ 🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄 │ ← Only persisting data
└──────────────────────┘
```

---

## 💡 Three Access Properties Explained

| Property | Symbol | Behavior | Use Case |
|----------|---------|----------|----------|
| `cudaAccessPropertyPersisting` | 🔄 | **High priority** for L2 retention | Frequently reused data |
| `cudaAccessPropertyStreaming` | 📥 | **Low priority** - evicted quickly | One-time use data |
| `cudaAccessPropertyNormal` | ⚖️ | **Resets** persisting status | Cleanup after algorithms |

---

## 📝 Complete Implementation Example

### Scenario: Matrix Operations with Repeated Data Access

```cpp
#include <cuda_runtime.h>

void optimized_matrix_operations() {
    // 1. Setup
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 2. Configure L2 set-aside (75% of total L2)
    size_t set_aside_size = min(int(prop.l2CacheSize * 0.75), 
                                prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_size);
    
    // 3. Setup access policy for frequently used matrix
    size_t matrix_size = sizeof(float) * rows * cols;
    size_t window_size = min(prop.accessPolicyMaxWindowSize, matrix_size);
    
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr  = matrix_A;
    stream_attr.accessPolicyWindow.num_bytes = window_size;
    stream_attr.accessPolicyWindow.hitRatio  = 0.8;  // 80% persisting
    stream_attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    
    // 4. Execute kernels that reuse matrix_A
    for(int i = 0; i < 10; i++) {
        matrix_multiply_kernel<<<grid, block, 0, stream>>>(matrix_A, matrix_B, result);
        matrix_transpose_kernel<<<grid, block, 0, stream>>>(matrix_A, temp);
    }
    
    // 5. Cleanup - Reset L2 cache for next operations
    stream_attr.accessPolicyWindow.num_bytes = 0;  // Disable window
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    cudaCtxResetPersistingL2Cache();  // Clear all persisting lines
    
    // 6. Now other data can use full L2 cache normally
    other_kernel<<<grid, block, 0, stream>>>(different_data);
}
```

---

## ⚖️ Hit Ratio Strategy Guide

### Scenario-Based Hit Ratio Selection

```
┌─────────────────────────────────────────────────────────────┐
│                    HIT RATIO STRATEGIES                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎯 hitRatio = 1.0                                          │
│  ├─ Use when: Single kernel, data fits in set-aside         │
│  ├─ Effect: Maximum caching attempt                         │
│  └─ Risk: Cache thrashing if data > set-aside size          │
│                                                             │
│  ⚖️ hitRatio = 0.5-0.8                                      │
│  ├─ Use when: Multiple concurrent kernels                   │
│  ├─ Effect: Balanced caching, reduces competition           │
│  └─ Best for: Most production scenarios                     │
│                                                             │
│  🎲 hitRatio = 0.2-0.4                                      │
│  ├─ Use when: Many concurrent streams                       │
│  ├─ Effect: Light caching, minimal interference             │
│  └─ Good for: High-concurrency applications                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example: Managing Multiple Concurrent Streams

When using **multiple CUDA streams**, each can be configured with **memory access policy hints**. These hints help the GPU's hardware-level **L2 cache management** and **streaming memory prioritization** optimize performance by understanding how each stream will access memory.

One such hint is the **access policy window**, which tells the GPU **how much of the L2 cache** the stream expects to use and **how aggressively** it wants to keep its data cached.

---

### ❌ **Bad Approach: Full Cache Contention**

```cpp
stream1_attr.accessPolicyWindow.hitRatio = 1.0;
stream2_attr.accessPolicyWindow.hitRatio = 1.0;
```

#### ✅ Meaning:

* `hitRatio = 1.0` tells the GPU that this stream wants **100% cache residency** for its working set (i.e., "please keep everything I touch in cache").
* Both `stream1` and `stream2` are configured to demand **all of the 16KB set-aside cache**.

#### ❌ Problem:

* If **both streams demand the full cache**, but **only 16KB** is available in total, they will **evict each other's data** continuously.
* This leads to **cache thrashing**, where neither stream gets the cache residency it wants, causing **cache misses** and **performance degradation**.

---

### ✅ **Good Approach: Cooperative Sharing**

```cpp
stream1_attr.accessPolicyWindow.hitRatio = 0.5;
stream2_attr.accessPolicyWindow.hitRatio = 0.5;
```

#### ✅ Meaning:

* Each stream now signals that it only needs **about 50% cache residency**.
* Effectively, both streams are saying: “I’m okay using \~8KB of the 16KB cache.”

#### ✅ Benefit:

* With a **cooperative policy**, the cache is **shared fairly**.
* Reduces cache eviction between streams.
* Improves **overall hit rate** and avoids **thrashing**.

---

## 🔄 L2 Reset Strategies

### Three Ways to Reset Persisting Cache

```
┌─────────────────────────────────────────────────────────────┐
│                    RESET METHODS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1️⃣ TARGETED RESET                                          │
│  ├─ Method: Set access property to `cudaAccessPropertyNormal` │
│  ├─ Scope: Specific memory region                           │
│  └─ Use: When you know exactly what to reset                │
│                                                             │
│  2️⃣ GLOBAL RESET                                            │
│  ├─ Method: `cudaCtxResetPersistingL2Cache()   `              │
│  ├─ Scope: All persisting cache lines                       │
│  └─ Use: Between different algorithm phases                 │
│                                                             │
│  3️⃣ AUTOMATIC RESET                                         │
│  ├─ Method: Hardware automatic (time-based)                 │
│  ├─ Scope: Unused persisting lines                          │
│  └─ Use: AVOID - timing is unpredictable                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚨 Important Limitations & Considerations

### Environment Restrictions

| Configuration | L2 Set-Aside Support |  Configuration Method  |
|---------------|----------------------|------------------------|
| **Normal Mode** | ✅ Fully supported | `cudaDeviceSetLimit()` |
| **MIG Mode** | ❌ Disabled | N/A |
| **MPS Mode** | ⚠️ Limited | Environment variable only |

### MPS Configuration
```bash
# For Multi-Process Service, set at startup:
export CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=75
```

---

## 📋 Quick Reference Checklist

### Before Implementation:
- [ ] Verify compute capability ≥ 8.0
- [ ] Check if MIG mode is enabled (disables feature)
- [ ] Measure baseline performance
- [ ] Identify data access patterns

### During Implementation:
- [ ] Query device properties first
- [ ] Set appropriate set-aside cache size
- [ ] Configure access policy windows
- [ ] Choose hit ratios based on concurrency
- [ ] Reset cache between algorithm phases

### Performance Tuning:
- [ ] Monitor cache hit rates
- [ ] Adjust hit ratios for concurrent kernels
- [ ] Profile memory bandwidth improvements
- [ ] Test different set-aside cache sizes

---

## 🎓 Best Practices Summary

### ✅ Do's
- **Profile first**: Measure performance before and after
- **Reset religiously**: Always reset L2 between different algorithms
- **Conservative hit ratios**: Start with 0.6-0.8 for most cases
- **Monitor concurrency**: Adjust for multiple streams
- **Size windows carefully**: Don't exceed `accessPolicyMaxWindowSize`

### ❌ Don'ts
- **Don't forget resets**: Persisting data can hurt subsequent kernels
- **Don't use hitRatio = 1.0 carelessly**: Can cause cache thrashing
- **Don't rely on automatic reset**: Timing is unpredictable
- **Don't ignore MIG/MPS limitations**: Check your environment
- **Don't over-complicate**: Start simple, optimize incrementally

---

## 📚 API Quick Reference

```cpp
// Query device properties
cudaGetDeviceProperties(&prop, device_id);

// Set L2 set-aside size
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

// Configure stream access policy
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// Configure graph node access policy  
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &attr);

// Reset all persisting cache lines
cudaCtxResetPersistingL2Cache();

// Query current set-aside size
cudaDeviceGetLimit(&current_size, cudaLimitPersistingL2CacheSize);
```

---
