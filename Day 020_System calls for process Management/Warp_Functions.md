# CUDA Warp Functions 

## 📋 Table of Contents

- [Warp Vote Functions](#-warp-vote-functions)
- [Warp Match Functions](#-warp-match-functions)
- [Warp Reduce Functions](#-warp-reduce-functions)
- [Warp Shuffle Functions](#-warp-shuffle-functions)
- [Code Examples](#-code-examples)
- [Hardware Requirements](#-hardware-requirements)

---

## 🗳️ Warp Vote Functions

Warp vote functions let **threads inside a warp (group of 32 threads)** collectively make decisions based on a condition (`predicate`). Each thread checks a condition, and all threads **vote** together.

### 🧠 Basic Concepts

- **mask**: A 32-bit value where each bit represents a thread (0 to 31)
- **predicate**: A condition that each thread checks (e.g., `x > 0`)
- **warp**: Group of 32 threads that execute together

### 🛠️ Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `__all_sync(mask, predicate)` | `true` if **all** predicates are true | All threads must agree |
| `__any_sync(mask, predicate)` | `true` if **any** predicate is true | At least one thread agrees |
| `__ballot_sync(mask, predicate)` | Bitmask showing which threads are true | Who agrees? (bit-wise) |
| `__activemask()` | Bitmask of currently active threads | Who's still running? |

### 📝 Examples

```cpp
// Check if all threads have positive values
bool everyonePositive = __all_sync(0xFFFFFFFF, x > 0);

// Check if any thread has positive value
bool anyonePositive = __any_sync(0xFFFFFFFF, x > 0);

// Get bitmask of which threads have positive values
unsigned votes = __ballot_sync(0xFFFFFFFF, x > 0);

// Get mask of currently active threads
unsigned mask = __activemask();
```

---

## 🔁 Warp Match Functions

These functions let threads **compare values with other threads** in a warp and **know who has the same value**.

### 🛠️ Functions

| Function | Returns | Use Case |
|----------|---------|----------|
| `__match_any_sync(mask, value)` | Bitmask of threads with **same value** | Find peers with same data |
| `__match_all_sync(mask, value, &pred)` | Full mask if all have same value, else 0 | Check for full agreement |

### 📝 Examples

```cpp
// Find threads with the same value as current thread
unsigned mask = 0xFFFFFFFF;
unsigned int peers = __match_any_sync(mask, thread_value);

// Check if all threads have the same value
int allEqual;
unsigned match = __match_all_sync(0xFFFFFFFF, thread_value, &allEqual);
```

**Example Result**: If thread 5 has `thread_value = 10`, and threads 2, 5, and 7 also have 10, then `peers = 0b10100100` (bits 2, 5, and 7 are set).

---

## ➕ Warp Reduce Functions

These perform a **reduction** (combine values) across threads in a warp. Each thread contributes a value, and the operation is applied across all active threads.

### 🛠️ Arithmetic Reductions

| Function | Operation | Example Result (values `{2, 3, 5}`) |
|----------|-----------|-------------------------------------|
| `__reduce_add_sync(mask, value)` | Sum | `10` |
| `__reduce_min_sync(mask, value)` | Minimum | `2` |
| `__reduce_max_sync(mask, value)` | Maximum | `5` |

### 🔣 Bitwise Reductions

| Function | Operation | Example Result (values `{2, 3, 5}`) |
|----------|-----------|-------------------------------------|
| `__reduce_and_sync(mask, value)` | Bitwise AND | `0` |
| `__reduce_or_sync(mask, value)` | Bitwise OR | `7` |
| `__reduce_xor_sync(mask, value)` | Bitwise XOR | `4` |

### 📝 Examples

```cpp
// Sum all thread values
int total = __reduce_add_sync(0xFFFFFFFF, my_val);

// Find minimum across all threads
int minimum = __reduce_min_sync(0xFFFFFFFF, my_val);

// Bitwise OR of all thread flags
unsigned result = __reduce_or_sync(0xFFFFFFFF, flags);
```

---

## 🔄 Warp Shuffle Functions

These functions allow **direct data exchange** between threads **within a warp**, without using shared memory. They're very fast and commonly used for reductions, scans, broadcasts, etc.

### 🛠️ Main Functions

| Function | Meaning | Use Case |
|----------|---------|----------|
| `__shfl_sync(mask, x, lane)` | Get `x` from thread `lane` | Broadcast a value |
| `__shfl_up_sync(mask, x, d)` | Get `x` from `laneId - d` | Inclusive scan, accumulate from lower lanes |
| `__shfl_down_sync(mask, x, d)` | Get `x` from `laneId + d` | Reverse scan, shift values downward |
| `__shfl_xor_sync(mask, x, m)` | Get `x` from `laneId ^ m` | Butterfly patterns for reductions |

### 📝 Common Patterns

#### 📣 Broadcast from Lane 0
```cpp
if (laneId == 0) value = arg;
value = __shfl_sync(0xFFFFFFFF, value, 0);  // All threads get value from lane 0
```

#### ➕ Inclusive Scan
```cpp
for (int i = 1; i <= 4; i *= 2) {
    int n = __shfl_up_sync(0xFFFFFFFF, value, i, 8);
    if ((laneId & 7) >= i) value += n;
}
```

#### 🌲 Butterfly Warp Reduction
```cpp
for (int i = 16; i >= 1; i /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, i);
}
```

---

## 💻 Code Examples

### Example 1: Broadcast
```cpp
__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    
    if (laneId == 0)
        value = arg;  // Only thread 0 sets value
    
    value = __shfl_sync(0xffffffff, value, 0);  // Broadcast to all
    
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}
```

### Example 2: Inclusive Scan
```cpp
__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;  // Initial values: 31, 30, ..., 0
    
    // Inclusive scan within groups of 8 threads
    for (int i=1; i<=4; i*=2) {
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }
    
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}
```

### Example 3: Warp Reduction
```cpp
__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;  // Initial values: 31, 30, ..., 0
    
    // Tree-based reduction using XOR shuffle
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    
    printf("Thread %d final value = %d\n", threadIdx.x, value);
    // All threads will print 496 (sum of 0+1+...+31)
}
```

---

## 🔧 Hardware Requirements

| Function Category | Minimum Compute Capability |
|-------------------|----------------------------|
| Vote Functions (`_sync` versions) | 7.0+ |
| Match Functions | 7.0+ |
| Reduce Functions | 8.0+ |
| Shuffle Functions (`_sync` versions) | 5.0+ |

---

## ⚠️ Important Rules

1. **All participating threads must call the same function with the same mask**
2. **Use only `_sync` versions** - older versions are deprecated/removed
3. **These functions do not synchronize memory** - they only coordinate values/decisions
4. **Target threads must be active** - otherwise behavior is undefined
5. **Mask must be consistent** across all participating threads

---

## 🚀 When to Use These Functions

- **Vote Functions**: Collective decision making, early termination conditions
- **Match Functions**: Finding threads with similar data, grouping operations
- **Reduce Functions**: Fast arithmetic operations across warps
- **Shuffle Functions**: Efficient data exchange, avoiding shared memory overhead

These warp-level primitives are essential for high-performance CUDA programming, especially for:
- Matrix operations
- Prefix sums and scans
- Reductions
- Histogram computations
- Graph algorithms

---

## 📚 Additional Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Warp-Level Primitives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-primitives)

---

