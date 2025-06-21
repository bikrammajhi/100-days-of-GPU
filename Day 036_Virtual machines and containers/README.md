# 🧠 Mojo GPU Vector Addition — (With Full Code)

In this blog, we’ll walk through a Mojo program that performs **vector addition on the GPU** — slowly, like you’re learning to walk 👶🚶.

We’ll start with the full code, and then break it down **one step at a time**. No prior Mojo or GPU experience needed.

---

## 🧾 Full Mojo Code

```mojo
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias float_dtype = DType.float32
alias VECTOR_WIDTH = 10
alias BLOCK_SIZE = 5
alias layout = Layout.row_major(VECTOR_WIDTH)

def main():
    constrained[
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator(),
        "This example requires a supported GPU",
    ]()

    # Get context for the attached GPU
    var ctx = DeviceContext()

    # Allocate data on the GPU address space
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    # Fill in values across the entire width
    _ = lhs_buffer.enqueue_fill(1.25)
    _ = rhs_buffer.enqueue_fill(2.5)

    # Wrap the device buffers in tensors
    var lhs_tensor = LayoutTensor[float_dtype, layout](lhs_buffer)
    var rhs_tensor = LayoutTensor[float_dtype, layout](rhs_buffer)
    var out_tensor = LayoutTensor[float_dtype, layout](out_buffer)

    # Calculate the number of blocks needed to cover the vector
    var grid_dim = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function[vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        VECTOR_WIDTH,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print("Resulting vector:", host_tensor)

fn vector_addition(
    lhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    size: Int,
):
    """The calculation to perform across the vector on the GPU."""
    var global_tid = global_idx.x
    if global_tid < size:
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]
````

---

## 🐣 Baby Step Walkthrough

### 🧱 Step 1: The Imports

```mojo
from math import ceildiv
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator
```

* `ceildiv`: Divide and round up (e.g., `ceildiv(10, 3)` → `4`)
* GPU checks: We ensure a supported GPU (NVIDIA or AMD) is present.

---

### 🔖 Step 2: Alias Setup

```mojo
alias float_dtype = DType.float32
alias VECTOR_WIDTH = 10
alias BLOCK_SIZE = 5
alias layout = Layout.row_major(VECTOR_WIDTH)
```

* Define the data type (`float32`)
* Vector size is 10
* Each GPU block will process 5 elements
* Data is stored row-wise in memory

---

### 🚀 Step 3: Inside the `main()` Function

```mojo
constrained[
    has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator(),
    "This example requires a supported GPU",
]()
```

Checks if you have a GPU — if not, it stops here.

---

### 🎮 Step 4: Connect to GPU

```mojo
var ctx = DeviceContext()
```

This gets you a **handle to the GPU**, like plugging in a controller.

---

### 🧠 Step 5: Allocate Buffers

```mojo
var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
```

We allocate memory on the GPU for:

* Input 1 (`lhs`)
* Input 2 (`rhs`)
* Output (`out`)

---

### 💾 Step 6: Fill Input Buffers

```mojo
_ = lhs_buffer.enqueue_fill(1.25)
_ = rhs_buffer.enqueue_fill(2.5)
```

Fill:

* `lhs`: \[1.25, 1.25, ..., 1.25]
* `rhs`: \[2.5, 2.5, ..., 2.5]

---

### 🔳 Step 7: Wrap Buffers as Tensors

```mojo
var lhs_tensor = LayoutTensor[float_dtype, layout](lhs_buffer)
var rhs_tensor = LayoutTensor[float_dtype, layout](rhs_buffer)
var out_tensor = LayoutTensor[float_dtype, layout](out_buffer)
```

This lets the GPU know how to interpret and access the buffer data (with layout info).

---

### 📐 Step 8: Compute Grid Dimensions

```mojo
var grid_dim = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)
```

* `VECTOR_WIDTH = 10`
* `BLOCK_SIZE = 5`
* So `grid_dim = 2`

We need 2 blocks, each handling 5 elements.

---

### 💥 Step 9: Launch the GPU Kernel

```mojo
ctx.enqueue_function[vector_addition](
    lhs_tensor,
    rhs_tensor,
    out_tensor,
    VECTOR_WIDTH,
    grid_dim=grid_dim,
    block_dim=BLOCK_SIZE,
)
```

This runs the actual **vector addition function on the GPU**.

---

## ⚙️ Inside `vector_addition()`

```mojo
var global_tid = global_idx.x
if global_tid < size:
    out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]
```

* `global_tid` is the thread index
* Each thread adds one element:

  * `out[i] = lhs[i] + rhs[i]`
* Only valid threads (less than `size`) do the work

---

### 🧳 Step 10: Copy Data Back to Host

```mojo
with out_buffer.map_to_host() as host_buffer:
    var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
    print("Resulting vector:", host_tensor)
```

* GPU → CPU memory transfer
* Wrap result as tensor
* Print result

📌 Output:

```text
Resulting vector: [3.75, 3.75, ..., 3.75]
```

Because `1.25 + 2.5 = 3.75` for each element.

---

## ✅ Final Thoughts

You just walked through a full Mojo GPU program — from memory allocation to kernel launch and result retrieval — **step by step**.

🥇 You understood:

* GPU setup
* Memory layout
* Kernel dispatch
* Result mapping

---

Want more Mojo walkthroughs like this? Or curious to build on top (e.g., matrix multiplication)?
Let me know — we can go step-by-step again.

Happy GPU hacking 💻⚡

```

---

