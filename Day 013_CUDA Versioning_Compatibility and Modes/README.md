
---

## 🚀 CUDA Versioning and Compatibility – Simplified

When you write CUDA programs, **two main version numbers** are important:

---

### 🧠 1. **Compute Capability (Device Feature Set)**

* Refers to **what your GPU can do**.
* It's a pair like `6.1`, `7.5`, `8.6`, etc.
* Example:

  * `Compute Capability 5.0` supports basic CUDA features.
  * `Compute Capability 8.6` supports newer features like asynchronous copies and tensor cores.

✅ You use compute capability to **compile code for a specific GPU** using flags like:

```bash
nvcc -arch=sm_86 my_kernel.cu
```

---

### 🧰 2. **CUDA Driver and Runtime Versions**

* **Driver API version**: Refers to the version of the **installed GPU driver**.
* **Runtime version**: Comes with the **CUDA Toolkit** you use to compile your code.

You can check:

```cpp
// In C++
printf("CUDA Runtime Version: %d\n", CUDA_VERSION);
```

---

## 🔁 Compatibility Rules

### 🔹 A. **Driver is Backward Compatible**

That means:

* A **new driver** can run **old CUDA applications**.

🧠 **Example**:

* You compiled your app with CUDA 11.
* You now installed a CUDA 12.5 driver.
* ✅ It **still works** because newer drivers support older apps.

---

### 🔹 B. **Driver is NOT Forward Compatible**

That means:

* An **old driver** ❌ **cannot run** applications compiled with a **newer CUDA version**.

🧠 **Example**:

* You compiled your app with CUDA 12.9.
* On the system, CUDA driver version is only 11.0.
* ❌ The app **will crash** or not load with an error like: `unspecified launch failure`.

---

### ⚠️ Driver Version Must Be >= Application Build Version

🧠 Example:

* App built with CUDA 12.2 (Driver API version 12020).
* System driver is 11.8 (Driver API version 11080).
* ❌ App won’t run.
* ✅ You must update your driver.

You can check your installed driver version:

```bash
nvidia-smi
```

---

## ⚙️ Mix-and-Match Version Rules

### 🔸 Rule 1: Only One CUDA Driver at a Time

* You can only install **one version of the CUDA driver** on a system.

✅ So all apps, plugins, and libraries on that machine must be compatible with **that installed version**.

---

### 🔸 Rule 2: Runtime Versions Must Match (Unless Statically Linked)

If your app loads shared libraries (e.g., `.so` files):

* All of them must use the **same CUDA Runtime** (unless they were statically linked).

🧠 Example:

Let’s say:

* App uses `cuFFT` and `cuBLAS`.
* `cuFFT` is compiled with CUDA 12.0.
* `cuBLAS` is compiled with CUDA 11.0.

❌ App will likely crash or behave unpredictably.
✅ Use same version, or statically link each library.

---

### 🧱 Static Linking Example

```bash
nvcc -o my_app my_app.cu -lcufft -lcublas --cudart=static
```

This makes your app include its own copy of the runtime — now version mismatches don’t matter as much.

---

## 🖼️ Visual Recap

```
           ┌────────────────────────────────────────┐
           │        Compiled with CUDA 11.2         │
           └────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────┐
       │     Installed Driver 12.9    │  ✅ Works (Backward Compatible)
       └──────────────────────────────┘

BUT

           ┌────────────────────────────────────────┐
           │        Compiled with CUDA 12.9         │
           └────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────┐
       │     Installed Driver 11.0    │  ❌ FAILS (Not Forward Compatible)
       └──────────────────────────────┘
```

---

## 🧵 TCC Mode Note

* **TCC (Tesla Compute Cluster)** mode:

  * Disables graphics (display output).
  * Used on **server GPUs** like Tesla, Quadro RTX, or A100.
  * Recommended for headless compute tasks.

---

## ✅ Summary Table

| Concept             | Meaning                                                 |
| ------------------- | ------------------------------------------------------- |
| Compute Capability  | What features your GPU hardware supports                |
| Driver API Version  | Installed driver version – must be >= app build version |
| Runtime Version     | Comes with CUDA Toolkit used during compilation         |
| Backward Compatible | New drivers run old CUDA apps ✅                         |
| Forward Compatible  | Old drivers can't run new CUDA apps ❌                   |
| Static Linking      | Avoids runtime version conflicts                        |
| TCC Mode            | Headless mode for compute-only devices                  |

---

### 🔹 **6.4. Compute Modes**

When using NVIDIA GPUs, especially on servers or HPC systems (like Tesla GPUs), you can set the **"compute mode"** of a GPU to control **how many processes or threads can access it**.

#### There are 3 Compute Modes:

| Compute Mode          | Description                                                                        | Simple Analogy                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Default**           | Multiple processes or threads can use the GPU at the same time.                    | Like a shared taxi – multiple passengers (apps) can ride together.                                            |
| **Exclusive-Process** | Only one process can use the GPU at a time. But that process can use many threads. | Like booking the entire taxi – only one customer (app) can use it, but they can bring many friends (threads). |
| **Prohibited**        | No CUDA context can be created. The GPU is **off-limits** for compute tasks.       | Like putting a "No Entry" sign – no one can use the taxi.                                                     |

---

#### 🛠 Example:

Let’s say your system has **GPU 0**, and three different programs are running:

1. **Default Mode**:

   * Program A, B, and C can all run GPU code on GPU 0.
   * All 3 use `cudaSetDevice(0)` and everything works.

2. **Exclusive-Process Mode**:

   * Only Program A can use GPU 0.
   * If B or C tries to use it, they'll get an error.
   * Inside A, multiple threads (like for deep learning) can still use the GPU.

3. **Prohibited Mode**:

   * Any call to `cudaSetDevice(0)` will fail.
   * No CUDA operations allowed on that GPU.

---

#### 💡 Notes:

* **`nvidia-smi`** is the tool used to **set these modes**.
* If `cudaSetDevice()` is not called explicitly, the CUDA runtime picks a device automatically — it might skip a prohibited/exclusive GPU.
* You can guide it using `cudaSetValidDevices()` to pick from valid ones only.

---

#### 🧠 Advanced: Compute Preemption

* On GPUs with **Pascal or newer** (Compute Capability ≥ 6.x), CUDA supports **preemption at instruction level**.
* **Why it matters**: Long-running GPU tasks won’t block the system completely. They can be interrupted more cleanly.
* Use `cudaDeviceGetAttribute()` with `cudaDevAttrComputePreemptionSupported` to check support.

---

### 🔹 **6.5. Mode Switches (Windows GPUs with Display Output)**

GPUs that drive monitors **use part of GPU memory** for the screen display (called the **primary surface**).

#### ⚠ What’s the issue?

If the user **changes screen resolution** (e.g., from 1280×1024 to 1600×1200), **more GPU memory is needed** to store the screen data.

If this happens while a CUDA program is running, it may **invalidate the GPU context** (i.e., break the app).

#### 🧪 Example:

* A CUDA program is running a model on GPU 0.
* The user **Alt+Tabs** or **launches a full-screen game** (DirectX).
* The GPU reallocates memory for the screen.
* CUDA app **crashes** with "invalid context".

---

### 🔹 **6.6. Tesla Compute Cluster (TCC) Mode**

On **Windows**, Tesla (and some Quadro) GPUs can run in a special mode called **TCC**:

#### ▶ What is TCC Mode?

* TCC = **Tesla Compute Cluster**
* It **disables all graphics/display functions** on the GPU.
* The GPU is used **only** for compute – like a CPU but massively parallel.

#### ✅ Advantages of TCC:

* No memory reserved for screen output.
* No interruptions due to display mode switches.
* Better performance for compute tasks.
* Ideal for headless servers and HPC environments.

#### ❌ Not for:

* GPUs connected to monitors.
* Interactive graphical applications.

---

#### 🧠 How to Enable TCC?

```bash
nvidia-smi -g 0 -dm 1
```

* Here `-g 0` selects GPU 0, `-dm 1` sets mode to TCC.

> Use `nvidia-smi -q -d COMPUTE` to view compute mode.

---

### ✅ Summary Table

| Feature           | What It Controls                                | Example                                                        |
| ----------------- | ----------------------------------------------- | -------------------------------------------------------------- |
| **Compute Mode**  | How many apps or threads can use the GPU.       | Only one app can access in **exclusive-process**.              |
| **Mode Switches** | Display changes causing memory reallocation.    | Alt+Tab or resolution change can crash CUDA.                   |
| **TCC Mode**      | Turns GPU into compute-only mode (no graphics). | Great for servers or AI training, avoids display interference. |
---
