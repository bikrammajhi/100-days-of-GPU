## Fixing CUDA Version Mismatch on Google Colab

While working on Google Colab with a Tesla T4 GPU, I encountered a **CUDA version mismatch** that prevented me from compiling CUDA code properly. Here's a quick breakdown of the issue, what caused it, and how I resolved it.

### 🚨 The Problem

When running `nvcc --version` and `nvidia-smi`, I saw conflicting CUDA versions:

```bash
$ nvcc --version
Cuda compilation tools, release 12.5, V12.5.82

$ nvidia-smi
Driver Version: 550.54.15
CUDA Version: 12.4
```

The problem is that `nvcc` was from CUDA 12.5, while the **NVIDIA driver only supported CUDA 12.4**. This mismatch can lead to compilation errors or runtime incompatibilities.

### ❓ Why Did This Happen?

Google Colab sometimes pre-installs a newer CUDA toolkit (like 12.5), but the T4 GPU driver supports only an earlier version (in this case, 12.4). This results in an inconsistency between what your compiler expects and what your GPU can actually run.

### ✅ The Solution

To fix the mismatch, I reinstalled **CUDA 12.4.1**, matching the driver version shown in `nvidia-smi`. Here's the full procedure I used:

```bash
# 1. Remove any existing CUDA installation
!sudo rm -rf /usr/local/cuda*

# 2. Download and install CUDA 12.4.1 toolkit
!wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
!chmod +x cuda_12.4.1_550.54.15_linux.run
!sudo ./cuda_12.4.1_550.54.15_linux.run --silent --toolkit

# 3. Update environment variables in Python
import os
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# 4. Verify the fix
!nvcc --version
```

After these steps, `nvcc` correctly showed CUDA 12.4, and my CUDA code compiled without issues.

### 🧠 TL;DR for Version Mismatch
* **Issue:** CUDA 12.5 `nvcc` with a driver that only supports CUDA 12.4
* **Fix:** Downgrade to CUDA 12.4.1 toolkit to match driver
* **Platform:** Google Colab (Tesla T4 GPU)

