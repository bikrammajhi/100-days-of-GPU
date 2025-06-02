# 🚀 CUDA & GPU Optimization Resources

> *A comprehensive collection of cutting-edge CUDA programming, GPU optimization, and high-performance computing resources*

---

## 📋 Table of Contents

- [🏆 Featured Resources](#-featured-resources)
- [📚 GitHub Repositories](#-github-repositories)
  - [⚡ CUDA Fundamentals & Optimization](#-cuda-fundamentals--optimization)
  - [🤖 LLM Inference Implementations](#-llm-inference-implementations)
  - [🎨 Diffusion Models & Computer Vision](#-diffusion-models--computer-vision)
  - [🔥 Triton & Advanced Kernels](#-triton--advanced-kernels)
  - [🛠️ WebUI & Development Tools](#️-webui--development-tools)
  - [⭐ Awesome Lists & Collections](#-awesome-lists--collections)
- [📖 Educational Content](#-educational-content)
  - [🧮 Matrix Operations & Linear Algebra](#-matrix-operations--linear-algebra)
  - [💻 CUDA Programming Techniques](#-cuda-programming-techniques)
  - [🖥️ CPU Optimization & General HPC](#️-cpu-optimization--general-hpc)
  - [🎯 Hardware Selection & Performance](#-hardware-selection--performance)
- [🎥 Video Content](#-video-content)
- [📰 Newsletter & Substacks](#-newsletter--substacks)

---

## 🏆 Featured Resources

<div align="center">

### 🌟 **Essential Starting Points** 🌟

[![GitHub](https://img.shields.io/badge/GitHub-LeetCUDA-181717?logo=github&style=for-the-badge)](https://github.com/xlite-dev/LeetCUDA) 
[![GitHub](https://img.shields.io/badge/GitHub-CUDA%20Optimizations-181717?logo=github&style=for-the-badge)](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)

[![Blog](https://img.shields.io/badge/Blog-Optimizing%20LayerNorm-e91e63?logo=hashnode&style=for-the-badge)](https://aryagxr.com/blogs/cuda-optimizing-layernorm)
[![Blog](https://img.shields.io/badge/Blog-Tensor%20Core%20MatMul-2196f3?logo=medium&style=for-the-badge)](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)

</div>

---

## 📚 GitHub Repositories

### ⚡ CUDA Fundamentals & Optimization

| Resource | Description | Focus Area |
|----------|-------------|------------|
| [![GitHub](https://img.shields.io/badge/GitHub-NVIDIA%20SGEMM%20Practice-181717?logo=github)](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) | 🔢 **SGEMM Implementation** | Matrix Multiplication |
| [![GitHub](https://img.shields.io/badge/GitHub-cugrad-181717?logo=github)](https://github.com/dmaivel/cugrad) | 🧠 **Automatic Differentiation** | Deep Learning Backend |
| [![GitHub](https://img.shields.io/badge/GitHub-maxas%20SGEMM-181717?logo=github)](https://github.com/NervanaSystems/maxas/wiki/SGEMM) | 📐 **Assembly-level SGEMM** | Low-level Optimization |
| [![GitHub](https://img.shields.io/badge/GitHub-CUDA%20Algorithm%20Optimization-181717?logo=github)](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | 🎯 **Algorithm Optimization** | Performance Tuning |
| [![GitHub](https://img.shields.io/badge/GitHub-Tiny%20Flash%20Attention-181717?logo=github)](https://github.com/66RING/tiny-flash-attention) | ⚡ **Attention Mechanism** | Transformer Optimization |

---

### 🤖 LLM Inference Implementations

<details>
<summary><strong>🔽 Pure CUDA Implementations</strong></summary>

| Language | Resource | Model Support |
|----------|----------|---------------|
| **CUDA C++** | [![GitHub](https://img.shields.io/badge/GitHub-llama3.cu-181717?logo=github)](https://github.com/abhisheknair10/llama3.cu) | 🦙 Llama 3 8B |
| **CUDA C++** | [![GitHub](https://img.shields.io/badge/GitHub-llama3.cuda-181717?logo=github)](https://github.com/likejazz/llama3.cuda) | 🦙 Llama 3 |
| **CUDA C++** | [![GitHub](https://img.shields.io/badge/GitHub-llama2.cu%20(rogerallen)-181717?logo=github)](https://github.com/rogerallen/llama2.cu) | 🦙 Llama 2 |
| **Pure CUDA** | [![GitHub](https://img.shields.io/badge/GitHub-llama2.cu%20(ankan--ban)-181717?logo=github)](https://github.com/ankan-ban/llama2.cu) | 🦙 Llama 2 |

</details>

<details>
<summary><strong>🔽 Multi-Language Implementations</strong></summary>

| Language | Resource | Highlights |
|----------|----------|------------|
| **Official** | [![GitHub](https://img.shields.io/badge/GitHub-Llama%20Official-181717?logo=github)](https://github.com/meta-llama/llama/tree/main) | 🏢 Meta's Reference |
| **Pure C** | [![GitHub](https://img.shields.io/badge/GitHub-llama2.c-181717?logo=github)](https://github.com/karpathy/llama2.c) | 🎯 Minimal Dependencies |
| **Python** | [![GitHub](https://img.shields.io/badge/GitHub-llama2.py-181717?logo=github)](https://github.com/tairov/llama2.py) | 🐍 Educational |
| **NumPy** | [![GitHub](https://img.shields.io/badge/GitHub-llama.np-181717?logo=github)](https://github.com/hscspring/llama.np) | 🔢 Pure NumPy |
| **NumPy** | [![GitHub](https://img.shields.io/badge/GitHub-llama3.np-181717?logo=github)](https://github.com/likejazz/llama3.np) | 🔢 Llama 3 NumPy |

</details>

---

### 🎨 Diffusion Models & Computer Vision

| Model Type | Resource | Specialization |
|------------|----------|----------------|
| **Stable Diffusion** | [![GitHub](https://img.shields.io/badge/GitHub-stable--diffusion.cpp-181717?logo=github)](https://github.com/leejet/stable-diffusion.cpp) | 🎨 C++ Implementation |
| **UNet** | [![GitHub](https://img.shields.io/badge/GitHub-unet.cu-181717?logo=github)](https://github.com/clu0/unet.cu) | 🏗️ Pure CUDA UNet |
| **Vision Transformer** | [![GitHub](https://img.shields.io/badge/GitHub-vit.cpp-181717?logo=github)](https://github.com/ggerganov/vit.cpp) | 👁️ ViT in C++ |
| **Minimal Diffusion** | [![GitHub](https://img.shields.io/badge/GitHub-minDiffusion-181717?logo=github)](https://github.com/cloneofsimo/minDiffusion) | 📚 Educational |
| **LoRA** | [![GitHub](https://img.shields.io/badge/GitHub-LoRA-181717?logo=github)](https://github.com/cloneofsimo/lora) | 🔧 Parameter Efficient |
| **Low VRAM** | [![GitHub](https://img.shields.io/badge/GitHub-Stable%20Diffusion%20Low%20VRAM-181717?logo=github)](https://github.com/basujindal/stable-diffusion) | 💾 Memory Optimized |

---

### 🔥 Triton & Advanced Kernels

| Framework | Resource | Innovation |
|-----------|----------|------------|
| **Triton** | [![GitHub](https://img.shields.io/badge/GitHub-Liger%20Kernel-181717?logo=github)](https://github.com/linkedin/Liger-Kernel) | 🦁 LinkedIn's Optimizations |
| **Flux** | [![GitHub](https://img.shields.io/badge/GitHub-Flux-181717?logo=github)](https://github.com/black-forest-labs/flux/tree/main) | 🌊 Advanced Diffusion |
| **Fast Inference** | [![GitHub](https://img.shields.io/badge/GitHub-stable--fast-181717?logo=github)](https://github.com/chengzeyi/stable-fast) | ⚡ Speed Optimizations |
| **Attention** | [![GitHub](https://img.shields.io/badge/GitHub-attorch-181717?logo=github)](https://github.com/BobMcDear/attorch/tree/main) | 🧠 Attention Kernels |
| **Transformers** | [![GitHub](https://img.shields.io/badge/GitHub-kernl-181717?logo=github)](https://github.com/ELS-RD/kernl) | 🚀 Kernel Optimizations |
| **LLM Serving** | [![GitHub](https://img.shields.io/badge/GitHub-lightllm-181717?logo=github)](https://github.com/ModelTC/lightllm/tree/main) | 🔥 Production Ready |

---

### 🛠️ WebUI & Development Tools

[![GitHub](https://img.shields.io/badge/GitHub-Stable%20Diffusion%20WebUI%20Discussion-181717?logo=github)](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/6601)
> 💬 **Community Discussions** - WebUI optimization techniques

---

### ⭐ Awesome Lists & Collections

| Collection | Focus | Curation Quality |
|------------|-------|------------------|
| [![Awesome](https://img.shields.io/badge/Awesome-CUDA%20&%20HPC-ff6b6b?logo=awesome-lists)](https://github.com/coderonion/awesome-cuda-and-hpc) | 🖥️ **CUDA & HPC** | Comprehensive |
| [![Awesome](https://img.shields.io/badge/Awesome-DiT%20Inference-4ecdc4?logo=awesome-lists)](https://github.com/xlite-dev/Awesome-DiT-Inference) | 🎯 **DiT Inference** | Specialized |

---

## 📖 Educational Content

### 🧮 Matrix Operations & Linear Algebra

<div align="center">

#### **🏅 GPU Matrix Multiplication Masters**

</div>

| Topic | Resource | Difficulty | GPU Focus |
|-------|----------|------------|-----------|
| **Tensor Cores** | [![Blog](https://img.shields.io/badge/Blog-Tensor%20Core%20MatMul-e91e63?logo=hashnode)](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) | 🔴 Advanced | H100/A100 |
| **MMA Instructions** | [![Blog](https://img.shields.io/badge/Blog-MMA%20MatMul-2196f3?logo=medium)](https://www.spatters.ca/mma-matmul) | 🔴 Advanced | Modern GPUs |
| **CUDA Optimization** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Matrix%20Multiplication%20Optimization-2196f3?logo=medium)](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) | 🟡 Intermediate | General |
| **SGEMV Optimization** | [![Blog](https://img.shields.io/badge/Blog-Optimizing%20SGEMV%20CUDA-2196f3?logo=medium)](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/) | 🟡 Intermediate | Vector Operations |
| **Softmax CUDA** | [![Blog](https://img.shields.io/badge/Blog-Optimizing%20Softmax%20CUDA-2196f3?logo=medium)](https://maharshi.bearblog.dev/optimizing-softmax-cuda/) | 🟡 Intermediate | Neural Networks |
| **Fast GPU MatMul** | [![Blog](https://img.shields.io/badge/Blog-Fast%20GPU%20Matrix%20Multiplication-2196f3?logo=medium)](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html) | 🟢 Beginner | General |
| **SGEMM GPU** | [![Blog](https://img.shields.io/badge/Blog-SGEMM%20GPU%20Optimization-2196f3?logo=medium)](https://salykova.github.io/sgemm-gpu) | 🟡 Intermediate | Single Precision |
| **SGEMM Notes** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20SGEMM%20Optimization%20Notes-2196f3?logo=medium)](https://linn-ylz.com/Computer-Science/CUDA/CUDA-SGEMM-optimization-notes/) | 🟡 Intermediate | Detailed Analysis |
| **FP32 GEMM** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20GEMM%20FP32%20Optimization-2196f3?logo=medium)](https://code.hitori.moe/post/cuda-gemm-fp32-optimization/) | 🔴 Advanced | Float32 |
| **Transpose Opt** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Transpose%20Optimization-2196f3?logo=medium)](https://code.hitori.moe/post/cuda-transpose-optimization/) | 🟡 Intermediate | Memory Access |
| **GPU MatMul** | [![Blog](https://img.shields.io/badge/Blog-GPU%20Matrix%20Multiply-2196f3?logo=medium)](https://indii.org/blog/gpu-matrix-multiply/) | 🟢 Beginner | Educational |
| **FP16 MMA** | [![Blog](https://img.shields.io/badge/Blog-Two%20Stage%20FP16%20MMA-2196f3?logo=medium)](https://www.spatters.ca/two-stage-fp16-mma) | 🔴 Advanced | Half Precision |

---

### 💻 CUDA Programming Techniques

<div align="center">

#### **⚙️ Core CUDA Concepts**

</div>

| Concept | Resource | Learning Path |
|---------|----------|---------------|
| **CUDA Basics** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Basics-2196f3?logo=medium)](https://tinkerd.net/blog/machine-learning/cuda-basics/#wrapping-up) | 🟢 Start Here |
| **Comprehensive Guide** | [![Blog](https://img.shields.io/badge/Blog-Lei%20Mao%20CUDA%20Tags-2196f3?logo=medium)](https://leimao.github.io/tags/CUDA/) | 🟡 Progressive |
| **GPU Programming** | [![Blog](https://img.shields.io/badge/Blog-BRRR%20Intro-2196f3?logo=medium)](https://horace.io/brrr_intro.html) | 🟢 Fun Intro |
| **Task Parallelism** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Task%20Parallelism-2196f3?logo=medium)](https://enccs.github.io/cuda/3.02_TaskParallelism/) | 🟡 Intermediate |
| **Parallel Reduction** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Parallel%20Reduction-2196f3?logo=medium)](https://enccs.github.io/cuda/3.01_ParallelReduction/) | 🟡 Intermediate |
| **Optimization Tips** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20Kernel%20Optimization%20Tips-2196f3?logo=medium)](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-ii---cuda-kernel-optimization-tips) | 🔴 Advanced |
| **Documentation** | [![Blog](https://img.shields.io/badge/Blog-Julian%20Roth%20CUDA%20Documentation-2196f3?logo=medium)](https://julianroth.org/documentation/cuda/) | 📚 Reference |

---

### 🖥️ CPU Optimization & General HPC

| Focus Area | Resource | Comparison |
|------------|----------|------------|
| **CPU MatMul** | [![Blog](https://img.shields.io/badge/Blog-Matrix%20Multiplication%20CPU-2196f3?logo=medium)](https://salykova.github.io/matmul-cpu) | 🖥️ vs GPU |
| **Fast CPU MMM** | [![Blog](https://img.shields.io/badge/Blog-Fast%20Matrix%20Multiplication%20CPU-2196f3?logo=medium)](https://siboehm.com/articles/22/Fast-MMM-on-CPU) | ⚡ Optimized |
| **Algorithmica** | [![Blog](https://img.shields.io/badge/Blog-Algorithmica%20Matrix%20Multiplication-2196f3?logo=medium)](https://en.algorithmica.org/hpc/algorithms/matmul/) | 📐 Theory |
| **Mathematical** | [![Blog](https://img.shields.io/badge/Blog-Strange%20Matrix%20Multiplications-2196f3?logo=medium)](https://www.thonking.ai/p/strangely-matrix-multiplications) | 🤔 Insights |
| **CUDA vs CPU** | [![Blog](https://img.shields.io/badge/Blog-CUDA%20MatMul%20Optimization-00d4aa?logo=dev.to)](https://siboehm.com/articles/22/CUDA-MMM) | ⚖️ Comparison |

---

### 🎯 Hardware Selection & Performance

| Topic | Resource | Audience |
|-------|----------|----------|
| **GPU Selection** | [![Blog](https://img.shields.io/badge/Blog-Which%20GPU%20for%20Deep%20Learning-2196f3?logo=medium)](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#more-6) | 🛒 Buyers Guide |
| **42dot LLM** | [![Blog](https://img.shields.io/badge/Blog-42dot%20LLM%201.3B-2196f3?logo=medium)](https://42dot.ai/blog/178) | 🏢 Industry |

---

## 🎥 Video Content

<div align="center">

### **🔥 Performance Deep Dives**

[![YouTube](https://img.shields.io/badge/YouTube-Outperforming%20cuBLAS-FF0000?logo=youtube&style=for-the-badge)](https://www.youtube.com/watch?v=ErTmTCRP1_U)

*Learn how to beat NVIDIA's optimized BLAS library with custom kernels*

</div>

---

## 📰 Newsletter & Substacks

<div align="center">

### **📊 Performance Engineering Insights**

[![Substack](https://img.shields.io/badge/Substack-Outperforming%20cuBLAS-ff6b35?logo=substack&style=for-the-badge)](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

*H100 optimization worklog and techniques*

</div>

---

## 🎯 Quick Navigation

<div align="center">

| 🚀 **Getting Started** | 🔬 **Deep Learning** | ⚡ **Performance** | 🛠️ **Tools** |
|:---:|:---:|:---:|:---:|
| [CUDA Basics](#-cuda-programming-techniques) | [LLM Inference](#-llm-inference-implementations) | [Matrix Ops](#-matrix-operations--linear-algebra) | [GitHub Repos](#-github-repositories) |
| [Hardware Guide](#-hardware-selection--performance) | [Diffusion Models](#-diffusion-models--computer-vision) | [Optimization](#-cuda-fundamentals--optimization) | [Awesome Lists](#-awesome-lists--collections) |

</div>

---

<div align="center">

### 💡 **Pro Tips for Navigation**

- 🔍 Use browser search (Ctrl+F) to find specific topics
- 📱 Difficulty levels: 🟢 Beginner | 🟡 Intermediate | 🔴 Advanced  
- ⭐ Star repositories you find valuable
- 🔄 Resources are regularly updated - check back often!

---

**🎉 Happy Optimizing! 🎉**

*Built with ❤️ for the CUDA community*

</div>