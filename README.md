# Hare: High-Performance Sparse MoE Inference via Hierarchical Structured Sparsity and Padding-Free Execution on GPU

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Powered By](https://img.shields.io/badge/Powered%20By-DeepGEMM-orange)](https://github.com/deepseek-ai/DeepGEMM)
[![Architecture](https://img.shields.io/badge/Arch-NVIDIA%20Hopper-green)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-76b900.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()

**Hare** is a novel and high-performance inference system designed for Mixture-of-Experts (MoE) Large Language Models (LLMs). By synergistically leveraging model sparsity and expert parallelism, Hare addresses the critical compute and memory bottlenecks inherent in scaling MoE models.

Built upon the high-performance foundations of **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**, Hare introduces a customized sparse kernel library optimized specifically for **NVIDIA Hopper** architectures using TMA (Tensor Memory Accelerator).

### Key Features & Innovations

* **🚫 Padding-Free Execution**: Hare introduces a **Residual Blocked Coordinate Format** for activations, eliminating redundant token padding and costly data-format generation typically found in MoE routing mechanisms.
* **🧩 Structured Sparsity**: Utilizes a hierarchical **Block N:M** sparse format for weights to fully exploit hardware acceleration and bridge the gap between sparsity and hardware utilization.
* **⚡ DeepGEMM Integration**: Leveraging DeepGEMM's state-of-the-art GEMM implementation, Hare extends these capabilities to support dynamic, sparse workloads with maximum efficiency.

### Performance

Extensive evaluations demonstrate that Hare consistently outperforms state-of-the-art baselines:
* **Kernel Level**: Average speedup of **6.16×** (up to 33.26×).
* **Model Level (E2E)**: Average speedup of **1.49×** (up to 1.88×).

---

## ⚙️ Prerequisites

**Hardware Requirements**
> ⚠️ **Strict Requirement**: Hare relies on **TMA (Tensor Memory Accelerator)** features.
* **GPU**: NVIDIA Hopper Architecture or newer (e.g., **H100**, **5090**, **PRO 6000**).
* **VRAM**: Sufficient memory to load the target MoE model size.

**Software Requirements**
* **OS**: Linux (Ubuntu 20.04/22.04 recommended)
* **CUDA Toolkit**: **Version 12.8** or higher (Required for latest TMA/Hopper APIs).
* **Compiler**: CMake >= 3.20, GCC >= 9.0
* **Python**: Version 3.10

---

## 📥 Installation

Follow these steps to set up the environment and build the custom kernels.

1. clone the repository.
```bash
# Replace with your actual repository URL
git clone --recursive  https://github.com/bwf02/Hare.git
cd hare
```

2. environment setup & compilation
```bash
# Create and activate environment
conda create -n hare python=3.10 -y
conda activate hare

# Install Python dependencies
pip install -r requirements.txt

# Compile custom kernels
./build.sh
```
## 📊 Benchmarks & Reproduction

Follow these steps to reproduce the experimental results and figures presented in the paper.

1. Reproduce Figure 11 (real-world matrix)
```bash
cd /benchmark
python kernel/test_realistic.py
# plot figure 11
python plot/plot_realistic.py
```
2. Reproduce Figure 12 (synthetic matrix)
```bash
python kernel/test_synthetic.py
# plot figure 12
python plot/plot_synthetic_vary.py
```

3. Reproduce end-to-end results.
```bash
python model/script.sh
# plot figure
python plot/plot_prefill_model.py
python plot/plot_decode_model.py
```

## Code Coming Soon....



