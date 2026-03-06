# Hare: High-Performance Sparse MoE Inference via Hierarchical Structured Sparsity and Padding-Free Execution on GPU

**Hare** is a novel and high-performance inference system designed for Mixture-of-Experts (MoE) Large Language Models (LLMs). By synergistically leveraging model sparsity and expert parallelism, Hare addresses the critical compute and memory bottlenecks inherent in scaling MoE models.

Built upon the high-performance foundations of **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**, Hare introduces a customized sparse kernel library optimized specifically for **NVIDIA Hopper** architectures using TMA (Tensor Memory Accelerator).

### Key Features & Innovations

* **Padding-Free Execution**: Hare introduces a **Residual Blocked Coordinate Format** for activations, eliminating redundant token padding and costly data-format generation typically found in MoE routing mechanisms.
* **Structured Sparsity**: Utilizes a hierarchical **Block N:M** sparse format for weights to fully exploit hardware acceleration and bridge the gap between sparsity and hardware utilization.
* **DeepGEMM Integration**: Leveraging DeepGEMM's state-of-the-art GEMM implementation, Hare extends these capabilities to support dynamic, sparse workloads with maximum efficiency.

---

## Prerequisites

**Hardware Requirements**
> **Strict Requirement**: Hare relies on **TMA (Tensor Memory Accelerator)** features.
* **GPU**: NVIDIA Hopper Architecture or newer (e.g., **H100**, **5090**, **PRO 6000**).
* **VRAM**: Sufficient memory to load the target MoE model size.

**Software Requirements**
* **OS**: Linux (Ubuntu 20.04/22.04 recommended)
* **CUDA Toolkit**: **Version 12.8** or higher (Required for latest TMA/Hopper APIs).
* **Compiler**: CMake >= 3.20, GCC >= 9.0
* **Python**: Version 3.10

---

## Project Structure

```
Hare/
├── sparse_warp_spec/     # Custom sparse kernel library (DeepGEMM-based)
│   ├── csrc/            # CUDA C++ source code for JIT kernels
│   ├── sparse_gemm/     # Sparse GEMM operations and utilities
│   └── tests/           # Kernel unit tests
├── hare/                # End-to-end MoE inference system
│   ├── model/           # MoE layer implementations (DeepSeek, Mixtral, Qwen)
│   ├── infer/           # Inference code and benchmarks
│   └── utils/           # Utility functions and sparsification tools
├── benchmark/           # Performance evaluation scripts
│   ├── kernel/          # Micro-benchmarks for sparse kernels
│   ├── model/           # End-to-end model benchmarks
│   └── plot/            # Visualization scripts for results
└── profiling/           # Profiling data
```

---

## Installation

Follow these steps to set up the environment and build the custom kernels.

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/bwf02/Hare.git
cd fast-sparse
```

### 2. Environment Setup & Compilation

```bash
# Create and activate conda environment
conda create -n hare python=3.10 -y
conda activate hare

# Install Python dependencies
pip install -r requirements.txt

# Compile custom kernels
./build.sh
```

**Note:** The build process requires CUDA 12.8+ and CMake 3.20+. Ensure your `LIBTORCH` environment variable is set correctly if using a custom PyTorch installation.

---

## Benchmarks & Reproduction

Follow these steps to reproduce the experimental results and figures presented in the paper.

### Reproduce Figure 11 (Real-World Matrix Shapes)

```bash
cd benchmark/
python kernel/test_realistic.py
# Generate Figure 11
python plot/plot_realistic.py
```

### Reproduce Figure 12 (Synthetic Matrix Shapes)

```bash
python kernel/test_synthetic.py
# Generate Figure 12
python plot/plot_synthetic_vary.py
```

### Reproduce End-to-End MoE Results

```bash
# Run end-to-end benchmarks
bash model/script.sh

# Generate performance plots
python plot/plot_prefill_model.py
python plot/plot_decode_model.py
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank the contributors of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepSpeed-MoE](https://github.com/microsoft/DeepSpeed), and [vLLM](https://github.com/vllm-project/vllm) for their excellent open-source work that made this project possible.





