
# fast-sparse

This repository contains the `fast-sparse` project and related components used for building high-performance sparse GEMM (and related) kernels. The instructions below follow the original minimal steps and expand them into a full, practical workflow for cloning, preparing submodules, building native C++/CUDA components, and installing the Python extension.

## Prerequisites

- Linux (Ubuntu recommended)
- Python 3.8+ (3.10 used in CI/dev container)
- CUDA toolkit (matching the installed GPU drivers)
- NVCC in `CUDA_HOME` (e.g. `/usr/local/cuda`)
- PyTorch with CUDA support (libtorch or `torch` Python package)
- CMake (>= 3.10)
- `git`, `make`, `gcc`/`g++`
- `python3-dev` / Python headers for building extensions

Optional but recommended:
- `pip` and a virtualenv or conda environment
- `pybind11` if building via CMake (the `setup.py` build flow will handle Python extension building via `torch.utils.cpp_extension`)

## Quick start (recommended)

These commands reproduce the original short steps and provide the typical flow used during development.

1. Clone the repository (with submodules):

```bash
git clone -b fp16 git@gitee.com:bu-wf/fast-sparse.git --recursive
cd fast-sparse
git submodule update --remote --recursive
git submodule update --init --recursive
```

2. Run the top-level build script (this is project-specific and may perform project-wide preparation):

```bash
./build.sh
```

3. Build and install the `sparse_warp_specialization` component (Py extension and helper libs):

```bash
cd sparse_warp_spec
./install.sh
```

The `install.sh` script in that folder typically calls `python setup.py build` and creates a `.so` / Python extension that is symlinked into the current directory. If you need an editable install, run `pip install -e .` from that folder instead.

4. Run Hare MoE
```bash
python benchmark/deepseek_hare.py --time --batch_size 1 --layer --flash --experts 8 --intermediate_size 2048
# multi-gpu
torchrun --nproc_per_node 2 benchmark/deepseek_hare.py --time --batch_size 1 --layer --flash --experts 8 --intermediate_size 2048
```


5. venom
```bash
# cusparseLt
./src/benchmark_spmm --sparsity-type csr --spmm cuSparseLt --gemm cuBlas --precision half --m 1024 --k 4096 --n 768 --d 0.5 --check
# CLASP
./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 2 --m 1024 --k 256 --n 256 --d 0.2 --check
# Spuntik 
./src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision half --acc_t fp16 --m 256 --k 256 --n 256 --d 0.1 
# Spatha 
./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m 1024 --k 4096 --n 4096 --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check
```

6. SSMM
```bash
./build/./benchmark/./horizontal_ssmm_benchmark -m 1024 -k 4096 -n 4096 --vector_length 128 --seed 42 -t
```

## run kernel benchmark