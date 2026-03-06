#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build
rm -rf *.egg-info
mkdir build

#https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/
mkdir -p third_party/cusparselt
cd third_party/cusparselt
rm -rf libcusparse_lt-linux-x86_64-0.5.0.1-archive*
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz
tar -xf libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz
cd $SCRIPT_DIR

pip install . --no-build-isolation -v

cd third_party/Samoyeds/
python setup.py install
cd $SCRIPT_DIR

cd third_party/Samoyeds/Samoyeds-Kernel
./build.sh
cd $SCRIPT_DIR

cd sparse_warp_spec
pip uninstall sparse_gemm -y && ./install.sh
cd $SCRIPT_DIR

export CUSPARSELT_PATH="$SCRIPT_DIR/third_party/cusparselt/libcusparse_lt-linux-x86_64-0.5.0.1-archive"
cd third_party/venom
mkdir -p build && cd build

GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

if [ "$GPU_CC" = "8.0" ]; then
    CUDA_COMPUTE_CAPABILITY=80
elif [ "$GPU_CC" = "8.6" ]; then
    CUDA_COMPUTE_CAPABILITY=86
elif [ "$GPU_CC" = "8.9" ]; then
    CUDA_COMPUTE_CAPABILITY=89
elif [ "$GPU_CC" = "9.0" ]; then
    CUDA_COMPUTE_CAPABILITY=90
elif [ "$GPU_CC" = "12.0" ]; then
    CUDA_COMPUTE_CAPABILITY=120
else
    echo "Unsupported GPU compute capability: $GPU_CC"
    exit 1
fi

cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="$CUDA_COMPUTE_CAPABILITY" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF && make -j 16
