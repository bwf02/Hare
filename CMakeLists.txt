cmake_minimum_required(VERSION 3.16)
project(hare_ops)

# 设置 libtorch 路径
set(Torch_DIR "${LIBTORCH}/share/cmake/Torch")
message(STATUS "Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${LIBTORCH}/lib")
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8/")
# set(CUDA_HOME "/usr/local/cuda-12.8/")
# set(CUDA_TOOLKIT_PATH "/usr/local/cuda-12.8/")
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_PATH}/bin/nvcc")
# set(CMAKE_BUILD TYPE Release)

# set(Python_SITELIB "/home/buwf/miniconda3/envs/megablock/lib/python3.10/site-packages")

find_package (Python COMPONENTS Interpreter Development)
set(pybind11_DIR "${Python_SITELIB}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HARE_OPS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hare/utils/csrc/hare_ops")
set(MEGABLOCKS_OPS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hare/utils/csrc/megablock_ops")

add_subdirectory(hare/utils/csrc)
