#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "utils/layout.hpp"

#include "jit_kernels/impls/smxx_layout.hpp"
#include "jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d2d.hpp"
#include "jit_kernels/impls/fastssd.hpp"
#include "jit_kernels/impls/fastdss.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME sparse_gemm_cpp
#endif

namespace sparse_gemm {

torch::Tensor ssd(
    const int &m, 
    const int &n,
    const int &k,
    const int &block_h,
    const int &block_w,
    const int &bn,
    const int &bm,
    const int &nnz_block_k,
    const torch::Tensor &row_indices,
    const torch::Tensor &column_indices,
    const torch::Tensor &height_offsets, 
    const torch::Tensor &gather_index, 
    const torch::Tensor &meta,
    const torch::Tensor &block_indices,
    const torch::Tensor &values,
    const torch::Tensor &rhs
) {
    const auto &[num_col_blocks] = get_shape<1>(column_indices);
    const auto &[num_blocks] = get_shape<1>(row_indices);
    const auto &[num_experts] = get_shape<1>(height_offsets);
    const auto num_row_blocks = num_blocks / num_col_blocks;

    if (get_env<int>("DG_JIT_DEBUG")){
        printf("ssd called with m=%d, n=%d, k=%d\n", m, n, k);
        printf("num_blocks=%d, num_col_blocks=%d, num_row_blocks=%d\n",
            num_blocks, num_col_blocks, num_row_blocks);
    }

    // Create output tensor with shape (num_row_blocks * 128, n) and dtype bfloat16.
    auto out = torch::zeros(
        {static_cast<int64_t>(num_row_blocks) * 128, static_cast<int64_t>(n)},
        torch::TensorOptions().dtype(torch::kBFloat16).device(rhs.device())
    );

    // Do nothing if the problem is empty
    if (m == 0)
        return out;

    const auto &arch_major = device_runtime->get_arch_major();

    if (arch_major >= 9) {
        fastssd(m, n, k, block_h, block_w, bn, bm, nnz_block_k,
            num_row_blocks, num_col_blocks, num_experts, 
            row_indices, column_indices, height_offsets, gather_index, 
            meta, block_indices, values, rhs, out);

    } else {
        DG_HOST_UNREACHABLE("Unknown kernel or architecture");
    }
    return out;
}

torch::Tensor dss(
    const int &m, 
    const int &n,
    const int &k,
    const int &block_h,
    const int &block_w,
    const int &bn,
    const int &bm,
    const int &nnz_block_k,
    const torch::Tensor &row_indices,
    const torch::Tensor &column_indices,
    const torch::Tensor &height_offsets, 
    const torch::Tensor &gather_index, 
    const torch::Tensor &meta,
    const torch::Tensor &block_indices,
    const torch::Tensor &values,
    const torch::Tensor &rhs
) {
    const auto &[num_col_blocks] = get_shape<1>(column_indices);
    const auto &[num_blocks] = get_shape<1>(row_indices);
    const auto &[num_experts] = get_shape<1>(height_offsets);
    const auto num_row_blocks = num_blocks / num_col_blocks;

    // Create output tensor with shape (m, n) and dtype bfloat16.
    auto out = torch::zeros(
        {static_cast<int64_t>(n), static_cast<int64_t>(m)},
        torch::TensorOptions().dtype(torch::kBFloat16).device(rhs.device())
    );

    // Do nothing if the problem is empty
    if (m == 0)
        return out;

    const auto &arch_major = device_runtime->get_arch_major();

    if (arch_major >= 9) {
        fastdss(m, n, k, block_h, block_w, bn, bm, nnz_block_k,
            num_row_blocks, num_col_blocks, num_experts, 
            row_indices, column_indices, height_offsets, gather_index, 
            meta, block_indices, values, rhs, out);

    } else {
        DG_HOST_UNREACHABLE("Unknown kernel or architecture");
    }
    return out;
}



torch::Tensor ssd_naive(
    const int &m, 
    const int &n,
    const int &k,
    const int &block_h,
    const int &block_w,
    const int &bn,
    const int &bm,
    const int &nnz_block_k,
    const torch::Tensor &row_indices,
    const torch::Tensor &column_indices,
    const torch::Tensor &height_offsets, 
    const torch::Tensor &gather_index, 
    const torch::Tensor &meta,
    const torch::Tensor &block_indices,
    const torch::Tensor &values,
    const torch::Tensor &rhs
) {
    const auto &[num_col_blocks] = get_shape<1>(column_indices);
    const auto &[num_blocks] = get_shape<1>(row_indices);
    const auto &[num_experts] = get_shape<1>(height_offsets);
    const auto num_row_blocks = num_blocks / num_col_blocks;

    if (get_env<int>("DG_JIT_DEBUG")){
        printf("ssd called with m=%d, n=%d, k=%d\n", m, n, k);
        printf("num_blocks=%d, num_col_blocks=%d, num_row_blocks=%d\n",
            num_blocks, num_col_blocks, num_row_blocks);
    }

    // Create output tensor with shape (num_row_blocks * 128, n) and dtype bfloat16.
    auto out = torch::zeros(
        {static_cast<int64_t>(num_row_blocks) * 128, static_cast<int64_t>(n)},
        torch::TensorOptions().dtype(torch::kBFloat16).device(rhs.device())
    );

    // Do nothing if the problem is empty
    if (m == 0)
        return out;

    const auto &arch_major = device_runtime->get_arch_major();

    if (arch_major >= 9) {
        fastssd_naive(m, n, k, block_h, block_w, bn, bm, nnz_block_k,
            num_row_blocks, num_col_blocks, num_experts, 
            row_indices, column_indices, height_offsets, gather_index, 
            meta, block_indices, values, rhs, out);

    } else {
        DG_HOST_UNREACHABLE("Unknown kernel or architecture");
    }
    return out;
}



} // namespace sparse_gemm

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace sparse_gemm;

    m.doc() = "DeepGEMM C++ library";

    // Runtime
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });

    // JIT
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_torch) {
        DG_HOST_ASSERT(get_env("DG_JIT_USE_NVRTC", 0) == 0 and "Currently only support NVCC");
        compiler = std::make_shared<NVCCCompiler>(library_root_path, cuda_home_path_by_torch);
        KernelRuntime::set_cuda_home(cuda_home_path_by_torch);
    });

    // Raw kernels or functions
    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
    
    m.def("ssd", &ssd,
        py::arg("m"),
        py::arg("n"),
        py::arg("k"),
        py::arg("block_h"),
        py::arg("block_w"),
        py::arg("bn"),
        py::arg("bm"),
        py::arg("nnz_block_k"),
        py::arg("row_indices"),
        py::arg("column_indices"),
        py::arg("height_offsets"),
        py::arg("gather_index"),
        py::arg("meta"),
        py::arg("block_indices"),
        py::arg("values"),
        py::arg("rhs"));
    
    m.def("ssd_naive", &ssd_naive,
        py::arg("m"),
        py::arg("n"),
        py::arg("k"),
        py::arg("block_h"),
        py::arg("block_w"),
        py::arg("bn"),
        py::arg("bm"),
        py::arg("nnz_block_k"),
        py::arg("row_indices"),
        py::arg("column_indices"),
        py::arg("height_offsets"),
        py::arg("gather_index"),
        py::arg("meta"),
        py::arg("block_indices"),
        py::arg("values"),
        py::arg("rhs"));

    m.def("dss", &dss,
        py::arg("m"),
        py::arg("n"),
        py::arg("k"),
        py::arg("block_h"),
        py::arg("block_w"),
        py::arg("bn"),
        py::arg("bm"),
        py::arg("nnz_block_k"),
        py::arg("row_indices"),
        py::arg("column_indices"),
        py::arg("height_offsets"),
        py::arg("gather_index"),
        py::arg("meta"),
        py::arg("block_indices"),
        py::arg("values"),
        py::arg("rhs"));
}
