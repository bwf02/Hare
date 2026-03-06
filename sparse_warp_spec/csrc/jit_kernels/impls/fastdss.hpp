#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace sparse_gemm {

class DSSRuntime final: public LaunchRuntime<DSSRuntime> {
public:
  struct Args {
    int m, n, k, num_groups;
    // BCSR
    int num_row_blocks, num_col_blocks;
    int num_experts;
    // Block N:M sparsity
    int block_w, block_h;
    int bn, bm;
    int nnz_block_k;

    SparseConfig sparse_config;
    LaunchArgs launch_args;

    void *row_indices, *columns_indices, *height_offsets, *gather_index;
    // Block NM sparsity
    void *meta, *indices;
    void *values;
    void *rhs;
    void *out;
    CUtensorMap a_desc, b_desc;

  };

  static std::string generate_impl(const Args &args) {
    return fmt::format(
        R"(
    #ifdef __CUDACC_RTC__
    #include <sparse_gemm/nvrtc_std.cuh>
    #else
    #include <cuda.h>
    #include <string>
    #endif
    
    #include <sparse_gemm/impls/fastdss.cuh>
    
    using namespace sparse_gemm;
    
    static void __instantiate_kernel() {{
        auto ptr = reinterpret_cast<void*>(&fastdss_impl<
            {}, {}, {},
            {}, {},
            {},
            {},
            {}, {}, {},
            {}, {},
            {}, {},
            {},
            {},
            {}, {},
            {}, {},
            {}, {},
            {}
        >);
    }};
    )",
        // TODO: add CD dtype
        args.m, args.n, args.k, 
        args.num_row_blocks, args.num_col_blocks,
        args.num_experts,
        args.num_groups,
        args.sparse_config.block_m, args.sparse_config.block_n, args.sparse_config.block_k, 
        args.block_w, args.block_h,
        args.bn, args.bm,
        args.nnz_block_k,
        args.sparse_config.smem_config.swizzle_cd_mode,
        args.sparse_config.num_stages, args.sparse_config.num_last_stages,
        args.sparse_config.thread_config.num_tma_threads,
        args.sparse_config.thread_config.num_math_threads,
        args.sparse_config.multicast_config.num_multicast,
        args.sparse_config.multicast_config.is_multicast_on_a,
        to_string(args.sparse_config.gemm_type));
        
  }

  static void launch_impl(const cudaKernel_t &kernel,
                          const cudaLaunchConfig_t &config, Args args) {
    // TODO: optimize `args` copy
    DG_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel, 
        args.row_indices, args.columns_indices, args.height_offsets, args.gather_index,
        args.meta, args.indices, args.values, args.rhs, args.out, args.a_desc, args.b_desc
      ));
  }
};

static void fastdss(
    const int& m, 
    const int& n, 
    const int& k, 
    const int &block_h,//block nm sparsity
    const int &block_w,
    const int &bn,
    const int &bm,
    const int &nnz_block_k,
    const int& num_row_blocks, 
    const int& num_col_blocks,
    const int& num_experts,
    const torch::Tensor &row_indices,
    const torch::Tensor &column_indices,
    const torch::Tensor &height_offsets,
    const torch::Tensor &gather_index, 
    const torch::Tensor &meta,//block nm sparsity                       
    const torch::Tensor &indices,
    const torch::Tensor &values,
    const torch::Tensor &rhs, 
    const torch::Tensor &out
) {

    const auto &aligned_k = align(k, 128);
    const int nnz_k = static_cast<int>(nnz_block_k * block_w / 2);

    // block m, block k, block n
    const auto &config = get_sparse_config<SM90ArchSpec, 128, 128, 64, 2, 512>(
        GemmType::Normal, KernelType::Kernel1D2D, m, n, k, 1, 
        device_runtime->get_num_sms()
    );

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k / 2);//使用sparse tensor core

    if (get_env<int>("DG_JIT_DEBUG")){
        printf("Common Params: m=%d, n=%d, k=%d, block_m=%d, block_k=%d, block_n=%d, swizzle_a_mode=%d, nnz_k=%d, block_h=%d, block_w=%d\n",
               m, n, k, config.block_m, config.block_k, config.block_n, config.smem_config.swizzle_a_mode, nnz_k, block_h, block_w);
    }

    const auto &a_desc = make_sparse_a_desc(values, m, nnz_k,
                                          config.block_m, config.block_k / 2,
                                          config.smem_config.swizzle_a_mode);
    const auto &b_desc = make_block_sparse_b_desc(rhs, num_row_blocks * 128, n,
                                                  block_w, config.block_n,
                                                  block_w);
    // Launch
    const DSSRuntime::Args args = {
        .m = m,
        .n = n,
        .k = aligned_k,
        .num_groups = 1,
        .num_row_blocks = num_row_blocks, 
        .num_col_blocks = num_col_blocks,
        .num_experts = num_experts,
        .block_w = block_w,
        .block_h = block_h,
        .bn = bn,
        .bm = bm,
        .nnz_block_k = nnz_block_k,
        .sparse_config = config,
        .launch_args = LaunchArgs(
            config.num_sms, config.thread_config.num_threads,
            config.smem_config.smem_size, config.multicast_config.num_multicast),
        .row_indices     = row_indices.data_ptr(),
        .columns_indices = column_indices.data_ptr(),
        .height_offsets  = height_offsets.data_ptr(),
        .gather_index    = gather_index.data_ptr(),
        .meta            = meta.data_ptr(),
        .indices         = indices.data_ptr(),
        .values          = values.data_ptr(),
        .rhs             = rhs.data_ptr(),
        .out             = out.data_ptr(),
        .a_desc          = a_desc,
        .b_desc          = b_desc
    };

    const auto &code = DSSRuntime::generate(args);
    const auto &runtime = compiler->build("fastdss", code);
    DSSRuntime::launch(runtime, args);
}

}//namespace fastsparse