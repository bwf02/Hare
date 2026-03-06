#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <sparse_gemm/common/utils.cuh>
#include <sparse_gemm/common/scheduler.cuh>
#include <sparse_gemm/common/sm90_utils.cuh>

#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

namespace sparse_gemm {

using namespace sparse_gemm::sm90;

template <int SHAPE_M, int SHAPE_N, int SHAPE_K,
          int NUM_ROW_BLOCK, int NUM_COL_BLOCK,
          int NUM_EXPERTS,
          int kNumGroups,
          int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int BLOCK_W, int BLOCK_H,
          int BN, int BM,
          int NNZ_BLOCK_K,
          int kSwizzleDMode,
          int kNumStages, int kNumLastStages,
          int kNumTMAThreads, int kNumMathThreads,
          int kNumTMAMulticast, bool kIsTMAMulticastOnA,
          GemmType kGemmType>
__global__ void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
    fastdss_impl(
        int* __restrict__ row_indices,
        int* __restrict__ column_indices,
        int* __restrict__ height_offset,
        int* __restrict__ gather_index,
        uint *__restrict__ meta,
        uint *__restrict__ indices,
        __nv_bfloat16 *__restrict__ value,
        __nv_bfloat16 *__restrict__ rhs,
        __nv_bfloat16 *__restrict__ out,
        const __grid_constant__ CUtensorMap a_desc,
        const __grid_constant__ CUtensorMap b_desc
    )
{
    static constexpr int PREFETCHEDCOL = kNumTMAThreads;
    static constexpr int NUM_BLOCK_K = SHAPE_K / BLOCK_W;

    static constexpr int NNZ_K = (NNZ_BLOCK_K * BLOCK_W) / 2;
    static constexpr int MCOL_K = NNZ_K / 8; //计算每行有多少个uint(meta的单位), 一个uint在一行里面付出16bit,就是8个元素的距离
    static constexpr int BLOCK_PER_TILE = BLOCK_K / BLOCK_W;//每个tile里面是多少个
    // metadata
    static constexpr int META_NNZ_K = NNZ_K / 8; //计算每行的meta需要多少个uint, 一个uint在一行里面付出16bit,就是8个元素的距离
    static constexpr int META_MROW = 2; //每个uint meta存储跨度是2行
    static constexpr int META_ELEMENT = 8;//每个uint meta在一行里面是8个元素
    // Shared memory
    static constexpr int SMEM_D_SIZE = BLOCK_M * BLOCK_N; //bf16
    static constexpr int SMEM_A_SIZE_PER_STAGE = BLOCK_M * (BLOCK_K / 2);
    static constexpr int SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K;
    static constexpr int SMEM_I_SIZE_PER_STAGE = PREFETCHEDCOL; //int
    static constexpr int VAILD_B_K = NUM_ROW_BLOCK * BLOCK_M;
    static constexpr int VAILD_A_K = (VAILD_B_K * BN / BM) / 2;
    
    // Configs
    const uint32_t num_iterations = ceil_div(VAILD_A_K, BLOCK_K); // num rows  block size

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("num_iterations=%u\n", num_iterations);
    //     printf("SMEM_A_SIZE_PER_STAGE=%d SMEM_B_SIZE_PER_STAGE=%d SMEM_I_SIZE_PER_STAGE=%d\n",
    //            SMEM_A_SIZE_PER_STAGE, SMEM_B_SIZE_PER_STAGE, SMEM_I_SIZE_PER_STAGE);
    //     printf("VAILD_B_K=%d VAILD_A_K=%d SMEM_D_SIZE=%d PREFETCHEDCOL=%d BLOCK_M=%d BLOCK_N=%d BLOCK_K=%d\n",
    //            VAILD_B_K, VAILD_A_K, SMEM_D_SIZE, PREFETCHEDCOL, BLOCK_M, BLOCK_N, BLOCK_K);
    // }
    // const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == kNumMathThreads) {
        // NOTES: `reinterpret_cast` must be here, or NVRTC will fail
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&a_desc));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&b_desc));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) int smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    uint *smem_indices = reinterpret_cast<uint *>(smem_buffer);
    __nv_bfloat16 *smem_d = reinterpret_cast<__nv_bfloat16 *>(smem_indices + SMEM_I_SIZE_PER_STAGE * 2); // 对indices开double buffer,方便预取
    __nv_bfloat16 *smem_a[kNumStages];
    __nv_bfloat16 *smem_b[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++i){
        smem_a[i] = reinterpret_cast<__nv_bfloat16 *>(smem_d + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_bfloat16 *>(smem_d + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    }

    // Fill barriers
    barrier *barrier_start_ptr = reinterpret_cast<barrier*>(smem_d + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    // TMA Barrier for both divisible and non-divisible cases
    barrier *full_barriers[kNumStages];
    barrier *empty_barriers[kNumStages];

    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 1, "Too many TMA multicast");

    #pragma nv_diag_suppress static_var_with_dynamic_init
    if (threadIdx.x == kNumMathThreads) {
        for (uint32_t i = 0; i < kNumStages; i++) { 
            init(full_barriers[i], blockDim.x);
            init(empty_barriers[i], blockDim.x);
        }
        // cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();// ensure all threads see initialized barriers

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 192;
    constexpr uint32_t kNumWarpTMA = kNumTMAThreads / 32;
    constexpr uint32_t kNumWarpMath = kNumMathThreads / 32;
    // if(threadIdx.x == 0)
    // printf("kNumStages:%d, kNumLastStages:%d, kNumTMAThreads:%d, kNumMathThreads:%d\n", kNumStages, kNumLastStages, kNumTMAThreads, kNumMathThreads);

    // Block scheduler
    uint32_t m_index, n_index, num_cols, k_index;
    auto scheduler = DSScheduler<kGemmType,
                        SHAPE_M, SHAPE_N,
                        BLOCK_M, BLOCK_N,
                        NUM_ROW_BLOCK, NUM_COL_BLOCK, NUM_EXPERTS, 
                        kNumTMAMulticast, true, 16>(row_indices, column_indices, height_offset);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data 
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_index, n_index, num_cols, k_index)) {
            int k_nnz_start = k_index * (BN / BM) / 2;
            int k_nnz_end = (k_index + VAILD_B_K) * (BN / BM) / 2;
            int indices_start = k_index / (BLOCK_W * BM) * BN;
            int indices_end = (k_index + VAILD_B_K) / (BLOCK_W * BM) * BN;

            int m_block_idx = m_index / BLOCK_M;
            const int *I_panel = reinterpret_cast<const int *>(indices + m_block_idx * NNZ_BLOCK_K + indices_start);

            // 预取indices
            int num_block = VAILD_B_K / (BLOCK_W * BM) * BN;
            int num_meta_tile = CEIL(num_block, PREFETCHEDCOL); // 每行需要取多少次
            int fetch_meta = 0;
            const int prefetchMetaStage = PREFETCHEDCOL / BLOCK_PER_TILE; // 取一次可以循环BLOCK_K多少次
            // load the first tile of indice
            // ldg_copy<kNumTMAThreads>(reinterpret_cast<uint *>(smem_indices), reinterpret_cast<const uint *>(I_panel),
            //               min(num_block, PrefetchedCol));
            // cutlass::arch::NamedBarrier(kNumTMAThreads).sync();//对producer进行同步
            cp_async_tile<PREFETCHEDCOL, kNumTMAThreads>(
                            reinterpret_cast<uint *>(smem_indices), reinterpret_cast<const uint *>(I_panel), min(num_block, PREFETCHEDCOL));
            cp_async_group_commit();
            fetch_meta++;

            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++k_iter){

                uint32_t stages_id = k_iter % kNumStages;
                auto& empty_barrier = *(empty_barriers[stages_id]);
                auto& full_barrier = *(full_barriers[stages_id]);

                // 预取下阶段的列坐标
                if (k_iter % prefetchMetaStage == 0){ // 读的是下一个阶段的tile indices
                    cp_async_wait_group<0>();
                    if (fetch_meta < num_meta_tile) {
                        uint *shared_tile_I = smem_indices + (fetch_meta % 2) * PREFETCHEDCOL;
                        const int *tile_I = I_panel + fetch_meta * PREFETCHEDCOL;
                        int meta_num = min(PREFETCHEDCOL,
                                           num_block - fetch_meta * PREFETCHEDCOL);

                        cp_async_tile<PREFETCHEDCOL, kNumTMAThreads>(
                            reinterpret_cast<uint *>(shared_tile_I), reinterpret_cast<const uint *>(tile_I), meta_num);
                        cp_async_group_commit();
                    }
                    fetch_meta++;
                }

                int k_idx = k_nnz_start + k_iter * (BLOCK_K / 2);
                uint *shared_tile_I = smem_indices + (fetch_meta % 2) * PREFETCHEDCOL;
                uint *tile_I = shared_tile_I + (k_iter % prefetchMetaStage) * BLOCK_PER_TILE;

                empty_barrier.arrive_and_wait();

                if (threadIdx.x == kNumMathThreads){ 
                    int load_size = 0;

                    // B是分散的，需要多次2d的拷贝
                    #pragma unroll
                    for(int load_b_idx = 0; load_b_idx < BLOCK_PER_TILE; load_b_idx++){
                        
                        // 恢复全局列坐标
                        int idx = k_iter * BLOCK_PER_TILE + load_b_idx;
                        uint k_base = tile_I[idx];
                        int b_k_idx_block = (idx / BN) * BM + k_iter * (BLOCK_PER_TILE / BN) * BM + k_base;
                        int b_k_idx = b_k_idx_block * BLOCK_W;
                        
                        __nv_bfloat16 *smem_ptr = smem_b[stages_id];
                        cde::cp_async_bulk_tensor_2d_global_to_shared(smem_ptr, &b_desc, b_k_idx, n_index, full_barrier);
                        load_size += BLOCK_W * BLOCK_N * sizeof(__nv_bfloat16);
                    }

                    // load a
                    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_a[stages_id], &a_desc, k_idx, m_index, full_barrier);
                    load_size += BLOCK_M * (BLOCK_K / 2) * sizeof(__nv_bfloat16);

                    auto token = cuda::device::barrier_arrive_tx(full_barrier, 1, load_size);
                }
                else{
                    auto token = full_barrier.arrive();
                }
            }//k iterations
        }//while (scheduler.get_next_block(m_index, n_index, num_rows)) 

    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        #pragma unroll
        for (int32_t s = 0; s < kNumStages; s++) {
            auto token = empty_barriers[s]->arrive();
        }

        static constexpr int WARP_M = 16, WARP_K = (BLOCK_K / 2), WARP_N = BLOCK_N;
        static constexpr int MMA_M = 16, MMA_K = 32, MMA_N = 8;

        static constexpr int iter_lds_a = WARP_M * WARP_K / 16 * 16; // ldsm 16x16
        static constexpr int iter_lds_b = BLOCK_K * MMA_N / (32 * 8); // ldsm 32 x 8
        static constexpr int iter_compute = WARP_N / MMA_N; // ldsm 32 x 8
        static constexpr int iter_store = iter_compute; // ldsm 32 x 8

        static constexpr int num_regs_a = (WARP_M * WARP_K) / (MMA_M * (MMA_K / 2)) * 4;
        static constexpr int num_regs_b = (BLOCK_K * MMA_N) / (MMA_N * MMA_K) * 4;
        static constexpr int num_regs_c = (WARP_M * WARP_N) / (MMA_M * MMA_N) * 4;

        static constexpr int LOAD_WARP_N = 16, LOAD_ELEMENT = 8, LOAD_GROUP = 8;
        static constexpr int num_iter_b_cp_async = (BLOCK_K * LOAD_WARP_N) / (32 * LOAD_ELEMENT);

        while (scheduler.get_next_block(m_index, n_index, num_cols, k_index)) {

            int k_nnz_start = k_index * (BN / BM) / 2;
            int k_nnz_end = (k_index + VAILD_B_K) * (BN / BM) / 2;
            int indices_start = k_index / (BLOCK_W * BM) * BN;
            int indices_end = (k_index + VAILD_B_K) / (BLOCK_W * BM) * BN;

            uint32_t regs_a[num_regs_a], regs_b[2][num_regs_b] = {};//对b regs使用double buffer
            uint32_t meta_regs[2] = {0};
            float accum[num_regs_c] = {0};
            
            // cp.async load a from global memory to shared memory.
            uint32_t itile2read_innstage = 0; // 用于标记读到哪一块smem tile, 用于tc
            uint32_t itile2write_innstage = 0; // 用于标记写到哪一块smem tile, 用于cp.async
            uint32_t itile2write = 0; // 用于标记写的tile的位置 最大值为K/BK

            int m_block_idx = m_index / BLOCK_M;
            const uint *M_panel = reinterpret_cast<const uint *>(
                meta + m_block_idx * (BLOCK_M / META_MROW) * META_NNZ_K + warp_idx * (WARP_M / META_MROW) + (lane_idx >> 2) * META_MROW + (lane_idx & 1) * (BLOCK_M / META_MROW)
            );

            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter){

                uint32_t stages_id = k_iter % kNumStages;
                auto& empty_barrier = *(empty_barriers[stages_id]);
                auto& full_barrier = *(full_barriers[stages_id]);

                // load metadata for this tile
                *((uint2 *)meta_regs) = *((uint2 *)(M_panel + k_iter * (BLOCK_K / 2 / META_ELEMENT) * (BLOCK_M / META_MROW)));

                int k_idx = k_iter * (BLOCK_K / 2);

                full_barrier.arrive_and_wait();

                // read a to regs
                #pragma unroll
                for(uint32_t ldsm_a_idx = 0; ldsm_a_idx < iter_lds_a; ldsm_a_idx++) {
                    ldmatrix<false, 4, 16>(
                        regs_a + ldsm_a_idx * 4,
                        reinterpret_cast<uint32_t*>(
                            smem_a[stages_id] + (lane_idx % 16) * (BLOCK_K / 2) + warp_idx * WARP_M * (BLOCK_K / 2) \
                            + (ldsm_a_idx * (MMA_K / 2) + (lane_idx / 16) * 8 + (lane_idx % 8) * 8) % (BLOCK_K / 2))
                    );
                }

                // read b to regs __ idx 0
                #pragma unroll
                for (uint32_t ldsm_b_idx = 0; ldsm_b_idx < iter_lds_b; ldsm_b_idx++) {
                    // Read B
                    ldmatrix<true, 4, 16>(
                        regs_b[0] + ldsm_b_idx * 4, 
                        reinterpret_cast<uint32_t*>(
                            smem_b[stages_id] + lane_idx * BLOCK_N + ldsm_b_idx * 32 * BLOCK_N \
                            + (0 + (lane_idx % 8) * 8) % BLOCK_N)
                    );
                }

                #pragma unroll
                for (uint32_t compute_idx = 0; compute_idx < iter_compute; ++compute_idx) {
                    auto n_offset = compute_idx * MMA_N;
                    int next_regs_buffer_idx = (compute_idx + 1) % 2;
                    int curr_regs_buffer_idx = (compute_idx) % 2;

                    if (compute_idx + 1 < iter_compute) {
                        #pragma unroll
                        for (uint32_t ldsm_b_idx = 0; ldsm_b_idx < iter_lds_b; ldsm_b_idx++) {
                            // Read B
                            ldmatrix<true, 4, 16>(
                                regs_b[next_regs_buffer_idx] + ldsm_b_idx * 4, 
                                reinterpret_cast<uint32_t*>(
                                    smem_b[stages_id] + lane_idx * BLOCK_N + ldsm_b_idx * 32 * BLOCK_N \
                                    + (((compute_idx + 1) * MMA_N) + (lane_idx % 8) * 8) % BLOCK_N)
                            );
                        }
                    }

                    mma_sparse_m16n8k32(regs_a, regs_b[curr_regs_buffer_idx], accum + compute_idx * 4, meta_regs);
                    // mma_sparse_m16n8k32(regs_a + 8, regs_b[curr_regs_buffer_idx] + 8, accum + compute_idx * 4, meta_regs);
                }

                auto token = empty_barrier.arrive();
            }

            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Write back to shared memory using STSM and issue TMA stores
            #pragma unroll
            for (uint32_t store_idx = 0; store_idx < iter_store; ++ store_idx) {
                auto shifted_accum = accum + store_idx * 4;
                
                uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(
                    smem_d + warp_idx * MMA_M * BLOCK_N + (lane_idx % 8) * BLOCK_M + store_idx * MMA_N * BLOCK_M + ((lane_idx / 8) * 8 + lane_idx % 8 * 8) % BLOCK_M
                );

                // NOTES: only 16 lanes' addresses are used
                SM90_U32x2_STSM_N_TRANS<__nv_bfloat162>::copy(
                    __float22bfloat162_rn({shifted_accum[0], shifted_accum[1]}),
                    __float22bfloat162_rn({shifted_accum[2], shifted_accum[3]}),
                    smem_ptr
                );

            }
            // sync
            cde::fence_proxy_async_shared_cta();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();
            
            // Use TMA store to write back to global memory
            int *gather_index_ptr = gather_index + n_index;

            if (threadIdx.x < num_cols) {
                // int global_n_index = gather_index_ptr[threadIdx.x];//out of range
                int global_n_index = 0;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    out + global_n_index * SHAPE_M + m_index + threadIdx.x * SHAPE_M,
                    smem_d + threadIdx.x * BLOCK_M,
                    BLOCK_N * sizeof(__nv_bfloat16)
                );
                cuda::ptx::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
            }
        }
    }

}

};  // namespace sparse_gemm

#pragma clang diagnostic pop
