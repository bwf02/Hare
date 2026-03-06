#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifdef __CLION_IDE__

__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y) ((CEIL((x), (y))) * (y))

namespace sparse_gemm {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    __device__ __host__
    auto operator [](const uint32_t& i) {
        return func(i);
    }
};

template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ T align(T a, T b) {
    return ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template<typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w));
}

template <typename old_t>
__device__ __forceinline__ int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

__device__ __forceinline__ unsigned get_smem_ptr(void *ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template<bool trans, int num_reg, int nbit>
__device__ __forceinline__ void ldmatrix(uint32_t *dst, uint32_t *src)
{
    // no f32 transpose is supported in current cuda
    static_assert((!trans) || nbit==16);

    unsigned smem_ptr = get_smem_ptr(src);

    uint* x = dst;

    if (!trans) {
        if (num_reg==4) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                    : "r"(smem_ptr));
        }
        else if (num_reg==2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(x[0]), "=r"(x[1])
                    : "r"(smem_ptr));
        }
        else if (num_reg==1) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                    : "=r"(x[0])
                    : "r"(smem_ptr));
        }
        else assert(0);
    }
    else { // trans
        if (num_reg==4) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                    : "r"(smem_ptr));
        }
        else if (num_reg==2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(x[0]), "=r"(x[1])
                    : "r"(smem_ptr));
        }
        else if (num_reg==1) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                    : "=r"(x[0])
                    : "r"(smem_ptr));
        }
        else assert(0);
    }
}


template<typename DataType, int SizeInBytes>
__device__ __forceinline__ void cp_async_srclimit(DataType *smem_ptr, const DataType *global_ptr, bool real_copy=true)
{
    int src_size = real_copy ? SizeInBytes : 0;

    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
                 "}\n" ::"r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes),
                 "r"(src_size));
}


__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}


template<int NumThreads>
__device__ __forceinline__ void ldg_copy(uint *dst, const uint *src, int size) {   
    int tid = threadIdx.x % NumThreads;
    for (int i = tid; i < size; i += NumThreads) {
        dst[i] = src[i];
        // (*(float4 *)(dst + i)) = (*(float4 *)(src + i));
    }
}


template<int tile_size, int NumThread> __device__ __forceinline__
void cp_async_tile(uint* smem_ptr, const uint* gmem_ptr, const int valid_size) {
        int tid = threadIdx.x % NumThread;
        for (int i = tid; i < ROUND_UP(tile_size, NumThread); i += NumThread) {
            bool valid = i < tile_size;
            int src_in_bytes = (i < valid_size) ? 4 : 0;
            unsigned dst = get_smem_ptr(smem_ptr + i);
            const uint *src = gmem_ptr + i;
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %0, 0;\n"
                "  @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
                "}\n" ::"r"((int)valid),
                "r"(dst), "l"(src), "n"(4), "r"(src_in_bytes));
        }
}

__device__ __forceinline__ void mma_sparse_m16n8k32(
    uint *a, uint *b, float *c, uint *meta
)
{
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(meta[0]));

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(meta[1]));
}

} // namespace `sparse_gemm`
