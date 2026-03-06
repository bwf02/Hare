#pragma once

namespace sparse_gemm {

enum class GemmType {
    Normal              = 0,
    MGroupedContiguous  = 1,
    MGroupedMasked      = 2,
    KGroupedContiguous  = 3,
};

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
};

} // namespace sparse_gemm
