#include <torch/extension.h>
#include "indices.cu"

// height offsets
torch::Tensor get_height_offsets_cuda(
    torch::Tensor tokens_per_expert,
    int block_size,
    int num_experts
) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(tokens_per_expert.device()).requires_grad(false);
    torch::Tensor residue_height_offsets = torch::full({num_experts}, block_size, options);

    getHeightOffsetKernelExec<int32_t>(
        tokens_per_expert.data_ptr<int32_t>(),
        residue_height_offsets.data_ptr<int32_t>(),
        block_size,
        num_experts
    );

    return residue_height_offsets;
}

torch::Tensor get_height_offsets(
    torch::Tensor tokens_per_expert,
    int block_size,
    int num_experts
) {
    return get_height_offsets_cuda(tokens_per_expert, block_size, num_experts);
}

// columns indices
torch::Tensor get_col_indices_cuda(
    torch::Tensor block_bins,
    int block_num_rows,
    int block_num_columns,
    int block_size,
    int num_bins // num experts
)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(block_bins.device()).requires_grad(false);
    torch::Tensor indices = torch::zeros({block_num_rows * block_num_columns}, options);//整数部分+残余部分排序

    getColIndicesExec<int32_t>(
        block_bins.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        block_num_rows,
        block_num_columns,
        block_size,
        num_bins
    );

    return indices;
}

torch::Tensor get_col_indices(
    torch::Tensor block_bins,
    int block_num_rows,
    int block_num_columns,
    int block_size,
    int num_bins // num experts
)
{
    return get_col_indices_cuda(block_bins, block_num_rows, block_num_columns, block_size, num_bins);
}

// row indices
torch::Tensor get_row_indices_cuda(
    torch::Tensor block_bins,
    torch::Tensor bins,
    int block_num_rows,
    int num_bins, // num experts
    int block_size
)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(block_bins.device()).requires_grad(false);
    torch::Tensor row_block_offsets = torch::zeros({block_num_rows}, options);

    getRowIndicesExec<int32_t>(
        block_bins.data_ptr<int32_t>(),
        bins.data_ptr<int32_t>(),
        row_block_offsets.data_ptr<int32_t>(),
        block_num_rows - num_bins,
        num_bins,
        block_size
    );

    return row_block_offsets;
}

torch::Tensor get_row_indices(
    torch::Tensor block_bins,
    torch::Tensor bins,
    int block_num_rows,
    int num_bins, // num experts
    int block_size
) {
    return get_row_indices_cuda(block_bins, bins, block_num_rows, num_bins, block_size);
}


PYBIND11_MODULE(hare_ops, m) {
    /**
     * @brief Registers custom Hare kernel functions to the Python module.
     *
     * The following functions are exposed to Python:
     * - `get_height_offsets`: Retrieves height offsets for the sparse matrix.
     * - `get_col_indices`: Retrieves column indices for the sparse matrix.
     * - `get_row_indices`: Retrieves row indices for the sparse matrix.
     *
     * These functions are designed to work with the Hare library and provide
     * efficient operations for sparse matrix computations.
     *
     * @note Ensure that the corresponding C++ functions (`get_height_offsets`, 
     * `get_col_indices`, `get_row_indices`) are implemented and properly linked.
     */
    m.def("get_height_offsets",
          &get_height_offsets,
          "Get height offsets for the sparse matrix",
          py::arg("tokens_per_expert"), py::arg("block_size"), py::arg("num_experts"));

    m.def("get_col_indices",
          &get_col_indices,
          "Get column indices for the sparse matrix",
          py::arg("block_bins"), py::arg("num_rows_block"), py::arg("num_columns_block"), py::arg("block_size"), py::arg("num_experts"));

    m.def("get_row_indices",
          &get_row_indices,
          "Get row indices for the sparse matrix",
          py::arg("block_bins"), py::arg("bins"), py::arg("num_rows_block"), py::arg("num_experts"), py::arg("block_size"));
}