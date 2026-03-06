#include <cuda_runtime.h>
#include <stdio.h>

const int kThreadsPerBlock = 32;
const int BLOCK_SIZE = 128;

// height offsets
template <typename IndexType>
void __global__ __launch_bounds__(kThreadsPerBlock)
    getHeightOffsetKernel(
        IndexType *__restrict__ tokens_per_expert,
        IndexType *__restrict__ residue_height_offsets,
        int num_experts,
        int block_size)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // 记录全部块的高度

  // if (tid < num_experts){
  //     int height_offsets_idx = block_bins[tid] - 1;
  //     if (tokens_per_expert[tid] % block_size != 0){
  //         height_offsets[height_offsets_idx] = tokens_per_expert[tid] % block_size;
  //     }
  // }

  // 只记录每个专家的最后一块

  if (tid < num_experts)
  {
    if (tokens_per_expert[tid] % block_size != 0)
    {
      residue_height_offsets[tid] = tokens_per_expert[tid] % block_size;
    }
  }

}

template <typename IndexType>
cudaError_t getHeightOffsetKernelExec(
    IndexType *__restrict__ tokens_per_expert,
    IndexType *__restrict__ residue_height_offsets,
    int block_size,
    int num_experts)
{
  dim3 block(32, 1, 1);
  dim3 grid((num_experts + block.x - 1) / block.x, 1, 1);
  getHeightOffsetKernel<IndexType><<<grid, block>>>(tokens_per_expert, residue_height_offsets, num_experts, block_size);

  return cudaGetLastError();
}

// columns indices
template <typename IndexType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    getColIndicesKernel(
        IndexType *__restrict__ block_bins,
        IndexType *__restrict__ indices,
        int num_rows_block,
        int num_columns_block,
        int block_size,
        int num_bins // num experts
    )
{
  // Load the offset for this bins indices.
  int start = 0;
  if (blockIdx.x > 0){
    start = __ldg(block_bins + blockIdx.x - 1);
  }
  int end = __ldg(block_bins + blockIdx.x);
  int num_rows = end - start;

  start -= blockIdx.x; // 减去每个专家的residue行
  IndexType *integer_indices = indices + (start + blockIdx.y) * num_columns_block + threadIdx.x;
  IndexType *residual_indices = indices + (num_rows_block - num_bins) * num_columns_block + blockIdx.x * num_columns_block + threadIdx.x; // 每个专家只有最后一行是residue, blockIdx.x = expert_idx

  for (int bin_offset = blockIdx.y; bin_offset < num_rows; bin_offset += gridDim.y)
  {
    IndexType *out = integer_indices;
    if (bin_offset == (num_rows - 1))
    {
      out = residual_indices;
    }

    for (int bid = threadIdx.x; bid < num_columns_block; bid += kThreadsPerBlock)
    {
      *out = bid * BLOCK_SIZE + (blockIdx.x * num_columns_block) * BLOCK_SIZE;
      out += kThreadsPerBlock;
    }
    integer_indices += gridDim.y * num_columns_block;
  }
}

template <typename IndexType>
cudaError_t getColIndicesExec(
    IndexType *__restrict__ block_bins,
    IndexType *__restrict__ indices,
    int block_num_rows,
    int block_num_columns,
    int block_size,
    int num_bins // num experts
)
{
  dim3 block_dim(kThreadsPerBlock);
  dim3 grid_dim(num_bins, (int)std::ceil((float)block_num_rows / num_bins)); // x 维度是切换到不太的expert负责的, y维度负责循环一个expert内部
  getColIndicesKernel<IndexType><<<grid_dim, block_dim>>>(
      block_bins,
      indices,
      block_num_rows,
      block_num_columns,
      block_size,
      num_bins);

  return cudaGetLastError();
}

// row indices
template <typename IndexType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    getRowIndicesKernel(
        IndexType *__restrict__ block_bins,//记录的是每个专家blocks数的累积
        IndexType *__restrict__ bins,//记录的是每个专家tokens数的累积
        IndexType *__restrict__ row_block_offsets,
        int num_blocks_integer,
        int num_experts,
        int block_size)
{
  // 记录每个block行的起始位置，并且按照整数+残余部分排序
  int start_block_bins = 0, start_bins = 0;
  if (blockIdx.x > 0){
    start_block_bins = __ldg(block_bins + blockIdx.x - 1);
    start_bins = __ldg(bins + blockIdx.x - 1);
  }
  int end_block_bins = __ldg(block_bins + blockIdx.x);

  int num_rows_block_bins = end_block_bins - start_block_bins;

  start_block_bins -= blockIdx.x;

  IndexType *integer_row_block_offsets = row_block_offsets + start_block_bins;
  IndexType *residue_row_block_offsets = row_block_offsets + num_blocks_integer + blockIdx.x; // blockIdx.x 代表的是专家id

  #pragma unroll
  for (int bin_offset = threadIdx.x; bin_offset < num_rows_block_bins; bin_offset += kThreadsPerBlock)
  {
    IndexType *out = integer_row_block_offsets;
    int offset = bin_offset;
    if (bin_offset == (num_rows_block_bins - 1))
    {
      out = residue_row_block_offsets;
      offset = 0;
    }
    *(out + offset) = start_bins + bin_offset * block_size;
  }
}

template <typename IndexType>
cudaError_t getRowIndicesExec(
    IndexType *__restrict__ block_bins,
    IndexType *__restrict__ bins,
    IndexType *__restrict__ row_block_offsets,
    int num_blocks_integer,
    int num_experts,
    int block_size)
{
  dim3 block_dim(kThreadsPerBlock);
  dim3 grid_dim(num_experts, 1); // x 维度是切换到不太的expert负责的, y维度负责循环一个expert内部
  getRowIndicesKernel<IndexType><<<grid_dim, block_dim>>>(
      block_bins,
      bins,
      row_block_offsets,
      num_blocks_integer,
      num_experts,
      block_size);

  return cudaGetLastError();
}