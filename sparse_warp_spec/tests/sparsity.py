from typing import List, Optional, Tuple

import os
import sys
CURRENT_FILE_PATH = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
sys.path.extend([f"{WORK_DIR}/third_party/megablocks", f"{WORK_DIR}/hare"])

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
import megablocks.ops as ops
import hare_ops as mops

from vllm.model_executor.models.deepseek import DeepseekMLP
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs

import sparse_gemm
KVCache = Tuple[torch.Tensor, torch.Tensor]

def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x

class SparseBlockNM:
    def __init__(self, m, k, bn=1, bm=2, bh=128, bw=16):
        self.m = m
        self.k = k
        self.bn = bn
        self.bm = bm
        self.bh = bh
        self.bw = bw
        self.density = bn / (2 * bm)
        # meta info
        self.bits_elem_meta = 2
        self.mrow_m = 2
        self.nelems = 32 // self.bits_elem_meta  # 一个uint可以存储多少个元素的meta信息
        self.nelems_col = self.nelems // self.mrow_m #一个uint在一行里面可以存储多少个元素的meta信息
        # matrix info
        self.num_block_k = k // bw
        self.num_block_m = m // bh
        # 计算每行有多少的非零块, 分为整除和不整除两种情况
        self.residue_block_window = self.num_block_k % bm
        self.nnz_block_k_integer = (self.num_block_k // bm) * bn
        self.nnz_block_k_residue = (self.residue_block_window * bn + bm - 1) // bm
        
        # bw的时候,一行的非零元素可能不是8的倍数,需要特殊处理,使其对齐
        if bw <= 8:
            nnz_integer = self.nnz_block_k_integer * bw // 2
            nnz_integer_align = (nnz_integer + 8 - 1) // 8 * 8
            padding_nnz_for_align = nnz_integer_align - nnz_integer
            self.align_residue_nnz_block(padding_nnz_for_align)
            
        self.nnz_block_k = self.nnz_block_k_integer + self.nnz_block_k_residue
        self.num_indices = self.num_block_m * self.nnz_block_k
        self.nnz_k = int(self.nnz_block_k * bw / 2)
        
        assert self.nnz_k % self.nelems_col == 0, "Please ensure nnz_k is divisible by nelems_col."
        
        self.nnz_meta_uint = m // self.mrow_m * self.nnz_k // self.nelems_col
        self.nnz = m * self.nnz_k
        
        self.values = torch.randn(size=(self.nnz, ), device="cuda", dtype=torch.bfloat16)
        self.block_indices = torch.zeros(size=(self.num_indices, ), device="cuda", dtype=torch.uint32)
        self.metadata = torch.zeros(size=(self.nnz_meta_uint, ), device="cuda", dtype=torch.uint32)
        
    def align_residue_nnz_block(self, padding_nnz_for_align):
        num_block_align = self.nelems_col // (self.bw // 2)
        if (padding_nnz_for_align):
            num_block_align_ = padding_nnz_for_align // (self.bw // 2)
            self.nnz_block_k_residue = (self.nnz_block_k_residue // num_block_align_) * num_block_align_ #ensure multiple of num_block_align_
            if self.nnz_block_k_residue == 0:
                self.nnz_block_k_integer -= (num_block_align - num_block_align_) #if out of range, remove residue
        else:
            # only consider the residue align to 8
            self.nnz_block_k_residue = (self.nnz_block_k_residue // num_block_align) * num_block_align #ensure multiple of num_block_align
            if self.nnz_block_k_residue > self.residue_block_window:
                self.nnz_block_k_residue = 0 #if out of range, remove residue
        
    def init_format_naive(self):
        # init meta
        arr = list(range(4))
        mbrow = self.bh
        for i in range(self.m // mbrow):
            for j in range(int(self.nnz_k // self.nelems_col)):
                for k in range(int(mbrow // self.mrow_m)):
                    meta = 0
                    for g in range(int(self.nelems // 2)):#每次处理2个元素
                        random.shuffle(arr)
                        arr_sorted = sorted(arr[:2])
                        for w, tmp in enumerate(arr_sorted):
                            meta |= tmp << (g * 2 * self.bits_elem_meta + w * self.bits_elem_meta)
                        
                            idx = (
                                i * (mbrow // self.mrow_m) * (self.nnz_k // self.nelems_col)
                                + j * (mbrow // self.mrow_m)
                                + k
                            )
                    self.metadata[idx] = meta
                    
        # init indices
        for i in range(self.m // self.bh):
            for j in range(0, self.nnz_block_k, self.bn):
                window_idx = j // self.bn
                window_size = self.bm if window_idx < (self.num_block_k // self.bm) else (self.num_block_k % self.bm)
                nnz_per_window = self.bn if window_idx < (self.num_block_k // self.bm) else self.nnz_block_k_residue
                arr2 = list(range(window_size))
                random.shuffle(arr2)
                arr2_sorted = sorted(arr2[:nnz_per_window])
                for w, tmp in enumerate(arr2_sorted):
                    idx = i * self.nnz_block_k + j + w
                    self.block_indices[idx] = tmp


    def init_format(self):
        # 性能优化过的
        device = self.metadata.device
        
        # -----------------------------------------------------------
        # 1. 优化 Metadata 初始化 (纯 PyTorch 实现)
        # -----------------------------------------------------------
        dim_i = self.m // self.bh
        dim_j = int(self.nnz_k // self.nelems_col)
        dim_k = int(self.bh // self.mrow_m)
        total_indices = dim_i * dim_j * dim_k
        
        # 预先定义 4 选 2 的组合 (Pairs Pool)
        # 注意：使用 int64 或 int32，稍后转换
        pairs_pool = torch.tensor([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]
        ], device=device, dtype=torch.long)
        
        n_groups = int(self.nelems // 2)
        
        # 1. 批量生成随机索引 [0, 6)
        rand_pair_indices = torch.randint(0, 6, (total_indices, n_groups), device=device)
        
        # 2. 查表获取选中的 pairs
        # shape: (total_indices, n_groups, 2)
        selected_pairs = pairs_pool[rand_pair_indices]
        
        # 3. 构造 shift 向量
        g_indices = torch.arange(n_groups, device=device)
        bits = self.bits_elem_meta
        
        # 广播计算 shift
        shift_w0 = g_indices * 2 * bits
        shift_w1 = g_indices * 2 * bits + bits
        
        # 4. 执行位运算
        # 强转为 self.metadata 的类型 (通常是 int32 或 int64) 以避免位运算溢出问题
        # 假设 metadata 是 int32，中间计算可用 int64 保证安全，最后转回
        calc_dtype = torch.int64 
        val0 = selected_pairs[:, :, 0].to(calc_dtype) << shift_w0
        val1 = selected_pairs[:, :, 1].to(calc_dtype) << shift_w1
        
        meta_values = val0 | val1
        
        # 5. 聚合 (OR reduce)
        # PyTorch 没有 bitwise_or.reduce，但因为位不重叠，sum 等价于 bitwise_or
        final_metas = torch.sum(meta_values, dim=1)
        
        # 赋值 (自动 cast 到 self.metadata 的 dtype)
        self.metadata[:total_indices] = final_metas.to(self.metadata.dtype)

        # -----------------------------------------------------------
        # 2. 优化 Block Indices 初始化 (纯 PyTorch 实现)
        # -----------------------------------------------------------
        num_i = self.m // self.bh
        steps_j = range(0, self.nnz_block_k, self.bn)
        num_j = len(steps_j)
        total_blocks = num_i * num_j
        
        # 生成随机矩阵
        # torch.rand 生成 [0, 1)
        rand_matrix = torch.rand((total_blocks, self.bm), device=device)
        
        # argsort 获取前 bn 个最小值的索引 (等价于随机抽取)
        # shape: (total_blocks, self.bm) -> slice -> (total_blocks, self.bn)
        sampled_indices = torch.argsort(rand_matrix, dim=1)[:, :self.bn]
        
        # 排序
        sampled_indices, _ = torch.sort(sampled_indices, dim=1)
        
        # 展平并赋值
        self.block_indices[:total_blocks * self.bn] = sampled_indices.flatten().to(self.block_indices.dtype)

class BlockCSR(nn.Module):

    def __init__(self, hidden_dim: int, feature_size: int, num_experts: int,
                 top_k=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = feature_size // num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        # gating
        self.gate = nn.Linear(self.hidden_dim,
                              self.num_experts,
                              bias=False,
                              device=torch.cuda.current_device())

        self.ffn_dim = feature_size // self.num_experts
        self.feature_size = self.ffn_dim * self.num_experts

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.ffn_dim * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)
        
        self.indices_with_ffn_dim, self.indices_with_seq_dim, self.height_offsets, self.indices = None, None, None, None

    def topology(self, x, block_bins, bins):
        tokens, _ = x.size()

        assert self.ffn_dim % self.blocking == 0, "ffn_dim_per_partition must be divisible by blocking"
        num_block_with_seq_dim = block_bins[-1]
        num_blocks_ffn_dim_per_expert = self.ffn_dim // self.blocking

        indices_with_ffn_dim =mops.get_col_indices(
            block_bins, 
            num_block_with_seq_dim,
            num_blocks_ffn_dim_per_expert,
            self.blocking,
            self.num_experts
        )

        indices_with_seq_dim = mops.get_row_indices(
            block_bins,
            bins,
            num_block_with_seq_dim,
            self.num_experts,
            self.blocking
        )

        return indices_with_ffn_dim, indices_with_seq_dim
    
    def indices_and_height_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_experts = top_experts.int()
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)
        
        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        block_row_per_expert = torch.div(tokens_per_expert + (self.blocking - 1), self.blocking, rounding_mode='trunc')

        bins = ops.inclusive_cumsum(tokens_per_expert, 0) #每个专家token数的累加
        block_bins = ops.inclusive_cumsum(block_row_per_expert, 0) #每个专家token数的累加，转换为block为单位

        height_offsets = mops.get_height_offsets(tokens_per_expert, self.blocking, self.num_experts)

        height_offsets = promote_scalar(height_offsets)
                
        return indices, bins, bin_ids, block_bins, height_offsets, tokens_per_expert
    
    def init_format(self, x):
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=1, dtype=torch.float) # score
        # weights, selected_experts: (sequence_length, top-k)
        
        weights, selected_experts = torch.topk(all_probs, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten().to(x.dtype)
        selected_experts = selected_experts.flatten()

        self.indices, bins, bin_ids, block_bins, self.height_offsets, tokens_per_expert = self.indices_and_height_bins(selected_experts)
        self.indices_with_ffn_dim, self.indices_with_seq_dim = self.topology(x, block_bins, bins)

    


