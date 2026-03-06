# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import List, Optional, Tuple

import os
import sys
CURRENT_FILE_PATH = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
sys.path.extend([f"{WORK_DIR}/third_party/megablocks/megablocks", f"{WORK_DIR}/hare"])

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from transformers import MistralConfig

import megablocks.ops as ops
import hare_ops as mops

import stk

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank, 
                                             get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs

import sparse_gemm
from utils.sparsify import gen_block_nm
from nvtx import annotate

KVCache = Tuple[torch.Tensor, torch.Tensor]


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


class HareMoE(nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int,
                 top_k: int, bw=16, naive_kernel=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.naive_kernel = naive_kernel
        if self.naive_kernel:
            self.ssd_func = sparse_gemm.ssd_naive
        else:
            self.ssd_func = sparse_gemm.ssd

        # gating
        self.gate = nn.Linear(self.hidden_dim,
                              self.num_experts,
                              bias=False,
                              device=torch.cuda.current_device())

        tp_size = get_tensor_model_parallel_world_size()
        assert self.ffn_dim % tp_size == 0
        self.ffn_dim_per_partition = self.ffn_dim // tp_size
        self.feature_size = self.ffn_dim_per_partition * self.num_experts
        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = gen_block_nm(self.feature_size, 
                                self.hidden_dim, 
                                bw=bw,
                                device=torch.cuda.current_device())
        self.w2 = gen_block_nm(self.hidden_dim,
                                self.feature_size, 
                                bw=bw,
                                device=torch.cuda.current_device())
        self.w3 = gen_block_nm(self.feature_size,
                                self.hidden_dim,
                                bw=bw,
                                device=torch.cuda.current_device())
        
        self.w1.values = nn.Parameter(self.w1.values)
        set_weight_attrs(self.w1, {"weight_loader": self.moe_weight_loader})
        self.w2.values = nn.Parameter(self.w2.values)
        set_weight_attrs(self.w2, {"weight_loader": self.moe_weight_loader})
        self.w3.values = nn.Parameter(self.w3.values)
        set_weight_attrs(self.w3, {"weight_loader": self.moe_weight_loader})

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

    def moe_weight_loader(self, param: nn.Parameter,
                          loaded_weight: torch.Tensor) -> None:
        """
        Load the weights for the MoE linear layer.
        """
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.ffn_dim_per_partition
        loaded_weight = loaded_weight.view(self.num_experts, self.ffn_dim, -1)
        loaded_weight = loaded_weight[:, shard_size * tp_rank:shard_size *
                                      (tp_rank + 1)]
        loaded_weight = loaded_weight.reshape_as(param)
        param.data.copy_(loaded_weight)

    def topology(self, x, block_bins, bins):
        tokens, _ = x.size()

        assert self.ffn_dim_per_partition % self.blocking == 0, "ffn_dim_per_partition must be divisible by blocking"
        num_block_with_seq_dim = block_bins[-1]
        num_blocks_ffn_dim_per_expert = self.ffn_dim_per_partition // self.blocking

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
            
    def indices_and_padded_bins_with_height(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        selected_experts = selected_experts.int()
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert,
                                                self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)
        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)

        block_row_per_expert = torch.div(padded_tokens_per_expert + (self.blocking - 1), self.blocking, rounding_mode='trunc')
        height_offsets = mops.get_height_offsets(padded_tokens_per_expert, self.blocking, self.num_experts)
        height_offsets = promote_scalar(height_offsets)

        block_bins = ops.inclusive_cumsum(block_row_per_expert, 0) #每个专家token数的累加，转换为block为单位

        return indices, bin_ids, bins, padded_bins, block_bins, height_offsets, tokens_per_expert

    @torch.inference_mode()
    @annotate("hare forward", color="red")
    def forward(self, x: torch.Tensor) -> torch.Tensor:

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

        (indices, bin_ids, bins, padded_bins, 
            block_bins, height_offsets, tokens_per_expert) = self.indices_and_padded_bins_with_height(selected_experts)
        gate_indices = indices
        indices_with_ffn_dim, indices_with_seq_dim = self.topology(x, block_bins, bins)

        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins,
                              self.top_k)
        
        seq_len = x.shape[0]
        # gate_proj
        gate_out = self.ssd_func(self.feature_size, seq_len, self.hidden_dim, 
                          self.w1.bh, self.w1.bw, self.w1.bn, self.w1.bm, self.w1.nnz_block_k, 
                          indices_with_ffn_dim, indices_with_seq_dim, height_offsets, gate_indices, 
                          self.w1.metadata, self.w1.block_indices, self.w1.values.data,
                          x)
        gate_out = F.silu(gate_out)
        up_out = self.ssd_func(self.feature_size, seq_len, self.hidden_dim, 
                          self.w3.bh, self.w3.bw, self.w3.bn, self.w3.bm, self.w3.nnz_block_k, 
                          indices_with_ffn_dim, indices_with_seq_dim, height_offsets, gate_indices, 
                          self.w3.metadata, self.w3.block_indices, self.w3.values.data,
                          x)
        x = gate_out * up_out

        x = sparse_gemm.dss(self.hidden_dim, seq_len * self.top_k, self.feature_size,
                          self.w2.bh, self.w2.bw, self.w2.bn, self.w2.bm, self.w2.nnz_block_k,
                          indices_with_ffn_dim, indices_with_seq_dim, height_offsets, gate_indices,
                          self.w2.metadata, self.w2.block_indices, self.w2.values.data,
                          x)

        x = tensor_model_parallel_all_reduce(x)

        x = ops.scatter(x, indices, bin_ids, weights, bins, self.top_k)
        
        return x.view(*input_shape)
