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
from transformers import PretrainedConfig

import megablocks.ops as ops
import hare_ops as mops

import stk

from vllm.model_executor.models.deepseek import DeepseekMLP
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank, 
                                             get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs

import sparse_gemm
from utils.sparsify import gen_block_nm
from nvtx import annotate

KVCache = Tuple[torch.Tensor, torch.Tensor]

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x

class Qwen2HareMoE(nn.Module):

    def __init__(self, 
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None,
                 bw=16
                 ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_experts,
                                     bias=False)
        
        # shared expert
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                reduce_results=False,
            )
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                                  1,
                                                  bias=False)

        # expert
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
        
        # Calculate the bin bounds for the sorted tokens.
        height_offsets = promote_scalar(height_offsets)
        
        return indices, bins, bin_ids, block_bins, height_offsets, tokens_per_expert

    @torch.inference_mode()
    @annotate("hare forward", color="red")
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(
                    self.shared_expert_gate(x)) * shared_output

        # gate_logits: (sequence_length, n_experts)
        gate_logits, _ = self.gate(x)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        weights, selected_experts = torch.topk(all_probs, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten().to(x.dtype)
        selected_experts = selected_experts.flatten()

        indices, bins, bin_ids, block_bins, height_offsets, tokens_per_expert = self.indices_and_height_bins(selected_experts)
        gate_indices = indices // self.top_k
        indices_with_ffn_dim, indices_with_seq_dim = self.topology(x, block_bins, bins)
        seq_len = x.shape[0]
        # gate_proj
        gate_out = sparse_gemm.ssd(self.feature_size, seq_len * self.top_k, self.hidden_dim, 
                          self.w1.bh, self.w1.bw, self.w1.bn, self.w1.bm, self.w1.nnz_block_k, 
                          indices_with_ffn_dim, indices_with_seq_dim, height_offsets, gate_indices, 
                          self.w1.metadata, self.w1.block_indices, self.w1.values.data,
                          x)
        gate_out = F.silu(gate_out)
        up_out = sparse_gemm.ssd(self.feature_size, seq_len * self.top_k, self.hidden_dim, 
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
        
        if shared_output is not None:
            x = x + shared_output

        return x.view(*input_shape)