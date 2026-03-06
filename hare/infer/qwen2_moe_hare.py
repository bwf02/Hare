#
# Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
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
#
import os
import sys
CURRENT_FILE_PATH = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
sys.path.extend([f"{WORK_DIR}/third_party/megablocks", f"{WORK_DIR}/hare", f"{WORK_DIR}/third_party/Samoyeds"])

import argparse
import time

import torch
from torch import nn

from vllm.distributed.parallel_state import (initialize_model_parallel, init_distributed_environment, 
                                             destroy_distributed_environment, destroy_model_parallel)
from vllm.distributed import get_tensor_model_parallel_rank
import torch.distributed as dist
from transformers import AutoConfig

from qwen2_moe.modeling_qwen2_moe_vllm import Qwen2MoeDecoderLayer
from qwen2_moe.modeling_qwen2_moe import Qwen2MoeModel
from model.qwen2 import Qwen2HareMoE

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--intermediate_size', type=int, default=1408)

parser.add_argument('--experts', type=int, default=8)
parser.add_argument('--bw', type=int, default=16)

parser.add_argument('--flash', action='store_true', default=False)
parser.add_argument('--huggingface_config', action='store_true', default=False)
parser.add_argument('--repo', type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
args = parser.parse_args()

m = args.intermediate_size
k = args.hidden_size
n = args.batch_size * args.seq_len
expert_num = args.experts
use_flash = args.flash

WARMUP = 10
ITER = 100

configuration = AutoConfig.from_pretrained(args.repo, trust_remote_code=True)
if not args.huggingface_config:
    configuration.moe_intermediate_size = m
    configuration.hidden_size = k
    configuration.num_experts = expert_num

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"

def qwen2_moe_mlp_run():
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen2HareMoE(configuration, bw=args.bw).to(torch.bfloat16).cuda()
    model.eval()

    input = torch.rand((args.batch_size, args.seq_len, configuration.hidden_size)).to(torch.bfloat16).cuda()

    for i in range(ITER + WARMUP):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.time()
        output = model(input)
    torch.cuda.synchronize()
    end = time.time()
    print(
        f"qwen2_moe,mlp,hare,{ITER},{args.batch_size},{args.seq_len},"
        f"{configuration.hidden_size},{configuration.moe_intermediate_size},"
        f"{configuration.num_experts},{(end - start) * 1000},"
        f"{configuration._attn_implementation}"
    )

def qwen2_moe_decoder_layer_run():
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen2MoeDecoderLayer(configuration, 0)
    model.mlp = Qwen2HareMoE(configuration)
    model = model.to(torch.bfloat16).cuda()
    model.eval()
    
    input = torch.rand((args.batch_size, args.seq_len, configuration.hidden_size)).to(torch.bfloat16).cuda()
    
    for i in range(ITER + WARMUP):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.time()
        output = model(input, position_ids=position_ids)
    torch.cuda.synchronize()
    end = time.time()
    print(
        f"qwen2_moe,layer,hare,{ITER},{args.batch_size},{args.seq_len},"
        f"{configuration.hidden_size},{configuration.moe_intermediate_size},"
        f"{configuration.num_experts},{(end - start) * 1000},"
        f"{configuration._attn_implementation}"
    )

def qwen2_moe_model_run():
    device = torch.device(f"cuda")
    torch.set_default_device("cuda")

    model = Qwen2MoeModel(configuration).to(device=device, dtype=torch.bfloat16)
    decoder_layers = nn.ModuleList()
    
    for i in range(configuration.num_hidden_layers):
        layer = Qwen2MoeDecoderLayer(configuration, i).to(
            device=device, 
            dtype=torch.bfloat16
        )
        
        layer.mlp = Qwen2HareMoE(configuration).to(device=device, dtype=torch.bfloat16)
        
        decoder_layers.append(layer)
        
    model.layers = decoder_layers
    model.eval()

    input = torch.randint(
        low=0,
        high=configuration.vocab_size,
        size=(args.batch_size, args.seq_len)
    ).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = model(input_ids=input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print(
            f"qwen2_moe,model,hare,{ITER},{args.batch_size},{args.seq_len},"
            f"{configuration.hidden_size},{configuration.moe_intermediate_size},"
            f"{configuration.num_experts},{(end - start) * 1000},"
            f"{configuration._attn_implementation}"
        )

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_grad_enabled(False)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed_environment(world_size=world_size, rank=rank)
    initialize_model_parallel(backend="nccl")
    
    if args.mlp:
        qwen2_moe_mlp_run()
    if args.layer:
        qwen2_moe_decoder_layer_run()
    if args.model:
        qwen2_moe_model_run()

    destroy_model_parallel()
    destroy_distributed_environment()