import os
import sys

import time
import argparse

from sparsity import BlockCSR, SparseBlockNM
import sparse_gemm
import torch
from cupy.cuda import profiler

WARMUP = 10
ITER = 200

def test_ssd(m, n, k, bn=1, bm=2, bh=128, bw=16, num_expert=4):
    """
    Docstring for test_ssd
    
    :param m: feature dim = ffn_dim_per_exoert * num expert
    :param n: seq_len
    :param k: hidden dim
    :param bn: 
    :param bm: 
    :param bh: 
    :param bw: 
    """
    feature_dim = m
    seq_len = n
    hidden_dim = k
    rhs_matrix = torch.randn(seq_len, hidden_dim).cuda()

    lhs = SparseBlockNM(m, k, bn, bm, bh, bw) # sparsity = bn / bm / 2
    lhs.init_format()
    out = BlockCSR(hidden_dim, feature_dim, num_expert).cuda() # sparsity = 1 / num_expert
    out.init_format(rhs_matrix)

    def test_func():
        out_value = sparse_gemm.ssd(m, n, k, 
            lhs.bh, lhs.bw, lhs.bn, lhs.bm, lhs.nnz_block_k, 
            out.indices_with_ffn_dim, out.indices_with_seq_dim, out.height_offsets, out.indices, 
            lhs.metadata, lhs.block_indices, lhs.values, rhs_matrix)
    
    for i in range(ITER + WARMUP):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.time()
            profiler.start()
        test_func()
    torch.cuda.synchronize()
    profiler.stop()
    end = time.time()
    print(f"m={m}, n={n}, k={k}, bn={bn}, bm={bm}, bh={bh}, bw={bw}, num_expert={num_expert}")
    print(f"Time per iteration: {(end-start)/(ITER-WARMUP):.4f} seconds")

def test_dss(m, n, k, bn=1, bm=2, bh=128, bw=16, num_expert=4):
    """
    Docstring for test_ssd
    
    :param m: hidden dim
    :param n: seq_len
    :param k: feature dim = ffn_dim_per_exoert * num expert
    :param bn: 
    :param bm: 
    :param bh: 
    :param bw: 
    """
    # TODO 正确的shape对应关系
    feature_dim = m
    seq_len = n
    hidden_dim = k
    input = torch.randn(seq_len, hidden_dim).cuda()

    lhs = SparseBlockNM(hidden_dim, feature_dim, bn, bm, bh, bw) # sparsity = bn / bm / 2
    lhs.init_format()
    rhs = BlockCSR(hidden_dim, feature_dim, num_expert) # sparsity = 1 / num_expert
    rhs.init_format(input)
    rhs_value = torch.randn(seq_len, feature_dim // num_expert).cuda()
   
    def test_func():
        out = sparse_gemm.dss(hidden_dim, seq_len, feature_dim,
            lhs.bh, lhs.bw, lhs.bn, lhs.bm, lhs.nnz_block_k, 
            rhs.indices_with_ffn_dim, rhs.indices_with_seq_dim, rhs.height_offsets, rhs.indices, 
            lhs.metadata, lhs.block_indices, lhs.values, rhs_value)
        
    for i in range(ITER + WARMUP):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.time()
            profiler.start()
        test_func()
    torch.cuda.synchronize()
    profiler.stop()
    end = time.time()
    print(f"m={m}, n={n}, k={k}, bn={bn}, bm={bm}, bh={bh}, bw={bw}, num_expert={num_expert}")
    print(f"Time per iteration: {(end-start)/(ITER-WARMUP):.4f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--bn', type=int, default=1)
    parser.add_argument('--bm', type=int, default=2)
    parser.add_argument('--bh', type=int, default=128)
    parser.add_argument('--bw', type=int, default=16)
    parser.add_argument('--num_expert', type=int, default=4)
    parser.add_argument('--ssd', action='store_true', default=False)
    parser.add_argument('--dss', action='store_true', default=False)
    args = parser.parse_args()

    args.dss = True
    # args.ssd = True

    if args.ssd:
        test_ssd(args.m, args.n, args.k, args.bn, args.bm, args.bh, args.bw, args.num_expert)
    elif args.dss:
        test_dss(args.m, args.n, args.k, args.bn, args.bm, args.bh, args.bw, args.num_expert)
    else:
        print("Please specify --ssd or --dss to run the corresponding test.")
