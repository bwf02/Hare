import torch
import random
import numpy as np

class SparseBlockNM:
    def __init__(self, m, k, bn, bm, bh, bw, device):
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
        
        self.values = torch.empty(size=(self.nnz, ), device=device, dtype=torch.bfloat16)
        self.block_indices = torch.zeros(size=(self.num_indices, ), device=device, dtype=torch.uint32)
        self.metadata = torch.zeros(size=(self.nnz_meta_uint, ), device=device, dtype=torch.uint32)
    
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
        
    def init_metadata(self):
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
                    
        # init block_indices
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
    
    def init_metadata_optimized(self):
        # 获取设备 (CPU 或 CUDA)，确保后续创建的 Tensor 都在同一个设备上
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
        

def gen_block_nm(m, k, bn=1, bm=2, bh=128, bw=16, device=None):
    assert (bn / bm) in [1, 0.5, 0.2, 0.1], "Please ensure vaild bn/bm."
    assert m % bh == 0
    assert k % bw == 0
    
    sparse_matrix = SparseBlockNM(m, k, bn, bm, bh, bw, device)
    # TODO 性能优化
    sparse_matrix.init_metadata_optimized()
    
    return sparse_matrix