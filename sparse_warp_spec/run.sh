#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="12.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$SCRIPT_DIR"

# export CUBLASLT_LOG_MASK=63
export CUDA_DEVICE_MAX_CONNECTIONS=1 #并行限制
# export DG_JIT_DEBUG=1
#model config
seq_length=1024
hidden_size=1024
ffn_hidden_size=$((4 * hidden_size))
# fixed
batch_size=8
moe_num_experts=8
topk=2

#计算m n k
m=$((batch_size * seq_length))
n=$((moe_num_experts * ffn_hidden_size))
k=$hidden_size

echo $m $n $k
timestamp=$(date +"%Y%m%d_%H%M%S")

ncu_output=${workspace_dir}/log/ncu/hare
mkdir -p $ncu_output

metrics="smsp__warps_issue_stalled_barrier,smsp__average_warp_latency_issue_stalled_barrier,smsp__average_warp_latency_issue_stalled_barrier,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active,l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_duration.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts_cache_bypass.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts_cache_access.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
section_option="--set full
                --section SpeedOfLight_HierarchicalTensorRooflineChart
                --section MemoryWorkloadAnalysis_Tables
                --section LaunchStats
                --section MemoryWorkloadAnalysis_Chart
                --section MemoryWorkloadAnalysis"

if [ "$1" == "ncu" ]; then
    ncu_options="ncu --kernel-name regex:.*d_kernel|fast* -f --cache-control all --metric ${metrics}"
    ncu_options_samoyeds="ncu --launch-count 10 --kernel-name regex:.*SsmmKernel -f --cache-control all --metric ${metrics}"
else
    ncu_options=""
    ncu_options_mega=""
fi

if [ "$2" == "section" ]; then
    ncu_options="ncu --kernel-name regex:.*d_kernel|fast* -f --cache-control all ${section_option} --metric ${metrics} -o ${ncu_output}/cuda_warp_specialization"
    ncu_options_samoyeds="ncu --launch-count 10 --kernel-name regex:.*SsmmKernel -f --cache-control all ${section_option} --metric ${metrics} -o ${ncu_output}/samoyeds_ssmm"
fi


# 根据m n k 计算生成格式生成参数, 并且收集megablock kernel的信息, _sdd_kernel + 
export RANK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -x
# ${ncu_options} ./fastsparse --m ${m} --k ${k} --n ${n} --expert_ffn_size ${expert_ffn_size} --warm_iterations 0 --iterations 1 |& tee log/kernel.log
${ncu_options} python sparse_warp_specialization/tests/test_core.py \
    --micro-batch-size $batch_size \
    --seq-length $seq_length \
    --hidden-size $hidden_size \
    --ffn-hidden-size $ffn_hidden_size \
    --moe-num-experts $moe_num_experts \
    --moe-top-k $topk

# cd /root/code/fast-sparse/third_part/Samoyeds-Kernel
# ${ncu_options_samoyeds} ./build/./benchmark/./horizontal_ssmm_benchmark -m $k -k $n -n $m -N 1 -M 2 --vector_length 128 --seed 42 -t #这里对应的是转置