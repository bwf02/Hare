## padding + naive 
# bw=8
# python deepseek_hare_breakdown.py --repo deepseek-ai/deepseek-moe-16b-base --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config --naive
# python MiniCPM_hare_breakdown.py --repo openbmb/MiniCPM-MoE-8x2B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config --naive
# python mixtral_hare_breakdown.py --repo cloudyu/Mixtral_7Bx2_MoE_13B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config --naive
# python mixtral_hare_breakdown.py --repo cloudyu/Mixtral_11Bx2_MoE_19B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config --naive
# python qwen2_moe_hare_breakdown.py --repo Qwen/Qwen1.5-MoE-A2.7B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config --naive

# padding
# bw=8
# python deepseek_hare_breakdown.py --repo deepseek-ai/deepseek-moe-16b-base --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python MiniCPM_hare_breakdown.py --repo openbmb/MiniCPM-MoE-8x2B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare_breakdown.py --repo cloudyu/Mixtral_7Bx2_MoE_13B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare_breakdown.py --repo cloudyu/Mixtral_11Bx2_MoE_19B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python qwen2_moe_hare_breakdown.py --repo Qwen/Qwen1.5-MoE-A2.7B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config

# # bw = 16 
# bw=8
# python deepseek_hare.py --repo deepseek-ai/deepseek-moe-16b-base --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python MiniCPM_hare.py --repo openbmb/MiniCPM-MoE-8x2B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare.py --repo cloudyu/Mixtral_7Bx2_MoE_13B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare.py --repo cloudyu/Mixtral_11Bx2_MoE_19B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python qwen2_moe_hare.py --repo Qwen/Qwen1.5-MoE-A2.7B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config

# bw = 32
bw=32
python deepseek_hare.py --repo deepseek-ai/deepseek-moe-16b-base --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
python MiniCPM_hare.py --repo openbmb/MiniCPM-MoE-8x2B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
python mixtral_hare.py --repo cloudyu/Mixtral_7Bx2_MoE_13B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
python mixtral_hare.py --repo cloudyu/Mixtral_11Bx2_MoE_19B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
python qwen2_moe_hare.py --repo Qwen/Qwen1.5-MoE-A2.7B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config

# ## bw = 32
# bw=64
# python deepseek_hare.py --repo deepseek-ai/deepseek-moe-16b-base --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python MiniCPM_hare.py --repo openbmb/MiniCPM-MoE-8x2B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare.py --repo cloudyu/Mixtral_7Bx2_MoE_13B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python mixtral_hare.py --repo cloudyu/Mixtral_11Bx2_MoE_19B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config
# python qwen2_moe_hare.py --repo Qwen/Qwen1.5-MoE-A2.7B --batch_size 1 --seq_len 4096 --bw $bw --mlp --flash --time --huggingface_config