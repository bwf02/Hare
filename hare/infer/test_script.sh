# model_name="deepseek"
# model_name="MiniCPM"
# model_name="mixtral"
model_name="qwen2_moe"

# --repo cloudyu/Mixtral_11Bx2_MoE_19B
set -x

python ${model_name}_hare.py --batch_size 1 --model --time --flash
