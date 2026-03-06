#!/bin/bash
# python test_model.py --batch_size 32 --seq_len 4096 --model --flash

# python test_model.py --batch_size 32 --seq_len 4096 --model --flash

# python test_model.py --batch_size 32 --seq_len 4096 --model --flash

# 定义起始、步长和结束值
START=32
STEP=32
END=2048

# 如果你确定显卡算力，可以在这里强制指定（例如 A100 是 80，RTX 3090/4090 是 86/90）
# export TORCH_CUDA_ARCH_LIST="8.0" 

for (( len=$START; len<=$END; len+=$STEP ))
do
    echo "------------------------------------------------"
    echo "Running: seq_len = $len"
    echo "------------------------------------------------"
    
    # 执行 Python 命令
    python test_model.py --batch_size 1 --seq_len $len --mlp --flash
    
    # 检查上一个命令是否崩溃（比如显存溢出 OOM）
    if [ $? -ne 0 ]; then
        echo "Error occurred at seq_len $len, skipping or stopping..."
        # break # 如果你想出错就停止，取消注释这一行
    fi
done

echo "All tests finished!"