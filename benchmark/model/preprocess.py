import re
import logging
import subprocess
from datetime import datetime

def setup_logging(filename=None, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | » %(message)s', 
        datefmt='%m-%d %H:%M'
    )

    if filename is None:
        filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.log")

    root = logging.getLogger()
    root.setLevel(level)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # 日志写入文件
    data_logger = logging.getLogger("file")
    data_logger.propagate = False
    
    fh = logging.FileHandler(filename, mode='w')
    fh.setFormatter(formatter)
    data_logger.addHandler(fh)

def parse_benchmark_data(text: str):
    # 定义正则表达式，严格匹配由逗号分隔的11个字段
    pattern = re.compile(
        r"^\s*"                  # 允许行首有空白字符
        r"([^,]+),"              # 1. model name
        r"([^,]+),"              # 2. model type
        r"([^,]+),"              # 3. kernel name
        r"(\d+),"                # 4. iter (整数)
        r"(\d+),"                # 5. batch size (整数)
        r"(\d+),"                # 6. args.seq_len (整数)
        r"(\d+),"                # 7. configuration.hidden_size (整数)
        r"(\d+),"                # 8. configuration.moe_intermediate_size (整数)
        r"(\d+),"                # 9. configuration.n_routed_experts (整数)
        r"([\d\.]+),"            # 10. (end - start)*1000 (浮点数)
        r"([^,\s]+)"             # 11. configuration._attn_implementation (字符串)
        r"\s*$"                  # 允许行尾有空白字符
    )

    result = None
    
    for line in text.splitlines():
        line = line.strip()
        m = pattern.match(line)
        if m:
            # 提取所有分组并转换为相应类型
            result = {
                "model_name": m.group(1),
                "model_type": m.group(2),
                "kernel_name": m.group(3),
                "iter": int(m.group(4)),
                "batch_size": int(m.group(5)),
                "seq_len": int(m.group(6)),
                "hidden_size": int(m.group(7)),
                "intermediate_size": int(m.group(8)),
                "num_experts": int(m.group(9)),
                "duration": float(m.group(10)),
                "attn_implementation": m.group(11)
            }
            break  # 找到后即可退出循环

    return result

# 测试代码
if __name__ == "__main__":
    log_text = """
    /root/code/fast-sparse/third_party/megablocks/megablocks/grouped_gemm_util.py:10: UserWarning: Grouped GEMM not available.
      warnings.warn('Grouped GEMM not available.')
    INFO 12-31 12:23:39 [__init__.py:216] Automatically detected platform cuda.
    WARNING 12-31 12:23:39 [cuda.py:682] Detected different devices in the system: NVIDIA GeForce RTX 5080, NVIDIA GeForce RTX 3090. Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior.
    WARNING 12-31 12:23:42 [__init__.py:3804] Current vLLM config is not set.
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    WARNING 12-31 12:23:43 [__init__.py:3804] Current vLLM config is not set.
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    INFO 12-31 12:23:43 [parallel_state.py:1166] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
    MiniCPM,mlp,hare,100,1,4096,2304,5760,8,273.6539840698242,eager
    """
    
    data = parse_benchmark_data(log_text)
    if data:
        print("提取成功:")
        for k, v in data.items():
            print(f"{k}: {v}")
    else:
        print("未找到匹配数据")