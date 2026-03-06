import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
# ============== 1. 数据准备 ==============
topk = 6

# 原始数据 [(Input_Dim0, Padded_Dim0), ...]
raw_data = [
    (32, 7936),
    (64, 8192),
    (128, 8192),
    (256, 8192),
    (2048, 16384),
    (4096, 28544)
]

# 提取数据
x_labels = [str(d[0]) for d in raw_data]  # X轴刻度标签
inputs = np.array([d[0] for d in raw_data])
actual_padded = np.array([d[1] for d in raw_data])

# 计算理论值 (Ideal = Input * TopK)
ideal_tokens = inputs * topk

# 计算 Padding 倍数 (用于展示程度)
padding_ratios = actual_padded / ideal_tokens

# ============== 2. 绘图 ==============
plt.figure(figsize=(10, 4.3))
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
# 使用索引作为X轴坐标，避免 log 坐标轴带来的视觉失真，让每个点均匀分布
indices = np.arange(len(inputs))

# --- 绘制线条 ---
# 1. 实际计算量 (橙色实线) - 代表现状
plt.plot(indices, actual_padded, marker='o', color='#E64A19', linewidth=3, 
         label='Real Computed Tokens (Padded)')

# 2. 理论最小计算量 (蓝色虚线) - 代表你的目标/理想状态
plt.plot(indices, ideal_tokens, marker='D', color='#1976D2', linewidth=2.5, linestyle='--', 
         label=f'Theoretical Computed Tokens')

# --- 填充区域 (关键) ---
# 填充中间区域，颜色用灰色或红色，代表“浪费的算力”
plt.fill_between(indices, ideal_tokens, actual_padded, color='gray', alpha=0.15, 
                 label='Padding Overhead (Zero Tokens)')

# ============== 3. 添加详细标注 (Padding 程度) ==============
for i in range(len(indices)):
    # A. 标注实际数值 (上方)
    plt.text(i, actual_padded[i] + 1000, f'{actual_padded[i]}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#E64A19')
    
    # B. 标注理论数值 (下方)
    # 对于前几个点，理论值很小，为了不重叠，稍微错开一点
    offset = -1500 if ideal_tokens[i] > 2000 else -3000
    if i < 4: offset = -500 # 小数值特殊处理
    plt.text(i, ideal_tokens[i] + offset, f'{ideal_tokens[i]}', 
             ha='center', va='top', fontsize=12, color='#1976D2')

    # C. 核心：标注膨胀倍数 (中间)
    # 只有当 padding 比较明显时 (>1.1x) 才标注，避免图表太乱
    ratio = padding_ratios[i]
    if ratio > 1.1: 
        # 计算中间位置
        mid_y = (actual_padded[i] + ideal_tokens[i]) / 2
        
        pct_increase = ((actual_padded[i] - ideal_tokens[i]) / ideal_tokens[i]) * 100
        
        label_text = f"+{pct_increase:.0f}%\nzero tokens"
        
        plt.text(i, mid_y, label_text, 
                 ha='center', va='center', fontsize=13, color='black',
                 )


ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True) # 使用 Latex 风格的指数
formatter.set_powerlimits((0, 0)) # 强制只要有0就转指数
ax.yaxis.set_major_formatter(formatter)
# 如果觉得左上角的 x10^4 太小，可以调大一点
ax.yaxis.get_offset_text().set_fontsize(11)

# X轴：强调输入规模
plt.xlabel('Input Tokens', fontsize=14, labelpad=-2)
plt.xticks(indices, x_labels) # 替换刻度

# Y轴：强调计算负载
plt.ylabel('Total Computed Tokens', fontsize=14, labelpad=1)

# 图例优化
plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=14)

# 网格
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(pad=0.1)
# 显示
plt.savefig("/root/code/fast-sparse/benchmark/plot/figures/padding_overhead_analysis.pdf", dpi=300)